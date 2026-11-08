"""The serving engine: scheduler + paged KV pool + model + sampler.

One :meth:`LLMEngine.step` is one iteration of the serving loop::

    schedule()          -> which sequences run, COW copies, preemptions
    copy blocks         -> apply copy-on-write before anything is written
    build flat batch    -> tokens/positions/slot_mapping for every scheduled seq
    model.forward()     -> one pass; prefills and decodes share the matmuls
    sample              -> per-sequence params and RNG
    bookkeeping         -> publish full blocks to the prefix cache, check stop
                           conditions, free finished sequences, stream deltas
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np

from .model import Transformer
from .paged import BlockManager, PagedBatch, PagedKVPool
from .sampling import SamplingParams, make_rng, sample
from .scheduler import Scheduler, SchedulerConfig, Sequence, SeqStatus
from .tokenizer import CharTokenizer


@dataclass
class EngineConfig:
    block_size: int = 16
    num_blocks: int = 256
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 512
    max_model_len: int | None = None       # defaults to the model's max_seq_len
    enable_prefix_caching: bool = True
    policy: str = "continuous"             # "continuous" | "static"
    attn: str = "gather"                   # "gather" | "paged"


@dataclass
class StepOutput:
    """An incremental update for one sequence (what a streaming client gets)."""
    request_id: str
    index: int
    new_token_ids: list[int]
    new_text: str
    finished: bool = False
    finish_reason: str | None = None


@dataclass
class CompletionOutput:
    request_id: str
    index: int
    text: str
    token_ids: list[int]
    finish_reason: str
    prompt_tokens: int
    num_preemptions: int = 0


@dataclass
class EngineStats:
    steps: int = 0
    prefill_tokens: int = 0       # prompt tokens actually computed
    cached_prompt_tokens: int = 0 # prompt tokens served from the prefix cache
    decode_tokens: int = 0
    generated_tokens: int = 0
    max_running: int = 0
    cow_copies: int = 0
    block_usage: list[float] = field(default_factory=list)


@dataclass
class _Request:
    request_id: str
    params: SamplingParams
    prompt_len: int
    seqs: list[Sequence]
    forked: bool = False


class LLMEngine:
    def __init__(self, model: Transformer, tokenizer: CharTokenizer,
                 config: EngineConfig | None = None, clock=time.monotonic):
        self.model, self.tok = model, tokenizer
        self.config = c = config or EngineConfig()
        self.max_model_len = c.max_model_len or model.cfg.max_seq_len
        if self.max_model_len > model.cfg.max_seq_len:
            raise ValueError("max_model_len exceeds the model's max_seq_len")
        # Without chunked prefill a preempted sequence must be re-prefilled in
        # one step, so the budget must fit the longest possible sequence, and
        # the pool must hold at least one - otherwise the engine can deadlock.
        if c.max_num_batched_tokens < self.max_model_len:
            raise ValueError("max_num_batched_tokens must be >= max_model_len")
        if c.num_blocks * c.block_size < self.max_model_len:
            raise ValueError("KV pool smaller than one max-length sequence")
        self.bm = BlockManager(c.num_blocks, c.block_size, c.enable_prefix_caching)
        self.pool = PagedKVPool(model.cfg, c.num_blocks, c.block_size, model.dtype)
        self.scheduler = Scheduler(
            SchedulerConfig(c.max_num_seqs, c.max_num_batched_tokens, c.policy), self.bm)
        self.clock = clock
        self.stats = EngineStats()
        self._requests: dict[str, _Request] = {}
        self._seq_ids = itertools.count()
        self._req_ids = itertools.count()

    # ------------------------------------------------------------------ API
    def add_request(self, prompt: str | list[int], params: SamplingParams | None = None,
                    request_id: str | None = None, arrival_time: float | None = None) -> str:
        params = params or SamplingParams()
        ids = self.tok.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if not ids:
            raise ValueError("empty prompt")
        if len(ids) >= self.max_model_len:
            raise ValueError(f"prompt of {len(ids)} tokens does not fit max_model_len "
                             f"{self.max_model_len}")
        rid = request_id or f"req-{next(self._req_ids)}"
        if rid in self._requests:
            raise ValueError(f"duplicate request id {rid}")
        arrival = self.clock() if arrival_time is None else arrival_time
        seq = Sequence(next(self._seq_ids), rid, 0, ids, len(ids),
                       make_rng(params.seed, 0), arrival)
        self._requests[rid] = _Request(rid, params, len(ids), [seq])
        self.scheduler.add(seq)
        return rid

    def abort(self, request_id: str) -> None:
        req = self._requests.pop(request_id, None)
        if req:
            for s in req.seqs:
                if s.status is not SeqStatus.FINISHED:
                    self.scheduler.finish(s, "abort")

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    def result(self, request_id: str) -> list[CompletionOutput]:
        req = self._requests[request_id]
        return [CompletionOutput(req.request_id, s.index, self._visible_text(s, final=True),
                                 s.output_tokens, s.finish_reason or "", req.prompt_len,
                                 s.num_preemptions) for s in req.seqs]

    def release(self, request_id: str) -> None:
        """Drop bookkeeping for a finished request (servers call this)."""
        self._requests.pop(request_id, None)

    def generate(self, prompts: list[str], params: SamplingParams | list[SamplingParams]
                 ) -> list[list[CompletionOutput]]:
        """Offline batch API: submit everything, step until done."""
        plist = params if isinstance(params, list) else [params] * len(prompts)
        rids = [self.add_request(p, sp) for p, sp in zip(prompts, plist)]
        while self.has_unfinished():
            self.step()
        outs = [self.result(r) for r in rids]
        for r in rids:
            self.release(r)
        return outs

    # ----------------------------------------------------------------- step
    def step(self) -> list[StepOutput]:
        batch = self.scheduler.schedule()
        if not batch:
            return []
        for src, dst in batch.copies:
            self.pool.copy_block(src, dst)
        self.stats.cow_copies += len(batch.copies)

        seqs = batch.seqs
        n_new = [s.num_uncomputed for s in seqs]
        starts = [s.num_computed for s in seqs]
        tokens = np.fromiter(itertools.chain.from_iterable(
            s.tokens[s.num_computed:] for s in seqs), dtype=np.int64, count=sum(n_new))
        kv = PagedBatch(self.pool, [self.bm.block_table(s.seq_id) for s in seqs],
                        starts, n_new, self.config.attn)
        logits_idx = np.cumsum(n_new) - 1
        logits = self.model.forward(tokens, kv.positions, kv, logits_idx)

        st = self.stats
        st.steps += 1
        st.max_running = max(st.max_running, len(self.scheduler.running))
        st.block_usage.append(self.bm.alloc.num_used / self.bm.alloc.num_blocks)
        now = self.clock()
        outputs: list[StepOutput] = []
        for seq, row, n in zip(seqs, logits, n_new):
            if n > 1 or seq.num_computed < seq.prompt_len:
                # a prefill (or recompute); count only real prompt work
                st.prefill_tokens += n
                if seq.num_preemptions == 0:
                    st.cached_prompt_tokens += seq.num_computed
            else:
                st.decode_tokens += n
            seq.num_computed = len(seq.tokens)
            self.bm.mark_computed(seq.seq_id, seq.tokens, seq.num_computed)
            req = self._requests[seq.request_id]
            group = [seq]
            if req.params.n > 1 and not req.forked:
                group += self._fork(req, seq)
            for s in group:
                outputs.append(self._append_token(s, req.params, row, now))
        return outputs

    # --------------------------------------------------------------- helpers
    def _fork(self, req: _Request, parent: Sequence) -> list[Sequence]:
        """Parallel sampling: n-1 children share the parent's prompt blocks.
        Only the partial last block is ever copied (on first divergent write)."""
        req.forked = True
        kids = []
        for i in range(1, req.params.n):
            child = Sequence(next(self._seq_ids), req.request_id, i, list(parent.tokens),
                             parent.prompt_len, make_rng(req.params.seed, i), parent.arrival,
                             num_computed=parent.num_computed)
            self.bm.fork(parent.seq_id, child.seq_id)
            self.scheduler.add_running(child)
            req.seqs.append(child)
            kids.append(child)
        return kids

    def _append_token(self, seq: Sequence, params: SamplingParams, logits: np.ndarray,
                      now: float) -> StepOutput:
        tok = sample(logits, params, seq.rng, seq.tokens)
        seq.tokens.append(tok)
        self.stats.generated_tokens += 1
        if seq.first_token_time is None:
            seq.first_token_time = now
        reason = None
        text = self.tok.decode(seq.output_tokens)
        if params.stop and any(s in text for s in params.stop):
            reason = "stop"
        elif len(seq.output_tokens) >= params.max_tokens or len(seq.tokens) >= self.max_model_len:
            reason = "length"
        if reason:
            self.scheduler.finish(seq, reason)
        visible = self._visible_text(seq, final=reason is not None, text=text)
        delta = visible[seq.emitted_chars:]
        seq.emitted_chars = len(visible)
        return StepOutput(seq.request_id, seq.index, [tok], delta, reason is not None, reason)

    def _visible_text(self, seq: Sequence, final: bool, text: str | None = None) -> str:
        """Output text safe to show the client.

        Stop strings are excluded from the output (OpenAI semantics). While
        streaming, a suffix that *could* be the start of a stop string is held
        back - otherwise we might stream "EN" and then discover the stop string
        was "END", with no way to take the characters back.
        """
        text = self.tok.decode(seq.output_tokens) if text is None else text
        stops = self._requests[seq.request_id].params.stop
        if not stops:
            return text
        cut = min((text.find(s) for s in stops if s in text), default=-1)
        if cut >= 0:
            return text[:cut]
        if final:
            return text
        hold = 0
        for s in stops:
            for k in range(min(len(s) - 1, len(text)), 0, -1):
                if text.endswith(s[:k]):
                    hold = max(hold, k)
                    break
        return text[:len(text) - hold]
