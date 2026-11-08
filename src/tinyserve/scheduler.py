"""Continuous-batching scheduler (iteration-level scheduling, Orca/vLLM style).

Static batching forms a batch, runs it until the *longest* member finishes and
only then admits new work: short requests sit idle in finished slots and new
arrivals queue behind the whole batch. Continuous batching re-decides the
batch at every decode step:

    step t:   [A A A A] [B B] [C C C]        <- running decodes, 1 token each
    step t+1: [A] [B] [C] [D D D D D D]      <- B finished; D's prefill joins
                                                 under the token budget

Every step this scheduler:

1. Schedules every RUNNING sequence for one decode token, oldest first. If the
   block pool cannot hold the next token, it *preempts* the youngest running
   sequence (frees all its blocks, sends it back to the front of the queue to
   be recomputed later) until the older one fits. Recompute-preemption (rather
   than swapping KV to CPU) is vLLM's default for short sequences: there is no
   CPU tier here, and a prefill is cheap compared to idling the batch.
2. If nothing was preempted, admits WAITING sequences FCFS while their prefill
   fits in the remaining token budget (``max_num_batched_tokens``), the seq
   cap, and the free blocks. Skipping admission after a preemption avoids
   thrashing: admitting new work into a pool that just ran dry would only
   cause the next preemption.

A sequence's uncomputed tokens are ``tokens[num_computed:]``. A fresh request
has its whole prompt uncomputed (minus any prefix-cache hit), a running
sequence has exactly one, and a preempted one has prompt+output again - so
prefill, decode and recompute are the same code path.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .paged import BlockManager


class SeqStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(eq=False)
class Sequence:
    seq_id: int
    request_id: str
    index: int                  # which of the request's n samples
    tokens: list[int]
    prompt_len: int
    rng: object
    arrival: float
    num_computed: int = 0
    status: SeqStatus = SeqStatus.WAITING
    finish_reason: str | None = None
    num_preemptions: int = 0
    emitted_chars: int = 0      # streaming cursor into the output text
    first_token_time: float | None = None

    @property
    def output_tokens(self) -> list[int]:
        return self.tokens[self.prompt_len:]

    @property
    def num_uncomputed(self) -> int:
        return len(self.tokens) - self.num_computed

    def priority(self):
        # FCFS by request arrival; seq_id breaks ties deterministically
        return (self.arrival, self.seq_id)


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 512
    policy: str = "continuous"   # or "static"

    def __post_init__(self):
        if self.policy not in ("continuous", "static"):
            raise ValueError(f"unknown policy {self.policy!r}")


@dataclass
class ScheduledBatch:
    seqs: list[Sequence] = field(default_factory=list)
    copies: list[tuple[int, int]] = field(default_factory=list)   # COW (src, dst)
    preempted: list[Sequence] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return sum(s.num_uncomputed for s in self.seqs)

    def __bool__(self) -> bool:
        return bool(self.seqs)


class Scheduler:
    def __init__(self, config: SchedulerConfig, block_manager: BlockManager):
        self.config = config
        self.bm = block_manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0

    def add(self, seq: Sequence) -> None:
        seq.status = SeqStatus.WAITING
        self.waiting.append(seq)

    def add_running(self, seq: Sequence) -> None:
        """For forked children: they already own (shared) blocks."""
        seq.status = SeqStatus.RUNNING
        self.running.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def _preempt(self, seq: Sequence) -> None:
        self.bm.free(seq.seq_id)
        seq.num_computed = 0
        seq.status = SeqStatus.WAITING
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(seq)

    def finish(self, seq: Sequence, reason: str) -> None:
        seq.status = SeqStatus.FINISHED
        seq.finish_reason = reason
        if seq in self.running:
            self.running.remove(seq)
        elif seq in self.waiting:
            self.waiting.remove(seq)
        if seq.seq_id in self.bm.tables:
            self.bm.free(seq.seq_id)

    def schedule(self) -> ScheduledBatch:
        cfg = self.config
        out = ScheduledBatch()
        budget = cfg.max_num_batched_tokens

        # ---- 1. running sequences, oldest first --------------------------
        queue = sorted(self.running, key=Sequence.priority)
        still_running: list[Sequence] = []
        while queue:
            seq = queue.pop(0)
            if seq.num_uncomputed > budget:
                still_running.append(seq)   # keeps its blocks, runs next step
                continue
            while not self.bm.can_append(seq.seq_id, len(seq.tokens)):
                victim = queue.pop() if queue else seq   # youngest first
                self._preempt(victim)
                out.preempted.append(victim)
                if victim is seq:
                    break
            else:
                out.copies += self.bm.ensure_capacity(seq.seq_id, len(seq.tokens),
                                                      seq.num_computed)
                out.seqs.append(seq)
                still_running.append(seq)
                budget -= seq.num_uncomputed
        self.running = sorted(still_running, key=Sequence.priority)

        # ---- 2. admit new work -------------------------------------------
        if out.preempted:
            return out
        if cfg.policy == "static" and self.running:
            return out   # static batching: wait for the whole batch to drain
        while self.waiting and len(self.running) < cfg.max_num_seqs:
            seq = self.waiting[0]
            hits, _ = self.bm.plan_prefix(seq.tokens)
            n_new = len(seq.tokens) - len(hits) * self.bm.block_size
            if n_new > budget or not self.bm.can_allocate(seq.tokens):
                break   # FCFS: do not let a later, smaller request jump the queue
            seq.num_computed = self.bm.allocate(seq.seq_id, seq.tokens)
            self.waiting.popleft()
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            out.seqs.append(seq)
            budget -= seq.num_uncomputed
        return out
