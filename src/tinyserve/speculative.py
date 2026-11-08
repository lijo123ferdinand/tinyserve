"""Speculative decoding (Leviathan et al. 2023; Chen et al. 2023).

Decoding is latency-bound: each token costs one full forward of the target
model, and a forward over 1 token costs nearly the same as a forward over
k+1 tokens (the weights are read once either way; here, the Python and
per-layer overhead is paid once either way). So:

1. a cheap draft model proposes k tokens autoregressively, ``x_1..x_k``
   with distributions ``q_1..q_k``;
2. the target scores all of them in ONE forward, giving ``p_1..p_{k+1}``;
3. walk the proposals: accept ``x_i`` with probability ``min(1, p_i(x_i)/q_i(x_i))``.
   On the first rejection, emit a token from the residual
   ``norm(max(0, p_i - q_i))`` and stop. If all k are accepted, emit a bonus
   token from ``p_{k+1}``.

Why the residual: P(emit x) = q(x)·min(1, p/q) + P(reject)·residual(x)
= min(q, p) + (1 - sum min(q, p)) · max(0, p - q) / sum max(0, p - q)
= min(q, p) + max(0, p - q) = p(x). The output distribution is *exactly* the
target's - speculation changes the speed, never the samples' law. A test
checks this empirically; a "shortcut" such as accepting iff the draft token is
the target's argmax would silently bias sampling.

Each step emits between 1 and k+1 tokens for one target forward.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .kv_cache import KVCache, forward_with_caches
from .model import Transformer
from .sampling import SamplingParams, draw, make_rng, probs


def verify(p_rows: np.ndarray, q_rows: np.ndarray, draft_tokens: list[int],
           rng: np.random.Generator) -> tuple[list[int], int]:
    """The acceptance rule, isolated so it can be tested on synthetic p and q.

    p_rows: (k+1, V) target distributions; q_rows: (k, V) draft distributions.
    Returns (emitted tokens, number of accepted draft tokens).
    """
    out = []
    for i, x in enumerate(draft_tokens):
        p, q = p_rows[i], q_rows[i]
        # q[x] > 0 always: x was sampled from q.
        if rng.random() < min(1.0, p[x] / q[x]):
            out.append(x)
            continue
        resid = np.maximum(p - q, 0.0)
        s = resid.sum()
        # s == 0 only when p == q, where rejection has probability 0; guard
        # anyway against float round-off.
        out.append(draw(resid / s if s > 0 else p, rng))
        return out, i
    out.append(draw(p_rows[len(draft_tokens)], rng))
    return out, len(draft_tokens)


@dataclass
class SpecStats:
    proposed: int = 0
    accepted: int = 0
    target_forwards: int = 0
    draft_forwards: int = 0
    emitted: int = 0
    seconds: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_target_forward(self) -> float:
        return self.emitted / self.target_forwards if self.target_forwards else 0.0


class SpeculativeDecoder:
    """Single-sequence speculative decoding with contiguous KV caches.

    Cache invariant (both models): the cache holds K/V for every context token
    except the last one, which is fed on the next forward. After verification
    the target cache is truncated to drop the rejected proposals - with a
    contiguous cache that is just moving the length pointer.
    """

    def __init__(self, target: Transformer, draft: Transformer, k: int = 4):
        if target.cfg.vocab_size != draft.cfg.vocab_size:
            raise ValueError("draft and target must share a vocabulary")
        if k < 1:
            raise ValueError("k must be >= 1")
        self.target, self.draft, self.k = target, draft, k

    def generate(self, prompt: list[int], params: SamplingParams
                 ) -> tuple[list[int], SpecStats]:
        st = SpecStats()
        t0 = time.perf_counter()
        rng = make_rng(params.seed, 0)
        max_len = min(self.target.cfg.max_seq_len, self.draft.cfg.max_seq_len)
        tc = KVCache(self.target.cfg, max_len, self.target.dtype)
        dc = KVCache(self.draft.cfg, max_len, self.draft.dtype)
        ctx = list(prompt)
        # prime the target with everything but the last prompt token
        if len(ctx) > 1:
            forward_with_caches(self.target, [tc], [ctx[:-1]])
        while len(ctx) - len(prompt) < params.max_tokens:
            room = max_len - len(ctx)
            if room <= 0:
                break
            k = min(self.k, params.max_tokens - (len(ctx) - len(prompt)) - 1, room - 1)
            n0 = len(ctx)
            drafts, q_rows = [], []
            # --- draft k tokens (feed whatever the draft cache is missing)
            for _ in range(k):
                full = ctx + drafts
                lg = forward_with_caches(self.draft, [dc], [full[dc.length:]])[0][-1]
                st.draft_forwards += 1
                q = probs(lg, params, full)
                drafts.append(draw(q, rng))
                q_rows.append(q)
            # --- verify with one target forward over [last, x_1..x_k]
            lg = forward_with_caches(self.target, [tc], [ctx[-1:] + drafts])[0]
            st.target_forwards += 1
            p_rows = np.stack([probs(lg[i], params, ctx + drafts[:i]) for i in range(k + 1)])
            if k:
                emitted, acc = verify(p_rows, np.stack(q_rows), drafts, rng)
            else:
                emitted, acc = [draw(p_rows[0], rng)], 0
            st.proposed += k
            st.accepted += acc
            ctx += emitted
            # --- roll back: keep K/V only for tokens before the new last one
            tc.truncate(n0 + acc)
            dc.truncate(min(dc.length, n0 + acc))
        st.emitted = len(ctx) - len(prompt)
        st.seconds = time.perf_counter() - t0
        return ctx[len(prompt):][:params.max_tokens], st


def autoregressive(model: Transformer, prompt: list[int], params: SamplingParams
                   ) -> tuple[list[int], float]:
    """Plain KV-cached decoding with the same sampler: the speed baseline and
    the distribution that speculative decoding must reproduce."""
    t0 = time.perf_counter()
    rng = make_rng(params.seed, 0)
    cache = KVCache(model.cfg, model.cfg.max_seq_len, model.dtype)
    ctx = list(prompt)
    feed = ctx
    while len(ctx) - len(prompt) < params.max_tokens and len(ctx) < model.cfg.max_seq_len:
        lg = forward_with_caches(model, [cache], [feed])[0][-1]
        ctx.append(draw(probs(lg, params, ctx), rng))
        feed = ctx[-1:]
    return ctx[len(prompt):], time.perf_counter() - t0
