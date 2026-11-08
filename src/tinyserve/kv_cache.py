"""Contiguous (per-sequence, pre-reserved) KV cache.

This is how serving worked before PagedAttention: each sequence gets one
buffer sized for the *maximum* length it might reach, because a contiguous
tensor cannot grow in place. Two kinds of waste follow, which
``benchmarks`` measures against the paged cache:

* internal fragmentation - the reserved tail ``max_len - len`` that is never
  written (most of the buffer, for a short answer);
* reservation waste - memory held for tokens that will be generated later.

It is still the right tool for a single sequence (see ``speculative.py``),
where ``truncate`` makes rollback after a rejected draft O(1).
"""

from __future__ import annotations

import numpy as np

from .model import ModelConfig, causal_attention


class KVCache:
    """KV buffer for ONE sequence: (layers, max_len, kv_heads, head_dim)."""

    def __init__(self, cfg: ModelConfig, max_len: int | None = None, dtype=np.float32):
        self.max_len = max_len or cfg.max_seq_len
        shape = (cfg.n_layers, self.max_len, cfg.n_kv_heads, cfg.head_dim)
        self.k = np.zeros(shape, dtype)
        self.v = np.zeros(shape, dtype)
        self.length = 0  # tokens whose K/V are stored

    def truncate(self, length: int) -> None:
        """Forget everything past ``length``. Data is left in place: it will be
        overwritten by the next write, so rollback costs nothing."""
        if not 0 <= length <= self.length:
            raise ValueError(f"cannot truncate cache of length {self.length} to {length}")
        self.length = length

    @property
    def reserved_bytes(self) -> int:
        return self.k.nbytes + self.v.nbytes


class ContiguousBatch:
    """KVBackend over a batch of contiguous caches.

    Sequence ``i`` contributes ``n_new[i]`` consecutive rows of the flat batch,
    at positions ``cache.length .. cache.length + n_new - 1``. Call
    :meth:`commit` after the forward pass to advance every cache's length
    (lengths must stay fixed while the layers run - every layer writes to the
    same slots).
    """

    def __init__(self, caches: list[KVCache], n_new: list[int]):
        self.caches = caches
        self.n_new = n_new
        self.offsets = np.concatenate([[0], np.cumsum(n_new)]).astype(int)
        for c, n in zip(caches, n_new):
            if c.length + n > c.max_len:
                raise ValueError("contiguous cache overflow: sequence exceeds its reservation")

    def positions(self) -> np.ndarray:
        return np.concatenate([np.arange(c.length, c.length + n)
                               for c, n in zip(self.caches, self.n_new)])

    def write_and_attend(self, layer, q, k, v):
        out = np.empty_like(q)
        for c, a, b in zip(self.caches, self.offsets[:-1], self.offsets[1:]):
            s, e = c.length, c.length + (b - a)
            c.k[layer, s:e] = k[a:b]
            c.v[layer, s:e] = v[a:b]
            out[a:b] = causal_attention(q[a:b], c.k[layer, :e], c.v[layer, :e], q_start=s)
        return out

    def commit(self) -> None:
        for c, n in zip(self.caches, self.n_new):
            c.length += n


def forward_with_caches(model, caches: list[KVCache], token_lists: list) -> list[np.ndarray]:
    """Convenience: run ``model`` on new tokens for several cached sequences and
    return each sequence's logits for *all* its new tokens."""
    n_new = [len(t) for t in token_lists]
    kv = ContiguousBatch(caches, n_new)
    tokens = np.concatenate([np.asarray(t, dtype=np.int64) for t in token_lists])
    logits = model.forward(tokens, kv.positions(), kv)
    kv.commit()
    return [logits[a:b] for a, b in zip(kv.offsets[:-1], kv.offsets[1:])]
