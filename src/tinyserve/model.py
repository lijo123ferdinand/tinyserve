"""A small decoder-only transformer in pure numpy (Llama-style).

Architecture per layer::

    h = h + Wo . Attn(RoPE(Wq . RMSNorm(h)), RoPE(Wk . RMSNorm(h)), Wv . RMSNorm(h))
    h = h + W2 . (SiLU(W1 . RMSNorm(h)) * (W3 . RMSNorm(h)))

with grouped-query attention (GQA): ``n_kv_heads`` key/value heads are shared
by ``n_heads // n_kv_heads`` query heads each. GQA matters for serving because
the KV cache - the thing a server spends its memory on - shrinks by that
factor while the query side keeps its full width.

The forward pass is written the way a serving engine needs it, not the way a
training loop does: the input is a *flat* list of tokens from many sequences
(``tokens``, ``positions``), and everything that touches the KV cache is
delegated to a :class:`KVBackend`. The model therefore never knows whether its
keys live in a contiguous buffer, in a paged block pool, or nowhere at all -
that is exactly the seam vLLM draws between the model and its cache engine.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

__all__ = [
    "ModelConfig",
    "Transformer",
    "KVBackend",
    "NoCache",
    "rms_norm",
    "rope_tables",
    "apply_rope",
    "causal_attention",
    "silu",
]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2
    ffn_dim: int = 352
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads (GQA groups)")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_bytes_per_token(self) -> int:
        """Bytes of KV cache one token occupies across all layers (float32)."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * 4


# --------------------------------------------------------------------------
# Building blocks (shared with the training code so both paths are the same math)
# --------------------------------------------------------------------------


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    # RMSNorm drops LayerNorm's mean-centring: cheaper and empirically as good.
    r = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * r * weight


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def rope_tables(cfg: ModelConfig, dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
    """Precompute cos/sin for every position up to ``max_seq_len``.

    Uses the "rotate-half" (GPT-NeoX) pairing: dimension ``i`` is paired with
    ``i + head_dim/2``. Precomputing once means a decode step is just a gather.
    """
    half = cfg.head_dim // 2
    inv_freq = cfg.rope_theta ** (-np.arange(half, dtype=np.float64) / half)
    angles = np.arange(cfg.max_seq_len, dtype=np.float64)[:, None] * inv_freq[None, :]
    return np.cos(angles).astype(dtype), np.sin(angles).astype(dtype)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Rotate ``x[..., n, heads, head_dim]`` by per-token angles.

    ``cos``/``sin`` have shape ``(..., n, head_dim/2)`` (already gathered by
    position). RoPE encodes position *relatively*: q_m . k_n depends only on
    m - n, which is why cached keys never need to be re-rotated when new tokens
    arrive - the property that makes KV caching sound in the first place.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[..., None, :]  # broadcast over heads
    s = sin[..., None, :]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


def causal_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, q_start: int
) -> np.ndarray:
    """Reference GQA attention for ONE sequence.

    q: (n, H, d) - queries for positions ``q_start .. q_start+n-1``
    k, v: (T, KVH, d) - keys/values for positions ``0 .. T-1`` (T >= q_start+n)
    returns (n, H, d)

    Query ``i`` may attend to key ``j`` iff ``j <= q_start + i``. Expressing the
    mask in absolute positions is what lets the same function serve a full
    prefill (q_start=0, n=T), a single decode step (n=1) and a speculative
    verification of k tokens (n=k+1) without special cases.
    """
    n, H, d = q.shape
    T, KVH, _ = k.shape
    g = H // KVH
    # (KVH, g, n, d): each KV head is matched with its group of query heads,
    # so we never materialise repeated K/V (that would defeat GQA's saving).
    qg = q.reshape(n, KVH, g, d).transpose(1, 2, 0, 3)
    kt = k.transpose(1, 2, 0)  # (KVH, d, T)
    scores = (qg @ kt[:, None]) * (1.0 / np.sqrt(d))  # (KVH, g, n, T)
    qpos = q_start + np.arange(n)[:, None]
    kpos = np.arange(T)[None, :]
    scores = np.where(kpos <= qpos, scores, -np.inf)
    scores = scores - scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=-1, keepdims=True)
    out = p @ v.transpose(1, 0, 2)[:, None]  # (KVH, g, n, d)
    return out.transpose(2, 0, 1, 3).reshape(n, H, d)


# --------------------------------------------------------------------------
# KV backends
# --------------------------------------------------------------------------


class KVBackend(Protocol):
    """Where the model's keys and values go.

    For each layer the model hands over the new tokens' q/k/v (flat across the
    whole batch, RoPE already applied) and gets back attention output. The
    backend owns the batch layout (which rows belong to which sequence and at
    what positions), the cache write, and the read.
    """

    def write_and_attend(
        self, layer: int, q: np.ndarray, k: np.ndarray, v: np.ndarray
    ) -> np.ndarray: ...


class NoCache:
    """Recompute-everything backend: the correctness oracle.

    Treats the whole flat batch as ONE sequence starting at position 0. Tests
    compare every cached path against this.
    """

    def write_and_attend(self, layer, q, k, v):
        return causal_attention(q, k, v, q_start=0)


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------


def init_weights(cfg: ModelConfig, seed: int = 0, dtype=np.float32) -> dict[str, np.ndarray]:
    """Seeded init. Residual-output projections are scaled by 1/sqrt(2L) (GPT-2
    style) so the residual stream's variance does not grow with depth."""
    rng = np.random.default_rng(seed)
    D, F, hd = cfg.dim, cfg.ffn_dim, cfg.head_dim

    def normal(shape, std):
        return (rng.standard_normal(shape) * std).astype(dtype)

    proj_std = 0.02 / np.sqrt(2 * cfg.n_layers)
    w: dict[str, np.ndarray] = {
        "tok_emb": normal((cfg.vocab_size, D), 0.02),
        "norm": np.ones(D, dtype),
        "lm_head": normal((D, cfg.vocab_size), 0.02),
    }
    for i in range(cfg.n_layers):
        p = f"layers.{i}."
        w[p + "attn_norm"] = np.ones(D, dtype)
        w[p + "wq"] = normal((D, cfg.n_heads * hd), 0.02)
        w[p + "wk"] = normal((D, cfg.n_kv_heads * hd), 0.02)
        w[p + "wv"] = normal((D, cfg.n_kv_heads * hd), 0.02)
        w[p + "wo"] = normal((cfg.n_heads * hd, D), proj_std)
        w[p + "ffn_norm"] = np.ones(D, dtype)
        w[p + "w1"] = normal((D, F), 0.02)
        w[p + "w3"] = normal((D, F), 0.02)
        w[p + "w2"] = normal((F, D), proj_std)
    return w


class Transformer:
    def __init__(self, cfg: ModelConfig, weights: dict[str, np.ndarray] | None = None,
                 seed: int = 0):
        self.cfg = cfg
        self.w = weights if weights is not None else init_weights(cfg, seed)
        self.cos, self.sin = rope_tables(cfg, self.w["tok_emb"].dtype)
        # Pre-split per-layer weights into tuples: avoids dict lookups + string
        # formatting in the hot loop, which at this model size is a measurable
        # fraction of a decode step.
        self._layers = [
            tuple(self.w[f"layers.{i}.{n}"] for n in
                  ("attn_norm", "wq", "wk", "wv", "wo", "ffn_norm", "w1", "w3", "w2"))
            for i in range(cfg.n_layers)
        ]

    @property
    def dtype(self):
        return self.w["tok_emb"].dtype

    def forward(
        self,
        tokens: np.ndarray,
        positions: np.ndarray,
        kv: KVBackend,
        logits_idx: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run the model on a flat batch of tokens.

        tokens, positions: (N,) int arrays. Rows from different sequences are
        simply concatenated; ``kv`` knows the segment boundaries.
        logits_idx: which rows need logits. During decode only the last row of
        each sequence matters, and the LM head (D x V) is often the single
        biggest matmul - so skipping the rest is a real saving.
        Returns (len(logits_idx) or N, vocab) float logits.
        """
        cfg = self.cfg
        if positions.size and positions.max() >= cfg.max_seq_len:
            raise ValueError(f"position {positions.max()} exceeds max_seq_len {cfg.max_seq_len}")
        N = tokens.shape[0]
        H, KVH, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        cos, sin = self.cos[positions], self.sin[positions]
        h = self.w["tok_emb"][tokens]
        for li, (an, wq, wk, wv, wo, fn, w1, w3, w2) in enumerate(self._layers):
            a = rms_norm(h, an, cfg.norm_eps)
            # Projections run over the whole flat batch at once: this is where
            # batching pays off - one (N x D) @ (D x D) instead of N tiny ones.
            q = apply_rope((a @ wq).reshape(N, H, hd), cos, sin)
            k = apply_rope((a @ wk).reshape(N, KVH, hd), cos, sin)
            v = (a @ wv).reshape(N, KVH, hd)
            o = kv.write_and_attend(li, q, k, v)
            h = h + o.reshape(N, H * hd) @ wo
            m = rms_norm(h, fn, cfg.norm_eps)
            h = h + (silu(m @ w1) * (m @ w3)) @ w2
        if logits_idx is not None:
            h = h[logits_idx]
        return rms_norm(h, self.w["norm"], cfg.norm_eps) @ self.w["lm_head"]

    def logits_full(self, tokens) -> np.ndarray:
        """No-cache reference forward over one sequence. O(T^2) per call."""
        tokens = np.asarray(tokens, dtype=np.int64)
        return self.forward(tokens, np.arange(len(tokens)), NoCache())

    # ---------------------------------------------------------------- io
    def save(self, path: str | Path, extra: dict | None = None) -> None:
        meta = {"config": asdict(self.cfg), **(extra or {})}
        np.savez(path, __meta__=np.array(json.dumps(meta)), **self.w)

    @classmethod
    def load(cls, path: str | Path) -> tuple["Transformer", dict]:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["__meta__"]))
            weights = {k: z[k] for k in z.files if k != "__meta__"}
        return cls(ModelConfig(**meta["config"]), weights), meta

    def num_params(self) -> int:
        return int(sum(a.size for a in self.w.values()))
