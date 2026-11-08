"""Sampling: logits -> next token.

Pipeline (same order as vLLM / HF): repetition penalty -> temperature ->
top-k -> top-p -> draw. :func:`probs` exposes the *processed* distribution
because speculative decoding must run its acceptance test against exactly the
distribution we would otherwise have sampled from - if the draft and the
verifier used different processing, the "lossless" guarantee would be void.

Determinism: every sequence owns a ``numpy.random.Generator`` seeded from
``(seed, sequence_index)``. Because the generator travels with the sequence,
its output does not depend on what else happened to be in the batch, or on
whether it was preempted and recomputed - a property the tests check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 1.0          # 0 => greedy
    top_k: int = 0                    # 0 => disabled
    top_p: float = 1.0                # 1 => disabled
    repetition_penalty: float = 1.0   # 1 => disabled; >1 discourages repeats
    seed: int | None = None
    n: int = 1                        # parallel samples sharing one prompt (COW fork)
    stop: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if self.n < 1:
            raise ValueError("n must be >= 1")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0


def apply_repetition_penalty(logits: np.ndarray, context, penalty: float) -> np.ndarray:
    """CTRL-style penalty (Keskar et al. 2019): divide positive logits, multiply
    negative ones, so a seen token always becomes *less* likely. Applied to
    prompt + output tokens, like HF and vLLM."""
    if penalty == 1.0 or len(context) == 0:
        return logits
    logits = logits.copy()
    seen = np.unique(np.asarray(context, dtype=np.int64))
    x = logits[seen]
    logits[seen] = np.where(x > 0, x / penalty, x * penalty)
    return logits


def probs(logits: np.ndarray, params: SamplingParams, context=()) -> np.ndarray:
    """The processed next-token distribution (float64, sums to 1)."""
    x = apply_repetition_penalty(np.asarray(logits, dtype=np.float64), context,
                                 params.repetition_penalty)
    V = x.shape[-1]
    if params.greedy:
        p = np.zeros(V)
        p[int(np.argmax(x))] = 1.0
        return p
    x = x / params.temperature
    if 0 < params.top_k < V:
        kth = np.partition(x, V - params.top_k)[V - params.top_k]
        x = np.where(x >= kth, x, -np.inf)
    x = x - x.max()
    p = np.exp(x)
    p /= p.sum()
    if params.top_p < 1.0:
        order = np.argsort(-p, kind="stable")
        cum = np.cumsum(p[order])
        # keep the smallest prefix whose mass reaches top_p (always >= 1 token)
        keep = cum - p[order] < params.top_p
        mask = np.zeros(V, dtype=bool)
        mask[order[keep]] = True
        p = np.where(mask, p, 0.0)
        p /= p.sum()
    return p


def draw(p: np.ndarray, rng: np.random.Generator) -> int:
    """Inverse-CDF draw. One uniform per token keeps RNG consumption fixed and
    therefore reproducible across code paths."""
    c = np.cumsum(p)
    return int(min(np.searchsorted(c, rng.random() * c[-1], side="right"), len(p) - 1))


def sample(logits: np.ndarray, params: SamplingParams, rng: np.random.Generator,
           context=()) -> int:
    if params.greedy:
        x = apply_repetition_penalty(np.asarray(logits, dtype=np.float64), context,
                                     params.repetition_penalty)
        return int(np.argmax(x))
    return draw(probs(logits, params, context), rng)


def make_rng(seed: int | None, index: int) -> np.random.Generator:
    return np.random.default_rng(None if seed is None else [seed, index])
