import numpy as np
import pytest

from tinyserve.kv_cache import ContiguousBatch, KVCache, forward_with_caches
from tinyserve.model import ModelConfig, Transformer, apply_rope, causal_attention, rope_tables

from conftest import tiny_model


def naive_attention(q, k, v):
    """Textbook attention with K/V explicitly repeated per query head."""
    n, H, d = q.shape
    g = H // k.shape[1]
    k, v = np.repeat(k, g, axis=1), np.repeat(v, g, axis=1)
    s = np.einsum("nhd,thd->hnt", q, k) / np.sqrt(d)
    s = np.where(np.tril(np.ones((n, n), bool)), s, -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return np.einsum("hnt,thd->nhd", p, v)


def test_gqa_attention_matches_naive(rng):
    q = rng.standard_normal((9, 6, 8))
    k = rng.standard_normal((9, 2, 8))
    v = rng.standard_normal((9, 2, 8))
    np.testing.assert_allclose(causal_attention(q, k, v, 0), naive_attention(q, k, v), atol=1e-12)


def test_rope_is_relative(rng):
    """q_m . k_n depends only on m - n: the property that makes caching rotated keys valid."""
    cfg = ModelConfig(vocab_size=4, dim=16, n_heads=1, n_kv_heads=1, max_seq_len=64)
    cos, sin = rope_tables(cfg, np.float64)
    q = rng.standard_normal((1, 1, 16))
    k = rng.standard_normal((1, 1, 16))

    def dot(m, n):
        return float(np.sum(apply_rope(q, cos[[m]], sin[[m]]) * apply_rope(k, cos[[n]], sin[[n]])))

    assert dot(5, 2) == pytest.approx(dot(40, 37), rel=1e-9)
    assert dot(5, 2) != pytest.approx(dot(5, 3), rel=1e-3)


def test_cached_decode_matches_uncached(model, rng):
    toks = rng.integers(0, model.cfg.vocab_size, 40).tolist()
    ref = model.logits_full(toks)
    cache = KVCache(model.cfg)
    got = [forward_with_caches(model, [cache], [toks[:1]])[0]]
    for t in toks[1:]:                       # one token at a time, like decode
        got.append(forward_with_caches(model, [cache], [[t]])[0])
    np.testing.assert_allclose(np.concatenate(got), ref, rtol=1e-4, atol=1e-4)
    assert cache.length == 40


def test_chunked_prefill_matches_uncached(model, rng):
    toks = rng.integers(0, model.cfg.vocab_size, 50).tolist()
    ref = model.logits_full(toks)
    cache = KVCache(model.cfg)
    chunks = [toks[:7], toks[7:30], toks[30:31], toks[31:]]
    got = np.concatenate([forward_with_caches(model, [cache], [c])[0] for c in chunks])
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_batched_contiguous_matches_single(model, rng):
    seqs = [rng.integers(0, model.cfg.vocab_size, n).tolist() for n in (5, 17, 1, 33)]
    caches = [KVCache(model.cfg) for _ in seqs]
    outs = forward_with_caches(model, caches, seqs)
    for s, o in zip(seqs, outs):
        np.testing.assert_allclose(o, model.logits_full(s), rtol=1e-4, atol=1e-4)
    # then a batched decode step on top of different-length histories
    nxt = [[int(rng.integers(model.cfg.vocab_size))] for _ in seqs]
    outs = forward_with_caches(model, caches, nxt)
    for s, n, o in zip(seqs, nxt, outs):
        np.testing.assert_allclose(o[0], model.logits_full(s + n)[-1], rtol=1e-4, atol=1e-4)


def test_cache_truncate_rolls_back(model, rng):
    toks = rng.integers(0, model.cfg.vocab_size, 20).tolist()
    cache = KVCache(model.cfg)
    forward_with_caches(model, [cache], [toks])
    cache.truncate(12)
    alt = [3, 1, 4]
    out = forward_with_caches(model, [cache], [alt])[0]
    np.testing.assert_allclose(out, model.logits_full(toks[:12] + alt)[12:], rtol=1e-4, atol=1e-4)
    with pytest.raises(ValueError):
        cache.truncate(99)


def test_contiguous_overflow_raises(model):
    cache = KVCache(model.cfg, max_len=4)
    with pytest.raises(ValueError):
        ContiguousBatch([cache], [5])


def test_position_limit(model):
    with pytest.raises(ValueError):
        model.logits_full([0] * (model.cfg.max_seq_len + 1))


def test_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=4, dim=32, n_heads=4, n_kv_heads=3)


def test_save_load_roundtrip(tmp_path, tok):
    m = tiny_model(tok.vocab_size, seed=7)
    m.save(tmp_path / "m.npz", {"tokenizer": tok.chars})
    m2, meta = Transformer.load(tmp_path / "m.npz")
    assert meta["tokenizer"] == tok.chars and m2.cfg == m.cfg
    np.testing.assert_array_equal(m2.logits_full([1, 2, 3]), m.logits_full([1, 2, 3]))
