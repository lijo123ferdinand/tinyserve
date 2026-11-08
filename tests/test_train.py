import numpy as np

from tinyserve.model import ModelConfig, Transformer, init_weights
from tinyserve.train import AdamW, loss_and_grads

CFG = ModelConfig(vocab_size=11, dim=16, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=24,
                  max_seq_len=32)


def _weights(seed=1):
    w = init_weights(CFG, seed, np.float64)
    r = np.random.default_rng(seed)
    for k in w:   # move off the trivial init so every term in the backward matters
        w[k] = w[k] * 10 if w[k].ndim == 2 else w[k] + r.standard_normal(w[k].shape) * 0.1
    return w


def test_backward_matches_finite_differences():
    w = _weights()
    rng = np.random.default_rng(0)
    idx, tgt = rng.integers(0, 11, (2, 7)), rng.integers(0, 11, (2, 7))
    _, g = loss_and_grads(w, CFG, idx, tgt)
    for name in w:
        for _ in range(3):
            i = tuple(int(rng.integers(0, s)) for s in w[name].shape)
            old = w[name][i]
            w[name][i] = old + 1e-5
            lp, _ = loss_and_grads(w, CFG, idx, tgt)
            w[name][i] = old - 1e-5
            lm, _ = loss_and_grads(w, CFG, idx, tgt)
            w[name][i] = old
            num = (lp - lm) / 2e-5
            assert abs(num - g[name][i]) <= 1e-6 + 1e-5 * abs(num), (name, num, g[name][i])


def test_training_forward_equals_inference_forward():
    w = _weights(2)
    idx = np.random.default_rng(3).integers(0, 11, (1, 9))
    loss, _ = loss_and_grads(w, CFG, idx, idx)
    lg = Transformer(CFG, w).logits_full(idx[0])
    lp = lg - lg.max(-1, keepdims=True)
    lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
    assert np.isclose(loss, -lp[np.arange(9), idx[0]].mean())


def test_a_few_steps_reduce_loss():
    w = init_weights(CFG, 0)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 11, (4, 16))
    tgt = (idx + 1) % 11               # learnable rule: next = current + 1
    opt = AdamW(w, lr=1e-2)
    first, _ = loss_and_grads(w, CFG, idx, tgt)
    for _ in range(60):
        loss, g = loss_and_grads(w, CFG, idx, tgt)
        opt.step(w, g, 1e-2)
    assert loss < 0.3 * first
