"""Train the tiny char-level models in pure numpy (hand-written backward pass).

Why train at all, when a server only needs *some* weights? Two reasons:

1. Random weights produce noise, and noise hides bugs. A model that writes
   recognisable pseudo-Shakespeare makes a broken cache or a wrong RoPE offset
   visible to the naked eye.
2. Speculative decoding is only interesting when the draft model actually
   predicts the target. Two random models agree ~1/vocab of the time; two
   models trained on the same text agree most of the time - which is the
   regime real deployments live in.

The backward pass is derived by hand for exactly the forward in
``model.py`` (RMSNorm, RoPE, GQA causal attention, SwiGLU) and is checked
against finite differences in ``tests/test_train.py``.

    python -m tinyserve.train --preset target --steps 2000 --out target.npz
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from .model import ModelConfig, Transformer, apply_rope, init_weights, rope_tables
from .tokenizer import CharTokenizer

DATA_PATH = Path(__file__).parent / "data" / "tinyshakespeare.txt"

PRESETS = {
    # ~0.8M params. Big enough to learn words, small enough to train in minutes.
    "target": dict(dim=128, n_layers=4, n_heads=4, n_kv_heads=2, ffn_dim=352),
    # ~0.06M params, 1 layer: ~4x cheaper per forward, the draft for speculation.
    "draft": dict(dim=64, n_layers=1, n_heads=2, n_kv_heads=1, ffn_dim=176),
}


# --------------------------------------------------------------------------
# forward with saved activations
# --------------------------------------------------------------------------


def _rms_fwd(x, g, eps):
    r = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    xh = x * r
    return xh * g, (xh, r, g)


def _rms_bwd(dy, cache):
    xh, r, g = cache
    dg = np.sum(dy * xh, axis=tuple(range(dy.ndim - 1)))
    dxh = dy * g
    # d/dx of x*r(x): the r term couples every element of the row.
    dx = r * (dxh - xh * np.mean(dxh * xh, axis=-1, keepdims=True))
    return dx, dg


def loss_and_grads(w: dict[str, np.ndarray], cfg: ModelConfig, idx: np.ndarray,
                   targets: np.ndarray, rope=None) -> tuple[float, dict[str, np.ndarray]]:
    """Mean cross-entropy over (B, T) and its gradient w.r.t. every weight."""
    B, T = idx.shape
    H, KVH, hd, D = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.dim
    g = H // KVH
    eps = cfg.norm_eps
    cos, sin = rope if rope is not None else rope_tables(cfg, w["tok_emb"].dtype)
    cos, sin = cos[:T], sin[:T]
    scale = 1.0 / math.sqrt(hd)
    mask = np.triu(np.ones((T, T), dtype=bool), 1)

    x = w["tok_emb"][idx]  # (B, T, D)
    caches = []
    for i in range(cfg.n_layers):
        p = f"layers.{i}."
        a, c_an = _rms_fwd(x, w[p + "attn_norm"], eps)
        q = apply_rope((a @ w[p + "wq"]).reshape(B, T, H, hd), cos, sin)
        k = apply_rope((a @ w[p + "wk"]).reshape(B, T, KVH, hd), cos, sin)
        v = (a @ w[p + "wv"]).reshape(B, T, KVH, hd)
        qh = q.transpose(0, 2, 1, 3)                       # (B, H, T, hd)
        kh = np.repeat(k, g, axis=2).transpose(0, 2, 1, 3)  # head h uses kv head h//g
        vh = np.repeat(v, g, axis=2).transpose(0, 2, 1, 3)
        s = (qh @ kh.transpose(0, 1, 3, 2)) * scale
        s[..., mask] = -np.inf
        s -= s.max(-1, keepdims=True)
        P = np.exp(s)
        P /= P.sum(-1, keepdims=True)
        o = (P @ vh).transpose(0, 2, 1, 3).reshape(B, T, H * hd)
        x = x + o @ w[p + "wo"]
        m, c_fn = _rms_fwd(x, w[p + "ffn_norm"], eps)
        u = m @ w[p + "w1"]
        gate = m @ w[p + "w3"]
        sig = 1.0 / (1.0 + np.exp(-u))
        act = u * sig
        hid = act * gate
        x = x + hid @ w[p + "w2"]
        caches.append((a, c_an, qh, kh, vh, P, o, m, c_fn, u, gate, sig, act, hid))

    f, c_f = _rms_fwd(x, w["norm"], eps)
    logits = f @ w["lm_head"]
    logits -= logits.max(-1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(-1, keepdims=True)
    N = B * T
    flat_t = targets.reshape(-1)
    pf = probs.reshape(N, -1)
    loss = float(-np.mean(np.log(pf[np.arange(N), flat_t] + 1e-12)))

    # ---------------------------------------------------------------- backward
    grads: dict[str, np.ndarray] = {}
    dlog = pf.copy()
    dlog[np.arange(N), flat_t] -= 1.0
    dlog = (dlog / N).reshape(B, T, -1)
    grads["lm_head"] = f.reshape(N, D).T @ dlog.reshape(N, -1)
    dx, grads["norm"] = _rms_bwd(dlog @ w["lm_head"].T, c_f)

    for i in reversed(range(cfg.n_layers)):
        p = f"layers.{i}."
        a, c_an, qh, kh, vh, P, o, m, c_fn, u, gate, sig, act, hid = caches[i]
        # SwiGLU
        grads[p + "w2"] = hid.reshape(N, -1).T @ dx.reshape(N, D)
        dhid = dx @ w[p + "w2"].T
        dgate = dhid * act
        du = dhid * gate * (sig * (1.0 + u * (1.0 - sig)))  # silu'(u)
        grads[p + "w1"] = m.reshape(N, D).T @ du.reshape(N, -1)
        grads[p + "w3"] = m.reshape(N, D).T @ dgate.reshape(N, -1)
        dm = du @ w[p + "w1"].T + dgate @ w[p + "w3"].T
        dxm, grads[p + "ffn_norm"] = _rms_bwd(dm, c_fn)
        dx = dx + dxm
        # attention
        grads[p + "wo"] = o.reshape(N, -1).T @ dx.reshape(N, D)
        do = (dx @ w[p + "wo"].T).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        dP = do @ vh.transpose(0, 1, 3, 2)
        dvh = P.transpose(0, 1, 3, 2) @ do
        dS = P * (dP - np.sum(dP * P, -1, keepdims=True)) * scale
        dqh = dS @ kh
        dkh = dS.transpose(0, 1, 3, 2) @ qh
        # undo the GQA repeat: gradients of shared KV heads sum over their group
        dk = dkh.transpose(0, 2, 1, 3).reshape(B, T, KVH, g, hd).sum(3)
        dv = dvh.transpose(0, 2, 1, 3).reshape(B, T, KVH, g, hd).sum(3)
        # RoPE is a rotation, so its adjoint is the inverse rotation (-sin).
        dq = apply_rope(dqh.transpose(0, 2, 1, 3), cos, -sin).reshape(B, T, -1)
        dk = apply_rope(dk, cos, -sin).reshape(B, T, -1)
        dv = dv.reshape(B, T, -1)
        af = a.reshape(N, D)
        grads[p + "wq"] = af.T @ dq.reshape(N, -1)
        grads[p + "wk"] = af.T @ dk.reshape(N, -1)
        grads[p + "wv"] = af.T @ dv.reshape(N, -1)
        da = dq @ w[p + "wq"].T + dk @ w[p + "wk"].T + dv @ w[p + "wv"].T
        dxa, grads[p + "attn_norm"] = _rms_bwd(da, c_an)
        dx = dx + dxa

    demb = np.zeros_like(w["tok_emb"])
    np.add.at(demb, idx.reshape(-1), dx.reshape(N, D))
    grads["tok_emb"] = demb
    return loss, grads


# --------------------------------------------------------------------------
# AdamW + loop
# --------------------------------------------------------------------------


class AdamW:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.05):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, *betas, eps, weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr):
        self.t += 1
        b1c, b2c = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            upd = (self.m[k] / b1c) / (np.sqrt(self.v[k] / b2c) + self.eps)
            if params[k].ndim == 2 and k != "tok_emb":  # no decay on norms/embeddings
                upd = upd + self.wd * params[k]
            params[k] -= (lr * upd).astype(params[k].dtype)


def train(preset: str, steps: int, batch: int, seq: int, lr: float, seed: int,
          out: Path, log_every: int = 100) -> dict:
    text = DATA_PATH.read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    data = tok.encode_np(text)
    split = int(0.9 * len(data))
    tr, va = data[:split], data[split:]
    # max_seq_len = the training context: RoPE positions the model never saw in
    # training are out of distribution, and generation visibly degrades past
    # them. Advertising a longer context than was trained would be a lie.
    cfg = ModelConfig(vocab_size=tok.vocab_size, max_seq_len=seq, **PRESETS[preset])
    w = init_weights(cfg, seed)
    rope = rope_tables(cfg)
    opt = AdamW(w, lr=lr)
    rng = np.random.default_rng(seed)

    def batch_of(src):
        ix = rng.integers(0, len(src) - seq - 1, batch)
        xb = np.stack([src[i:i + seq] for i in ix])
        yb = np.stack([src[i + 1:i + seq + 1] for i in ix])
        return xb, yb

    warmup = max(1, steps // 20)
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        # linear warmup then cosine to 10% of peak
        if step <= warmup:
            cur = lr * step / warmup
        else:
            prog = (step - warmup) / max(1, steps - warmup)
            cur = lr * (0.1 + 0.45 * (1 + math.cos(math.pi * prog)))
        xb, yb = batch_of(tr)
        loss, grads = loss_and_grads(w, cfg, xb, yb, rope)
        # global-norm clipping: char models spike early when attention is sharp
        gn = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        if gn > 1.0:
            for g in grads.values():
                g *= 1.0 / gn
        opt.step(w, grads, cur)
        if step % log_every == 0 or step == steps:
            print(f"[{preset}] step {step:5d} loss {loss:.3f} lr {cur:.2e} "
                  f"{time.perf_counter() - t0:6.1f}s", flush=True)

    vals = []
    for _ in range(20):
        xb, yb = batch_of(va)
        vals.append(loss_and_grads(w, cfg, xb, yb, rope)[0])
    stats = {"preset": preset, "steps": steps, "batch": batch, "seq": seq,
             "val_loss": round(float(np.mean(vals)), 4),
             "train_seconds": round(time.perf_counter() - t0, 1)}
    Transformer(cfg, w).save(out, {"tokenizer": tok.chars, "train": stats})
    print(f"[{preset}] saved {out}  val_loss={stats['val_loss']}")
    return stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", choices=sorted(PRESETS), default="target")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    train(a.preset, a.steps, a.batch, a.seq, a.lr, a.seed, a.out)


if __name__ == "__main__":
    main()
