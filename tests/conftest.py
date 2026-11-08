import numpy as np
import pytest

from tinyserve.model import ModelConfig, Transformer
from tinyserve.tokenizer import CharTokenizer
from tinyserve.train import DATA_PATH


@pytest.fixture(scope="session")
def tok() -> CharTokenizer:
    return CharTokenizer.from_text(DATA_PATH.read_text(encoding="utf-8"))


def tiny_model(vocab: int, seed: int = 0, layers: int = 2, dim: int = 32, heads: int = 4,
               kv_heads: int = 2, max_seq_len: int = 256) -> Transformer:
    cfg = ModelConfig(vocab_size=vocab, dim=dim, n_layers=layers, n_heads=heads,
                      n_kv_heads=kv_heads, ffn_dim=2 * dim + 16, max_seq_len=max_seq_len)
    m = Transformer(cfg, seed=seed)
    # Default init is tiny (std 0.02) which makes every logit ~equal; scale up
    # so attention is peaky and argmax decisions are far from ties - a
    # random-but-opinionated model is a much stronger test of the cache paths.
    for k, w in m.w.items():
        if w.ndim == 2:
            w *= 20.0
    return m


@pytest.fixture(scope="session")
def model(tok) -> Transformer:
    return tiny_model(tok.vocab_size)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
