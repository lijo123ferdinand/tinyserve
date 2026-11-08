"""tinyserve - a vLLM-style LLM inference server built from scratch in numpy."""

from __future__ import annotations

from pathlib import Path

from .engine import CompletionOutput, EngineConfig, LLMEngine, StepOutput
from .model import ModelConfig, Transformer
from .sampling import SamplingParams
from .speculative import SpeculativeDecoder
from .tokenizer import CharTokenizer

__version__ = "0.1.0"

WEIGHTS_DIR = Path(__file__).parent / "weights"


def load_pretrained(name: str = "target") -> tuple[Transformer, CharTokenizer]:
    """Load one of the bundled char-level Shakespeare models ("target" or "draft")."""
    path = WEIGHTS_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"no bundled model {name!r} (looked for {path})")
    model, meta = Transformer.load(path)
    return model, CharTokenizer(meta["tokenizer"])


__all__ = [
    "CharTokenizer", "CompletionOutput", "EngineConfig", "LLMEngine", "ModelConfig",
    "SamplingParams", "SpeculativeDecoder", "StepOutput", "Transformer", "load_pretrained",
]
