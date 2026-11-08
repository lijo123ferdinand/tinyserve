"""Character-level tokenizer.

Char-level keeps the project self-contained (no BPE merges to ship) and makes
streaming detokenization trivial: one token is always one complete character,
so there is no "half a UTF-8 code point" buffering problem that real servers
have to solve for byte-level BPE.
"""

from __future__ import annotations

import numpy as np


class CharTokenizer:
    def __init__(self, chars: str):
        if len(set(chars)) != len(chars):
            raise ValueError("duplicate characters in vocabulary")
        self.chars = chars
        self._stoi = {c: i for i, c in enumerate(chars)}
        # Unknown characters map to space if we have one: a prompt containing
        # an emoji should degrade gracefully, not 500 the server.
        self._unk = self._stoi.get(" ", 0)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls("".join(sorted(set(text))))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self._stoi.get(c, self._unk) for c in text]

    def encode_np(self, text: str) -> np.ndarray:
        return np.fromiter((self._stoi.get(c, self._unk) for c in text), dtype=np.int64,
                           count=len(text))

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)
