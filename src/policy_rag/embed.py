"""Local embedding models. Nothing here calls a paid API.

- Model2VecEmbedder: a real static embedding model (minishlab/potion-retrieval-32M, 512 dims),
  downloaded once from Hugging Face and run on CPU with numpy. This is what the eval uses.
- HashEmbedder: a deterministic bag-of-character-n-grams vector with no download. It is a
  lexical stand-in for unit tests that must run offline; it is NOT a semantic model and is
  never used for reported retrieval numbers.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "minishlab/potion-retrieval-32M"


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized rows."""
        ...


def _normalize(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float32, copy=False)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class Model2VecEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        from model2vec import StaticModel

        self.model = StaticModel.from_pretrained(model_name)
        self.name = f"model2vec:{model_name}"
        self.dim = int(self.model.dim)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.asarray(self.model.encode(texts)))


class HashEmbedder:
    """Feature hashing over word unigrams and character 4-grams. Test-only, lexical."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"hash:{dim}"

    def _features(self, text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        feats = [f"w:{w}" for w in words]
        for w in words:
            padded = f"_{w}_"
            feats.extend(f"c:{padded[i : i + 4]}" for i in range(max(1, len(padded) - 3)))
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for f in self._features(text):
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        return _normalize(out)
