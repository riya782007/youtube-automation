"""Embedding helper for anti-repetition memory.

Tries sentence-transformers; falls back to a deterministic char-trigram hash
embedding so the pipeline still works without the heavy ML dep installed.
"""
from __future__ import annotations

import hashlib
import math
import os
from functools import lru_cache
from typing import Iterable

import numpy as np

from .logger import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _model():
    name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer(name)
    except Exception as e:
        log.warning("sentence-transformers unavailable (%s) — using hash fallback", e)
        return None


def embed(text: str) -> list[float]:
    m = _model()
    if m is not None:
        v = m.encode(text, normalize_embeddings=True)
        return v.tolist()
    return _hash_embed(text)


def embed_many(texts: Iterable[str]) -> list[list[float]]:
    return [embed(t) for t in texts]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        # Hash-based vectors share length; mismatch means model swap mid-run.
        return 0.0
    av, bv = np.array(a), np.array(b)
    denom = (np.linalg.norm(av) * np.linalg.norm(bv)) or 1e-12
    return float(np.dot(av, bv) / denom)


# ---------- Fallback hash embedding ------------------------------------------
_DIM = 384  # match MiniLM dim so the system feels consistent


def _hash_embed(text: str) -> list[float]:
    """Char-trigram fingerprint → fixed-dim sparse-ish vector. Not great, but
    survives without ML deps and produces stable cosine similarity."""
    text = (text or "").lower().strip()
    vec = np.zeros(_DIM, dtype=np.float32)
    if not text:
        return vec.tolist()
    padded = f"  {text}  "
    for i in range(len(padded) - 2):
        tri = padded[i : i + 3]
        h = int(hashlib.md5(tri.encode("utf-8")).hexdigest(), 16)
        vec[h % _DIM] += 1.0
    norm = math.sqrt(float((vec * vec).sum())) or 1.0
    vec /= norm
    return vec.tolist()
