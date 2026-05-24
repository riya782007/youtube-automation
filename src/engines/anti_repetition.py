"""Engine 4 — Anti-Repetition Memory.

Computes embeddings over hook / topic / ending / structure and rejects
candidates whose cosine similarity to ANY past entry exceeds the threshold
(default 0.55, configurable via COSINE_REJECT_THRESHOLD).

This is the single biggest defense against "AI slop feel".
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.embeddings import cosine, embed
from ..core.logger import get_logger
from ..core.state import DATA_DIR, load_json, write_json_atomic

log = get_logger(__name__)

THRESHOLD = float(os.getenv("COSINE_REJECT_THRESHOLD", "0.55"))


@dataclass
class RepetitionVerdict:
    accepted: bool
    field: str
    similarity: float
    closest_text: str
    closest_id: str | None


class AntiRepetitionMemory:
    """Stores fingerprints in 4 separate JSON files, one per dimension."""

    FILES = {
        "hook":     "used_hooks.json",
        "topic":    "used_topics.json",
        "ending":   "used_endings.json",
        "structure":"used_structures.json",
    }

    def __init__(self) -> None:
        self.threshold = THRESHOLD

    # ---- Read --------------------------------------------------------------
    def _load(self, field: str) -> dict[str, Any]:
        return load_json(DATA_DIR / self.FILES[field])

    # ---- Check -------------------------------------------------------------
    def check(self, fingerprint: dict[str, str]) -> list[RepetitionVerdict]:
        verdicts: list[RepetitionVerdict] = []
        for field, text in fingerprint.items():
            if field not in self.FILES or not text:
                continue
            v = embed(text)
            store = self._load(field)
            best_sim = 0.0
            best_text = ""
            best_id: str | None = None
            for entry in store.get("entries", []):
                sim = cosine(v, entry.get("embedding", []))
                if sim > best_sim:
                    best_sim = sim
                    best_text = entry.get("text", "")
                    best_id = entry.get("id")
            verdicts.append(RepetitionVerdict(
                accepted=best_sim < self.threshold,
                field=field,
                similarity=best_sim,
                closest_text=best_text,
                closest_id=best_id,
            ))
        return verdicts

    def is_accepted(self, fingerprint: dict[str, str]) -> tuple[bool, list[RepetitionVerdict]]:
        verdicts = self.check(fingerprint)
        ok = all(v.accepted for v in verdicts)
        for v in verdicts:
            log.info("[anti-rep] %-9s sim=%.2f accepted=%s",
                     v.field, v.similarity, v.accepted)
        return ok, verdicts

    # ---- Commit ------------------------------------------------------------
    def commit(self, video_id: str, fingerprint: dict[str, str], channel_id: str) -> None:
        for field, text in fingerprint.items():
            if field not in self.FILES or not text:
                continue
            store = self._load(field)
            entry = {
                "id": video_id,
                "channel_id": channel_id,
                "text": text,
                "embedding": embed(text),
                "created_at": dt.datetime.utcnow().isoformat() + "Z",
            }
            store.setdefault("entries", []).append(entry)
            # Cap memory size — last N per channel — to keep checks fast.
            store["entries"] = _cap_per_channel(store["entries"], cap_per_channel=300)
            write_json_atomic(DATA_DIR / self.FILES[field], store)
        log.info("[anti-rep] committed fingerprint for video=%s", video_id)


def _cap_per_channel(entries: list[dict[str, Any]], cap_per_channel: int) -> list[dict[str, Any]]:
    by_ch: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        by_ch.setdefault(e.get("channel_id", "_"), []).append(e)
    out: list[dict[str, Any]] = []
    for ch, lst in by_ch.items():
        out.extend(lst[-cap_per_channel:])
    # Keep stable order
    out.sort(key=lambda e: e.get("created_at", ""))
    return out
