"""Engine 8 — Attention Reset.

Human attention decays in seconds. We schedule a visual reset every 1.8–4s.
A reset = one of: zoom_punch, camera_shake, meme_insert, flash, emoji_sticker,
sfx_hit, scene_change. Cadence is pulled from EmotionPlan.visual.reset_interval_sec.

Output is a list of `ResetCue` records the renderer schedules at exact seconds.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Any

from ..core.logger import get_logger
from .emotion_engine import EmotionPlan

log = get_logger(__name__)


RESET_TYPES = ["zoom_punch", "camera_shake", "meme_insert", "flash",
               "emoji_sticker", "sfx_hit", "scene_change"]


@dataclass
class ResetCue:
    at_sec: float
    kind: str
    intensity: float           # 0..1
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AttentionReset:
    def __init__(self, *, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def schedule(self, *, total_seconds: float, emotion_plan: EmotionPlan,
                 hard_cuts_at: list[float] | None = None) -> list[ResetCue]:
        if total_seconds <= 0:
            return []
        base = float(emotion_plan.visual.get("reset_interval_sec", 2.4))
        shake = float(emotion_plan.visual.get("shake", 0.3))
        flash = float(emotion_plan.visual.get("flash", 0.2))
        zoom = float(emotion_plan.visual.get("zoom", 0.5))
        weights = self._weights(shake, flash, zoom)

        cues: list[ResetCue] = []
        t = 0.0
        # First reset always between 0.6–1.2s — earliest possible attention test.
        t += self.rng.uniform(0.6, 1.2)
        while t < total_seconds:
            kind = self._weighted_pick(weights)
            cues.append(ResetCue(
                at_sec=round(t, 2),
                kind=kind,
                intensity=round(self.rng.uniform(0.55, 0.95), 2),
                note=f"reset@{t:.2f}s",
            ))
            t += self.rng.uniform(max(1.8, base - 0.6), max(2.4, base + 1.2))

        # Force a strong reset at every hard cut (beat boundary).
        for cut in (hard_cuts_at or []):
            if cut <= 0 or cut >= total_seconds:
                continue
            cues.append(ResetCue(
                at_sec=round(cut, 2),
                kind="scene_change",
                intensity=0.95,
                note="beat boundary",
            ))
        cues.sort(key=lambda c: c.at_sec)
        log.info("[attn-reset] scheduled %d cues over %.1fs", len(cues), total_seconds)
        return cues

    @staticmethod
    def _weights(shake: float, flash: float, zoom: float) -> dict[str, float]:
        return {
            "zoom_punch":    0.6 + zoom,
            "camera_shake":  0.4 + shake,
            "meme_insert":   0.7,
            "flash":         0.4 + flash,
            "emoji_sticker": 0.5,
            "sfx_hit":       0.6,
            "scene_change":  0.9,
        }

    def _weighted_pick(self, weights: dict[str, float]) -> str:
        items = list(weights.items())
        total = sum(w for _, w in items) or 1.0
        r = self.rng.uniform(0, total)
        acc = 0.0
        for k, w in items:
            acc += w
            if r <= acc:
                return k
        return items[-1][0]
