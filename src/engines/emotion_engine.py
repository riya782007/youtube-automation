"""Engine 5 — Emotion Engine.

Maps the chosen emotion → voice parameters (speed/pitch), pause patterns,
breath probability and visual reset cadence. Annotates the script with
provider-agnostic markers like <<pause:long>> and <<breath>>.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Any

from ..core.logger import get_logger
from ..core.state import load_emotions

log = get_logger(__name__)


@dataclass
class EmotionPlan:
    emotion: str
    voice: dict[str, float]                       # {speed, pitch}
    pause_ms: dict[str, int]
    breath_probability: float
    visual: dict[str, float]
    music_intensity: float
    annotated_script: list[str]                   # sentences with markers
    ssml_hint: str                                # provider-agnostic SSML-ish blob

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EmotionEngine:
    def __init__(self, *, seed: int | None = None) -> None:
        self.seed = seed
        self.cfg = load_emotions()
        self.markers = self.cfg.get("markers", {})

    def plan(self, sentences: list[str], emotion: str) -> EmotionPlan:
        rng = random.Random(self.seed)
        emap = self.cfg["emotions"].get(emotion) or self.cfg["emotions"]["curiosity"]

        pause = emap["pause_ms"]
        breath_p = float(emap["breath_probability"])
        annotated: list[str] = []
        ssml_lines: list[str] = []

        for i, raw in enumerate(sentences):
            s = raw.strip()
            if not s:
                continue

            # Punctuation-driven pause selection.
            if s.endswith("..."):
                tail_marker = self.markers["pause_long"]
                tail_ms = pause["dramatic"]
            elif s.endswith("?") or s.endswith("!"):
                tail_marker = self.markers["pause_medium"]
                tail_ms = int(pause["sentence"] * 1.4)
            else:
                tail_marker = self.markers["pause_short"]
                tail_ms = pause["sentence"]

            # Hook (i==0) and twist sentences get a breath.
            with_breath = (i == 0 or "twist" in s.lower()) or rng.random() < breath_p

            line = s
            if with_breath:
                line = f"{self.markers['breath']} {line}"

            # Emphasis: capitalised tokens get wrapped.
            line = _wrap_emphasis(line, self.markers)

            annotated.append(f"{line} {tail_marker}")
            ssml_lines.append(
                f"<prosody rate=\"{emap['voice']['speed']}\" pitch=\"{emap['voice']['pitch']}\">"
                f"{_strip_markers(s)}</prosody>"
                f"<break time=\"{tail_ms}ms\"/>"
            )

        ssml = (
            "<speak>"
            + ("<break time=\"120ms\"/>".join(ssml_lines))
            + "</speak>"
        )
        log.info("[emotion] mapped emotion=%s sentences=%d breath_p=%.2f",
                 emotion, len(annotated), breath_p)
        return EmotionPlan(
            emotion=emotion,
            voice=emap["voice"],
            pause_ms=pause,
            breath_probability=breath_p,
            visual=emap["visual"],
            music_intensity=float(emap["music_intensity"]),
            annotated_script=annotated,
            ssml_hint=ssml,
        )


def _wrap_emphasis(line: str, markers: dict[str, str]) -> str:
    """Wrap ALL-CAPS tokens (≥3 chars) in emphasis markers."""
    out_words = []
    for w in line.split():
        if w.isupper() and len(w) >= 3 and w.isalpha():
            out_words.append(f"{markers['emphasis_open']}{w}{markers['emphasis_close']}")
        else:
            out_words.append(w)
    return " ".join(out_words)


def _strip_markers(s: str) -> str:
    import re
    return re.sub(r"<<[^>]+>>", "", s).strip()
