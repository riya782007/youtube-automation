"""Engine 11 — Quality Gate.

Scores every video plan on 6 axes (0-10):
  • hook
  • voice
  • visual_density
  • emotion
  • editing
  • retention_prediction

Reject if average < 8.5 → caller regenerates. The cheap heuristic gate (in
core/filter.py) already ran during script generation; this gate is the final
LLM-backed sanity check before render.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Any

from ..core.llm import LLM
from ..core.logger import get_logger

log = get_logger(__name__)

REJECT_THRESHOLD = float(os.getenv("QUALITY_REJECT_THRESHOLD", "8.5"))


@dataclass
class QualityScore:
    hook: float
    voice: float
    visual_density: float
    emotion: float
    editing: float
    retention_prediction: float
    notes: str = ""

    def average(self) -> float:
        vals = [self.hook, self.voice, self.visual_density,
                self.emotion, self.editing, self.retention_prediction]
        return round(sum(vals) / len(vals), 2)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["average"] = self.average()
        d["passes"] = self.passes()
        return d

    def passes(self) -> bool:
        return self.average() >= REJECT_THRESHOLD


class QualityGate:
    def __init__(self) -> None:
        self.llm = LLM()

    def score(self, plan_summary: dict[str, Any]) -> QualityScore:
        system = (
            "You are a YouTube Shorts retention specialist. "
            "Score the given short-form Hinglish video plan on 6 axes "
            "(0–10). Be HARSH. 8 means good; 9 means excellent; 10 must be earned. "
            "Return only JSON."
        )
        user = f"""\
Channel: {plan_summary.get('channel_id')}
Title: {plan_summary.get('title')}
Duration: {plan_summary.get('duration_sec')}s
Emotion: {plan_summary.get('emotion')}
Hook (first sentence): {plan_summary.get('hook_sentence')}
Loop (last sentence): {plan_summary.get('loop_sentence')}
Asset mix: {plan_summary.get('asset_mix')}
Reset cadence: every {plan_summary.get('avg_reset_interval')}s
Sentence count: {plan_summary.get('sentence_count')}
Comment trigger: {plan_summary.get('comment_trigger')}
Pinned comment: {plan_summary.get('pinned_comment')}
Forbidden phrase hits: {plan_summary.get('forbidden_hits')}

Score JSON:
{{
  "hook": <float>,
  "voice": <float>,
  "visual_density": <float>,
  "emotion": <float>,
  "editing": <float>,
  "retention_prediction": <float>,
  "notes": "<one sentence on the single biggest weakness>"
}}
"""
        data = self.llm.complete_json(system, user, max_tokens=400, temperature=0.2)

        def f(k: str, default: float) -> float:
            try:
                return float(data.get(k, default))
            except Exception:
                return default

        score = QualityScore(
            hook=f("hook", 7.0),
            voice=f("voice", 7.0),
            visual_density=f("visual_density", 7.0),
            emotion=f("emotion", 7.0),
            editing=f("editing", 7.0),
            retention_prediction=f("retention_prediction", 7.0),
            notes=str(data.get("notes", "")),
        )
        log.info("[quality] avg=%.2f passes=%s notes=%s",
                 score.average(), score.passes(), score.notes[:80])
        return score
