"""Engine 2 — Audience Psychology.

For every video, BEFORE scripting, lock the six psychological pillars:
  Pain, Fear, Desire, Curiosity, EmotionalTrigger, CommentTrigger

This is what differentiates "another AI video" from "this hit me".
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from ..core.llm import LLM
from ..core.logger import get_logger

log = get_logger(__name__)


@dataclass
class PsychProfile:
    pain: str
    fear: str
    desire: str
    curiosity: str
    emotional_trigger: str
    comment_trigger: str
    channel_id: str
    topic: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AudiencePsychology:
    def __init__(self) -> None:
        self.llm = LLM()

    def profile(self, channel: dict[str, Any], topic: str) -> PsychProfile:
        ch_name = channel["name"]
        psych = channel["audience"]["core_psychology"]
        comment_triggers = channel.get("comment_triggers", [])

        system = (
            "You are a behavioral economist + human psychologist. "
            "Reduce a topic to its sharpest emotional levers in ONE line per field. "
            "Output spoken Hinglish where it sounds natural; otherwise English. "
            "Be specific — never say 'people'; name the actual reaction."
        )
        user = f"""\
Channel: {ch_name}
Audience age: {channel['audience']['age_range']}
Known pains:   {psych.get('pains')}
Known desires: {psych.get('desires')}
Topic for this video: {topic}

Return JSON:
{{
  "pain":              "<the precise discomfort the viewer feels right now>",
  "fear":              "<what they secretly fear about this topic>",
  "desire":            "<what they want to be true>",
  "curiosity":         "<the unanswered question we will dangle>",
  "emotional_trigger": "<one of: surprise|romantic|serious|suspense|excitement|curiosity|empathy>",
  "comment_trigger":   "<one short prompt that forces them to comment — pick from: {comment_triggers} or invent something tighter>"
}}
Each value MUST be ≤14 words. No fluff. No 'people often' phrasing.
"""
        data = self.llm.complete_json(system, user, max_tokens=600, temperature=0.5)

        prof = PsychProfile(
            pain=str(data.get("pain", "")).strip() or "unspecified discomfort",
            fear=str(data.get("fear", "")).strip() or "missing out",
            desire=str(data.get("desire", "")).strip() or "feeling understood",
            curiosity=str(data.get("curiosity", "")).strip() or "what happens next",
            emotional_trigger=_normalize_emotion(data.get("emotional_trigger", "curiosity")),
            comment_trigger=str(data.get("comment_trigger", "")).strip() or "What do you think?",
            channel_id=_channel_id(channel),
            topic=topic,
        )
        log.info("[psych] topic=%r emotion=%s comment=%r",
                 topic, prof.emotional_trigger, prof.comment_trigger)
        return prof


_VALID = {"surprise", "romantic", "serious", "suspense", "excitement", "curiosity", "empathy"}


def _normalize_emotion(raw: Any) -> str:
    s = str(raw).strip().lower()
    return s if s in _VALID else "curiosity"


def _channel_id(channel: dict[str, Any]) -> str:
    name = channel.get("name", "").lower()
    if "decoder" in name:
        return "human_decoder"
    if "money" in name:
        return "ai_money_lab"
    if "drama" in name:
        return "dramaverse"
    return name.replace(" ", "_")
