"""Engine 3 — Script Engine.

Generates spoken Hinglish in five timed beats:
  0–2s   : extreme curiosity hook
  2–8s   : curiosity escalation
  8–18s  : reveal
  18–25s : twist
  25–30s : loop ending (forces re-watch)

Rules enforced:
  • max 8 words per sentence
  • spoken Hinglish only — never textbook
  • no "psychology says", "studies show", etc.
  • last sentence must loop back to the hook
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any

from ..core.filter import BrainHoldFilter
from ..core.llm import LLM
from ..core.logger import get_logger
from .audience_psychology import PsychProfile

log = get_logger(__name__)

BEATS = [
    ("hook",       0,  2,  "extreme curiosity hook — must be a hook the viewer cannot ignore"),
    ("escalation", 2,  8,  "raise the stakes — make them NEED the answer"),
    ("reveal",     8,  18, "deliver the answer with concrete specifics"),
    ("twist",      18, 25, "introduce one unexpected angle that recontextualizes the reveal"),
    ("loop",       25, 30, "loop ending that mirrors the hook so re-watch feels natural"),
]


@dataclass
class ScriptBeat:
    name: str
    start_sec: float
    end_sec: float
    sentences: list[str]


@dataclass
class Script:
    channel_id: str
    topic: str
    title: str
    description: str
    tags: list[str]
    pinned_comment: str
    beats: list[ScriptBeat]
    emotion: str
    psych: dict[str, Any]
    full_text: str = ""
    fingerprint: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


class ScriptEngine:
    def __init__(self) -> None:
        self.llm = LLM()
        self.filter = BrainHoldFilter()

    def generate(self, channel: dict[str, Any], psych: PsychProfile) -> Script:
        beats_spec = "\n".join(
            f"  - {b[0]} ({b[1]}-{b[2]}s): {b[3]}" for b in BEATS
        )
        forbidden = channel.get("forbidden_phrases", [])
        archetypes = channel.get("hook_archetypes", [])

        system = f"""\
You are an attention engineer + film editor + scriptwriter for a short-form
Hinglish YouTube channel called "{channel['name']}".

Tone: {", ".join(channel['voice']['tone'])}
Audience: {channel['audience']['age_range']}
Language: spoken Hinglish — natural mix, NEVER textbook.

ABSOLUTE RULES:
 • Max 8 words per sentence.
 • Every sentence must sound spoken, not written.
 • Forbidden phrases: {forbidden}
 • Never start with "in this video", "today", "let's talk about".
 • Last sentence must loop to the hook so re-watch feels seamless.
 • No factual claims you cannot defend ("studies show", "scientists found", "$X overnight").

Hook archetypes you may riff on (do not copy verbatim):
{chr(10).join("  • " + a for a in archetypes)}
"""

        user = f"""\
Topic: {psych.topic}
Pain:  {psych.pain}
Fear:  {psych.fear}
Desire: {psych.desire}
Curiosity: {psych.curiosity}
Target emotion: {psych.emotional_trigger}
Comment trigger to plant: "{psych.comment_trigger}"

Produce JSON:
{{
  "title":         "<≤55 chars, scroll-stopping, NOT clickbait lie>",
  "description":   "<2-3 lines, hashtags okay>",
  "tags":          ["<10-15 short tags>"],
  "pinned_comment":"<a comment we'll pin to seed the conversation>",
  "beats": [
    {{"name":"hook",       "sentences": ["...","..."]}},
    {{"name":"escalation", "sentences": ["...","..."]}},
    {{"name":"reveal",     "sentences": ["...","..."]}},
    {{"name":"twist",      "sentences": ["...","..."]}},
    {{"name":"loop",       "sentences": ["...","..."]}}
  ]
}}

Beat timing (be disciplined — do NOT exceed sentence counts):
{beats_spec}
"""
        data = self.llm.complete_json(system, user, max_tokens=1600, temperature=0.85)

        beats = self._parse_beats(data.get("beats", []))
        full_text = "\n".join(s for b in beats for s in b.sentences)
        script = Script(
            channel_id=_channel_id(channel),
            topic=psych.topic,
            title=str(data.get("title", psych.topic))[:80],
            description=str(data.get("description", "")),
            tags=[str(t) for t in (data.get("tags") or [])][:18],
            pinned_comment=str(data.get("pinned_comment", psych.comment_trigger)),
            beats=beats,
            emotion=psych.emotional_trigger,
            psych=psych.to_dict(),
            full_text=full_text,
            fingerprint={
                "hook":   beats[0].sentences[0] if beats and beats[0].sentences else "",
                "ending": beats[-1].sentences[-1] if beats and beats[-1].sentences else "",
                "topic":  psych.topic,
                "structure": "|".join(b.name for b in beats),
            },
        )

        # Heuristic gate — cheap before TTS spend.
        result = self.filter.score_script(s for b in script.beats for s in b.sentences)
        log.info("[script] brain-hold score=%.2f passes=%s reasons=%s",
                 result.score, result.passes, result.reasons[:3])
        if not result.passes:
            log.info("[script] regenerating once due to filter fail")
            return self._repair(channel, psych, script, result.reasons + result.suggestions)
        return script

    def _repair(self, channel: dict[str, Any], psych: PsychProfile,
                draft: Script, hints: list[str]) -> Script:
        system = "You repair short-form Hinglish scripts. Keep all rules. Apply hints."
        user = f"""\
Original draft (each line is one sentence):

{draft.full_text}

Issues to fix:
{chr(10).join('- ' + h for h in hints)}

Constraints:
 • Max 8 words per sentence.
 • Beats must remain hook / escalation / reveal / twist / loop.
 • Spoken Hinglish.
 • Last sentence must loop back to first.

Return JSON in the same shape:
{{
 "title":..., "description":..., "tags":[...], "pinned_comment":...,
 "beats":[{{"name":..., "sentences":[...]}}, ...]
}}
"""
        data = self.llm.complete_json(system, user, max_tokens=1500, temperature=0.7)
        beats = self._parse_beats(data.get("beats", []))
        full_text = "\n".join(s for b in beats for s in b.sentences)
        return Script(
            channel_id=draft.channel_id,
            topic=draft.topic,
            title=str(data.get("title", draft.title))[:80],
            description=str(data.get("description", draft.description)),
            tags=[str(t) for t in (data.get("tags") or draft.tags)][:18],
            pinned_comment=str(data.get("pinned_comment", draft.pinned_comment)),
            beats=beats,
            emotion=draft.emotion,
            psych=draft.psych,
            full_text=full_text,
            fingerprint={
                "hook":   beats[0].sentences[0] if beats and beats[0].sentences else "",
                "ending": beats[-1].sentences[-1] if beats and beats[-1].sentences else "",
                "topic":  draft.topic,
                "structure": "|".join(b.name for b in beats),
            },
        )

    @staticmethod
    def _parse_beats(raw: list[dict[str, Any]]) -> list[ScriptBeat]:
        spec = {b[0]: (b[1], b[2]) for b in BEATS}
        out: list[ScriptBeat] = []
        for b in raw:
            name = str(b.get("name", "")).strip().lower()
            if name not in spec:
                continue
            sentences = [_clean_sentence(s) for s in (b.get("sentences") or []) if str(s).strip()]
            sentences = [s for s in sentences if s]
            start, end = spec[name]
            out.append(ScriptBeat(name=name, start_sec=start, end_sec=end, sentences=sentences))
        # Ensure all five beats exist; fill missing with empty so downstream code never crashes.
        existing = {b.name for b in out}
        for n, s, e, _ in BEATS:
            if n not in existing:
                out.append(ScriptBeat(n, s, e, []))
        out.sort(key=lambda x: x.start_sec)
        return out


def _clean_sentence(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    # Strip leading bullet punctuation if model added it.
    s = re.sub(r"^[-\*\u2022]\s*", "", s)
    return s


def _channel_id(channel: dict[str, Any]) -> str:
    name = channel.get("name", "").lower()
    if "decoder" in name:
        return "human_decoder"
    if "money" in name:
        return "ai_money_lab"
    if "drama" in name:
        return "dramaverse"
    return name.replace(" ", "_")
