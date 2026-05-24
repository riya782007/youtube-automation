"""Engine 6 — Voice Engine.

Sarvam primary → ElevenLabs fallback. Normalizes text for natural Hinglish:
  • numbers and currency to spoken words ($1000 → "one thousand dollars")
  • removes <<markers>> for providers that don't accept SSML
  • respects emotion-driven speed/pitch from EmotionPlan

Falls back to a silent placeholder mp3 if no provider is reachable, so the
rest of the pipeline keeps running and the failure is visible in logs.
"""
from __future__ import annotations

import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..adapters import elevenlabs as el
from ..adapters import sarvam
from ..core.logger import get_logger
from ..core.state import ARTIFACT_DIR
from .emotion_engine import EmotionPlan

log = get_logger(__name__)


@dataclass
class VoiceArtifact:
    audio_path: Path
    provider: str           # "sarvam" | "elevenlabs" | "placeholder"
    seconds_estimate: float
    text_used: str


class VoiceEngine:
    def __init__(self) -> None:
        pass

    def render(self, *, channel: dict[str, Any], video_id: str,
               annotated_sentences: list[str], emotion_plan: EmotionPlan) -> VoiceArtifact:
        text = self._for_tts(annotated_sentences)
        out = ARTIFACT_DIR / video_id / "voice.mp3"

        ch_voice = channel.get("voice", {})
        speed = float(emotion_plan.voice.get("speed", 1.0))
        pitch = float(emotion_plan.voice.get("pitch", 1.0))

        # 1) Sarvam
        path = sarvam.synthesize(
            text, out_path=out,
            voice=ch_voice.get("sarvam_voice", "anushka"),
            speed=speed, pitch=pitch,
        )
        if path is not None:
            return VoiceArtifact(path, "sarvam", _estimate_seconds(text, speed), text)

        # 2) ElevenLabs
        path = el.synthesize(
            text, out_path=out,
            voice_id_env=ch_voice.get("elevenlabs_voice_env"),
            speed=speed,
        )
        if path is not None:
            return VoiceArtifact(path, "elevenlabs", _estimate_seconds(text, speed), text)

        # 3) Placeholder so pipeline keeps going
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_silent_wav(out.with_suffix(".wav"), seconds=_estimate_seconds(text, speed))
        log.warning("[voice] all providers failed → placeholder silent file at %s", out)
        return VoiceArtifact(out.with_suffix(".wav"), "placeholder",
                             _estimate_seconds(text, speed), text)

    # ---- Text normalization ----------------------------------------------
    @staticmethod
    def _for_tts(annotated: list[str]) -> str:
        joined = " ".join(annotated)
        joined = _strip_markers(joined)
        joined = _spoken_numbers(joined)
        joined = _spoken_currency(joined)
        joined = re.sub(r"\s+", " ", joined).strip()
        return joined


def _strip_markers(t: str) -> str:
    # Convert markers to natural pause cues that most TTS handles via punctuation.
    t = t.replace("<<pause:short>>", ", ")
    t = t.replace("<<pause:medium>>", ". ")
    t = t.replace("<<pause:long>>", "... ")
    t = t.replace("<<breath>>", " ")  # most providers infer breath from punctuation
    t = t.replace("<<emph>>", "").replace("<</emph>>", "")
    return t


def _spoken_currency(t: str) -> str:
    def repl(m: re.Match) -> str:
        sym, amt = m.group(1), m.group(2).replace(",", "")
        unit = {"$": "dollars", "₹": "rupees", "€": "euros", "£": "pounds"}.get(sym, "")
        return f"{_int_to_words(int(amt))} {unit}".strip()
    return re.sub(r"([\$₹€£])\s?([0-9][0-9,]*)", repl, t)


def _spoken_numbers(t: str) -> str:
    # Only convert standalone integers; leave dates/years alone if 4 digits start with 19/20.
    def repl(m: re.Match) -> str:
        n = m.group(0)
        if len(n) == 4 and (n.startswith("19") or n.startswith("20")):
            return n
        return _int_to_words(int(n))
    return re.sub(r"\b\d+\b", repl, t)


def _int_to_words(n: int) -> str:
    if n == 0:
        return "zero"
    units = ["", "one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
             "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty",
            "seventy", "eighty", "ninety"]

    def below_thousand(x: int) -> str:
        parts = []
        if x >= 100:
            parts.append(units[x // 100] + " hundred")
            x %= 100
            if x:
                parts.append("and")
        if x >= 20:
            t = tens[x // 10]
            u = x % 10
            parts.append(t + ("-" + units[u] if u else ""))
        elif x > 0:
            parts.append(units[x])
        return " ".join(parts).strip()

    if n < 0:
        return "minus " + _int_to_words(-n)
    chunks = []
    for power, label in [(1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand")]:
        if n >= power:
            chunks.append(below_thousand(n // power) + " " + label)
            n %= power
    if n > 0:
        chunks.append(below_thousand(n))
    return " ".join(chunks).strip()


def _estimate_seconds(text: str, speed: float) -> float:
    words = max(1, len(text.split()))
    base_wpm = 165
    return round((words / base_wpm) * 60 / max(0.6, speed), 2)


def _write_silent_wav(path: Path, *, seconds: float, sample_rate: int = 24000) -> None:
    n = int(max(1.0, seconds) * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * n)
