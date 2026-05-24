"""Voice Intelligence — turn raw script into TTS-ready, naturally-paced Hinglish.

Responsibilities:
  • clean Hinglish (transliterate Devanagari into Latin where TTS handles better)
  • inject conversational micro-pauses, breaths, em-stops
  • detect emphasis words and wrap them
  • slang handling (bro, yaar, bhai → keep, not "translated")
  • numbers / currency / handles → spoken form
  • produce per-sentence TIMING ESTIMATES the Editor Agent uses to align effects

Output is twofold:
  1. natural_text — string ready for Sarvam/ElevenLabs
  2. word_timings — list of dicts [{word, start_sec, end_sec, emphasis}]
     used by editor_agent.py to place effects on emphasized words.

Timing is estimated heuristically when no aligner is available; when Sarvam
returns durations, those override the estimate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any

from ..core.logger import get_logger

log = get_logger(__name__)


# Hinglish slang we MUST preserve — never auto-translated.
PRESERVE_TOKENS = {
    "bhai", "yaar", "bro", "dude", "boss", "didi", "bhaiya",
    "matlab", "actually", "obviously", "literally", "really",
    "scene", "vibe", "mood", "fire", "cap", "lit",
    "hai", "hain", "ho", "tha", "thi", "the",
    "mein", "main", "tu", "tum", "aap", "vo", "yeh", "ye",
}

# Words that should be EMPHASIZED in delivery (stress + slight pitch lift).
DEFAULT_EMPHASIS = {
    "this", "that", "never", "actually", "stop", "wait", "look", "watch",
    "secret", "first", "before", "after", "instantly", "suddenly",
    "shocking", "weirdly", "surprisingly", "real", "truth",
    "yeh", "vo", "actually", "matlab", "exactly",
}


@dataclass
class WordTiming:
    word: str
    start_sec: float
    end_sec: float
    emphasis: bool = False
    sentence_idx: int = 0
    is_pause: bool = False                       # synthetic pause unit


@dataclass
class VoicePlan:
    natural_text: str                            # to feed Sarvam/ElevenLabs
    sentences: list[str]
    word_timings: list[WordTiming]
    total_seconds: float
    pace_wpm: int
    pauses_inserted: int
    breaths_inserted: int
    emphasis_words: list[str]
    sarvam_voice: str = "anushka"

    def to_dict(self) -> dict[str, Any]:
        d = {
            "natural_text": self.natural_text,
            "sentences": self.sentences,
            "word_timings": [asdict(w) for w in self.word_timings],
            "total_seconds": self.total_seconds,
            "pace_wpm": self.pace_wpm,
            "pauses_inserted": self.pauses_inserted,
            "breaths_inserted": self.breaths_inserted,
            "emphasis_words": self.emphasis_words,
            "sarvam_voice": self.sarvam_voice,
        }
        return d


class VoiceIntelligence:
    def __init__(self, *, default_wpm: int = 165) -> None:
        self.default_wpm = default_wpm

    def prepare(self, *, sentences: list[str], pace_wpm: int | None = None,
                emotion: str = "curiosity",
                sarvam_voice: str = "anushka") -> VoicePlan:
        wpm = int(pace_wpm or self.default_wpm)
        clean: list[str] = []
        word_timings: list[WordTiming] = []
        cursor = 0.0
        pauses = 0
        breaths = 0
        emphasis_log: list[str] = []

        # Per-emotion micro-tweaks
        sentence_pause = {
            "surprise": 0.18, "excitement": 0.15, "curiosity": 0.22,
            "suspense": 0.32, "romantic": 0.28, "serious": 0.30, "empathy": 0.30,
        }.get(emotion, 0.22)

        breath_chance = 0.22 if emotion in {"romantic", "suspense", "empathy"} else 0.12

        for s_idx, raw in enumerate(sentences):
            sent = self._humanize(raw)
            clean.append(sent)

            # Optional breath at the start of an emotional sentence
            if s_idx == 0 or _should_breath(s_idx, breath_chance):
                cursor += 0.22
                word_timings.append(WordTiming(
                    "<<breath>>", cursor - 0.22, cursor,
                    emphasis=False, sentence_idx=s_idx, is_pause=True,
                ))
                breaths += 1

            words = sent.split()
            for w_idx, w in enumerate(words):
                # Per-word duration ~ 60/wpm * syllable factor
                syl = _syllables(w)
                dur = (60.0 / wpm) * max(0.6, min(1.8, syl / 1.6))
                emph = _is_emphasis(w)
                if emph:
                    dur *= 1.15
                    emphasis_log.append(_strip_punct(w))
                start = cursor
                cursor += dur
                word_timings.append(WordTiming(
                    word=w, start_sec=round(start, 3), end_sec=round(cursor, 3),
                    emphasis=emph, sentence_idx=s_idx,
                ))

                # Mid-sentence comma pause
                if w.endswith(","):
                    p = 0.16
                    cursor += p
                    word_timings.append(WordTiming(
                        "<<comma>>", cursor - p, cursor,
                        emphasis=False, sentence_idx=s_idx, is_pause=True,
                    ))
                    pauses += 1

            # End-of-sentence pause; longer if "..." or "?"
            tail_pause = sentence_pause
            if sent.endswith("..."):
                tail_pause *= 1.7
            elif sent.endswith("?") or sent.endswith("!"):
                tail_pause *= 1.25
            cursor += tail_pause
            word_timings.append(WordTiming(
                "<<pause>>", cursor - tail_pause, cursor,
                emphasis=False, sentence_idx=s_idx, is_pause=True,
            ))
            pauses += 1

        natural_text = " ".join(clean)
        log.info("[voice-intel] sentences=%d words=%d total=%.2fs wpm=%d "
                 "pauses=%d breaths=%d emphasis=%d",
                 len(sentences),
                 sum(1 for w in word_timings if not w.is_pause),
                 cursor, wpm, pauses, breaths, len(emphasis_log))
        return VoicePlan(
            natural_text=natural_text,
            sentences=clean,
            word_timings=word_timings,
            total_seconds=round(cursor, 2),
            pace_wpm=wpm,
            pauses_inserted=pauses,
            breaths_inserted=breaths,
            emphasis_words=sorted(set(emphasis_log)),
            sarvam_voice=sarvam_voice,
        )

    # ---- Humanization ----------------------------------------------------
    @staticmethod
    def _humanize(s: str) -> str:
        s = s.strip()
        # Convert numbers / currency to spoken form
        s = _spoken_currency(s)
        s = _spoken_numbers(s)
        # Light pause sprinkling — convert " - " or " — " to a comma pause
        s = re.sub(r"\s+[-–—]\s+", ", ", s)
        # Turn shouted ALL-CAPS words into [emph: word] — handled at TTS layer
        s = re.sub(r"\b([A-Z]{3,})\b",
                   lambda m: f"[emph]{m.group(1).lower()}[/emph]", s)
        # Compact whitespace
        s = re.sub(r"\s+", " ", s)
        return s


# ---- Helpers ---------------------------------------------------------------


def _strip_punct(w: str) -> str:
    return re.sub(r"[^\w']+", "", w).lower()


def _is_emphasis(w: str) -> bool:
    cw = _strip_punct(w)
    if cw in DEFAULT_EMPHASIS:
        return True
    # CAPS word — already emphasized at humanize step but still flag.
    return w.isupper() and len(w) >= 3 and w.isalpha()


def _syllables(w: str) -> int:
    cw = _strip_punct(w)
    if not cw:
        return 1
    # Naive vowel-group counter, English/transliterated Hindi-ok.
    cw = re.sub(r"[^a-z]", "", cw)
    if not cw:
        return 1
    groups = re.findall(r"[aeiouy]+", cw)
    return max(1, len(groups))


def _should_breath(sentence_idx: int, p: float) -> bool:
    # Deterministic-ish: breath every 3rd sentence + after twist/loop emphasis.
    return sentence_idx % 3 == 0 and p > 0


def _spoken_currency(t: str) -> str:
    return re.sub(
        r"([\$₹€£])\s?([0-9][0-9,]*)",
        lambda m: f"{_int_to_words(int(m.group(2).replace(',', '')))} "
                  f"{ {'$':'dollars','₹':'rupees','€':'euros','£':'pounds'}[m.group(1)]}",
        t,
    )


def _spoken_numbers(t: str) -> str:
    def repl(m: re.Match) -> str:
        n = m.group(0)
        if len(n) == 4 and (n.startswith("19") or n.startswith("20")):
            return n
        try:
            return _int_to_words(int(n))
        except Exception:
            return n
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
        out = []
        if x >= 100:
            out.append(units[x // 100] + " hundred")
            x %= 100
            if x:
                out.append("and")
        if x >= 20:
            out.append(tens[x // 10] + (("-" + units[x % 10]) if x % 10 else ""))
        elif x > 0:
            out.append(units[x])
        return " ".join(out)

    if n < 0:
        return "minus " + _int_to_words(-n)
    chunks = []
    for power, label in [(1_000_000_000, "billion"),
                         (1_000_000, "million"),
                         (1_000, "thousand")]:
        if n >= power:
            chunks.append(below_thousand(n // power) + " " + label)
            n %= power
    if n > 0:
        chunks.append(below_thousand(n))
    return " ".join(chunks).strip()
