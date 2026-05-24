"""The brain-holding filter — every decision in the system passes through this.

Core operating rule:
    Never ask "How do I generate a video?"
    Always ask "How do I hold a human brain for one more second?"

This module implements that rule as code. Any candidate (hook, sentence, scene,
ending) can be scored against `BrainHoldFilter.score(...)` and rejected if it
doesn't credibly extend attention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# Heuristic word lists — fast pre-LLM gate. The LLM-backed quality gate runs after.
ATTENTION_BOOSTERS = {
    "secret", "actually", "this", "these", "before", "never", "first",
    "wait", "stop", "look", "watch", "notice", "you", "your", "she",
    "he", "they", "nobody", "everyone", "the truth", "real reason",
    "hidden", "ignore", "shocking", "weirdly", "unexpectedly", "instantly",
}

ATTENTION_KILLERS = {
    "in this video",
    "today we will discuss",
    "let me tell you",
    "psychology says",
    "studies show",
    "experts believe",
    "as we all know",
    "first of all",
    "without further ado",
    "scientists found",
}

# Phrases that imply long, written, textbook tone — banned per user spec.
TEXTBOOK_PHRASES = {
    "research indicates", "it is observed that", "one might say",
    "in conclusion", "to summarize", "moreover", "furthermore",
    "therefore", "as previously stated",
}

MAX_WORDS_PER_SENTENCE = 8


@dataclass
class FilterResult:
    passes: bool
    score: float                  # 0..10
    reasons: list[str]
    suggestions: list[str]


class BrainHoldFilter:
    """Cheap heuristic gate. Apply BEFORE expensive TTS / render calls."""

    def score_sentence(self, sentence: str) -> FilterResult:
        s = sentence.strip()
        low = s.lower()
        reasons: list[str] = []
        suggestions: list[str] = []
        score = 6.0

        word_count = len(s.split())
        if word_count == 0:
            return FilterResult(False, 0.0, ["empty sentence"], [])
        if word_count > MAX_WORDS_PER_SENTENCE:
            score -= 2.5
            reasons.append(f"sentence is {word_count} words (>{MAX_WORDS_PER_SENTENCE})")
            suggestions.append("split the sentence; one idea per line")

        for kill in ATTENTION_KILLERS | TEXTBOOK_PHRASES:
            if kill in low:
                score -= 3.0
                reasons.append(f"attention killer: '{kill}'")
                suggestions.append(f"remove '{kill}' and start mid-thought")

        boosters_hit = sum(1 for b in ATTENTION_BOOSTERS if b in low)
        score += min(boosters_hit * 0.6, 2.4)

        # Specificity check — vague abstract words drain attention.
        if any(v in low.split() for v in {"things", "stuff", "something"}):
            score -= 1.0
            reasons.append("vague nouns ('things'/'stuff'/'something')")
            suggestions.append("name the actual thing")

        # Reward sentence-ending mystery
        if s.endswith("...") or s.endswith("?"):
            score += 0.6

        score = max(0.0, min(10.0, score))
        passes = score >= 6.0 and not reasons
        return FilterResult(passes=passes, score=round(score, 2),
                            reasons=reasons, suggestions=suggestions)

    def score_script(self, sentences: Iterable[str]) -> FilterResult:
        sents = [s for s in sentences if s.strip()]
        if not sents:
            return FilterResult(False, 0.0, ["empty script"], [])

        per = [self.score_sentence(s) for s in sents]
        avg = sum(p.score for p in per) / len(per)
        all_reasons: list[str] = []
        for s, r in zip(sents, per):
            for reason in r.reasons:
                all_reasons.append(f"'{s[:40]}…' → {reason}")

        # Penalty for weak first sentence — the 0–2 sec hook is everything.
        first = per[0]
        if first.score < 7.5:
            avg -= 1.0
            all_reasons.append("hook (first sentence) below 7.5 — rewrite")

        suggestions = sorted({sug for r in per for sug in r.suggestions})
        return FilterResult(
            passes=avg >= 7.0 and first.score >= 7.5,
            score=round(max(0.0, min(10.0, avg)), 2),
            reasons=all_reasons,
            suggestions=suggestions,
        )
