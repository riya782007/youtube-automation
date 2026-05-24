"""Unified LLM client. Claude primary (better Hinglish nuance), OpenAI fallback.

Designed to degrade gracefully:
 - if both keys are missing, returns a deterministic mock response so the
   pipeline can still be wired and tested end-to-end without spend.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .logger import get_logger

log = get_logger(__name__)


@dataclass
class LLMResponse:
    text: str
    provider: str          # "anthropic" | "openai" | "mock"
    raw: Any = None


class LLM:
    def __init__(self) -> None:
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self._anthropic = None
        self._openai = None
        self.dry_run = os.getenv("DRY_RUN", "false").lower() == "true"

    # ---- Provider lazy-init ------------------------------------------------
    def _claude(self):
        if self._anthropic is None and self.anthropic_key:
            try:
                import anthropic
                self._anthropic = anthropic.Anthropic(api_key=self.anthropic_key)
            except Exception as e:
                log.warning("Anthropic SDK unavailable: %s", e)
        return self._anthropic

    def _gpt(self):
        if self._openai is None and self.openai_key:
            try:
                from openai import OpenAI
                self._openai = OpenAI(api_key=self.openai_key)
            except Exception as e:
                log.warning("OpenAI SDK unavailable: %s", e)
        return self._openai

    # ---- Public API --------------------------------------------------------
    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Single-turn completion. Tries Claude first, falls back to GPT, then mock."""
        if self.dry_run:
            return self._mock(system, user, json_mode)

        c = self._claude()
        if c is not None:
            try:
                msg = c.messages.create(
                    model="claude-3-5-sonnet-latest",
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                text = "".join(
                    blk.text for blk in msg.content if getattr(blk, "type", "") == "text"
                )
                return LLMResponse(text=text, provider="anthropic", raw=msg)
            except Exception as e:
                log.warning("Claude failed (%s) — falling back to GPT", e)

        g = self._gpt()
        if g is not None:
            try:
                kwargs: dict[str, Any] = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = g.chat.completions.create(**kwargs)
                return LLMResponse(
                    text=resp.choices[0].message.content or "",
                    provider="openai",
                    raw=resp,
                )
            except Exception as e:
                log.warning("OpenAI failed (%s) — using mock", e)

        return self._mock(system, user, json_mode)

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2000,
        temperature: float = 0.4,
    ) -> dict[str, Any]:
        """Force JSON output, parse safely."""
        sys2 = system + "\n\nRespond with a single valid JSON object. No prose. No markdown fences."
        resp = self.complete(sys2, user, max_tokens=max_tokens,
                             temperature=temperature, json_mode=True)
        return _safe_json(resp.text)

    # ---- Mock --------------------------------------------------------------
    def _mock(self, system: str, user: str, json_mode: bool) -> LLMResponse:
        log.info("LLM running in MOCK mode (no API keys / DRY_RUN=true).")
        sys_l, user_l = system.lower(), user.lower()
        # Structurally valid script JSON so the pipeline can be smoke-tested.
        if "beats" in user_l and json_mode or "title" in user_l and "pinned_comment" in user_l:
            stub = {
                "title": "Why she stops replying suddenly",
                "description": "What her silence really means. #relationships #hinglish",
                "tags": ["relationships", "psychology", "texting", "hinglish",
                         "shorts", "human decoder", "crush", "attraction"],
                "pinned_comment": "Comment YES if this happened to you.",
                "beats": [
                    {"name": "hook", "sentences": [
                        "She stopped replying suddenly...",
                        "Yeh actually kya matlab hai?"]},
                    {"name": "escalation", "sentences": [
                        "Aapko lagta hai vo busy hai.",
                        "Lekin truth different hai bro."]},
                    {"name": "reveal", "sentences": [
                        "Vo overthink kar rahi hai.",
                        "Har message ka weight feel karti hai.",
                        "Silence matlab tension, not boredom."]},
                    {"name": "twist", "sentences": [
                        "But ek galti most boys karte hain.",
                        "Double-text pressure dete hain instantly."]},
                    {"name": "loop", "sentences": [
                        "Kal phir same situation aayegi.",
                        "Ab pata hai kya karna hai..."]},
                ],
            }
            return LLMResponse(text=json.dumps(stub), provider="mock")
        # Quality scoring stub.
        if "score" in sys_l and "hook" in user_l:
            stub = {
                "hook": 8.7, "voice": 8.4, "visual_density": 8.6,
                "emotion": 8.8, "editing": 8.5, "retention_prediction": 8.6,
                "notes": "[mock] looks tight; verify reveal lands within 18s.",
            }
            return LLMResponse(text=json.dumps(stub), provider="mock")
        # Audience psychology stub.
        if "pain" in user_l and "comment_trigger" in user_l:
            stub = {
                "pain": "she stopped replying mid-conversation",
                "fear": "she lost interest because of me",
                "desire": "to know exactly what she's thinking",
                "curiosity": "what does her silence actually mean",
                "emotional_trigger": "curiosity",
                "comment_trigger": "True or false?",
            }
            return LLMResponse(text=json.dumps(stub), provider="mock")
        # Generic JSON stub.
        if json_mode or "JSON" in system.upper():
            stub = {"_mock": True,
                    "note": "set ANTHROPIC_API_KEY or OPENAI_API_KEY for real output"}
            return LLMResponse(text=json.dumps(stub), provider="mock")
        return LLMResponse(
            text=(
                "[MOCK] Girls actually notice THIS first...\n"
                "before looks.\n"
                "Their brain locks on energy.\n"
                "Confidence beats everything.\n"
                "Watch what happens next..."
            ),
            provider="mock",
        )


def _safe_json(text: str) -> dict[str, Any]:
    """Parse JSON even if the model wrapped it in ```json fences or prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    # Find first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error("JSON parse failed: %s — text: %r", e, text[:300])
        return {"_parse_error": str(e), "raw": text[:1000]}
