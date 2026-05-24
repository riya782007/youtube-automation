"""Sarvam AI TTS adapter — primary Hinglish voice."""
from __future__ import annotations

import base64
import os
from pathlib import Path

import httpx

from ..core.logger import get_logger

log = get_logger(__name__)

API_URL = "https://api.sarvam.ai/text-to-speech"


def synthesize(
    text: str,
    *,
    out_path: Path,
    voice: str = "anushka",
    speed: float = 1.0,
    pitch: float = 1.0,
    sample_rate: int = 24000,
) -> Path | None:
    """Render `text` to mp3 using Sarvam. Returns path on success, None on
    failure (caller should fall back to ElevenLabs)."""
    api_key = os.getenv("SARVAM_API_KEY", "").strip()
    if not api_key:
        log.info("[sarvam] no API key — skipping (ElevenLabs fallback)")
        return None

    payload = {
        "inputs": [text[:1500]],
        "target_language_code": "hi-IN",
        "speaker": voice,
        "pitch": pitch - 1.0,    # Sarvam expects offset from 1.0
        "pace": speed,
        "loudness": 1.0,
        "speech_sample_rate": sample_rate,
        "enable_preprocessing": True,
        "model": "bulbul:v1",
    }
    headers = {"api-subscription-key": api_key, "Content-Type": "application/json"}
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(API_URL, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            audios = data.get("audios") or []
            if not audios:
                log.warning("[sarvam] empty response")
                return None
            audio_bytes = base64.b64decode(audios[0])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(audio_bytes)
            log.info("[sarvam] wrote %s (%d bytes)", out_path, len(audio_bytes))
            return out_path
    except Exception as e:
        log.warning("[sarvam] failed: %s", e)
        return None
