"""ElevenLabs TTS adapter — fallback voice."""
from __future__ import annotations

import os
from pathlib import Path

import httpx

from ..core.logger import get_logger

log = get_logger(__name__)


def synthesize(
    text: str,
    *,
    out_path: Path,
    voice_id: str | None = None,
    voice_id_env: str | None = None,
    speed: float = 1.0,
    stability: float = 0.45,
    similarity: float = 0.75,
) -> Path | None:
    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        log.info("[elevenlabs] no API key — skipping")
        return None

    vid = voice_id or (os.getenv(voice_id_env) if voice_id_env else None)
    if not vid:
        log.warning("[elevenlabs] no voice_id (env=%s) — skipping", voice_id_env)
        return None

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "accept": "audio/mpeg",
    }
    payload = {
        "text": text[:5000],
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity,
            "speed": speed,
            "use_speaker_boost": True,
        },
    }
    try:
        with httpx.Client(timeout=90.0) as client:
            r = client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(r.content)
            log.info("[elevenlabs] wrote %s (%d bytes)", out_path, len(r.content))
            return out_path
    except Exception as e:
        log.warning("[elevenlabs] failed: %s", e)
        return None
