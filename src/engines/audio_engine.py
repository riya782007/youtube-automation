"""Engine 10 — Audio Engine.

Layered audio plan:
  • voice (already produced by VoiceEngine)
  • background music (channel-mood-based selection)
  • micro-SFX per emotional beat (heartbeat, whoosh, click, ping, vibration…)
  • transitions (whoosh + sub-bass on every reset cue)
  • ambient (channel-specific bed)

Outputs an `AudioPlan` for the renderer to mix. Royalty-safe library names
only — never copyrighted tracks.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Any

from ..adapters import free_music
from ..core.logger import get_logger
from .attention_reset import ResetCue
from .emotion_engine import EmotionPlan

log = get_logger(__name__)


@dataclass
class AudioCue:
    at_sec: float
    layer: str                 # music | sfx | transition | ambient
    label: str                 # human-readable: "heartbeat", "whoosh", etc.
    gain_db: float = -6.0
    duration_sec: float = 0.4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AudioPlan:
    music_track_query: str
    music_intensity: float
    voice_track: str           # path to voice mp3
    cues: list[AudioCue] = field(default_factory=list)
    ambient_bed: str | None = None
    music_candidates: list[dict[str, Any]] = field(default_factory=list)
    selected_music: dict[str, Any] | None = None


class AudioEngine:
    def __init__(self, *, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def plan(self, *, channel: dict[str, Any], voice_path: str,
             total_seconds: float,
             emotion_plan: EmotionPlan,
             reset_cues: list[ResetCue]) -> AudioPlan:
        palette = channel.get("audio_palette", {})
        music_moods: list[str] = palette.get("music_mood", ["cinematic"])
        sfx_pool: list[str] = palette.get("micro_sfx", ["whoosh", "click"])

        music_query = f"{music_moods[0]} short hindi background loop royalty free"
        ambient_bed = music_moods[-1] if len(music_moods) > 1 else None

        cues: list[AudioCue] = []

        # 1) Hook punch at t≈0.0 — sub-bass + whoosh
        cues.append(AudioCue(0.0, "transition", "sub_bass_punch", gain_db=-3.0, duration_sec=0.6))

        # 2) Per reset cue: SFX matched to channel palette
        for r in reset_cues:
            label = self._sfx_for(r.kind, sfx_pool)
            cues.append(AudioCue(
                at_sec=r.at_sec,
                layer="sfx",
                label=label,
                gain_db=-8.0 + 4.0 * (r.intensity - 0.5),
                duration_sec=0.35,
            ))

        # 3) Final loop sting at end-2s
        cues.append(AudioCue(
            at_sec=max(0.0, total_seconds - 2.0),
            layer="transition",
            label="loop_sting",
            gain_db=-4.0,
            duration_sec=1.5,
        ))

        # 4) Ambient bed across whole length, if defined
        if ambient_bed:
            cues.append(AudioCue(
                at_sec=0.0,
                layer="ambient",
                label=f"ambient_{ambient_bed}",
                gain_db=-22.0,
                duration_sec=total_seconds,
            ))

        log.info("[audio] %d cues, music=%r intensity=%.2f",
                 len(cues), music_query, emotion_plan.music_intensity)

        # Open-source music selection (Pixabay → Jamendo → FMA → curated fallback)
        music_candidates = free_music.search_music(
            music_query, limit=5,
            min_duration=max(15, int(total_seconds)), max_duration=180,
        )
        selected = music_candidates[0] if music_candidates else None
        if selected:
            log.info("[audio] selected free track: %s — %s (%s)",
                     selected.get("title"), selected.get("artist"), selected.get("license"))

        return AudioPlan(
            music_track_query=music_query,
            music_intensity=emotion_plan.music_intensity,
            voice_track=str(voice_path),
            cues=cues,
            ambient_bed=ambient_bed,
            music_candidates=music_candidates,
            selected_music=selected,
        )

    def _sfx_for(self, reset_kind: str, pool: list[str]) -> str:
        # Map reset kind → semantic SFX preference, fall back to channel pool.
        mapping = {
            "zoom_punch":    "whoosh",
            "camera_shake":  "rumble",
            "meme_insert":   "boing",
            "flash":         "ding",
            "emoji_sticker": "pop",
            "sfx_hit":       "impact",
            "scene_change":  "transition",
        }
        ideal = mapping.get(reset_kind, "click")
        if ideal in pool:
            return ideal
        return self.rng.choice(pool) if pool else ideal
