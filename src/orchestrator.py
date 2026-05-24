"""The orchestrator — runs the 12-step pipeline for one video, end-to-end.

Flow:
    Trend Intel (cached)
 →  Audience Psychology
 →  Script Engine          (regenerates if filter rejects)
 →  Anti-Repetition Memory (regenerates if duplicate)
 →  Emotion Engine
 →  Voice Engine           (Sarvam → ElevenLabs → placeholder)
 →  Visual Composition
 →  Attention Reset
 →  HyperFrames Engine
 →  Audio Engine
 →  Quality Gate           (regenerates if avg < 8.5)
 →  Learning Engine        (records the new video, will absorb analytics later)
 →  Persist VideoPlan JSON for renderer pickup.
"""
from __future__ import annotations

import datetime as dt
import os
import random
import re
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .core.logger import get_logger, retention_event
from .core.state import (
    ARTIFACT_DIR,
    OUTPUT_DIR,
    load_channel,
    write_json_atomic,
)
from .engines.anti_repetition import AntiRepetitionMemory
from .engines.attention_reset import AttentionReset
from .engines.audience_psychology import AudiencePsychology, PsychProfile
from .engines.audio_engine import AudioEngine
from .engines.emotion_engine import EmotionEngine
from .engines.hyperframes_engine import HyperFramesEngine
from .engines.learning_engine import LearningEngine
from .engines.quality_gate import QualityGate
from .engines.render_engine import RenderEngine
from .engines.script_engine import Script, ScriptEngine
from .engines.visual_composition import VisualComposition
from .engines.voice_engine import VoiceEngine

log = get_logger(__name__)

MAX_REGENS = int(os.getenv("MAX_REGENERATION_ATTEMPTS", "3"))


@dataclass
class VideoPlan:
    id: str
    channel_id: str
    created_at: str
    title: str
    description: str
    tags: list[str]
    pinned_comment: str
    duration_sec: float
    emotion: str
    psych: dict[str, Any]
    script: dict[str, Any]
    fingerprint: dict[str, str]
    voice: dict[str, Any]
    visual: dict[str, Any]
    hyperframes: dict[str, Any]
    attention_resets: list[dict[str, Any]]
    audio: dict[str, Any]
    quality: dict[str, Any] = field(default_factory=dict)
    repetition_check: list[dict[str, Any]] = field(default_factory=list)
    artifact_dir: str = ""
    render: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Orchestrator:
    def __init__(self) -> None:
        self.psych = AudiencePsychology()
        self.script = ScriptEngine()
        self.repetition = AntiRepetitionMemory()
        self.emotion = EmotionEngine()
        self.voice = VoiceEngine()
        self.visual = VisualComposition()
        self.attn = AttentionReset()
        self.hf = HyperFramesEngine()
        self.audio = AudioEngine()
        self.quality = QualityGate()
        self.learn = LearningEngine()
        self.renderer = RenderEngine()

    def generate_video(self, channel_id: str, *, topic: str | None = None,
                       render: bool = False) -> VideoPlan:
        retention_event("start", channel=channel_id, topic=topic)
        channel = load_channel(channel_id)

        topic = topic or self._pick_topic(channel)
        log.info("[orchestrator] channel=%s topic=%r", channel_id, topic)

        # ---- 2. Audience Psychology --------------------------------------
        psych = self.psych.profile(channel, topic)

        # ---- 3,4. Script + Anti-Repetition (with regen loop) -------------
        script = self._script_with_repetition_guard(channel, psych)

        # ---- 5. Emotion Engine -------------------------------------------
        all_sentences = [s for b in script.beats for s in b.sentences]
        emotion_plan = self.emotion.plan(all_sentences, script.emotion)

        # ---- 6. Voice Engine ---------------------------------------------
        video_id = _new_video_id(channel_id)
        voice_artifact = self.voice.render(
            channel=channel,
            video_id=video_id,
            annotated_sentences=emotion_plan.annotated_script,
            emotion_plan=emotion_plan,
        )
        duration = max(8.0, voice_artifact.seconds_estimate)

        # ---- 7. Visual Composition ---------------------------------------
        visual_plan = self.visual.plan(
            channel=channel, script_beats=script.beats, voice_seconds=duration,
        )

        # ---- 8. Attention Reset ------------------------------------------
        hard_cuts = [b.start_sec for b in script.beats if b.start_sec > 0]
        reset_cues = self.attn.schedule(
            total_seconds=duration,
            emotion_plan=emotion_plan,
            hard_cuts_at=hard_cuts,
        )

        # ---- 9. HyperFrames ----------------------------------------------
        hf_plan = self.hf.build(visual_plan=visual_plan)

        # ---- 10. Audio Engine --------------------------------------------
        audio_plan = self.audio.plan(
            channel=channel,
            voice_path=str(voice_artifact.audio_path),
            total_seconds=duration,
            emotion_plan=emotion_plan,
            reset_cues=reset_cues,
        )

        # ---- 11. Quality Gate (with regen loop) --------------------------
        plan_summary = _summary_for_quality(
            channel_id=channel_id, script=script, duration=duration,
            visual_plan=visual_plan, reset_cues=reset_cues,
            forbidden=channel.get("forbidden_phrases", []),
        )
        quality = self.quality.score(plan_summary)
        attempts = 0
        while not quality.passes() and attempts < MAX_REGENS:
            attempts += 1
            log.info("[orchestrator] quality fail (avg=%.2f) → regen #%d note=%s",
                     quality.average(), attempts, quality.notes[:80])
            script = self._script_with_repetition_guard(channel, psych)
            all_sentences = [s for b in script.beats for s in b.sentences]
            emotion_plan = self.emotion.plan(all_sentences, script.emotion)
            visual_plan = self.visual.plan(
                channel=channel, script_beats=script.beats, voice_seconds=duration,
            )
            reset_cues = self.attn.schedule(
                total_seconds=duration, emotion_plan=emotion_plan,
                hard_cuts_at=[b.start_sec for b in script.beats if b.start_sec > 0],
            )
            hf_plan = self.hf.build(visual_plan=visual_plan)
            plan_summary = _summary_for_quality(
                channel_id=channel_id, script=script, duration=duration,
                visual_plan=visual_plan, reset_cues=reset_cues,
                forbidden=channel.get("forbidden_phrases", []),
            )
            quality = self.quality.score(plan_summary)

        # ---- Commit fingerprint + record ---------------------------------
        self.repetition.commit(video_id, script.fingerprint, channel_id)

        plan = VideoPlan(
            id=video_id,
            channel_id=channel_id,
            created_at=dt.datetime.utcnow().isoformat() + "Z",
            title=script.title,
            description=script.description,
            tags=script.tags,
            pinned_comment=script.pinned_comment,
            duration_sec=duration,
            emotion=script.emotion,
            psych=psych.to_dict(),
            script={
                "full_text": script.full_text,
                "beats": [
                    {"name": b.name, "start_sec": b.start_sec,
                     "end_sec": b.end_sec, "sentences": b.sentences}
                    for b in script.beats
                ],
            },
            fingerprint=script.fingerprint,
            voice={
                "provider": voice_artifact.provider,
                "audio_path": str(voice_artifact.audio_path),
                "seconds_estimate": voice_artifact.seconds_estimate,
                "ssml_hint": emotion_plan.ssml_hint[:6000],
            },
            visual={
                "asset_mix_actual": visual_plan.asset_mix_actual,
                "scenes": [s.to_dict() for s in visual_plan.scenes],
            },
            hyperframes={
                "aspect_ratio": hf_plan.aspect_ratio,
                "fps": hf_plan.fps,
                "scenes": [s.to_dict() for s in hf_plan.scenes],
            },
            attention_resets=[c.to_dict() for c in reset_cues],
            audio={
                "music_track_query": audio_plan.music_track_query,
                "music_intensity": audio_plan.music_intensity,
                "voice_track": audio_plan.voice_track,
                "ambient_bed": audio_plan.ambient_bed,
                "cues": [c.to_dict() for c in audio_plan.cues],
                "music_candidates": audio_plan.music_candidates,
                "selected_music": audio_plan.selected_music,
            },
            quality=quality.to_dict() | {"regen_attempts": attempts},
            artifact_dir=str(ARTIFACT_DIR / video_id),
        )

        out = OUTPUT_DIR / f"{video_id}.json"
        write_json_atomic(out, plan.to_dict())
        log.info("[orchestrator] DONE id=%s quality_avg=%.2f → %s",
                 video_id, quality.average(), out)

        # Optional: actually render the mp4 with the open-source FFmpeg/MoviePy engine.
        if render:
            try:
                rr = self.renderer.render(video_plan=plan.to_dict())
                plan.render = {
                    "video_path": str(rr.video_path) if rr.video_path else None,
                    "music_track": rr.music_track,
                    "sfx_used": rr.sfx_used,
                    "skipped_reason": rr.skipped_reason,
                }
                # Re-write JSON with render info attached.
                write_json_atomic(out, plan.to_dict())
                log.info("[orchestrator] render → %s", plan.render.get("video_path"))
            except Exception as e:
                log.warning("[orchestrator] render failed: %s", e)
                plan.render = {"video_path": None, "skipped_reason": f"error:{e}"}

        # Record stub in analytics history (will be enriched on ingest)
        self.learn.record_video({
            "id": video_id,
            "channel_id": channel_id,
            "fingerprint": script.fingerprint,
            "emotion": script.emotion,
            "plan_summary": plan_summary,
            "first_seen_at": plan.created_at,
        })
        retention_event("done", id=video_id, q=quality.average(), regens=attempts)
        return plan

    # ---- Helpers ---------------------------------------------------------
    def _script_with_repetition_guard(
        self, channel: dict[str, Any], psych: PsychProfile,
    ) -> Script:
        for attempt in range(MAX_REGENS):
            script = self.script.generate(channel, psych)
            ok, verdicts = self.repetition.is_accepted(script.fingerprint)
            if ok:
                return script
            sims = ", ".join(f"{v.field}={v.similarity:.2f}" for v in verdicts)
            log.info("[orchestrator] repetition rejected (%s) — regen #%d", sims, attempt + 1)
        # Last script wins even if too similar — at least it's better than nothing.
        log.warning("[orchestrator] repetition guard exhausted. Returning last script.")
        return script

    @staticmethod
    def _pick_topic(channel: dict[str, Any]) -> str:
        seeds = channel.get("topic_seeds") or ["lifestyle"]
        return random.choice(seeds)


def _new_video_id(channel_id: str) -> str:
    return f"{channel_id}_{dt.datetime.utcnow():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"


def _summary_for_quality(
    *, channel_id: str, script: Script, duration: float,
    visual_plan: Any, reset_cues: list[Any], forbidden: list[str],
) -> dict[str, Any]:
    flat = [s for b in script.beats for s in b.sentences]
    body = " ".join(flat).lower()
    forbidden_hits = [p for p in forbidden if p.lower() in body]
    avg_reset = (
        round(duration / max(1, len(reset_cues)), 2) if reset_cues else duration
    )
    hook_sentence = flat[0] if flat else ""
    loop_sentence = flat[-1] if flat else ""
    return {
        "channel_id": channel_id,
        "title": script.title,
        "duration_sec": duration,
        "emotion": script.emotion,
        "hook_sentence": hook_sentence,
        "loop_sentence": loop_sentence,
        "asset_mix": visual_plan.asset_mix_actual,
        "avg_reset_interval": avg_reset,
        "sentence_count": len(flat),
        "comment_trigger": script.psych.get("comment_trigger"),
        "pinned_comment": script.pinned_comment,
        "forbidden_hits": forbidden_hits,
    }
