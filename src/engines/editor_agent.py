"""The Editor Agent — acts like a human Shorts editor, not a renderer.

Reads the script's emotional arc, the voice's word-level timings, the chosen
emotion, and the channel's competitor patterns. Decides:

  • where to cut, where to dwell
  • which words deserve a kinetic-text reveal
  • where to drop a meme insert (humor)
  • where to drop a focus circle / arrow (focus)
  • where camera pushes / shakes / ramps live (surprise / retention)
  • where music swells, where it ducks
  • where character expression CHANGES (emotion)
  • when to end the loop with a sting

Every decision in the timeline carries a `reason` field — never random.

Output:
  EditingTimeline  — list[TimelineEvent], dumped as editing_timeline.json
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict
from typing import Any

from ..core.logger import get_logger
from .character_engine import VALID_EXPRESSIONS
from .motion_engine import MotionEngine, MotionSpec
from .music_intelligence import IntensityKeyframe, MusicPlan
from .voice_intelligence import VoicePlan, WordTiming

log = get_logger(__name__)


REASONS = {"retention", "emotion", "humor", "surprise", "focus", "loop"}


@dataclass
class TimelineEvent:
    time: float                 # seconds
    action: str
    reason: str                 # MUST be in REASONS
    duration_sec: float = 0.0
    intensity: float = 0.7
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["time"] = round(d["time"], 3)
        return d


@dataclass
class EditingTimeline:
    video_id: str
    channel_id: str
    total_seconds: float
    events: list[TimelineEvent]
    music_curve: list[dict[str, Any]]
    character_expression_track: list[dict[str, Any]]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "channel_id": self.channel_id,
            "total_seconds": round(self.total_seconds, 2),
            "events": [e.to_dict() for e in self.events],
            "music_curve": self.music_curve,
            "character_expression_track": self.character_expression_track,
            "rationale": self.rationale,
        }


# ---------- The agent ----------------------------------------------------


class EditorAgent:
    """Decides timeline; never renders."""

    def __init__(self, *, seed: int | None = None) -> None:
        self.rng = random.Random(seed or 7)
        self.motion = MotionEngine()

    # ---- Public API ------------------------------------------------------

    def cut(self, *, video_id: str, channel_id: str,
            beats: list[Any],
            voice_plan: VoicePlan,
            music_plan: MusicPlan,
            emotion: str,
            psych: dict[str, Any],
            competitor_patterns: dict[str, Any] | None = None) -> EditingTimeline:
        T = max(8.0, voice_plan.total_seconds)
        events: list[TimelineEvent] = []

        # --- 1. The hook (0 → first sentence end) ------------------------
        hook_end = self._first_sentence_end(voice_plan)
        events += self._hook_events(hook_end, emotion)

        # --- 2. Per-beat cuts + motion ----------------------------------
        beat_boundaries = self._beat_boundaries(beats, voice_plan, T)
        events += self._beat_events(beat_boundaries, emotion)

        # --- 3. Word-level kinetic text on emphasis & on hook/twist/loop -
        events += self._kinetic_text_events(voice_plan, beats, beat_boundaries)

        # --- 4. Strategic effect inserts based on emotional curve --------
        events += self._emotional_inserts(beats, beat_boundaries, voice_plan,
                                          emotion, psych)

        # --- 5. Cadence backstop: never go > 2.4s without a visual change
        events = self._enforce_cadence(events, T,
                                       max_gap_sec=2.4, min_gap_sec=1.5)

        # --- 6. Sort + dedupe near-duplicates
        events = self._compress(events)

        # --- 7. Music curve as event stream (sample at music plan keyframes)
        music_curve = [{"time": kf.t_sec, "intensity": kf.intensity,
                        "label": kf.label} for kf in music_plan.intensity_curve]

        # --- 8. Character expression track (per-beat)
        char_track = self._character_track(beats, beat_boundaries, emotion, psych)

        rationale = self._rationale(beats, beat_boundaries, voice_plan,
                                    music_plan, emotion)
        log.info("[editor] %d events | T=%.2fs | hook=%.2fs | beats=%d",
                 len(events), T, hook_end, len(beat_boundaries))
        return EditingTimeline(
            video_id=video_id,
            channel_id=channel_id,
            total_seconds=T,
            events=events,
            music_curve=music_curve,
            character_expression_track=char_track,
            rationale=rationale,
        )

    # ---- Step 1: Hook ----------------------------------------------------

    def _hook_events(self, hook_end: float, emotion: str) -> list[TimelineEvent]:
        """Hard-hit the first 0-2s. Face zoom + sub-bass + first kinetic word."""
        return [
            TimelineEvent(0.00, "character_in", "retention",
                          duration_sec=0.35,
                          payload={"expression": "curious", "scale_in": 1.10}),
            TimelineEvent(0.00, "push_zoom", "focus",
                          duration_sec=min(1.0, hook_end),
                          intensity=0.9,
                          payload=self.motion.push_zoom(
                              t0=0.0, t1=min(1.0, hook_end), zoom_to=1.18,
                              reason="focus").to_dict()),
            TimelineEvent(0.00, "sub_bass_punch", "surprise",
                          duration_sec=0.6, intensity=0.95,
                          payload={"layer": "transition"}),
            TimelineEvent(0.20, "heartbeat_sfx", "emotion",
                          duration_sec=1.4, intensity=0.6,
                          payload={"layer": "sfx", "loop": False}),
            TimelineEvent(0.05, "depth_parallax", "retention",
                          duration_sec=hook_end + 0.5, intensity=0.5,
                          payload=self.motion.depth_parallax(
                              t0=0.05, t1=hook_end + 0.5, layers=3,
                              reason="retention").to_dict()),
        ]

    # ---- Step 2: Beat-driven cuts ----------------------------------------

    def _beat_boundaries(self, beats: list[Any],
                         voice_plan: VoicePlan,
                         T: float) -> list[tuple[str, float, float]]:
        """Map each beat to actual time window from voice timings."""
        # Group word timings by beat sentence_idx → sentence → beat.
        sent_to_beat: list[str] = []
        for b in beats:
            sent_to_beat.extend([b.name] * len(b.sentences))
        if not sent_to_beat:
            return [("hook", 0.0, T)]

        # Find time per sentence by scanning word_timings.
        sent_starts: dict[int, float] = {}
        sent_ends: dict[int, float] = {}
        for w in voice_plan.word_timings:
            if w.is_pause:
                continue
            sent_starts.setdefault(w.sentence_idx, w.start_sec)
            sent_ends[w.sentence_idx] = w.end_sec

        bounds: dict[str, list[float]] = {}
        for s_idx, beat_name in enumerate(sent_to_beat):
            if s_idx not in sent_starts:
                continue
            bounds.setdefault(beat_name, [sent_starts[s_idx], sent_ends[s_idx]])
            bounds[beat_name][0] = min(bounds[beat_name][0], sent_starts[s_idx])
            bounds[beat_name][1] = max(bounds[beat_name][1], sent_ends[s_idx])

        # Stable beat order
        order = ["hook", "escalation", "reveal", "twist", "loop"]
        out: list[tuple[str, float, float]] = []
        for name in order:
            if name in bounds:
                a, b = bounds[name]
                out.append((name, round(a, 2), round(b, 2)))
        if not out:
            out = [("hook", 0.0, T)]
        return out

    def _beat_events(self, boundaries: list[tuple[str, float, float]],
                     emotion: str) -> list[TimelineEvent]:
        out: list[TimelineEvent] = []
        for i, (name, t0, t1) in enumerate(boundaries):
            if i == 0:
                continue   # hook handled separately
            # Cut + transition at the beat boundary
            out.append(TimelineEvent(
                t0, "scene_cut", "retention",
                duration_sec=0.0, intensity=0.85,
                payload={"to_beat": name},
            ))
            out.append(TimelineEvent(
                t0, "blur_transition", "retention",
                duration_sec=0.18, intensity=0.7,
                payload=self.motion.blur_transition(t=t0, dur=0.18,
                                                     reason="retention").to_dict(),
            ))
            # Camera personality per beat
            if name == "escalation":
                out.append(TimelineEvent(
                    t0 + 0.05, "camera_drift", "retention",
                    duration_sec=t1 - t0, intensity=0.4,
                    payload=self.motion.camera_drift(
                        t0=t0 + 0.05, t1=t1, dx=0.012, dy=-0.004,
                        reason="retention").to_dict(),
                ))
            elif name == "reveal":
                out.append(TimelineEvent(
                    t0 + 0.10, "push_zoom", "focus",
                    duration_sec=min(2.0, t1 - t0), intensity=0.7,
                    payload=self.motion.push_zoom(
                        t0=t0 + 0.10, t1=min(t0 + 2.0, t1), zoom_to=1.10,
                        reason="focus").to_dict(),
                ))
                out.append(TimelineEvent(
                    t0 + 0.20, "focus_circle", "focus",
                    duration_sec=1.4, intensity=0.7,
                    payload=self.motion.focus_circle(
                        t0=t0 + 0.20, t1=t0 + 1.6, x=0.5, y=0.42,
                        reason="focus").to_dict(),
                ))
            elif name == "twist":
                out.append(TimelineEvent(
                    t0, "speed_ramp", "surprise",
                    duration_sec=0.6, intensity=0.85,
                    payload=self.motion.speed_ramp(
                        t0=t0, t1=t0 + 0.6, ramp=(0.85, 1.20),
                        reason="surprise").to_dict(),
                ))
                out.append(TimelineEvent(
                    t0 + 0.05, "camera_shake", "surprise",
                    duration_sec=0.45, intensity=0.65,
                    payload=self.motion.camera_shake(
                        t=t0 + 0.05, dur=0.45, intensity=0.65,
                        reason="surprise").to_dict(),
                ))
            elif name == "loop":
                out.append(TimelineEvent(
                    max(0.0, t1 - 1.6), "music_fade", "loop",
                    duration_sec=1.4, intensity=0.4,
                    payload={"to_db": -34.0},
                ))
                out.append(TimelineEvent(
                    max(0.0, t1 - 0.5), "loop_sting", "loop",
                    duration_sec=0.5, intensity=0.85,
                    payload={"layer": "transition"},
                ))
        return out

    # ---- Step 3: Word-level kinetic text --------------------------------

    def _kinetic_text_events(self, voice_plan: VoicePlan,
                             beats: list[Any],
                             boundaries: list[tuple[str, float, float]],
                             ) -> list[TimelineEvent]:
        out: list[TimelineEvent] = []

        # Map sentence_idx to beat name
        sent_to_beat: dict[int, str] = {}
        s_idx = 0
        for b in beats:
            for _ in b.sentences:
                sent_to_beat[s_idx] = b.name
                s_idx += 1

        # Group word timings into sentences, emit a kinetic text per sentence
        # but ONLY for hook / twist / loop beats. Reveal gets subtitles
        # (lower visual weight so the focus_circle stays the lead).
        from itertools import groupby
        for s_id, group in groupby(
            (w for w in voice_plan.word_timings if not w.is_pause),
            key=lambda w: w.sentence_idx,
        ):
            words: list[WordTiming] = list(group)
            if not words:
                continue
            t0 = words[0].start_sec
            t1 = words[-1].end_sec
            text = " ".join(w.word for w in words)
            beat = sent_to_beat.get(s_id, "escalation")
            if beat in {"hook", "twist", "loop"}:
                out.append(TimelineEvent(
                    t0, "kinetic_text", "focus",
                    duration_sec=t1 - t0, intensity=0.9,
                    payload=self.motion.kinetic_text(
                        text=text, t0=t0, t1=t1, style="punchy",
                        reason="focus").to_dict(),
                ))
            else:
                out.append(TimelineEvent(
                    t0, "subtitle", "retention",
                    duration_sec=t1 - t0, intensity=0.5,
                    payload={"text": text, "style": "subtitle"},
                ))

            # Per-emphasis word: elastic burst (capped to 1 per sentence
            # so it stays meaningful, never spammy).
            emp = next((w for w in words if w.emphasis), None)
            if emp is not None:
                out.append(TimelineEvent(
                    emp.start_sec, "elastic_burst", "emotion",
                    duration_sec=0.35, intensity=0.85,
                    payload=self.motion.elastic_burst(
                        t=emp.start_sec, label=emp.word,
                        reason="emotion").to_dict(),
                ))
        return out

    # ---- Step 4: Emotional inserts (memes / reactions / arrows) ---------

    def _emotional_inserts(self, beats: list[Any],
                           boundaries: list[tuple[str, float, float]],
                           voice_plan: VoicePlan,
                           emotion: str,
                           psych: dict[str, Any]) -> list[TimelineEvent]:
        out: list[TimelineEvent] = []
        bounds_map = {n: (a, b) for n, a, b in boundaries}

        # Reaction meme at the start of escalation — only if it serves humor
        # or surprise (not romantic/serious).
        if emotion in {"surprise", "excitement", "curiosity"}:
            if "escalation" in bounds_map:
                t = bounds_map["escalation"][0] + 0.3
                out.append(TimelineEvent(
                    t, "reaction_meme", "humor",
                    duration_sec=0.7, intensity=0.7,
                    payload={"asset": "auto://meme/raised-eyebrow",
                             "x": 0.78, "y": 0.18, "scale": 0.22},
                ))

        # Animated arrow at reveal — points to the focus circle target
        if "reveal" in bounds_map:
            t0 = bounds_map["reveal"][0] + 0.55
            out.append(TimelineEvent(
                t0, "animated_arrow", "focus",
                duration_sec=1.0, intensity=0.8,
                payload=self.motion.animated_arrow(
                    t0=t0, t1=t0 + 1.0,
                    from_xy=(0.18, 0.78), to_xy=(0.45, 0.46),
                    reason="focus").to_dict(),
            ))

        # Emoji explosion at the twist for high-energy emotions
        if emotion in {"surprise", "excitement"} and "twist" in bounds_map:
            t = bounds_map["twist"][0] + 0.25
            out.append(TimelineEvent(
                t, "emoji_explosion", "humor",
                duration_sec=0.7, intensity=0.85,
                payload=self.motion.emoji_explosion(
                    t=t, emoji="🤯", count=14, reason="humor").to_dict(),
            ))

        # Glow on the comment-trigger word during loop ending
        if "loop" in bounds_map:
            a, b = bounds_map["loop"]
            t = max(a, b - 1.2)
            out.append(TimelineEvent(
                t, "glow_indicator", "retention",
                duration_sec=1.0, intensity=0.55,
                payload=self.motion.glow_indicator(
                    t0=t, t1=t + 1.0, x=0.5, y=0.92,
                    reason="retention").to_dict(),
            ))
            out.append(TimelineEvent(
                t, "comment_prompt", "retention",
                duration_sec=1.0, intensity=0.7,
                payload={"text": psych.get("comment_trigger", "What do you think?"),
                         "y": 0.90, "style": "pill"},
            ))
        return out

    # ---- Step 5: Cadence enforcement -------------------------------------

    def _enforce_cadence(self, events: list[TimelineEvent], T: float,
                         *, max_gap_sec: float, min_gap_sec: float
                         ) -> list[TimelineEvent]:
        """If a stretch of >max_gap_sec has no visual-change event, insert
        a fake-handheld + slight drift so the screen never goes still."""
        ev = sorted(events, key=lambda e: e.time)
        VISUAL_KINDS = {"scene_cut", "blur_transition", "kinetic_text",
                        "elastic_burst", "speed_ramp", "camera_shake",
                        "push_zoom", "focus_circle", "animated_arrow",
                        "reaction_meme", "emoji_explosion"}
        visual_times = [e.time for e in ev if e.action in VISUAL_KINDS]
        if not visual_times:
            return ev
        cur = 0.0
        gaps_filled = 0
        i = 0
        while cur < T - 0.5 and i < len(visual_times):
            if visual_times[i] - cur > max_gap_sec:
                t = cur + min_gap_sec
                ev.append(TimelineEvent(
                    t, "fake_handheld", "retention",
                    duration_sec=min(1.6, visual_times[i] - t - 0.1),
                    intensity=0.35,
                    payload=self.motion.fake_handheld(
                        t0=t, t1=min(visual_times[i] - 0.05, t + 1.6),
                        reason="retention").to_dict(),
                ))
                gaps_filled += 1
            cur = visual_times[i]
            i += 1
        if gaps_filled:
            log.info("[editor] cadence backstop filled %d dead-air gaps", gaps_filled)
        return sorted(ev, key=lambda e: e.time)

    # ---- Step 6: Compress (drop near-duplicates) -------------------------

    @staticmethod
    def _compress(events: list[TimelineEvent]) -> list[TimelineEvent]:
        out: list[TimelineEvent] = []
        for ev in sorted(events, key=lambda e: (e.time, e.action)):
            # Drop a duplicate of the same action within 60ms
            if out and out[-1].action == ev.action and abs(out[-1].time - ev.time) < 0.06:
                continue
            out.append(ev)
        return out

    # ---- Step 7: Character expression track ------------------------------

    def _character_track(self, beats: list[Any],
                         boundaries: list[tuple[str, float, float]],
                         emotion: str,
                         psych: dict[str, Any]) -> list[dict[str, Any]]:
        # Map beat → expression with a sensible default per channel emotion.
        beat_expr = {
            "hook":       "curious",
            "escalation": "curious",
            "reveal":     "shocked"     if emotion in {"surprise", "excitement"} else "curious",
            "twist":      "shocked"     if emotion != "romantic" else "embarrassed",
            "loop":       "happy",
        }
        if emotion == "romantic":
            beat_expr["reveal"] = "romantic"
            beat_expr["twist"] = "romantic"
        if emotion in {"serious", "empathy"}:
            beat_expr["reveal"] = "confused"

        track: list[dict[str, Any]] = []
        for name, t0, t1 in boundaries:
            track.append({
                "from_sec": t0,
                "to_sec": t1,
                "beat": name,
                "expression": beat_expr.get(name, "curious"),
            })
        return track

    # ---- Helpers --------------------------------------------------------

    @staticmethod
    def _first_sentence_end(voice_plan: VoicePlan) -> float:
        for w in voice_plan.word_timings:
            if w.is_pause and w.word == "<<pause>>":
                return float(w.end_sec)
        return 2.0

    @staticmethod
    def _rationale(beats: list[Any], boundaries: list[tuple[str, float, float]],
                   voice_plan: VoicePlan, music_plan: MusicPlan,
                   emotion: str) -> str:
        parts = [
            f"emotion={emotion}",
            f"voice_total={voice_plan.total_seconds:.1f}s @ {voice_plan.pace_wpm}wpm",
            f"emphasis_words={len(voice_plan.emphasis_words)}",
            "beats=" + "→".join(n for n, _, _ in boundaries),
            f"music={(music_plan.track or {}).get('title','curated')}"
            f" / target_bpm={music_plan.target_bpm}",
        ]
        return "; ".join(parts)
