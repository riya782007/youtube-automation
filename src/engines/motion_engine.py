"""Motion graphics primitives.

Each primitive is a self-contained scheduling unit the Editor Agent can request:
  kinetic_text, elastic_burst, animated_arrow, focus_circle, glow_indicator,
  emoji_explosion, speed_ramp, camera_drift, camera_shake, push_zoom,
  fake_handheld, blur_transition, depth_parallax.

The engine does NOT decide WHEN to fire these — that's the Editor Agent's job.
This module returns deterministic JSON specs that the renderer consumes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal


MotionKind = Literal[
    "kinetic_text", "elastic_burst", "animated_arrow", "focus_circle",
    "glow_indicator", "emoji_explosion", "speed_ramp", "camera_drift",
    "camera_shake", "push_zoom", "fake_handheld", "blur_transition",
    "depth_parallax",
]


@dataclass
class MotionSpec:
    kind: MotionKind
    start_sec: float
    end_sec: float
    intensity: float = 0.7              # 0..1
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""                    # human-readable WHY

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MotionEngine:
    """Builds MotionSpecs the renderer can execute."""

    # ---- Text primitives --------------------------------------------------

    def kinetic_text(self, *, text: str, t0: float, t1: float,
                     style: str = "punchy", reason: str = "focus") -> MotionSpec:
        words = text.strip().split()
        return MotionSpec(
            kind="kinetic_text", start_sec=t0, end_sec=t1,
            intensity=0.9 if style == "punchy" else 0.6,
            params={
                "text": text,
                "words": words,
                "style": style,           # punchy | subtitle | whisper
                "max_words_visible": 4,
                "reveal_per_chunk_sec": max(0.18, (t1 - t0) / max(1, len(words) / 3)),
                "color": "#FFFFFF",
                "stroke": "#000000",
                "shadow_blur": 6,
                "scale_in": 1.18,
            },
            reason=reason,
        )

    def elastic_burst(self, *, t: float, label: str = "", reason: str = "emotion") -> MotionSpec:
        return MotionSpec(
            kind="elastic_burst", start_sec=t, end_sec=t + 0.35,
            intensity=0.9,
            params={"label": label, "scale_peak": 1.45, "ease": "elastic"},
            reason=reason,
        )

    # ---- Indicator primitives --------------------------------------------

    def focus_circle(self, *, t0: float, t1: float, x: float = 0.5, y: float = 0.4,
                     radius: float = 0.18, reason: str = "focus") -> MotionSpec:
        return MotionSpec(
            kind="focus_circle", start_sec=t0, end_sec=t1,
            intensity=0.7,
            params={"x": x, "y": y, "radius": radius,
                    "stroke_color": "#FFD86B", "stroke_w": 6, "pulse": True},
            reason=reason,
        )

    def animated_arrow(self, *, t0: float, t1: float,
                       from_xy: tuple[float, float] = (0.15, 0.85),
                       to_xy: tuple[float, float] = (0.45, 0.45),
                       reason: str = "focus") -> MotionSpec:
        return MotionSpec(
            kind="animated_arrow", start_sec=t0, end_sec=t1,
            intensity=0.8,
            params={"from": list(from_xy), "to": list(to_xy),
                    "color": "#FF4D4D", "thickness": 8, "wobble": 0.12},
            reason=reason,
        )

    def glow_indicator(self, *, t0: float, t1: float, x: float = 0.5, y: float = 0.5,
                       reason: str = "focus") -> MotionSpec:
        return MotionSpec(
            kind="glow_indicator", start_sec=t0, end_sec=t1,
            intensity=0.6,
            params={"x": x, "y": y, "color": "#7CF6FF", "radius": 0.22},
            reason=reason,
        )

    def emoji_explosion(self, *, t: float, emoji: str = "🔥", count: int = 14,
                        reason: str = "humor") -> MotionSpec:
        return MotionSpec(
            kind="emoji_explosion", start_sec=t, end_sec=t + 0.7,
            intensity=0.85,
            params={"emoji": emoji, "count": count, "spread": 0.6, "duration_sec": 0.7},
            reason=reason,
        )

    # ---- Camera primitives -----------------------------------------------

    def speed_ramp(self, *, t0: float, t1: float, ramp: tuple[float, float] = (0.85, 1.20),
                   reason: str = "surprise") -> MotionSpec:
        return MotionSpec(
            kind="speed_ramp", start_sec=t0, end_sec=t1,
            intensity=0.7,
            params={"ramp": list(ramp)},
            reason=reason,
        )

    def camera_drift(self, *, t0: float, t1: float,
                     dx: float = 0.02, dy: float = 0.0, reason: str = "retention") -> MotionSpec:
        return MotionSpec(
            kind="camera_drift", start_sec=t0, end_sec=t1,
            intensity=0.4,
            params={"dx": dx, "dy": dy},
            reason=reason,
        )

    def camera_shake(self, *, t: float, dur: float = 0.45, intensity: float = 0.6,
                     reason: str = "surprise") -> MotionSpec:
        return MotionSpec(
            kind="camera_shake", start_sec=t, end_sec=t + dur,
            intensity=intensity,
            params={"freq": 28, "amp_px": 14},
            reason=reason,
        )

    def push_zoom(self, *, t0: float, t1: float, zoom_to: float = 1.18,
                  reason: str = "focus") -> MotionSpec:
        return MotionSpec(
            kind="push_zoom", start_sec=t0, end_sec=t1,
            intensity=0.6,
            params={"zoom_to": zoom_to, "ease": "easeOutCubic"},
            reason=reason,
        )

    def fake_handheld(self, *, t0: float, t1: float, reason: str = "retention") -> MotionSpec:
        return MotionSpec(
            kind="fake_handheld", start_sec=t0, end_sec=t1,
            intensity=0.35,
            params={"freq": 1.6, "amp_px": 6, "rotation_deg": 0.6},
            reason=reason,
        )

    # ---- Transitions -----------------------------------------------------

    def blur_transition(self, *, t: float, dur: float = 0.18,
                        reason: str = "retention") -> MotionSpec:
        return MotionSpec(
            kind="blur_transition", start_sec=t, end_sec=t + dur,
            intensity=0.7,
            params={"sigma_peak": 12},
            reason=reason,
        )

    def depth_parallax(self, *, t0: float, t1: float,
                       layers: int = 3, reason: str = "retention") -> MotionSpec:
        return MotionSpec(
            kind="depth_parallax", start_sec=t0, end_sec=t1,
            intensity=0.5,
            params={"layers": layers, "max_offset_px": 22},
            reason=reason,
        )
