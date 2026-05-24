"""Engine 9 — HyperFrames Master Mode.

Builds a HyperFrames-compatible scene specification covering:
  • parallax (foreground / midground / background depth layers)
  • kinetic typography (word-by-word reveal, max 3-6 visible words)
  • camera_push, drift, shake
  • speed_ramps
  • depth layers

This module does NOT render. It produces a deterministic JSON spec that the
HyperFrames renderer (or any drop-in replacement) can consume.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from ..core.logger import get_logger
from .visual_composition import Scene, VisualPlan

log = get_logger(__name__)

MAX_WORDS_PER_REVEAL = 6
MIN_WORDS_PER_REVEAL = 3


@dataclass
class HFTextReveal:
    text: str
    start_sec: float
    end_sec: float
    style: str = "kinetic"
    chunk_words: int = 4


@dataclass
class HFSceneSpec:
    index: int
    beat: str
    start_sec: float
    end_sec: float
    primary_asset_url: str | None
    asset_kind: str
    layers: list[dict[str, Any]]                    # parallax depth layers
    camera: dict[str, Any]                          # push / drift / shake / ramp
    text_reveals: list[HFTextReveal]
    overlays: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class HyperFramesPlan:
    scenes: list[HFSceneSpec]
    aspect_ratio: str = "9:16"
    fps: int = 30
    safe_area_pct: float = 0.88


class HyperFramesEngine:
    def build(self, *, visual_plan: VisualPlan) -> HyperFramesPlan:
        specs: list[HFSceneSpec] = []
        for s in visual_plan.scenes:
            specs.append(self._build_scene(s))
        log.info("[hyperframes] built %d scene specs", len(specs))
        return HyperFramesPlan(scenes=specs)

    def _build_scene(self, scene: Scene) -> HFSceneSpec:
        cam = self._camera_for(scene.beat)
        layers = self._layers_for(scene)
        reveals = self._text_reveals(scene)
        return HFSceneSpec(
            index=scene.index,
            beat=scene.beat,
            start_sec=scene.start_sec,
            end_sec=scene.end_sec,
            primary_asset_url=scene.primary_asset.selected_url,
            asset_kind=scene.primary_asset.kind,
            layers=layers,
            camera=cam,
            text_reveals=reveals,
            overlays=[asdict(o) for o in scene.overlays],
        )

    @staticmethod
    def _camera_for(beat: str) -> dict[str, Any]:
        # Per-beat camera personality.
        return {
            "hook":       {"push": 0.25, "drift": [0.02, 0.0], "shake": 0.30, "ramp": [1.0, 1.10]},
            "escalation": {"push": 0.10, "drift": [0.01, -0.005], "shake": 0.10, "ramp": [1.0, 1.0]},
            "reveal":     {"push": -0.05, "drift": [0.0, 0.0], "shake": 0.05, "ramp": [1.0, 0.95]},
            "twist":      {"push": 0.30, "drift": [-0.02, 0.01], "shake": 0.40, "ramp": [0.85, 1.20]},
            "loop":       {"push": 0.18, "drift": [0.01, 0.0], "shake": 0.20, "ramp": [1.0, 1.05]},
        }.get(beat, {"push": 0.10, "drift": [0.0, 0.0], "shake": 0.10, "ramp": [1.0, 1.0]})

    @staticmethod
    def _layers_for(scene: Scene) -> list[dict[str, Any]]:
        # Parallax depth — three default layers; renderer picks ai/stock per kind.
        return [
            {"z": 0.0, "asset_kind": "background", "blur": 0.0, "opacity": 1.0},
            {"z": 0.5, "asset_kind": scene.primary_asset.kind,
             "asset_url": scene.primary_asset.selected_url, "opacity": 1.0},
            {"z": 1.0, "asset_kind": "foreground_overlay",
             "tag": "kinetic_text" if scene.beat in {"hook", "twist", "loop"} else "particles"},
        ]

    @staticmethod
    def _text_reveals(scene: Scene) -> list[HFTextReveal]:
        words = scene.sentence.split()
        if not words:
            return []
        # Chunk into 3-6 word groups
        chunks: list[list[str]] = []
        chunk: list[str] = []
        for w in words:
            chunk.append(w)
            if len(chunk) >= MAX_WORDS_PER_REVEAL:
                chunks.append(chunk)
                chunk = []
        if chunk:
            if chunks and len(chunk) < MIN_WORDS_PER_REVEAL:
                chunks[-1].extend(chunk)
            else:
                chunks.append(chunk)

        duration = max(0.4, scene.end_sec - scene.start_sec)
        per = duration / len(chunks)
        reveals = []
        for i, c in enumerate(chunks):
            reveals.append(HFTextReveal(
                text=" ".join(c),
                start_sec=round(scene.start_sec + i * per, 2),
                end_sec=round(scene.start_sec + (i + 1) * per, 2),
                style="kinetic" if scene.beat in {"hook", "twist", "loop"} else "subtitle",
                chunk_words=len(c),
            ))
        return reveals
