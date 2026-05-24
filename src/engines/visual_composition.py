"""Engine 7 — Visual Composition.

Composition target per channel.yaml asset_mix (40/30/20/10 default):
  • stock      — Pexels videos
  • ai_visual  — placeholder slot for AI-generated frames (renderer-specific)
  • overlay    — UI captures, screenshots, kinetic text panes
  • meme       — reaction clips / meme cutaways

Outputs a per-scene asset list. Renderer (HyperFrames) is responsible for
actually downloading + cutting; we only PLAN here so all decisions are
inspectable and deterministic given the same inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from ..adapters import pexels
from ..core.llm import LLM
from ..core.logger import get_logger

log = get_logger(__name__)


@dataclass
class SceneAsset:
    kind: str                    # stock | ai_visual | overlay | meme
    query: str
    candidates: list[dict[str, Any]] = field(default_factory=list)
    selected_url: str | None = None
    notes: str = ""


@dataclass
class Scene:
    index: int
    beat: str                    # hook | escalation | reveal | twist | loop
    start_sec: float
    end_sec: float
    sentence: str
    primary_asset: SceneAsset
    overlays: list[SceneAsset] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class VisualPlan:
    scenes: list[Scene]
    asset_mix_actual: dict[str, float]


class VisualComposition:
    def __init__(self) -> None:
        self.llm = LLM()

    def plan(self, *, channel: dict[str, Any], script_beats: list[Any],
            voice_seconds: float) -> VisualPlan:
        # Map beat sentences to even time slots within the beat window.
        scenes: list[Scene] = []
        idx = 0
        for beat in script_beats:
            sentences = list(beat.sentences)
            if not sentences:
                continue
            window = max(0.6, (beat.end_sec - beat.start_sec) / len(sentences))
            for j, sent in enumerate(sentences):
                start = beat.start_sec + j * window
                end = start + window
                kind = self._kind_for(channel, beat.name, j)
                query = self._query_for(channel, beat.name, sent)
                primary = self._fetch_asset(kind, query)
                overlays = self._overlays_for(channel, beat.name, sent)
                scenes.append(Scene(
                    index=idx, beat=beat.name,
                    start_sec=round(start, 2), end_sec=round(end, 2),
                    sentence=sent,
                    primary_asset=primary,
                    overlays=overlays,
                ))
                idx += 1

        # Stretch/compress to actual voice length
        if scenes and voice_seconds > 0:
            self._rescale(scenes, voice_seconds)

        # Compute actual asset mix
        counts: dict[str, int] = {}
        for s in scenes:
            counts[s.primary_asset.kind] = counts.get(s.primary_asset.kind, 0) + 1
        total = sum(counts.values()) or 1
        actual = {k: round(v / total, 2) for k, v in counts.items()}
        log.info("[visual] %d scenes  asset_mix=%s", len(scenes), actual)
        return VisualPlan(scenes=scenes, asset_mix_actual=actual)

    # ---- Helpers ---------------------------------------------------------
    def _kind_for(self, channel: dict[str, Any], beat: str, j: int) -> str:
        mix = channel["visual_style"].get(
            "asset_mix",
            {"stock": 0.4, "ai_visual": 0.3, "overlay": 0.2, "meme": 0.1},
        )
        # Beat-aware bias: hooks should hit hard with overlays/memes, reveals get stock,
        # twists get AI visuals, loops mirror hooks.
        bias = {
            "hook":       {"overlay": 1.4, "meme": 1.4, "stock": 0.7, "ai_visual": 0.7},
            "escalation": {"stock": 1.2, "overlay": 1.1},
            "reveal":     {"stock": 1.4, "ai_visual": 1.1},
            "twist":      {"ai_visual": 1.5, "meme": 1.2, "stock": 0.7},
            "loop":       {"overlay": 1.3, "meme": 1.3},
        }.get(beat, {})
        # Pick deterministically using j for stable visual rhythm.
        scored = sorted(
            ((k, mix.get(k, 0) * bias.get(k, 1.0)) for k in mix),
            key=lambda kv: kv[1], reverse=True,
        )
        return scored[j % len(scored)][0]

    def _query_for(self, channel: dict[str, Any], beat: str, sentence: str) -> str:
        # Cheap query synthesis: 3-5 keyword phrase from sentence + channel topic.
        words = [w.strip(",.!?\"'") for w in sentence.split() if len(w) > 3]
        keywords = " ".join(words[:4])
        seed = (channel.get("topic_seeds") or ["lifestyle"])[0]
        return f"{keywords} {seed}".strip() or seed

    def _fetch_asset(self, kind: str, query: str) -> SceneAsset:
        if kind == "stock":
            results = pexels.search_videos(query, per_page=3)
            return SceneAsset(
                kind=kind, query=query,
                candidates=results,
                selected_url=results[0]["url"] if results else None,
                notes="pexels-video",
            )
        if kind == "ai_visual":
            return SceneAsset(
                kind=kind, query=query,
                notes="hand-off to HyperFrames AI image gen",
            )
        if kind == "overlay":
            photos = pexels.search_photos(query, per_page=2)
            return SceneAsset(
                kind=kind, query=query,
                candidates=photos,
                selected_url=photos[0]["url"] if photos else None,
                notes="overlay/UI capture surface",
            )
        # meme
        return SceneAsset(
            kind=kind, query=query,
            notes="renderer pulls from /assets/memes/<channel>/",
        )

    def _overlays_for(self, channel: dict[str, Any], beat: str, sentence: str) -> list[SceneAsset]:
        overlays: list[SceneAsset] = []
        # Always overlay kinetic typography for the primary noun in hook/twist/loop.
        if beat in {"hook", "twist", "loop"}:
            overlays.append(SceneAsset(
                kind="overlay", query="kinetic_typography",
                notes=f"3-6 word reveal over: {sentence[:60]}",
            ))
        return overlays

    def _rescale(self, scenes: list[Scene], total_sec: float) -> None:
        cur_total = max(s.end_sec for s in scenes)
        if cur_total <= 0:
            return
        ratio = total_sec / cur_total
        for s in scenes:
            s.start_sec = round(s.start_sec * ratio, 2)
            s.end_sec = round(s.end_sec * ratio, 2)
