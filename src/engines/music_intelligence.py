"""Music Intelligence — emotional matching, NOT trending-song picking.

Builds an explicit `intensity_curve` per the user's spec:
  0–2s:   high (hook punch)
  2–8s:   medium (set up)
  8–18s:  build (toward reveal)
  18–25s: spike (twist)
  25–30s: fade (loop ending)

Music is selected from the open-source library by matching:
  - emotion (love/wealth/drama → mood map)
  - BPM (target by channel)
  - intensity envelope match
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from ..adapters import free_music
from ..core.logger import get_logger

log = get_logger(__name__)


# Channel → emotional music DNA (mapped to mood keywords + BPM range).
CHANNEL_DNA = {
    "human_decoder": {
        "moods_primary":   ["heartbeat", "lofi", "emotional", "mysterious"],
        "moods_secondary": ["ambient", "cinematic"],
        "bpm_target":      [80, 110],
        "ducking_db":      -14,
    },
    "ai_money_lab": {
        "moods_primary":   ["future bass", "tech", "energetic"],
        "moods_secondary": ["edm", "lo-fi tech"],
        "bpm_target":      [120, 145],
        "ducking_db":      -16,
    },
    "dramaverse": {
        "moods_primary":   ["dramatic", "tense", "humorous"],
        "moods_secondary": ["romantic", "cinematic"],
        "bpm_target":      [85, 115],
        "ducking_db":      -15,
    },
}


@dataclass
class IntensityKeyframe:
    t_sec: float
    intensity: float        # 0..1, multiplied against base music gain
    label: str              # "hook", "setup", "build", "spike", "fade"


@dataclass
class MusicPlan:
    track: dict[str, Any] | None
    candidates: list[dict[str, Any]]
    intensity_curve: list[IntensityKeyframe]
    duck_db: float                              # how much music ducks under voice
    target_bpm: list[int]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


class MusicIntelligence:
    def select(self, *, channel_id: str, emotion: str,
               total_seconds: float) -> MusicPlan:
        dna = CHANNEL_DNA.get(channel_id, CHANNEL_DNA["human_decoder"])

        # Bias mood selection by the emotion tag chosen for this video.
        bias_map = {
            "romantic":   ["heartbeat", "emotional", "romantic", "lofi"],
            "suspense":   ["tense", "dramatic", "mysterious"],
            "surprise":   ["future bass", "energetic", "tech"],
            "excitement": ["energetic", "future bass", "edm"],
            "curiosity":  ["mysterious", "lofi", "emotional", "cinematic"],
            "empathy":    ["lofi", "emotional", "ambient"],
            "serious":    ["cinematic", "ambient", "tense"],
        }
        moods = bias_map.get(emotion, dna["moods_primary"]) + dna["moods_secondary"]
        query = f"{moods[0]} {moods[1] if len(moods) > 1 else ''} background loop"

        # Pull candidates from open-source libs (Pixabay/Jamendo/FMA/curated).
        bpm_lo, bpm_hi = dna["bpm_target"]
        candidates = free_music.search_music(
            query, limit=8,
            min_duration=max(15, int(total_seconds)), max_duration=180,
        )

        # Score by BPM proximity to channel target + license preference.
        scored: list[tuple[float, dict[str, Any]]] = []
        for t in candidates:
            bpm = int(t.get("bpm") or 0)
            bpm_score = 1.0
            if bpm:
                center = (bpm_lo + bpm_hi) / 2
                bpm_score = max(0.1, 1.0 - abs(bpm - center) / max(1, bpm_hi - bpm_lo))
            license_bonus = 1.2 if "cc0" in str(t.get("license", "")).lower() else 1.0
            source_bonus = {"pixabay": 1.10, "jamendo": 1.05,
                            "fma": 1.0, "curated": 0.9}.get(t.get("source"), 0.9)
            scored.append((bpm_score * license_bonus * source_bonus, t))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        track = scored[0][1] if scored else None

        # Build the explicit intensity curve. Anchored relative to total_seconds
        # so it scales for 15s and 60s pieces alike.
        T = max(8.0, total_seconds)
        curve = self._build_curve(T)

        rationale = (
            f"channel={channel_id}; emotion={emotion}; moods={moods[:3]}; "
            f"target_bpm={dna['bpm_target']}; "
            f"selected={track.get('title') if track else 'none'}; "
            f"intensity peaks at hook(0-2s) and twist(~75% mark), fades on loop ending"
        )
        log.info("[music-intel] %s", rationale)
        return MusicPlan(
            track=track,
            candidates=candidates,
            intensity_curve=curve,
            duck_db=dna["ducking_db"],
            target_bpm=dna["bpm_target"],
            rationale=rationale,
        )

    @staticmethod
    def _build_curve(T: float) -> list[IntensityKeyframe]:
        """User-spec curve, scaled to T seconds.

        Anchor points (fraction of T):
          0.00  hook       1.00
          0.07  hook end   0.95   (tight punch)
          0.27  setup low  0.55   (let voice breathe)
          0.60  build mid  0.78
          0.83  spike      1.00   (twist)
          0.93  fade pre   0.55
          1.00  end        0.18
        """
        anchors = [
            (0.00, 1.00, "hook"),
            (0.07, 0.95, "hook"),
            (0.27, 0.55, "setup"),
            (0.60, 0.78, "build"),
            (0.83, 1.00, "spike"),
            (0.93, 0.55, "fade"),
            (1.00, 0.18, "fade"),
        ]
        return [IntensityKeyframe(round(frac * T, 2), inten, label)
                for frac, inten, label in anchors]
