"""Music-trend tracker — pulls the current trending sounds / styles from
open-source sources and emits per-channel BPM + mood weightings the AudioEngine
can use to bias its track picks.

Sources (open / no-key by default):
  • Pixabay Music popular endpoint (key-required, free)
  • Jamendo "popularity_week" ordering (key-required, free)
  • Reddit r/futurebeats, r/synthwave, r/lofi etc. (PRAW)

Output written to data/music_trends.json:

{
  "updated_at": "...",
  "by_channel": {
    "human_decoder":  {"bpm": [80, 110], "moods": ["mysterious", "emotional"], "tags": [...]},
    "ai_money_lab":   {"bpm": [120, 145], "moods": ["future bass", "tech"], "tags": [...]},
    "dramaverse":     {"bpm": [85, 115], "moods": ["dramatic", "tense", "romantic"], "tags": [...]}
  }
}
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Any

import httpx

from ..core.logger import get_logger
from ..core.state import DATA_DIR, load_channel, write_json_atomic

log = get_logger(__name__)

PATH = DATA_DIR / "music_trends.json"


# Per-channel mood seed → forces sources to surface relevant tracks.
SEED_MOODS = {
    "human_decoder": ["mysterious", "emotional", "ambient", "cinematic", "dark beat"],
    "ai_money_lab":  ["future bass", "tech", "edm", "hip hop", "lo-fi"],
    "dramaverse":    ["dramatic", "tense", "romantic", "cinematic", "indie pop"],
}


class MusicTrendTracker:
    def refresh(self, *, channel_ids: list[str]) -> dict[str, Any]:
        out: dict[str, Any] = {"updated_at": dt.datetime.utcnow().isoformat() + "Z",
                               "by_channel": {}}
        for ch in channel_ids:
            try:
                _ = load_channel(ch)
            except Exception:
                continue
            seeds = SEED_MOODS.get(ch, ["cinematic"])
            tracks: list[dict[str, Any]] = []
            for seed in seeds[:3]:
                tracks += _pixabay_popular(seed, limit=8)
                tracks += _jamendo_popular(seed, limit=8)
            bpms = [t["bpm"] for t in tracks if t.get("bpm")]
            tags = []
            for t in tracks:
                if t.get("mood"):
                    tags.extend([s.strip() for s in str(t["mood"]).split(",") if s.strip()])

            bpm_range = _percentile_range(bpms, lo=20, hi=80) if bpms else _default_bpm(ch)
            top_tags = _top_n([s.lower() for s in tags], n=10)

            out["by_channel"][ch] = {
                "bpm": bpm_range,
                "moods": seeds,
                "tags": top_tags,
                "sample_titles": [t.get("title") for t in tracks[:6] if t.get("title")],
            }
            log.info("[music-trends] %s bpm=%s tags=%s", ch, bpm_range, top_tags[:5])
        write_json_atomic(PATH, out)
        return out

    @staticmethod
    def for_channel(channel_id: str) -> dict[str, Any]:
        if not PATH.exists():
            return {"bpm": _default_bpm(channel_id), "moods": SEED_MOODS.get(channel_id, []),
                    "tags": []}
        from ..core.state import load_json
        data = load_json(PATH)
        return data.get("by_channel", {}).get(channel_id, {"bpm": _default_bpm(channel_id),
                                                           "moods": SEED_MOODS.get(channel_id, []),
                                                           "tags": []})


# ------------------------- helpers ------------------------------------------


def _pixabay_popular(seed: str, *, limit: int) -> list[dict[str, Any]]:
    key = os.getenv("PIXABAY_API_KEY", "").strip()
    if not key:
        return []
    try:
        with httpx.Client(timeout=20.0) as c:
            r = c.get("https://pixabay.com/api/music/",
                      params={"key": key, "q": seed, "per_page": limit,
                              "order": "popular"})
            r.raise_for_status()
            data = r.json()
            return [{
                "title": t.get("title"),
                "bpm": int(t.get("bpm", 0)),
                "mood": t.get("tags", ""),
            } for t in data.get("hits", [])]
    except Exception as e:
        log.warning("[music-trends/pixabay] %s", e)
        return []


def _jamendo_popular(seed: str, *, limit: int) -> list[dict[str, Any]]:
    key = os.getenv("JAMENDO_CLIENT_ID", "").strip()
    if not key:
        return []
    try:
        with httpx.Client(timeout=20.0) as c:
            r = c.get("https://api.jamendo.com/v3.0/tracks/",
                      params={"client_id": key, "format": "json", "limit": limit,
                              "tags": seed, "order": "popularity_week",
                              "include": "musicinfo"})
            r.raise_for_status()
            data = r.json()
            return [{
                "title": t.get("name"),
                "bpm": int((t.get("musicinfo") or {}).get("bpm") or 0),
                "mood": ", ".join((t.get("musicinfo") or {})
                                  .get("tags", {}).get("vartags", []))[:80],
            } for t in data.get("results", [])]
    except Exception as e:
        log.warning("[music-trends/jamendo] %s", e)
        return []


def _percentile_range(values: list[int], *, lo: int, hi: int) -> list[int]:
    if not values:
        return [80, 130]
    s = sorted(values)
    return [int(s[max(0, len(s) * lo // 100)]), int(s[min(len(s) - 1, len(s) * hi // 100)])]


def _default_bpm(channel_id: str) -> list[int]:
    return {
        "human_decoder": [80, 110],
        "ai_money_lab":  [120, 145],
        "dramaverse":    [85, 115],
    }.get(channel_id, [90, 130])


def _top_n(items: list[str], *, n: int) -> list[str]:
    from collections import Counter
    c = Counter(items)
    return [k for k, _ in c.most_common(n)]
