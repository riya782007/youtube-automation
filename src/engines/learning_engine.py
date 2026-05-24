"""Engine 12 — Learning Engine.

Ingests YouTube Analytics (or manually-imported CSV) and detects:
  • winning hook fingerprints
  • winning structures (e.g. hook|escalation|reveal|twist|loop)
  • winning emotions
  • winning pacing (avg reset interval)

Down-weights losers, amplifies winners. Stored in analytics_history.json
under `winning_signals` so the script engine + emotion engine can pull from
it on every generation.

The actual YouTube Analytics ingestion is intentionally pluggable — provide
either the YouTube Analytics API or a CSV export.
"""
from __future__ import annotations

import csv
import datetime as dt
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..core.logger import get_logger
from ..core.state import DATA_DIR, load_json, load_retention_targets, write_json_atomic

log = get_logger(__name__)


class LearningEngine:
    def __init__(self) -> None:
        self.path = DATA_DIR / "analytics_history.json"
        self.targets = load_retention_targets()

    # ---- Read --------------------------------------------------------------
    def _store(self) -> dict[str, Any]:
        return load_json(self.path)

    # ---- Write -------------------------------------------------------------
    def record_video(self, video_record: dict[str, Any]) -> None:
        store = self._store()
        store.setdefault("videos", []).append(video_record)
        write_json_atomic(self.path, store)
        log.info("[learn] recorded video id=%s", video_record.get("id"))

    def ingest_analytics(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge fresh analytics rows by video id. Each row should at minimum contain:
        id, channel_id, avg_view_pct, swipe_away_3s_pct, likes_per_view,
        comments_per_view, repeat_view_rate, subs_gained, published_at."""
        store = self._store()
        videos = store.get("videos", [])
        index = {v.get("id"): i for i, v in enumerate(videos) if v.get("id")}

        for row in rows:
            vid = row.get("id")
            if not vid:
                continue
            if vid in index:
                videos[index[vid]].setdefault("analytics", {}).update(row.get("analytics", row))
            else:
                videos.append({
                    "id": vid,
                    "channel_id": row.get("channel_id"),
                    "analytics": row.get("analytics", row),
                    "first_seen_at": dt.datetime.utcnow().isoformat() + "Z",
                })

        store["videos"] = videos
        store["winning_signals"] = self._compute_winning_signals(videos)
        store["last_ingest_at"] = dt.datetime.utcnow().isoformat() + "Z"
        write_json_atomic(self.path, store)
        log.info("[learn] ingested %d rows. winners now: %s",
                 len(rows), {k: len(v) for k, v in store["winning_signals"].items()})
        return store["winning_signals"]

    def ingest_csv(self, csv_path: Path | str) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "id": r.get("id") or r.get("video_id"),
                    "channel_id": r.get("channel_id") or r.get("channel"),
                    "analytics": {
                        "avg_view_pct": _f(r, "avg_view_pct"),
                        "swipe_away_3s_pct": _f(r, "swipe_away_3s_pct"),
                        "likes_per_view": _f(r, "likes_per_view"),
                        "comments_per_view": _f(r, "comments_per_view"),
                        "repeat_view_rate": _f(r, "repeat_view_rate"),
                        "subs_gained": _f(r, "subs_gained"),
                    },
                })
        return self.ingest_analytics(rows)

    # ---- Signal extraction -------------------------------------------------
    def _compute_winning_signals(self, videos: list[dict[str, Any]]) -> dict[str, Any]:
        learning = self.targets.get("learning", {})
        decay = float(learning.get("decay_per_day", 0.97))
        recency = int(learning.get("recency_window_days", 14))
        min_samples = int(learning.get("min_samples_for_adoption", 5))
        amp = float(learning.get("amplification_factor_for_top_decile", 1.6))
        now = dt.datetime.utcnow()

        scored: list[tuple[float, dict[str, Any]]] = []
        for v in videos:
            a = v.get("analytics", {}) or {}
            avg_view_pct = float(a.get("avg_view_pct", 0) or 0)
            likes_per_view = float(a.get("likes_per_view", 0) or 0)
            comments_per_view = float(a.get("comments_per_view", 0) or 0)
            repeat = float(a.get("repeat_view_rate", 0) or 0)
            # composite north-star: retention × (likes + comments + repeat)
            composite = (avg_view_pct / 100.0) * (1 + likes_per_view * 4
                         + comments_per_view * 6 + repeat * 3)
            published = v.get("published_at") or v.get("first_seen_at")
            if published:
                try:
                    p = dt.datetime.fromisoformat(published.replace("Z", ""))
                    days = max(0, (now - p).days)
                    composite *= decay ** days
                    if days > recency * 3:
                        continue
                except Exception:
                    pass
            scored.append((composite, v))

        if len(scored) < min_samples:
            log.info("[learn] only %d samples (need %d) — keeping prior signals",
                     len(scored), min_samples)
            return self._store().get("winning_signals", {})

        scored.sort(key=lambda kv: kv[0], reverse=True)
        top_n = max(1, math.ceil(len(scored) * 0.10))
        top = [kv[1] for kv in scored[:top_n]]

        hooks_c: Counter = Counter()
        structures_c: Counter = Counter()
        emotions_c: Counter = Counter()
        pacing_buckets: dict[str, list[float]] = defaultdict(list)

        for v in top:
            fp = v.get("fingerprint", {}) or {}
            hooks_c[fp.get("hook", "")] += amp
            structures_c[fp.get("structure", "")] += amp
            emotions_c[v.get("emotion", "")] += amp
            ar = (v.get("plan_summary", {}) or {}).get("avg_reset_interval")
            if ar:
                pacing_buckets[v.get("channel_id", "_")].append(float(ar))

        def topk(c: Counter, k: int = 12) -> list[dict[str, Any]]:
            return [{"value": k_, "weight": round(v_, 3)}
                    for k_, v_ in c.most_common(k) if k_]

        return {
            "hooks":      topk(hooks_c),
            "structures": topk(structures_c, 8),
            "emotions":   topk(emotions_c, 6),
            "pacing":     {ch: round(sum(b) / len(b), 2)
                           for ch, b in pacing_buckets.items() if b},
        }

    def winning_signals(self) -> dict[str, Any]:
        return self._store().get("winning_signals", {})


def _f(r: dict[str, Any], k: str) -> float:
    try:
        return float(r.get(k, 0) or 0)
    except Exception:
        return 0.0
