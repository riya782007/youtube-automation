"""Engine 1 — Trend Intelligence.

Pulls signals from:
  • Reddit (PRAW)
  • Google Trends (pytrends)
  • YouTube Search (youtube-data-api) — competitor titles + comments

Distills into competitor_patterns.json with FOUR abstracted layers:
  • hook_patterns       — sentence frames, NEVER raw copies
  • title_structures    — token-level templates
  • emotional_patterns  — surprise/curiosity/empathy/etc. distribution
  • pacing_styles       — fast / medium / slow per channel category
  • comment_triggers    — what made viewers comment

Rule: never blindly copy. Translate to original templates.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter
from typing import Any

from ..core.llm import LLM
from ..core.logger import get_logger
from ..core.state import DATA_DIR, load_channel, load_json, write_json_atomic

log = get_logger(__name__)


class TrendIntelligence:
    def __init__(self) -> None:
        self.llm = LLM()
        self.path = DATA_DIR / "competitor_patterns.json"

    # ---- Public ------------------------------------------------------------
    def scan(self, channel_id: str, *, max_items: int = 30) -> dict[str, Any]:
        log.info("[trend-intel] scanning channel=%s", channel_id)
        channel = load_channel(channel_id)
        seeds: list[str] = channel.get("topic_seeds", [])

        signals: list[dict[str, Any]] = []
        signals += self._reddit(seeds, max_items=max_items)
        signals += self._google_trends(seeds)
        signals += self._youtube_search(seeds, max_items=max_items)

        patterns = self._distill(channel_id, signals)
        self._merge_and_save(channel_id, patterns, signals)
        log.info("[trend-intel] %d signals → %d hook patterns",
                 len(signals), len(patterns.get("hook_patterns", [])))
        return patterns

    # ---- Sources -----------------------------------------------------------
    def _reddit(self, seeds: list[str], *, max_items: int) -> list[dict[str, Any]]:
        try:
            import praw  # type: ignore
            import os
            cid = os.getenv("REDDIT_CLIENT_ID")
            csec = os.getenv("REDDIT_CLIENT_SECRET")
            if not (cid and csec):
                return []
            r = praw.Reddit(
                client_id=cid,
                client_secret=csec,
                user_agent=os.getenv("REDDIT_USER_AGENT", "yt-os/0.1"),
            )
            out = []
            for seed in seeds[:5]:
                for post in r.subreddit("all").search(seed, limit=max_items // 5):
                    out.append({
                        "source": "reddit",
                        "seed": seed,
                        "title": post.title,
                        "score": post.score,
                        "comments": post.num_comments,
                    })
            return out
        except Exception as e:
            log.warning("reddit source skipped: %s", e)
            return []

    def _google_trends(self, seeds: list[str]) -> list[dict[str, Any]]:
        try:
            from pytrends.request import TrendReq  # type: ignore
            tr = TrendReq()
            out = []
            for seed in seeds[:3]:
                tr.build_payload([seed], timeframe="now 7-d", geo="IN")
                related = tr.related_queries().get(seed, {}) or {}
                rising = related.get("rising")
                if rising is None:
                    continue
                for _, row in rising.head(8).iterrows():
                    out.append({
                        "source": "google_trends",
                        "seed": seed,
                        "title": row["query"],
                        "score": int(row.get("value", 0)),
                    })
            return out
        except Exception as e:
            log.warning("google-trends source skipped: %s", e)
            return []

    def _youtube_search(self, seeds: list[str], *, max_items: int) -> list[dict[str, Any]]:
        try:
            import os
            from googleapiclient.discovery import build  # type: ignore
            key = os.getenv("YOUTUBE_API_KEY")
            if not key:
                return []
            svc = build("youtube", "v3", developerKey=key, cache_discovery=False)
            out = []
            for seed in seeds[:5]:
                resp = svc.search().list(
                    q=seed, part="snippet", type="video",
                    videoDuration="short",
                    maxResults=min(10, max_items // max(1, len(seeds))),
                    regionCode="IN", relevanceLanguage="hi",
                ).execute()
                for item in resp.get("items", []):
                    out.append({
                        "source": "youtube",
                        "seed": seed,
                        "title": item["snippet"]["title"],
                        "channel": item["snippet"]["channelTitle"],
                    })
            return out
        except Exception as e:
            log.warning("youtube-search source skipped: %s", e)
            return []

    # ---- Distillation via LLM ---------------------------------------------
    def _distill(self, channel_id: str, signals: list[dict[str, Any]]) -> dict[str, Any]:
        if not signals:
            log.info("[trend-intel] no signals — returning empty pattern set")
            return {"hook_patterns": [], "title_structures": [],
                    "emotional_patterns": [], "pacing_styles": [],
                    "comment_triggers": []}

        sample_titles = [s["title"] for s in signals if s.get("title")][:60]
        system = (
            "You are an attention engineer for short-form Hinglish YouTube. "
            "Your job: distill viral PATTERNS, not raw copies. "
            "Never reproduce a title verbatim — abstract the shape."
        )
        user = f"""\
Channel: {channel_id}
Recent viral / search-spike titles (mixed Hinglish + English):

{chr(10).join('- ' + t for t in sample_titles)}

Return JSON with exactly these keys:
{{
  "hook_patterns":      [ "<token-level frame using {{slot}} placeholders>", ... ],   // ~8 entries
  "title_structures":   [ "<token-level frame>", ... ],                                // ~6 entries
  "emotional_patterns": [ {{"emotion":"surprise","weight":0.0..1.0}}, ... ],
  "pacing_styles":      [ "fast-cut" | "slow-build" | "loop-bait" | "list-of-3" ],
  "comment_triggers":   [ "<short imperative or A/B>" , ... ]
}}
Rules:
 - hook_patterns must NEVER be a verbatim title. Use slots like {{verb}}, {{thing}}, {{age}}.
 - emotional_patterns weights must sum to ~1.0
 - return ONLY the JSON object, no prose
"""
        out = self.llm.complete_json(system, user, max_tokens=1400, temperature=0.3)
        # Sanity defaults
        for k in ("hook_patterns", "title_structures", "comment_triggers", "pacing_styles"):
            out.setdefault(k, [])
        out.setdefault("emotional_patterns", [])
        return out

    # ---- Merge into store --------------------------------------------------
    def _merge_and_save(self, channel_id: str, patterns: dict[str, Any],
                        raw: list[dict[str, Any]]) -> None:
        store = load_json(self.path) if self.path.exists() else {
            "schema_version": 1,
            "patterns": {"hook_patterns": [], "title_structures": [],
                         "emotional_patterns": [], "pacing_styles": [],
                         "comment_triggers": []},
            "raw_samples_archive": [],
        }
        # Per-channel namespacing
        per = store["patterns"].setdefault("_by_channel", {}).setdefault(channel_id, {})
        for k, v in patterns.items():
            existing = per.get(k, [])
            merged = _dedupe_merge(existing, v)
            per[k] = merged
        store["last_updated"] = dt.datetime.utcnow().isoformat() + "Z"
        store["raw_samples_archive"] = (
            store.get("raw_samples_archive", []) + raw
        )[-500:]   # cap archive
        write_json_atomic(self.path, store)


def _dedupe_merge(a: list, b: list) -> list:
    seen, out = set(), []
    for item in (a or []) + (b or []):
        key = repr(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def emotion_distribution(channel_id: str) -> dict[str, float]:
    """Convenience: read current emotion weights from competitor_patterns."""
    store = load_json(DATA_DIR / "competitor_patterns.json")
    per = store.get("patterns", {}).get("_by_channel", {}).get(channel_id, {})
    raw = per.get("emotional_patterns", [])
    counter: Counter = Counter()
    for entry in raw:
        if isinstance(entry, dict):
            counter[entry.get("emotion", "curiosity")] += float(entry.get("weight", 0))
    if not counter:
        return {"curiosity": 1.0}
    total = sum(counter.values()) or 1.0
    return {k: v / total for k, v in counter.items()}
