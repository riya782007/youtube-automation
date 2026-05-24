"""Pexels stock-footage adapter."""
from __future__ import annotations

import os
from typing import Any

import httpx

from ..core.logger import get_logger

log = get_logger(__name__)

VIDEO_URL = "https://api.pexels.com/videos/search"
PHOTO_URL = "https://api.pexels.com/v1/search"


def search_videos(query: str, *, per_page: int = 5,
                  orientation: str = "portrait") -> list[dict[str, Any]]:
    key = os.getenv("PEXELS_API_KEY", "").strip()
    if not key:
        return []
    try:
        with httpx.Client(timeout=30.0) as c:
            r = c.get(
                VIDEO_URL,
                headers={"Authorization": key},
                params={"query": query, "per_page": per_page,
                        "orientation": orientation, "size": "medium"},
            )
            r.raise_for_status()
            data = r.json()
            return [_normalize_video(v) for v in data.get("videos", [])]
    except Exception as e:
        log.warning("[pexels-video] %r → %s", query, e)
        return []


def search_photos(query: str, *, per_page: int = 5) -> list[dict[str, Any]]:
    key = os.getenv("PEXELS_API_KEY", "").strip()
    if not key:
        return []
    try:
        with httpx.Client(timeout=30.0) as c:
            r = c.get(
                PHOTO_URL,
                headers={"Authorization": key},
                params={"query": query, "per_page": per_page, "orientation": "portrait"},
            )
            r.raise_for_status()
            data = r.json()
            return [{
                "id": p["id"],
                "url": p["src"]["large"],
                "photographer": p.get("photographer"),
                "alt": p.get("alt"),
            } for p in data.get("photos", [])]
    except Exception as e:
        log.warning("[pexels-photo] %r → %s", query, e)
        return []


def _normalize_video(v: dict[str, Any]) -> dict[str, Any]:
    files = sorted(
        v.get("video_files", []),
        key=lambda f: (f.get("height", 0), f.get("width", 0)),
        reverse=True,
    )
    best = next((f for f in files if 720 <= (f.get("height") or 0) <= 1920), files[0] if files else {})
    return {
        "id": v.get("id"),
        "duration": v.get("duration"),
        "url": best.get("link"),
        "width": best.get("width"),
        "height": best.get("height"),
        "user": v.get("user", {}).get("name"),
    }
