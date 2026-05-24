"""Open-source / royalty-free music & SFX adapter.

Sources (all CC0 / CC-BY / explicit royalty-free):
  • Pixabay Music    — https://pixabay.com/api/docs/  (CC0-style content license)
  • Jamendo Music    — https://developer.jamendo.com/v3.0  (CC licensed)
  • Free Music Archive — open catalog (no key, public JSON via fma-archive mirrors)
  • Freesound        — https://freesound.org/apiv2/  (mostly CC0 / CC-BY for SFX)

Each function returns a list of normalized track dicts:
  {id, title, artist, license, url, preview_url, duration, bpm, mood, source}

If no API keys are present, returns curated fallback tracks bundled by mood
so the system stays operational without any external network call.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from ..core.logger import get_logger

log = get_logger(__name__)


# ---------- PUBLIC API -------------------------------------------------------


def search_music(query: str, *, limit: int = 5,
                 min_duration: int = 15, max_duration: int = 90,
                 sources: tuple[str, ...] = ("pixabay", "jamendo", "fma")) -> list[dict[str, Any]]:
    """Search across all configured open-source sources, dedupe by title+artist."""
    results: list[dict[str, Any]] = []
    if "pixabay" in sources:
        results += _pixabay_music(query, limit=limit)
    if "jamendo" in sources:
        results += _jamendo_music(query, limit=limit, min_duration=min_duration,
                                  max_duration=max_duration)
    if "fma" in sources:
        results += _fma_music(query, limit=limit)

    seen, deduped = set(), []
    for t in results:
        key = (t.get("title", "").lower(), t.get("artist", "").lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)

    if not deduped:
        deduped = _fallback_tracks(query, limit=limit)
    return deduped[:limit]


def search_sfx(label: str, *, limit: int = 3) -> list[dict[str, Any]]:
    """Find an SFX clip on Freesound by label (whoosh, ding, heartbeat, etc.)."""
    out = _freesound(label, limit=limit)
    if out:
        return out
    return _fallback_sfx(label, limit=limit)


# ---------- PIXABAY ---------------------------------------------------------


def _pixabay_music(query: str, *, limit: int) -> list[dict[str, Any]]:
    key = os.getenv("PIXABAY_API_KEY", "").strip()
    if not key:
        return []
    try:
        with httpx.Client(timeout=20.0) as c:
            r = c.get(
                "https://pixabay.com/api/music/",   # documented music endpoint
                params={"key": key, "q": query, "per_page": limit, "safesearch": "true"},
            )
            r.raise_for_status()
            data = r.json()
            return [{
                "id": str(t.get("id")),
                "title": t.get("title", ""),
                "artist": t.get("user", ""),
                "license": "Pixabay Content License (royalty-free)",
                "url": t.get("audio") or t.get("preview"),
                "preview_url": t.get("preview"),
                "duration": int(t.get("duration", 0)),
                "bpm": int(t.get("bpm", 0)),
                "mood": ", ".join(t.get("tags", "").split(",")[:3]),
                "source": "pixabay",
            } for t in data.get("hits", [])]
    except Exception as e:
        log.warning("[free-music/pixabay] %s", e)
        return []


# ---------- JAMENDO ---------------------------------------------------------


def _jamendo_music(query: str, *, limit: int,
                   min_duration: int, max_duration: int) -> list[dict[str, Any]]:
    key = os.getenv("JAMENDO_CLIENT_ID", "").strip()
    if not key:
        return []
    try:
        with httpx.Client(timeout=20.0) as c:
            r = c.get(
                "https://api.jamendo.com/v3.0/tracks/",
                params={
                    "client_id": key, "format": "json", "limit": limit,
                    "search": query, "include": "musicinfo licenses",
                    "audioformat": "mp32",
                    "durationbetween": f"{min_duration}_{max_duration}",
                    "ccsa": "true",   # share-alike & remix safe
                },
            )
            r.raise_for_status()
            data = r.json()
            return [{
                "id": str(t.get("id")),
                "title": t.get("name", ""),
                "artist": t.get("artist_name", ""),
                "license": (t.get("license_ccurl") or "Jamendo CC"),
                "url": t.get("audio"),
                "preview_url": t.get("audio"),
                "duration": int(t.get("duration", 0)),
                "bpm": int((t.get("musicinfo") or {}).get("bpm") or 0),
                "mood": ", ".join((t.get("musicinfo") or {}).get("tags", {}).get("vartags", []))[:60],
                "source": "jamendo",
            } for t in data.get("results", [])]
    except Exception as e:
        log.warning("[free-music/jamendo] %s", e)
        return []


# ---------- FMA -------------------------------------------------------------


def _fma_music(query: str, *, limit: int) -> list[dict[str, Any]]:
    """Free Music Archive — public mirrors. Best-effort, no key required."""
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as c:
            r = c.get(
                "https://freemusicarchive.org/api/get/tracks.json",
                params={"api_key": os.getenv("FMA_API_KEY", "").strip() or "FREEFM",
                        "search": query, "limit": limit},
            )
            if r.status_code != 200:
                return []
            data = r.json()
            return [{
                "id": str(t.get("track_id")),
                "title": t.get("track_title", ""),
                "artist": t.get("artist_name", ""),
                "license": t.get("license_title", "FMA / Creative Commons"),
                "url": t.get("track_file_url"),
                "preview_url": t.get("track_file_url"),
                "duration": int(t.get("track_duration_seconds") or 0),
                "bpm": 0,
                "mood": t.get("track_genres", ""),
                "source": "fma",
            } for t in (data.get("dataset") or [])]
    except Exception as e:
        log.warning("[free-music/fma] %s", e)
        return []


# ---------- FREESOUND (SFX) -------------------------------------------------


def _freesound(label: str, *, limit: int) -> list[dict[str, Any]]:
    token = os.getenv("FREESOUND_API_KEY", "").strip()
    if not token:
        return []
    try:
        with httpx.Client(timeout=20.0) as c:
            r = c.get(
                "https://freesound.org/apiv2/search/text/",
                params={
                    "token": token, "query": label,
                    "filter": "license:\"Creative Commons 0\"",
                    "fields": "id,name,username,license,previews,duration,tags",
                    "page_size": limit,
                },
            )
            r.raise_for_status()
            data = r.json()
            return [{
                "id": str(t.get("id")),
                "title": t.get("name", ""),
                "artist": t.get("username", ""),
                "license": t.get("license", "CC0"),
                "url": (t.get("previews") or {}).get("preview-hq-mp3"),
                "preview_url": (t.get("previews") or {}).get("preview-hq-mp3"),
                "duration": float(t.get("duration", 0)),
                "bpm": 0,
                "mood": ", ".join(t.get("tags", [])[:3]),
                "source": "freesound",
            } for t in data.get("results", [])]
    except Exception as e:
        log.warning("[free-music/freesound] %s", e)
        return []


# ---------- FALLBACK CURATED LIST ------------------------------------------


_CURATED: dict[str, list[dict[str, Any]]] = {
    "mysterious": [{
        "title": "Mysterious Drone (curated)", "artist": "Public Domain",
        "license": "CC0", "duration": 60, "bpm": 92,
        "mood": "mysterious", "source": "curated",
    }],
    "emotional": [{
        "title": "Soft Piano Tears (curated)", "artist": "Public Domain",
        "license": "CC0", "duration": 60, "bpm": 70,
        "mood": "emotional", "source": "curated",
    }],
    "energetic": [{
        "title": "Tech Hustle Loop (curated)", "artist": "Public Domain",
        "license": "CC0", "duration": 60, "bpm": 128,
        "mood": "energetic", "source": "curated",
    }],
    "future bass": [{
        "title": "Future Bass Pulse (curated)", "artist": "Public Domain",
        "license": "CC0", "duration": 60, "bpm": 140,
        "mood": "future bass", "source": "curated",
    }],
    "lo-fi tech": [{
        "title": "LoFi Tech Beat (curated)", "artist": "Public Domain",
        "license": "CC0", "duration": 60, "bpm": 90,
        "mood": "lo-fi tech", "source": "curated",
    }],
    "dramatic": [{
        "title": "Dramatic Cinematic Bed (curated)", "artist": "Public Domain",
        "license": "CC0", "duration": 60, "bpm": 100,
        "mood": "dramatic", "source": "curated",
    }],
    "tense": [{
        "title": "Tense Suspense Pulse (curated)", "artist": "Public Domain",
        "license": "CC0", "duration": 60, "bpm": 110,
        "mood": "tense", "source": "curated",
    }],
    "romantic": [{
        "title": "Romantic Strings (curated)", "artist": "Public Domain",
        "license": "CC0", "duration": 60, "bpm": 78,
        "mood": "romantic", "source": "curated",
    }],
    "cinematic": [{
        "title": "Cinematic Hook (curated)", "artist": "Public Domain",
        "license": "CC0", "duration": 60, "bpm": 100,
        "mood": "cinematic", "source": "curated",
    }],
}


def _fallback_tracks(query: str, *, limit: int) -> list[dict[str, Any]]:
    q = query.lower()
    matches: list[dict[str, Any]] = []
    for mood, tracks in _CURATED.items():
        if mood in q:
            matches.extend(tracks)
    if not matches:
        matches = _CURATED["cinematic"]
    return matches[:limit]


_SFX_FALLBACK = {
    "whoosh":     {"title": "whoosh-fast", "license": "CC0"},
    "rumble":     {"title": "low-rumble", "license": "CC0"},
    "boing":      {"title": "comedic-boing", "license": "CC0"},
    "ding":       {"title": "tiny-ding", "license": "CC0"},
    "pop":        {"title": "soft-pop", "license": "CC0"},
    "impact":     {"title": "drum-impact", "license": "CC0"},
    "transition": {"title": "swoosh-transition", "license": "CC0"},
    "click":      {"title": "ui-click", "license": "CC0"},
    "heartbeat":  {"title": "heartbeat-soft", "license": "CC0"},
    "pulse":      {"title": "neon-pulse", "license": "CC0"},
    "keyboard":   {"title": "keyboard-tap", "license": "CC0"},
    "cash register": {"title": "cash-register", "license": "CC0"},
    "notification ping": {"title": "notif-ping", "license": "CC0"},
    "ui swoosh":  {"title": "ui-swoosh", "license": "CC0"},
    "typing":     {"title": "phone-typing", "license": "CC0"},
    "notification": {"title": "phone-notif", "license": "CC0"},
    "vibration":  {"title": "phone-vibrate", "license": "CC0"},
    "phone buzz": {"title": "phone-buzz", "license": "CC0"},
    "vinyl crackle": {"title": "vinyl-crackle", "license": "CC0"},
    "soft tick": {"title": "soft-tick", "license": "CC0"},
    "sub_bass_punch": {"title": "sub-bass-punch", "license": "CC0"},
    "loop_sting": {"title": "loop-sting", "license": "CC0"},
}


def _fallback_sfx(label: str, *, limit: int) -> list[dict[str, Any]]:
    label_l = label.lower()
    cur = _SFX_FALLBACK.get(label_l) or _SFX_FALLBACK.get("click")
    return [{
        "id": f"curated_{label_l.replace(' ', '_')}",
        "title": cur["title"],
        "artist": "curated",
        "license": cur["license"],
        "url": None,
        "preview_url": None,
        "duration": 0.5,
        "bpm": 0,
        "mood": label,
        "source": "curated",
    }][:limit]
