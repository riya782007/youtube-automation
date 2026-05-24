"""Filesystem-backed JSON/YAML state with safe atomic writes.

Used by every engine that needs to read/write memory:
data/used_hooks.json, data/used_topics.json, data/competitor_patterns.json, etc.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
ARTIFACT_DIR = ROOT / "artifacts"


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path | str, data: Any) -> None:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        shutil.move(tmp, p)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def append_entry(path: Path | str, entry: dict[str, Any]) -> None:
    """Append to a JSON file with `entries: []` shape."""
    data = load_json(path)
    data.setdefault("entries", []).append(entry)
    write_json_atomic(path, data)


def load_channel(channel_id: str) -> dict[str, Any]:
    cfg = load_yaml(CONFIG_DIR / "channels.yaml")
    channels = cfg.get("channels", {})
    if channel_id not in channels:
        raise KeyError(f"Unknown channel '{channel_id}'. Known: {list(channels)}")
    return channels[channel_id]


def load_emotions() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "emotions.yaml")


def load_retention_targets() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "retention_targets.yaml")


def all_channel_ids() -> list[str]:
    cfg = load_yaml(CONFIG_DIR / "channels.yaml")
    return list(cfg.get("channels", {}).keys())


_ensure_dirs()
