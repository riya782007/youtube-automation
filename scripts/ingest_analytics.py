#!/usr/bin/env python3
"""Direct entry: `python scripts/ingest_analytics.py /path/to/analytics.csv`.

Expected CSV columns (any subset works):
  id, channel_id, avg_view_pct, swipe_away_3s_pct, likes_per_view,
  comments_per_view, repeat_view_rate, subs_gained
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.engines.learning_engine import LearningEngine

load_dotenv()


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: ingest_analytics.py <csv_path>")
        sys.exit(1)
    signals = LearningEngine().ingest_csv(sys.argv[1])
    print("Winning signals updated.")
    for k, v in signals.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
