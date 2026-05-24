#!/usr/bin/env python3
"""Direct entry: `python scripts/run_pipeline.py human_decoder`."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.orchestrator import Orchestrator

load_dotenv()


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: run_pipeline.py <channel_id> [topic ...]")
        sys.exit(1)
    channel = sys.argv[1]
    topic = " ".join(sys.argv[2:]) or None
    plan = Orchestrator().generate_video(channel, topic=topic)
    print(f"OK {plan.id}  q={plan.quality.get('average')}  → {plan.artifact_dir}")


if __name__ == "__main__":
    main()
