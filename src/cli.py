"""yt-os — command-line interface for the YouTube Content OS.

Commands:
  yt-os trend-scan [--channel <id>]
  yt-os generate   --channel <id> [--topic "<topic>"]
  yt-os batch      [--videos-per-channel N]
  yt-os ingest-analytics --csv <path>
  yt-os status
"""
from __future__ import annotations

import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.table import Table

from .core.logger import get_logger
from .core.state import (
    DATA_DIR,
    OUTPUT_DIR,
    all_channel_ids,
    load_json,
    load_retention_targets,
)
from .engines.learning_engine import LearningEngine
from .engines.music_trends import MusicTrendTracker
from .engines.review_stage import ReviewStage
from .engines.trend_intelligence import TrendIntelligence
from .orchestrator import Orchestrator

load_dotenv()
log = get_logger("yt-os.cli")
app = typer.Typer(help="YouTube Content OS — autonomous retention-engineered content.")
console = Console()


@app.command("trend-scan")
def trend_scan(
    channel: str = typer.Option(None, "--channel", help="Channel id, or all if omitted."),
):
    """Refresh competitor_patterns.json from Reddit + Google Trends + YouTube Search."""
    ti = TrendIntelligence()
    targets = [channel] if channel else all_channel_ids()
    for ch in targets:
        ti.scan(ch)
    console.print(f"[green]✓[/green] Trend scan complete for: {', '.join(targets)}")


@app.command("music-scan")
def music_scan(
    channel: str = typer.Option(None, "--channel", help="Channel id, or all if omitted."),
):
    """Refresh data/music_trends.json from Pixabay/Jamendo popular endpoints."""
    targets = [channel] if channel else all_channel_ids()
    out = MusicTrendTracker().refresh(channel_ids=targets)
    console.print(f"[green]✓[/green] music trends refreshed at {out['updated_at']}")
    for ch, info in out.get("by_channel", {}).items():
        console.print(f"  [cyan]{ch}[/cyan]  bpm={info['bpm']}  moods={info['moods'][:3]}")


@app.command("generate")
def generate(
    channel: str = typer.Option(..., "--channel", help="Channel id."),
    topic: str = typer.Option(None, "--topic", help="Override the auto-picked topic."),
    render: bool = typer.Option(False, "--render", help="Also render mp4 with open-source FFmpeg renderer."),
):
    """Run the full 12-step pipeline once for one channel."""
    plan = Orchestrator().generate_video(channel, topic=topic, render=render)
    console.print(
        f"[bold green]✓ {plan.id}[/bold green]  "
        f"q={plan.quality.get('average', '?')}  "
        f"emotion={plan.emotion}  "
        f"duration={plan.duration_sec}s"
    )
    console.print(f"  title:  [cyan]{plan.title}[/cyan]")
    console.print(f"  hook:   {plan.fingerprint.get('hook', '')}")
    console.print(f"  loop:   {plan.fingerprint.get('ending', '')}")
    console.print(f"  output: [dim]{OUTPUT_DIR / (plan.id + '.json')}[/dim]")
    if plan.render.get("video_path"):
        console.print(f"  video:  [green]{plan.render['video_path']}[/green]")
        if plan.render.get("music_track"):
            mt = plan.render["music_track"]
            console.print(f"  music:  {mt.get('title')} — {mt.get('artist')} ({mt.get('license')})")


@app.command("batch")
def batch(
    videos_per_channel: int = typer.Option(
        2, "--videos-per-channel", help="How many videos to plan per channel.",
    ),
):
    """Generate multiple videos across all channels, one daily run."""
    orch = Orchestrator()
    for ch in all_channel_ids():
        for _ in range(videos_per_channel):
            try:
                plan = orch.generate_video(ch)
                console.print(f"[green]✓[/green] {ch} {plan.id} q={plan.quality.get('average')}")
            except Exception as e:
                console.print(f"[red]✗[/red] {ch} failed: {e}")


@app.command("sample")
def sample(
    channel: str = typer.Option(..., "--channel", help="Channel id."),
    topic: str = typer.Option(None, "--topic", help="Override the auto-picked topic."),
):
    """Produce ONE sample (mp4 + thumbnail + script + timeline + reports), then STOP.

    No bulk production runs until the sample is approved.
    """
    bundle = ReviewStage().produce_sample(channel, topic=topic)
    console.rule("[bold green]SAMPLE READY — review before producing more[/bold green]")
    console.print(f"Sample dir: [cyan]{bundle.sample_dir}[/cyan]")
    console.print(f"  1. MP4              {bundle.mp4_path}")
    console.print(f"  2. Thumbnail        {bundle.thumbnail_path}")
    console.print(f"  3. Script           {bundle.script_path}")
    console.print(f"  4. Timeline         {bundle.timeline_path}")
    console.print(f"  5. Voice analysis   {bundle.voice_analysis_path}")
    console.print(f"  6. Retention pred.  {bundle.retention_prediction_path}")
    console.print(f"  7. Editing notes    {bundle.editing_explanation_path}")
    console.print(f"  Quality report      {bundle.quality_report_path}")
    console.print(f"  Summary             {bundle.summary_path}")
    console.print()
    console.print("[bold yellow]STOP — production halts pending your approval.[/bold yellow]")


@app.command("ingest-analytics")
def ingest_analytics(
    csv_path: Path = typer.Option(..., "--csv", exists=True, help="CSV export from YouTube Studio."),
):
    """Ingest a YouTube Analytics CSV and recompute winning signals."""
    signals = LearningEngine().ingest_csv(csv_path)
    console.print("[green]✓[/green] winning signals updated")
    console.print(signals)


@app.command("status")
def status():
    """Show retention targets, memory health, and last winning signals."""
    t = load_retention_targets()
    console.rule("[bold]Retention targets[/bold]")
    sf = t.get("targets", {}).get("short_form", {})
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("metric", style="cyan")
    table.add_column("target")
    for k, v in sf.items():
        table.add_row(k, str(v))
    console.print(table)

    console.rule("[bold]Memory[/bold]")
    mem = Table(box=box.SIMPLE_HEAVY)
    mem.add_column("file", style="cyan")
    mem.add_column("entries")
    for f in ("used_hooks.json", "used_topics.json", "used_endings.json", "used_structures.json"):
        try:
            data = load_json(DATA_DIR / f)
            mem.add_row(f, str(len(data.get("entries", []))))
        except Exception as e:
            mem.add_row(f, f"err: {e}")
    console.print(mem)

    console.rule("[bold]Winning signals[/bold]")
    try:
        ws = LearningEngine().winning_signals()
        if not ws:
            console.print("[yellow]none yet — run ingest-analytics[/yellow]")
        else:
            for k, v in ws.items():
                console.print(f"[cyan]{k}[/cyan] → {v}")
    except Exception as e:
        console.print(f"[red]signal error: {e}[/red]")

    keys_present = {
        "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
        "OPENAI_API_KEY":    bool(os.getenv("OPENAI_API_KEY")),
        "SARVAM_API_KEY":    bool(os.getenv("SARVAM_API_KEY")),
        "ELEVENLABS_API_KEY":bool(os.getenv("ELEVENLABS_API_KEY")),
        "PEXELS_API_KEY":    bool(os.getenv("PEXELS_API_KEY")),
        "YOUTUBE_API_KEY":   bool(os.getenv("YOUTUBE_API_KEY")),
        "REDDIT_CLIENT_ID":  bool(os.getenv("REDDIT_CLIENT_ID")),
    }
    console.rule("[bold]API keys present[/bold]")
    kt = Table(box=box.SIMPLE_HEAVY)
    kt.add_column("env", style="cyan")
    kt.add_column("ok")
    for k, ok in keys_present.items():
        kt.add_row(k, "[green]yes[/green]" if ok else "[red]no[/red]")
    console.print(kt)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
