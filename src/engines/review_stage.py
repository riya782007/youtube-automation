"""Review Stage — produce ONE sample, present 7 artifacts, STOP.

Pipeline (per user spec):

  Generate script
   ↓
  Generate voice (intelligence-prepped)
   ↓
  Generate visuals (character + scene)
   ↓
  Editor agent creates timeline
   ↓
  Render sample
   ↓
  Generate thumbnail
   ↓
  Generate quality report
   ↓
  STOP — wait for explicit human approval

No bulk production. No scheduling. No upload. The CLI command `yt-os sample`
calls this module and dumps every artifact under output/<video_id>/ for review.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.logger import get_logger
from ..core.state import OUTPUT_DIR, ROOT, load_channel, write_json_atomic
from .audience_psychology import AudiencePsychology
from .character_engine import CharacterEngine, VALID_EXPRESSIONS
from .editor_agent import EditorAgent
from .music_intelligence import MusicIntelligence
from .quality_gate import QualityGate
from .script_engine import ScriptEngine
from .studio_renderer import StudioRenderer
from .voice_intelligence import VoiceIntelligence

log = get_logger(__name__)


@dataclass
class SampleBundle:
    """Everything produced by the sample pipeline. Saved under output/<id>/."""
    video_id: str
    channel_id: str
    sample_dir: Path
    mp4_path: Path | None
    thumbnail_path: Path | None
    script_path: Path
    timeline_path: Path
    voice_analysis_path: Path
    retention_prediction_path: Path
    editing_explanation_path: Path
    quality_report_path: Path
    summary_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {k: (str(v) if isinstance(v, Path) and v else v)
                for k, v in self.__dict__.items()}


class ReviewStage:
    def __init__(self) -> None:
        self.psych = AudiencePsychology()
        self.script = ScriptEngine()
        self.voice_intel = VoiceIntelligence()
        self.music_intel = MusicIntelligence()
        self.character = CharacterEngine()
        self.editor = EditorAgent()
        self.renderer = StudioRenderer()
        self.quality = QualityGate()

    def produce_sample(self, channel_id: str, *, topic: str | None = None) -> SampleBundle:
        ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        video_id = f"sample_{channel_id}_{ts}_{uuid.uuid4().hex[:6]}"
        sample_dir = OUTPUT_DIR / video_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        log.info("[review] producing sample %s", video_id)

        channel = load_channel(channel_id)

        # 1) Script ------------------------------------------------------
        topic = topic or (channel.get("topic_seeds") or ["lifestyle"])[0]
        psych = self.psych.profile(channel, topic)
        script = self.script.generate(channel, psych)
        all_sentences = [s for b in script.beats for s in b.sentences]

        # 2) Voice (intelligence-prepped, with timings) ------------------
        voice_plan = self.voice_intel.prepare(
            sentences=all_sentences,
            pace_wpm=channel.get("voice", {}).get("pace_words_per_minute"),
            emotion=script.emotion,
            sarvam_voice=channel.get("voice", {}).get("sarvam_voice", "anushka"),
        )

        # 3) Music (emotional + intensity curve) -------------------------
        music_plan = self.music_intel.select(
            channel_id=channel_id,
            emotion=script.emotion,
            total_seconds=voice_plan.total_seconds,
        )

        # 4) Editor Agent → timeline ------------------------------------
        timeline = self.editor.cut(
            video_id=video_id,
            channel_id=channel_id,
            beats=script.beats,
            voice_plan=voice_plan,
            music_plan=music_plan,
            emotion=script.emotion,
            psych=psych.to_dict(),
        )

        # 5) Character expressions PNGs (one per used expression) -------
        char_dir = sample_dir / "characters"
        char_dir.mkdir(exist_ok=True)
        used_expr = {tr["expression"] for tr in timeline.character_expression_track}
        used_expr |= {"curious", "shocked", "happy"}
        for expr in used_expr:
            self.character.render(
                channel_id=channel_id, expression=expr,
                video_id=video_id, persist_png=True,
            )

        # Pillow-level fallback PNGs in case cairosvg isn't available.
        from ..core.state import ARTIFACT_DIR
        char_artifacts = ARTIFACT_DIR / video_id / "characters"
        for expr in used_expr:
            png_path = char_artifacts / f"{self.character.for_channel(channel_id)['name'].lower()}_{expr}.png"
            if not png_path.exists() or png_path.stat().st_size < 1024:
                self._fallback_character_png(channel_id, expr, png_path)

        # 6) Render -----------------------------------------------------
        render = self.renderer.render(
            video_id=video_id, channel=channel,
            timeline=timeline.to_dict(),
            voice_plan=voice_plan.to_dict(),
            music_plan=music_plan.to_dict(),
            characters_dir=char_artifacts,
        )

        # 7) Save script / timeline / voice / etc. as inspectable files -
        script_path = sample_dir / "script.txt"
        script_path.write_text(self._format_script(script, voice_plan), encoding="utf-8")

        timeline_path = sample_dir / "editing_timeline.json"
        write_json_atomic(timeline_path, timeline.to_dict())

        voice_analysis = self._voice_analysis(voice_plan, channel)
        va_path = sample_dir / "voice_analysis.json"
        write_json_atomic(va_path, voice_analysis)

        retention = self._retention_prediction(timeline, voice_plan, music_plan, script)
        rp_path = sample_dir / "retention_prediction.json"
        write_json_atomic(rp_path, retention)

        explanation = self._editing_explanation(timeline, script, voice_plan, music_plan)
        ee_path = sample_dir / "editing_explanation.md"
        ee_path.write_text(explanation, encoding="utf-8")

        # 8) Quality scorecard (LLM-backed; falls back to heuristic mock) -
        plan_summary = {
            "channel_id": channel_id,
            "title": script.title,
            "duration_sec": voice_plan.total_seconds,
            "emotion": script.emotion,
            "hook_sentence": script.beats[0].sentences[0] if script.beats and script.beats[0].sentences else "",
            "loop_sentence": script.beats[-1].sentences[-1] if script.beats and script.beats[-1].sentences else "",
            "asset_mix": {"character": 0.45, "kinetic_text": 0.30,
                          "overlay": 0.15, "meme": 0.10},
            "avg_reset_interval": round(
                voice_plan.total_seconds / max(1, len(timeline.events)), 2),
            "sentence_count": len(all_sentences),
            "comment_trigger": psych.comment_trigger,
            "pinned_comment": script.pinned_comment,
            "forbidden_hits": [],
        }
        quality = self.quality.score(plan_summary)
        qr_path = sample_dir / "quality_report.json"
        write_json_atomic(qr_path, quality.to_dict())

        # 9) Copy mp4 + thumbnail next to the rest of the artifacts ----
        mp4_path = None
        if render.video_path and render.video_path.exists():
            mp4_path = sample_dir / f"{video_id}.mp4"
            shutil.copyfile(render.video_path, mp4_path)
        thumb_path = None
        if render.thumbnail_path and render.thumbnail_path.exists():
            thumb_path = sample_dir / "thumbnail.jpg"
            shutil.copyfile(render.thumbnail_path, thumb_path)

        # 10) The summary document -------------------------------------
        summary = self._summary(
            video_id=video_id, channel=channel, script=script,
            voice_plan=voice_plan, music_plan=music_plan,
            timeline=timeline, retention=retention,
            quality=quality.to_dict(),
            mp4_path=mp4_path, thumb_path=thumb_path,
            artifact_paths={
                "script": str(script_path),
                "timeline": str(timeline_path),
                "voice_analysis": str(va_path),
                "retention_prediction": str(rp_path),
                "editing_explanation": str(ee_path),
                "quality_report": str(qr_path),
            },
        )
        summary_path = sample_dir / "SAMPLE_REVIEW.md"
        summary_path.write_text(summary, encoding="utf-8")

        log.info("[review] DONE %s → %s", video_id, sample_dir)
        return SampleBundle(
            video_id=video_id, channel_id=channel_id, sample_dir=sample_dir,
            mp4_path=mp4_path, thumbnail_path=thumb_path,
            script_path=script_path, timeline_path=timeline_path,
            voice_analysis_path=va_path,
            retention_prediction_path=rp_path,
            editing_explanation_path=ee_path,
            quality_report_path=qr_path,
            summary_path=summary_path,
        )

    # ---- Reports -------------------------------------------------------

    @staticmethod
    def _format_script(script, voice_plan) -> str:
        lines = [f"# {script.title}", "", f"_Pinned comment_: {script.pinned_comment}", ""]
        for b in script.beats:
            lines.append(f"## {b.name}  ({b.start_sec}s → {b.end_sec}s)")
            for s in b.sentences:
                lines.append(f"- {s}")
            lines.append("")
        lines.append("---")
        lines.append(f"Pace: {voice_plan.pace_wpm} WPM | "
                     f"Total: {voice_plan.total_seconds:.1f}s | "
                     f"Emphasis words: {voice_plan.emphasis_words}")
        return "\n".join(lines)

    @staticmethod
    def _voice_analysis(voice_plan, channel) -> dict[str, Any]:
        v = channel.get("voice", {})
        return {
            "language": v.get("language", "hinglish"),
            "tone": v.get("tone", []),
            "pace_wpm": voice_plan.pace_wpm,
            "total_seconds": voice_plan.total_seconds,
            "sentences": len(voice_plan.sentences),
            "words": sum(1 for w in voice_plan.word_timings if not w.is_pause),
            "pauses_inserted": voice_plan.pauses_inserted,
            "breaths_inserted": voice_plan.breaths_inserted,
            "emphasis_words": voice_plan.emphasis_words,
            "tts_provider_target": v.get("tts_provider", "sarvam"),
            "natural_text_preview": voice_plan.natural_text[:240],
            "notes": (
                "Sample audio synthesized via offline pyttsx3 fallback; "
                "Sarvam (primary) or ElevenLabs (fallback) will produce "
                "the production-quality Hinglish voice when API keys are set."
            ),
        }

    @staticmethod
    def _retention_prediction(timeline, voice_plan, music_plan, script) -> dict[str, Any]:
        events = timeline.events
        T = timeline.total_seconds
        kinetic = sum(1 for e in events if e.action == "kinetic_text")
        cuts = sum(1 for e in events if e.action == "scene_cut")
        focus = sum(1 for e in events if e.reason == "focus")
        emotion_events = sum(1 for e in events if e.reason in {"emotion", "humor", "surprise"})
        gaps = []
        last = 0.0
        visual_kinds = {"scene_cut", "kinetic_text", "elastic_burst",
                        "speed_ramp", "camera_shake", "focus_circle",
                        "emoji_explosion", "reaction_meme"}
        for e in sorted(events, key=lambda x: x.time):
            if e.action in visual_kinds:
                gaps.append(round(e.time - last, 2))
                last = e.time
        max_gap = max(gaps) if gaps else T
        avg_gap = round(sum(gaps) / len(gaps), 2) if gaps else T

        # Heuristic prediction
        score = 70
        if kinetic >= 5: score += 6
        if cuts >= 3: score += 4
        if focus >= 2: score += 3
        if emotion_events >= 4: score += 5
        if avg_gap <= 2.0: score += 6
        if max_gap > 3.5: score -= 5
        score = max(0, min(95, score))
        pred_view_pct = score
        return {
            "predicted_avg_view_pct": pred_view_pct,
            "confidence": "low" if voice_plan.pace_wpm < 120 else "medium",
            "drivers": {
                "kinetic_text_count": kinetic,
                "scene_cut_count": cuts,
                "focus_events": focus,
                "emotion_events": emotion_events,
                "avg_visual_gap_sec": avg_gap,
                "max_visual_gap_sec": max_gap,
            },
            "risk_factors": (
                ["max_visual_gap_too_long"] if max_gap > 3.5 else []
            ) + (
                ["too_few_kinetic_reveals"] if kinetic < 3 else []
            ),
            "loop_strength": (
                "strong" if "loop_sting" in {e.action for e in events} else "weak"
            ),
            "music_intensity_match": "aligned" if music_plan.intensity_curve else "missing",
            "expected_repeat_view_rate_pct": round(min(35, 12 + kinetic * 1.2 + cuts * 0.6), 1),
        }

    @staticmethod
    def _editing_explanation(timeline, script, voice_plan, music_plan) -> str:
        # Group events by reason
        by_reason: dict[str, list] = {}
        for e in timeline.events:
            by_reason.setdefault(e.reason, []).append(e)

        out = [
            f"# Editing explanation — {script.title}",
            "",
            f"**Total length:** {timeline.total_seconds:.1f}s",
            f"**Events:** {len(timeline.events)}",
            f"**Rationale:** {timeline.rationale}",
            "",
            "## Events grouped by reason",
            "",
        ]
        for reason in ("retention", "emotion", "humor", "surprise", "focus", "loop"):
            arr = by_reason.get(reason, [])
            if not arr:
                continue
            out.append(f"### {reason}  ({len(arr)} events)")
            for e in sorted(arr, key=lambda x: x.time)[:30]:
                out.append(f"- `{e.time:5.2f}s` **{e.action}**  "
                           f"int={e.intensity:.2f}  dur={e.duration_sec:.2f}s")
            out.append("")
        out.append("## Character expression track")
        out.append("")
        for tr in timeline.character_expression_track:
            out.append(f"- `{tr['from_sec']:5.2f}–{tr['to_sec']:5.2f}s`  "
                       f"**{tr['beat']}**  →  expression `{tr['expression']}`")
        out.append("")
        out.append("## Music intensity curve")
        out.append("")
        for kf in music_plan.intensity_curve:
            out.append(f"- `{kf.t_sec:5.2f}s`  intensity={kf.intensity:.2f}  ({kf.label})")
        return "\n".join(out)

    @staticmethod
    def _summary(*, video_id, channel, script, voice_plan, music_plan,
                 timeline, retention, quality, mp4_path, thumb_path,
                 artifact_paths) -> str:
        ch_name = channel.get("name", "")
        return f"""# Sample for review — {ch_name}

**Video id:** `{video_id}`
**Channel:** {ch_name}
**Topic:** {script.psych.get('topic')}
**Emotion:** {script.emotion}
**Total length:** {timeline.total_seconds:.1f}s
**Quality avg:** {quality.get('average', '?')}  passes={quality.get('passes')}

---

## 1. MP4 sample
- {mp4_path or '_(render failed; see logs)_'}

## 2. Thumbnail
- {thumb_path or '_(thumbnail failed)_'}

## 3. Script
- {artifact_paths['script']}

## 4. Timeline
- {artifact_paths['timeline']}

## 5. Voice analysis
- {artifact_paths['voice_analysis']}

## 6. Retention prediction
- {artifact_paths['retention_prediction']}
- predicted avg view pct: **{retention['predicted_avg_view_pct']}%**
- expected repeat view rate: **{retention['expected_repeat_view_rate_pct']}%**

## 7. Editing explanation
- {artifact_paths['editing_explanation']}

## Quality scorecard
| axis | score |
|------|-------|
| hook | {quality.get('hook')} |
| voice | {quality.get('voice')} |
| visual_density | {quality.get('visual_density')} |
| emotion | {quality.get('emotion')} |
| editing | {quality.get('editing')} |
| retention_prediction | {quality.get('retention_prediction')} |
| **average** | **{quality.get('average')}** |

> {quality.get('notes', '')}

---

**STOP.** Production halts here pending your approval.
"""

    # ---- Pillow fallback character (when cairosvg is missing) ----------

    def _fallback_character_png(self, channel_id: str, expression: str,
                                out_path: Path) -> None:
        """Pillow-only character render so we never block on cairosvg."""
        from PIL import Image, ImageDraw
        dna = self.character.for_channel(channel_id)
        from .character_engine import EXPRESSIONS
        p = EXPRESSIONS.get(expression, EXPRESSIONS["curious"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        W, H = 720, 1080
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img, "RGBA")

        cx, cy = W // 2, int(H * 0.42)
        head_r = int(W * 0.30)

        # outfit
        out_color = _hx(dna["outfit"])
        draw.polygon([
            (cx - int(head_r * 1.7), cy + int(head_r * 1.25)),
            (cx + int(head_r * 1.7), cy + int(head_r * 1.25)),
            (cx + int(head_r * 2.0), H), (cx - int(head_r * 2.0), H),
        ], fill=out_color, outline=_hx(dna["outline"]), width=4)

        # hair back
        draw.ellipse((cx - int(head_r * 1.30), cy - int(head_r * 1.10),
                      cx + int(head_r * 1.30), cy + int(head_r * 1.20)),
                     fill=_hx(dna["hair"]))

        # face
        draw.ellipse((cx - head_r, cy - int(head_r * 1.08),
                      cx + head_r, cy + int(head_r * 1.08)),
                     fill=_hx(dna["skin"]),
                     outline=_hx(dna["outline"]), width=4)

        # hair fringe
        draw.polygon([
            (cx - head_r, cy - int(head_r * 0.10)),
            (cx - int(head_r * 0.30), cy - int(head_r * 0.95)),
            (cx + int(head_r * 0.40), cy - int(head_r * 0.85)),
            (cx + head_r, cy - int(head_r * 0.10)),
            (cx, cy - int(head_r * 0.45)),
        ], fill=_hx(dna["hair_hl"]))

        # blush
        ba = p["blush"]
        if ba > 0.05:
            for sx in (-1, 1):
                bx = cx + sx * int(head_r * 0.55)
                by = cy + int(head_r * 0.20)
                bw = int(head_r * 0.20); bh = int(head_r * 0.10)
                draw.ellipse((bx - bw, by - bh, bx + bw, by + bh),
                             fill=_hx_rgba(dna["lip"], int(160 * ba)))

        # eyes
        eye_dx = int(head_r * 0.36)
        eye_y = cy - int(head_r * 0.10)
        eye_w = int(head_r * 0.20)
        eye_h = max(3, int(head_r * 0.22 * p["eye_open"]))
        for sx in (-1, 1):
            ex = cx + sx * eye_dx
            draw.ellipse((ex - eye_w, eye_y - eye_h, ex + eye_w, eye_y + eye_h),
                         fill=(255, 255, 255), outline=_hx(dna["outline"]), width=3)
            ir = int(eye_w * 0.55)
            draw.ellipse((ex - ir, eye_y - ir, ex + ir, eye_y + ir),
                         fill=_hx(dna["eye"]))
            pr = int(eye_w * 0.25)
            draw.ellipse((ex - pr, eye_y - pr, ex + pr, eye_y + pr), fill=(0, 0, 0))
            # sparkle
            if p["sparkle"] > 0.2:
                sr = max(2, int(eye_w * 0.18))
                draw.ellipse((ex - sr + 8, eye_y - sr - 8,
                              ex + sr + 8, eye_y + sr - 8), fill=(255, 255, 255))

        # brows
        bw = int(head_r * 0.22); bt = max(3, int(head_r * 0.05))
        bly = eye_y - int(head_r * 0.30) - int(head_r * 0.10 * p["brow_lift"])
        for sx in (-1, 1):
            ex = cx + sx * eye_dx
            draw.line([(ex - bw, bly), (ex, bly - int(bt * 1.3)), (ex + bw, bly)],
                      fill=_hx(dna["hair"]), width=bt, joint="curve")

        # mouth
        mx = cx; my = cy + int(head_r * 0.45)
        mw = int(head_r * 0.34)
        m_open = p["mouth_open"]
        curve = p["mouth_curve"]
        if m_open > 0.05:
            mh = int(head_r * 0.10 * (1 + 1.5 * m_open))
            draw.ellipse((mx - mw, my - mh, mx + mw, my + mh),
                         fill=_hx("#5A1F2A"), outline=_hx(dna["outline"]), width=3)
        else:
            # Smile/frown via two-point curve approximated as polyline
            cy_off = int(head_r * 0.10 * (-curve * 1.4))
            pts = []
            for i in range(13):
                fx = i / 12
                x = mx - mw + int(2 * mw * fx)
                y = my + int(cy_off * math.sin(math.pi * fx))
                pts.append((x, y))
            draw.line(pts, fill=_hx("#7A2230"), width=6, joint="curve")

        # accent (earring)
        ar = int(head_r * 0.06)
        ax = cx + int(head_r * 1.05)
        ay = cy + int(head_r * 0.55)
        draw.ellipse((ax - ar, ay - ar, ax + ar, ay + ar), fill=_hx(dna["accent"]))

        img.save(out_path)


def _hx(s: str):
    s = (s or "#000000").lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def _hx_rgba(s: str, a: int):
    return _hx(s) + (a,)
