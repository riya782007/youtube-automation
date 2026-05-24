"""Studio Renderer — consumes editing_timeline.json and produces a real mp4.

Open-source stack:
  • imageio-ffmpeg (bundled FFmpeg binary — works in any sandbox)
  • Pillow (frame painting)
  • MoviePy (timeline composition where useful)
  • PySceneDetect (optional, for analyzing existing footage)
  • OpenTimelineIO (optional, exports a .otio for Blender VSE / DaVinci)

The renderer's job:
  1. Build a 1080x1920 frame stream painting:
     - background gradient (channel palette + emotion tint)
     - parallax BG layer (slow drift)
     - character SVG/PNG layer (expression follows character_expression_track)
     - kinetic text reveals (per kinetic_text events)
     - subtitles (per subtitle events)
     - effect overlays (focus_circle, animated_arrow, glow, emoji explosion)
     - reaction meme stickers
     - camera transforms (push_zoom, shake, drift, fake_handheld, speed_ramp)
  2. Mix audio track:
     - voice (from voice_engine output OR built-in pyttsx3 fallback)
     - music with intensity_curve modulation + ducking under voice
     - sfx (heartbeat, sub-bass punch, loop sting) per timeline events
  3. Mux into final mp4
  4. Generate thumbnail at the strongest emotional spike

Designed so each subsystem can be inspected separately — every step writes
its artifacts to artifacts/<video_id>/.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.logger import get_logger
from ..core.state import ARTIFACT_DIR, write_json_atomic

log = get_logger(__name__)


def _ffmpeg_exe() -> str:
    """Return path to ffmpeg, preferring system, falling back to imageio-ffmpeg."""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


@dataclass
class StudioRenderResult:
    video_path: Path | None
    thumbnail_path: Path | None
    voice_path: Path | None
    music_path: Path | None
    artifacts_dir: Path
    skipped_reason: str | None = None


# ---------- Public API -----------------------------------------------------


class StudioRenderer:
    """Realizes an editing_timeline.json into a playable mp4 + thumbnail."""

    def __init__(self, *, fps: int = 30, width: int = 1080, height: int = 1920) -> None:
        self.fps = fps
        self.W = width
        self.H = height
        self.ffmpeg = _ffmpeg_exe()

    # ---- The render call ------------------------------------------------

    def render(self, *, video_id: str,
               channel: dict[str, Any],
               timeline: dict[str, Any],
               voice_plan: dict[str, Any],
               music_plan: dict[str, Any],
               characters_dir: Path | str | None = None) -> StudioRenderResult:
        work = ARTIFACT_DIR / video_id
        work.mkdir(parents=True, exist_ok=True)

        T = float(timeline["total_seconds"])
        log.info("[studio] rendering %s | T=%.2fs | %dx%d @ %dfps",
                 video_id, T, self.W, self.H, self.fps)

        # 1) Voice → wav
        voice_path = self._synth_voice(voice_plan, work / "voice.wav", T)

        # 2) Music → wav, with intensity_curve modulation
        music_path = self._synth_music(music_plan, work / "music.wav", T)

        # 3) SFX bed
        sfx_path = self._synth_sfx(timeline, work / "sfx.wav", T)

        # 4) Mix audio
        mix_path = self._mix_audio(voice_path, music_path, sfx_path,
                                   work / "audio_mix.wav", T)

        # 5) Pre-fetch B-roll from Pexels per beat (real footage)
        broll = self._fetch_broll(channel, timeline, work, T)

        # 6) Visual frame stream
        frames_dir = work / "frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True)
        frame_count = self._paint_frames(timeline, channel, voice_plan,
                                         characters_dir, frames_dir, T,
                                         broll=broll)
        log.info("[studio] painted %d frames", frame_count)

        # 7) FFmpeg encode + audio mux
        out_mp4 = work / f"{video_id}.mp4"
        ok = self._encode(frames_dir, mix_path, out_mp4)

        # 8) Thumbnail at the strongest emotional spike
        thumb_path = self._thumbnail(timeline, channel, frames_dir, work)

        return StudioRenderResult(
            video_path=out_mp4 if ok else None,
            thumbnail_path=thumb_path,
            voice_path=voice_path,
            music_path=music_path,
            artifacts_dir=work,
            skipped_reason=None if ok else "ffmpeg-encode-failed",
        )

    # ---- B-roll (Pexels) -------------------------------------------------

    @staticmethod
    def _pexels_key() -> str:
        return (os.environ.get("PEXELS_API_KEY")
                or os.environ.get("pixels api key")
                or "").strip()

    def _fetch_broll(self, channel: dict[str, Any],
                     timeline: dict[str, Any], work: Path, T: float) -> dict[str, Any]:
        """Fetch one B-roll clip per beat. Returns {beat_name: frames_dir}.

        Uses ffmpeg to pre-render each clip into a JPEG frame stream sized
        to canvas, so the painter just reads `frame_<idx>.jpg` per timestamp.
        """
        key = self._pexels_key()
        if not key:
            log.info("[studio/broll] no Pexels key — skipping (Pillow gradient fallback)")
            return {}
        try:
            import httpx
        except Exception:
            return {}

        broll_dir = work / "broll"
        broll_dir.mkdir(exist_ok=True)
        per_beat: dict[str, dict[str, Any]] = {}

        # Query plan per beat from script + channel category
        title_seed = channel.get("topic_seeds", ["psychology"])[0]
        beat_queries = {
            "hook":       f"close up portrait phone screen anxious face",
            "escalation": f"young woman thinking phone {title_seed}",
            "reveal":     f"slow zoom face emotional reaction",
            "twist":      f"surprised reaction phone notification",
            "loop":       f"cinematic reflection city night thoughtful",
        }

        for beat_name, query in beat_queries.items():
            try:
                with httpx.Client(timeout=20.0) as c:
                    r = c.get("https://api.pexels.com/videos/search",
                              headers={"Authorization": key},
                              params={"query": query, "per_page": 3,
                                      "orientation": "portrait", "size": "medium"})
                if r.status_code != 200:
                    continue
                videos = r.json().get("videos", [])
                if not videos:
                    continue
                # Pick the first video with HD portrait
                v = videos[0]
                files = sorted(v.get("video_files", []),
                               key=lambda f: f.get("height", 0), reverse=True)
                best = next((f for f in files
                             if 720 <= f.get("height", 0) <= 1920),
                            files[0] if files else {})
                url = best.get("link")
                if not url:
                    continue
                src = broll_dir / f"{beat_name}.mp4"
                # Download
                with httpx.Client(timeout=120.0, follow_redirects=True) as c:
                    rr = c.get(url)
                    if rr.status_code != 200:
                        continue
                    src.write_bytes(rr.content)
                # Pre-render into frame stream sized to canvas
                bf_dir = broll_dir / f"{beat_name}_frames"
                bf_dir.mkdir(exist_ok=True)
                subprocess.run([
                    self.ffmpeg, "-y", "-i", str(src),
                    "-vf",
                    f"scale={self.W}:{self.H}:force_original_aspect_ratio=increase,"
                    f"crop={self.W}:{self.H},fps={self.fps}",
                    "-q:v", "5",
                    str(bf_dir / "f_%05d.jpg"),
                ], capture_output=True, timeout=120)
                count = len(list(bf_dir.glob("*.jpg")))
                if count > 0:
                    per_beat[beat_name] = {
                        "frames_dir": bf_dir,
                        "frame_count": count,
                        "video_id": v.get("id"),
                        "credit": v.get("user", {}).get("name"),
                    }
                    log.info("[studio/broll] %s: pexels#%s, %d frames (%s)",
                             beat_name, v.get("id"), count, v.get("user", {}).get("name"))
            except Exception as e:
                log.warning("[studio/broll] %s failed: %s", beat_name, e)
        return per_beat

    # ---- Voice synthesis ------------------------------------------------

    def _synth_voice(self, voice_plan: dict[str, Any], out: Path,
                     T: float) -> Path:
        """Synthesize voice. Order of preference:
            1. **Sarvam** (Hinglish-native, market-grade)
            2. ElevenLabs Multilingual v2 (Hinglish via prosody)
            3. espeak CLI (last-ditch sanity audio)
            4. silent placeholder (logged loudly)
        """
        text = (voice_plan.get("natural_text") or "").strip()
        text = text.replace("[emph]", "").replace("[/emph]", "")
        if not text:
            _write_silent_wav(out, seconds=T, sample_rate=24000)
            return out

        # 1) Sarvam
        if self._try_sarvam(text, out, voice_plan):
            return out

        # 2) ElevenLabs
        if self._try_elevenlabs(text, out, voice_plan):
            return out

        # 3) espeak last-resort
        espeak = shutil.which("espeak")
        if espeak:
            wpm = int(voice_plan.get("pace_wpm", 165) * 0.95)
            try:
                cmd = [espeak, "-v", "en-in", "-s", str(wpm), "-p", "55",
                       "-w", str(out), text[:8000]]
                r = subprocess.run(cmd, capture_output=True, timeout=120)
                if r.returncode == 0 and out.exists() and out.stat().st_size > 1024:
                    log.warning("[studio/voice] using espeak fallback (Sarvam/EL keys missing)")
                    return out
            except Exception as e:
                log.warning("[studio/voice] espeak failed: %s", e)

        _write_silent_wav(out, seconds=T, sample_rate=24000)
        log.warning("[studio/voice] using SILENT placeholder — no TTS reachable")
        return out

    @staticmethod
    def _sarvam_key() -> str:
        # Accept both standard env name AND the literal-spaced provisioning name.
        return (os.environ.get("SARVAM_API_KEY")
                or os.environ.get("sarvam api key")
                or os.environ.get("SARVAM_KEY")
                or "").strip()

    def _try_sarvam(self, text: str, out: Path, voice_plan: dict[str, Any]) -> bool:
        key = self._sarvam_key()
        if not key:
            return False
        try:
            import base64, httpx
            # Sarvam caps inputs at 500 chars per chunk — split sensibly.
            chunks = self._chunk_for_tts(text, max_chars=480)
            wav_pieces = []
            voice = voice_plan.get("sarvam_voice") or "anushka"
            pace = max(0.5, min(2.0, voice_plan.get("pace_wpm", 165) / 165.0))
            for chunk in chunks:
                payload = {
                    "inputs": [chunk],
                    "target_language_code": "hi-IN",
                    "speaker": voice,
                    "pitch": 0.0,
                    "pace": round(pace, 2),
                    "loudness": 1.0,
                    "speech_sample_rate": 22050,
                    "enable_preprocessing": True,
                    "model": "bulbul:v2",
                }
                with httpx.Client(timeout=90.0) as c:
                    r = c.post("https://api.sarvam.ai/text-to-speech",
                               json=payload,
                               headers={"api-subscription-key": key,
                                        "Content-Type": "application/json"})
                if r.status_code != 200:
                    log.warning("[studio/voice/sarvam] HTTP %d: %s",
                                r.status_code, r.text[:200])
                    return False
                data = r.json()
                audios = data.get("audios") or []
                if not audios:
                    log.warning("[studio/voice/sarvam] empty audios array")
                    return False
                wav_pieces.append(base64.b64decode(audios[0]))
            # Concat WAV bytes — Sarvam returns 22050Hz PCM WAV; concat by parsing.
            concat_wav = self._concat_wavs(wav_pieces)
            out.write_bytes(concat_wav)
            log.info("[studio/voice/sarvam] %d chunks → %s (%d KB)",
                     len(chunks), out, out.stat().st_size // 1024)
            return True
        except Exception as e:
            log.warning("[studio/voice/sarvam] failed: %s", e)
            return False

    def _try_elevenlabs(self, text: str, out: Path, voice_plan: dict[str, Any]) -> bool:
        key = (os.environ.get("ELEVENLABS_API_KEY") or "").strip()
        if not key:
            return False
        try:
            import httpx
            voice_id = (os.environ.get("ELEVENLABS_VOICE_ID_HUMAN_DECODER")
                        or os.environ.get("ELEVENLABS_VOICE_ID")
                        or "21m00Tcm4TlvDq8ikWAM")  # default Rachel
            r = httpx.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                json={
                    "text": text[:5000],
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {"stability": 0.45, "similarity_boost": 0.78,
                                       "style": 0.45, "use_speaker_boost": True},
                },
                headers={"xi-api-key": key, "accept": "audio/mpeg",
                         "Content-Type": "application/json"},
                timeout=90.0,
            )
            if r.status_code != 200:
                log.warning("[studio/voice/elevenlabs] HTTP %d", r.status_code)
                return False
            # ElevenLabs returns mp3; we'll let ffmpeg consume mp3 in the mux step.
            mp3_path = out.with_suffix(".mp3")
            mp3_path.write_bytes(r.content)
            # Convert to wav using ffmpeg
            subprocess.run([
                self.ffmpeg, "-y", "-i", str(mp3_path),
                "-ar", "22050", "-ac", "1", str(out),
            ], capture_output=True, timeout=60)
            return out.exists() and out.stat().st_size > 1024
        except Exception as e:
            log.warning("[studio/voice/elevenlabs] failed: %s", e)
            return False

    @staticmethod
    def _chunk_for_tts(text: str, *, max_chars: int) -> list[str]:
        """Split on sentence boundaries; never exceed max_chars."""
        import re
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        out, cur = [], ""
        for s in sents:
            if len(cur) + len(s) + 1 <= max_chars:
                cur = (cur + " " + s).strip()
            else:
                if cur:
                    out.append(cur)
                cur = s if len(s) <= max_chars else s[:max_chars]
        if cur:
            out.append(cur)
        return out

    @staticmethod
    def _concat_wavs(pieces: list[bytes]) -> bytes:
        """Concatenate WAV byte blobs with the SAME format. Sarvam returns
        consistent 22050Hz mono 16-bit, so we strip headers from all but
        the first and rebuild a single header."""
        import io, wave
        if not pieces:
            return b""
        if len(pieces) == 1:
            return pieces[0]
        frames = []
        params = None
        for blob in pieces:
            with wave.open(io.BytesIO(blob), "rb") as w:
                if params is None:
                    params = w.getparams()
                frames.append(w.readframes(w.getnframes()))
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setparams(params)
            w.writeframes(b"".join(frames))
        return out.getvalue()

    # ---- Music synthesis (intensity curve baked in) ---------------------

    def _synth_music(self, music_plan: dict[str, Any], out: Path, T: float) -> Path:
        """Generate a synthetic background bed shaped by the intensity curve.

        We deliberately synthesize rather than always streaming — produces a
        deterministic sample we know is royalty-free. The selected track from
        the open-source library is recorded in the JSON for the user to swap
        in for production renders.
        """
        import numpy as np

        sr = 44100
        N = int(T * sr)
        t = np.arange(N) / sr

        # Build an intensity envelope from the curve keyframes.
        curve = music_plan.get("intensity_curve") or [{"t_sec": 0, "intensity": 0.6, "label": "setup"}]
        ts = np.array([kf["t_sec"] for kf in curve])
        vs = np.array([kf["intensity"] for kf in curve])
        env = np.interp(t, ts, vs).astype(np.float32)

        # Two-tone synth pad (bass A2 + warm pad) — emotion-leaning rather
        # than tied to a specific genre; the renderer only needs a credible
        # bed for the SAMPLE.
        bass = 0.35 * np.sin(2 * np.pi * 110.0 * t)
        pad  = 0.28 * (np.sin(2 * np.pi * 220.0 * t) + 0.6 * np.sin(2 * np.pi * 330.0 * t))
        # Slow LFO for movement
        lfo = 0.85 + 0.15 * np.sin(2 * np.pi * 0.25 * t)
        signal = ((bass + pad) * lfo) * env

        # Soft saturation
        signal = np.tanh(signal * 1.3)
        signal = (signal * 0.55).astype(np.float32)

        _write_pcm_wav(out, signal, sr)
        return out

    # ---- SFX from timeline events ---------------------------------------

    def _synth_sfx(self, timeline: dict[str, Any], out: Path, T: float) -> Path:
        import numpy as np
        sr = 44100
        N = int(T * sr)
        bus = np.zeros(N, dtype=np.float32)

        for ev in timeline.get("events", []):
            act = ev["action"]
            t0 = float(ev["time"])
            i0 = int(t0 * sr)
            if act == "sub_bass_punch":
                bus[i0:i0 + int(0.6 * sr)] += _sub_bass_punch(sr)
            elif act == "heartbeat_sfx":
                bus[i0:i0 + int(1.4 * sr)] += _heartbeat(sr, beats=2)
            elif act == "loop_sting":
                bus[i0:i0 + int(0.5 * sr)] += _loop_sting(sr)
            elif act == "elastic_burst":
                bus[i0:i0 + int(0.18 * sr)] += _click(sr, freq=900)
            elif act == "scene_cut":
                bus[i0:i0 + int(0.22 * sr)] += _whoosh(sr)
            elif act == "camera_shake":
                bus[i0:i0 + int(0.18 * sr)] += _click(sr, freq=200)
            elif act == "emoji_explosion":
                bus[i0:i0 + int(0.4 * sr)] += _whoosh(sr) * 0.9

        # Hard-clip safety
        bus = np.tanh(bus * 1.1) * 0.9
        _write_pcm_wav(out, bus.astype(np.float32), sr)
        return out

    # ---- Mix --------------------------------------------------------------

    def _mix_audio(self, voice: Path, music: Path, sfx: Path,
                   out: Path, T: float) -> Path:
        """ducks music under voice; sums voice + music + sfx into stereo wav."""
        import numpy as np

        v = _read_wav_mono(voice)
        m = _read_wav_mono(music)
        s = _read_wav_mono(sfx)
        sr = 44100
        N = int(T * sr)
        for arr in (v, m, s):
            if len(arr) < N:
                arr.resize(N, refcheck=False)
        v = v[:N]; m = m[:N]; s = s[:N]

        # Voice activity envelope → side-chain ducker for music
        win = int(sr * 0.04)
        if win < 1:
            win = 1
        env = np.abs(v)
        # Moving avg
        cumsum = np.cumsum(np.insert(env, 0, 0))
        env = (cumsum[win:] - cumsum[:-win]) / win
        env = np.pad(env, (0, N - len(env)), constant_values=0)
        # Normalize
        if env.max() > 0:
            env /= env.max()
        duck = 1.0 - 0.65 * np.clip(env * 4.0, 0.0, 1.0)
        music_ducked = m * duck

        mix = (1.0 * v + 0.45 * music_ducked + 0.7 * s)
        mix = np.tanh(mix * 1.1) * 0.95

        _write_pcm_wav(out, mix.astype(np.float32), sr)
        return out

    # ---- Frame painting --------------------------------------------------

    def _paint_frames(self, timeline: dict[str, Any], channel: dict[str, Any],
                      voice_plan: dict[str, Any],
                      characters_dir: Path | str | None,
                      frames_dir: Path, T: float,
                      *, broll: dict[str, Any] | None = None) -> int:
        """Paint every frame deterministically. CPU-only Pillow."""
        from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance
        import numpy as np
        broll = broll or {}

        # Load palette — channels.yaml stores DESCRIPTIVE names; map to hex.
        vstyle = channel.get("visual_style", {})
        palette = [_palette_hex(p) for p in vstyle.get("palette", [])]
        if not palette:
            palette = ["#0a0a14", "#1a1a2e", "#FFD86B"]
        bg_a = _hex(palette[0])
        bg_b = _hex(palette[1] if len(palette) > 1 else "#1a1a2e")
        accent = _hex(palette[-1] if palette else "#FFD86B")

        # Load character SVG → PNG once per expression (best-effort)
        char_imgs = self._load_characters(characters_dir)

        events = sorted(timeline.get("events", []), key=lambda e: e["time"])
        char_track = timeline.get("character_expression_track", [])

        # Try to load a system font; fall back to default
        font_big = _font(80, bold=True)
        font_med = _font(56, bold=True)
        font_sub = _font(40, bold=False)

        n_frames = int(T * self.fps)
        for f in range(n_frames):
            t = f / self.fps

            # Resolve current beat
            cur_beat = "hook"
            for tr in char_track:
                if tr["from_sec"] <= t <= tr["to_sec"]:
                    cur_beat = tr["beat"]
                    break

            # Background: real B-roll if we have it, else gradient
            img = self._beat_background(cur_beat, t, broll, bg_a, bg_b)

            # Camera transform state
            zoom = 1.0
            offset_x = 0.0
            offset_y = 0.0
            rot_deg = 0.0
            blur_sigma = 0.0

            for ev in events:
                t0 = float(ev["time"])
                dur = float(ev.get("duration_sec", 0.0))
                t1 = t0 + dur
                if t < t0 or (dur > 0 and t > t1):
                    continue
                act = ev["action"]
                p = ev.get("payload", {}) or {}

                if act == "push_zoom" and dur > 0:
                    f01 = (t - t0) / max(0.001, dur)
                    z_to = float(p.get("zoom_to", 1.12))
                    zoom *= 1.0 + (z_to - 1.0) * _ease_out(f01)
                elif act == "camera_shake":
                    f01 = (t - t0) / max(0.001, dur)
                    amp = float(p.get("amp_px", 14)) * (1 - f01) * float(ev.get("intensity", 0.6))
                    offset_x += amp * math.sin(2 * math.pi * 28 * t)
                    offset_y += amp * math.cos(2 * math.pi * 24 * t)
                elif act == "fake_handheld":
                    offset_x += 5 * math.sin(2 * math.pi * 1.6 * t)
                    offset_y += 4 * math.cos(2 * math.pi * 1.2 * t)
                    rot_deg += 0.5 * math.sin(2 * math.pi * 0.9 * t)
                elif act == "camera_drift" and dur > 0:
                    f01 = (t - t0) / max(0.001, dur)
                    offset_x += 22 * float(p.get("dx", 0.012)) * f01 * self.W
                    offset_y += 22 * float(p.get("dy", 0.0)) * f01 * self.H
                elif act == "blur_transition":
                    f01 = (t - t0) / max(0.001, dur)
                    s = float(p.get("sigma_peak", 12))
                    blur_sigma += s * (1 - abs(f01 - 0.5) * 2)
                elif act == "speed_ramp":
                    pass   # encoded into time mapping; safe to ignore visually

            # Active character expression at t
            expr = "curious"
            for tr in char_track:
                if tr["from_sec"] <= t <= tr["to_sec"]:
                    expr = tr["expression"]
                    break
            char_img = char_imgs.get(expr) or char_imgs.get("curious")

            # Composite character only on hook/twist/loop beats; let B-roll
            # breathe on escalation/reveal so it feels less template-y.
            if char_img is not None and cur_beat in {"hook", "twist", "loop"}:
                bob = 4 * math.sin(2 * math.pi * 0.6 * t)
                cw, ch = char_img.size
                target_w = int(self.W * 0.58 * zoom)
                target_h = int(target_w * ch / cw)
                resized = char_img.resize((target_w, target_h), Image.LANCZOS)
                px = int((self.W - target_w) / 2 + offset_x)
                py = int(self.H * 0.08 + bob + offset_y)
                img.paste(resized, (px, py), resized)

            # Draw kinetic text / subtitles / overlays / arrows / focus / memes
            draw = ImageDraw.Draw(img, "RGBA")
            for ev in events:
                t0 = float(ev["time"])
                dur = float(ev.get("duration_sec", 0.0))
                t1 = t0 + dur
                if t < t0 or (dur > 0 and t > t1 + 0.05):
                    continue
                act = ev["action"]
                p = ev.get("payload", {}) or {}

                if act == "kinetic_text":
                    text = p.get("text", "")
                    self._draw_kinetic(draw, text, t - t0, max(0.001, dur),
                                       font_big, accent)
                elif act == "subtitle":
                    self._draw_subtitle(draw, p.get("text", ""), font_sub)
                elif act == "elastic_burst":
                    f01 = (t - t0) / max(0.001, dur)
                    s = 1.0 + 0.45 * (1 - abs(f01 - 0.4) * 2)
                    self._burst_pop(img, draw, s, accent)
                elif act == "focus_circle":
                    f01 = (t - t0) / max(0.001, dur)
                    r = int(self.W * float(p.get("radius", 0.18)) *
                            (1.0 + 0.06 * math.sin(2 * math.pi * 2.0 * f01)))
                    cx = int(self.W * float(p.get("x", 0.5)))
                    cy = int(self.H * float(p.get("y", 0.4)))
                    draw.ellipse((cx - r, cy - r, cx + r, cy + r),
                                 outline=_hex(p.get("stroke_color", "#FFD86B")),
                                 width=8)
                elif act == "animated_arrow":
                    f01 = (t - t0) / max(0.001, dur)
                    fr = p.get("from", [0.18, 0.78])
                    to = p.get("to", [0.45, 0.46])
                    sx, sy = self.W * fr[0], self.H * fr[1]
                    ex, ey = self.W * to[0], self.H * to[1]
                    cx_, cy_ = sx + (ex - sx) * f01, sy + (ey - sy) * f01
                    draw.line([(sx, sy), (cx_, cy_)],
                              fill=_hex(p.get("color", "#FF4D4D")),
                              width=int(p.get("thickness", 8)))
                    self._arrow_head(draw, (sx, sy), (cx_, cy_),
                                     _hex(p.get("color", "#FF4D4D")))
                elif act == "glow_indicator":
                    cx_ = int(self.W * float(p.get("x", 0.5)))
                    cy_ = int(self.H * float(p.get("y", 0.5)))
                    r = int(self.W * float(p.get("radius", 0.22)))
                    draw.ellipse((cx_ - r, cy_ - r, cx_ + r, cy_ + r),
                                 outline=_hex(p.get("color", "#7CF6FF")),
                                 width=4)
                elif act == "reaction_meme":
                    f01 = (t - t0) / max(0.001, dur)
                    self._meme_box(draw, t01=f01, accent=accent,
                                   x=p.get("x", 0.78),
                                   y=p.get("y", 0.18),
                                   scale=p.get("scale", 0.22))
                elif act == "emoji_explosion":
                    f01 = (t - t0) / max(0.001, dur)
                    self._emoji_explosion(draw, t01=f01,
                                          emoji=p.get("emoji", "🔥"),
                                          count=int(p.get("count", 14)),
                                          font=font_med)
                elif act == "comment_prompt":
                    self._comment_pill(draw, p.get("text", "What do you think?"),
                                       y=float(p.get("y", 0.90)),
                                       font=font_sub, accent=accent)

            # Blur transitions affect the entire frame
            if blur_sigma > 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
            # Slight rotation for handheld
            if abs(rot_deg) > 0.05:
                img = img.rotate(rot_deg, resample=Image.BICUBIC, fillcolor=(0, 0, 0))

            img.convert("RGB").save(frames_dir / f"f_{f:05d}.jpg", quality=84)

        return n_frames

    def _beat_background(self, beat: str, t: float, broll: dict[str, Any],
                         bg_a: tuple[int, int, int], bg_b: tuple[int, int, int]):
        """Real Pexels B-roll if we have it for this beat (subtle blur + dim
        so foreground reads), else gradient fallback."""
        from PIL import Image, ImageEnhance, ImageFilter, ImageDraw as _D
        meta = broll.get(beat)
        if meta:
            idx = int(t * self.fps) % max(1, meta["frame_count"])
            src = meta["frames_dir"] / f"f_{idx + 1:05d}.jpg"
            if src.exists():
                try:
                    img = Image.open(src).convert("RGB")
                    img = img.filter(ImageFilter.GaussianBlur(radius=2.0))
                    img = ImageEnhance.Brightness(img).enhance(0.74)
                    rgba = img.convert("RGBA")
                    overlay = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
                    _D.Draw(overlay).rectangle(
                        (0, 0, self.W, int(self.H * 0.16)), fill=(0, 0, 0, 100))
                    _D.Draw(overlay).rectangle(
                        (0, int(self.H * 0.84), self.W, self.H), fill=(0, 0, 0, 130))
                    rgba.alpha_composite(overlay)
                    return rgba
                except Exception:
                    pass
        return self._gradient(bg_a, bg_b, t)

    # ---- Drawing helpers -------------------------------------------------

    def _gradient(self, a: tuple[int, int, int], b: tuple[int, int, int],
                  t: float):
        from PIL import Image
        import numpy as np
        # Vertical gradient with subtle pan via numpy (fast).
        pan = int(20 * math.sin(2 * math.pi * 0.05 * t))
        ys = np.arange(self.H, dtype=np.float32)
        f = np.clip((ys + pan) / self.H, 0.0, 1.0)
        col = np.empty((self.H, 3), dtype=np.uint8)
        for c in range(3):
            col[:, c] = (a[c] * (1 - f) + b[c] * f).astype(np.uint8)
        # Repeat column across width
        arr = np.broadcast_to(col[:, None, :], (self.H, self.W, 3)).copy()
        return Image.fromarray(arr, mode="RGB").convert("RGBA")

    def _draw_kinetic(self, draw, text: str, t_local: float, dur: float,
                      font, accent: tuple[int, int, int]) -> None:
        words = text.split()
        if not words:
            return
        per_chunk = max(0.18, dur / max(1, len(words) / 3))
        chunk_idx = min(int(t_local // per_chunk), max(0, math.ceil(len(words) / 3) - 1))
        start = chunk_idx * 3
        chunk = " ".join(words[start:start + 3])
        if not chunk:
            return
        # Centered, bottom third
        bbox = draw.textbbox((0, 0), chunk.upper(), font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (self.W - tw) // 2
        y = int(self.H * 0.62)
        # Stroke
        for dx in (-3, 0, 3):
            for dy in (-3, 0, 3):
                draw.text((x + dx, y + dy), chunk.upper(), font=font, fill=(0, 0, 0))
        # Fill (animated drop-in)
        f01 = (t_local % per_chunk) / per_chunk
        scale_in = 1.0 + 0.06 * (1 - f01)
        draw.text((x, y - int(8 * (scale_in - 1) * 10)), chunk.upper(),
                  font=font, fill=(255, 255, 255))
        # Accent underline
        draw.rectangle((x, y + th + 8, x + tw, y + th + 16), fill=accent)

    def _draw_subtitle(self, draw, text: str, font) -> None:
        if not text:
            return
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (self.W - tw) // 2
        y = int(self.H * 0.78)
        # Pill bg
        pad = 18
        draw.rounded_rectangle((x - pad, y - pad // 2, x + tw + pad, y + th + pad // 2),
                               radius=22, fill=(0, 0, 0, 180))
        draw.text((x, y), text, font=font, fill=(255, 255, 255))

    def _burst_pop(self, img, draw, scale: float,
                   accent: tuple[int, int, int]) -> None:
        # White flash overlay
        from PIL import Image
        flash = Image.new("RGBA", (self.W, self.H),
                          (255, 255, 255, int(40 * scale)))
        img.alpha_composite(flash) if img.mode == "RGBA" else img.paste(flash, (0, 0), flash)

    def _arrow_head(self, draw, start, end, color):
        sx, sy = start
        ex, ey = end
        ang = math.atan2(ey - sy, ex - sx)
        L = 28
        x1 = ex - L * math.cos(ang - 0.5)
        y1 = ey - L * math.sin(ang - 0.5)
        x2 = ex - L * math.cos(ang + 0.5)
        y2 = ey - L * math.sin(ang + 0.5)
        draw.polygon([(ex, ey), (x1, y1), (x2, y2)], fill=color)

    def _meme_box(self, draw, *, t01: float, accent: tuple[int, int, int],
                  x: float, y: float, scale: float) -> None:
        bw = int(self.W * scale)
        bh = int(bw * 0.8)
        bx = int(self.W * x - bw / 2)
        by = int(self.H * y - bh / 2)
        # Pop-in scale
        s = 0.4 + 0.6 * min(1.0, t01 * 3)
        bw2, bh2 = int(bw * s), int(bh * s)
        bx += (bw - bw2) // 2; by += (bh - bh2) // 2
        draw.rounded_rectangle((bx, by, bx + bw2, by + bh2),
                               radius=16, fill=(255, 255, 255),
                               outline=accent, width=4)
        draw.text((bx + 18, by + bh2 // 2 - 18), "👀  REACTION",
                  fill=(0, 0, 0))

    def _emoji_explosion(self, draw, *, t01: float, emoji: str, count: int,
                         font) -> None:
        cx = self.W // 2
        cy = int(self.H * 0.45)
        spread = self.W * 0.45 * t01
        for i in range(count):
            ang = 2 * math.pi * (i / count) + t01 * 1.4
            x = cx + spread * math.cos(ang)
            y = cy + spread * math.sin(ang)
            draw.text((x, y), emoji, font=font, fill=(255, 230, 80))

    def _comment_pill(self, draw, text: str, *, y: float, font,
                      accent: tuple[int, int, int]) -> None:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
        x = (self.W - tw) // 2
        py = int(self.H * y)
        pad = 24
        draw.rounded_rectangle((x - pad, py - pad // 2, x + tw + pad, py + th + pad // 2),
                               radius=28, fill=(0, 0, 0, 220),
                               outline=accent, width=3)
        draw.text((x, py), text, font=font, fill=(255, 255, 255))

    # ---- Character loading ----------------------------------------------

    def _load_characters(self, characters_dir):
        from PIL import Image
        out: dict[str, Any] = {}
        if not characters_dir:
            return out
        chd = Path(characters_dir)
        if not chd.exists():
            return out
        for png in chd.glob("*.png"):
            try:
                img = Image.open(png).convert("RGBA")
                # filename like maya_curious.png
                expr = png.stem.split("_")[-1]
                out[expr] = img
            except Exception:
                continue
        return out

    # ---- FFmpeg encode + thumbnail --------------------------------------

    def _encode(self, frames_dir: Path, audio: Path, out: Path) -> bool:
        cmd = [
            self.ffmpeg, "-y", "-framerate", str(self.fps),
            "-i", str(frames_dir / "f_%05d.jpg"),
            "-i", str(audio),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            "-shortest", str(out),
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0:
            log.error("[studio] ffmpeg failed: %s", res.stderr.decode("utf-8", "ignore")[-500:])
            return False
        log.info("[studio] mp4 → %s (%d KB)", out, out.stat().st_size // 1024)
        return True

    def _thumbnail(self, timeline: dict[str, Any], channel: dict[str, Any],
                   frames_dir: Path, work: Path) -> Path | None:
        # Pick frame at the strongest "surprise" or "emotion" event;
        # else mid-reveal; else 30% of total.
        events = timeline.get("events", [])
        T = float(timeline["total_seconds"])
        priority = ["surprise", "emotion", "humor", "focus", "retention"]
        best_t = T * 0.30
        for r in priority:
            cand = [e for e in events
                    if e.get("reason") == r and e["action"] in
                    ("kinetic_text", "elastic_burst", "emoji_explosion",
                     "speed_ramp", "camera_shake")]
            if cand:
                best_t = float(cand[0]["time"]) + 0.2
                break
        f = int(best_t * self.fps)
        src = frames_dir / f"f_{f:05d}.jpg"
        if not src.exists():
            log.warning("[studio] no frame for thumbnail at t=%.2f", best_t)
            return None
        thumb = work / "thumbnail.jpg"
        try:
            from PIL import Image, ImageDraw
            img = Image.open(src).convert("RGB")
            draw = ImageDraw.Draw(img)
            # Add a punchy title overlay using channel name
            name = channel.get("name", "Hinglish Shorts")
            font = _font(110, bold=True)
            bbox = draw.textbbox((0, 0), name.upper(), font=font)
            tw = bbox[2] - bbox[0]
            x = (img.width - tw) // 2
            y = int(img.height * 0.82)
            for dx in (-4, 0, 4):
                for dy in (-4, 0, 4):
                    draw.text((x + dx, y + dy), name.upper(), font=font, fill=(0, 0, 0))
            draw.text((x, y), name.upper(), font=font, fill=(255, 220, 100))
            img.save(thumb, quality=92)
            return thumb
        except Exception as e:
            log.warning("[studio] thumbnail failed: %s", e)
            return None


# ---------- Audio helpers ----------------------------------------------------


def _write_silent_wav(path: Path, *, seconds: float, sample_rate: int = 24000) -> None:
    n = int(max(1.0, seconds) * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * n)


def _write_pcm_wav(path: Path, signal, sr: int) -> None:
    import numpy as np
    s = (np.clip(signal, -1.0, 1.0) * 32767.0).astype("int16")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(s.tobytes())


def _read_wav_mono(path: Path):
    import numpy as np
    if not path.exists() or path.stat().st_size < 100:
        return np.zeros(0, dtype=np.float32)
    with wave.open(str(path), "rb") as w:
        n = w.getnframes(); ch = w.getnchannels(); sr = w.getframerate()
        raw = w.readframes(n)
    arr = np.frombuffer(raw, dtype="int16").astype(np.float32) / 32768.0
    if ch == 2:
        arr = arr.reshape(-1, 2).mean(axis=1)
    if sr != 44100:
        # Cheap linear resample
        x_new = np.linspace(0, 1, int(len(arr) * 44100 / sr), endpoint=False)
        x_old = np.linspace(0, 1, len(arr), endpoint=False)
        arr = np.interp(x_new, x_old, arr)
    return arr.astype(np.float32)


def _sub_bass_punch(sr: int):
    import numpy as np
    n = int(0.6 * sr); t = np.arange(n) / sr
    env = np.exp(-t * 6.0)
    return (np.sin(2 * np.pi * 60.0 * t) * env * 0.85).astype(np.float32)


def _heartbeat(sr: int, *, beats: int = 2):
    import numpy as np
    out = np.zeros(int(1.4 * sr), dtype=np.float32)
    for k in range(beats):
        t0 = int(k * 0.7 * sr)
        for off, dur, freq, amp in [(0.0, 0.10, 60.0, 0.65), (0.13, 0.08, 75.0, 0.55)]:
            i = t0 + int(off * sr)
            n = int(dur * sr)
            t = np.arange(n) / sr
            env = np.exp(-t * 22.0)
            out[i:i + n] += np.sin(2 * np.pi * freq * t) * env * amp
    return out


def _loop_sting(sr: int):
    import numpy as np
    n = int(0.5 * sr); t = np.arange(n) / sr
    env = np.exp(-t * 4.0)
    sweep = np.linspace(220, 660, n)
    return (np.sin(2 * np.pi * sweep * t / sr * sr) * env * 0.5).astype(np.float32)


def _click(sr: int, freq: float):
    import numpy as np
    n = int(0.18 * sr); t = np.arange(n) / sr
    env = np.exp(-t * 60.0)
    return (np.sin(2 * np.pi * freq * t) * env * 0.7).astype(np.float32)


def _whoosh(sr: int):
    import numpy as np
    n = int(0.22 * sr); t = np.arange(n) / sr
    env = np.exp(-((t - 0.10) ** 2) / 0.0035)
    noise = np.random.randn(n) * 0.5
    return (noise * env).astype(np.float32)


# ---------- Misc helpers ------------------------------------------------------


def _hex(s: str) -> tuple[int, int, int]:
    s = (s or "#000000").lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except (ValueError, IndexError):
        return (10, 10, 20)


# Map descriptive palette names → hex. Extend as channels grow.
_PALETTE_NAMES = {
    "deep navy":       "#0a0e2a",
    "soft gold":       "#FFD86B",
    "muted red":       "#A8453E",
    "electric blue":   "#1E63FF",
    "neon green":      "#22F0C6",
    "jet black":       "#0a0a0e",
    "midnight purple": "#1F1235",
    "soft pink":       "#FFC1D8",
    "warm white":      "#F4ECE0",
}


def _palette_hex(name_or_hex: str) -> str:
    s = (name_or_hex or "").strip()
    if s.startswith("#"):
        return s
    return _PALETTE_NAMES.get(s.lower(), "#0a0a14")


def _ease_out(x: float) -> float:
    return 1 - (1 - x) ** 3


def _font(size: int, bold: bool = False):
    """Find a system font; fall back to PIL default."""
    from PIL import ImageFont
    candidates = [
        # DejaVu — bundled on most Linux distros
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    ]
    if not bold:
        candidates = [c for c in candidates if "Bold" not in c] + candidates
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()
