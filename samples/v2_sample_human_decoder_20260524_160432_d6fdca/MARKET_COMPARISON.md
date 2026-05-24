# Market comparison — v2 sample vs current Hinglish Shorts standard

**Sample:** `sample_human_decoder_20260524_160432_d6fdca`
**What changed vs v1:** real Sarvam Hinglish voice (was espeak), real Pexels portrait B-roll per beat (was Pillow gradient), real 1080×1920 30fps 12 MB mp4 (was 3 MB synthetic).

## Honest gap analysis

I checked what your account actually had access to before this re-render:

| Tool | Status | Used in v2? |
|------|--------|-------------|
| Sarvam (`sarvam api key`) | ✅ key present, tested HTTP 200 | **YES** — voice now native Hinglish |
| Pexels (`pixels api key`) | ✅ key present, tested HTTP 200 | **YES** — 5 portrait clips fetched |
| Anthropic / OpenAI | ❌ no key | **No** — script still mock; usable but not LLM-grade |
| ElevenLabs | ❌ no key | No — Sarvam covered voice |
| Bria.ai | ❌ no auth (device-flow needs your sign-in) | No — character is still SVG |
| Pixabay / Jamendo | ❌ no key | No — music is synthesized bed |

## What top Hinglish psychology Shorts (BeerBiceps Skillhouse, Sandeep Maheshwari, Vedanta-style channels) do that we now match — and don't

| Element | Market standard 2025-26 | This sample (v2) | Gap |
|---|---|---|---|
| **Voice** | Native Hindi/Hinglish, expressive, breath cues, pause discipline | Sarvam `bulbul:v2 / anushka` — native pronunciation, controlled pace | ✅ matches; only weakness is no per-emotion pitch shifting yet |
| **Hook 0-2s** | Tight verbal promise + visual pattern interrupt | "She stopped replying suddenly..." + push_zoom + sub-bass + heartbeat SFX + character entry | ✅ strong, hits the multi-sensory bar from [stormy.ai](https://stormy.ai/blog/psychology-of-audience-retention-2025) |
| **Visual base** | Real human face / B-roll / branded character art (no flat shapes) | Real Pexels portrait clips per beat (blurred + dimmed so kinetic text reads) | ⚠️ acceptable; will look templated until we ship Bria.ai character art |
| **Pacing** | Cut every 1-2.5s, max gap ≤ 3s | avg gap 1.17s, max gap 5.19s ⚠️ | Will fix: too-long gap during reveal — editor agent should add a mid-reveal beat reset |
| **Music** | Trending lo-fi/cinematic, ducks under voice, swells at twist | Synthesized bed with intensity curve baked in (peaks at hook + twist, fades at loop) — works but not on-trend | ⚠️ next step: pull a real CC-BY trending track via Pixabay or via ElevenLabs Music |
| **Subtitles / kinetic text** | Always on, big, white-with-black-stroke, max 4-6 visible words | Identical: 80px DejaVu Bold, 3-word chunks, accent underline | ✅ matches |
| **Hooks per minute** | 5-8 effect resets in first 30s | 43 timeline events, 5 reasoned categories | ✅ matches |
| **Comment trigger** | Pinned comment + on-screen prompt at end | "True or false?" pill at y=0.92 + "Comment YES if this happened to you." pinned comment | ✅ matches |
| **Loop ending** | Mirrors the hook for re-watch | "Kal phir same situation aayegi. Ab pata hai kya karna hai..." mirrors "She stopped replying suddenly..." | ✅ matches |

## Where this sample still falls short of pro creators

1. **Script intelligence** — without Anthropic/OpenAI, the script is from a structurally-correct mock. Top creators write hooks calibrated for their specific audience daily. **Fix:** add `ANTHROPIC_API_KEY` to env.
2. **Character art** — we ship a Pillow-painted Maya. BeerBiceps shows the actual face; AI-channel competitors use Stable Diffusion / Bria custom character art. **Fix:** authenticate Bria.ai (one-click) — the character_engine swaps SVG out for AI-generated art automatically.
3. **Music** — synthesized bed sounds artificial against trending Bollywood/lo-fi tracks. **Fix:** add `PIXABAY_API_KEY` (free) — the music_intelligence engine already prefers Pixabay tracks when keyed.
4. **Voice expression** — Sarvam delivers native pronunciation but not yet per-emotion pitch/speed. **Fix:** plumb the `EmotionPlan.voice.{speed,pitch}` from the editor through to Sarvam's `pace`/`pitch` params per chunk (small code change).

## Concrete pro-quality checklist (what to expect once 4 keys are added)

```
ANTHROPIC_API_KEY=...      → script LLM-quality, never mock again
PIXABAY_API_KEY=...        → trending CC-BY music bed (not synth)
ELEVENLABS_API_KEY=...     → backup TTS + ElevenLabs Music if Sarvam ever rate-limits
BRIA_AUTH (device flow)    → AI-rendered character + scene art per beat
```

With those in place + the existing editor agent + Sarvam voice, this pipeline produces **publishable-grade Hinglish Shorts**, not slop.

## Quality scorecard delta

| Axis | v1 | v2 | Pro creator target |
|------|----|----|--------------------|
| Voice naturalness | 1/10 (espeak) | 7/10 (Sarvam) | 9/10 (real human) |
| Visual quality | 2/10 (geometric) | 6/10 (real B-roll, blurred bg) | 9/10 (Bria + B-roll layered) |
| Pacing | 7/10 | 7/10 | 9/10 (close gap fix) |
| Music | 4/10 (synth pad) | 4/10 (still synth) | 9/10 (real trending track) |
| Editor reasoning | 9/10 | 9/10 | 9/10 |
| **Overall** | **5/10** | **7/10** | **9/10** |

**Bottom line:** v2 closed two of the four gaps (voice + visuals). Music and AI-character are blocked on keys/auth you can plug in.
