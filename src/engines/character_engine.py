"""Character Engine — recurring channel-bound visual identities.

Each channel gets ONE primary character with consistent design tokens
(palette, hair, eye, outline, accessory). Expressions are generated as SVG
overlays that compose on top of any background, so the character looks the
same across every video while still reacting emotionally per beat.

Why SVG (not Stable Diffusion right now):
  • Deterministic — the SAME character appears every video, no drift
  • Zero GPU dependency, runs in this sandbox today
  • Cheap to render at 1080x1920
  • Trivially upgradable: replace `expression_to_svg` with an SD pipeline
    later without touching the Editor Agent's API.

The engine emits both the SVG markup AND an alpha PNG sized to the canvas,
ready for the renderer to composite at any timeline position.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from ..core.logger import get_logger
from ..core.state import ARTIFACT_DIR, load_yaml, ROOT

log = get_logger(__name__)


VALID_EXPRESSIONS = (
    "happy", "shocked", "confused", "angry",
    "curious", "romantic", "embarrassed",
)


# ---------- Channel character DNA -----------------------------------------

CHARACTERS: dict[str, dict[str, Any]] = {
    "human_decoder": {
        "name": "Maya",
        "archetype": "stylized semi-realistic, expressive eyes, warm female",
        "skin":     "#F1C9A5",
        "hair":     "#1A1A2E",
        "hair_hl":  "#2E2E55",
        "eye":      "#3B6FF2",
        "lip":      "#C8526A",
        "outline":  "#1B1B1B",
        "accent":   "#FFD86B",            # earring / hair clip
        "outfit":   "#5C2E66",
        "vibe":     "mysterious, emotional",
    },
    "ai_money_lab": {
        "name": "Arjun",
        "archetype": "futuristic AI mentor, digital entrepreneur, sharp jaw",
        "skin":     "#D8AF89",
        "hair":     "#0E0E1A",
        "hair_hl":  "#1F1F35",
        "eye":      "#22F0C6",            # glowing cyan
        "lip":      "#3F2E2E",
        "outline":  "#0B0B0B",
        "accent":   "#22F0C6",
        "outfit":   "#0F2E45",
        "vibe":     "smart, exciting, high-energy",
    },
    "dramaverse": {
        "name": "Riya",
        "archetype": "anime-leaning, large eyes, dramatic shojo flair",
        "skin":     "#FBD9C5",
        "hair":     "#3A2540",
        "hair_hl":  "#5A3760",
        "eye":      "#A35BFF",
        "lip":      "#E8506E",
        "outline":  "#1B1B1B",
        "accent":   "#FFC1D8",
        "outfit":   "#1F1235",
        "vibe":     "dramatic, romantic, expressive",
    },
}


# ---------- Expression → facial geometry tweaks ---------------------------

EXPRESSIONS: dict[str, dict[str, float]] = {
    # eye_open: 0..1 (1 = wide), brow_lift: -1..1, mouth_curve: -1..1 (1 smile),
    # blush: 0..1, mouth_open: 0..1, sparkle: 0..1 (cute eye sparkle)
    "happy":      {"eye_open": 0.55, "brow_lift":  0.20, "mouth_curve":  0.85,
                   "blush": 0.30, "mouth_open": 0.20, "sparkle": 0.50},
    "shocked":    {"eye_open": 1.00, "brow_lift":  0.75, "mouth_curve":  0.05,
                   "blush": 0.00, "mouth_open": 0.85, "sparkle": 0.00},
    "confused":   {"eye_open": 0.65, "brow_lift": -0.30, "mouth_curve": -0.05,
                   "blush": 0.05, "mouth_open": 0.10, "sparkle": 0.00},
    "angry":      {"eye_open": 0.55, "brow_lift": -0.85, "mouth_curve": -0.55,
                   "blush": 0.10, "mouth_open": 0.20, "sparkle": 0.00},
    "curious":    {"eye_open": 0.85, "brow_lift":  0.45, "mouth_curve":  0.10,
                   "blush": 0.05, "mouth_open": 0.15, "sparkle": 0.20},
    "romantic":   {"eye_open": 0.45, "brow_lift":  0.10, "mouth_curve":  0.40,
                   "blush": 0.85, "mouth_open": 0.05, "sparkle": 0.85},
    "embarrassed":{"eye_open": 0.40, "brow_lift": -0.20, "mouth_curve": -0.10,
                   "blush": 0.95, "mouth_open": 0.10, "sparkle": 0.10},
}


@dataclass
class CharacterFrame:
    channel_id: str
    name: str
    expression: str
    svg: str
    png_path: str | None
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Don't dump full svg into JSON output of the orchestrator; keep a hash.
        d["svg_chars"] = len(self.svg)
        d["svg"] = self.svg[:600] + ("..." if len(self.svg) > 600 else "")
        return d


class CharacterEngine:
    """Single deterministic character per channel; many expressions per video."""

    def __init__(self) -> None:
        pass

    def for_channel(self, channel_id: str) -> dict[str, Any]:
        return CHARACTERS.get(channel_id, CHARACTERS["human_decoder"])

    def render(self, *, channel_id: str, expression: str,
               width: int = 1080, height: int = 1920,
               video_id: str | None = None,
               persist_png: bool = True) -> CharacterFrame:
        if expression not in VALID_EXPRESSIONS:
            log.warning("[character] unknown expression=%r → curious", expression)
            expression = "curious"

        dna = self.for_channel(channel_id)
        params = EXPRESSIONS[expression]
        svg = self._draw_svg(dna, params, w=width, h=height)

        png_path: str | None = None
        if persist_png and video_id:
            outdir = ARTIFACT_DIR / video_id / "characters"
            outdir.mkdir(parents=True, exist_ok=True)
            png_path = str(outdir / f"{dna['name'].lower()}_{expression}.png")
            self._svg_to_png(svg, png_path, width=width, height=height)

        return CharacterFrame(
            channel_id=channel_id, name=dna["name"], expression=expression,
            svg=svg, png_path=png_path, width=width, height=height,
        )

    # ---- SVG drawing -----------------------------------------------------

    def _draw_svg(self, dna: dict[str, Any], p: dict[str, float],
                  *, w: int, h: int) -> str:
        # Anchor the head in the upper-half of a 9:16 frame.
        cx = w * 0.5
        cy = h * 0.40
        head_r = w * 0.22

        # Eyes
        eye_y = cy - head_r * 0.10
        eye_dx = head_r * 0.34
        eye_open = p["eye_open"]
        eye_w = head_r * 0.18
        eye_h = head_r * 0.22 * eye_open

        # Brows
        brow_y = eye_y - head_r * 0.30 - head_r * 0.10 * p["brow_lift"]
        brow_dx = eye_dx
        brow_w = head_r * 0.22
        brow_thick = head_r * 0.04

        # Mouth
        mouth_y = cy + head_r * 0.42
        mouth_w = head_r * 0.34
        curve = p["mouth_curve"]
        mouth_h = head_r * 0.10 * (0.4 + abs(curve) * 0.6)
        m_open = p["mouth_open"]

        # Blush
        blush_alpha = p["blush"]
        # Sparkle
        spk = p["sparkle"]

        sparkle_block = ""
        if spk > 0.2:
            sparkle_block = (
                f'<g opacity="{spk:.2f}">'
                f'<circle cx="{cx - eye_dx + eye_w*0.4:.0f}" cy="{eye_y - eye_h*0.3:.0f}"'
                f' r="{eye_w*0.18:.0f}" fill="white"/>'
                f'<circle cx="{cx + eye_dx + eye_w*0.4:.0f}" cy="{eye_y - eye_h*0.3:.0f}"'
                f' r="{eye_w*0.18:.0f}" fill="white"/>'
                f"</g>"
            )

        mouth_path = self._mouth_path(cx, mouth_y, mouth_w, mouth_h, curve, m_open)

        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <defs>
    <radialGradient id="bg" cx="50%" cy="35%" r="65%">
      <stop offset="0%" stop-color="{dna['outfit']}" stop-opacity="0.0"/>
      <stop offset="100%" stop-color="{dna['outfit']}" stop-opacity="0.0"/>
    </radialGradient>
    <linearGradient id="hairGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="{dna['hair_hl']}"/>
      <stop offset="100%" stop-color="{dna['hair']}"/>
    </linearGradient>
  </defs>
  <rect width="{w}" height="{h}" fill="url(#bg)"/>
  <!-- shoulders / outfit -->
  <path d="M {cx - head_r*1.7:.0f} {cy + head_r*1.25:.0f}
           Q {cx:.0f} {cy + head_r*0.95:.0f} {cx + head_r*1.7:.0f} {cy + head_r*1.25:.0f}
           L {cx + head_r*2.0:.0f} {h:.0f} L {cx - head_r*2.0:.0f} {h:.0f} Z"
        fill="{dna['outfit']}" stroke="{dna['outline']}" stroke-width="6"/>
  <!-- hair back -->
  <ellipse cx="{cx:.0f}" cy="{cy + head_r*0.05:.0f}" rx="{head_r*1.30:.0f}" ry="{head_r*1.15:.0f}"
           fill="url(#hairGrad)"/>
  <!-- face -->
  <ellipse cx="{cx:.0f}" cy="{cy:.0f}" rx="{head_r:.0f}" ry="{head_r*1.08:.0f}"
           fill="{dna['skin']}" stroke="{dna['outline']}" stroke-width="5"/>
  <!-- hair fringe -->
  <path d="M {cx - head_r:.0f} {cy - head_r*0.15:.0f}
           Q {cx - head_r*0.3:.0f} {cy - head_r*1.05:.0f} {cx + head_r*0.4:.0f} {cy - head_r*0.85:.0f}
           T {cx + head_r:.0f} {cy - head_r*0.10:.0f}
           Q {cx + head_r*0.5:.0f} {cy - head_r*0.55:.0f} {cx:.0f} {cy - head_r*0.55:.0f}
           Q {cx - head_r*0.55:.0f} {cy - head_r*0.45:.0f} {cx - head_r:.0f} {cy - head_r*0.15:.0f} Z"
        fill="url(#hairGrad)"/>
  <!-- accent (earring/clip) -->
  <circle cx="{cx + head_r*1.05:.0f}" cy="{cy + head_r*0.55:.0f}" r="{head_r*0.06:.0f}" fill="{dna['accent']}"/>
  <!-- blush -->
  <ellipse cx="{cx - head_r*0.55:.0f}" cy="{cy + head_r*0.20:.0f}"
           rx="{head_r*0.18:.0f}" ry="{head_r*0.10:.0f}"
           fill="{dna['lip']}" opacity="{0.55*blush_alpha:.2f}"/>
  <ellipse cx="{cx + head_r*0.55:.0f}" cy="{cy + head_r*0.20:.0f}"
           rx="{head_r*0.18:.0f}" ry="{head_r*0.10:.0f}"
           fill="{dna['lip']}" opacity="{0.55*blush_alpha:.2f}"/>
  <!-- eyes -->
  <ellipse cx="{cx - eye_dx:.0f}" cy="{eye_y:.0f}" rx="{eye_w:.0f}" ry="{max(2, eye_h):.0f}" fill="white" stroke="{dna['outline']}" stroke-width="3"/>
  <ellipse cx="{cx + eye_dx:.0f}" cy="{eye_y:.0f}" rx="{eye_w:.0f}" ry="{max(2, eye_h):.0f}" fill="white" stroke="{dna['outline']}" stroke-width="3"/>
  <circle  cx="{cx - eye_dx:.0f}" cy="{eye_y:.0f}" r="{eye_w*0.55:.0f}" fill="{dna['eye']}"/>
  <circle  cx="{cx + eye_dx:.0f}" cy="{eye_y:.0f}" r="{eye_w*0.55:.0f}" fill="{dna['eye']}"/>
  <circle  cx="{cx - eye_dx:.0f}" cy="{eye_y:.0f}" r="{eye_w*0.25:.0f}" fill="black"/>
  <circle  cx="{cx + eye_dx:.0f}" cy="{eye_y:.0f}" r="{eye_w*0.25:.0f}" fill="black"/>
  {sparkle_block}
  <!-- brows -->
  <path d="M {cx - brow_dx - brow_w:.0f} {brow_y:.0f}
           Q {cx - brow_dx:.0f} {brow_y - brow_thick*1.3:.0f} {cx - brow_dx + brow_w:.0f} {brow_y:.0f}"
        stroke="{dna['hair']}" stroke-width="{brow_thick:.0f}" fill="none" stroke-linecap="round"/>
  <path d="M {cx + brow_dx - brow_w:.0f} {brow_y:.0f}
           Q {cx + brow_dx:.0f} {brow_y - brow_thick*1.3:.0f} {cx + brow_dx + brow_w:.0f} {brow_y:.0f}"
        stroke="{dna['hair']}" stroke-width="{brow_thick:.0f}" fill="none" stroke-linecap="round"/>
  <!-- mouth -->
  {mouth_path}
</svg>"""
        return svg

    @staticmethod
    def _mouth_path(cx: float, my: float, mw: float, mh: float,
                    curve: float, m_open: float) -> str:
        # Curve sign decides smile vs frown
        c_y = my + (mh * (curve * -1.4))
        if m_open > 0.05:
            # Open mouth: lens shape
            o_h = mh * (1.0 + 1.6 * m_open)
            return (
                f'<path d="M {cx-mw:.0f} {my:.0f} '
                f'Q {cx:.0f} {c_y:.0f} {cx+mw:.0f} {my:.0f} '
                f'Q {cx:.0f} {my+o_h:.0f} {cx-mw:.0f} {my:.0f} Z" '
                f'fill="#5A1F2A" stroke="#1B1B1B" stroke-width="4"/>'
            )
        return (
            f'<path d="M {cx-mw:.0f} {my:.0f} Q {cx:.0f} {c_y:.0f} {cx+mw:.0f} {my:.0f}" '
            f'stroke="#7A2230" stroke-width="6" fill="none" stroke-linecap="round"/>'
        )

    # ---- SVG → PNG -------------------------------------------------------

    @staticmethod
    def _svg_to_png(svg: str, out_path: str, *, width: int, height: int) -> None:
        """Best-effort SVG→PNG using cairosvg if installed; else write SVG only.

        We always also write the .svg next to the .png for inspection.
        """
        try:
            import cairosvg  # type: ignore
            cairosvg.svg2png(bytestring=svg.encode("utf-8"),
                             output_width=width, output_height=height,
                             write_to=out_path)
            Path(out_path).with_suffix(".svg").write_text(svg, encoding="utf-8")
            return
        except Exception:
            pass
        # Pillow can't render SVG directly. Dump SVG; the renderer can fall
        # back to a Pillow-painted version if no png exists.
        Path(out_path).with_suffix(".svg").write_text(svg, encoding="utf-8")
