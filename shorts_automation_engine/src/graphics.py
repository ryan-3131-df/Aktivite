"""
Pillow-based motion-graphics renderer.

Everything that is drawn on top of the footage is pre-rendered once as an
RGBA "sprite" (captions for every active-word state, the glowing rank badge,
the list tracker for every reveal state, title cards). The per-frame
compositor then only alpha-blits small regions, which keeps a 60 fps
1080x1920 render fast.
"""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .config import FONTS_DIR, CaptionSettings
from .utils import log

RGB = tuple[int, int, int]

# Rank accent colours (#6 cool → #1 gold).
RANK_COLORS: dict[int, RGB] = {
    6: (0, 200, 255),
    5: (70, 130, 255),
    4: (165, 90, 255),
    3: (255, 70, 190),
    2: (255, 135, 20),
    1: (255, 205, 0),
}
YELLOW: RGB = (255, 226, 0)
WHITE: RGB = (255, 255, 255)
BLACK: RGB = (0, 0, 0)

# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #
_SYSTEM_BOLD = [
    # macOS
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    # Windows
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/ariblk.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
]

_ROLE_FONTS = {
    # Hormozi/CapCut captions: heavy geometric sans.
    "caption": ["Montserrat-Black.ttf", "Montserrat-ExtraBold.ttf", "Anton-Regular.ttf"],
    # Badges, titles, headers: condensed display face.
    "display": ["Anton-Regular.ttf", "Montserrat-Black.ttf"],
    # List tracker rows.
    "ui": ["Montserrat-ExtraBold.ttf", "Montserrat-Black.ttf", "Anton-Regular.ttf"],
}
_ROLE_ENV = {"caption": "CAPTION_FONT", "display": "DISPLAY_FONT", "ui": "UI_FONT"}


@functools.lru_cache(maxsize=None)
def font_path(role: str) -> Path | None:
    override = os.environ.get(_ROLE_ENV.get(role, ""), "").strip()
    candidates = ([Path(override)] if override else []) + [FONTS_DIR / f for f in _ROLE_FONTS[role]]
    candidates += [Path(p) for p in _SYSTEM_BOLD]
    for c in candidates:
        if c.is_file():
            return c
    return None


@functools.lru_cache(maxsize=None)
def get_font(role: str, size: int) -> ImageFont.FreeTypeFont:
    path = font_path(role)
    if path is None:
        log.warning("No bold TTF font found - captions will use Pillow's default font. "
                    "Run `python main.py --init-assets` to download fonts.")
        return ImageFont.load_default(size=size)
    return ImageFont.truetype(str(path), size=size)


# --------------------------------------------------------------------------- #
# Sprite + blitting
# --------------------------------------------------------------------------- #
class Sprite:
    """Pre-multiplied RGBA image ready for fast alpha compositing onto RGB frames."""

    __slots__ = ("rgba", "w", "h", "_premul", "_alpha", "_scaled")

    def __init__(self, image: Image.Image | np.ndarray):
        arr = np.asarray(image.convert("RGBA") if isinstance(image, Image.Image) else image, dtype=np.uint8)
        self.rgba = arr
        self.h, self.w = arr.shape[:2]
        alpha = arr[..., 3:4].astype(np.float32) / 255.0
        self._alpha = alpha
        self._premul = arr[..., :3].astype(np.float32) * alpha
        self._scaled: dict[int, Sprite] = {}

    def scaled(self, scale: float) -> "Sprite":
        """Cached resized copy (scale quantised to 1%)."""
        key = int(round(scale * 100))
        if key == 100:
            return self
        if key not in self._scaled:
            s = max(key, 1) / 100.0
            w, h = max(1, int(self.w * s)), max(1, int(self.h * s))
            interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
            self._scaled[key] = Sprite(cv2.resize(self.rgba, (w, h), interpolation=interp))
        return self._scaled[key]

    def blit(self, frame: np.ndarray, x: int, y: int, opacity: float = 1.0) -> None:
        """Alpha-composite onto `frame` (HxWx3 uint8, modified in place) with top-left at (x, y)."""
        if opacity <= 0.003:
            return
        fh, fw = frame.shape[:2]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + self.w, fw), min(y + self.h, fh)
        if x1 <= x0 or y1 <= y0:
            return
        sx0, sy0 = x0 - x, y0 - y
        sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
        region = frame[y0:y1, x0:x1].astype(np.float32)
        a = self._alpha[sy0:sy1, sx0:sx1]
        p = self._premul[sy0:sy1, sx0:sx1]
        if opacity >= 0.997:
            out = region * (1.0 - a) + p
        else:
            out = region * (1.0 - a * opacity) + p * opacity
        frame[y0:y1, x0:x1] = (out + 0.5).astype(np.uint8)

    def blit_centered(self, frame: np.ndarray, cx: float, cy: float, scale: float = 1.0, opacity: float = 1.0) -> None:
        spr = self.scaled(scale)
        spr.blit(frame, int(round(cx - spr.w / 2)), int(round(cy - spr.h / 2)), opacity)

    def blit_anchored(self, frame: np.ndarray, x: int, y: int, scale: float = 1.0, opacity: float = 1.0,
                      anchor: tuple[float, float] = (0.0, 0.0)) -> None:
        """Scale around an anchor point given in sprite-relative coordinates (0..1)."""
        spr = self.scaled(scale)
        ax, ay = anchor
        px = x + self.w * ax - spr.w * ax
        py = y + self.h * ay - spr.h * ay
        spr.blit(frame, int(round(px)), int(round(py)), opacity)


# --------------------------------------------------------------------------- #
# Primitive drawing helpers
# --------------------------------------------------------------------------- #
def _text_width(font: ImageFont.FreeTypeFont, text: str) -> float:
    return font.getlength(text)


def _line_height(font: ImageFont.FreeTypeFont) -> int:
    ascent, descent = font.getmetrics()
    return ascent + descent


def render_runs(
    lines: Sequence[Sequence[tuple[str, RGB]]],
    font: ImageFont.FreeTypeFont,
    *,
    stroke: int = 10,
    stroke_color: RGB = BLACK,
    line_gap: int = 0,
    shadow_offset: tuple[int, int] = (5, 8),
    shadow_blur: int = 7,
    shadow_alpha: int = 190,
    align: str = "center",
) -> Image.Image:
    """Render lines of coloured word runs with a thick stroke and soft drop shadow."""
    space = _text_width(font, " ")
    lh = _line_height(font)
    widths = [sum(_text_width(font, w) for w, _ in line) + space * max(len(line) - 1, 0) for line in lines]
    pad = stroke + shadow_blur * 2 + max(abs(shadow_offset[0]), abs(shadow_offset[1])) + 4
    W = int(max(widths or [1]) + pad * 2)
    H = int(lh * len(lines) + line_gap * max(len(lines) - 1, 0) + pad * 2)

    text_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow_mask = Image.new("L", (W, H), 0)
    td, sd = ImageDraw.Draw(text_layer), ImageDraw.Draw(shadow_mask)
    y = pad
    for line, lw in zip(lines, widths):
        x = pad + ((W - 2 * pad - lw) / 2 if align == "center" else 0)
        for word, color in line:
            td.text((x, y), word, font=font, fill=(*color, 255), stroke_width=stroke, stroke_fill=(*stroke_color, 255))
            sd.text((x + shadow_offset[0], y + shadow_offset[1]), word, font=font, fill=255,
                    stroke_width=stroke, stroke_fill=255)
            x += _text_width(font, word) + space
        y += lh + line_gap

    if shadow_alpha > 0:
        shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(shadow_blur))
        shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        shadow.putalpha(shadow_mask.point(lambda v: v * shadow_alpha // 255))
        out = Image.alpha_composite(shadow, text_layer)
    else:
        out = text_layer
    bbox = out.getbbox()
    return out.crop(bbox) if bbox else out


def _vertical_gradient(size: tuple[int, int], top: RGB, bottom: RGB) -> Image.Image:
    w, h = size
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    grad = np.array(top, np.float32) * (1 - t) + np.array(bottom, np.float32) * t
    return Image.fromarray(np.repeat(grad, w, axis=1).astype(np.uint8), "RGB")


def _shade(color: RGB, factor: float) -> RGB:
    """factor > 1 lightens toward white, < 1 darkens."""
    if factor >= 1:
        return tuple(int(c + (255 - c) * (factor - 1)) for c in color)  # type: ignore[return-value]
    return tuple(int(c * factor) for c in color)  # type: ignore[return-value]


def _glow(mask: Image.Image, color: RGB, radius: int, strength: float = 1.0) -> Image.Image:
    blurred = mask.filter(ImageFilter.GaussianBlur(radius))
    if strength != 1.0:
        blurred = blurred.point(lambda v: min(255, int(v * strength)))
    layer = Image.new("RGBA", mask.size, (*color, 0))
    layer.putalpha(blurred)
    return layer


def fit_font(role: str, text: str, max_width: int, start: int, minimum: int, stroke_ratio: float = 0.0) -> ImageFont.FreeTypeFont:
    size = start
    while size > minimum:
        f = get_font(role, size)
        if _text_width(f, text) + 2 * size * stroke_ratio <= max_width:
            return f
        size -= 2
    return get_font(role, minimum)


def _ellipsize(font: ImageFont.FreeTypeFont, text: str, max_width: int) -> str:
    if _text_width(font, text) <= max_width:
        return text
    while text and _text_width(font, text + "…") > max_width:
        text = text[:-1]
    return text.rstrip() + "…"


def _wrap_balanced(words: list[str], font: ImageFont.FreeTypeFont, max_width: int) -> list[list[str]]:
    """Split words into at most 2 lines with the most even widths."""
    if _text_width(font, " ".join(words)) <= max_width or len(words) < 2:
        return [words]
    best, best_score = [words], float("inf")
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        wa, wb = _text_width(font, a), _text_width(font, b)
        score = max(wa, wb) + abs(wa - wb) * 0.25
        if score < best_score:
            best, best_score = [words[:i], words[i:]], score
    return best


# --------------------------------------------------------------------------- #
# Captions
# --------------------------------------------------------------------------- #
def render_caption_states(words: Sequence[str], highlight: RGB, cfg: CaptionSettings) -> list[Sprite]:
    """
    One sprite per active word: the chunk is drawn in white with word *i* in the
    highlight colour. All states share the same layout so switching is seamless.
    """
    words = [w for w in words if w]
    if not words:
        return []
    joined = " ".join(words)
    # Prefer one line, shrinking down to min_font_size; otherwise wrap to two lines.
    layout_font, lines = None, None
    for size in range(cfg.font_size, cfg.min_font_size - 1, -4):
        f = get_font("caption", size)
        if _text_width(f, joined) + 2 * cfg.stroke_width <= cfg.max_width:
            layout_font, lines = f, [list(words)]
            break
    if layout_font is None:
        layout_font = get_font("caption", cfg.font_size)
        lines = _wrap_balanced(list(words), layout_font, cfg.max_width)
        while max(_text_width(layout_font, " ".join(l)) for l in lines) > cfg.max_width and layout_font.size > cfg.min_font_size:
            layout_font = get_font("caption", layout_font.size - 4)
            lines = _wrap_balanced(list(words), layout_font, cfg.max_width)

    stroke = max(4, int(cfg.stroke_width * layout_font.size / cfg.font_size))
    states: list[Sprite] = []
    for active in range(len(words)):
        idx, runs = 0, []
        for line in lines:
            row = []
            for w in line:
                row.append((w, highlight if idx == active else cfg.base_color))
                idx += 1
            runs.append(row)
        img = render_runs(runs, layout_font, stroke=stroke, line_gap=-int(layout_font.size * 0.12),
                          shadow_offset=cfg.shadow_offset, shadow_blur=cfg.shadow_blur,
                          shadow_alpha=cfg.shadow_alpha)
        states.append(img)
    # Pad every state to a common canvas so their centres line up exactly.
    W = max(i.width for i in states)
    H = max(i.height for i in states)
    sprites = []
    for img in states:
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        canvas.paste(img, ((W - img.width) // 2, (H - img.height) // 2))
        sprites.append(Sprite(canvas))
    return sprites


# --------------------------------------------------------------------------- #
# Ranking overlay: glowing badge + list tracker
# --------------------------------------------------------------------------- #
def render_badge(label: str, color: RGB, height: int = 136) -> Sprite:
    """Glossy pill badge (e.g. '#6') with an outer neon glow."""
    font = get_font("display", int(height * 0.74))
    text_w = _text_width(font, label)
    w = int(max(height * 1.45, text_w + height * 0.75))
    glow_r = 26
    margin = glow_r * 2
    W, H = w + margin * 2, height + margin * 2
    radius = height // 2

    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rounded_rectangle((margin, margin, margin + w, margin + height), radius=radius, fill=255)
    out = Image.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, 0)), _glow(mask, color, glow_r, 1.6))

    pill = _vertical_gradient((W, H), _shade(color, 1.35), _shade(color, 0.62)).convert("RGBA")
    pill.putalpha(mask)
    out = Image.alpha_composite(out, pill)

    d = ImageDraw.Draw(out)
    # Glossy highlight on the upper half.
    gloss = Image.new("L", (W, H), 0)
    ImageDraw.Draw(gloss).rounded_rectangle((margin + 8, margin + 6, margin + w - 8, margin + height // 2),
                                            radius=radius - 8, fill=70)
    out = Image.alpha_composite(out, Image.merge("RGBA", (*[Image.new("L", (W, H), 255)] * 3, gloss)))
    d = ImageDraw.Draw(out)
    d.rounded_rectangle((margin, margin, margin + w, margin + height), radius=radius, outline=(255, 255, 255, 230), width=5)

    text = render_runs([[(label, WHITE)]], font, stroke=6, shadow_offset=(3, 5), shadow_blur=5, shadow_alpha=150)
    tx = margin + (w - text.width) // 2
    ty = margin + (height - text.height) // 2
    out.alpha_composite(text, (tx, ty))
    return Sprite(out)


def render_list_panel(titles: dict[int, str | None], highlight: int | None, *, width: int = 620,
                      row_h: int = 70, show_titles: bool = True) -> Sprite:
    """
    The dynamic list tracker: rows 1..6 top-to-bottom. Revealed ranks show their
    title; the current rank row gets a glowing bar in its accent colour.
    """
    pad = 16
    glow_m = 22
    rows = [1, 2, 3, 4, 5, 6]
    W = width + glow_m * 2
    H = row_h * len(rows) + pad * 2 + glow_m * 2
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    panel = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(panel).rounded_rectangle((glow_m, glow_m, glow_m + width, H - glow_m), radius=24,
                                            fill=(8, 8, 14, 135))
    out = Image.alpha_composite(out, panel)

    num_font = get_font("display", int(row_h * 0.72))
    title_font_size = int(row_h * 0.56)
    chip = int(row_h * 0.78)
    text_x = glow_m + pad + chip + 16
    max_title_w = width - (text_x - glow_m) - pad

    for i, rank in enumerate(rows):
        y0 = glow_m + pad + i * row_h
        color = RANK_COLORS[rank]
        revealed = titles.get(rank)
        is_hl = highlight == rank
        if is_hl:
            bar_mask = Image.new("L", (W, H), 0)
            ImageDraw.Draw(bar_mask).rounded_rectangle((glow_m + 6, y0 + 3, glow_m + width - 6, y0 + row_h - 3),
                                                       radius=16, fill=255)
            out = Image.alpha_composite(out, _glow(bar_mask, color, 14, 1.2))
            bar = Image.new("RGBA", (W, H), (*_shade(color, 0.55), 0))
            bar.putalpha(bar_mask.point(lambda v: v * 215 // 255))
            out = Image.alpha_composite(out, bar)

        d = ImageDraw.Draw(out)
        cx0 = glow_m + pad
        cy0 = y0 + (row_h - chip) // 2
        chip_fill = (*color, 255) if (revealed or is_hl) else (70, 70, 80, 220)
        d.rounded_rectangle((cx0, cy0, cx0 + chip, cy0 + chip), radius=12, fill=chip_fill,
                            outline=(255, 255, 255, 200 if is_hl else 90), width=3)
        num = render_runs([[(str(rank), WHITE)]], num_font, stroke=4, shadow_alpha=0)
        out.alpha_composite(num, (cx0 + (chip - num.width) // 2, cy0 + (chip - num.height) // 2))

        if revealed and show_titles:
            label = revealed.upper()
            f = fit_font("ui", label, max_title_w, title_font_size, int(title_font_size * 0.72))
            label = _ellipsize(f, label, max_title_w - 12)
            txt = render_runs([[(label, WHITE)]], f, stroke=4, shadow_offset=(2, 3), shadow_blur=3,
                              shadow_alpha=140, align="left")
        else:
            f = get_font("ui", title_font_size)
            txt = render_runs([[("• • •", (150, 150, 160))]], f, stroke=0, shadow_alpha=0, align="left")
        out.alpha_composite(txt, (text_x, y0 + (row_h - txt.height) // 2))
    return Sprite(out)


# --------------------------------------------------------------------------- #
# Titles
# --------------------------------------------------------------------------- #
def render_title_card(kicker: str, title: str, max_width: int = 960) -> Sprite:
    """Hook title card: big yellow kicker ('TOP 6') over a white uppercase title."""
    kicker_font = fit_font("display", kicker.upper(), max_width, 200, 110, 0.06)
    k_img = render_runs([[(kicker.upper(), YELLOW)]], kicker_font, stroke=11, shadow_offset=(6, 10),
                        shadow_blur=9, shadow_alpha=210)
    words = title.upper().split()
    t_font = get_font("display", 118)
    lines = _wrap_balanced(words, t_font, max_width)
    while max(_text_width(t_font, " ".join(l)) for l in lines) > max_width and t_font.size > 64:
        t_font = get_font("display", t_font.size - 6)
        lines = _wrap_balanced(words, t_font, max_width)
    t_img = render_runs([[(w, WHITE) for w in line] for line in lines], t_font, stroke=9,
                        line_gap=-int(t_font.size * 0.08), shadow_offset=(5, 9), shadow_blur=8, shadow_alpha=210)
    return Sprite(stack_vertical([k_img, t_img], gap=-6))


def render_header(text: str, max_width: int = 1000) -> Sprite:
    font = fit_font("display", text.upper(), max_width, 70, 44, 0.05)
    img = render_runs([[(w, WHITE) for w in text.upper().split()]], font, stroke=6, shadow_offset=(3, 6),
                      shadow_blur=6, shadow_alpha=200)
    return Sprite(img)


def stack_vertical(images: Sequence[Image.Image], gap: int = 0) -> Image.Image:
    W = max(i.width for i in images)
    H = sum(i.height for i in images) + gap * (len(images) - 1)
    canvas = Image.new("RGBA", (W, max(H, 1)), (0, 0, 0, 0))
    y = 0
    for img in images:
        canvas.alpha_composite(img, ((W - img.width) // 2, max(y, 0)))
        y += img.height + gap
    return canvas


def blend_white(frame: np.ndarray, alpha: float) -> np.ndarray:
    """Flash effect: mix the frame toward white."""
    if alpha <= 0.003:
        return frame
    return cv2.addWeighted(frame, 1.0 - alpha, np.full_like(frame, 255), alpha, 0)


if sys.platform == "win32":  # pragma: no cover - keep cv2 from spawning too many threads on Windows
    cv2.setNumThreads(max(1, (os.cpu_count() or 4) // 2))
