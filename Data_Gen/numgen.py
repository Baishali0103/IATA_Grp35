# -*- coding: utf-8 -*-
"""
numgen.py  v2 — Synthetic number image data generator.

NEW IN v2:
  • Handwriting-style distortion: elastic deformation, rotation/shear,
    localised (non-global) random blur.  Controlled by distortion_level 0-1.
  • Free placement: numbers placed at random (x, y) with collision detection.
  • Per-digit colouring: each digit in a number can have its own colour.
  • Partial overlap: numbers may overlap; perceptual colour-contrast is
    enforced between overlapping regions.
  • Varying backgrounds: plain, ruled, grid, tinted.
  • Crowding parameter: low / medium / high controls how many numbers appear
    and how densely packed they are.
  • Occlusion tracking: manifest records what fraction of each number is
    covered by numbers in front of it.
  • Ambiguous flag: scenes where any number is >50 % occluded.
  • Counting sentences: "Three numbers are visible …" etc.

Notation format (backward-compatible extension):
    single digit, single colour  : 5:red,large
    multi-digit, single colour   : 67:blue,large
    multi-digit, per-digit colour: 68:red|blue,large      (6=red, 8=blue)
    multi-number scene           : 67:blue,large | 42:red|green,small

Output layout (identical to v1 / shapes generator):
    Numbers_Data/
        manifest.csv
        train / val / test /
            text / meta / images / matrices /
"""

import csv
import math
import random
import itertools
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tqdm import tqdm


# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).resolve().parent / "Numbers_Data"
SPLITS        = ["train", "val", "test"]
SUBDIRS       = ["text", "meta", "images", "matrices"]
MANIFEST_PATH = BASE_DIR / "manifest.csv"
MANIFEST_FIELDS = [
    "filename", "notation",
    "num_numbers", "values", "colors", "sizes", "num_digits_list",
    "background_type", "crowding", "distortion_level",
    "colour", "has_overlap", "occlusion_pcts", "ambiguous",
    "split",
]


def _ensure_dirs() -> None:
    for split in SPLITS:
        for sub in SUBDIRS:
            (BASE_DIR / split / sub).mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)


def _init_manifest() -> None:
    if not MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(MANIFEST_FIELDS)


def _append_manifest(row: dict) -> None:
    with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([row[k] for k in MANIFEST_FIELDS])


def _dir(split: str, kind: str) -> Path:
    return BASE_DIR / split / kind


# ── Vocabulary ──────────────────────────────────────────────────────────────────

COLORS: dict[str, tuple[int, int, int]] = {
    "red":    (210,  50,  55),
    "blue":   ( 55,  90, 210),
    "green":  ( 55, 175,  80),
    "yellow": (215, 190,  45),
    "purple": (145,  55, 195),
    "orange": (225, 125,  45),
    "pink":   (225,  95, 155),
    "cyan":   ( 45, 195, 215),
}
COLOR_NAMES = list(COLORS.keys())

# Minimum perceptual colour distance to keep overlapping digits readable.
MIN_COLOR_DISTANCE = 130.0

SIZES      = ["small", "medium", "large"]
SIZE_WORDS = {
    "small":  ["small",        "tiny",         "little"],
    "medium": ["medium-sized", "medium",        "mid-sized"],
    "large":  ["large",        "big",           "sizeable"],
}

DIGIT_LENGTHS  = [1, 2, 3, 4]
CROWDING_MODES = ["low", "medium", "high"]

CROWDING_COUNT   = {"low": (1, 2), "medium": (2, 3), "high": (3, 4)}
CROWDING_MAX_IOU = {"low": 0.04,   "medium": 0.35,   "high": 0.65}

BACKGROUND_TYPES = ["plain", "ruled", "grid", "tinted"]
TINT_OPTIONS: dict[str, tuple[int, int, int]] = {
    "cream":     (250, 245, 225),
    "blue_grey": (225, 230, 240),
    "yellow":    (250, 248, 218),
    "rose":      (248, 235, 235),
    "sage":      (232, 245, 232),
}

IMG_SIZE = 256

FONT_SIZES: dict[tuple[int, str], int] = {
    (1, "small"): 52,  (1, "medium"): 88,  (1, "large"): 126,
    (2, "small"): 46,  (2, "medium"): 76,  (2, "large"): 108,
    (3, "small"): 34,  (3, "medium"): 56,  (3, "large"): 82,
    (4, "small"): 26,  (4, "medium"): 44,  (4, "large"): 66,
}

BW_FILL = 55


# ── Font helpers ────────────────────────────────────────────────────────────────

_FONT_PATHS = [
    # ── Sans-serif bold (clean, modern) ───────────────────────────────────────
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/google-fonts/Poppins-Medium.ttf",
    "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf",
    "/usr/share/fonts/opentype/tlwg/Loma-Bold.otf",

    # ── Sans-serif regular (lighter stroke weight, more variety) ──────────────
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/google-fonts/Poppins-Regular.ttf",

    # ── Serif bold (classic, ink-like stroke contrast) ────────────────────────
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/crosextra/Caladea-Bold.ttf",

    # ── Monospace bold (fixed-width, technical feel) ──────────────────────────
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",

    # ── macOS fallbacks ───────────────────────────────────────────────────────
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Times.ttc",
    "/System/Library/Fonts/Courier.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",

    # ── Windows fallbacks ─────────────────────────────────────────────────────
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/timesbd.ttf",
    "C:/Windows/Fonts/courbd.ttf",
    "C:/Windows/Fonts/verdanab.ttf",
    "C:/Windows/Fonts/georgiabd.ttf",
    "C:/Windows/Fonts/consolab.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/cambriab.ttf",
]
_AVAILABLE_FONTS: list[str] = [p for p in _FONT_PATHS if Path(p).exists()]


def _pick_font_path() -> str | None:
    return random.choice(_AVAILABLE_FONTS) if _AVAILABLE_FONTS else None


def _load_font(path: str | None, px: int) -> ImageFont.FreeTypeFont:
    if path:
        try:
            return ImageFont.truetype(path, px)
        except (IOError, OSError):
            pass
    try:
        return ImageFont.load_default(size=px)       # type: ignore[call-arg]
    except TypeError:
        return ImageFont.load_default()               # type: ignore[return-value]


def _measure_char(char: str,
                  font: ImageFont.FreeTypeFont) -> tuple[int, int, int, int]:
    """Return (width, height, x_off, y_off) for a single character."""
    tmp = ImageDraw.Draw(Image.new("L", (2000, 500), 255))
    bb  = tmp.textbbox((0, 0), char, font=font)
    return bb[2] - bb[0], bb[3] - bb[1], bb[0], bb[1]


# ── Colour helpers ──────────────────────────────────────────────────────────────

def _color_distance(a: str, b: str) -> float:
    r1, g1, b1_ = COLORS[a]
    r2, g2, b2_ = COLORS[b]
    return math.sqrt(2*(r1-r2)**2 + 4*(g1-g2)**2 + 3*(b1_-b2_)**2)


def _contrasting_color(existing: list[str]) -> str:
    """Pick a colour with sufficient contrast against every colour in existing."""
    pool = COLOR_NAMES.copy()
    random.shuffle(pool)
    for c in pool:
        if all(_color_distance(c, e) >= MIN_COLOR_DISTANCE for e in existing):
            return c
    # Fallback: most distant
    return max(COLOR_NAMES,
               key=lambda c: min(_color_distance(c, e) for e in existing))


def _assign_digit_colors(num_digits: int,
                          forbidden: list[str] | None = None) -> list[str]:
    """
    Assign per-digit colours.
    Adjacent digits within the number contrast with each other.
    All chosen colours also contrast against `forbidden` (used for overlaps).
    """
    forbidden = forbidden or []
    colors: list[str] = []
    for _ in range(num_digits):
        avoid = colors[-1:] + forbidden
        colors.append(_contrasting_color(avoid) if avoid else random.choice(COLOR_NAMES))
    return colors


# ── Data classes ────────────────────────────────────────────────────────────────

@dataclass
class DigitSpec:
    char:  str    # single character '0'–'9'
    color: str    # colour name (set even for B&W — kept for notation/manifest)


@dataclass
class NumberInstance:
    value:            int
    digit_specs:      list[DigitSpec]
    size:             str
    font_path:        str | None
    z_order:          int           # higher z = drawn in front
    distortion_level: float
    x:    int   = 0                 # centre x on canvas (set during layout)
    y:    int   = 0                 # centre y on canvas
    bbox: tuple = field(default_factory=lambda: (0, 0, 0, 0))  # x0,y0,x1,y1

    @property
    def num_digits(self) -> int:
        return len(str(self.value))

    @property
    def all_same_color(self) -> bool:
        return len({d.color for d in self.digit_specs}) == 1

    @property
    def main_color(self) -> str:
        return self.digit_specs[0].color

    def color_notation(self) -> str:
        if self.all_same_color:
            return self.main_color
        return "|".join(d.color for d in self.digit_specs)

    def to_token(self) -> str:
        return f"{self.value}:{self.color_notation()},{self.size}"


@dataclass
class Scene:
    numbers:          list[NumberInstance]
    background_type:  str
    crowding:         str
    colour:           bool
    distortion_level: float

    def to_notation(self) -> str:
        return " | ".join(n.to_token() for n in self.numbers)

    @property
    def has_overlap(self) -> bool:
        return any(
            _bbox_iou(a.bbox, b.bbox) > 0.01
            for i, a in enumerate(self.numbers)
            for b in self.numbers[i+1:]
        )

    def occlusion_pcts(self) -> list[float]:
        """Fraction of each number's bbox occluded by numbers with higher z_order."""
        result = []
        for inst in self.numbers:
            above = [o for o in self.numbers if o.z_order > inst.z_order]
            occ   = min(1.0, sum(_bbox_occlusion(inst.bbox, a.bbox) for a in above))
            result.append(round(occ, 3))
        return result

    @property
    def ambiguous(self) -> bool:
        return any(p > 0.50 for p in self.occlusion_pcts())


# ── Bounding-box helpers ────────────────────────────────────────────────────────

def _bbox_iou(a: tuple, b: tuple) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    inter  = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1, (a[2]-a[0]) * (a[3]-a[1]))
    area_b = max(1, (b[2]-b[0]) * (b[3]-b[1]))
    return inter / (area_a + area_b - inter)


def _bbox_occlusion(lower: tuple, upper: tuple) -> float:
    """Fraction of lower's area covered by upper."""
    ix0, iy0 = max(lower[0], upper[0]), max(lower[1], upper[1])
    ix1, iy1 = min(lower[2], upper[2]), min(lower[3], upper[3])
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    inter      = (ix1 - ix0) * (iy1 - iy0)
    lower_area = max(1, (lower[2]-lower[0]) * (lower[3]-lower[1]))
    return inter / lower_area


# ── Layout / placement ──────────────────────────────────────────────────────────

def _layout_numbers(instances: list[NumberInstance],
                    crowding:  str,
                    img_size:  int = IMG_SIZE,
                    max_tries: int = 120) -> list[NumberInstance]:
    """
    Assign (x, y, bbox) to each NumberInstance via rejection sampling.
    """
    max_iou = CROWDING_MAX_IOU[crowding]
    placed: list[tuple] = []

    for inst in instances:
        font_px = FONT_SIZES.get((inst.num_digits, inst.size), 60)
        font    = _load_font(inst.font_path, font_px)

        # Measure total text extent
        total_w, total_h = 0, 0
        for spec in inst.digit_specs:
            cw, ch, _, _ = _measure_char(spec.char, font)
            total_w += cw
            total_h  = max(total_h, ch)

        pad = max(14, int(font_px * 0.22))
        sw  = total_w + 2 * pad
        sh  = total_h + 2 * pad
        margin = 8

        chosen = None
        for _ in range(max_tries):
            cx = random.randint(sw // 2 + margin, max(sw // 2 + margin + 1, img_size - sw // 2 - margin))
            cy = random.randint(sh // 2 + margin, max(sh // 2 + margin + 1, img_size - sh // 2 - margin))
            x0, y0 = cx - sw // 2, cy - sh // 2
            candidate = (x0, y0, x0 + sw, y0 + sh)
            if all(_bbox_iou(candidate, p) <= max_iou for p in placed):
                chosen = (cx, cy, candidate)
                break

        if chosen is None:
            cx = random.randint(sw // 2 + margin, max(sw // 2 + margin + 1, img_size - sw // 2 - margin))
            cy = random.randint(sh // 2 + margin, max(sh // 2 + margin + 1, img_size - sh // 2 - margin))
            x0, y0 = cx - sw // 2, cy - sh // 2
            chosen = (cx, cy, (x0, y0, x0 + sw, y0 + sh))

        inst.x, inst.y, inst.bbox = chosen
        placed.append(inst.bbox)

    return instances


# ── Overlap contrast enforcement ────────────────────────────────────────────────

def _enforce_overlap_contrast(instances: list[NumberInstance]) -> None:
    """
    For every pair of overlapping numbers, reassign digit colours on the front
    number so they contrast against all digit colours of the back number.
    """
    for front in instances:
        for back in instances:
            if back.z_order >= front.z_order:
                continue
            if _bbox_iou(front.bbox, back.bbox) < 0.01:
                continue
            back_colors = [d.color for d in back.digit_specs]
            for spec in front.digit_specs:
                if any(_color_distance(spec.color, bc) < MIN_COLOR_DISTANCE
                       for bc in back_colors):
                    spec.color = _contrasting_color(back_colors)


# ── Background rendering ────────────────────────────────────────────────────────

def _render_background(size: int,
                        bg_type: str,
                        colour:  bool) -> np.ndarray:
    bg_val = random.randint(232, 248)
    mode   = "RGB" if colour else "L"

    if not colour or bg_type == "plain":
        bg  = (bg_val, bg_val, bg_val) if colour else bg_val
        img = Image.new(mode, (size, size), color=bg)

    elif bg_type == "tinted":
        name = random.choice(list(TINT_OPTIONS.keys()))
        tint = tuple(max(215, min(255, c + random.randint(-8, 8)))
                     for c in TINT_OPTIONS[name])
        img  = Image.new("RGB", (size, size), color=tint)

    elif bg_type == "ruled":
        img  = Image.new("RGB", (size, size), (bg_val, bg_val, bg_val))
        draw = ImageDraw.Draw(img)
        sp   = random.randint(14, 22)
        lc   = (max(175, bg_val - 40),) * 3
        for y in range(sp, size, sp):
            draw.line([(0, y), (size, y)], fill=lc, width=1)

    elif bg_type == "grid":
        img  = Image.new("RGB", (size, size), (bg_val, bg_val, bg_val))
        draw = ImageDraw.Draw(img)
        sp   = random.randint(14, 22)
        lc   = (max(180, bg_val - 32),) * 3
        for y in range(sp, size, sp):
            draw.line([(0, y), (size, y)], fill=lc, width=1)
        for x in range(sp, size, sp):
            draw.line([(x, 0), (x, size)], fill=lc, width=1)

    else:
        bg  = (bg_val, bg_val, bg_val) if colour else bg_val
        img = Image.new(mode, (size, size), color=bg)

    arr = np.array(img, dtype=np.int16)
    arr = np.clip(arr + np.random.normal(0, 2.0, arr.shape).astype(np.int16),
                  0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode=img.mode)
    if not colour and img.mode != "L":
        img = img.convert("L")

    return np.asarray(img, dtype=np.float32) / 255.0


# ── Distortion helpers ──────────────────────────────────────────────────────────

def _elastic_deform(rgba: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
    """Elastic deformation via smooth random displacement fields."""
    if alpha < 0.5:
        return rgba
    H, W = rgba.shape[:2]

    def _smooth(f: np.ndarray) -> np.ndarray:
        norm = ((f.clip(-3, 3) + 3) / 6 * 255).astype(np.uint8)
        sm   = np.array(
            Image.fromarray(norm, mode="L").filter(ImageFilter.GaussianBlur(radius=sigma)),
            dtype=np.float32,
        )
        return sm / 255.0 * 6 - 3

    dx = _smooth(np.random.randn(H, W).astype(np.float32)) * alpha
    dy = _smooth(np.random.randn(H, W).astype(np.float32)) * alpha

    y_g, x_g = np.mgrid[0:H, 0:W].astype(np.float32)
    sx = (x_g + dx).clip(0, W - 1.001)
    sy = (y_g + dy).clip(0, H - 1.001)

    x0 = np.floor(sx).astype(np.int32);  x1 = np.minimum(x0 + 1, W - 1)
    y0 = np.floor(sy).astype(np.int32);  y1 = np.minimum(y0 + 1, H - 1)
    wx = (sx - x0)[..., np.newaxis]
    wy = (sy - y0)[..., np.newaxis]

    return (
        rgba[y0, x0] * (1-wx) * (1-wy) +
        rgba[y0, x1] * wx     * (1-wy) +
        rgba[y1, x0] * (1-wx) * wy     +
        rgba[y1, x1] * wx     * wy
    ).astype(np.float32)


def _localised_blur(rgba: np.ndarray, level: float) -> np.ndarray:
    """
    Blur a random elliptical sub-region — simulates uneven pen pressure.
    The rest of the digit stays sharp.
    """
    if level < 0.08:
        return rgba
    H, W = rgba.shape[:2]
    cx = random.randint(W // 5, 4*W // 5)
    cy = random.randint(H // 5, 4*H // 5)
    rx = random.randint(W // 7, W // 2)
    ry = random.randint(H // 7, H // 2)

    yg, xg = np.mgrid[0:H, 0:W]
    dist     = ((xg - cx) / rx)**2 + ((yg - cy) / ry)**2
    raw_mask = np.where(dist <= 1,
                        level * random.uniform(0.4, 1.0),
                        0.0).astype(np.float32)
    mask = np.array(
        Image.fromarray((raw_mask * 255).astype(np.uint8), mode="L")
             .filter(ImageFilter.GaussianBlur(radius=5)),
        dtype=np.float32,
    ) / 255.0

    br      = 1.5 + level * 3.0
    blurred = np.array(
        Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")
             .filter(ImageFilter.GaussianBlur(radius=br)),
        dtype=np.float32,
    ) / 255.0

    m = mask[..., np.newaxis]
    return (rgba * (1 - m) + blurred * m).astype(np.float32)


# ── Per-number sub-canvas renderer ─────────────────────────────────────────────

def _render_number_rgba(inst: NumberInstance,
                         colour: bool) -> tuple[np.ndarray, int, int]:
    """
    Render one number to a float32 RGBA sub-canvas with all distortions applied.

    Returns (rgba, x_off, y_off) where x_off/y_off place the sub-canvas on
    the main canvas such that inst.x, inst.y lands at the visual centre.
    """
    font_px = FONT_SIZES.get((inst.num_digits, inst.size), 60)
    font    = _load_font(inst.font_path, font_px)
    pad     = max(14, int(font_px * 0.22))

    # Measure each digit individually
    char_ws, char_hs, char_xoffs, char_yoffs = [], [], [], []
    for spec in inst.digit_specs:
        cw, ch, cx, cy = _measure_char(spec.char, font)
        char_ws.append(cw); char_hs.append(ch)
        char_xoffs.append(cx); char_yoffs.append(cy)

    total_w = sum(char_ws)
    total_h = max(char_hs) if char_hs else font_px
    sub_w   = total_w + 2 * pad
    sub_h   = total_h + 2 * pad

    sub  = Image.new("RGBA", (sub_w, sub_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sub)

    x_cur = pad
    for i, spec in enumerate(inst.digit_specs):
        y_pos = pad - char_yoffs[i]
        if colour:
            rgb  = COLORS[spec.color]
            fill: tuple = tuple(max(0, min(255, c + random.randint(-12, 12)))
                                 for c in rgb) + (255,)
        else:
            v    = max(20, min(100, BW_FILL + random.randint(-10, 10)))
            fill = (v, v, v, 255)

        # Drop shadow (colour mode, 25 % chance)
        if colour and random.random() < 0.25:
            shd = tuple(max(0, c - 60) for c in COLORS[spec.color]) + (180,)
            draw.text((x_cur + 2, y_pos + 3), spec.char, font=font, fill=shd)

        draw.text((x_cur - char_xoffs[i], y_pos), spec.char, font=font, fill=fill)
        x_cur += char_ws[i]

    # Distortion pipeline
    rgba = np.array(sub, dtype=np.float32) / 255.0

    if inst.distortion_level > 0.0:
        lv = inst.distortion_level
        alpha = lv * max(sub_w, sub_h) * 0.16
        sigma = 2.5 + lv * 2.5
        rgba  = _elastic_deform(rgba, alpha, sigma)

        angle = lv * random.uniform(-13, 13)
        sub2  = Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")
        sub2  = sub2.rotate(angle, resample=Image.BILINEAR, expand=False)
        rgba  = np.array(sub2, dtype=np.float32) / 255.0

        rgba = _localised_blur(rgba, lv)

    # Final light anti-alias blur
    pil  = Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")
    pil  = pil.filter(ImageFilter.GaussianBlur(radius=0.5))
    rgba = np.array(pil, dtype=np.float32) / 255.0

    content_cx = pad + total_w // 2
    content_cy = pad + total_h // 2
    x_off      = inst.x - content_cx
    y_off      = inst.y - content_cy

    return rgba, x_off, y_off


# ── Alpha compositing ───────────────────────────────────────────────────────────

def _composite(main: np.ndarray,
               rgba: np.ndarray,
               x_off: int, y_off: int,
               colour: bool) -> np.ndarray:
    H_m, W_m = main.shape[:2]
    H_s, W_s = rgba.shape[:2]
    x0m = max(0, x_off);       y0m = max(0, y_off)
    x1m = min(W_m, x_off+W_s); y1m = min(H_m, y_off+H_s)
    x0s = x0m - x_off;         y0s = y0m - y_off
    x1s = x0s + (x1m - x0m);   y1s = y0s + (y1m - y0m)

    if x0m >= x1m or y0m >= y1m:
        return main

    alpha = rgba[y0s:y1s, x0s:x1s, 3:4]
    src   = rgba[y0s:y1s, x0s:x1s, :3]

    if colour:
        main[y0m:y1m, x0m:x1m] = (
            main[y0m:y1m, x0m:x1m] * (1 - alpha) + src * alpha
        )
    else:
        grey = (src[..., 0]*0.299 + src[..., 1]*0.587 + src[..., 2]*0.114)
        a2d  = alpha[..., 0]
        main[y0m:y1m, x0m:x1m] = (
            main[y0m:y1m, x0m:x1m] * (1 - a2d) + grey * a2d
        )
    return main


# ── Full scene renderer ─────────────────────────────────────────────────────────

def render_scene(scene: Scene, size: int = IMG_SIZE) -> np.ndarray:
    """
    Render the full scene to a float32 numpy array in [0, 1].

    Returns (H, W, 3) for colour=True or (H, W) for colour=False.
    """
    main = _render_background(size, scene.background_type, scene.colour)
    for inst in sorted(scene.numbers, key=lambda n: n.z_order):
        rgba, x_off, y_off = _render_number_rgba(inst, scene.colour)
        main = _composite(main, rgba, x_off, y_off, scene.colour)
    return main


# ── Language helpers ────────────────────────────────────────────────────────────

def _sw(size: str) -> str:
    return random.choice(SIZE_WORDS[size])


def _a(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def _dw(n: int) -> str:
    return "digit" if n == 1 else "digits"


def _canvas_region(cx: int, cy: int, size: int = IMG_SIZE) -> str:
    t   = size // 3
    col = "left"   if cx < t      else ("right"  if cx >= 2*t else "centre")
    row = "top"    if cy < t      else ("bottom" if cy >= 2*t else "middle")
    if row == "middle" and col == "centre":
        return random.choice(["the centre", "the middle of the image"])
    elif row == "middle":
        return f"the middle-{col}"
    elif col == "centre":
        return f"the {row}-centre"
    else:
        return f"the {row}-{col}"


def _color_inline(inst: NumberInstance, colour: bool) -> str | None:
    """Returns colour adjective for single-colour numbers, None for multi-colour."""
    if not colour or not inst.all_same_color:
        return None
    return inst.main_color


def _color_suffix(inst: NumberInstance, colour: bool) -> str:
    """Returns per-digit colour suffix for multi-colour numbers, empty string otherwise."""
    if not colour or inst.all_same_color:
        return ""
    parts = [f"{d.char} in {d.color}" for d in inst.digit_specs]
    joined = ", ".join(parts[:-1]) + " and " + parts[-1]
    return f", with {joined}"


def _np_inst(inst: NumberInstance, colour: bool,
             include_pos: bool = False,
             include_color_suffix: bool = False) -> str:
    sw  = _sw(inst.size)
    col = _color_inline(inst, colour)
    val = str(inst.value)
    base = (f"{_a(sw)} {sw} {col} {val}" if col
            else f"{_a(sw)} {sw} {val}")
    if include_color_suffix:
        base += _color_suffix(inst, colour)
    if include_pos:
        base += f" at {_canvas_region(inst.x, inst.y)}"
    return base


# ── Sentence generation ─────────────────────────────────────────────────────────

def describe(scene: Scene) -> str:
    """Return a varied English sentence describing the full scene."""
    colour  = scene.colour
    nums    = scene.numbers
    n       = len(nums)
    has_ov  = scene.has_overlap

    # ── 1 number ──────────────────────────────────────────────────────────────
    if n == 1:
        inst = nums[0]
        sw   = _sw(inst.size)
        col  = _color_inline(inst, colour)
        suf  = _color_suffix(inst, colour)   # "" for single-colour
        val  = str(inst.value)
        dw   = _dw(inst.num_digits)
        be   = "is" if inst.num_digits == 1 else "are"

        if colour:
            if col:
                # single-colour templates
                t = [
                    f"{_a(sw).capitalize()} {sw} {col} number {val}.",
                    f"The {dw} {val}, {sw} and {col}.",
                    f"{_a(sw).capitalize()} {sw} {col} {val}.",
                    f"{sw.capitalize()} {col} {val}.",
                    f"The {dw} {val} {be} {sw} and {col}.",
                    f"{val}, written {sw} in {col}.",
                    f"In {sw} {col}: {val}.",
                    f"A {col}, {sw} number: {val}.",
                    f"Number {val} \u2014 {sw}, {col}.",
                    f"Here is {_a(sw)} {sw} {col} {val}.",
                ]
            else:
                # multi-colour templates — colour description after value
                t = [
                    f"{_a(sw).capitalize()} {sw} number {val}{suf}.",
                    f"{_a(sw).capitalize()} {sw} {val}{suf}.",
                    f"{sw.capitalize()} {val}{suf}.",
                    f"The {dw} {val} {be} {sw}{suf}.",
                    f"{val}, written {sw}{suf}.",
                    f"Number {val} \u2014 {sw}{suf}.",
                    f"Here is {_a(sw)} {sw} {val}{suf}.",
                ]
        else:
            t = [
                f"{_a(sw).capitalize()} {sw} number {val}.",
                f"The {dw} {val}, {sw}.",
                f"{_a(sw).capitalize()} {sw} {val}.",
                f"{sw.capitalize()} {val}.",
                f"The {dw} {val} {be} {sw}.",
                f"{val}, written {sw}.",
                f"A {sw} number: {val}.",
                f"Number {val} \u2014 {sw}.",
                f"Here is {_a(sw)} {sw} {val}.",
            ]
        return random.choice(t)

    # ── 2 numbers ─────────────────────────────────────────────────────────────
    if n == 2:
        a, b = nums[0], nums[1]
        # helper: noun phrase with colour suffix included
        def _np2(inst: NumberInstance, pos: bool = False) -> str:
            return _np_inst(inst, colour, include_pos=pos, include_color_suffix=True)

        if has_ov:
            front, back = (a, b) if a.z_order > b.z_order else (b, a)
            ov_phrs = [
                f"{_np2(front)} partially overlapping {_np2(back)}.",
                f"{_np2(front)} overlapping {_np2(back)}.",
                f"{_np2(front)} in front of {_np2(back)}.",
                f"{_np2(back).capitalize()} with {_np2(front)} overlapping it.",
            ]
            ov = random.choice(ov_phrs).capitalize()

        t = [
            lambda: f"{_np2(a, True).capitalize()} and {_np2(b, True)} appear in the scene.",
            lambda: f"Two numbers are visible: {_np2(a, True)} and {_np2(b, True)}.",
            lambda: f"The scene shows {_np2(a)} and {_np2(b)}.",
            lambda: f"{_np2(a, True).capitalize()}; also {_np2(b, True)}.",
            lambda: (ov if has_ov else
                     f"{_np2(a, True).capitalize()} alongside {_np2(b, True)}."),
            lambda: f"Here we see {_np2(a)} and {_np2(b)}.",
        ]
        return random.choice(t)()

    # ── 3 or 4 numbers ────────────────────────────────────────────────────────
    word   = {3: "Three", 4: "Four"}.get(n, f"{n}")
    labels = [_np_inst(inst, colour, include_color_suffix=True) for inst in nums]

    if n == 3:
        listing = f"{labels[0]}, {labels[1]}, and {labels[2]}"
    else:
        listing = f"{labels[0]}, {labels[1]}, {labels[2]}, and {labels[3]}"

    # Most prominent overlap pair
    best_iou, best_pair = 0.0, None
    for i, x in enumerate(nums):
        for y in nums[i+1:]:
            iou = _bbox_iou(x.bbox, y.bbox)
            if iou > best_iou:
                best_iou, best_pair = iou, (x, y)

    ov_clause = ""
    if has_ov and best_pair:
        fx, bx = (best_pair if best_pair[0].z_order > best_pair[1].z_order
                  else (best_pair[1], best_pair[0]))
        ov_clause = (f" {_np_inst(fx, colour, include_color_suffix=True).capitalize()} overlaps "
                     f"{_np_inst(bx, colour, include_color_suffix=True)}.")

    t = [
        f"{word} numbers are visible: {listing}.",
        f"{word} numbers appear in the scene: {listing}.{ov_clause}",
        f"The scene contains {listing}.",
        f"In total, {word.lower()} numbers: {listing}.{ov_clause}",
    ]
    return random.choice(t)


# ── Scene generator ─────────────────────────────────────────────────────────────

def random_scene(num_numbers:      int | None   = None,
                 crowding:         str | None   = None,
                 distortion_level: float | None = None,
                 background_type:  str | None   = None,
                 colour:           bool | None  = None,
                 img_size:         int           = IMG_SIZE) -> Scene:
    """
    Generate a fully-specified random Scene.  All arguments default to a
    random choice if not given.
    """
    if crowding is None:
        crowding = random.choice(CROWDING_MODES)
    if num_numbers is None:
        lo, hi_ = CROWDING_COUNT[crowding]
        num_numbers = random.randint(lo, hi_)
    if distortion_level is None:
        distortion_level = random.uniform(0.0, 1.0)
    if background_type is None:
        background_type = random.choice(BACKGROUND_TYPES)
    if colour is None:
        colour = random.choice([True, False])

    z_orders = list(range(num_numbers))
    random.shuffle(z_orders)

    instances: list[NumberInstance] = []
    for i in range(num_numbers):
        dl    = random.choice(DIGIT_LENGTHS)
        size  = random.choice(SIZES)
        lo_v  = 10 ** (dl - 1) if dl > 1 else 0
        hi_v  = 10 ** dl - 1
        value = random.randint(lo_v, hi_v)

        digit_colors = _assign_digit_colors(dl)
        digit_specs  = [DigitSpec(char=c, color=col)
                        for c, col in zip(str(value), digit_colors)]

        instances.append(NumberInstance(
            value            = value,
            digit_specs      = digit_specs,
            size             = size,
            font_path        = _pick_font_path(),
            z_order          = z_orders[i],
            distortion_level = distortion_level,
        ))

    instances = _layout_numbers(instances, crowding, img_size)
    _enforce_overlap_contrast(instances)

    return Scene(
        numbers          = instances,
        background_type  = background_type,
        crowding         = crowding,
        colour           = colour,
        distortion_level = distortion_level,
    )


# ── File saving ─────────────────────────────────────────────────────────────────

def save_sample(n:        int,
                scene:    Scene,
                sentence: str,
                split:    str = "train",
                prefix:   str = "sample",
                size:     int = IMG_SIZE) -> None:
    """
    Save four files for sample n and append one manifest row.

    .mat convention (mirrors shapes generator):
        colour    → (H×3, W)   reshape to (H, W, 3)
        greyscale → (H, W)
    """
    _ensure_dirs()
    _init_manifest()

    arr      = render_scene(scene, size=size)
    filename = f"{prefix}_{n:06d}"

    (_dir(split, "text") / f"{filename}.txt").write_text(
        sentence + "\n", encoding="utf-8")

    occ = scene.occlusion_pcts()
    with open(_dir(split, "meta") / f"{filename}.meta", "w", encoding="utf-8") as f:
        f.write(sentence + "\n")
        f.write(scene.to_notation() + "\n")
        f.write(
            f"num_numbers={len(scene.numbers)}  bg={scene.background_type}  "
            f"crowding={scene.crowding}  distortion={scene.distortion_level:.2f}  "
            f"colour={'yes' if scene.colour else 'no'}  "
            f"has_overlap={scene.has_overlap}  ambiguous={scene.ambiguous}\n"
        )
        for i, inst in enumerate(scene.numbers):
            f.write(
                f"  [{i}] {inst.value}  color={inst.color_notation()}  "
                f"size={inst.size}  z={inst.z_order}  "
                f"pos=({inst.x},{inst.y})  occ={occ[i]:.3f}\n"
            )

    pil_mode = "RGB" if scene.colour else "L"
    Image.fromarray((arr * 255).astype(np.uint8), mode=pil_mode).save(
        _dir(split, "images") / f"{filename}.png")

    mat = arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr
    np.savetxt(_dir(split, "matrices") / f"{filename}.mat", mat, fmt="%.4f")

    nums = scene.numbers
    _append_manifest({
        "filename":        f"{split}/images/{filename}.png",
        "notation":        scene.to_notation(),
        "num_numbers":     len(nums),
        "values":          ",".join(str(i_.value) for i_ in nums),
        "colors":          ",".join(i_.color_notation() for i_ in nums),
        "sizes":           ",".join(i_.size for i_ in nums),
        "num_digits_list": ",".join(str(i_.num_digits) for i_ in nums),
        "background_type": scene.background_type,
        "crowding":        scene.crowding,
        "distortion_level": round(scene.distortion_level, 3),
        "colour":          scene.colour,
        "has_overlap":     scene.has_overlap,
        "occlusion_pcts":  ",".join(f"{p:.3f}" for p in occ),
        "ambiguous":       scene.ambiguous,
        "split":           split,
    })


# ── Main generation loop ────────────────────────────────────────────────────────

if __name__ == "__main__":
    TOTAL    = 5000   # ← edit to generate more / fewer samples
    IMG_SIZE = 256    # ← raise to 512 for more detail

    # Balance across:
    #   crowding         ∈ {low, medium, high}
    #   distortion_level ∈ {0.0, 0.35, 0.70, 1.0}
    #   colour           ∈ {True, False}
    # background_type randomised per sample for variety
    DIST_LEVELS = [0.0, 0.35, 0.70, 1.0]
    COMBOS      = list(itertools.product(CROWDING_MODES, DIST_LEVELS, [True, False]))
    per_combo   = max(1, TOTAL // len(COMBOS))   # 24 combos → ~41 each

    print(f"=== Generating {TOTAL} number image samples (v2) ===")
    print(f"    {len(COMBOS)} param combos  ×  ~{per_combo} each\n")

    n = 0
    for crowding, dist_level, colour in tqdm(COMBOS):
        for i in range(per_combo):
            if n >= TOTAL:
                break
            r     = i / per_combo
            split = "train" if r < 0.70 else "val" if r < 0.85 else "test"

            scene    = random_scene(crowding=crowding,
                                    distortion_level=dist_level,
                                    colour=colour,
                                    img_size=IMG_SIZE)
            sentence = describe(scene)
            save_sample(n, scene, sentence, split=split, size=IMG_SIZE)

            # if n % 100 == 0:
            #     tag  = f"{'RGB' if colour else 'BW'}/{crowding}/d={dist_level:.2f}"
            #     vals = "+".join(str(inst.value) for inst in scene.numbers)
            #     print(f"  [{n:>4}/{TOTAL}] {split:<5} | {tag} | {vals}")
            n += 1
        if n >= TOTAL:
            break

    # Top-up to exactly TOTAL
    while n < TOTAL:
        r     = (n % max(per_combo, 1)) / max(per_combo, 1)
        split = "train" if r < 0.70 else "val" if r < 0.85 else "test"
        scene    = random_scene(colour=True, img_size=IMG_SIZE)
        sentence = describe(scene)
        save_sample(n, scene, sentence, split=split, size=IMG_SIZE)
        n += 1

    # print(f"\nDone.  {n} samples saved to {BASE_DIR}")
    counts = {s: len(list((BASE_DIR / s / "images").glob("*.png")))
              for s in SPLITS}
    for s, c in counts.items():
        print(f"  {s:<6}: {c}")