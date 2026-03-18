# -*- coding: utf-8 -*-
"""
numbers.py — Synthetic data generator for number image descriptions.

Generates matched (image, English sentence) pairs for neural-network training.
Each sample renders a single number (1–4 digits) in a chosen size and colour.

Example sentences:
    "A large blue number 67."
    "The digits 1337, small and red."
    "A tiny green 42."
    "In sizeable orange: 5."
    "The digit 9 is big and purple."

Parameters exposed in the generation loop (all can be overridden):
    digit_length : 1 | 2 | 3 | 4  — how many digits the number has
    size         : "small" | "medium" | "large"
    colour       : bool — True → RGB image; False → greyscale image

Output layout (identical to shapes generator):
    Numbers_Data/
        manifest.csv
        train/ val/ test/
            text/ meta/ images/ matrices/

manifest.csv columns:
    filename, notation, value, num_digits, color, size, colour, split

Usage:
    python numbers.py          # 1 000 samples, balanced across all combos
    Change TOTAL near the bottom to generate more or fewer samples.
"""

import random
import itertools
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pathlib import Path
import csv


# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).resolve().parent / "Numbers_Data"
SPLITS          = ["train", "val", "test"]
SUBDIRS         = ["text", "meta", "images", "matrices"]
MANIFEST_PATH   = BASE_DIR / "manifest.csv"
MANIFEST_FIELDS = [
    "filename", "notation", "value", "num_digits",
    "color", "size", "colour", "split",
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

SIZES      = ["small", "medium", "large"]
SIZE_WORDS = {
    "small":  ["small",        "tiny",         "little"],
    "medium": ["medium-sized", "medium",        "mid-sized"],
    "large":  ["large",        "big",           "sizeable"],
}

DIGIT_LENGTHS = [1, 2, 3, 4]

# B&W rendering constants (mirrors shapes.py)
BW_BG   = 242   # near-white background
BW_FILL = 55    # dark ink

IMG_SIZE = 256


# ── Font loading ───────────────────────────────────────────────────────────────

# Several bold sans-serif fonts for visual variety across samples
_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf",
    # macOS / Windows fallbacks
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arialbd.ttf",
]
_AVAILABLE_FONTS: list[str] = [p for p in _FONT_PATHS if Path(p).exists()]


def _load_font(px: int) -> ImageFont.FreeTypeFont:
    """Load a random available bold font at the requested pixel size."""
    if _AVAILABLE_FONTS:
        path = random.choice(_AVAILABLE_FONTS)
        try:
            return ImageFont.truetype(path, px)
        except (IOError, OSError):
            pass
    # Pillow ≥ 10.0 supports size=
    try:
        return ImageFont.load_default(size=px)   # type: ignore[call-arg]
    except TypeError:
        return ImageFont.load_default()           # type: ignore[return-value]


# Font-size table: (digit_length, size) → px
# Tuned so numbers fill the 256 × 256 canvas attractively.
FONT_SIZES: dict[tuple[int, str], int] = {
    (1, "small"): 52,  (1, "medium"): 88,  (1, "large"): 126,
    (2, "small"): 46,  (2, "medium"): 76,  (2, "large"): 108,
    (3, "small"): 34,  (3, "medium"): 56,  (3, "large"): 82,
    (4, "small"): 26,  (4, "medium"): 44,  (4, "large"): 66,
}


# ── Data class ──────────────────────────────────────────────────────────────────

@dataclass
class NumberSample:
    value: int
    color: str
    size:  str

    @property
    def num_digits(self) -> int:
        return len(str(self.value))

    def to_notation(self) -> str:
        """Compact notation: <value>:<color>,<size>  e.g. '67:blue,large'"""
        return f"{self.value}:{self.color},{self.size}"


# ── Language helpers ────────────────────────────────────────────────────────────

def _a(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def _sw(size: str) -> str:
    return random.choice(SIZE_WORDS[size])


def _digit_word(n: int) -> str:
    """'digit' for 1, 'digits' for 2–4."""
    return "digit" if n == 1 else "digits"


def describe(sample: NumberSample, colour: bool = True) -> str:
    """
    Return a varied English sentence describing the sample.

    colour=False → colour adjectives are omitted (greyscale images have no
    colour information the network can use).
    """
    sw  = _sw(sample.size)
    val = str(sample.value)
    col = sample.color
    dw  = _digit_word(sample.num_digits)
    be  = "is" if sample.num_digits == 1 else "are"

    if colour:
        templates = [
            # "A large blue number 67."
            lambda: f"{_a(sw).capitalize()} {sw} {col} number {val}.",
            # "The digits 1337, small and red."
            lambda: f"The {dw} {val}, {sw} and {col}.",
            # "A tiny red 1337."
            lambda: f"{_a(sw).capitalize()} {sw} {col} {val}.",
            # "Large blue 67."
            lambda: f"{sw.capitalize()} {col} {val}.",
            # "The digit 9 is big and purple."
            lambda: f"The {dw} {val} {be} {sw} and {col}.",
            # "67, written large in blue."
            lambda: f"{val}, written {sw} in {col}.",
            # "In sizeable orange: 42."
            lambda: f"In {sw} {col}: {val}.",
            # "A blue, large-sized number: 1337."
            lambda: f"A {col}, {sw} number: {val}.",
            # "Number 67 — large, blue."
            lambda: f"Number {val} \u2014 {sw}, {col}.",
            # "Here is a small red 5."
            lambda: f"Here is {_a(sw)} {sw} {col} {val}.",
        ]
    else:
        templates = [
            lambda: f"{_a(sw).capitalize()} {sw} number {val}.",
            lambda: f"The {dw} {val}, {sw}.",
            lambda: f"{_a(sw).capitalize()} {sw} {val}.",
            lambda: f"{sw.capitalize()} {val}.",
            lambda: f"The {dw} {val} {be} {sw}.",
            lambda: f"{val}, written {sw}.",
            lambda: f"A {sw} number: {val}.",
            lambda: f"Number {val} \u2014 {sw}.",
            lambda: f"Here is {_a(sw)} {sw} {val}.",
        ]

    return random.choice(templates)()


# ── Scene generator ─────────────────────────────────────────────────────────────

def random_sample(digit_length: int | None = None,
                  size:         str | None = None) -> NumberSample:
    """
    Generate a random NumberSample.

    Args:
        digit_length: 1–4.  Random if None.
        size:         "small" | "medium" | "large".  Random if None.

    The number value is drawn uniformly from all integers with exactly
    digit_length digits (10^(n-1) … 10^n − 1; 0–9 for n=1).
    Colour is always random.
    """
    if digit_length is None:
        digit_length = random.choice(DIGIT_LENGTHS)
    if size is None:
        size = random.choice(SIZES)

    lo    = 10 ** (digit_length - 1) if digit_length > 1 else 0
    hi    = 10 ** digit_length - 1
    value = random.randint(lo, hi)
    color = random.choice(COLOR_NAMES)

    return NumberSample(value=value, color=color, size=size)


# ── Rendering ───────────────────────────────────────────────────────────────────

def render_image(sample: NumberSample,
                 size:   int  = IMG_SIZE,
                 colour: bool = True) -> np.ndarray:
    """
    Render sample to a numpy float32 array in [0, 1].

    Returns:
        colour=True  → (H, W, 3)  RGB
        colour=False → (H, W)     greyscale
    """
    # ── Background ──────────────────────────────────────────────────────────────
    bg_val = random.randint(234, 248)
    mode   = "RGB" if colour else "L"
    bg     = (bg_val, bg_val, bg_val) if colour else bg_val
    img    = Image.new(mode, (size, size), color=bg)

    # Low-amplitude Gaussian noise (simulates paper texture)
    arr = np.array(img, dtype=np.int16)
    arr = np.clip(
        arr + np.random.normal(0, 2.0, arr.shape).astype(np.int16),
        0, 255,
    ).astype(np.uint8)
    img  = Image.fromarray(arr, mode=mode)
    draw = ImageDraw.Draw(img)

    # ── Font & text colour ───────────────────────────────────────────────────────
    font_px   = FONT_SIZES.get((sample.num_digits, sample.size), 60)
    font      = _load_font(font_px)
    text      = str(sample.value)

    if colour:
        rgb        = COLORS[sample.color]
        # Tiny random tint variation for realism
        text_color: tuple[int, ...] | int = tuple(
            max(0, min(255, c + random.randint(-10, 10))) for c in rgb
        )
    else:
        text_color = max(20, min(90, BW_FILL + random.randint(-8, 8)))

    # ── Layout: centre the text with slight jitter ───────────────────────────────
    bbox = draw.textbbox((0, 0), text, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]
    jx   = random.randint(-6, 6)
    jy   = random.randint(-6, 6)
    tx   = (size - tw) // 2 - bbox[0] + jx
    ty   = (size - th) // 2 - bbox[1] + jy

    # ── Optional drop-shadow (colour mode, 30 % of samples) ─────────────────────
    if colour and random.random() < 0.30:
        rgb_base = COLORS[sample.color]
        shadow   = tuple(max(0, c - 55) for c in rgb_base)
        draw.text((tx + 2, ty + 3), text, font=font, fill=shadow)

    draw.text((tx, ty), text, font=font, fill=text_color)

    # Light anti-aliasing blur (mirrors shapes.py)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.55))

    return np.asarray(img, dtype=np.float32) / 255.0


# ── File saving ─────────────────────────────────────────────────────────────────

def save_sample(n:        int,
                sample:   NumberSample,
                sentence: str,
                colour:   bool,
                split:    str = "train",
                prefix:   str = "sample") -> None:
    """
    Save four files for sample n and append one manifest row.

    Files written:
        {split}/text/{prefix}_{n:06d}.txt      ← English sentence
        {split}/meta/{prefix}_{n:06d}.meta     ← sentence + notation + info
        {split}/images/{prefix}_{n:06d}.png    ← rendered PNG
        {split}/matrices/{prefix}_{n:06d}.mat  ← flat float matrix

    .mat convention (mirrors shapes generator):
        colour  → shape (H×3, W),  reshape to (H, W, 3) to use
        BW      → shape (H, W)
    """
    _ensure_dirs()
    _init_manifest()

    arr      = render_image(sample, colour=colour)
    filename = f"{prefix}_{n:06d}"

    # Text
    (_dir(split, "text") / f"{filename}.txt").write_text(
        sentence + "\n", encoding="utf-8")

    # Meta
    with open(_dir(split, "meta") / f"{filename}.meta", "w", encoding="utf-8") as f:
        f.write(sentence + "\n")
        f.write(sample.to_notation() + "\n")
        f.write(
            f"value={sample.value}  digits={sample.num_digits}  "
            f"color={sample.color}  size={sample.size}  colour={'yes' if colour else 'no'}\n"
        )

    # PNG
    pil_mode = "RGB" if colour else "L"
    Image.fromarray((arr * 255).astype(np.uint8), mode=pil_mode).save(
        _dir(split, "images") / f"{filename}.png")

    # .mat  (same stacking convention as shapes.py)
    mat = arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr
    np.savetxt(_dir(split, "matrices") / f"{filename}.mat", mat, fmt="%.4f")

    # Manifest row
    _append_manifest({
        "filename":   f"{split}/images/{filename}.png",
        "notation":   sample.to_notation(),
        "value":      sample.value,
        "num_digits": sample.num_digits,
        "color":      sample.color,
        "size":       sample.size,
        "colour":     colour,
        "split":      split,
    })


# ── Main generation loop ────────────────────────────────────────────────────────

if __name__ == "__main__":
    TOTAL    = 1000    # ← edit this to generate more / fewer samples
    IMG_SIZE = 256     # ← raise to 512 for more detail

    # Balance across all three axes:
    #   digit_length ∈ {1, 2, 3, 4}
    #   size         ∈ {"small", "medium", "large"}
    #   colour       ∈ {True, False}
    COMBOS    = list(itertools.product(DIGIT_LENGTHS, SIZES, [True, False]))
    per_combo = max(1, TOTAL // len(COMBOS))   # 24 combos → ~41 each

    print(f"=== Generating {TOTAL} number image samples ===")
    print(f"    {len(COMBOS)} param combos  ×  ~{per_combo} each\n")

    n = 0
    for digit_length, size, colour in COMBOS:
        for i in range(per_combo):
            if n >= TOTAL:
                break
            r     = i / per_combo
            split = "train" if r < 0.70 else "val" if r < 0.85 else "test"

            sample   = random_sample(digit_length=digit_length, size=size)
            sentence = describe(sample, colour=colour)
            save_sample(n, sample, sentence, colour=colour, split=split)

            if n % 100 == 0:
                tag = f"{'RGB' if colour else 'BW'}/{size}/{digit_length}d"
                print(f"  [{n:>4}/{TOTAL}] split={split} | {tag} "
                      f"| {sample.value} ({sample.color})")
            n += 1
        if n >= TOTAL:
            break

    # Top-up to reach exactly TOTAL (avoids rounding shortfall)
    while n < TOTAL:
        r     = (n % max(per_combo, 1)) / max(per_combo, 1)
        split = "train" if r < 0.70 else "val" if r < 0.85 else "test"
        sample   = random_sample()
        sentence = describe(sample, colour=True)
        save_sample(n, sample, sentence, colour=True, split=split)
        n += 1

    print(f"\nDone.  {n} samples saved to {BASE_DIR}")
    counts = {s: len(list((BASE_DIR / s / "images").glob("*.png")))
              for s in SPLITS}
    for s, c in counts.items():
        print(f"  {s:<6}: {c}")