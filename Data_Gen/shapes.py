# -*- coding: utf-8 -*-
"""
shapes.py — Synthetic data generator for geometric shape arrangement descriptions.

Generates matched (image, English sentence) pairs for neural-network training.

NEW IN v2:
  • 2-D shapes  (circle, square, triangle, rectangle, pentagon, star)
  • 3-D shapes  (sphere, cube, cylinder, cone, pyramid)  — controlled via shape_mode
  • Containment / overlap — one shape drawn inside another in the same cell
  • Black-and-white images in addition to colour RGB images

Parameters exposed in the generation loop (all can be overridden):
  shape_mode   : "2d" | "3d" | None  (None → random 50/50 per scene)
  allow_overlap: bool   — if True, one cell in the scene gets a containment pair
  colour       : bool   — True → RGB image; False → greyscale image

Output layout (identical to TicTacToe generator):
  Shapes_Data/
    manifest.csv
    train/ val/ test/
      text/ meta/ images/ matrices/

manifest.csv columns:
  filename, notation, num_objects, shapes, colors,
  shape_mode, colour, has_overlap, split

Usage:
    python shapes.py          # 1 000 samples, balanced across all parameter combos
    Change TOTAL near the bottom to generate more or fewer samples.
"""

import math
import random
import itertools
from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
import csv
from tqdm import tqdm


# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).resolve().parent / "Shapes_Data"
SPLITS          = ["train", "val", "test"]
SUBDIRS         = ["text", "meta", "images", "matrices"]
MANIFEST_PATH   = BASE_DIR / "manifest.csv"
MANIFEST_FIELDS = [
    "filename", "notation", "num_objects", "shapes", "colors",
    "shape_mode", "colour", "has_overlap", "split",
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

SHAPES_2D = ["circle", "square", "triangle", "rectangle", "pentagon", "star"]
SHAPES_3D = ["sphere", "cube", "cylinder", "cone", "pyramid"]
ALL_SHAPES = SHAPES_2D + SHAPES_3D

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
SIZE_SCALE = {"small": 0.30, "medium": 0.48, "large": 0.66}

POSITIONS: dict[str, tuple[int, int]] = {
    "TL": (0, 0), "TM": (0, 1), "TR": (0, 2),
    "ML": (1, 0), "C":  (1, 1), "MR": (1, 2),
    "BL": (2, 0), "BM": (2, 1), "BR": (2, 2),
}

POSITION_PHRASES: dict[str, list[str]] = {
    "TL": ["top-left",      "upper-left corner"],
    "TM": ["top-centre",    "top-middle"],
    "TR": ["top-right",     "upper-right corner"],
    "ML": ["middle-left",   "left side"],
    "C":  ["centre",        "middle of the scene"],
    "MR": ["middle-right",  "right side"],
    "BL": ["bottom-left",   "lower-left corner"],
    "BM": ["bottom-centre", "bottom-middle"],
    "BR": ["bottom-right",  "lower-right corner"],
}

SIZE_WORDS: dict[str, list[str]] = {
    "small":  ["small",        "tiny",         "little"],
    "medium": ["medium-sized", "medium",       "mid-sized"],
    "large":  ["large",        "big",          "sizeable"],
}

# Greyscale rendering constants
BW_BG           = 242   # near-white background
BW_FILL         = 65    # dark ink (outer shapes)
BW_OUTLINE      = 30
BW_INNER_FILL   = 195   # light fill (inner / contained shapes)
BW_INNER_OUTLINE = 30


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class SceneObject:
    shape:      str
    color:      str
    size:       str
    position:   str
    shape_mode: str                                # "2d" or "3d"
    inside:     Optional["SceneObject"] = field(default=None, repr=False)

    def token(self) -> str:
        base = f"{self.shape}:{self.color},{self.size},{self.position},{self.shape_mode}"
        if self.inside is not None:
            base += f">{self.inside.token()}"
        return base


@dataclass
class Scene:
    objects: list[SceneObject]

    @property
    def all_objects(self) -> list[SceneObject]:
        result = []
        for o in self.objects:
            result.append(o)
            if o.inside is not None:
                result.append(o.inside)
        return result

    @property
    def has_overlap(self) -> bool:
        return any(o.inside is not None for o in self.objects)

    @property
    def shape_mode(self) -> str:
        modes = {o.shape_mode for o in self.all_objects}
        return modes.pop() if len(modes) == 1 else "mixed"

    def to_notation(self) -> str:
        return " | ".join(o.token() for o in self.objects)

    def ascii_grid(self) -> str:
        cell_w   = 14
        sep_line = "─" * (cell_w * 3 + 4)
        cell_map: dict[str, str] = {}
        for o in self.objects:
            abbr = f"{o.color[:3].upper()}/{o.shape[:3].upper()}"
            if o.inside is not None:
                abbr += f"[{o.inside.shape[:3].upper()}]"
            cell_map[o.position] = abbr
        row_keys = [["TL", "TM", "TR"], ["ML", "C", "MR"], ["BL", "BM", "BR"]]
        lines    = []
        for ri, row in enumerate(row_keys):
            cells = [cell_map.get(pos, "---").center(cell_w) for pos in row]
            lines.append(" | ".join(cells))
            if ri < 2:
                lines.append(sep_line)
        return "\n".join(lines)


# ── Spatial / language helpers ─────────────────────────────────────────────────

def _relation(a: SceneObject, b: SceneObject) -> str:
    ra, ca = POSITIONS[a.position]
    rb, cb = POSITIONS[b.position]
    dr, dc = ra - rb, ca - cb
    v = {-2: "well above",          -1: "above",
          1: "below",                2: "well below"}.get(dr)
    h = {-2: "well to the left of", -1: "to the left of",
          1: "to the right of",      2: "well to the right of"}.get(dc)
    if v and h:
        return f"{v} and {h}"
    return v or h or "adjacent to"

def _pos_phrase(pos: str) -> str:
    return random.choice(POSITION_PHRASES[pos])

def _a(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"

def _np(obj: SceneObject,
        definite:   bool = False,
        omit_size:  bool = False,
        omit_color: bool = False,
        colour:     bool = True) -> str:
    """
    Build a noun phrase for obj.
    When colour=False the colour adjective is always omitted.
    """
    if not colour:
        omit_color = True
    sw  = random.choice(SIZE_WORDS[obj.size])
    det = "the" if definite else (_a(obj.color) if omit_size else _a(sw))
    parts = [det]
    if not omit_size:
        parts.append(sw)
    if not omit_color:
        parts.append(obj.color)
    parts.append(obj.shape)
    return " ".join(parts)


# ── Sentence templates ─────────────────────────────────────────────────────────

def _color_shape(obj: SceneObject, colour: bool) -> str:
    """'red sphere'  or just 'sphere' for B&W."""
    return f"{obj.color} {obj.shape}" if colour else obj.shape

def _containment_phrase(outer: SceneObject, inner: SceneObject,
                        colour: bool = True) -> str:
    templates = [
        lambda: f"{_np(inner, colour=colour)} inside {_np(outer, colour=colour)}",
        lambda: f"{_np(inner, colour=colour)} enclosed within {_np(outer, colour=colour)}",
        lambda: f"{_np(outer, colour=colour)} containing {_np(inner, colour=colour)}",
        lambda: f"{_np(inner, colour=colour)} nested inside {_np(outer, colour=colour)}",
        lambda: (f"inside the {_color_shape(outer, colour)} "
                 f"sits {_np(inner, colour=colour)}"),
        lambda: (f"{_np(outer, colour=colour)} with {_np(inner, colour=colour)} "
                 f"inside it"),
    ]
    return random.choice(templates)()


def _describe_2(a: SceneObject, b: SceneObject, colour: bool = True) -> str:
    rel_ab = _relation(a, b)
    rel_ba = _relation(b, a)
    templates = [
        lambda: f"{_np(a, colour=colour).capitalize()} is {rel_ab} {_np(b, colour=colour)}.",
        lambda: f"{_np(b, colour=colour).capitalize()} is {rel_ba} {_np(a, colour=colour)}.",
        lambda: f"There is {_np(a, colour=colour)} {rel_ab} {_np(b, colour=colour)}.",
        lambda: f"{_np(a, colour=colour).capitalize()} sits {rel_ab} {_np(b, colour=colour)}.",
        lambda: f"{_np(a, colour=colour).capitalize()} can be seen {rel_ab} {_np(b, colour=colour)}.",
        lambda: (f"{_np(a, colour=colour).capitalize()} occupies the "
                 f"{_pos_phrase(a.position)}, while {_np(b, colour=colour)} "
                 f"is at the {_pos_phrase(b.position)}."),
        lambda: (f"{_np(a, colour=colour).capitalize()} and {_np(b, colour=colour)} "
                 f"appear in the scene; {_np(a, colour=colour, definite=True)} is "
                 f"{random.choice(SIZE_WORDS[a.size])} and rests "
                 f"{rel_ab} {_np(b, colour=colour, definite=True)}."),
        lambda: (f"Two shapes are visible: {_np(a, colour=colour)} at the "
                 f"{_pos_phrase(a.position)} and {_np(b, colour=colour)} "
                 f"at the {_pos_phrase(b.position)}."),
        lambda: (f"{_np(a, colour=colour).capitalize()} is positioned {rel_ab} "
                 f"{_np(b, colour=colour)} on a plain background."),
        lambda: (f"Where is {_np(a, colour=colour, definite=True)}? "
                 f"It is {rel_ab} {_np(b, colour=colour, definite=True)}."),
    ]
    return random.choice(templates)()


def _describe_3(a: SceneObject, b: SceneObject, c: SceneObject,
                colour: bool = True) -> str:
    rel_ab = _relation(a, b)
    rel_cb = _relation(c, b)
    rel_ac = _relation(a, c)
    templates = [
        lambda: (f"Three shapes are arranged in the scene: "
                 f"{_np(a, colour=colour)}, {_np(b, colour=colour)}, "
                 f"and {_np(c, colour=colour)}."),
        lambda: (f"{_np(a, colour=colour).capitalize()} is {rel_ab} "
                 f"{_np(b, colour=colour)}, while {_np(c, colour=colour)} "
                 f"is {rel_cb} {_np(b, colour=colour, definite=True)}."),
        lambda: (f"The scene shows {_np(a, colour=colour)} at the "
                 f"{_pos_phrase(a.position)}, {_np(b, colour=colour)} at the "
                 f"{_pos_phrase(b.position)}, and {_np(c, colour=colour)} "
                 f"at the {_pos_phrase(c.position)}."),
        lambda: (f"{_np(a, colour=colour).capitalize()} is {rel_ac} "
                 f"{_np(c, colour=colour)}; also visible is "
                 f"{_np(b, colour=colour)} at the {_pos_phrase(b.position)}."),
        lambda: _sorted_desc([a, b, c], axis=1, direction="left to right", colour=colour),
        lambda: _sorted_desc([a, b, c], axis=0, direction="top to bottom", colour=colour),
        lambda: (f"{_np(a,colour=colour).capitalize()}, "
                 f"{_np(b,colour=colour)}, and {_np(c,colour=colour)} are present; "
                 f"{_np(a,colour=colour,definite=True)} is {rel_ab} "
                 f"{_np(b,colour=colour,definite=True)}."),
    ]
    return random.choice(templates)()


def _describe_4(objects: list[SceneObject], colour: bool = True) -> str:
    a, b, c, d = objects
    templates = [
        lambda: (f"Four shapes populate the scene: {_np(a, colour=colour)}, "
                 f"{_np(b, colour=colour)}, {_np(c, colour=colour)}, "
                 f"and {_np(d, colour=colour)}."),
        lambda: (f"{_np(a, colour=colour).capitalize()} is {_relation(a,b)} "
                 f"{_np(b, colour=colour)}, and {_np(c, colour=colour)} is "
                 f"{_relation(c,d)} {_np(d, colour=colour)}."),
        lambda: _sorted_desc(objects, axis=1, direction="left to right", colour=colour),
        lambda: _sorted_desc(objects, axis=0, direction="top to bottom", colour=colour),
        lambda: (
            "The scene contains "
            + ", ".join(f"{_np(o, colour=colour)} at the {_pos_phrase(o.position)}"
                        for o in objects[:-1])
            + f", and {_np(objects[-1], colour=colour)} at the "
              f"{_pos_phrase(objects[-1].position)}."
        ),
    ]
    return random.choice(templates)()


def _sorted_desc(objects: list[SceneObject], axis: int,
                 direction: str, colour: bool = True) -> str:
    other = 1 - axis
    srt   = sorted(objects,
                   key=lambda o: (POSITIONS[o.position][axis],
                                  POSITIONS[o.position][other]))
    labels = [_np(o, colour=colour) for o in srt]
    listed = ", ".join(labels[:-1]) + ", and " + labels[-1]
    return f"From {direction}: {listed}."


def describe(scene: Scene, colour: bool = True) -> str:
    """
    Return an English sentence describing the scene.

    colour=False -> colour adjectives are omitted (greyscale image has no colour info).
    Containment pairs are described first; remaining plain objects follow.
    """
    pairs   = [(o, o.inside) for o in scene.objects if o.inside is not None]
    singles = [o for o in scene.objects if o.inside is None]
    sentences: list[str] = []

    for outer, inner in pairs:
        cp = _containment_phrase(outer, inner, colour=colour)
        if random.random() < 0.45:
            cp += f" at the {_pos_phrase(outer.position)}"
        cp = cp.capitalize()
        if not cp.endswith("."): cp += "."
        sentences.append(cp)

    n         = len(singles)
    has_prior = bool(sentences)

    if n == 0:
        pass
    elif n == 1:
        s  = singles[0]
        sw = random.choice(SIZE_WORDS[s.size])
        if has_prior:
            connector = random.choice([
                f", alongside {_np(s, colour=colour)} at the {_pos_phrase(s.position)}",
                f", with {_np(s, colour=colour)} at the {_pos_phrase(s.position)}",
                f", and {_np(s, colour=colour)} nearby at the {_pos_phrase(s.position)}",
            ])
            sentences[-1] = sentences[-1].rstrip(".") + connector + "."
        else:
            col_part = f"{s.color} " if colour else ""
            sentences.append(
                f"{_a(sw).capitalize()} {sw} {col_part}{s.shape} "
                f"occupies the {_pos_phrase(s.position)}."
            )
    elif n == 2:
        clause = _describe_2(*singles, colour=colour)
        if not clause.endswith("."): clause += "."
        if has_prior: clause = "Also visible: " + clause[0].lower() + clause[1:]
        sentences.append(clause)
    elif n == 3:
        objs = list(singles); random.shuffle(objs)
        clause = _describe_3(*objs, colour=colour)
        if not clause.endswith("."): clause += "."
        if has_prior: clause = "Additionally, " + clause[0].lower() + clause[1:]
        sentences.append(clause)
    else:
        objs = list(singles); random.shuffle(objs)
        clause = _describe_4(objs, colour=colour)
        if not clause.endswith("."): clause += "."
        if has_prior: clause = "Additionally, " + clause[0].lower() + clause[1:]
        sentences.append(clause)

    return " ".join(sentences)

def random_scene(num_objects:   int | None = None,
                 shape_mode:    str | None = None,
                 allow_overlap: bool       = False) -> Scene:
    """
    Generate a random scene.

    Args:
        num_objects:   Number of grid cells to fill (2–4).  Random if None.
        shape_mode:    "2d" | "3d" | None  (None → randomly pick one per call).
        allow_overlap: If True, one cell will hold a containment pair
                       (large outer + small inner with distinct shape & colour).

    Guarantees:
        • All occupied cells are distinct.
        • All (shape, color) pairs are distinct across the whole scene,
          including the inner object of any containment pair.
    """
    if num_objects is None:
        num_objects = random.randint(2, 4)
    if num_objects not in (2, 3, 4):
        raise ValueError("num_objects must be 2, 3, or 4.")

    mode  = shape_mode if shape_mode in ("2d", "3d") else random.choice(("2d", "3d"))
    pool  = SHAPES_2D if mode == "2d" else SHAPES_3D

    # Number of unique (shape, color) combos needed
    total_needed = num_objects + (1 if allow_overlap else 0)
    all_combos   = [(s, c) for s in pool for c in COLOR_NAMES]
    total_needed = min(total_needed, len(all_combos))

    chosen_combos = random.sample(all_combos, total_needed)
    positions     = random.sample(list(POSITIONS.keys()), num_objects)

    objects = [
        SceneObject(shape=s, color=c,
                    size=random.choice(SIZES),
                    position=positions[i],
                    shape_mode=mode)
        for i, (s, c) in enumerate(chosen_combos[:num_objects])
    ]

    if allow_overlap and total_needed > num_objects:
        host      = random.choice(objects)
        host.size = "large"
        inner_s, inner_c = chosen_combos[num_objects]
        host.inside = SceneObject(
            shape=inner_s, color=inner_c, size="small",
            position=host.position, shape_mode=mode,
        )

    return Scene(objects=objects)


# ── Colour / tone helpers ──────────────────────────────────────────────────────

def _dk(rgb: tuple[int, int, int], f: float) -> tuple[int, int, int]:
    return tuple(max(0, int(c * f)) for c in rgb)    # type: ignore[return-value]

def _lt(rgb: tuple[int, int, int], f: float) -> tuple[int, int, int]:
    return tuple(min(255, int(c + (255 - c) * f)) for c in rgb)  # type: ignore[return-value]

def _gdk(v: int, f: float) -> int:
    return max(0, int(v * f))

def _glt(v: int, f: float) -> int:
    return min(255, int(v + (255 - v) * f))

def _fill(color_name: str, colour: bool, is_inner: bool = False):
    return COLORS[color_name] if colour else (BW_INNER_FILL if is_inner else BW_FILL)

def _otl(color_name: str, colour: bool, is_inner: bool = False):
    return _dk(COLORS[color_name], 0.40) if colour else (BW_INNER_OUTLINE if is_inner else BW_OUTLINE)


# ── 2-D shape drawers ──────────────────────────────────────────────────────────

def _draw_circle(draw, cx, cy, r, fill, outline, lw):
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=fill, outline=outline, width=lw)

def _draw_square(draw, cx, cy, r, fill, outline, lw):
    draw.rectangle([cx-r, cy-r, cx+r, cy+r], fill=fill, outline=outline, width=lw)

def _draw_triangle(draw, cx, cy, r, fill, outline, lw):
    pts = [(cx, cy-r),
           (cx - int(r*0.866), cy + r//2),
           (cx + int(r*0.866), cy + r//2)]
    draw.polygon(pts, fill=fill, outline=outline)
    draw.line(pts + [pts[0]], fill=outline, width=lw)

def _draw_rectangle(draw, cx, cy, r, fill, outline, lw):
    rh = max(4, int(r * 0.58))
    draw.rectangle([cx-r, cy-rh, cx+r, cy+rh], fill=fill, outline=outline, width=lw)

def _draw_pentagon(draw, cx, cy, r, fill, outline, lw):
    pts = [(cx + r*math.cos(2*math.pi*i/5 - math.pi/2),
            cy + r*math.sin(2*math.pi*i/5 - math.pi/2))
           for i in range(5)]
    draw.polygon(pts, fill=fill, outline=outline)
    draw.line(pts + [pts[0]], fill=outline, width=lw)

def _draw_star(draw, cx, cy, r, fill, outline, lw):
    inner_r = max(3, int(r * 0.40))
    pts = [(cx + (r if i%2==0 else inner_r) * math.cos(math.pi*i/5 - math.pi/2),
            cy + (r if i%2==0 else inner_r) * math.sin(math.pi*i/5 - math.pi/2))
           for i in range(10)]
    draw.polygon(pts, fill=fill, outline=outline)
    draw.line(pts + [pts[0]], fill=outline, width=lw)

_DRAW_2D = {
    "circle":    _draw_circle,
    "square":    _draw_square,
    "triangle":  _draw_triangle,
    "rectangle": _draw_rectangle,
    "pentagon":  _draw_pentagon,
    "star":      _draw_star,
}


# ── 3-D shape drawers ──────────────────────────────────────────────────────────

def _draw_sphere(draw, cx, cy, r, color_name, colour, is_inner=False):
    fill = _fill(color_name, colour, is_inner)
    lw   = max(1, r // 14)
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=fill)
    h_r = max(2, r//3);  h_cx = cx - r//3;  h_cy = cy - r//3
    if colour:
        rgb = COLORS[color_name]
        draw.ellipse([h_cx-h_r, h_cy-h_r, h_cx+h_r, h_cy+h_r], fill=_lt(rgb, 0.55))
        s_r = max(1, r//8)
        draw.ellipse([h_cx-s_r-4, h_cy-s_r-4, h_cx+s_r-4, h_cy+s_r-4],
                     fill=(255, 255, 255))
        draw.arc([cx-r, cy-r, cx+r, cy+r], start=30, end=150,
                 fill=_dk(rgb, 0.45), width=lw)
    else:
        hl = _glt(fill, 0.40)                                 # type: ignore[arg-type]
        draw.ellipse([h_cx-h_r, h_cy-h_r, h_cx+h_r, h_cy+h_r], fill=hl)
        draw.arc([cx-r, cy-r, cx+r, cy+r], start=30, end=150,
                 fill=_gdk(fill, 0.50), width=lw)             # type: ignore[arg-type]


def _draw_cube(draw, cx, cy, r, color_name, colour, is_inner=False):
    base = COLORS[color_name] if colour else (BW_INNER_FILL if is_inner else BW_FILL)
    if colour:
        top_c = _lt(base, 0.40); left_c = _dk(base, 0.70)
        right_c = _dk(base, 0.50); edge_c = _dk(base, 0.28)
    else:
        top_c = _glt(base, 0.35); left_c = _gdk(base, 0.68)  # type: ignore[arg-type]
        right_c = _gdk(base, 0.52); edge_c = _gdk(base, 0.30)  # type: ignore[arg-type]
    top   = [(cx, cy-r), (cx+r, cy-r//2), (cx, cy), (cx-r, cy-r//2)]
    left  = [(cx-r, cy-r//2), (cx, cy), (cx, cy+r), (cx-r, cy+r//2)]
    right = [(cx, cy), (cx+r, cy-r//2), (cx+r, cy+r//2), (cx, cy+r)]
    lw    = max(1, r // 18)
    draw.polygon(top,   fill=top_c)
    draw.polygon(left,  fill=left_c)
    draw.polygon(right, fill=right_c)
    for poly in (top, left, right):
        draw.line(poly + [poly[0]], fill=edge_c, width=lw)


def _draw_cylinder(draw, cx, cy, r, color_name, colour, is_inner=False):
    base = COLORS[color_name] if colour else (BW_INNER_FILL if is_inner else BW_FILL)
    if colour:
        lt_c = _lt(base, 0.28); dk_c = _dk(base, 0.55)
        cap_c = _dk(base, 0.60); top_c = _lt(base, 0.35); edge_c = _dk(base, 0.35)
    else:
        lt_c = _glt(base, 0.25); dk_c = _gdk(base, 0.58)      # type: ignore[arg-type]
        cap_c = _gdk(base, 0.62); top_c = _glt(base, 0.30)    # type: ignore[arg-type]
        edge_c = _gdk(base, 0.38)                              # type: ignore[arg-type]
    h = int(r*1.7); ew = r; eh = max(3, r//4); lw = max(1, r//18)
    ty = cy - h//2;  by = cy + h//2
    draw.rectangle([cx-ew, ty+eh, cx+ew, by], fill=base)
    draw.rectangle([cx-ew, ty+eh, cx-ew*2//3, by], fill=lt_c)
    draw.rectangle([cx+ew*2//3, ty+eh, cx+ew, by], fill=dk_c)
    draw.ellipse([cx-ew, by-eh, cx+ew, by+eh], fill=cap_c)
    draw.ellipse([cx-ew, ty-eh, cx+ew, ty+eh], fill=top_c)
    draw.ellipse([cx-ew, ty-eh, cx+ew, ty+eh], outline=edge_c, width=lw)
    draw.line([(cx-ew, ty), (cx-ew, by)], fill=edge_c, width=lw)
    draw.line([(cx+ew, ty), (cx+ew, by)], fill=edge_c, width=lw)


def _draw_cone(draw, cx, cy, r, color_name, colour, is_inner=False):
    base = COLORS[color_name] if colour else (BW_INNER_FILL if is_inner else BW_FILL)
    if colour:
        lit_c = _lt(base, 0.18); drk_c = _dk(base, 0.55)
        base_c = _dk(base, 0.65); edge_c = _dk(base, 0.35)
    else:
        lit_c = _glt(base, 0.22); drk_c = _gdk(base, 0.55)    # type: ignore[arg-type]
        base_c = _gdk(base, 0.65); edge_c = _gdk(base, 0.38)   # type: ignore[arg-type]
    h = int(r*1.9); ew = r; eh = max(3, r//4); lw = max(1, r//18)
    apex = (cx, cy - h//2);  by = cy + h//2
    draw.polygon([apex, (cx-ew, by), (cx, by)],  fill=lit_c)
    draw.polygon([apex, (cx, by),    (cx+ew, by)], fill=drk_c)
    draw.ellipse([cx-ew, by-eh, cx+ew, by+eh], fill=base_c)
    draw.line([apex, (cx-ew, by)], fill=edge_c, width=lw)
    draw.line([apex, (cx+ew, by)], fill=edge_c, width=lw)
    draw.ellipse([cx-ew, by-eh, cx+ew, by+eh], outline=edge_c, width=lw)


def _draw_pyramid(draw, cx, cy, r, color_name, colour, is_inner=False):
    base = COLORS[color_name] if colour else (BW_INNER_FILL if is_inner else BW_FILL)
    if colour:
        bk = _dk(base, 0.40); lf = _dk(base, 0.62)
        rt = _dk(base, 0.72); fr = _lt(base, 0.12); ed = _dk(base, 0.28)
    else:
        bk = _gdk(base, 0.42); lf = _gdk(base, 0.62)           # type: ignore[arg-type]
        rt = _gdk(base, 0.72); fr = _glt(base, 0.15)            # type: ignore[arg-type]
        ed = _gdk(base, 0.30)                                    # type: ignore[arg-type]
    h = int(r*1.65); lw = max(1, r//18)
    apex = (cx, cy - h//2)
    bfl = (cx-r, cy+h//3); bfr = (cx+r, cy+h//3)
    bbl = (cx-r//2, cy-h//6); bbr = (cx+r//2, cy-h//6)
    draw.polygon([apex, bbl, bbr], fill=bk)
    draw.polygon([apex, bbl, bfl], fill=lf)
    draw.polygon([apex, bfr, bbr], fill=rt)
    draw.polygon([apex, bfl, bfr], fill=fr)
    for pt in (bfl, bfr, bbl, bbr):
        draw.line([apex, pt], fill=ed, width=lw)
    for pair in [(bfl, bfr), (bfl, bbl), (bfr, bbr), (bbl, bbr)]:
        draw.line(list(pair), fill=ed, width=lw)


_DRAW_3D = {
    "sphere":   _draw_sphere,
    "cube":     _draw_cube,
    "cylinder": _draw_cylinder,
    "cone":     _draw_cone,
    "pyramid":  _draw_pyramid,
}


# ── Unified shape dispatcher ───────────────────────────────────────────────────

def _draw_object(draw: ImageDraw.ImageDraw,
                 cx: int, cy: int, r: int,
                 obj: SceneObject,
                 colour: bool,
                 is_inner: bool = False) -> None:
    if obj.shape in _DRAW_2D:
        fill = _fill(obj.color, colour, is_inner)
        otl  = _otl(obj.color, colour, is_inner)
        lw   = max(1, r // 11)
        _DRAW_2D[obj.shape](draw, cx, cy, r, fill, otl, lw)
    else:
        _DRAW_3D[obj.shape](draw, cx, cy, r, obj.color, colour, is_inner)


# ── Scene renderer ──────────────────────────────────────────────────────────────

IMG_SIZE    = 256
CELL_MARGIN = 14

def _cell_px(size: int = IMG_SIZE) -> int:
    return (size - 2 * CELL_MARGIN) // 3


def render_scene_image(scene: Scene,
                       size:   int  = IMG_SIZE,
                       colour: bool = True) -> np.ndarray:
    """
    Render the scene to a numpy float32 array in [0, 1].

    Returns:
        colour=True  → (H, W, 3)  standard convention (0=dark, 1=bright)
        colour=False → (H, W)     same convention, single channel
    """
    bg_val = random.randint(235, 248)
    if colour:
        bg   = (bg_val, bg_val, bg_val)
        mode = "RGB"
        img  = Image.new(mode, (size, size), color=bg)
    else:
        mode = "L"
        img  = Image.new(mode, (size, size), color=bg_val)

    # Background noise
    arr = np.array(img, dtype=np.int16)
    arr = np.clip(arr + np.random.normal(0, 2.0, arr.shape).astype(np.int16),
                  0, 255).astype(np.uint8)
    img  = Image.fromarray(arr, mode=mode)
    draw = ImageDraw.Draw(img)
    cell = _cell_px(size)

    for obj in scene.objects:
        row, col = POSITIONS[obj.position]
        cx = CELL_MARGIN + col*cell + cell//2 + random.randint(-4, 4)
        cy = CELL_MARGIN + row*cell + cell//2 + random.randint(-4, 4)
        r  = max(6, int(cell * SIZE_SCALE[obj.size] * 0.5))

        _draw_object(draw, cx, cy, r, obj, colour, is_inner=False)

        if obj.inside is not None:
            # Inner shape at ~40% of outer's radius — clearly contained
            r_inner = max(4, int(r * 0.42))
            _draw_object(draw, cx, cy, r_inner, obj.inside, colour, is_inner=True)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return np.asarray(img, dtype=np.float32) / 255.0


# ── File saving ─────────────────────────────────────────────────────────────────

def save_sample(n: int,
                scene:    Scene,
                sentence: str,
                colour:   bool,
                split:    str = "train",
                prefix:   str = "sample") -> None:
    """
    Save four files for sample n and one manifest row.

        {split}/text/{prefix}_{n:06d}.txt
        {split}/meta/{prefix}_{n:06d}.meta
        {split}/images/{prefix}_{n:06d}.png
        {split}/matrices/{prefix}_{n:06d}.mat

    .mat convention (mirrors TicTacToe generator):
        colour  → shape (H*3, W),   reshape to (H,W,3) to use
        BW      → shape (H,  W)
    """
    _ensure_dirs()
    _init_manifest()

    arr      = render_scene_image(scene, colour=colour)
    filename = f"{prefix}_{n:06d}"

    (_dir(split, "text") / f"{filename}.txt").write_text(
        sentence + "\n", encoding="utf-8")

    with open(_dir(split, "meta") / f"{filename}.meta", "w", encoding="utf-8") as f:
        f.write(sentence + "\n")
        f.write(scene.to_notation() + "\n")
        f.write(scene.ascii_grid() + "\n")

    pil_mode = "RGB" if colour else "L"
    Image.fromarray((arr * 255).astype(np.uint8), mode=pil_mode).save(
        _dir(split, "images") / f"{filename}.png")

    mat = arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr
    np.savetxt(_dir(split, "matrices") / f"{filename}.mat", mat, fmt="%.4f")

    all_objs = scene.all_objects
    _append_manifest({
        "filename":    f"{split}/images/{filename}.png",
        "notation":    scene.to_notation(),
        "num_objects": len(all_objs),
        "shapes":      ",".join(o.shape for o in all_objs),
        "colors":      ",".join(o.color for o in all_objs),
        "shape_mode":  scene.shape_mode,
        "colour":      colour,
        "has_overlap": scene.has_overlap,
        "split":       split,
    })


# ── Main generation loop ────────────────────────────────────────────────────────

if __name__ == "__main__":
    TOTAL = 5000    # ← edit this to generate more / fewer samples

    # Balance across all four axes:
    #   num_objects ∈ {2, 3, 4}
    #   shape_mode  ∈ {"2d", "3d"}
    #   colour      ∈ {True, False}
    #   overlap     ∈ {True, False}
    COMBOS    = list(itertools.product([2,3,4], ["2d","3d"], [True,False], [True,False]))
    per_combo = max(1, TOTAL // len(COMBOS))   # ≈ 41 for TOTAL=1000

    print(f"=== Generating {TOTAL} shape-arrangement samples ===")
    print(f"    {len(COMBOS)} param combos  ×  ~{per_combo} each\n")

    n = 0
    for num_objects, shape_mode, colour, overlap in tqdm(COMBOS):
        for i in range(per_combo):
            if n >= TOTAL:
                break
            r     = i / per_combo
            split = "train" if r < 0.70 else "val" if r < 0.85 else "test"

            scene    = random_scene(num_objects=num_objects,
                                    shape_mode=shape_mode,
                                    allow_overlap=overlap)
            sentence = describe(scene, colour=colour)
            save_sample(n, scene, sentence, colour=colour, split=split)

            # if n % 100 == 0:
            #     tag = (f"{'RGB' if colour else 'BW'}/{shape_mode}"
            #            f"{'+overlap' if overlap else ''}")
            #     print(f"  [{n:>4}/{TOTAL}] split={split} | {tag} "
            #           f"| n_obj={num_objects} "
            #           f"| {','.join(o.shape for o in scene.objects)}")
            n += 1
        if n >= TOTAL:
            break

    while n < TOTAL:
        r     = (n % max(per_combo,1)) / max(per_combo,1)
        split = "train" if r < 0.70 else "val" if r < 0.85 else "test"
        scene    = random_scene()
        sentence = describe(scene, colour=True)
        save_sample(n, scene, sentence, colour=True, split=split)
        n += 1

    # print(f"\nDone.  {n} samples saved to {BASE_DIR}")
    counts = {s: len(list((BASE_DIR / s / "images").glob("*.png")))
              for s in SPLITS}
    for s, c in counts.items():
        print(f"  {s:<6}: {c}")