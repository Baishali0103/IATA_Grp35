# -*- coding: utf-8 -*-
"""


This gives a simple example for tictactoe board generation; it makes a
game position, converts it into a sample sentance and makes a picture
of the corresponding game position; it produces four files for each
example, the .txt file holds the sentence, the .meta file an ascii
drawing and the game postion in symbols, .mat a greyscale image matrix
for the game and .png the corresponding picture. There is some naive
noise added to the picture and some slight randomness for the
sentences.

This is just an example, feel free to start with less noise, or to
progress to a more natural set of drawing, maybe using scanned X's and
O's and more noise types. The sentences could be made more complex!
This is only for tictactoe, the other examples of images and sentences
are also available, but this kind of shows the logic, find a sort of
"logical language" out of which both images and sentences can be
generated.


Symbolic language for tic-tac-toe board positions.

Board positions are named by their grid location:

    TL | TM | TR
    ---+----+---
    ML |  C | MR
    ---+----+---
    BL | BM | BR

Notation format:  "X:TL,C,TM O:TR,BL"
  - Players separated by a space
  - Each player's positions are comma-separated
  - Either player may be omitted if they have no moves yet
  - X always moves first, so |X| == |O| or |X| == |O| + 1
"""

import math
import random
from dataclasses import dataclass, field
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import csv


# --- Paths ---
BASE_DIR      = Path(__file__).resolve().parent / "TicTacToe_Data"
SPLITS        = ["train", "val", "test"]
SUBDIRS       = ["text", "meta", "images", "matrices"]
MANIFEST_PATH = BASE_DIR / "manifest.csv"
MANIFEST_FIELDS = ["filename", "notation", "colour", "num_moves", "winner", "is_draw", "split"]

def _ensure_dirs():
    for split in SPLITS:
        for sub in SUBDIRS:
            (BASE_DIR / split / sub).mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

def _dir(split: str, kind: str) -> Path:
    return BASE_DIR / split / kind

def _init_manifest():
    """Write header row if manifest does not exist yet."""
    if not MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(MANIFEST_FIELDS)

def _append_manifest(row: dict):
    with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([row[k] for k in MANIFEST_FIELDS])

# --- Colour ----
X_COLOR  = (200, 30,  30)   # red
O_COLOR  = (30,  30,  200)  # blue
WHITE_RGB = (255, 255, 255)


# --- Position vocabulary ---

POSITIONS = {
    "TL": (0, 0), "TM": (0, 1), "TR": (0, 2),
    "ML": (1, 0), "C":  (1, 1), "MR": (1, 2),
    "BL": (2, 0), "BM": (2, 1), "BR": (2, 2),
}

COORD_TO_NAME = {v: k for k, v in POSITIONS.items()}

ALIASES = {
    "TC": "TM",
    "BC": "BM",
    "CTR": "C",
    "M":  "C",
    "MM": "C",
}


def resolve(pos: str) -> tuple[int, int]:
    """Return (row, col) for a position name, handling aliases."""
    pos = pos.strip().upper()
    pos = ALIASES.get(pos, pos)
    if pos not in POSITIONS:
        valid = ", ".join(sorted(POSITIONS))
        raise ValueError(f"Unknown position '{pos}'. Valid positions: {valid}")
    return POSITIONS[pos]


# --- Board ---

@dataclass
class Board:
    x: list[str] = field(default_factory=list)
    o: list[str] = field(default_factory=list)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        x_coords = [resolve(p) for p in self.x]
        o_coords = [resolve(p) for p in self.o]
        all_coords = x_coords + o_coords

        if len(all_coords) != len(set(all_coords)):
            raise ValueError("Duplicate positions detected.")

        if not (len(self.x) == len(self.o) or len(self.x) == len(self.o) + 1):
            raise ValueError(
                f"Invalid move counts: X={len(self.x)}, O={len(self.o)}. "
                "X moves first, so X must have equal or one more move than O."
            )

    def to_grid(self) -> list[list[str]]:
        """Return a 3x3 list of lists with 'X', 'O', or ' '."""
        grid = [[" "] * 3 for _ in range(3)]
        for pos in self.x:
            r, c = resolve(pos)
            grid[r][c] = "X"
        for pos in self.o:
            r, c = resolve(pos)
            grid[r][c] = "O"
        return grid

    def winner(self) -> str | None:
        """Return 'X', 'O', or None if no winner yet."""
        grid = self.to_grid()
        lines = [
            [(0,0),(0,1),(0,2)], [(1,0),(1,1),(1,2)], [(2,0),(2,1),(2,2)],
            [(0,0),(1,0),(2,0)], [(0,1),(1,1),(2,1)], [(0,2),(1,2),(2,2)],
            [(0,0),(1,1),(2,2)], [(0,2),(1,1),(2,0)],
        ]
        for line in lines:
            values = {grid[r][c] for r, c in line}
            if values == {"X"}:
                return "X"
            if values == {"O"}:
                return "O"
        return None

    def __str__(self) -> str:
        grid = self.to_grid()
        rows = []
        for i, row in enumerate(grid):
            rows.append(" " + " | ".join(row) + " ")
            if i < 2:
                rows.append("---+---+---")
        return "\n".join(rows)


# --- Parser ---

def parse(notation: str) -> Board:
    """
    Parse a position notation string into a Board.

    Format: "X:TL,C,TM O:TR,BL"

    Examples:
        parse("X:C")                   # X in center only
        parse("X:C,TL O:TR")          # X center + top-left, O top-right
        parse("O:BR X:C,TL")          # player order doesn't matter
        parse("")                      # empty board
    """
    x_positions = []
    o_positions = []

    for token in notation.upper().split():
        if ":" not in token:
            raise ValueError(
                f"Expected 'PLAYER:POS,...' format, got '{token}'. "
                "Example: \"X:TL,C O:TR\""
            )
        player, _, pos_str = token.partition(":")
        player = player.strip()
        positions = [p.strip() for p in pos_str.split(",") if p.strip()]

        if player == "X":
            x_positions = positions
        elif player == "O":
            o_positions = positions
        else:
            raise ValueError(f"Unknown player '{player}'. Use 'X' or 'O'.")

    return Board(x=x_positions, o=o_positions)


# --- Serialiser ---

def to_notation(board: Board) -> str:
    """Convert a Board back to a canonical notation string."""
    parts = []
    if board.x:
        parts.append("X:" + ",".join(p.upper() for p in board.x))
    if board.o:
        parts.append("O:" + ",".join(p.upper() for p in board.o))
    return " ".join(parts)


# --- Random board generator ---

def random_board(num_moves: int | None = None) -> Board:
    """
    Generate a random legitimate board position.

    'Legitimate' means the position is actually reachable in a real game:
    - X has equal or one more move than O (X goes first)
    - Neither player won before the final move was played

    Args:
        num_moves: total moves played (0-9). Chosen randomly if None.
    """
    all_positions = list(POSITIONS.keys())

    while True:
        if num_moves is None:
            n = random.randint(0, 9)
        else:
            if not 0 <= num_moves <= 9:
                raise ValueError("num_moves must be between 0 and 9.")
            n = num_moves

        chosen = random.sample(all_positions, n)
        x_count = (n + 1) // 2
        x_pos = chosen[:x_count]
        o_pos = chosen[x_count:]

        board = Board(x=x_pos, o=o_pos)

        if _is_reachable(board):
            return board


def _is_reachable(board: Board) -> bool:
    """Return True if this board state is reachable in a real game."""
    w = board.winner()
    if w == "X" and len(board.x) != len(board.o) + 1:
        return False
    if w == "O" and len(board.x) != len(board.o):
        return False
    last_player = "x" if len(board.x) > len(board.o) else "o"
    prev_x = board.x[:-1] if last_player == "x" else board.x
    prev_o = board.o[:-1] if last_player == "o" else board.o
    if prev_x or prev_o or len(board.x) + len(board.o) > 1:
        prev_board = Board(x=prev_x, o=prev_o)
        if prev_board.winner() is not None:
            return False
    return True


# --- English description ---

NAMES = ["Sean", "Eoin", "Niamh", "Aoife", "Siobhan", "Saoirse", "Liam"]

POSITION_PHRASES = {
    "C":  ["middle",              "center"],
    "ML": ["middle row left",     "left of centre"],
    "MR": ["middle row right",    "right of centre"],
    "TM": ["top middle",          "above the center"],
    "BM": ["bottom middle",       "below the center"],
    "TL": ["top left corner",     "up and left from center"],
    "TR": ["top right corner",    "up and right from center"],
    "BL": ["bottom left corner",  "down and left from center"],
    "BR": ["bottom right corner", "down and right from center"],
}

CORNER_POS = {"TL", "TR", "BL", "BR"}
EDGE_POS   = {"TM", "ML", "MR", "BM"}

def _phrase(pos: str) -> str:
    """Pick a random English phrase for a position."""
    pos = pos.strip().upper()
    pos = ALIASES.get(pos, pos)
    return random.choice(POSITION_PHRASES[pos])


def _join(phrases: list[str]) -> str:
    """Join a list of phrases naturally: 'a', 'a and b', 'a, b and c'."""
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]

def _game_state_desc(board: Board, name_x: str, name_o: str,
                     x_label: str, o_label: str, winner: str | None) -> str | None:
    """Return a clause describing turn, move count, or empty board."""
    total = len(board.x) + len(board.o)
    if total == 0:
        return random.choice(["the board is empty", "no moves have been made yet"])
    if winner is not None or total == 9:
        return None   # game over — no turn description needed
    next_player = "X" if len(board.x) == len(board.o) else "O"
    next_name   = name_x  if next_player == "X" else name_o
    return random.choice([
        f"it is {next_name}'s turn",
        f"{total} move{'s have' if total > 1 else ' has'} been played",
        f"{next_name} is next to move",
    ])

def _relative_desc(board: Board, name_x: str, name_o: str,
                   x_label: str, o_label: str) -> str | None:
    """Return a clause summarising positional advantage, or None."""
    if not board.x and not board.o:
        return None
    x_set = {p.upper() for p in board.x}
    o_set = {p.upper() for p in board.o}

    x_corners = len(x_set & CORNER_POS)
    o_corners = len(o_set & CORNER_POS)
    x_center  = "C" in x_set
    o_center  = "C" in o_set

    facts = []
    if x_center:
        facts.append(f"{name_x} controls the center")
    if o_center:
        facts.append(f"{name_o} controls the center")
    if x_corners >= 2:
        facts.append(f"{name_x} has {x_corners} corners")
    if o_corners >= 2:
        facts.append(f"{name_o} has {o_corners} corners")
    return random.choice(facts) if facts else None


def describe(board: Board, x_name: str | None = None, o_name: str | None = None, colour: bool = False) -> str:
    """
    Return an English sentence describing the board position.

    Names are chosen randomly from NAMES if not provided.
    Each position is described using one of its two alternative phrasings.
    """
    # --- Outcome Detection ---
    winner = board.winner()
    total_moves = len(board.x) + len(board.o)
    is_draw = winner is None and total_moves == 9
    # ----------------------
    
    name_x, name_o = _pick_names(x_name, o_name)
    
    x_label = "red X"   if colour else "X"
    o_label = "blue O"  if colour else "O"
    # print("colour------>" + str(colour))

    parts = []

    if board.x:
        x_phrases = _join([_phrase(p) for p in board.x])
        verb = random.choice(["gone for", "taken"])
        parts.append(random.choice([f"{name_x} is X and has {verb} {x_phrases}",f"{name_x} is {x_label} and has {verb} {x_phrases}"]))
        # parts.append(f"{name_x} is {x_label} and has {verb} {x_phrases}")
    else:
        parts.append(random.choice([f"{name_x} is X and has not moved yet",f"{name_x} is {x_label} and has not moved yet"]))

    if board.o:
        o_phrases = _join([_phrase(p) for p in board.o])
        verb = random.choice(["gone for", "taken"])
        parts.append(random.choice([f"{name_o} is O and has {verb} {o_phrases}",f"{name_o} is {o_label} and has {verb} {o_phrases}"]))
    else:
        parts.append(random.choice([f"{name_o} is O and has not moved yet",f"{name_o} is {o_label} and has not moved yet"]))
    
    # --- Game state (turn / move count / empty board) ---
    gs = _game_state_desc(board, name_x, name_o, x_label, o_label, winner)
    if gs:
        parts.append(gs)
        
    # --- Relative / positional description ---
    rel = _relative_desc(board, name_x, name_o, x_label, o_label)
    if rel:
        parts.append(rel)
    
    # --- Outcome ---
    if not _is_reachable(board):
        parts.append(random.choice([
            "Though this position could not occur in a real game",
            "However this is not a reachable position in a real game",
        ]))
    
    if winner == "X":
        parts.append(random.choice([
            f"{name_x} is X and has won",
            f"{x_label} and wins",
        ]))
    elif winner == "O":
        parts.append(random.choice([
            f"{name_o} is O and has won",
            f"{o_label} wins",
        ]))
    elif is_draw:
        parts.append(random.choice([
            "The game is a draw",
            "It is a draw",
        ]))

    return "; ".join(parts) + "."


def _pick_names(x_name: str | None, o_name: str | None) -> tuple[str, str]:
    """Return (x_name, o_name), filling in random names as needed."""
    if x_name and o_name:
        return x_name, o_name
    available = [n for n in NAMES if n not in (x_name, o_name)]
    random.shuffle(available)
    if not x_name:
        x_name = available.pop()
    if not o_name:
        o_name = available.pop()
    return x_name, o_name


# --- Image rendering ---

def _rotate_translate(img: Image.Image, angle_deg: float, dx: int, dy: int, fillcolor=255) -> Image.Image:
    """Rotate then translate a PIL image, filling gaps with white."""
    img = img.rotate(angle_deg, fillcolor=fillcolor, resample=Image.Resampling.BILINEAR)
    img = img.transform(
        img.size, Image.Transform.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        fillcolor=fillcolor, resample=Image.Resampling.BILINEAR,
    )
    return img

def _perspective_coeffs(src_quad, dst_quad):
    """Compute 8 PIL PERSPECTIVE coefficients mapping dst_quad -> src_quad."""
    matrix = []
    for (xs, ys), (xd, yd) in zip(src_quad, dst_quad):
        matrix.append([xd, yd, 1, 0, 0, 0, -xs * xd, -xs * yd])
        matrix.append([0, 0, 0, xd, yd, 1, -ys * xd, -ys * yd])
    A = np.array(matrix, dtype=float)
    b = np.array([c for s in src_quad for c in s], dtype=float)
    return np.linalg.lstsq(A, b, rcond=None)[0].tolist()

def _add_perspective(img: Image.Image, magnitude: float = 0.04, fillcolor=255) -> Image.Image:
    """Apply a small random perspective warp to simulate a non-flat photo angle."""
    w, h = img.size
    d = magnitude
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [
        (random.uniform(0, d * w),     random.uniform(0, d * h)),
        (w - random.uniform(0, d * w), random.uniform(0, d * h)),
        (w - random.uniform(0, d * w), h - random.uniform(0, d * h)),
        (random.uniform(0, d * w),     h - random.uniform(0, d * h)),
    ]
    coeffs = _perspective_coeffs(src, dst)
    return img.transform(img.size, Image.Transform.PERSPECTIVE, coeffs,
                         Image.Resampling.BILINEAR, fillcolor=fillcolor)


def _blur_edges(arr: np.ndarray) -> np.ndarray:
    """
    Randomly soften pixels at foreground/background boundaries.
    - Background pixels (≈0) adjacent to foreground: random value in [0, 0.5]
    - Foreground pixels (≈1) adjacent to background: random value in [0.5, 1]
    """
    
    if arr.ndim == 3:
        return np.stack([_blur_edges(arr[:, :, c]) for c in range(3)], axis=2)
    
    result = arr.copy()
    fg = arr > 0.5

    def _dilate(mask):
        return (
            np.roll(mask, 1, axis=0) | np.roll(mask, -1, axis=0)
            | np.roll(mask, 1, axis=1) | np.roll(mask, -1, axis=1)
        )

    bg_adj = ~fg & _dilate(fg)
    fg_adj =  fg & _dilate(~fg)

    result[bg_adj] = np.random.uniform(0.0, 0.5, bg_adj.sum())
    result[fg_adj] = np.random.uniform(0.5, 1.0, fg_adj.sum())
    return result

def _add_background_noise(arr: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    """Add Gaussian noise to simulate paper texture."""
    return np.clip(arr + np.random.normal(0, sigma, arr.shape), 0.0, 1.0)


def render_board_image(board: Board, size: int = 500, colour: bool = False) -> np.ndarray:
    """
    Render the board as a (size x size) numpy array with values in [0, 1].
    0 = background (white), 1 = foreground (black ink).

    Noise added:
    - Grid and each symbol independently rotated by a random angle in [-pi/16, pi/16]
      and translated by up to 3 pixels in each axis.
    - Edge pixels are randomly softened.
    
    B&W:    returns (size, size)    float array, 0=white 1=black ink.
    Colour: returns (size, size, 3) float array, X=red, O=blue, grid=black.
    """
    margin = random.randint(size // 12, size // 7)
    grid_size = size - 2 * margin
    cell = grid_size // 3
    # variable line thickness
    lw = random.randint(max(2, size // 150), max(5, size // 60))
    # variable symbol size
    pad = random.randint(cell // 8, cell // 4)

    fill = WHITE_RGB if colour else 255
    mode = "RGB"     if colour else "L"
    grid_ink = (0, 0, 0) if colour else 0

    # background noise baked into starting canvas
    noise_sigma = 3.0
    if colour:
        canvas = np.clip(np.random.normal(255, noise_sigma, (size, size, 3)), 0, 255)
    else:
        canvas = np.clip(np.random.normal(255, noise_sigma, (size, size)),    0, 255)

    # --- Grid lines ---
    grid_img = Image.new(mode, (size, size), color=fill)
    gd = ImageDraw.Draw(grid_img)
    for i in (1, 2):
        x = margin + i * cell
        gd.line([(x, margin), (x, margin + grid_size)], fill=grid_ink, width=lw)
        y = margin + i * cell
        gd.line([(margin, y), (margin + grid_size, y)], fill=grid_ink, width=lw)
    angle = random.uniform(-math.pi / 16, math.pi / 16) * 180 / math.pi
    dx, dy = random.randint(-3, 3), random.randint(-3, 3)
    grid_img = _rotate_translate(grid_img, angle, dx, dy, fillcolor=fill)
    grid_img = _add_perspective(grid_img, magnitude=0.03, fillcolor=fill)   # adding perspective
    canvas = np.minimum(canvas, np.array(grid_img, dtype=float))

    # --- Symbols ---
    board_grid = board.to_grid()
    for r in range(3):
        for c in range(3):
            sym = board_grid[r][c]
            if sym == " ":
                continue
            sym_ink = X_COLOR if (colour and sym == "X") else O_COLOR if (colour and sym == "O") else 0
            sym_img = Image.new(mode, (cell, cell), color=fill)
            sd = ImageDraw.Draw(sym_img)
            if sym == "X":
                sd.line([(pad, pad), (cell - pad, cell - pad)], fill=sym_ink, width=lw * 2)
                sd.line([(cell - pad, pad), (pad, cell - pad)], fill=sym_ink, width=lw * 2)
            elif sym == "O":
                sd.ellipse([(pad, pad), (cell - pad, cell - pad)], outline=sym_ink, width=lw * 2)
            angle = random.uniform(-math.pi / 16, math.pi / 16) * 180 / math.pi
            dx, dy = random.randint(-3, 3), random.randint(-3, 3)
            sym_img = _rotate_translate(sym_img, angle, dx, dy, fillcolor=fill)
            sym_img = _add_perspective(sym_img, magnitude=0.03, fillcolor=fill)   # adding perspective
            stamp = np.full((size, size, 3) if colour else (size, size), 255.0)
            py, px = margin + r * cell, margin + c * cell
            stamp[py:py + cell, px:px + cell] = np.array(sym_img, dtype=float)
            canvas = np.minimum(canvas, stamp)

    arr = canvas / 255.0 if colour else 1.0 - canvas / 255.0
    arr = _add_background_noise(arr, sigma=0.015)   # final texture pass
    return _blur_edges(arr)


# --- Test file generator ---

def save_test_files(n: int, board: Board, sentence: str, colour: bool,
                    split: str = "train", prefix: str = "test"):
    """
    Save four files for test case n:
      Data/text/{prefix}_{n}.txt
      Data/meta/{prefix}_{n}.meta
      Data/images/{prefix}_{n}.png
      Data/matrices/{prefix}_{n}.mat
    """
    
    _ensure_dirs()
    _init_manifest()

    arr      = render_board_image(board, colour=colour)
    filename = f"{prefix}_{n}"

    (_dir(split, "text") / f"{filename}.txt").write_text(sentence + "\n", encoding="utf-8")

    with open(_dir(split, "meta") / f"{filename}.meta", "w", encoding="utf-8") as f:
        f.write(sentence + "\n")
        f.write(to_notation(board) + "\n")
        f.write(str(board) + "\n")

    img_mode = "RGB" if colour else "L"
    img = Image.fromarray((arr * 255).astype(np.uint8), mode=img_mode)
    img.save(_dir(split, "images") / f"{filename}.png")

    mat = arr.reshape(-1, arr.shape[1]) if colour else arr
    np.savetxt(_dir(split, "matrices") / f"{filename}.mat", mat, fmt="%.4f")

    winner = board.winner()
    total  = len(board.x) + len(board.o)
    _append_manifest({
        "filename":  f"{split}/images/{filename}.png",
        "notation":  to_notation(board),
        "colour":    colour,
        "num_moves": total,
        "winner":    winner if winner else "none",
        "is_draw":   winner is None and total == 9,
        "split":     split,
    })


def _generate_board(num_moves: int, outcome: str) -> Board:
    """
    Try to generate a board with both a specific num_moves and outcome.
    Falls back to any valid board with num_moves if the combo is impossible.
    """
    x_count = (num_moves + 1) // 2
    o_count = num_moves // 2
    feasible = not (
        (outcome == "x_win"       and x_count < 3) or
        (outcome == "o_win"       and o_count < 3) or
        (outcome == "draw"        and num_moves != 9) or
        (outcome == "in_progress" and num_moves == 9)
    )
    if not feasible:
        return random_board(num_moves=num_moves)
    for _ in range(500):
        board = random_board(num_moves=num_moves)
        w     = board.winner()
        total = len(board.x) + len(board.o)
        if outcome == "x_win"       and w == "X":                 return board
        if outcome == "o_win"       and w == "O":                 return board
        if outcome == "draw"        and w is None and total == 9: return board
        if outcome == "in_progress" and w is None and total < 9:  return board
    return random_board(num_moves=num_moves)   # graceful fallback


# ----------------------- OPTION A -------------------------------------------------

# if __name__ == "__main__":
#     TOTAL    = 100                           # change to generate more/fewer samples
#     OUTCOMES = ["x_win", "o_win", "draw", "in_progress"]
#     per_moves = TOTAL // 10                   # equal samples per num_moves bucket
#
#     print(f"=== Generating {TOTAL} samples ===\n")
#     n = 0
#
#     for num_moves in range(10):
#         for i in range(per_moves):
#             outcome = OUTCOMES[i % len(OUTCOMES)]          # cycle through outcomes
#             colour  = (i % 2 == 0)                         # strict 50/50 colour
#
#             # deterministic split assignment within each bucket
#             r     = i / per_moves
#             split = "train" if r < 0.70 else "val" if r < 0.85 else "test"
#
#             board    = _generate_board(num_moves, outcome)
#             sentence = describe(board, colour=colour)
#             save_test_files(n, board, sentence, colour=colour, split=split)
#
#             if n % 100 == 0:
#                 print(f"  [{n:>4}/{TOTAL}] split={split} | moves={num_moves} "
#                       f"| outcome={outcome:<12} | colour={colour}")
#             n += 1
#
#     print(f"\nDone. {n} samples saved to {BASE_DIR}")
    
    
# ----------------------- OPTION B -------------------------------------------------

if __name__ == "__main__":
    TRAIN_TARGET = 1000                        # training samples you actually want
    SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
    TOTAL        = round(TRAIN_TARGET / SPLIT_RATIOS["train"])  # ~1429 total
    OUTCOMES     = ["x_win", "o_win", "draw", "in_progress"]
    per_moves    = TOTAL // 10                 # samples per num_moves bucket (~142)

    print(f"=== Generating {TOTAL} samples (~{TRAIN_TARGET} training) ===\n")
    n = 0

    for num_moves in range(10):
        for i in range(per_moves):
            outcome = OUTCOMES[i % len(OUTCOMES)]
            colour  = (i % 2 == 0)

            r     = i / per_moves
            split = "train" if r < SPLIT_RATIOS["train"] else \
                    "val"   if r < SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"] else \
                    "test"

            board    = _generate_board(num_moves, outcome)
            sentence = describe(board, colour=colour)
            save_test_files(n, board, sentence, colour=colour, split=split)

            if n % 100 == 0:
                print(f"  [{n:>4}/{TOTAL}] split={split} | moves={num_moves} "
                      f"| outcome={outcome:<12} | colour={colour}")
            n += 1

    print(f"\nDone. {n} total samples (~{round(n * SPLIT_RATIOS['train'])} train, "
          f"~{round(n * SPLIT_RATIOS['val'])} val, "
          f"~{round(n * SPLIT_RATIOS['test'])} test) saved to {BASE_DIR}")