# Shape Arrangement Data Generator

## What This Project Does

Generates matched **(image, English sentence)** pairs for training image-to-text
neural networks. Each sample shows 2–4 geometric shapes arranged on a 3×3 grid.

Example sentences:

> *"A tiny blue square nested inside a big cyan triangle at the bottom-centre.
> Additionally, the scene shows a sizeable orange square at the centre."*

> *"A tiny rectangle enclosed within a sizeable circle, with a large square at
> the top-centre."*  ← B&W sentence (no colour adjectives)

> *"From left to right: a large red cylinder, a big yellow cube, and a medium
> green sphere."*

---

## What's New

| Feature | Detail |
|---------|--------|
| **2-D shapes** | circle, square, triangle, rectangle, pentagon, star |
| **3-D shapes** | sphere, cube, cylinder, cone, pyramid (unchanged from v1) |
| **`shape_mode` parameter** | `"2d"`, `"3d"`, or `None` (random 50/50 per scene) |
| **Containment / overlap** | One shape drawn inside another; sentence describes it naturally |
| **B&W images** | Greyscale renders; sentences omit colour adjectives automatically |

---

## The Symbolic Vocabulary

Three independent axes combine freely:

| Axis | 2-D elements | 3-D elements |
|------|-------------|-------------|
| **Shape** | circle, square, triangle, rectangle, pentagon, star | sphere, cube, cylinder, cone, pyramid |
| **Colour** | red, blue, green, yellow, purple, orange, pink, cyan | ← same |
| **Size** | small, medium, large | ← same |

**6 × 8 × 3 = 144** possible 2-D triples; **5 × 8 × 3 = 120** possible 3-D triples.

---

## Containment / Overlap

When `allow_overlap=True`, one grid cell holds a **containment pair**:

- The **outer** object is forced to `large` (so the inner clearly fits inside).
- The **inner** object is always `small`, drawn at ~42 % of the outer's radius.
- The two objects always have distinct `(shape, color)` pairs.

**Colour images** — the inner shape uses its own full colour.  
**B&W images** — outer is dark grey; inner is light grey, so both remain visible.

Six sentence templates vary the phrasing:

```
A small blue square inside a large orange circle at the top-left.
Inside the orange circle sits a small blue square at the top-left.
A large orange circle containing a small blue square.
A large orange circle with a small blue square inside it.
A small blue square enclosed within a large orange circle.
A small blue square nested inside a large orange circle at the top-left.
```

---

## B&W vs Colour

| | Colour (RGB) | Black-and-white |
|--|--------------|-----------------|
| **Image channels** | 3-channel RGB | 1-channel greyscale |
| **Sentence** | includes colour adjectives | colour adjectives omitted |
| **Example** | *"a large red cylinder above a small blue cube"* | *"a large cylinder above a small cube"* |
| **.mat shape** | `(H×3, W)` | `(H, W)` |

The B&W task is deliberately harder — the network must rely on shape and spatial
position, not colour.

---

## Grid Position Names

Positions are named exactly as in the TicTacToe generator:

```
 TL | TM | TR
----+----+----
 ML |  C | MR
----+----+----
 BL | BM | BR
```

| Name | Meaning |
|------|---------|
| TL | Top Left | TM | Top Middle | TR | Top Right |
| ML | Middle Left | C | Centre | MR | Middle Right |
| BL | Bottom Left | BM | Bottom Middle | BR | Bottom Right |

---

## Scene Notation Format

```
circle:blue,large,TL,2d | square:red,small,BR,2d
```

Each token: `shape:color,size,position,mode`.  
A `>` suffix attaches an inner (contained) object:

```
cone:green,large,TM,3d>pyramid:pink,small,TM,3d
```

The full notation for a containment scene with two cells occupied:

```
square:cyan,large,ML,2d | rectangle:yellow,large,BL,2d>square:pink,small,BL,2d
```

---

## Spatial Relations Used in Sentences

| Row offset | Col offset | Phrase |
|-----------|-----------|--------|
| −1 | 0 | *above* |
| −2 | 0 | *well above* |
| +1 | 0 | *below* |
| +2 | 0 | *well below* |
| 0 | −1 | *to the left of* |
| 0 | −2 | *well to the left of* |
| 0 | +1 | *to the right of* |
| 0 | +2 | *well to the right of* |
| ±dr | ±dc | combined, e.g. *above and to the right of* |

---

## Sentence Templates

### 2-object scenes — 10 structural templates

| # | Template | Example |
|---|---------|---------|
| 1 | Simple copula | *"A large red sphere is above a small blue cube."* |
| 2 | Reversed perspective | *"A small blue cube is below a large red sphere."* |
| 3 | Existential | *"There is a large red sphere above a small blue cube."* |
| 4 | 'sits' | *"A large red sphere sits above a small blue cube."* |
| 5 | 'can be seen' | *"A large red sphere can be seen above a small blue cube."* |
| 6 | Named positions | *"A large red sphere occupies the top-left, while a small blue cube is at the lower-right corner."* |
| 7 | Enumeration + size back-ref | *"A large red sphere and a small blue cube appear in the scene; the large red sphere is sizeable and rests above the small blue cube."* |
| 8 | Two-shape enumeration | *"Two shapes are visible: a large red sphere at the top-left and a small blue cube at the lower-right corner."* |
| 9 | 'positioned' | *"A large red sphere is positioned above a small blue cube on a plain background."* |
| 10 | Q&A | *"Where is the large red sphere? It is above the small blue cube."* |

7 templates for 3-object scenes, 5 for 4-object scenes.

Size synonyms vary randomly: `small / tiny / little`, `medium / medium-sized / mid-sized`, `large / big / sizeable`.

---

## Output Structure

```
Shapes_Data/
│
├── manifest.csv              ← Index of every sample
│
├── train/                    ← 70 % of samples
│   ├── text/                 ← .txt  (one sentence per file)
│   ├── meta/                 ← .meta (sentence + notation + ASCII grid)
│   ├── images/               ← .png  (256×256 image, RGB or greyscale)
│   └── matrices/             ← .mat  (flat float matrix)
│
├── val/                      ← 15 % of samples
│   └── …
│
└── test/                     ← 15 % of samples
    └── …
```

### manifest.csv columns

| Column | Meaning |
|--------|---------|
| `filename` | Path to the PNG, relative to `Shapes_Data/` |
| `notation` | Full scene notation string |
| `num_objects` | Total shape instances (including inner/contained objects) |
| `shapes` | Comma-separated shape names |
| `colors` | Comma-separated colour names |
| `shape_mode` | `"2d"`, `"3d"`, or `"mixed"` |
| `colour` | `True` = RGB image, `False` = greyscale image |
| `has_overlap` | `True` if the scene has at least one containment pair |
| `split` | `train`, `val`, or `test` |

---

## How To Install and Run

### Step 1 — Check Python

Requires **Python 3.10 or higher**:

```bash
python --version
```

### Step 2 — Install libraries

```bash
pip install numpy Pillow
```

### Step 3 — Run the generator

```bash
python shapes.py
```

Progress is printed every 100 samples. All output folders are created automatically.
Generating 1 000 samples at 256 px takes roughly **2 minutes** on a typical laptop.

### Step 4 — Tune the generation

Open `shapes.py` and find these two lines near the bottom:

```python
TOTAL    = 1000   # total number of samples to generate
IMG_SIZE = 256     # image resolution in pixels; raise to 512 for more detail
```

---

## Generation Balance

The main loop cycles through **24 parameter combinations**:

```
num_objects ∈ {2, 3, 4}
shape_mode  ∈ {"2d", "3d"}
colour      ∈ {True, False}
overlap     ∈ {True, False}
```

Each combination gets `TOTAL // 24` samples, so all axes are balanced.

---

## Using `random_scene()` Directly

```python
from shapes import random_scene, describe, render_scene_image, save_sample

# 3-D colour scene, 3 objects, one containment pair
scene    = random_scene(num_objects=3, shape_mode="3d", allow_overlap=True)
sentence = describe(scene, colour=True)
arr      = render_scene_image(scene, size=256, colour=True)  # (256,256,3) float32

# 2-D greyscale scene, 2 objects, no overlap
scene    = random_scene(num_objects=2, shape_mode="2d", allow_overlap=False)
sentence = describe(scene, colour=False)   # colour adjectives omitted
arr      = render_scene_image(scene, colour=False)           # (256,256) float32

# Save to disk with the standard four-file format
save_sample(n=0, scene=scene, sentence=sentence, colour=False, split="train")
```

### `shape_mode=None` (the default)

```python
scene = random_scene()          # shape_mode chosen randomly per call (50 % 2D / 50 % 3D)
```

---

## Loading in PyTorch

```python
import csv
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class ShapesDataset(Dataset):
    def __init__(self, manifest_path, split="train", transform=None):
        base = Path(manifest_path).parent
        rows = list(csv.DictReader(open(manifest_path, encoding="utf-8")))
        self.samples = [
            (base / r["filename"], r["notation"],
             r["colour"] == "True", r["has_overlap"] == "True")
            for r in rows if r["split"] == split
        ]
        self.transform = transform or T.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, notation, colour, has_overlap = self.samples[idx]
        img = Image.open(path).convert("RGB")   # normalise to 3-channel
        return self.transform(img), notation
```

Use `manifest.csv` to build your `Dataset`. Filter by `split`, load images from
`filename`, and use `notation` as the ground-truth label.

---

## .mat File Convention

| Image type | `.mat` shape | How to load |
|------------|-------------|-------------|
| Colour RGB | `(H×3, W)` | `np.loadtxt(f).reshape(H, W, 3)` |
| Greyscale | `(H, W)` | `np.loadtxt(f)` |

Values are in `[0.0, 1.0]` where `0.0 = dark` and `1.0 = bright`.

---

## Example .meta File

```
A tiny blue square nested inside a big cyan triangle at the bottom-centre.
triangle:cyan,large,BM,2d>square:blue,small,BM,2d
    ---       |      ---       |      ---     
──────────────────────────────────────────────
    ---       |      ---       |      ---     
──────────────────────────────────────────────
    ---       |  CYA/TRI[SQU]  |      ---     
```

---

## How the Rendering Works

### 2-D shapes

Each 2-D shape is drawn as a filled, outlined polygon or ellipse using
`PIL.ImageDraw`. A small random position jitter (±4 px) is added per object.

| Shape | Method |
|-------|--------|
| circle | `ellipse` |
| square | `rectangle` |
| triangle | equilateral triangle polygon |
| rectangle | wider-than-tall rectangle |
| pentagon | 5-point regular polygon |
| star | 10-point star polygon (inner radius 40 % of outer) |

### 3-D shapes

Each 3-D shape uses face-based shading to simulate lighting from the upper-left:

| Shape | Technique |
|-------|-----------|
| sphere | Filled circle + diffuse highlight oval + specular dot + shadow arc |
| cube | Three isometric parallelograms (top lightest, right darkest) |
| cylinder | Rectangle body with light/dark strips + top and bottom elliptical caps |
| cone | Two triangular faces (lit left, dark right) + elliptical base |
| pyramid | Four triangular faces in isometric projection |

### B&W shading

In greyscale mode:
- Outer shapes: dark fill (~65 / 255) on a near-white background (~242 / 255)
- Inner (contained) shapes: light fill (~195 / 255) with a dark outline, so they
  stand out clearly against the darker outer shape

### Background

Soft off-white with low-amplitude Gaussian noise (σ ≈ 2 px) to simulate paper
texture. A final `GaussianBlur(radius=0.6)` softens aliasing on shape edges.

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| numpy | ≥ 1.21 | Array operations and background noise |
| Pillow | ≥ 9.0 | Drawing shapes and saving images |
| Python | ≥ 3.10 | `str \| None` type-hint syntax |

---

## File Summary

| File | Purpose |
|------|---------|
| `shapes.py` | Main script — run this to generate data |
| `Shapes_Data/` | All generated output — do not edit by hand |
| `manifest.csv` | Index of every generated sample |
| `shapes_README.md` | This file |