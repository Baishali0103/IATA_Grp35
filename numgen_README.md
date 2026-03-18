# Number Image Data Generator

## What This Project Does

Generates matched **(image, English sentence)** pairs for training image-to-text
neural networks. Each sample renders a single number (1–4 digits) in a chosen
size and colour on a plain off-white background.

Example sentences:

> *"A large blue number 67."*

> *"The digits 1337, small and red."*

> *"In sizeable orange: 42."*

> *"A tiny number 9."*  ← B&W sentence (no colour adjective)

---

## The Symbolic Vocabulary

Three independent axes combine freely:

| Axis | Values |
|------|--------|
| **Value** | Any integer with the chosen number of digits (random) |
| **Digit length** | 1, 2, 3, 4 |
| **Colour** | red, blue, green, yellow, purple, orange, pink, cyan |
| **Size** | small, medium, large |

**4 lengths × 8 colours × 3 sizes = 96** distinct attribute combinations
(value is random within each digit-length bucket).

---

## Notation Format

Each sample is described by a compact notation string:

```
<value>:<color>,<size>
```

Examples:

```
67:blue,large
1337:red,small
5:orange,medium
```

---

## B&W vs Colour

| | Colour (RGB) | Black-and-white |
|--|--------------|-----------------|
| **Image channels** | 3-channel RGB | 1-channel greyscale |
| **Sentence** | includes colour adjectives | colour adjectives omitted |
| **Example** | *"A large blue number 67."* | *"A large number 67."* |
| **.mat shape** | `(H×3, W)` | `(H, W)` |

---

## Sentence Templates

Ten sentence structures vary randomly. For colour images:

| # | Template | Example |
|---|---------|---------|
| 1 | article + size + colour + "number" | *"A large blue number 67."* |
| 2 | "The digits …, size and colour" | *"The digits 1337, small and red."* |
| 3 | article + size + colour + bare value | *"A tiny red 1337."* |
| 4 | size + colour + bare value | *"Large blue 67."* |
| 5 | "The digit/digits … is/are size and colour" | *"The digit 9 is big and purple."* |
| 6 | value + "written size in colour" | *"67, written large in blue."* |
| 7 | "In size colour: value" | *"In sizeable orange: 42."* |
| 8 | "A colour, size number: value" | *"A blue, large-sized number: 1337."* |
| 9 | "Number … — size, colour" | *"Number 67 — large, blue."* |
| 10 | "Here is a size colour value" | *"Here is a tiny red 5."* |

B&W sentences use 9 equivalent templates with colour adjectives removed.

Size synonyms vary randomly: `small / tiny / little`,
`medium / medium-sized / mid-sized`, `large / big / sizeable`.

---

## Font Sizes

Font size is chosen from a table keyed on `(digit_length, size)` so that
numbers fill the canvas attractively regardless of how many digits they have:

| | small | medium | large |
|--|-------|--------|-------|
| **1 digit** | 52 px | 88 px | 126 px |
| **2 digits** | 46 px | 76 px | 108 px |
| **3 digits** | 34 px | 56 px | 82 px |
| **4 digits** | 26 px | 44 px | 66 px |

Multiple bold sans-serif fonts are rotated randomly per sample for visual
variety: DejaVu Sans Bold, DejaVu Sans Mono Bold, Liberation Sans Bold,
FreeSans Bold, Poppins Bold, Carlito Bold.

---

## Rendering Details

- **Background**: soft off-white (`234–248` grey) with low-amplitude Gaussian
  noise (σ ≈ 2 px) to simulate paper texture.
- **Jitter**: text position varies by ±6 px in each axis.
- **Drop shadow**: 30 % of colour-mode samples receive a subtle 2 px shadow.
- **Anti-aliasing**: a final `GaussianBlur(radius=0.55)` softens hard edges.
- **B&W fill**: dark grey (~55 / 255) on near-white (~242 / 255) background.

---

## Output Structure

```
Numbers_Data/
│
├── manifest.csv              ← Index of every sample
│
├── train/                    ← 70 % of samples
│   ├── text/                 ← .txt  (one sentence per file)
│   ├── meta/                 ← .meta (sentence + notation + attributes)
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
| `filename` | Path to the PNG, relative to `Numbers_Data/` |
| `notation` | Notation string, e.g. `67:blue,large` |
| `value` | The integer value rendered |
| `num_digits` | Number of digits (1–4) |
| `color` | Colour name |
| `size` | `small`, `medium`, or `large` |
| `colour` | `True` = RGB image, `False` = greyscale image |
| `split` | `train`, `val`, or `test` |

---

## Example .meta File

```
A large blue number 67.
67:blue,large
value=67  digits=2  color=blue  size=large  colour=yes
```

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
python numgen.py
```

Progress is printed every 100 samples. All output folders are created
automatically. Generating 1 000 samples at 256 px takes roughly
**30–60 seconds** on a typical laptop.

### Step 4 — Tune the generation

Open `numgen.py` and find these two lines near the bottom:

```python
TOTAL    = 1000   # total number of samples to generate
IMG_SIZE = 256    # image resolution in pixels; raise to 512 for more detail
```

---

## Generation Balance

The main loop cycles through **24 parameter combinations**:

```
digit_length ∈ {1, 2, 3, 4}
size         ∈ {"small", "medium", "large"}
colour       ∈ {True, False}
```

Each combination gets `TOTAL // 24` samples, so all axes are balanced.
Colour is randomised per sample within each combo.

---

## Using `random_sample()` Directly

```python
from numgen import random_sample, describe, render_image, save_sample

# Large blue two-digit number, colour image
sample   = random_sample(digit_length=2, size="large")
sample.color = "blue"          # override colour if desired
sentence = describe(sample, colour=True)
arr      = render_image(sample, colour=True)   # (256, 256, 3) float32

# Small greyscale 4-digit number
sample   = random_sample(digit_length=4, size="small")
sentence = describe(sample, colour=False)      # colour adjectives omitted
arr      = render_image(sample, colour=False)  # (256, 256) float32

# Save with the standard four-file format
save_sample(n=0, sample=sample, sentence=sentence, colour=False, split="train")
```

---

## Loading in PyTorch

```python
import csv
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class NumbersDataset(Dataset):
    def __init__(self, manifest_path, split="train", transform=None):
        base = Path(manifest_path).parent
        rows = list(csv.DictReader(open(manifest_path, encoding="utf-8")))
        self.samples = [
            (base / r["filename"], r["notation"],
             r["colour"] == "True", int(r["value"]))
            for r in rows if r["split"] == split
        ]
        self.transform = transform or T.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, notation, colour, value = self.samples[idx]
        img = Image.open(path).convert("RGB")   # normalise to 3-channel
        return self.transform(img), notation
```

---

## .mat File Convention

| Image type | `.mat` shape | How to load |
|------------|-------------|-------------|
| Colour RGB | `(H×3, W)` | `np.loadtxt(f).reshape(H, W, 3)` |
| Greyscale | `(H, W)` | `np.loadtxt(f)` |

Values are in `[0.0, 1.0]` where `0.0 = dark` and `1.0 = bright`.

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| numpy | ≥ 1.21 | Array operations and background noise |
| Pillow | ≥ 9.0 | Text rendering and image saving |
| Python | ≥ 3.10 | `str \| None` type-hint syntax |

---

## File Summary

| File | Purpose |
|------|---------|
| `numgen.py` | Main script — run this to generate data |
| `Numbers_Data/` | All generated output — do not edit by hand |
| `manifest.csv` | Index of every generated sample |
| `numbers_README.md` | This file |