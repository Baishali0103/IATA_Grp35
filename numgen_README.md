# numgen.py — Number Image Generator (v2)

## What Does This Do?

This script creates a large collection of **number images** paired with
**plain English descriptions** of what is in each image.

For example, it might produce an image of the number **68** where the 6 is
red and the 8 is blue, slightly wobbly as if handwritten, sitting next to
a small green **42** — and alongside that image it saves the sentence:

> *"Two numbers are visible: a large 68, with 6 in red and 8 in blue, at the
> top-left, and a small green 42 at the bottom-right."*

These paired (image, sentence) sets are used to **train AI models** to look
at an image and describe what they see.

---

## What Kind of Images Does It Make?

Each image is **256 × 256 pixels** (you can change this) and can contain:

- **1 to 4 numbers** placed freely around the canvas
- Numbers with **1 to 4 digits** (e.g. 7, 42, 318, 1337)
- Numbers in **8 different colours**: red, blue, green, yellow, purple,
  orange, pink, cyan
- Each digit within a number can be a **different colour**
  (e.g. the 6 is red and the 8 is blue)
- Three **sizes**: small, medium, large
- Numbers that **partially overlap** each other (with different colours used
  so you can still tell them apart)
- A **handwriting-style distortion** that makes the numbers look wobbly and
  organic, like they were written by hand

---

## What Makes the Images Look Realistic?

The script applies several effects to avoid making images look too
computer-perfect:

| Effect | What it does |
|--------|-------------|
| **Elastic deformation** | Warps the number slightly, like wobbly handwriting |
| **Rotation and tilt** | Tilts the number by a small random angle |
| **Localised blur** | Blurs one part of the number (e.g. just the top of an 8) while the rest stays sharp — simulating uneven pen pressure |
| **Drop shadow** | Adds a faint shadow behind some numbers (colour images only) |
| **Background noise** | Adds tiny random speckles to mimic paper texture |

You control how strong all of this is using the **distortion level** (0 = clean
and sharp, 1 = heavily distorted).

---

## Background Styles

The background behind the numbers is not always plain white. Four styles are
available:

| Style | Description |
|-------|-------------|
| **plain** | Simple off-white background |
| **ruled** | Faint horizontal lines, like lined notebook paper |
| **grid** | Faint grid lines, like graph paper |
| **tinted** | A soft colour tint — cream, pale yellow, rose, sage, or blue-grey |

---

## Colour vs Black-and-White

The generator can produce either **colour (RGB)** or **black-and-white
(greyscale)** images.

- **Colour images** — the sentence includes colour words ("a large blue 42")
- **B&W images** — colour words are left out, since you cannot see the colour
  ("a large 42")

This is deliberate. Training on B&W forces the AI to recognise shapes and
positions without relying on colour as a shortcut.

---

## How Dense Are the Scenes?

The **crowding** setting controls how many numbers appear and how closely
packed they are:

| Crowding | Typical number count | Overlap allowed |
|----------|---------------------|-----------------|
| **low** | 1–2 | Very little |
| **medium** | 2–3 | Moderate |
| **high** | 3–4 | A lot |

When numbers overlap, the script automatically checks that their colours
are different enough to stay readable.

---

## What Files Does It Save?

For every image generated, the script saves **four files** and one line in
a summary spreadsheet:

```
Numbers_Data/
│
├── manifest.csv              ← Summary spreadsheet of all samples
│
├── train/                    ← 70% of all samples (used for training)
│   ├── text/                 ← Plain .txt file with the English sentence
│   ├── meta/                 ← .meta file with extra details
│   ├── images/               ← The actual .png image
│   └── matrices/             ← The image saved as numbers in a .mat file
│
├── val/                      ← 15% of samples (used for validation)
│   └── …
│
└── test/                     ← 15% of samples (used for testing)
    └── …
```

The 70 / 15 / 15 split is standard practice in machine learning — you train
on one chunk, tune on another, and test on the last.

---

## The Summary Spreadsheet (manifest.csv)

Every generated sample gets one row in `manifest.csv`. Here is what each
column means:

| Column | What it means |
|--------|--------------|
| `filename` | Where the image file is saved |
| `notation` | Short code describing the image, e.g. `68:red\|blue,large` |
| `num_numbers` | How many numbers are in the image |
| `values` | The actual number values, e.g. `68,42` |
| `colors` | The colour(s) used, e.g. `red\|blue,green` |
| `sizes` | Size of each number: small, medium, or large |
| `num_digits_list` | How many digits each number has, e.g. `2,2` |
| `background_type` | plain, ruled, grid, or tinted |
| `crowding` | low, medium, or high |
| `distortion_level` | How distorted the image is, from 0.0 (clean) to 1.0 (heavy) |
| `colour` | True = colour image, False = black-and-white |
| `has_overlap` | True if any two numbers overlap |
| `occlusion_pcts` | What fraction of each number is hidden behind another one |
| `ambiguous` | True if a number is more than 50% hidden (hard even for humans) |
| `split` | train, val, or test |

---

## The Notation Format

Each image has a short **notation code** that describes it precisely.
It is designed to be quick to read and easy to parse by code.

```
<value>:<colour>,<size>
```

**Examples:**

| Notation | Meaning |
|----------|---------|
| `5:red,large` | A large red 5 |
| `67:blue,small` | A small blue 67 |
| `68:red\|blue,large` | A large 68 where the 6 is red and the 8 is blue |
| `67:blue,large \| 42:green,small` | Two numbers: a large blue 67 and a small green 42 |

---

## How To Install and Run

### Step 1 — Make sure you have Python 3.10 or newer

```bash
python --version
```

### Step 2 — Install the two required libraries

```bash
pip install numpy Pillow, tqdm
```

### Step 3 — Run the script

```bash
python numgen.py
```

The script will print progress as it goes. All folders are created
automatically. Generating **1 000 images** at 256 × 256 pixels takes
roughly **1–3 minutes** on a typical laptop.

### Step 4 — Change how many images are made

Open `numgen.py` and find these two lines near the bottom:

```python
TOTAL    = 1000   # change this number to make more or fewer images
IMG_SIZE = 256    # change to 512 for sharper, more detailed images
```

---

## How Are the 1 000 Images Balanced?

The generator cycles through **24 combinations** of settings so that the
dataset is evenly spread:

```
crowding         : low, medium, high          (3 options)
distortion_level : 0.0, 0.35, 0.70, 1.0      (4 options)
colour           : True, False                 (2 options)
```

3 × 4 × 2 = **24 combinations**, each getting roughly 41 samples.
The background style and exact number values are randomised within each
combination to add further variety.

---

## Using the Generator in Your Own Code

You do not have to run the full loop. You can call individual functions
directly:

```python
from numgen import random_scene, describe, render_scene, save_sample

# Make a single clean colour scene with one large number
scene    = random_scene(num_numbers=1, crowding="low",
                        distortion_level=0.0, colour=True)
sentence = describe(scene)
arr      = render_scene(scene)   # numpy array, shape (256, 256, 3)
print(sentence)

# Make a heavily distorted B&W scene with two overlapping numbers
scene    = random_scene(num_numbers=2, crowding="high",
                        distortion_level=0.9, colour=False)
sentence = describe(scene)
arr      = render_scene(scene)   # numpy array, shape (256, 256)

# Save both to disk with the standard folder structure
save_sample(n=0, scene=scene, sentence=sentence, split="train")
```

---

## Loading the Data in PyTorch

```python
import csv
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class NumbersDataset(Dataset):
    def __init__(self, manifest_path, split="train", transform=None):
        base = Path(manifest_path).parent
        rows = list(csv.DictReader(open(manifest_path, encoding="utf-8")))
        self.samples = [
            (base / r["filename"], r["notation"])
            for r in rows if r["split"] == split
        ]
        self.transform = transform or T.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, notation = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), notation
```

---

## Reading the .mat Files

Each image is also saved as a plain text file of numbers (a `.mat` file)
in case you prefer to load data that way.

| Image type | Shape of the .mat file | How to load it |
|------------|------------------------|----------------|
| Colour | rows = height × 3, columns = width | `np.loadtxt(f).reshape(H, W, 3)` |
| Black-and-white | rows = height, columns = width | `np.loadtxt(f)` |

All values are between **0.0** (black) and **1.0** (white).

---

## Requirements

| Library | Minimum version | What it is used for |
|---------|----------------|---------------------|
| Python | 3.10 | The programming language |
| numpy | 1.21 | Maths and image arrays |
| Pillow | 9.0 | Drawing and saving images |

---

## File Summary

| File | What it is |
|------|-----------|
| `numgen.py` | The main script — run this |
| `Numbers_Data/` | All generated output — do not edit by hand |
| `manifest.csv` | Summary spreadsheet of every sample |
| `numbers_README.md` | This file |