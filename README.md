# TicTacToe Data Generator

## What This Project Does

This project is a **data generator**.
It automatically creates training data for a neural network.

The goal of the neural network is simple:
- **Input**:  A picture of a Tic-Tac-Toe board
- **Output**: A sentence in English that describes what is in the picture

This generator creates thousands of matched pairs of (picture, sentence) automatically.
You do not need to draw any pictures by hand or write any sentences by hand.

---

## What Is Tic-Tac-Toe?

Tic-Tac-Toe is a simple two-player game played on a 3x3 grid.
- One player uses the symbol **X**
- The other player uses the symbol **O**
- Players take turns placing their symbol in an empty cell
- The first player to place three symbols in a row (horizontal, vertical, or diagonal) wins
- If all 9 cells are filled and nobody has three in a row, the game is a draw

Example board:

```
 X | O | X
---+---+---
   | X |
---+---+---
 O |   | O
```

---

## Board Position Names

Every cell on the board has a name. We use these names throughout the code and in the output files.

```
 TL | TM | TR
----+----+----
 ML |  C | MR
----+----+----
 BL | BM | BR
```

| Name | Meaning            |
|------|--------------------|
| TL   | Top Left           |
| TM   | Top Middle         |
| TR   | Top Right          |
| ML   | Middle Left        |
| C    | Center             |
| MR   | Middle Right       |
| BL   | Bottom Left        |
| BM   | Bottom Middle      |
| BR   | Bottom Right       |

There are also aliases (alternative names for the same cell):
- `TC` = `TM` (Top Center = Top Middle)
- `BC` = `BM` (Bottom Center = Bottom Middle)
- `M`  = `C`  (Middle = Center)
- `MM` = `C`  (Middle Middle = Center)

---

## Project Structure

After you run the generator, the following folders and files will be created automatically:

```
TicTacToe_Data/
│
├── manifest.csv              ← Index file listing every sample
│
├── train/                    ← Training data (70% of all samples)
│   ├── text/                 ← .txt files (one sentence per file)
│   ├── meta/                 ← .meta files (sentence + notation + ASCII board)
│   ├── images/               ← .png files (pictures of the board)
│   └── matrices/             ← .mat files (the picture as a grid of numbers)
│
├── val/                      ← Validation data (15% of all samples)
│   ├── text/
│   ├── meta/
│   ├── images/
│   └── matrices/
│
└── test/                     ← Test data (15% of all samples)
    ├── text/
    ├── meta/
    ├── images/
    └── matrices/
```

### What each file type contains

| Extension | Contents                                                                 |
|-----------|--------------------------------------------------------------------------|
| `.txt`    | One English sentence describing the board                                |
| `.meta`   | The sentence + the board notation + an ASCII drawing of the board        |
| `.png`    | A picture of the board (may be black-and-white or colour)                |
| `.mat`    | The picture stored as a matrix of numbers between 0.0 and 1.0            |

### What the manifest.csv contains

The `manifest.csv` file is a table that lists every single sample in one place.
It has these columns:

| Column     | Meaning                                                               |
|------------|-----------------------------------------------------------------------|
| filename   | Path to the image file, relative to TicTacToe_Data/                  |
| notation   | The board position in text notation e.g. `X:TL,C O:TR`               |
| colour     | True = colour image, False = black and white image                    |
| num_moves  | Total number of moves played (0 to 9)                                 |
| winner     | `X`, `O`, or `none` if nobody has won yet                             |
| is_draw    | True if all 9 cells are filled and nobody won                         |
| split      | Which split the sample belongs to: `train`, `val`, or `test`          |

---

## How To Install and Run

### Step 1 — Make sure you have Python installed

This project requires **Python 3.10 or higher**.
You can check your version by running:

```bash
python --version
```

### Step 2 — Install the required libraries

This project uses three external libraries. Install them with this single command:

```bash
pip install numpy Pillow
```

All other imports (`math`, `random`, `dataclasses`, `pathlib`, `csv`) are part of
Python's standard library. You do not need to install them.

### Step 3 — Run the generator

Navigate to the folder where `tictactoe_generator.py` is saved, then run:

```bash
python tictactoe_generator.py
```

The script will print progress to the terminal and save all files automatically.
You do not need to create any folders first — the script creates them for you.

### Step 4 — Change how many samples are generated (optional)

Open the file and find this line near the bottom:

```python
TOTAL = 1000
```

Change `1000` to any number you want.
The samples will be split automatically: 70% train, 15% val, 15% test.

---

## How The Code Works — Step By Step

### 1. Board Representation (`Board` class)

A board is stored as two lists of position names:
- `x` — list of positions where X has played, e.g. `["TL", "C", "TM"]`
- `o` — list of positions where O has played, e.g. `["TR", "BL"]`

The board automatically checks:
- No position appears twice
- X has equal or one more move than O (because X always goes first)

### 2. Notation Format

Boards are written as a short text string. Examples:

```
"X:C"              → X has only played center
"X:TL,C O:TR"      → X played top-left and center; O played top-right
"X:TL,C,BM O:TR,ML"→ X has 3 moves, O has 2 moves
""                 → Empty board, no moves yet
```

### 3. Generating Random Boards (`random_board`)

The generator creates random boards that are **reachable** — meaning they could
actually appear in a real game.

The `_is_reachable` function checks that nobody won the game before the last move
was played. For example, if X already has three in a row but O has played more
moves after that, the position is not reachable.

### 4. Balanced Dataset Generation

To make sure the training data is balanced, samples are generated in groups:

- **By number of moves**: Equal numbers of boards with 0, 1, 2, ... 9 total moves
- **By outcome**: The generator cycles through `x_win`, `o_win`, `draw`,
  `in_progress` equally
- **By colour**: Exactly half of all images are colour, half are black and white

### 5. Generating Sentences (`describe`)

Each board gets an English description. The sentence has several parts:

1. **X's moves**: e.g. `"Seán is X and has taken the center and top left corner"`
2. **O's moves**: e.g. `"Niamh is O and has gone for top right corner"`
3. **Game state**: e.g. `"it is Seán's turn"` or `"5 moves have been played"`
4. **Relative description**: e.g. `"Seán controls the center"` or `"Niamh has 2 corners"`
5. **Outcome**: e.g. `"Seán has won"` or `"the game is a draw"`

To add variety, each part has multiple phrasings chosen randomly.
Player names are chosen randomly from a list of Irish names:
`Seán, Eoin, Niamh, Aoife, Siobhán, Saoirse, Liam`

For colour images, the sentence uses `"red X"` and `"blue O"` instead of just
`"X"` and `"O"`.

### 6. Generating Images (`render_board_image`)

Each board is drawn as a picture. Several sources of randomness are added to
make the pictures look different from each other:

| Feature                  | Description                                                         |
|--------------------------|---------------------------------------------------------------------|
| Variable margin          | The empty border around the grid is slightly different each time    |
| Variable line thickness  | Grid lines and symbols are drawn with different thicknesses         |
| Variable symbol size     | X and O symbols are slightly bigger or smaller each time            |
| Random rotation          | The grid and each symbol is slightly rotated                        |
| Random translation       | The grid and each symbol is shifted by a few pixels                 |
| Perspective distortion   | A small warp is applied to simulate a photo taken at a slight angle |
| Background noise         | Faint random noise is added to simulate paper texture               |
| Edge blurring            | The edges of lines and symbols are softened slightly                |

For **colour images**:
- X is drawn in **red**
- O is drawn in **blue**
- Grid lines are drawn in **black**

For **black and white images**:
- Everything is drawn in **black** on a white background

### 7. Saving Files (`save_test_files`)

For each sample, four files are saved and one row is added to `manifest.csv`.
The split (`train`, `val`, or `test`) determines which subfolder the files go into.

---

## Example Output

**Sentence (.txt file):**
```
Seán is red X and has taken the center and top left corner; Niamh is blue O
and has gone for top right corner; it is Niamh's turn; Seán controls the center.
```

**Notation + ASCII board (.meta file):**
```
Seán is red X and has taken the center and top left corner; ...
X:C,TL O:TR
 X |   | O
---+---+---
   | X |
---+---+---
   |   |
```

**Image (.png file):**
A 500x500 pixel picture of the board with the symbols drawn in the correct positions.

**Matrix (.mat file):**
A grid of numbers (500 rows × 500 columns for B&W, 1500 rows × 500 columns for colour)
where each number is between 0.0 and 1.0. This is the numerical version of the image
used directly by the neural network.

---

## Important Notes for Neural Network Training

- The `.mat` files and `.png` files contain the **same information** in different formats.
  Use `.png` files if you load images with a standard library (e.g. PyTorch `torchvision`).
  Use `.mat` files if you want to load raw numerical arrays directly.

- For **colour images**, the `.mat` file has shape `(1500, 500)` — this is the
  `(500, 500, 3)` colour array reshaped so it can be saved as a flat text file.
  When loading, reshape it back: `arr.reshape(500, 500, 3)`.

- For **black and white images**, the `.mat` file has shape `(500, 500)`.
  Values close to `1.0` are ink (dark). Values close to `0.0` are background (white).
  Note: colour images use the opposite convention — values close to `1.0` are bright.

- Use `manifest.csv` to build your PyTorch `Dataset` class. Load the CSV once,
  filter by the `split` column, and use the `filename` column to find each image.

---

## Frequently Asked Questions

**Q: Do I need a GPU to run the generator?**
No. The generator only uses the CPU. A GPU is only needed when you train the neural network.

**Q: How long does it take to generate 1000 samples?**
On a typical laptop, generating 1000 samples takes approximately 1 to 3 minutes.

**Q: Can I run the generator multiple times?**
Yes, but new samples will overwrite old ones if they have the same sample number.
Change the starting index in the loop or delete the old data folder first.

**Q: Why are Irish names used?**
The original code used Irish names. They contain accented characters (e.g. Seán, Siobhán)
which also tests that the text files are saved correctly in UTF-8 encoding.

**Q: What does UTF-8 mean?**
UTF-8 is a way of storing text that supports characters from all languages including
Chinese, Irish, Arabic, etc. All text files in this project are saved in UTF-8.

---

## Dependencies

| Library  | Version  | Purpose                              | Install command       |
|----------|----------|--------------------------------------|-----------------------|
| numpy    | >= 1.21  | Matrix operations and noise          | `pip install numpy`   |
| Pillow   | >= 9.0   | Drawing images                       | `pip install Pillow`  |
| Python   | >= 3.10  | Required for `str | None` type hints | (install Python)      |

---

## File Summary

| File                      | Purpose                                      |
|---------------------------|----------------------------------------------|
| `tictactoe_generator.py`  | The main script — run this to generate data  |
| `TicTacToe_Data/`         | All generated data — do not edit by hand     |
| `manifest.csv`            | Index of all generated samples               |
| `README.md`               | This file                                    |

---

*This project was written in Python. All random choices use Python's built-in
`random` module seeded by the system clock, so each run produces different data.*