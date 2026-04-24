# Image-to-Sentence Mapping — Task 5 Pipeline

Group 35 · EMATM0067 Introduction to AI and Text Analytics · Spring 2026

An end-to-end pipeline that trains neural networks to describe synthetic
images (shapes, numbers, tic-tac-toe boards) in natural language. Supports
three CNN encoder variants, two tokenization strategies, multi-metric
evaluation, error analysis, and attention visualisation.

This README describes the code — the rationale for each design decision, and
how to run the pipeline end-to-end.

---

## 1. What this pipeline does

Given a folder of `(image, caption)` pairs across three datasets, the pipeline:

1. Loads and tokenizes captions (custom word-level or BERT WordPiece).
2. Trains a CNN + Transformer model to map images to captions.
3. Generates captions on the held-out test set.
4. Scores those captions with four standard metrics.
5. Categorises failure types and produces plots.
6. Extracts cross-attention weights and overlays them on images.

All scripts are reproducible and read their configuration from `config.py`.

---

## 2. Project structure

---

## 3. Design decisions and rationale (the "why")

This section explains why each major choice was made. Read this before
modifying anything.

### 3.1 CNN + Transformer, not CNN-RNN

A CNN encoder extracts spatial image features. A Transformer decoder
generates captions token-by-token while attending over those features.

Why not LSTM/GRU decoder? Transformers parallelise training better,
scale to longer captions (our Numbers dataset has captions up to 108 words),
and the cross-attention mechanism gives us a clean way to visualise
where the model looks when generating each word. This visualisation is
used in our error analysis and in the report.

### 3.2 Three CNN variants (first analytic axis)

- **ShallowCNN** (2 conv layers, ~215K params)
- **DeepCNN** (5 conv layers, ~1.0M params)
- **ResNetCNN** (2 residual blocks with skip connections, ~1.5M params)

All three produce the same output shape `[batch, 1024 patches, 256 dim]`
so they are drop-in comparable for the decoder. This isolates the encoder
as the only varying factor — a fair controlled comparison.

We expected ResNet to win due to skip connections aiding gradient flow.
Results confirmed this: ResNet 28.1 BLEU-4 > Deep 26.5 > Shallow 24.8.

### 3.3 Two tokenizers (second analytic axis)

- **Custom word-level tokenizer** — built from training captions. 3,102
  whole-word tokens + 4 specials (`<PAD>`, `<START>`, `<END>`, `<UNK>`).
- **BERT WordPiece tokenizer** — `bert-base-uncased`, 30,522 subword tokens.
  `[CLS]` and `[SEP]` are repurposed as start/end markers.

Both expose the same interface (`encode`, `decode`, `vocab_size`) so
`dataset.py` and `models.py` are tokenizer-agnostic.

Why both? To test whether richer subword vocabulary beats a compact
word-level one for this domain. Result: it depends on the metric.
BERT wins METEOR (semantic matching), custom wins BLEU-4 (n-gram
precision). This metric-dependent result is one of our key findings.

### 3.4 Image resolution 128×128

Generated images are 256×256 RGB. We downsample to 128×128 at load time.

Why? Our training hardware is an RTX 4050 Laptop GPU with 6 GB VRAM.
At 256×256 with batch size 16, the forward+backward pass exceeds
available memory. 128×128 with fp16 training fits comfortably.

This is acknowledged as a limitation — Numbers dataset in particular
suffers, as small digit strokes blur at low resolution. The report
discusses this.

### 3.5 Caption length 150 tokens

Our longest caption (Numbers, 108 words) plus `<START>` plus `<END>`
plus buffer fits in 150 tokens under the BERT tokenizer's worst-case
subword fragmentation. Captions shorter than 150 are padded; the
`ignore_index=pad_id` argument to `CrossEntropyLoss` ensures padding
contributes no loss.

### 3.6 All three datasets combined for training

Rather than training three specialist models, we combine all
~10,524 training images into one training set. The Transformer
decoder therefore sees shapes, numbers, and tic-tac-toe captions in
one mixed stream.

Why? Shared linguistic structure (colour words, position words)
transfers across datasets. One unified model also lets us test
whether a single architecture can handle all three tasks — a
legitimate research question. Evaluation still reports per-dataset
metrics so we can identify where the combined model struggles.

### 3.7 Training loop choices

- **Adam optimizer, lr=1e-3** — standard for Transformers.
- **Cosine annealing, T_max=NUM_EPOCHS** — smooth decay from 1e-3 to
  1e-5. Critical: `T_max=NUM_EPOCHS` (not a smaller value). An earlier
  prototype had `T_max=10` which caused the schedule to restart
  mid-training. Fixed here.
- **Mixed precision (AMP)** — halves memory use, no accuracy cost.
  Essential for 6 GB GPUs.
- **Gradient clipping at 1.0** — prevents occasional Transformer
  divergence. Without this, one bad batch can destabilise training.
- **Early stopping, patience=5** — stops when validation loss doesn't
  improve for 5 consecutive epochs. Saves compute and prevents
  overfitting.
- **Random seed = 42** — reproducibility across teammates' machines.

### 3.8 Four evaluation metrics

No single metric captures caption quality. We report four:

| Metric | What it measures | Range |
|---|---|---|
| Exact Match | Word-for-word string identity | 0-1 |
| BLEU-4 | Corpus n-gram overlap, up to 4-grams | 0-100 |
| METEOR | Unigram match with WordNet synonym awareness | 0-1 |
| BERTScore F1 | Contextual embedding cosine similarity | 0-1 |

Each catches different qualities. Disagreements between metrics are
themselves informative (e.g. our BLEU vs METEOR split on tokenizers).

### 3.9 Error categorisation

`src/error_analysis.py` reads saved predictions and classifies each
failure into types based on which attribute words differ between
prediction and reference:

- colour_error, size_error, shape_error, position_error
- length_error (caption length ratio < 0.5 or > 2.0)
- xo_content_error (TicTacToe: board-position word set differs)

These heuristics are imperfect — "big" vs "large" would register as a
size_error even though semantically equivalent. We acknowledge this in
the report and use these numbers to identify patterns, not as absolute
truth.

### 3.10 Cross-attention visualisation

`src/attention_viz.py` monkey-patches the decoder's `MultiheadAttention`
to expose weights on every forward pass, then uses a forward hook to
capture them at each generation step. The resulting `[n_tokens, n_patches]`
tensor is reshaped to the CNN's spatial grid (32×32) and bilinearly
upsampled to image size (128×128) for overlay.

This is how `attention_report_resnet.png` is produced.

---

## 4. Setup

### 4.1 Prerequisites

- Python 3.10 or newer
- Windows, macOS, or Linux
- NVIDIA GPU recommended. CPU works but training will be very slow.

### 4.2 Environment

```powershell
# In the project root
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate         # macOS/Linux

pip install -r requirements.txt
```

Verify CUDA is detected:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

If `False` on a machine with an NVIDIA GPU, install the CUDA build:

```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4.3 Data

The `Data/` folder is NOT in this repository (it's ~27 GB including
matrices/meta subfolders we don't need). Generate it locally or
copy only the `images/` and `text/` subfolders from a teammate's
build.

Required structure:


Data/
├── Shapes_Data/
│   ├── manifest.csv                  (optional but useful)
│   ├── train/
│   │   ├── images/.png
│   │   └── text/.txt
│   ├── val/ (same structure)
│   └── test/ (same structure)
├── Numbers_Data/   (same structure)
└── TicTacToe_Data/ (same structure)


Edit `config.py` if your data lives elsewhere:

```python
PROJECT_ROOT = Path(r"C:\path\to\your\project")
# DATA_DIR defaults to PROJECT_ROOT / "Data"
```

---

## 5. How to run the full pipeline

Run these in order. Each command prints progress and produces output
files listed below.

```powershell
# 1. Sanity check the data is where the code expects it
python scripts/diagnose_data.py

# 2. Build the custom tokenizer from training captions
python scripts/save_tokenizer.py
# Produces: Model_Results/custom_vocab.json

# 3. Train the three CNN variants (~30-60 min each on RTX 4050)
python scripts/run_training.py --approach shallow
python scripts/run_training.py --approach deep
python scripts/run_training.py --approach resnet

# 4. Train the BERT tokenizer variant (second analytic axis)
python scripts/run_training.py --approach resnet --tokenizer bert --tag bert

# 5. Evaluate all four models (~15-30 min total; BERTScore is the slow part)
python scripts/run_evaluation.py --checkpoints shallow,deep,resnet
python scripts/run_evaluation.py --checkpoints resnet --tokenizer bert --tag bert

# 6. Error analysis
python scripts/run_error_analysis.py --approaches shallow,deep,resnet

# 7. Attention visualisation for the best model
python scripts/run_attention_viz.py --approach resnet --n 4

# 8. Tokenizer comparison
python scripts/compare_tokenizers.py --approach resnet

# 9. Master summary for the report
python scripts/build_report_summary.py

# 10. (Optional) LLM comparison - requires OpenAI API key
# $env:OPENAI_API_KEY = "sk-..."
# python scripts/llm_compare.py --n 100 --stratify
```

---

## 6. Output files and what they contain

After the full pipeline, `outputs_dir/` contains:

### Headline files for the report

- **`REPORT_MASTER_SUMMARY.md`** — all metrics for all four models,
  per-dataset breakdown. This is the main table for Section 4.
- **`metrics_summary.csv`** — same data in machine-readable form.

### Tokenizer comparison

- **`tokenizer_comparison.md`** — side-by-side custom vs BERT per metric.
- **`tokenizer_comparison.png`** — grouped bar chart.
- **`tokenizer_vocab_stats.md`** — vocab sizes, UNK rate, subword rate.

### Error analysis

- **`error_summary_all.md`** — combined error-type table for shallow/deep/resnet.
- **`error_breakdown_<model>.png`** — stacked bar chart per model.
- **`length_vs_accuracy_<model>.png`** — accuracy by caption length.
- **`approach_comparison_exact_match.png`** — all models side by side.

### Attention

- **`attention_report_resnet.png`** — 4-row figure for the report
  showing how attention concentrates on relevant regions.
- **`attention_resnet/example_*_grid.png`** — per-example heatmaps.

### Per-model detail

- **`metrics_<model>.json`** — raw metric values.
- **`metrics_by_dataset_colour_<model>.csv`** — colour vs B&W breakdown.
- **`predictions_<model>_full.csv`** — every test prediction with
  reference and manifest metadata (not in git; regenerate via eval).

---

## 7. Hyperparameters (edit in `config.py`)

```python
EMBED_DIM = 256              # Transformer embedding size
NUM_HEADS = 8                # attention heads
NUM_DECODER_LAYERS = 3       # decoder depth
FF_DIM = 512                 # feed-forward layer size
DROPOUT = 0.1
MAX_LENGTH = 150             # max caption tokens
IMAGE_SIZE = 128             # input image size (px)

BATCH_SIZE = 16              # lower to 8 if OOM
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50              # early stopping usually fires by ~20
EARLY_STOPPING_PATIENCE = 5
GRAD_CLIP = 1.0
USE_AMP = True               # mixed precision
SEED = 42
BERTSCORE_SUBSET = None      # set to 500 if BERTScore is too slow
```

---

## 8. Known limitations

Documented here and discussed in the report's Conclusions section.

1. **Resolution bottleneck.** 128×128 input loses detail, particularly
   for distorted multi-digit numbers. Numbers dataset accuracy is
   substantially worse than the others partly because of this.

2. **Fluency-accuracy dissociation.** All four models produce captions
   in the right linguistic style but with frequently incorrect content,
   especially on TicTacToe (36 BLEU alongside 79% content errors).
   Suggests the decoder memorises language patterns faster than it
   grounds them in visual evidence.

3. **Greedy decoding only.** We use greedy decoding for efficiency.
   Beam search would likely add 2-4 BLEU-4 points but is not explored.

4. **Error categorisation is heuristic.** Synonyms like "big"/"large"
   register as mismatches despite semantic equivalence. Numbers may be
   inflated.

5. **Small training set per dataset.** ~3,500 training examples per
   dataset is small for captioning. Larger corpora would likely
   improve all metrics meaningfully.

---

## 9. Reproducibility

- All random seeds are fixed to 42 (`SEED` in `config.py`).
- Training uses deterministic cosine annealing from 1e-3 to 1e-5.
- Same data + same seed + same config should produce identical checkpoints.
- Exception: mixed precision introduces tiny non-determinism in floating
  point; metric values may differ by <0.1% across identical runs.

---

## 10. Branches and contributions

- `main` — group-reviewed version
- `chirag-evaluation` — evaluation pipeline, error analysis, BERT
  tokenizer comparison, attention visualisation (this work)
- Other branches — see teammates' contributions in the repo

---

## 11. References and third-party libraries

- **PyTorch** — model training
- **transformers** (Hugging Face) — BERT tokenizer
- **sacrebleu** — BLEU-4
- **NLTK** — METEOR with WordNet
- **bert-score** — BERTScore metric
- **matplotlib / pandas** — plots and tables

`bert-base-uncased` weights are downloaded automatically from
Hugging Face on first use.

---

## 12. Contact

For questions about this branch: Chirag Sharma (group member).
For questions about the data generation scripts: see teammates' branches.
