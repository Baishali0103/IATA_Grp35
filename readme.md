# Neural Network Image-to-Sentence Mapping

**Repository:** IATA_Grp35 · Branch: `ux25878_thilokesh`

A neural network that looks at an image and generates a natural language sentence describing it. Three types of synthetic images are used — geometric shapes, multi-digit numbers, and tic-tac-toe board states — all trained in a single model with a CNN encoder and Transformer decoder.

---

## Repository Structure

```
IATA_Grp35/
├── Data_Gen/               # Data generation scripts
│   ├── numgen.py           # Generates number images and captions (dataset 1b)
│   ├── shapes.py           # Generates shape arrangement images (dataset 1a)
│   └── tictactoe.py        # Generates tic-tac-toe board images (dataset 1c)
├── Neural_Network_Image_to_Sentence_Mapping.ipynb   # Main training notebook
├── numgen_README.md        # Documentation for the number generator
├── shapes_README.md        # Documentation for the shapes generator
└── tictactoe_README.md     # Documentation for the tic-tac-toe generator
```

> Generated data, saved models, and config files are stored on Google Drive — see link below.

---

## Google Drive

All data, saved models, and config files are stored here:

**[ITAITA_Project on Google Drive](YOUR_GOOGLE_DRIVE_LINK_HERE)**

```
ITAITA_Project/
├── Config/
│   ├── word2idx.json               # Vocabulary: word → index mapping
│   ├── idx2word.json               # Vocabulary: index → word mapping
│   └── test_data_with_colour.json  # Test set metadata with colour flag
│
├── Data/
│   ├── Shapes_Data/                # Dataset 1a — geometric shapes
│   │   ├── train/
│   │   ├── val/
│   │   ├── test/
│   │   └── manifest.csv
│   ├── Numbers_Data/               # Dataset 1b — multi-digit numbers
│   │   ├── train/
│   │   ├── val/
│   │   ├── test/
│   │   └── manifest.csv
│   ├── TicTacToe_Data/             # Dataset 1c — tic-tac-toe boards
│   │   ├── train/
│   │   ├── val/
│   │   ├── test/
│   │   └── manifest.csv
│   ├── master_data.csv             # Combined manifest across all three datasets
│   └── master_encoded_data.csv     # Tokenised and encoded version of master_data.csv
│
└── Model_Results/
    ├── ShallowCNN_simple/          # Evaluation outputs — Shallow CNN, learned embeddings
    ├── ShallowCNN_bert/            # Evaluation outputs — Shallow CNN, BERT embeddings
    ├── DeepCNN_simple/             # Evaluation outputs — Deep CNN, learned embeddings
    ├── DeepCNN_bert/               # Evaluation outputs — Deep CNN, BERT embeddings
    ├── ResNetCNN_simple/           # Evaluation outputs — ResNet CNN, learned embeddings
    ├── ResNetCNN_bert/             # Evaluation outputs — ResNet CNN, BERT embeddings
    └── evaluation_results.json     # Aggregated evaluation metrics across all models
```

---

## Datasets

Each dataset produces matched (image, sentence) pairs in both colour and black-and-white variants.

| Dataset | Description | Example Sentence |
|---------|-------------|-----------------|
| **1a — Shapes** | Geometric shape arrangements with colour, size, and spatial relations | *a large blue rectangle containing a small green square at the bottom-right.* |
| **1b — Numbers** | Multi-digit numbers (up to 4 digits) with per-digit colour descriptions | *a medium sixty three at the middle-left and a medium six hundred sixty four at the bottom-centre appear in the scene.* |
| **1c — Tic-tac-toe** | Board state descriptions including moves, positions, and win conditions | *PLAYER_X has taken top right corner. PLAYER_O has gone for bottom left. PLAYER_X has won.* |

**Split sizes:** 5,000 train / 1,000 val / 1,000 test per dataset (15,000 / 3,000 / 3,000 total).

---

## Model Architecture

```
Image  →  CNN Encoder  →  Transformer Decoder  →  Sentence
```

Three CNN encoder variants are each paired with two text representation strategies, giving six models in total.

**CNN Encoders**

| Approach | Architecture | Key Feature |
|----------|-------------|-------------|
| **A — Shallow CNN** | 2 conv layers + pool + dense | Fast, minimal parameters |
| **B — Deep CNN** | 5 conv layers + pool + dense | More capacity, deeper features |
| **C — ResNet CNN** | 4 conv layers with skip connections | Residual learning, stable training |

**Text Representations**

| Variant | Description |
|---------|-------------|
| **Simple** | Token embeddings learned from scratch during training |
| **BERT** | Pre-trained BERT token embeddings fed into the decoder |

All six models share the same Transformer decoder and are trained under identical conditions (same data, random seed, and hyperparameters) to ensure a fair comparison.

---

## Results

Evaluated on 2,238 test samples across all three datasets.

| Model | Exact Match | BLEU-4 | METEOR | BERTScore |
|-------|-------------|--------|--------|-----------|
| ShallowCNN — simple | 0.0000 | 0.1398 | 0.4065 | 0.9086 |
| ShallowCNN — bert   | 0.0000 | 0.1542 | 0.4236 | 0.9216 |
| DeepCNN — simple    | 0.0000 | 0.1987 | 0.4807 | 0.9183 |
| DeepCNN — bert      | 0.0004 | 0.2190 | 0.4994 | 0.9292 |
| ResNetCNN — simple  | 0.0000 | 0.1895 | 0.4677 | 0.9172 |
| ResNetCNN — bert    | 0.0036 | 0.2371 | 0.5040 | 0.9299 |

Per-dataset and colour vs B&W breakdowns are available in `Model_Results/evaluation_results.json`.

---

## How to Run

### 1. Generate data

Run each generator script from the `Data_Gen/` directory:

```bash
python shapes.py
python numgen.py
python tictactoe.py
```

Each script produces a structured output folder with colour and B&W images and a `manifest.csv`. See the individual README files (`shapes_README.md`, `numgen_README.md`, `tictactoe_README.md`) for full options.

### 2. Train and evaluate

Open `Neural_Network_Image_to_Sentence_Mapping.ipynb` in Google Colab (T4 GPU recommended). Mount your Google Drive and set the project root path at the top of the notebook. The notebook covers:

- Preprocessing and tokenisation
- DataLoader setup
- Training all six model variants
- Evaluation with Exact Match, BLEU-4, METEOR, and BERTScore
- Attention heatmap visualisation

---

## Notes

- All images are converted to RGB internally (greyscale images have their single channel repeated 3 times)
- Numbers are converted to words during preprocessing (e.g. `63` → `sixty three`) to allow the model to learn digit names
- Vocabulary is built from training data only — no data leakage into val/test
- Max sequence length is set to 150 tokens to accommodate number-to-word expansion
- Random seeds are fixed for reproducibility across all six training runs