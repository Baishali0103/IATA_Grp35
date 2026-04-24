"""
Central configuration. Edit PROJECT_ROOT to point at the folder
that contains your Data/ folder.
"""
from pathlib import Path
import torch

# ==== Paths ====
# If Data/ sits next to this file, leave as-is.
# Otherwise set an absolute path, e.g.:
#   PROJECT_ROOT = Path(r"C:\Users\chirag\ITAITA_Project")
PROJECT_ROOT = Path(__file__).parent

DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "Model_Results"
OUTPUTS_DIR = PROJECT_ROOT / "outputs_dir"

DATASETS = ["Shapes_Data", "Numbers_Data", "TicTacToe_Data"]
SPLITS = ["train", "val", "test"]

# ==== Model hyperparameters ====
EMBED_DIM = 256
NUM_HEADS = 8
NUM_DECODER_LAYERS = 3
FF_DIM = 512
DROPOUT = 0.1
MAX_LENGTH = 150
IMAGE_SIZE = 128

# ==== Training ====
BATCH_SIZE = 32          # drop to 16 or 8 if VRAM runs out
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
GRAD_CLIP = 1.0
USE_AMP = True
SEED = 42

# ==== Device ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Evaluation ====
BERTSCORE_SUBSET = None  # set to e.g. 500 if BERTScore is too slow

# ==== Tokenizer ====
DEFAULT_TOKENIZER = "custom"
PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<UNK>"

# ==== LLM comparison ====
LLM_MODEL = "gpt-4o-mini"
LLM_PROMPT = "Describe this image in one sentence. Use the same style as the examples you see: shapes with colours and positions, numbers with colours and positions, or tic-tac-toe board states."