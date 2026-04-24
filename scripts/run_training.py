"""
Train one or all three CNN + Transformer variants.

Usage examples:
    # train one at a time (recommended first time)
    python scripts/run_training.py --approach shallow
    python scripts/run_training.py --approach deep
    python scripts/run_training.py --approach resnet

    # train all three in a row
    python scripts/run_training.py --all

    # train resnet with BERT tokenizer (the second analytic axis)
    python scripts/run_training.py --approach resnet --tokenizer bert --tag bert
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.dataset import load_raw_dataframes, ImageCaptionDataset
from src.tokenizer import build_tokenizer
from src.models import build_model, count_parameters
from src.train import train, set_seed


def train_one(approach: str, tokenizer, tag: str,
              train_loader, val_loader, device):
    print(f"\n=== Training {approach} (tag={tag or '(none)'}) ===")
    cnn, decoder = build_model(approach, vocab_size=tokenizer.vocab_size,
                               embed_dim=config.EMBED_DIM,
                               max_length=config.MAX_LENGTH,
                               num_heads=config.NUM_HEADS,
                               num_decoder_layers=config.NUM_DECODER_LAYERS,
                               ff_dim=config.FF_DIM, dropout=config.DROPOUT)
    cnn = cnn.to(device)
    decoder = decoder.to(device)

    print(f"CNN params:     {count_parameters(cnn):,}")
    print(f"Decoder params: {count_parameters(decoder):,}")

    params = list(cnn.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    save_name = f"{approach}_best.pt" if tag == "" else f"{approach}_{tag}_best.pt"
    save_path = config.RESULTS_DIR / save_name

    history = train(cnn, decoder, train_loader, val_loader, optimizer, scheduler,
                    criterion, device,
                    num_epochs=config.NUM_EPOCHS,
                    patience=config.EARLY_STOPPING_PATIENCE,
                    grad_clip=config.GRAD_CLIP, use_amp=config.USE_AMP,
                    save_path=str(save_path), verbose=True)
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", default=None,
                    choices=["shallow", "deep", "resnet"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tokenizer", default=config.DEFAULT_TOKENIZER,
                    choices=["custom", "bert"])
    ap.add_argument("--tokenizer-path", default=None)
    ap.add_argument("--tag", default="",
                    help="Optional suffix for checkpoint name, e.g. --tag bert")
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    set_seed(args.seed)
    device = config.DEVICE
    print(f"Device: {device}")

    # --- Tokenizer ---
    if args.tokenizer == "custom":
        tokenizer = build_tokenizer("custom")
        path = args.tokenizer_path or str(config.RESULTS_DIR / "custom_vocab.json")
        if Path(path).exists():
            tokenizer.load(path)
            print(f"Loaded custom tokenizer from {path}")
        else:
            raw = load_raw_dataframes(config.DATA_DIR)
            texts = []
            for df in raw.values():
                if df.empty:
                    continue
                texts.extend(df[df["split"] == "train"]["text"].tolist())
            tokenizer.fit(texts)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save(path)
            print(f"Fitted + saved custom tokenizer to {path}")
    else:
        tokenizer = build_tokenizer("bert")
    print(f"Tokenizer: {args.tokenizer}   vocab_size: {tokenizer.vocab_size}")

    # --- Data ---
    raw = load_raw_dataframes(config.DATA_DIR)
    train_df = pd.concat([df[df["split"] == "train"] for df in raw.values()
                          if not df.empty], ignore_index=True)
    val_df = pd.concat([df[df["split"] == "val"] for df in raw.values()
                        if not df.empty], ignore_index=True)
    print(f"Train: {len(train_df)}   Val: {len(val_df)}")

    train_ds = ImageCaptionDataset(train_df, tokenizer,
                                    image_size=config.IMAGE_SIZE,
                                    max_length=config.MAX_LENGTH, augment=False)
    val_ds = ImageCaptionDataset(val_df, tokenizer,
                                  image_size=config.IMAGE_SIZE,
                                  max_length=config.MAX_LENGTH, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    approaches = ["shallow", "deep", "resnet"] if args.all else [args.approach]
    if not any(approaches):
        raise SystemExit("Pass --approach <name> or --all.")

    for a in approaches:
        train_one(a, tokenizer, args.tag, train_loader, val_loader, device)


if __name__ == "__main__":
    main()