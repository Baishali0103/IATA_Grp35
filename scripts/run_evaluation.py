"""
Run evaluation on trained checkpoints. Produces metric tables + raw predictions.

Usage:
    # Evaluate all three custom-tokenizer models
    python scripts/run_evaluation.py --checkpoints shallow,deep,resnet

    # Evaluate the BERT-tokenizer variant of ResNet
    python scripts/run_evaluation.py --checkpoints resnet --tokenizer bert --tag bert

Outputs (in outputs_dir/):
    predictions_<label>.csv              raw predictions (for error analysis)
    predictions_<label>_full.csv         predictions + manifest metadata
    metrics_<label>.json                 overall + per-dataset metrics
    metrics_by_colour_<label>.csv        colour vs B&W breakdown
    metrics_by_dataset_colour_<label>.csv per-dataset + per-colour breakdown
    metrics_summary.csv                  combined table (all runs)
    metrics_summary.md                   markdown version for the report
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.dataset import ImageCaptionDataset, load_raw_dataframes, combine
from src.tokenizer import build_tokenizer
from src.models import build_model
from src.evaluate import (
    generate_predictions, compute_all_metrics,
    results_to_dataframe, save_predictions, save_results,
    exact_match, bleu4, meteor, bertscore,
)


def breakdown_by_colour(preds, refs, colour_flags, device):
    out = {}
    for label, mask_val in (("colour", True), ("bw", False)):
        idx = [i for i, c in enumerate(colour_flags) if c == mask_val]
        if not idx:
            continue
        p = [preds[i] for i in idx]
        r = [refs[i] for i in idx]
        print(f"  breakdown: {label} (n={len(p)})")
        bs = bertscore(p, r, device=device)
        out[label] = {
            "n": len(p),
            "exact_match": exact_match(p, r),
            "bleu4": bleu4(p, r),
            "meteor": meteor(p, r),
            "bertscore_f1": bs["F1"],
        }
    return out


def evaluate_one(approach, tag, tokenizer, test_df, test_loader, device):
    ckpt_name = f"{approach}_best.pt" if tag == "" else f"{approach}_{tag}_best.pt"
    ckpt_path = config.RESULTS_DIR / ckpt_name
    if not ckpt_path.exists():
        print(f"[skip] checkpoint missing: {ckpt_path}")
        return None

    label = approach if tag == "" else f"{approach}_{tag}"
    print(f"\n=== Evaluating {label} ===")

    cnn, decoder = build_model(approach, vocab_size=tokenizer.vocab_size,
                               embed_dim=config.EMBED_DIM,
                               max_length=config.MAX_LENGTH,
                               num_heads=config.NUM_HEADS,
                               num_decoder_layers=config.NUM_DECODER_LAYERS,
                               ff_dim=config.FF_DIM, dropout=config.DROPOUT)
    cnn = cnn.to(device)
    decoder = decoder.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    cnn.load_state_dict(ckpt["cnn"])
    decoder.load_state_dict(ckpt["decoder"])

    preds, refs = generate_predictions(cnn, decoder, test_loader, tokenizer, device,
                                        max_length=config.MAX_LENGTH)
    dataset_labels = test_df["dataset"].tolist()

    save_predictions(preds, refs, dataset_labels,
                     config.OUTPUTS_DIR / f"predictions_{label}.csv")

    # predictions + metadata for richer error analysis
    preds_df = test_df[["dataset", "file_name", "text"]].copy()
    preds_df["prediction"] = preds
    preds_df.rename(columns={"text": "reference"}, inplace=True)
    for col in ("colour", "has_overlap", "distortion_level", "crowding",
                "shape_mode", "num_moves", "winner", "ambiguous"):
        if col in test_df.columns:
            preds_df[col] = test_df[col].values
    preds_df.to_csv(config.OUTPUTS_DIR / f"predictions_{label}_full.csv", index=False)

    # core metrics + per-dataset breakdown
    results = compute_all_metrics(preds, refs, dataset_labels,
                                  bertscore_subset=config.BERTSCORE_SUBSET,
                                  device=str(device))
    save_results(results, config.OUTPUTS_DIR / f"metrics_{label}.json")

    # colour / B&W breakdown
    if "colour" in test_df.columns:
        col_flags = test_df["colour"].tolist()
        colour_results = breakdown_by_colour(preds, refs, col_flags, str(device))
        pd.DataFrame(colour_results).T.to_csv(
            config.OUTPUTS_DIR / f"metrics_by_colour_{label}.csv")

        rows = []
        for ds in sorted(set(dataset_labels)):
            for col_flag, col_name in ((True, "colour"), (False, "bw")):
                idx = [i for i, (d, c) in enumerate(zip(dataset_labels, col_flags))
                       if d == ds and c == col_flag]
                if not idx:
                    continue
                p = [preds[i] for i in idx]
                r = [refs[i] for i in idx]
                bs = bertscore(p, r, device=str(device))
                rows.append({
                    "dataset": ds,
                    "variant": col_name,
                    "n": len(p),
                    "exact_match": exact_match(p, r),
                    "bleu4": bleu4(p, r),
                    "meteor": meteor(p, r),
                    "bertscore_f1": bs["F1"],
                })
        pd.DataFrame(rows).to_csv(
            config.OUTPUTS_DIR / f"metrics_by_dataset_colour_{label}.csv",
            index=False)

    df = results_to_dataframe(results)
    df.insert(0, "approach", label)
    print(df.to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="shallow,deep,resnet")
    ap.add_argument("--tokenizer", default=config.DEFAULT_TOKENIZER,
                    choices=["custom", "bert"])
    ap.add_argument("--tokenizer-path", default=None)
    ap.add_argument("--tag", default="",
                    help="Optional checkpoint suffix, e.g. --tag bert")
    args = ap.parse_args()

    device = config.DEVICE
    print(f"Device: {device}")

    if args.tokenizer == "custom":
        path = args.tokenizer_path or str(config.RESULTS_DIR / "custom_vocab.json")
        if not Path(path).exists():
            raise SystemExit(f"Custom tokenizer not found at {path}. "
                             "Run scripts/save_tokenizer.py first.")
        tokenizer = build_tokenizer("custom")
        tokenizer.load(path)
    else:
        tokenizer = build_tokenizer("bert")
    print(f"Tokenizer: {args.tokenizer}   vocab_size: {tokenizer.vocab_size}")

    raw = load_raw_dataframes(config.DATA_DIR)
    test_df = combine(raw, "test")
    print(f"Test set size: {len(test_df)}")
    print(test_df["dataset"].value_counts().to_string())
    if "colour" in test_df.columns:
        print("\nColour / B&W split:")
        print(test_df.groupby(["dataset", "colour"]).size().to_string())

    test_ds = ImageCaptionDataset(test_df, tokenizer,
                                   image_size=config.IMAGE_SIZE,
                                   max_length=config.MAX_LENGTH, augment=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=0)

    all_dfs = []
    for approach in args.checkpoints.split(","):
        approach = approach.strip()
        if not approach:
            continue
        df = evaluate_one(approach, args.tag, tokenizer, test_df, test_loader, device)
        if df is not None:
            all_dfs.append(df)

    if all_dfs:
        summary = pd.concat(all_dfs, ignore_index=True)
        summary.to_csv(config.OUTPUTS_DIR / "metrics_summary.csv", index=False)
        with open(config.OUTPUTS_DIR / "metrics_summary.md", "w") as f:
            f.write("# Evaluation summary\n\n")
            f.write(summary.to_markdown(index=False))
        print(f"\nSaved combined summary to {config.OUTPUTS_DIR}")


if __name__ == "__main__":
    main()