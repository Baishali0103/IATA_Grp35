"""
Side-by-side comparison of custom word-level vs BERT WordPiece tokenizer.

Reads evaluation JSONs for resnet (custom) and resnet_bert, then produces
comparison tables, bar charts, and vocabulary statistics for the report.

Usage (after evaluation has been run for both variants):
    python scripts/compare_tokenizers.py --approach resnet
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.tokenizer import build_tokenizer
from src.dataset import load_raw_dataframes, combine


def load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def flatten(metrics: dict, label: str) -> pd.DataFrame:
    rows = []
    if "overall" in metrics:
        rows.append({"variant": label, "split": "overall", **metrics["overall"]})
    for name, m in metrics.get("per_dataset", {}).items():
        rows.append({"variant": label, "split": name, **m})
    return pd.DataFrame(rows)


def vocab_stats():
    """Compare vocab size + UNK rate (custom) and subword rate (BERT)."""
    raw = load_raw_dataframes(config.DATA_DIR)
    train_df = combine(raw, "train")
    val_df = combine(raw, "val")
    test_df = combine(raw, "test")

    rows = []
    for kind in ("custom", "bert"):
        if kind == "custom":
            tok = build_tokenizer("custom")
            tok.fit(train_df["text"].tolist())

            def oov_rate(df):
                total, unk = 0, 0
                for t in df["text"]:
                    ids = tok.encode(t, max_length=config.MAX_LENGTH)
                    for i in ids:
                        if i == tok.pad_id:
                            continue
                        total += 1
                        if i == tok.unk_id:
                            unk += 1
                return unk / max(1, total)

            val_m, test_m = oov_rate(val_df), oov_rate(test_df)
            metric_name = "unk_rate"
        else:
            tok = build_tokenizer("bert")

            def subword_rate(df):
                extra, total_words = 0, 0
                for t in df["text"]:
                    words = t.split()
                    total_words += len(words)
                    ids = tok.encode(t, max_length=config.MAX_LENGTH)
                    non_special = sum(
                        1 for i in ids
                        if i not in (tok.pad_id, tok.start_id, tok.end_id))
                    extra += max(0, non_special - len(words))
                return extra / max(1, total_words)

            val_m, test_m = subword_rate(val_df), subword_rate(test_df)
            metric_name = "extra_subwords_per_word"

        rows.append({
            "tokenizer": kind,
            "vocab_size": tok.vocab_size,
            "val_metric": round(val_m, 4),
            "test_metric": round(test_m, 4),
            "metric_name": metric_name,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", default="resnet")
    ap.add_argument("--custom-tag", default="")
    ap.add_argument("--bert-tag", default="bert")
    args = ap.parse_args()

    custom_label = (args.approach if args.custom_tag == ""
                    else f"{args.approach}_{args.custom_tag}")
    bert_label = f"{args.approach}_{args.bert_tag}"

    custom_metrics = load_metrics(
        config.OUTPUTS_DIR / f"metrics_{custom_label}.json")
    bert_metrics = load_metrics(
        config.OUTPUTS_DIR / f"metrics_{bert_label}.json")

    if not custom_metrics:
        print(f"[error] metrics_{custom_label}.json missing.")
    if not bert_metrics:
        print(f"[error] metrics_{bert_label}.json missing.")
    if not custom_metrics or not bert_metrics:
        return

    df_custom = flatten(custom_metrics, "custom")
    df_bert = flatten(bert_metrics, "bert")
    combined = pd.concat([df_custom, df_bert], ignore_index=True)
    combined.to_csv(config.OUTPUTS_DIR / "tokenizer_comparison.csv", index=False)

    # Markdown report table
    md_path = config.OUTPUTS_DIR / "tokenizer_comparison.md"
    with open(md_path, "w") as f:
        f.write(f"# Tokenizer comparison ({args.approach} encoder)\n\n")
        f.write("Same encoder and decoder hyperparameters; only tokenizer "
                "differs.\n\n")
        for col in ("exact_match", "bleu4", "meteor", "bertscore_f1"):
            if col in combined.columns:
                pv = combined.pivot(index="split", columns="variant",
                                     values=col).round(4)
                f.write(f"## {col}\n\n")
                f.write(pv.to_markdown())
                f.write("\n\n")
    print(f"Saved {md_path}")

    # Grouped bar plot
    try:
        metrics_to_plot = [m for m in
                           ["exact_match", "bleu4", "meteor", "bertscore_f1"]
                           if m in combined.columns]
        fig, axes = plt.subplots(1, len(metrics_to_plot),
                                 figsize=(4.2 * len(metrics_to_plot), 4.5))
        if len(metrics_to_plot) == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics_to_plot):
            pv = combined.pivot(index="split", columns="variant", values=metric)
            pv.plot(kind="bar", ax=ax)
            ax.set_title(metric)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(config.OUTPUTS_DIR / "tokenizer_comparison.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved tokenizer_comparison.png")
    except Exception as e:
        print(f"[warn] plot failed: {e}")

    # Vocab stats
    print("\nComputing vocabulary statistics (this takes a moment)...")
    vs = vocab_stats()
    vs_path = config.OUTPUTS_DIR / "tokenizer_vocab_stats.md"
    with open(vs_path, "w") as f:
        f.write("# Tokenizer vocabulary statistics\n\n")
        f.write(vs.to_markdown(index=False))
        f.write("\n\n")
        f.write("**Interpretation:**\n\n")
        f.write("- `unk_rate` (custom): fraction of non-PAD tokens mapping to "
                "`<UNK>`. Lower is better.\n")
        f.write("- `extra_subwords_per_word` (BERT): subword splits per "
                "whitespace word. Higher means BERT needs more tokens per "
                "caption.\n")
    print(f"Saved {vs_path}")


if __name__ == "__main__":
    main()