"""
Runs error analysis on saved predictions.

Usage:
    python scripts/run_error_analysis.py --approaches shallow,deep,resnet

Optionally include BERT variant:
    python scripts/run_error_analysis.py --approaches shallow,deep,resnet,resnet_bert
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.error_analysis import (
    analyse, summary_counts, plot_error_breakdown,
    plot_length_vs_accuracy, plot_approach_comparison,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approaches", default="shallow,deep,resnet")
    args = ap.parse_args()

    approaches = [a.strip() for a in args.approaches.split(",") if a.strip()]
    all_dfs = {}

    for approach in approaches:
        pred_path = config.OUTPUTS_DIR / f"predictions_{approach}.csv"
        if not pred_path.exists():
            print(f"[skip] {pred_path} not found. Run run_evaluation.py first.")
            continue

        print(f"\n=== Error analysis: {approach} ===")
        preds_df = pd.read_csv(pred_path).fillna("")
        df = analyse(
            preds_df["prediction"].tolist(),
            preds_df["reference"].tolist(),
            preds_df["dataset"].tolist(),
            approach_label=approach,
        )
        df.to_csv(config.OUTPUTS_DIR / f"errors_{approach}.csv", index=False)

        summary = summary_counts(df, group_by="dataset")
        print(summary.to_string())
        summary.to_csv(config.OUTPUTS_DIR / f"error_summary_{approach}.csv")

        plot_error_breakdown(
            df, config.OUTPUTS_DIR / f"error_breakdown_{approach}.png",
            title=f"Error types per dataset ({approach})",
        )
        plot_length_vs_accuracy(
            df, config.OUTPUTS_DIR / f"length_vs_accuracy_{approach}.png",
            title=f"Accuracy vs reference length ({approach})",
        )
        all_dfs[approach] = df

    if len(all_dfs) > 1:
        plot_approach_comparison(
            all_dfs, config.OUTPUTS_DIR / "approach_comparison_exact_match.png",
            metric="exact_match",
        )

    if all_dfs:
        md = config.OUTPUTS_DIR / "error_summary_all.md"
        with open(md, "w") as f:
            f.write("# Error analysis summary\n\n")
            for name, df in all_dfs.items():
                f.write(f"## {name}\n\n")
                f.write(summary_counts(df, group_by="dataset").to_markdown())
                f.write("\n\n")
        print(f"\nWrote combined summary: {md}")


if __name__ == "__main__":
    main()