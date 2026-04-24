"""
Error analysis: categorise failures into interpretable types per dataset.

Produces tables like:
    | dataset   | colour_err | size_err | position_err | count_err | length_err |
    | Shapes    |   18.3%    |  12.4%   |    41.9%     |   7.2%    |    5.1%    |
    ...

Used for the Discussion section of the report.
"""
from typing import List, Dict, Optional
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COLOUR_WORDS = {"red", "blue", "green", "yellow", "purple", "pink", "orange",
                "black", "white", "grey", "gray", "brown", "cyan", "magenta"}

SIZE_WORDS = {"large", "medium", "small", "big", "little", "tiny", "huge",
              "mid-sized", "medium-sized", "sizeable"}

SHAPE_WORDS = {"circle", "square", "triangle", "rectangle", "pentagon", "star",
               "sphere", "cube", "cylinder", "cone", "pyramid"}

POSITION_WORDS = {"top", "bottom", "left", "right", "centre", "center", "middle",
                  "corner", "side", "upper", "lower", "top-left", "top-right",
                  "bottom-left", "bottom-right", "top-centre", "bottom-centre",
                  "middle-left", "middle-right"}

XO_POSITION_WORDS = {"corner", "row", "centre", "center", "middle", "top",
                     "bottom", "left", "right"}


def _words(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    return [w for w in text.split() if w]


def _multiset_mismatch(a_words, b_words, vocab):
    ca = Counter(w for w in a_words if w in vocab)
    cb = Counter(w for w in b_words if w in vocab)
    return (ca - cb) + (cb - ca) != Counter()


def _norm(s): return " ".join(str(s).lower().strip().split())


def classify_errors(pred, ref, dataset=None):
    p = _words(pred)
    r = _words(ref)
    exact = _norm(pred) == _norm(ref)

    lp, lr = max(1, len(str(pred))), max(1, len(str(ref)))
    length_ratio = lp / lr

    xo_pred_pos = {w for w in p if w in XO_POSITION_WORDS}
    xo_ref_pos = {w for w in r if w in XO_POSITION_WORDS}

    return {
        "exact_match": exact,
        "colour_error": _multiset_mismatch(p, r, COLOUR_WORDS) and not exact,
        "size_error": _multiset_mismatch(p, r, SIZE_WORDS) and not exact,
        "shape_error": _multiset_mismatch(p, r, SHAPE_WORDS) and not exact,
        "position_error": _multiset_mismatch(p, r, POSITION_WORDS) and not exact,
        "length_error": (length_ratio < 0.5 or length_ratio > 2.0) and not exact,
        "xo_content_error": (xo_pred_pos != xo_ref_pos) and not exact
                             and dataset == "TicTacToe_Data",
        "length_ratio": length_ratio,
        "pred_len_words": len(p),
        "ref_len_words": len(r),
    }


def analyse(preds, refs, dataset_labels=None, approach_label=None):
    rows = []
    for i, (p, r) in enumerate(zip(preds, refs)):
        ds = dataset_labels[i] if dataset_labels else None
        row = {"idx": i, "dataset": ds, "approach": approach_label,
               "prediction": p, "reference": r}
        row.update(classify_errors(p, r, ds))
        rows.append(row)
    return pd.DataFrame(rows)


def summary_counts(df, group_by="dataset"):
    cols = ["exact_match", "colour_error", "size_error", "shape_error",
            "position_error", "length_error", "xo_content_error"]
    available = [c for c in cols if c in df.columns]
    g = df.groupby(group_by)[available].mean() * 100
    g["n"] = df.groupby(group_by).size()
    return g.round(2)


def length_vs_accuracy(df, bins=10):
    df = df.copy()
    df["len_bin"] = pd.cut(df["ref_len_words"], bins=bins)
    return df.groupby("len_bin", observed=True).agg(
        n=("idx", "count"), exact_match_rate=("exact_match", "mean"))


def plot_error_breakdown(df, out_path, title="Error types per dataset"):
    cols = ["colour_error", "size_error", "shape_error", "position_error",
            "length_error", "xo_content_error"]
    cols = [c for c in cols if c in df.columns]
    err_df = df[~df["exact_match"]].copy()
    if len(err_df) == 0:
        print("No errors to plot (all predictions exact). Skipping.")
        return
    pivot = err_df.groupby("dataset")[cols].mean() * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, colormap="tab10")
    ax.set_ylabel("% of error cases with this error type")
    ax.set_xlabel("Dataset")
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.xticks(rotation=0)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_length_vs_accuracy(df, out_path, title="Accuracy vs reference length"):
    la = length_vs_accuracy(df)
    fig, ax = plt.subplots(figsize=(9, 4))
    la["exact_match_rate"].plot(kind="bar", ax=ax, color="steelblue")
    ax.set_ylabel("Exact match rate")
    ax.set_xlabel("Reference length (words)")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_approach_comparison(dfs, out_path, metric="exact_match"):
    records = []
    for name, df in dfs.items():
        g = df.groupby("dataset")[metric].mean() * 100
        for dataset, v in g.items():
            records.append({"approach": name, "dataset": dataset, metric: v})
    r = pd.DataFrame(records)
    pivot = r.pivot(index="dataset", columns="approach", values=metric)

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel(f"{metric} (%)")
    ax.set_title(f"{metric} by approach and dataset")
    plt.xticks(rotation=0)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")