"""
Compare your best trained model against GPT-4o on the same test set.

Before running:
  1. pip install openai
  2. Set environment variable (PowerShell):
       $env:OPENAI_API_KEY = "sk-..."
  3. To save cost, use --n 100 first to spend ~$1-2 on a sanity subset.

Usage:
    python scripts/llm_compare.py --n 100 --stratify
    python scripts/llm_compare.py --n 300 --stratify --model gpt-4o

Produces:
    outputs_dir/predictions_llm.csv
    outputs_dir/metrics_llm.json
"""
import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.dataset import load_raw_dataframes, combine
from src.evaluate import compute_all_metrics, save_predictions, save_results


def encode_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def call_llm(client, model, img_b64, prompt, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ],
                }],
                max_tokens=200,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == retries - 1:
                print(f"[error] {e}")
                return ""
            time.sleep(2 ** attempt)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.LLM_MODEL)
    ap.add_argument("--prompt", default=config.LLM_PROMPT)
    ap.add_argument("--n", type=int, default=None,
                    help="Sample size (omit for full test set).")
    ap.add_argument("--stratify", action="store_true",
                    help="Sample equally from each dataset.")
    args = ap.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("pip install openai")

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY environment variable.")

    client = OpenAI()

    raw = load_raw_dataframes(config.DATA_DIR)
    test_df = combine(raw, "test")
    print(f"Full test set: {len(test_df)}")

    if args.n:
        if args.stratify:
            parts = []
            n_per = max(1, args.n // test_df["dataset"].nunique())
            for _, sub in test_df.groupby("dataset"):
                parts.append(sub.sample(min(len(sub), n_per), random_state=42))
            test_df = pd.concat(parts, ignore_index=True)
        else:
            test_df = test_df.sample(min(len(test_df), args.n),
                                     random_state=42).reset_index(drop=True)
    print(f"Sampled: {len(test_df)}")

    preds, refs, labels = [], [], []
    for row in tqdm(test_df.itertuples(index=False), total=len(test_df),
                    desc="LLM"):
        out = call_llm(client, args.model, encode_b64(row.file_path), args.prompt)
        preds.append(out)
        refs.append(row.text)
        labels.append(row.dataset)

    save_predictions(preds, refs, labels,
                     config.OUTPUTS_DIR / "predictions_llm.csv")
    results = compute_all_metrics(preds, refs, labels,
                                  bertscore_subset=config.BERTSCORE_SUBSET,
                                  device=str(config.DEVICE))
    save_results(results, config.OUTPUTS_DIR / "metrics_llm.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()