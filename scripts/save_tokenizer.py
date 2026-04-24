"""
Build and save the custom tokenizer from training captions.

Usage:
    python scripts/save_tokenizer.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.tokenizer import build_tokenizer
from src.dataset import load_raw_dataframes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(config.RESULTS_DIR / "custom_vocab.json"))
    args = ap.parse_args()

    raw = load_raw_dataframes(config.DATA_DIR)
    train_texts = []
    for df in raw.values():
        if df.empty:
            continue
        train_texts.extend(df[df["split"] == "train"]["text"].tolist())

    if not train_texts:
        raise SystemExit("No training captions found. Run diagnose_data.py first.")

    print(f"Fitting vocab on {len(train_texts)} training captions...")
    tok = build_tokenizer("custom", texts=train_texts)
    print(f"Vocabulary size: {tok.vocab_size}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tok.save(args.output)
    print(f"Saved to {args.output}")

    print("\nRoundtrip sanity check on first 3 captions:")
    for t in train_texts[:3]:
        ids = tok.encode(t)
        back = tok.decode(ids)
        print(f"  in  : {t[:80]}")
        print(f"  out : {back[:80]}")
        print()


if __name__ == "__main__":
    main()