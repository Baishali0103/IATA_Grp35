"""
Data sanity check. RUN THIS FIRST every time you move data around.
Usage:
    python scripts/diagnose_data.py
"""
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.dataset import load_raw_dataframes


def main():
    print(f"PROJECT_ROOT = {config.PROJECT_ROOT}")
    print(f"DATA_DIR     = {config.DATA_DIR}")
    print(f"RESULTS_DIR  = {config.RESULTS_DIR}")
    print(f"OUTPUTS_DIR  = {config.OUTPUTS_DIR}")
    print()

    if not config.DATA_DIR.exists():
        print(f"[ERROR] {config.DATA_DIR} does not exist.")
        print("        Edit config.py so PROJECT_ROOT points at the folder "
              "that contains `Data/`.")
        return

    print("=== File counts per dataset/split ===")
    for ds_name in config.DATASETS:
        ds_path = config.DATA_DIR / ds_name
        if not ds_path.exists():
            print(f"  {ds_name}: MISSING")
            continue
        line = f"  {ds_name}:"
        for split in config.SPLITS:
            img_dir = ds_path / split / "images"
            txt_dir = ds_path / split / "text"
            n_img = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
            n_txt = len(list(txt_dir.glob("*.txt"))) if txt_dir.exists() else 0
            line += f" {split}={n_img}img/{n_txt}txt"
        manifest = ds_path / "manifest.csv"
        line += f" manifest={'yes' if manifest.exists() else 'NO'}"
        print(line)
    print()

    print("=== Loading dataframes (this can take a moment) ===")
    raw = load_raw_dataframes(config.DATA_DIR)
    for ds_name, df in raw.items():
        print(f"\n--- {ds_name} (total rows: {len(df)}) ---")
        if df.empty:
            print("  empty")
            continue
        print(df["split"].value_counts().to_string())
        print("\n  Columns found:", list(df.columns))
        if "colour" in df.columns:
            print("  colour/BW split:")
            print("   ", df["colour"].value_counts().to_dict())
        for i in range(min(2, len(df))):
            row = df.iloc[i]
            print(f"\n  sample {i}:")
            print(f"    file: {row['file_name']}")
            print(f"    text: {row['text'][:120]}")

    print("\n=== Caption length stats (words) ===")
    for ds_name, df in raw.items():
        if df.empty:
            continue
        n_words = df["text"].str.split().str.len()
        print(f"  {ds_name:18s} min={n_words.min()}  mean={n_words.mean():.1f}  "
              f"median={n_words.median():.0f}  max={n_words.max()}")

    print("\n=== Image sanity (first image of first dataset) ===")
    for ds_name, df in raw.items():
        if df.empty:
            continue
        row = df.iloc[0]
        try:
            img = Image.open(row["file_path"])
            print(f"  {ds_name}: {img.size} mode={img.mode}")
        except Exception as e:
            print(f"  {ds_name}: ERROR {e}")
        break

    print("\n=== Checkpoints present ===")
    if not config.RESULTS_DIR.exists():
        print(f"  {config.RESULTS_DIR} does not exist yet.")
    else:
        ckpts = list(config.RESULTS_DIR.glob("*.pt"))
        if not ckpts:
            print("  no .pt files yet. Train with scripts/run_training.py")
        else:
            for c in ckpts:
                mb = c.stat().st_size / (1024*1024)
                print(f"  {c.name}  ({mb:.1f} MB)")

    print("\nDone.")


if __name__ == "__main__":
    main()