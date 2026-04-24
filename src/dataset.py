"""
Dataset loading. Reads images + captions from:
    <DATA_DIR>/<dataset>/manifest.csv
    <DATA_DIR>/<dataset>/<split>/images/*.png
    <DATA_DIR>/<dataset>/<split>/text/*.txt

Joins with manifest.csv to tag each row with colour flag, distortion level,
crowding, shape_mode, etc. for richer breakdowns later.
"""
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def _read_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(manifest_path)
    for col in ("colour", "has_overlap", "ambiguous", "is_draw"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin({"true", "1", "yes"})
    return df


def load_raw_dataframes(data_dir: Path,
                        datasets: Iterable[str] = ("Shapes_Data", "Numbers_Data",
                                                    "TicTacToe_Data"),
                        splits: Iterable[str] = ("train", "val", "test")) -> dict:
    from tqdm import tqdm

    out = {}
    for ds_name in datasets:
        ds_path = Path(data_dir) / ds_name
        if not ds_path.exists():
            print(f"[warn] dataset folder missing: {ds_path}")
            out[ds_name] = pd.DataFrame()
            continue

        manifest = _read_manifest(ds_path / "manifest.csv")
        if not manifest.empty and "filename" in manifest.columns:
            manifest["file_name"] = manifest["filename"].apply(lambda p: Path(p).name)

        frames = []
        for split in splits:
            img_dir = ds_path / split / "images"
            txt_dir = ds_path / split / "text"
            if not img_dir.exists():
                continue
            names = sorted(p.name for p in img_dir.glob("*.png"))
            rows = []
            for name in tqdm(names, desc=f"{ds_name}/{split}", leave=False):
                stem = Path(name).stem
                txt_path = txt_dir / f"{stem}.txt"
                if not txt_path.exists():
                    continue
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                rows.append({
                    "file_name": name,
                    "file_path": str(img_dir / name),
                    "text": text,
                    "split": split,
                    "dataset": ds_name,
                })
            if rows:
                df = pd.DataFrame(rows)
                if not manifest.empty:
                    meta_cols = [c for c in manifest.columns
                                 if c not in {"filename", "split", "file_name"}]
                    df = df.merge(manifest[["file_name"] + meta_cols],
                                  on="file_name", how="left")
                frames.append(df)
        out[ds_name] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out


def combine(raw: dict, split: str) -> pd.DataFrame:
    parts = []
    for df in raw.values():
        if df.empty:
            continue
        parts.append(df[df["split"] == split])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


class ImageCaptionDataset(Dataset):
    """Produces (image_tensor[3,H,W], token_ids[max_length])."""

    def __init__(self, dataframe: pd.DataFrame, tokenizer,
                 image_size: int = 128, max_length: int = 150,
                 augment: bool = False):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_length = max_length

        if augment:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ColorJitter(brightness=0.10, contrast=0.10),
                T.ToTensor(),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row["file_path"]).convert("RGB")
        img_tensor = self.transform(image)
        ids = self.tokenizer.encode(row["text"], max_length=self.max_length)
        return img_tensor, torch.tensor(ids, dtype=torch.long)

    def get_meta(self, idx) -> dict:
        return self.data.iloc[idx].to_dict()