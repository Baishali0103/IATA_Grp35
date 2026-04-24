"""
Run attention visualisation for a chosen model.

Usage:
    python scripts/run_attention_viz.py --approach resnet --n 4
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.tokenizer import build_tokenizer
from src.dataset import load_raw_dataframes, ImageCaptionDataset
from src.models import build_model
from src.attention_viz import (
    generate_with_attention, build_report_figure, plot_per_token_attention,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", default="resnet",
                    choices=["shallow", "deep", "resnet"])
    ap.add_argument("--tokenizer", default=config.DEFAULT_TOKENIZER,
                    choices=["custom", "bert"])
    ap.add_argument("--tokenizer-path", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    device = config.DEVICE

    if args.tokenizer == "custom":
        tokenizer = build_tokenizer("custom")
        tokenizer.load(args.tokenizer_path
                       or str(config.RESULTS_DIR / "custom_vocab.json"))
    else:
        tokenizer = build_tokenizer("bert")

    ckpt_name = f"{args.approach}_best.pt" if args.tag == "" else f"{args.approach}_{args.tag}_best.pt"
    ckpt_path = config.RESULTS_DIR / ckpt_name
    cnn, decoder = build_model(
        args.approach, vocab_size=tokenizer.vocab_size,
        embed_dim=config.EMBED_DIM, max_length=config.MAX_LENGTH,
        num_heads=config.NUM_HEADS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        ff_dim=config.FF_DIM, dropout=config.DROPOUT)
    cnn = cnn.to(device); decoder = decoder.to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    cnn.load_state_dict(ckpt["cnn"]); decoder.load_state_dict(ckpt["decoder"])

    raw = load_raw_dataframes(config.DATA_DIR)
    test_df = pd.concat([df[df["split"] == "test"] for df in raw.values()
                         if not df.empty], ignore_index=True)

    rng = torch.Generator().manual_seed(args.seed)
    picks = []
    for ds in ["Shapes_Data", "Numbers_Data", "TicTacToe_Data", "TicTacToe_Data"]:
        sub = test_df[test_df["dataset"] == ds].reset_index(drop=True)
        if len(sub) == 0:
            continue
        i = int(torch.randint(0, len(sub), (1,), generator=rng).item())
        picks.append(sub.iloc[i].to_dict())
        if len(picks) >= args.n:
            break

    ds = ImageCaptionDataset(test_df, tokenizer, image_size=config.IMAGE_SIZE,
                              max_length=config.MAX_LENGTH, augment=False)
    examples = []
    label = args.approach if args.tag == "" else f"{args.approach}_{args.tag}"
    out_dir = config.OUTPUTS_DIR / f"attention_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for j, row in enumerate(picks):
        idx = test_df[test_df["file_path"] == row["file_path"]].index[0]
        image_tensor, _ = ds[idx]
        words, attn = generate_with_attention(
            cnn, decoder, image_tensor, tokenizer, device,
            max_length=config.MAX_LENGTH)
        plot_per_token_attention(
            image_tensor, words, attn, out_dir / f"example_{j}_grid.png")
        highlights = ([0, len(words) // 2, len(words) - 1]
                      if len(words) >= 3 else list(range(len(words)))[:3])
        examples.append({
            "image": image_tensor, "words": words, "attn": attn,
            "caption": " ".join(words),
            "highlight_indices": highlights,
        })

    if examples:
        build_report_figure(
            examples,
            config.OUTPUTS_DIR / f"attention_report_{label}.png")
        print("\nDone. Figures saved:")
        print(f"  - outputs_dir/attention_{label}/example_*_grid.png")
        print(f"  - outputs_dir/attention_report_{label}.png")


if __name__ == "__main__":
    main()