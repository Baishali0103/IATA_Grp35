"""
Attention visualisation: extract decoder cross-attention during generation
and overlay it on the input image.
"""
from typing import List, Tuple
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image


class CrossAttentionRecorder:
    """Hooks the decoder's last cross-attention layer to capture weights."""

    def __init__(self, decoder: nn.Module):
        self.decoder = decoder
        self.step_weights = []
        self._hooks = []

        for layer in decoder.transformer.layers:
            self._patch_mha(layer.multihead_attn)

        self._last = decoder.transformer.layers[-1].multihead_attn
        h = self._last.register_forward_hook(self._hook)
        self._hooks.append(h)

    @staticmethod
    def _patch_mha(mha):
        orig = mha.forward
        def patched(query, key, value, **kw):
            kw["need_weights"] = True
            kw["average_attn_weights"] = True
            return orig(query, key, value, **kw)
        mha.forward = patched

    def _hook(self, module, inputs, output):
        attn = output[1]
        if attn is not None:
            self.step_weights.append(attn[:, -1, :].detach().cpu())

    def reset(self):
        self.step_weights = []

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


@torch.no_grad()
def generate_with_attention(cnn, decoder, image, tokenizer, device,
                             max_length=150):
    """Greedy decode while capturing per-step attention."""
    cnn.eval(); decoder.eval()
    recorder = CrossAttentionRecorder(decoder)
    try:
        image = image.unsqueeze(0).to(device)
        memory = cnn(image)
        tokens = [tokenizer.start_id]
        words = []
        attns = []
        for _ in range(max_length):
            inp = torch.tensor(tokens, device=device).unsqueeze(0)
            recorder.reset()
            out = decoder(inp, memory)
            next_id = torch.argmax(out[0, -1, :]).item()
            if recorder.step_weights:
                attns.append(recorder.step_weights[-1].squeeze(0).numpy())
            if next_id == tokenizer.end_id:
                break
            tokens.append(next_id)
            word = tokenizer.decode([tokenizer.start_id, next_id])
            words.append(word)
        attns = attns[:len(words)]
        attn_maps = np.stack(attns) if attns else np.zeros((0, memory.shape[1]))
        return words, attn_maps
    finally:
        recorder.close()


def attn_to_heatmap(attn_vec, target_size=128):
    n = attn_vec.shape[0]
    side = int(round(math.sqrt(n)))
    if side * side == n:
        m = attn_vec.reshape(side, side)
    else:
        m = attn_vec.reshape(1, n)
    t = torch.tensor(m, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(target_size, target_size), mode="bilinear",
                      align_corners=False)
    return t.squeeze().numpy()


def _to_pil(image_tensor):
    arr = (image_tensor.detach().cpu().numpy().transpose(1, 2, 0) * 255
           ).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def plot_per_token_attention(image_tensor, words, attn_maps, out_path,
                              max_cols=6, figsize_per=2.0):
    img = _to_pil(image_tensor)
    W, H = img.size
    n = len(words)
    if n == 0:
        return
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per, rows * figsize_per))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        if i < n:
            heat = attn_to_heatmap(attn_maps[i], target_size=W)
            ax.imshow(img)
            ax.imshow(heat, cmap="jet", alpha=0.5)
            ax.set_title(words[i], fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def build_report_figure(examples, out_path):
    """4-row figure for the report: (image, heat1, heat2, heat3) per row."""
    n = len(examples)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    axes = np.atleast_2d(axes)
    for r, ex in enumerate(examples):
        img = _to_pil(ex["image"])
        W, H = img.size
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(ex["caption"][:60], fontsize=9)
        axes[r, 0].axis("off")
        idxs = ex["highlight_indices"]
        for c, ti in enumerate(idxs, start=1):
            ax = axes[r, c]
            if ti < len(ex["words"]):
                heat = attn_to_heatmap(ex["attn"][ti], target_size=W)
                ax.imshow(img)
                ax.imshow(heat, cmap="jet", alpha=0.55)
                ax.set_title(f'"{ex["words"][ti]}"', fontsize=9)
            ax.axis("off")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")