"""
Training loop with:
  - mixed precision (AMP) to fit on 6GB GPUs
  - gradient clipping
  - random seeding
  - early stopping based on validation loss
  - best-checkpoint saving
"""
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(cnn, decoder, loader, optimizer, criterion, device,
                scaler: Optional[torch.cuda.amp.GradScaler] = None,
                grad_clip: float = 1.0, use_amp: bool = True):
    cnn.train()
    decoder.train()
    total = 0.0
    for images, seqs in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        seqs = seqs.to(device, non_blocking=True)
        inp = seqs[:, :-1]
        tgt = seqs[:, 1:]

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                memory = cnn(images)
                out = decoder(inp, memory)
                loss = criterion(out.reshape(-1, out.shape[-1]), tgt.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(cnn.parameters()) + list(decoder.parameters()), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            memory = cnn(images)
            out = decoder(inp, memory)
            loss = criterion(out.reshape(-1, out.shape[-1]), tgt.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(cnn.parameters()) + list(decoder.parameters()), grad_clip)
            optimizer.step()

        total += loss.item()
    return total / max(1, len(loader))


@torch.no_grad()
def eval_epoch(cnn, decoder, loader, criterion, device, use_amp: bool = True):
    cnn.eval()
    decoder.eval()
    total = 0.0
    for images, seqs in loader:
        images = images.to(device, non_blocking=True)
        seqs = seqs.to(device, non_blocking=True)
        inp = seqs[:, :-1]
        tgt = seqs[:, 1:]
        if use_amp:
            with torch.cuda.amp.autocast():
                memory = cnn(images)
                out = decoder(inp, memory)
                loss = criterion(out.reshape(-1, out.shape[-1]), tgt.reshape(-1))
        else:
            memory = cnn(images)
            out = decoder(inp, memory)
            loss = criterion(out.reshape(-1, out.shape[-1]), tgt.reshape(-1))
        total += loss.item()
    return total / max(1, len(loader))


def train(cnn, decoder, train_loader, val_loader, optimizer, scheduler,
          criterion, device, num_epochs: int = 50, patience: int = 5,
          grad_clip: float = 1.0, use_amp: bool = True,
          save_path: Optional[str] = None, verbose: bool = True):
    best = float("inf")
    counter = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and torch.cuda.is_available())

    for epoch in range(num_epochs):
        tr = train_epoch(cnn, decoder, train_loader, optimizer, criterion, device,
                         scaler=scaler, grad_clip=grad_clip, use_amp=use_amp)
        vl = eval_epoch(cnn, decoder, val_loader, criterion, device, use_amp=use_amp)
        history["train_loss"].append(tr)
        history["val_loss"].append(vl)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()

        if verbose:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | train {tr:.4f} | "
                  f"val {vl:.4f} | lr {history['lr'][-1]:.2e}")

        if vl < best:
            best = vl
            counter = 0
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"cnn": cnn.state_dict(),
                            "decoder": decoder.state_dict(),
                            "history": history}, save_path)
                if verbose:
                    print(f"  saved -> {save_path}")
        else:
            counter += 1
            if counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

    return history


def load_checkpoint(path, cnn, decoder, device):
    ckpt = torch.load(path, map_location=device)
    cnn.load_state_dict(ckpt["cnn"])
    decoder.load_state_dict(ckpt["decoder"])
    return ckpt.get("history", {})