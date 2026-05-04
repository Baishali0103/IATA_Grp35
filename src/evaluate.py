"""
Evaluation: generation + four metrics (Exact Match, BLEU-4, METEOR, BERTScore),
with per-dataset breakdowns.

Key functions:
    generate_predictions(cnn, decoder, loader, tokenizer, device) -> preds, refs
    compute_all_metrics(preds, refs, dataset_labels) -> dict
    save_predictions(...), save_results(...)
"""
from typing import List, Dict, Optional
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_one(cnn, decoder, image, tokenizer, device, max_length: int = 150):
    """Greedy decode a single image. Returns the decoded string."""
    cnn.eval()
    decoder.eval()
    image = image.unsqueeze(0).to(device)
    memory = cnn(image)

    tokens = [tokenizer.start_id]
    for _ in range(max_length):
        inp = torch.tensor(tokens, device=device).unsqueeze(0)
        out = decoder(inp, memory)
        next_id = torch.argmax(out[0, -1, :]).item()
        if next_id == tokenizer.end_id:
            break
        tokens.append(next_id)
    return tokenizer.decode(tokens)


@torch.no_grad()
def generate_batch(cnn, decoder, images, tokenizer, device, max_length: int = 150):
    """Greedy-decode a batch at once. Returns list of decoded strings."""
    cnn.eval()
    decoder.eval()
    images = images.to(device)
    B = images.shape[0]
    memory = cnn(images)

    tokens = torch.full((B, 1), tokenizer.start_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_length):
        out = decoder(tokens, memory)
        next_ids = out[:, -1, :].argmax(dim=-1)
        next_ids = torch.where(finished, torch.full_like(next_ids, tokenizer.pad_id), next_ids)
        tokens = torch.cat([tokens, next_ids.unsqueeze(1)], dim=1)
        finished = finished | (next_ids == tokenizer.end_id)
        if finished.all():
            break

    preds = [tokenizer.decode(row) for row in tokens.tolist()]
    return preds


def generate_predictions(cnn, decoder, data_loader, tokenizer, device,
                         max_length: int = 150, show_progress: bool = True):
    """Run generation over a whole DataLoader. Returns (preds, refs)."""
    preds, refs = [], []
    it = tqdm(data_loader, desc="Generating") if show_progress else data_loader
    for images, sequences in it:
        batch_preds = generate_batch(cnn, decoder, images, tokenizer, device, max_length)
        preds.extend(batch_preds)
        for row in sequences.tolist():
            refs.append(tokenizer.decode(row))
    return preds, refs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def exact_match(preds: List[str], refs: List[str]) -> float:
    assert len(preds) == len(refs)
    if len(preds) == 0:
        return 0.0
    hits = sum(1 for p, r in zip(preds, refs) if _normalize(p) == _normalize(r))
    return hits / len(preds)


def bleu4(preds: List[str], refs: List[str]) -> float:
    """Corpus-level BLEU-4 via sacrebleu. Returns 0-100 scale."""
    import sacrebleu
    if len(preds) == 0:
        return 0.0
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return bleu.score


def meteor(preds: List[str], refs: List[str]) -> float:
    """Corpus METEOR (mean of sentence-level scores)."""
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.translate.meteor_score import meteor_score

    scores = []
    for p, r in zip(preds, refs):
        try:
            s = meteor_score([r.split()], p.split())
        except Exception:
            s = 0.0
        scores.append(s)
    return float(np.mean(scores)) if scores else 0.0


def bertscore(preds: List[str], refs: List[str],
              model_type: str = "bert-base-uncased",
              batch_size: int = 32,
              device: Optional[str] = None,
              subset: Optional[int] = None) -> Dict[str, float]:
    """Corpus BERTScore precision/recall/F1."""
    from bert_score import score
    if subset is not None and len(preds) > subset:
        idx = np.random.RandomState(42).choice(len(preds), subset, replace=False)
        preds = [preds[i] for i in idx]
        refs = [refs[i] for i in idx]
    if len(preds) == 0:
        return {"P": 0.0, "R": 0.0, "F1": 0.0}
    P, R, F1 = score(preds, refs, model_type=model_type, batch_size=batch_size,
                     device=device, verbose=False)
    return {"P": float(P.mean()), "R": float(R.mean()), "F1": float(F1.mean())}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def compute_all_metrics(preds: List[str], refs: List[str],
                        dataset_labels: Optional[List[str]] = None,
                        bertscore_subset: Optional[int] = None,
                        device: Optional[str] = None) -> Dict:
    """
    Returns:
        {
          "overall": {n, exact_match, bleu4, meteor, bertscore_p/r/f1},
          "per_dataset": { dataset_name: {same keys} }
        }
    """
    results = {"overall": {}}

    print("Computing overall metrics...")
    results["overall"]["n"] = len(preds)
    results["overall"]["exact_match"] = exact_match(preds, refs)
    results["overall"]["bleu4"] = bleu4(preds, refs)
    results["overall"]["meteor"] = meteor(preds, refs)
    bs = bertscore(preds, refs, subset=bertscore_subset, device=device)
    results["overall"]["bertscore_p"] = bs["P"]
    results["overall"]["bertscore_r"] = bs["R"]
    results["overall"]["bertscore_f1"] = bs["F1"]

    if dataset_labels is not None:
        assert len(dataset_labels) == len(preds)
        results["per_dataset"] = {}
        buckets = defaultdict(lambda: {"preds": [], "refs": []})
        for p, r, d in zip(preds, refs, dataset_labels):
            buckets[d]["preds"].append(p)
            buckets[d]["refs"].append(r)

        for name, bucket in buckets.items():
            print(f"Computing metrics for {name} (n={len(bucket['preds'])})...")
            bs = bertscore(bucket["preds"], bucket["refs"],
                           subset=bertscore_subset, device=device)
            results["per_dataset"][name] = {
                "n": len(bucket["preds"]),
                "exact_match": exact_match(bucket["preds"], bucket["refs"]),
                "bleu4": bleu4(bucket["preds"], bucket["refs"]),
                "meteor": meteor(bucket["preds"], bucket["refs"]),
                "bertscore_p": bs["P"],
                "bertscore_r": bs["R"],
                "bertscore_f1": bs["F1"],
            }

    return results


def results_to_dataframe(results: Dict) -> pd.DataFrame:
    rows = [{"split": "overall", **results["overall"]}]
    for name, metrics in results.get("per_dataset", {}).items():
        rows.append({"split": name, **metrics})
    return pd.DataFrame(rows)


def save_predictions(preds, refs, dataset_labels, path):
    df = pd.DataFrame({
        "prediction": preds,
        "reference": refs,
        "dataset": dataset_labels if dataset_labels is not None else [""] * len(preds),
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def save_results(results: Dict, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)