"""
Tokenizer module. Two variants with the same interface:
  - CustomTokenizer      : word-level vocab built from training captions
  - BertTokenizerWrapper : BERT WordPiece (bert-base-uncased)

Interface:
  .encode(text, max_length) -> List[int] (padded to max_length)
  .decode(ids) -> str (strips special tokens)
  .vocab_size : int
  .pad_id, .start_id, .end_id, .unk_id : int
"""
from collections import Counter
from typing import List, Iterable, Optional
import re

from config import PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN, MAX_LENGTH


def _simple_tokenize(text: str) -> List[str]:
    """Lowercase, isolate punctuation as separate tokens, split on whitespace."""
    text = text.lower()
    text = re.sub(r"([.,])", r" \1 ", text)
    return text.split()


class CustomTokenizer:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.pad_id = 0
        self.start_id = 1
        self.end_id = 2
        self.unk_id = 3
        self._specials = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    def fit(self, texts: Iterable[str]):
        counter = Counter()
        for t in texts:
            counter.update(_simple_tokenize(t))
        vocab = list(self._specials) + sorted(counter.keys())
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        return self

    def encode(self, text: str, max_length: int = MAX_LENGTH) -> List[int]:
        tokens = [START_TOKEN] + _simple_tokenize(text) + [END_TOKEN]
        ids = [self.word2idx.get(t, self.unk_id) for t in tokens]
        if len(ids) > max_length:
            ids = ids[: max_length - 1] + [self.end_id]
        ids = ids + [self.pad_id] * (max_length - len(ids))
        return ids

    def decode(self, ids: List[int]) -> str:
        words = []
        for i in ids:
            if i in (self.pad_id, self.start_id):
                continue
            if i == self.end_id:
                break
            words.append(self.idx2word.get(i, UNK_TOKEN))
        text = " ".join(words)
        text = re.sub(r"\s+([.,])", r"\1", text)
        return text

    def save(self, path):
        import json
        with open(path, "w") as f:
            json.dump({
                "word2idx": self.word2idx,
                "pad_id": self.pad_id,
                "start_id": self.start_id,
                "end_id": self.end_id,
                "unk_id": self.unk_id,
            }, f)

    def load(self, path):
        import json
        with open(path) as f:
            d = json.load(f)
        self.word2idx = d["word2idx"]
        self.idx2word = {int(i): w for w, i in self.word2idx.items()}
        self.pad_id = d["pad_id"]
        self.start_id = d["start_id"]
        self.end_id = d["end_id"]
        self.unk_id = d["unk_id"]
        return self


class BertTokenizerWrapper:
    def __init__(self, model_name: str = "bert-base-uncased"):
        from transformers import BertTokenizerFast
        self._tok = BertTokenizerFast.from_pretrained(model_name)
        self.pad_id = self._tok.pad_token_id
        self.start_id = self._tok.cls_token_id
        self.end_id = self._tok.sep_token_id
        self.unk_id = self._tok.unk_token_id

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    def fit(self, texts):
        return self

    def encode(self, text: str, max_length: int = MAX_LENGTH) -> List[int]:
        enc = self._tok(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            return_tensors=None,
        )
        return enc["input_ids"]

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)

    def save(self, path):
        self._tok.save_pretrained(path)

    def load(self, path):
        from transformers import BertTokenizerFast
        self._tok = BertTokenizerFast.from_pretrained(path)
        self.pad_id = self._tok.pad_token_id
        self.start_id = self._tok.cls_token_id
        self.end_id = self._tok.sep_token_id
        self.unk_id = self._tok.unk_token_id
        return self


def build_tokenizer(kind: str = "custom", texts: Optional[Iterable[str]] = None):
    if kind == "custom":
        t = CustomTokenizer()
        if texts is not None:
            t.fit(texts)
        return t
    elif kind == "bert":
        return BertTokenizerWrapper()
    else:
        raise ValueError(f"Unknown tokenizer kind: {kind}")