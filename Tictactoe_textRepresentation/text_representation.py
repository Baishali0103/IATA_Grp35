from pathlib import Path
from collections import Counter
import re
import numpy as np
from gensim.models import Word2Vec

SPECIAL_TOKENS = ["<pad>", "<unk>", "<start>", "<end>"]


# -------------------------------
# Method 1: Whitespace tokenisation
# -------------------------------
def tokenize_whitespace(sentence):
    sentence = sentence.lower()
    sentence = sentence.replace(".", "")
    sentence = sentence.replace(",", "")
    sentence = sentence.replace(";", "")
    return sentence.split()


# -------------------------------
# Method 2: Regex tokenisation
# -------------------------------
def tokenize_regex(sentence):
    sentence = sentence.lower()
    tokens = re.findall(r"\b\w+\b", sentence)
    return tokens


# -------------------------------
# Method 3: Character-level tokenisation
# -------------------------------
def tokenize_character(sentence):
    sentence = sentence.lower()
    sentence = sentence.replace("\n", " ")
    return list(sentence)


def build_vocab(sentences, tokenizer):
    counter = Counter()

    for sentence in sentences:
        tokens = tokenizer(sentence)
        counter.update(tokens)

    vocab = {}
    idx = 0

    for token in SPECIAL_TOKENS:
        vocab[token] = idx
        idx += 1

    for item in counter:
        vocab[item] = idx
        idx += 1

    return vocab


def encode_sentence(sentence, vocab, tokenizer):
    tokens = tokenizer(sentence)

    ids = [vocab["<start>"]]

    for token in tokens:
        ids.append(vocab.get(token, vocab["<unk>"]))

    ids.append(vocab["<end>"])

    return ids


def pad_sequence(seq, max_len, pad_id):
    if len(seq) < max_len:
        seq = seq + [pad_id] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq


def load_sentences(text_dir):
    text_dir = Path(text_dir)
    sentences = []

    for txt_file in text_dir.glob("*.txt"):
        sentence = txt_file.read_text(encoding="utf-8").strip()
        if sentence:
            sentences.append(sentence)

    return sentences


# -------------------------------
# Method 4: Word2Vec
# -------------------------------
def train_word2vec(sentences, tokenizer=tokenize_regex, vector_size=50, window=5):
    tokenized_sentences = [tokenizer(s) for s in sentences]

    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=4
    )
    return model


def sentence_embedding(sentence, model, tokenizer=tokenize_regex):
    tokens = tokenizer(sentence)
    vectors = []

    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])

    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)