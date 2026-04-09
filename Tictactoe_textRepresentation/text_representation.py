from pathlib import Path
from collections import Counter

SPECIAL_TOKENS = ["<pad>", "<unk>", "<start>", "<end>"]


def tokenize(sentence):
    sentence = sentence.lower()
    sentence = sentence.replace(".", "")
    sentence = sentence.replace(",", "")
    return sentence.split()


def build_vocab(sentences):
    counter = Counter()

    for sentence in sentences:
        tokens = tokenize(sentence)
        counter.update(tokens)

    vocab = {}
    idx = 0

    for token in SPECIAL_TOKENS:
        vocab[token] = idx
        idx += 1

    for word in counter:
        vocab[word] = idx
        idx += 1

    return vocab


def encode_sentence(sentence, vocab):

    tokens = tokenize(sentence)

    ids = [vocab["<start>"]]

    for t in tokens:
        ids.append(vocab.get(t, vocab["<unk>"]))

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
        sentence = txt_file.read_text().strip()
        sentences.append(sentence)

    return sentences