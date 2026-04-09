from text_representation import *

DATA_PATH = "TicTacToe_Data/train/text"

sentences = load_sentences(DATA_PATH)

print("Example sentence:")
print(sentences[0])

vocab = build_vocab(sentences)

print("\nVocabulary size:")
print(len(vocab))

encoded = encode_sentence(sentences[0], vocab)

print("\nEncoded sentence:")
print(encoded)

padded = pad_sequence(encoded, 40, vocab["<pad>"])

print("\nPadded sentence:")
print(padded)