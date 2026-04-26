from text_representation import *

DATA_PATH = "TicTacToe_Data/train/text"

sentences = load_sentences(DATA_PATH)

print("Example sentence:")
print(sentences[0])

# vocab = build_vocab(sentences)
vocab = build_vocab(sentences, tokenize_character)        # For character-level tokenisation

print("\nVocabulary size:")
print(len(vocab))

#encoded = encode_sentence(sentences[0], vocab)
encoded = encode_sentence(sentences[0], vocab, tokenize_character)    # For character-level tokenisation

print("\nEncoded sentence:")
print(encoded)

padded = pad_sequence(encoded, 100, vocab["<pad>"])

print("\nPadded sentence:")
print(padded)