from text_representation import *

DATA_PATH = "TicTacToe_Data/train/text"

sentences = load_sentences(DATA_PATH)

print("Example sentence:")
print(sentences[0])

# training Word2Vec
w2v_model = train_word2vec(sentences, tokenizer=tokenize_regex, vector_size=50)

print("\nWord2Vec vocabulary size:")
print(len(w2v_model.wv))

print("\nExample tokens:")
print(tokenize_regex(sentences[0]))

print("\nVector for word 'red':")
print(w2v_model.wv["red"][:10])   

sent_vec = sentence_embedding(sentences[0], w2v_model, tokenizer=tokenize_regex)

print("\nSentence embedding shape:")
print(sent_vec.shape)

print("\nSentence embedding (first 10 dims):")
print(sent_vec[:10])