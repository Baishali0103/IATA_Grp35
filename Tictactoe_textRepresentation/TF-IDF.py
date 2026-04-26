from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------
# Load sentences
# ---------------------------
def load_sentences(text_dir):
    text_dir = Path(text_dir)
    sentences = []

    for txt_file in text_dir.glob("*.txt"):
        sentence = txt_file.read_text(encoding="utf-8").strip()
        if sentence:
            sentences.append(sentence)

    return sentences


# ---------------------------
# Main
# ---------------------------
DATA_PATH = "TicTacToe_Data/train/text"

sentences = load_sentences(DATA_PATH)

print("Number of sentences:")
print(len(sentences))

print("\nExample sentence:")
print(sentences[0])


# ---------------------------
# TF-IDF representation
# ---------------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    token_pattern=r"\b\w+\b"
)

tfidf_matrix = vectorizer.fit_transform(sentences)


# ---------------------------
# Results
# ---------------------------
print("\nVocabulary size:")
print(len(vectorizer.vocabulary_))

print("\nTF-IDF matrix shape:")
print(tfidf_matrix.shape)

print("\nFirst sentence TF-IDF vector (first 10 values):")
print(tfidf_matrix[0].toarray()[0][:10])

print("\nExample vocabulary tokens:")
print(list(vectorizer.vocabulary_.keys())[:10])