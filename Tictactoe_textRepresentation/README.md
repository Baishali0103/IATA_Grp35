# Text Representation Methods Used

In this project, I convert TicTacToe board descriptions into numerical representations so they can be processed by machine learning models.

The following text representation methods are used:

### 1. Character-level Encoding

In `main.py`, sentences are tokenized at the **character level**.
Each character is converted into an integer using a vocabulary dictionary.
The encoded sequence is then **padded to a fixed length** so that all inputs have the same size.

Pipeline:

Sentence → Character Tokenization → Vocabulary Mapping → Integer Encoding → Padding

---

### 2. TF-IDF Representation

In `TF-IDF.py`, I use **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert sentences into vectors based on the importance of words in the dataset.

This method provides a simple statistical representation of the text.

---

### 3. Word2Vec Embedding

In `main_word2vec.py`, I use **Word2Vec embeddings** to represent words as dense vectors.
Sentence representations are obtained by combining the embeddings of the words in the sentence.

---

