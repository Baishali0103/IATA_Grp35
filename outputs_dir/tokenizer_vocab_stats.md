# Tokenizer vocabulary statistics

| tokenizer   |   vocab_size |   val_metric |   test_metric | metric_name             |
|:------------|-------------:|-------------:|--------------:|:------------------------|
| custom      |         3102 |       0.0067 |        0.0062 | unk_rate                |
| bert        |        30522 |       0.2966 |        0.2928 | extra_subwords_per_word |

**Interpretation:**

- `unk_rate` (custom): fraction of non-PAD tokens mapping to `<UNK>`. Lower is better.
- `extra_subwords_per_word` (BERT): subword splits per whitespace word. Higher means BERT needs more tokens per caption.
