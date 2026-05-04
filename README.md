# Image-to-Sentence Mapping — Group 35

**EMATM0067 Introduction to AI and Text Analytics**

A deep learning pipeline that trains neural networks to generate natural language descriptions of synthetic images across three domains: geometric shapes, handwritten numbers, and tic-tac-toe boards.

## Overview

The system uses a CNN encoder + Transformer decoder architecture to map input images to captions. We compare three CNN encoders and two tokenisation strategies, and benchmark against GPT-4o as a commercial LLM baseline.

**Best model: ResNetCNN + BERT tokeniser** (BLEU-4: 0.2371, METEOR: 0.5040, BERTScore: 0.9299)

## Datasets

| Dataset | Description | Test Size |
|---|---|---|
| Shapes\_Data | Geometric shapes with colour, size, position attributes | 744 |
| Numbers\_Data | Handwritten numbers of varying length and colour | 744 |
| TicTacToe\_Data | Tic-tac-toe board states with X/O positions | 750 |

---

## Models

**CNN Encoder variants (Axis 1):**
- ShallowCNN — 2 conv layers, ~215K parameters
- DeepCNN — 5 conv layers, ~1.0M parameters
- ResNetCNN — 2 residual blocks with skip connections, ~1.5M parameters

**Tokenisation strategies (Axis 2):**
- Custom word-level tokeniser (3,102 tokens)
- BERT WordPiece tokeniser (`bert-base-uncased`, 30,522 tokens)

---

## Results

| Model | BLEU-4 | METEOR | BERTScore |
|---|---|---|---|
| ShallowCNN — Simple | 0.1398 | 0.4065 | 0.9086 |
| ShallowCNN — BERT | 0.1542 | 0.4236 | 0.9216 |
| DeepCNN — Simple | 0.1987 | 0.4807 | 0.9183 |
| DeepCNN — BERT | 0.2190 | 0.4994 | 0.9292 |
| ResNetCNN — Simple | 0.1895 | 0.4677 | 0.9172 |
| **ResNetCNN — BERT** | **0.2371** | **0.5040** | **0.9299** |
| GPT-4o (n=150) | 0.0243 | 0.1404 | 0.7693 |

---

## Key Findings

- ResNetCNN outperforms shallower variants — skip connections improve gradient flow and feature richness
- BERT tokenisation consistently improves METEOR and BERTScore across all CNN variants
- GPT-4o scores low on BLEU-4 despite producing semantically accurate descriptions — highlighting a fundamental limitation of n-gram metrics when evaluating lexically diverse outputs

---

## Team

| Member | Branch |
|---|---|
| Chirag Sharma (2738012) | `chirag-sharma` |
| Baishali Guha (2801408) | `ck25321-baishali` |
| Thilokesh (2800592) |  `ux25878_thilokesh`|
| Xinran Wang (2715539) | `Xinran-Wang` |
| Yikun Deng (2727723) | `nx25274_Deng-Yikun` |
