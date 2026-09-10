# Assignment 7 — HMM-based Noun Phrase (NP) Chunking

## Objective

Identify Noun Phrases (NP) in sentences using a Hidden Markov Model (HMM).

- Tags: `B-NP` (Beginning of NP), `I-NP` (Inside NP), `O` (Other)
- **Task 1:** Use the Forward Algorithm to find the probability of each test sentence.
- **Task 2:** Use Viterbi Decoding to predict NP tags and evaluate with `seqeval`.
- Add-K smoothing (K = 1) is used for zero counts.

## Data

| File | Description |
|------|-------------|
| `Data/train_NP.txt` | Training data — 823 sentences, 19,172 tokens |
| `Data/test_NP.txt` | Test data — 77 sentences, 1,896 tokens |

Each file has lines of `word\ttag`, with blank lines separating sentences.

## HMM Components

An HMM for sequence labeling consists of:

| Component | Symbol | Description |
|---|---|---|
| **States** | S | The tag set: {B-NP, I-NP, O} |
| **Initial probabilities** | π(s) | P(tag₁ = s) |
| **Transition probabilities** | A(s' \| s) | P(tagₜ = s' \| tagₜ₋₁ = s) |
| **Emission probabilities** | B(w \| s) | P(wordₜ = w \| tagₜ = s) |

All probability estimates use **Add-K smoothing** (K = 1) to handle zero counts. Computations are done in **log-space** for numerical stability.

## Task 1 — Forward Algorithm

The Forward Algorithm computes the total probability of an observed sequence under the HMM:

```
P(w₁, w₂, ..., wₜ) = Σₛ αₜ(s)
```

where the forward variable αₜ(s) is defined recursively:

```
α₁(s) = π(s) · B(w₁ | s)

αₜ(s) = [Σₛ' αₜ₋₁(s') · A(s | s')] · B(wₜ | s)
```

We use the **log-sum-exp trick** for numerical stability.

## Task 2 — Viterbi Decoding

The Viterbi Algorithm finds the most likely tag sequence given the observation:

```
ŝ₁:ₜ = argmax P(s₁:ₜ | w₁:ₜ)
```

We define the Viterbi variable:

```
δ₁(s) = π(s) · B(w₁ | s)

δₜ(s) = maxₛ' [δₜ₋₁(s') · A(s | s')] · B(wₜ | s)
```

and backtrack to recover the best path. Evaluation uses the [`seqeval`](https://pypi.org/project/seqeval/) library for entity-level metrics.

## Results

| Metric | Value |
|---|---|
| Entity-level F1 (NP) | 0.6702 |
| Token-level Accuracy | 0.8476 |

### Per-Tag Token Accuracy

| Tag | Correct | Total | Accuracy |
|-----|---------|-------|----------|
| B-NP | 377 | 475 | 0.7937 |
| I-NP | 480 | 579 | 0.8290 |
| O | 750 | 842 | 0.8907 |

### Confusion Matrix

```
         B-NP    I-NP       O
B-NP      377      32      66
I-NP       43     480      56
   O       45      47     750
```

## Key Observations

- The HMM captures basic sequential patterns of NP chunks (B-NP → I-NP transitions are strong).
- Add-K smoothing handles unseen words in the test set.
- Entity-level evaluation via `seqeval` is more stringent than token-level accuracy, since an entire NP span must be correctly identified.
- The `O` tag has the highest token accuracy (0.89), while `B-NP` is the hardest to predict (0.79) — boundary detection is inherently harder.

## Dependencies

- `numpy`
- `seqeval`

