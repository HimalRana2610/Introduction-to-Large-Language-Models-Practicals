## Notebook

Implementation and results live in [`main.ipynb`](main.ipynb).

# Assignment 1 - Language Identification on IndicCorp v2

**Course:** Introduction to Large Language Models

## Problem statement

Build a language identification (LID) model over the languages of the
[IndicCorp v2](https://huggingface.co/datasets/ai4bharat/IndicCorpV2) corpus.

1. Construct the dataset: 1000 **sentences** per language (each row of the corpus is a
   *paragraph*, so paragraphs must be sentence-tokenized first). Split 80:10:10 into
   train / validation / test using **stratified** sampling. Use the corpus's own labels.
2. Represent every sentence with a **TF-IDF** vector built from
   * word unigrams + word bigrams, and
   * character 2-, 3- and 4-grams.
3. Train a **logistic regression** language identifier on those vectors.
4. Evaluate with **macro-F1**.

**Constraint:** no `sklearn` (or any other pre-built ML library) class may be used for the
vectorizer, the classifier or the metrics. Everything below is implemented from scratch on
top of `numpy` / `scipy.sparse` only (`scipy.sparse` is used purely as a sparse-matrix
container, not as an ML component).

## 1. Building the dataset

### 1.1 Discovering the languages in the repository

IndicCorp v2 stores one plain-text file per *language-script* pair (that is how the corpus
encodes the fact that a few languages, e.g. Kashmiri and Sindhi, are written in more than one
script). We discover those files at runtime instead of hard-coding them, and use the file
names verbatim as class labels - the assignment asks for "the same labels as given in the
repository".

### 1.2 Streaming the raw paragraphs

The per-language files are several gigabytes each, but we only need a few thousand
paragraphs from every language. We therefore stream each file over HTTP and stop reading as
soon as we have enough - nothing is ever downloaded in full.

### 1.3 Sentence tokenization

Each row of IndicCorp is a paragraph, so it has to be split into sentences. One tokenizer has
to cope with three different families of sentence terminators present in the corpus:

| script family | terminators |
|---|---|
| Brahmic (Devanagari, Gujarati, Bengali, ...) | danda `U+0964`, double danda `U+0965`, `?`, `!` |
| Perso-Arabic (Urdu, Kashmiri, Sindhi) | Arabic full stop `U+06D4`, Arabic question mark `U+061F` |
| Latin (English) | `.`, `?`, `!` |

The full stop needs extra care: it is also a decimal separator and an abbreviation marker, so
we split on it only when it is *not* surrounded by digits and not preceded by a single capital
letter.

### 1.4 Sentence quality filter

Web-crawled text contains a lot of boilerplate. We keep a sentence only if it has a reasonable
length, enough words, is mostly made of letters (not digits / punctuation / markup), and has
not already been seen for that language.

### 1.5 Collecting 1000 sentences per language

### 1.6 Stratified 80 : 10 : 10 split

Every language contributes exactly the same number of sentences to every split, so the splits
stay perfectly balanced (800 / 100 / 100 sentences per language).

## 2. TF-IDF from scratch

### 2.1 Feature extraction

Each sentence becomes a bag of five kinds of features. Every family gets a prefix so that,
say, the word `the` and the character 3-gram `the` remain distinct features:

| prefix | feature |
|---|---|
| `w1:` | word unigram |
| `w2:` | word bigram |
| `c2:` `c3:` `c4:` | character 2 / 3 / 4-grams |

Character n-grams are extracted **within word boundaries** (every word is padded with one
space on each side). That is what makes character n-grams so effective for language ID: the
model can learn word-initial and word-final shapes instead of n-grams that straddle two
unrelated words.

### 2.2 The vectorizer

`TfIdfVectorizer` below is written from scratch. It is deliberately parameterised over the TF
and IDF weighting schemes, because **Assignment 2** asks for six combinations of normalized
and unnormalized TF / IDF - Assignment 1 uses the unnormalized (`raw` / `standard`) setting:

$$\mathrm{tfidf}(t,d) = \mathrm{tf}(t,d)\cdot \mathrm{idf}(t),
  \qquad \mathrm{tf}(t,d)=c(t,d), \qquad \mathrm{idf}(t)=\log\frac{N}{df(t)}$$

Rows are finally scaled to unit L2 length. That is a *vector length* normalization applied
identically to every configuration; it keeps gradient descent well conditioned and is
independent of the TF/IDF weighting schemes compared in Assignment 2.

## 3. Multinomial logistic regression from scratch

With $C$ languages we use the multinomial (softmax) formulation directly rather than
one-vs-rest:

$$P(y=c\mid x)=\frac{\exp(w_c^\top x + b_c)}{\sum_{c'}\exp(w_{c'}^\top x + b_{c'})}$$

trained by minimising the L2-regularised cross-entropy

$$\mathcal{L}=-\frac{1}{B}\sum_{i}\log P(y_i\mid x_i)+\frac{\lambda}{2}\lVert W\rVert^2 .$$

The gradient with respect to $W$ is $\frac{1}{B}X^\top(P-Y)+\lambda W$. Optimisation is
mini-batch gradient descent with **Adam**; the softmax uses the standard max-subtraction trick
for numerical stability. Training stops early when validation macro-F1 has not improved for
`patience` epochs, and the best weights are restored.

## 4. Evaluation metrics from scratch

For every language $c$:

$$P_c=\frac{TP_c}{TP_c+FP_c},\qquad R_c=\frac{TP_c}{TP_c+FN_c},\qquad
  F1_c=\frac{2P_cR_c}{P_c+R_c}$$

and **macro-F1** is the unweighted mean of the per-class $F1_c$ - every language counts
equally, regardless of how many test sentences it has.

## 5. Training the language identifier

## 6. Results

## 7. What the model learned

Logistic regression is linear, so the largest weights per class are directly readable: they
are the n-grams the model treats as evidence for a language.

## 8. Ablation - which feature family carries the signal?

Not required by the assignment, but it makes the choice of feature set concrete: character
n-grams alone come close to the full model, while word n-grams alone are noticeably weaker -
many of these languages share a script, so the discriminating evidence lives in character
sequences and word shapes rather than in whole-word identity.

## 9. Summary

* Built a balanced language-identification dataset from IndicCorp v2 - 1000 sentence-tokenized
  sentences per language, split 80:10:10 by stratified sampling so every split holds exactly
  the same number of sentences for every language.
* Implemented TF-IDF from scratch over word unigrams/bigrams and word-boundary character
  2/3/4-grams, with document-frequency pruning per feature family.
* Implemented multinomial logistic regression from scratch (softmax + cross-entropy + Adam,
  early stopping on validation macro-F1).
* Implemented precision / recall / F1 / macro-F1 / confusion matrix from scratch and reported
  the test macro-F1 above.
* The splits saved under `data/` are reused unchanged in **Assignment 2**, which compares six
  combinations of normalized and unnormalized TF and IDF weighting on this same model.
