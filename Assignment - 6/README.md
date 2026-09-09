## Notebook

Implementation and results live in [`main.ipynb`](main.ipynb).

# Assignment 6 - Neural classifiers for language identification

**Course:** Introduction to Large Language Models

## Problem statement

Using **the language identification dataset built in Assignment 1**, train the following
classifiers on **fastText wiki word vectors** and compare them with **macro-F1**:

1. **Logistic regression** on the average of a sentence's word vectors
2. **Multi-layer perceptron** with $n = 0, 1, 2$ hidden layers
3. **LSTM**
4. **GRU**
5. **fastText supervised classification** (the fastText library may be used here)

**Constraint:** no pre-built `sklearn` class may be used for the evaluation. Precision,
recall, F1 and macro-F1 are therefore implemented from scratch, as in Assignment 1 - and so
are classifiers 1-4, including the LSTM and GRU forward passes and their
backpropagation-through-time, all on top of `numpy` alone. Classifier 5 is the one model the
assignment allows us to take off the shelf.

> ### Dependencies
>
> This notebook needs **`fasttext`** for classifier 5 only
> (`pip install fasttext-wheel`, which ships prebuilt wheels and needs no C++ toolchain).
> Everything else uses `numpy` + `matplotlib`.
>
> The first run downloads the fastText wiki vectors (see §2.1) and caches them under
> `data/wiki_vectors/`; later runs read the cache and start immediately.

## 1. The dataset

Assignment 1 built a balanced language-identification corpus from
[IndicCorp v2](https://huggingface.co/datasets/ai4bharat/IndicCorpV2): **1000
sentence-tokenized sentences for each of 23 languages**, split 80:10:10 by stratified
sampling. This notebook reads those `train.tsv` / `val.tsv` / `test.tsv` files unchanged, so
every number below is directly comparable with the TF-IDF + logistic-regression baseline of
Assignment 1.

| split | sentences | per language |
|---|---|---|
| train | 18 400 | 800 |
| val | 2 300 | 100 |
| test | 2 300 | 100 |

The 23 labels are `as bd bn dg en gom gu kha kn ks mai ml mni mr ne or pa sa sat sd ta te ur`.

Tokenization reuses Assignment 1's Unicode-aware tokenizer. `\w+` is unusable on this corpus:
Indic vowel signs are combining marks (Unicode category `Mn` / `Mc`) and `str.isalnum()`
reports those as `False`, so a `\w+` tokenizer tears every Devanagari word apart at its
matras. The dataset contains **120 336 word types**; sentences are 15.2 tokens long on
average (median 12, p95 36).

## 2. Pretrained fastText wiki vectors

### 2.1 Streaming only what we need

fastText publishes one 300-dimensional model per Wikipedia language at
`dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.<lang>.vec`. The files are plain text and
enormous - `wiki.en.vec` alone is **6.6 GB**, and the 18 files we need come to roughly 11 GB.

Downloading them in full is unnecessary. We only ever need the 120 336 word types that occur
in this dataset, and fastText stores words in **descending corpus frequency**. So each file is
*streamed*, parsed line by line, filtered against the corpus vocabulary, and abandoned after
`max_vector_lines = 200 000` words. Three languages hit that cap (`en`, `ta`, `te`); the other
15 are exhausted before it. Roughly 4 GB crosses the network instead of 11, and about 70 000
distinct vectors survive the merge.

### 2.2 Five languages have no wiki model

fastText covers the Wikipedia editions that existed in 2016, and five of our 23 languages had
none - the server answers `403` for them:

| language | code | script | wiki model |
|---|---|---|---|
| Bodo | `bd` | Devanagari | no |
| Dogri | `dg` | Devanagari | no |
| Khasi | `kha` | Latin | no |
| Manipuri | `mni` | Meetei Mayek | no |
| Santali | `sat` | Ol Chiki | no |

This is not a bug to work around - it is the central fact that shapes the whole results table,
and §9 traces its consequences.

### 2.3 One merged embedding table

Each fastText model lives in **its own vector space**, so the 18 tables cannot simply be
concatenated: the Hindi vector for a word and the Marathi vector for the same word are not
comparable. For language identification that turns out not to matter. What the classifier
needs is a lookup that is (a) defined for the words of every language and (b) *different* for
words of different languages - and merging the per-language tables gives exactly that. A word
claimed by several models (shared scripts, loan words, digits, Latin strings) is resolved to
the model in which it is **most frequent**, i.e. the one where it has the smallest frequency
rank.

Two reserved rows are prepended: index 0 is `<pad>` and index 1 is `<unk>`, both the zero
vector, so an unknown token contributes nothing to a sentence average. Every other row is
scaled to unit length - fastText norms encode corpus frequency, which would otherwise let
frequent words dominate a sentence average for no linguistic reason.

The result is a **69 795 x 300** embedding table.

### 2.4 Coverage

| coverage of the corpus tokens | languages |
|---|---|
| > 90 % | `en` 95.3, `ur` 95.2, `sd` 91.7 |
| 80-90 % | `kha` 89.5, `pa` 89.1, `gu` 86.5, `ne` 86.4, `te` 86.2, `mr` 85.1, `gom` 83.8, `ta` 83.8, `kn` 83.6, `bn` 82.4, `as` 80.2 |
| 60-80 % | `mai` 74.8, `or` 71.2, `ml` 64.0 |
| < 60 % | `dg` 57.4, `sa` 45.3, `bd` 34.0, `ks` 34.0, **`mni` 3.6**, **`sat` 1.3** |

**Overall token coverage: 69.9 %.** Three things are worth reading out of this table:

* Having no wiki model is not automatically fatal. Khasi is written in the **Latin** script and
  reaches 89.5 % coverage through the *English* model; Bodo and Dogri are written in
  **Devanagari** and pick up 34 % / 57 % from Hindi-adjacent models.
* Having a wiki model is not automatically enough. Kashmiri's Wikipedia had only 1489 words in
  2016, so `ks` sits at 34 % despite having its own model; Sanskrit's rich morphology leaves
  `sa` at 45 %.
* **Manipuri and Santali are effectively invisible.** Meetei Mayek and Ol Chiki are used by no
  other language in the corpus, so nothing bleeds across - 881 of 1000 Manipuri sentences and
  896 of 1000 Santali sentences contain *no* word with a pretrained vector at all, and their
  averaged vector is exactly zero.

## 3. Evaluation metrics, from scratch

For every language $c$, from the confusion matrix,

$$P_c=\frac{TP_c}{TP_c+FP_c},\qquad R_c=\frac{TP_c}{TP_c+FN_c},\qquad
  F1_c=\frac{2P_cR_c}{P_c+R_c}$$

and **macro-F1** is the unweighted mean of the per-class $F1_c$ - every language counts
equally, regardless of how easy it is. The implementation is checked against a tiny
hand-computable example (expected 0.8222) before anything is trained.

Macro-F1 is the right metric to have chosen here: the splits are perfectly balanced, so
accuracy and micro-F1 would let 21 easy languages hide the two that the pretrained vectors
cannot see. Macro-F1 makes each of those cost a full $1/23 = 4.3$ points.

## 4. Classifier 1 - logistic regression on averaged word vectors

A sentence becomes the mean of the unit-norm fastText vectors of its in-vocabulary words:

$$x_d=\frac{1}{|\{w \in d : w \in V\}|}\sum_{w\in d,\; w\in V} \frac{v_w}{\lVert v_w\rVert}$$

The 300 features are then standardized with training-set statistics - gradient descent on raw
averaged vectors is badly conditioned because the dimensions have very different means and
scales. On top of that sits multinomial (softmax) logistic regression,

$$P(y=c\mid x)=\frac{\exp(w_c^\top x + b_c)}{\sum_{c'}\exp(w_{c'}^\top x + b_{c'})}$$

trained by minimising the L2-regularised cross-entropy with mini-batch **Adam**, early
stopping on validation macro-F1 and restoring the best weights.

## 5. Classifier 2 - multi-layer perceptron, $n = 0, 1, 2$

The same `MLP` class covers all three settings; `hidden = ()` is $n = 0$, `hidden = (256,)` is
$n = 1$ and `hidden = (256, 256)` is $n = 2$. Hidden layers use ReLU with He initialisation
and inverted dropout (0.2); the output layer is the same softmax as above. Backpropagation is
the textbook loop, written out explicitly.

**$n = 0$ has no hidden layer, so it *is* the logistic regression of §4.** It is trained again
anyway, from a different random seed, and lands within 0.002 macro-F1 of §4 - a useful check
that both code paths and the optimiser are behaving.

## 6. Classifiers 3 and 4 - LSTM and GRU

The recurrent models read the sentence word by word instead of collapsing it into a mean. Each
sentence becomes a length-40 array of row indices into the embedding table (40 covers all but
3 % of the training sentences); padding is masked at every time step, so a short sentence
simply stops updating its state, and the sentence representation is the hidden state **after
its last real token**. A linear softmax layer on top produces the language distribution. The
embeddings stay **frozen** - the assignment asks for classifiers over the fastText vectors,
not for a second round of embedding training.

**LSTM** (gates stacked in the order $i, f, o, g$; the forget-gate bias is initialised to 1 so
the cell remembers by default):

$$\begin{aligned}
i_t &= \sigma(x_tW_{xi}+h_{t-1}W_{hi}+b_i) & f_t &= \sigma(x_tW_{xf}+h_{t-1}W_{hf}+b_f)\\
o_t &= \sigma(x_tW_{xo}+h_{t-1}W_{ho}+b_o) & g_t &= \tanh(x_tW_{xg}+h_{t-1}W_{hg}+b_g)\\
c_t &= f_t\odot c_{t-1}+i_t\odot g_t & h_t &= o_t\odot\tanh(c_t)
\end{aligned}$$

**GRU** (gates stacked in the order $z, r, n$):

$$\begin{aligned}
z_t &= \sigma(x_tW_{xz}+h_{t-1}W_{hz}+b_z), \qquad r_t = \sigma(x_tW_{xr}+h_{t-1}W_{hr}+b_r)\\
n_t &= \tanh\big(x_tW_{xn}+r_t\odot(h_{t-1}W_{hn})+b_n\big)\\
h_t &= (1-z_t)\odot n_t+z_t\odot h_{t-1}
\end{aligned}$$

Both are trained with **backpropagation through time**, derived by hand and implemented in one
reverse loop per cell type, with global gradient-norm clipping at 5.0 and Adam. Masked steps
pass their state through unchanged, so their gradient passes through unchanged too - that is
the `dh_skip` / `dc_skip` term in the backward pass.

Two implementation details make this fast enough to run in `numpy`: the input projection
$XW_x$ is computed for **all** time steps in one matrix multiply before the recurrence starts,
and each mini-batch is trimmed to the length of its longest sentence rather than to
`max_len`. An epoch takes about 16 s for the LSTM and 11 s for the GRU, and both stop early
at epoch 14.

### 6.1 Gradient check

Hand-derived BPTT is exactly the kind of code that is subtly wrong in a way that still trains,
so before spending minutes on training the notebook compares the analytic gradients against
central finite differences on a tiny random problem:

```
lstm  worst relative error between analytic and numerical gradients: 3.36e-07  OK
gru   worst relative error between analytic and numerical gradients: 2.51e-06  OK
```

## 7. Classifier 5 - fastText supervised

fastText's supervised classifier is a linear softmax over the average of **learned** word and
n-gram embeddings - architecturally the same thing as §4, with two differences that turn out to
matter enormously:

* the embeddings are trained on **this task** instead of on Wikipedia, so every word of every
  language gets one, and
* the **character n-grams** of each word are embedded too (`minn=2`, `maxn=5`), which is
  precisely the evidence language identification lives on - and the reason Assignment 1's
  character n-gram TF-IDF worked so well.

The notebook trains it twice: once from random initialisation, and once with its word
embeddings **initialised from the merged wiki vectors** of §2.3 (`pretrainedVectors`, the one
place where the pretrained vectors and the supervised model can be combined). Training runs on
a single thread, which is the only way fastText is reproducible.

## 8. Results

All scores are on the held-out **test** split; models are selected on validation macro-F1.

| # | classifier | configuration | parameters | val macro-F1 | **test macro-F1** | test accuracy | train time |
|---|---|---|---|---|---|---|---|
| 1 | Logistic regression | avg. word vectors | 6 923 | 0.8499 | **0.8451** | 0.8561 | 0.4 s |
| 2 | MLP (*n* = 0) | avg. word vectors, no hidden layer | 6 923 | 0.8493 | **0.8464** | 0.8578 | 0.7 s |
| 2 | MLP (*n* = 1) | avg. word vectors, hidden = [256] | 82 967 | 0.8536 | **0.8591** | 0.8696 | 3 s |
| 2 | MLP (*n* = 2) | avg. word vectors, hidden = [256, 256] | 148 759 | 0.8516 | **0.8581** | 0.8696 | 18 s |
| 3 | LSTM | sequence of word vectors, hidden = 128 | 222 615 | 0.8715 | **0.8789** | 0.8791 | 235 s |
| 4 | GRU | sequence of word vectors, hidden = 128 | 167 703 | 0.8770 | **0.8841** | 0.8835 | 152 s |
| 5 | fastText supervised | dim 100, word bigrams, char 2-5 grams | 210 264 900 | 0.9434 | **0.9402** | 0.9404 | 38 s |
| 5 | fastText, wiki-initialised | dim 300, initialised from `wiki.*.vec` | 633 531 600 | 0.9421 | **0.9479** | 0.9478 | 111 s |

The parameter counts for fastText are not comparable with the rest: almost all of them are the
2 000 000 hashed character-n-gram buckets that fastText allocates by default, the vast
majority of which are never touched by an 18 400-sentence corpus.

The ordering is clean and reads as a single story:

* **Averaging destroys word order, and word order barely matters here.** Adding hidden layers
  to the averaged features buys 1.4 points and then stops - $n = 2$ is no better than
  $n = 1$. With 800 sentences per language there is not enough signal for a deeper network.
* **Reading the sentence sequentially buys another 2.5 points.** The LSTM and GRU see the same
  frozen vectors as the MLPs; the gain comes from being able to weight and combine tokens
  rather than average them. The **GRU beats the LSTM** (0.8841 vs 0.8789) with 25 % fewer
  parameters and 35 % less training time, which is the usual outcome on short texts and small
  datasets - the LSTM's extra cell state is capacity this task cannot use.
* **fastText wins by 5.6 points over the best from-scratch model, and it wins for a reason
  that has nothing to do with its architecture** - it is the *shallowest* model in the table.
  It wins because it learns embeddings for character n-grams on this corpus, so it has
  features for Manipuri and Santali, which the pretrained vectors simply do not cover.
* **Initialising fastText from the wiki vectors adds a further 0.8 points** (0.9402 → 0.9479),
  confirming the pretrained vectors are genuinely useful - just not sufficient on their own.

For reference, **Assignment 1 reached 0.9465** on this same test split, with nothing but
from-scratch TF-IDF over word 1/2-grams and character 2/3/4-grams and the same logistic
regression - better than every model here except the wiki-initialised fastText (0.9479), and
better than fastText trained from scratch (0.9402). It is the same lesson from a third
direction: on this task the evidence lives in **character sequences**, and 300-dimensional
averaged word vectors throw most of it away. Assignment 1's ablation is worth putting next to
the table above - character 2/3/4-grams alone scored 0.9435 there, word 1/2-grams alone only
0.9123.

## 9. Reading the per-language scores

Per-language test F1, sorted by the coverage of §2.4, makes the mechanism explicit:

| language | coverage | LR | MLP *n*=1 | LSTM | GRU | fastText | fastText + wiki |
|---|---|---|---|---|---|---|---|
| `sat` | 1.3 % | 0.601 | 0.590 | 0.545 | 0.564 | **1.000** | **1.000** |
| `mni` | 3.6 % | 0.039 | 0.019 | 0.565 | 0.606 | **0.990** | **0.990** |
| `bd` | 34.0 % | 0.688 | 0.821 | 0.804 | 0.839 | 0.969 | 0.969 |
| `ks` | 34.0 % | 0.949 | 0.950 | 0.949 | 0.954 | 0.995 | 0.990 |
| `sa` | 45.3 % | 0.881 | 0.892 | 0.873 | 0.900 | 0.975 | 0.975 |
| `dg` | 57.4 % | 0.785 | 0.806 | 0.780 | 0.844 | 0.941 | 0.950 |
| `mai` | 74.8 % | 0.853 | 0.893 | 0.881 | 0.923 | 0.951 | 0.970 |
| `gom` | 83.8 % | 0.673 | 0.734 | 0.702 | 0.670 | 0.713 | 0.754 |
| `mr` | 85.1 % | 0.735 | 0.733 | 0.735 | 0.737 | 0.776 | 0.800 |
| `kha` | 89.5 % | 0.684 | 0.725 | 0.698 | 0.690 | 0.680 | 0.721 |
| `en` | 95.3 % | 0.685 | 0.701 | 0.765 | 0.667 | 0.680 | 0.720 |
| 12 others | 64-95 % | 0.94-1.00 | 0.95-1.00 | 0.95-1.00 | 0.95-1.00 | 0.99-1.00 | 0.99-1.00 |

The first seven rows are the low-coverage languages, and every one of them is recovered once a
classifier can see their words - fastText scores 0.94-1.00 on all seven. The last four are
high-coverage languages that *no* classifier improves: across all eight models, `gom`, `mr`,
`kha` and `en` each span less than 0.10 F1. That split is the whole story.

There are **two entirely different failure modes** in this table, and they respond to
different fixes.

**Failure mode 1 - the model cannot see the language.** Manipuri scores **0.019** with the
MLP: 88 % of its sentences average to the zero vector, so the classifier has literally no
input to work with. This is a coverage problem, and it is the reason the recurrent models beat
the feed-forward ones by more than their architecture would suggest - an LSTM reading a
sequence of mostly-`<unk>` tokens still sees *how many* tokens there were and where the few
known ones sat, which is enough to lift `mni` from 0.019 to 0.565. fastText fixes it
completely (0.990 / 1.000) because it builds features from the characters themselves.

**Failure mode 2 - the languages are genuinely confusable.** The remaining errors are almost
entirely two pairs, and no classifier in the table fixes them:

```
most frequent confusions (true -> predicted), fastText + wiki init
   28  en     -> kha        20  gom    -> mr
   25  kha    -> en         19  mr     -> gom
```

Assignment 1's TF-IDF model failed on exactly the same two pairs (`en` 0.697, `kha` 0.702,
`gom` 0.751, `mr` 0.780) while scoring 0.98-1.00 on `mni`, `sat`, `bd` and `ks` - which is the
clearest possible statement that these two failure modes are independent.

* **English / Khasi.** Khasi is written in the Latin script, and the IndicCorp `kha` file is
  heavily contaminated with English - many `kha` sentences *are* English sentences. The very
  first `kha` row of the training split is
  *"We are engaging with farmers to decide on next date of talks: Agriculture Minister Tomar"*.
  This is a property of the corpus, not of the classifiers, and it caps `en` and `kha` at
  ~0.72 for every model.
* **Konkani / Marathi.** Goan Konkani and Marathi are closely related Devanagari languages that
  share a large amount of vocabulary; separating them from one sentence is hard for a human too.

Everything else is essentially solved: 12 of the 23 languages sit above 0.93 for *every*
classifier in the table, and 7 of them are scored perfectly by the wiki-initialised fastText.

## 10. Ablation - how much of the gap is coverage rather than capacity?

Not required by the assignment, but it settles the question §9 raises. The averaged-vector
features are rebuilt with one change: every corpus word type *without* a pretrained vector
(50 541 of them) gets a **fixed random unit vector** instead of the zero vector. That adds no
linguistic information whatsoever - only the ability to tell those words apart.

| classifier | pretrained vectors only | + random vector for OOV | delta |
|---|---|---|---|
| Logistic regression | 0.8451 | 0.8926 | **+0.0474** |
| MLP (*n* = 1) | 0.8591 | 0.8986 | **+0.0396** |

Four to five macro-F1 points - more than the entire gain from adding hidden layers, and more
than the gain from switching to a recurrent architecture - come back from doing nothing but
*distinguishing* the words the pretrained table cannot see. Half of logistic regression's
9.5-point deficit against fastText closes without adding a single real feature. The ceiling on
classifiers 1-4 is set by embedding coverage, not by model capacity - which is the same
conclusion the fastText result reaches from the other direction.

## 11. Summary

* Reused the Assignment 1 language-identification splits unchanged - 23 languages x 1000
  sentence-tokenized sentences, 80:10:10 stratified.
* Streamed the fastText **wiki** vectors for all 23 languages, keeping only the ~70 000 word
  types the corpus actually uses; 18 languages have a published model and 5 do not, giving
  **69.9 %** overall token coverage.
* Merged the 18 per-language tables into one 69 795 x 300 lookup, resolving words claimed by
  several models to their most frequent source.
* Implemented from scratch, on `numpy` only: precision / recall / F1 / macro-F1, multinomial
  logistic regression, an MLP with $n = 0, 1, 2$ hidden layers, and an **LSTM** and a **GRU**
  with hand-derived backpropagation through time - the latter two verified against numerical
  gradients to $< 3\times10^{-6}$ relative error.
* Trained the fastText supervised classifier both from scratch and initialised from the merged
  wiki vectors.
* **Test macro-F1: 0.8451 (LR) < 0.8464 / 0.8591 / 0.8581 (MLP n = 0/1/2) < 0.8789 (LSTM) <
  0.8841 (GRU) < 0.9402 (fastText) < 0.9479 (fastText + wiki init).**
* Showed that the gap between the from-scratch models and fastText is a **vocabulary-coverage**
  gap, not an architecture gap: giving OOV words nothing but distinguishable random vectors
  recovers 4.7 of logistic regression's 9.5-point deficit - more than hidden layers (+1.4) and
  recurrence (+2.5) combined. The residual errors are two genuinely confusable pairs
  (English/Khasi, Konkani/Marathi) that no classifier in the table resolves.
