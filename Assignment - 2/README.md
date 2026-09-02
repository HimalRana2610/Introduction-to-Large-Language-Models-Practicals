## Notebook

Implementation and results live in [`main.ipynb`](main.ipynb).

# Assignment 2 - Normalized vs unnormalized TF and IDF

**Course:** Introduction to Large Language Models

## Problem statement

Using **the same dataset built in Assignment 1** (1000 sentence-tokenized sentences per
IndicCorp v2 language, split 80:10:10 by stratified sampling):

1. Implement two ways of normalizing **TF**
   * (a) by the total number of words in the sentence,
   * (b) by the frequency of the most frequent word in the sentence.
2. Implement a normalization of **IDF** based on the most frequent word across sentences.
3. Train the logistic-regression language identifier on the normalized TF/IDF vectors and
   compare against the unnormalized vectors of Assignment 1. All combinations of normalized
   and unnormalized TF and IDF are required - **6 combinations** in total.

## The six combinations

| | IDF unnormalized $\log\frac{N}{df(t)}$ | IDF normalized $\log\frac{\max_{t'} df(t')}{df(t)}$ |
|---|---|---|
| **TF raw** $c(t,d)$ | 1 - Assignment 1 baseline | 4 |
| **TF / total words** $\dfrac{c(t,d)}{\sum_{t'} c(t',d)}$ | 2 | 5 |
| **TF / max word freq** $\dfrac{c(t,d)}{\max_{t'} c(t',d)}$ | 3 | 6 |

Everything (vectorizer, classifier, metrics) is again implemented from scratch - no `sklearn`.

## 1. Code carried over from Assignment 1

The tokenizers, the TF-IDF vectorizer, the softmax classifier and the metrics are exactly the
ones written in Assignment 1, repeated here so that this notebook runs standalone. The only
part that matters for *this* assignment is that `TfIdfVectorizer` was already parameterised
over `tf_scheme` and `idf_scheme`.

## 2. The dataset

The splits produced by Assignment 1 are loaded from `../Assignment - 1/data/`. If they are not
there (e.g. this notebook is run on its own), the identical dataset is rebuilt from
IndicCorp v2 with the same seed, so the splits come out exactly the same either way.

## 3. The TF and IDF normalization schemes

### 3.1 TF - three schemes

For a sentence $d$ and a feature $t$, with $c(t,d)$ the raw count:

| name | formula | meaning |
|---|---|---|
| `raw` | $\mathrm{tf}=c(t,d)$ | unnormalized - what Assignment 1 used |
| `len` | $\mathrm{tf}=\dfrac{c(t,d)}{\sum_{w} c(w,d)}$ | divide by the **total number of words** in the sentence |
| `max` | $\mathrm{tf}=\dfrac{c(t,d)}{\max_{w} c(w,d)}$ | divide by the **frequency of the most frequent word** in the sentence |

In both normalized schemes $w$ ranges over the **word unigrams** of the sentence, exactly as
the assignment states ("total number of words", "most frequent word"). The resulting
per-sentence factor is then applied to every feature of that sentence, character n-grams
included, so the whole vector stays on one consistent scale.

### 3.2 IDF - two schemes

| name | formula | meaning |
|---|---|---|
| `standard` | $\mathrm{idf}(t)=\log\dfrac{N}{df(t)}$ | unnormalized - what Assignment 1 used |
| `maxdf` | $\mathrm{idf}(t)=\log\dfrac{\max_{t'} df(t')}{df(t)}$ | document frequency normalized by the **largest document frequency across sentences** |

### 3.3 A property worth being explicit about

Both TF normalizations multiply an entire row by one **document-dependent constant**. If the
final vectors were rescaled to unit L2 length, that constant would cancel exactly and all
three TF schemes would produce *identical* matrices - the comparison would be vacuous. The
grid below therefore uses **no vector-length normalization** (`norm=None`), so that the TF and
IDF weighting schemes are the only thing that changes. Section 6 verifies the cancellation
numerically.

## 4. The classifier (unchanged from Assignment 1)

## 5. Running all six combinations

The n-gram counts do not depend on the weighting scheme, so they are computed **once** and all
six configurations re-weight the same counts. Only the IDF vector, the per-document TF factor,
and the classifier change.

## 6. Verifying the L2 cancellation claim

Section 3.3 argued that unit-length row normalization would make the three TF schemes
collapse into one. Rather than take that on trust, here it is checked numerically.

## 7. Comparison of the six combinations

## 8. Discussion

**What each normalization actually does.**

* **TF / total words** turns raw counts into within-sentence *relative frequencies*. It removes
  sentence length as a confound: without it a long sentence simply produces larger feature
  values than a short one, and the classifier sees length as if it were evidence.
* **TF / most frequent word** is the classic *augmented* term frequency. It also divides by a
  per-sentence constant, but a much smaller one (the top word count is usually 1-3, while the
  word total is 10-40), so it compresses less aggressively and keeps more of the raw
  count magnitude.
* **IDF / max df** is an *additive shift in log space*, since
  $\log\frac{\max df}{df(t)} = \log\frac{N}{df(t)} - \log\frac{N}{\max df}$.
  Every IDF value drops by the same constant $\log\frac{N}{\max df}$ and is then floored at 0.
  The effect is therefore not a rescaling but a *compression of the IDF range from below*: the
  most common features are pushed to (or through) zero and drop out of the representation
  entirely, while rare features keep almost all of their weight. In other words this
  normalization behaves like an automatic, corpus-driven stop-word filter.

**Why the differences here are small.** Language identification over these scripts is close to
a solved problem for character n-gram features - most of the languages use mutually exclusive
Unicode blocks, so the character n-grams alone already separate them almost perfectly. The
weighting scheme therefore has very little room to change the outcome, and the six numbers sit
within a narrow band. Where they do differ is on the *genuinely hard* pairs, the languages that
share a script (the Devanagari group: Hindi / Marathi / Nepali / Maithili / Konkani /
Sanskrit / Dogri, and the Perso-Arabic group: Urdu / Kashmiri / Sindhi). The per-language plot
and the sensitivity table above are where the effect of the normalization is visible.

**Practical takeaway.** Normalizing TF matters most when the documents vary a lot in length;
normalizing IDF by the maximum document frequency mostly acts as a soft stop-word filter. On
short, single-sentence inputs of roughly comparable length, and with a feature set as
discriminative as character n-grams, neither is decisive - but the length normalization is the
safer default because it makes the representation independent of how long the input happens to
be.

## 9. Summary

* Implemented two TF normalizations (by sentence word count, and by maximum word frequency)
  and one IDF normalization (by the maximum document frequency across sentences), all from
  scratch, on the Assignment 1 dataset.
* Trained and evaluated the from-scratch logistic regression on all **6** TF x IDF
  combinations, reporting validation and test macro-F1 for each.
* Showed numerically that vector-length (L2) normalization would collapse the three TF schemes
  into one, and therefore ran the comparison without it.
* Analysed *where* the schemes differ - not in overall macro-F1, but on the languages that
  share a script with another language in the corpus.
