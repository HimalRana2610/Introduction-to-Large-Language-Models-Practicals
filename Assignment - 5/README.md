## Notebook

Implementation and results live in [`main.ipynb`](main.ipynb).

# Assignment 5 - Embeddings and clustering for Gujarati

**Course:** Introduction to Large Language Models
**Language:** Gujarati (`gu`)

## Problem statement

Using **the same 1 000 000-sentence training data built in Assignment 3**:

1. Train three embedding models on the training data:
   * **(a) word2vec**, **(b) fastText**, **(c) Doc2Vec**
2. Use **K-means** to cluster the words of the training data.
   * **(a)** represent each cluster by the **20 words closest to its centroid**;
   * **(b)** represent every **validation and test** sentence as a vector, using both word
     embeddings and Doc2Vec embeddings, and **for each sentence find the closest sentence**.

The three models are trained with `gensim`, following the tutorials the assignment links to
([word2vec](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html),
[fastText](https://radimrehurek.com/gensim/auto_examples/tutorials/run_fasttext.html),
[Doc2Vec](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html)).
K-means, the sentence encoders and the nearest-neighbour search are written from scratch on
top of `numpy`, in keeping with the earlier assignments.

> ### Dependencies
>
> This notebook needs **`gensim`** (>= 4.3), which the earlier assignments did not use.
> Install it into the project environment before running:
>
> ```
> pip install "gensim>=4.3"
> ```
>
> `numpy`, `scipy` and `matplotlib` come with it or are already present.

> ### A note on runtime
>
> Training three embedding models over 1 000 000 sentences is the expensive part of this
> notebook - roughly **30-70 minutes in total** on a 4-core machine, fastText and Doc2Vec being
> the slow ones. Every model is cached under `models/`, so re-running the notebook reloads
> instead of retraining. If you want a quick pass first, drop
> `CONFIG["train_sentences"]` to e.g. 100 000: every qualitative conclusion below survives,
> only the embedding quality drops.

## 1. The data

The splits come straight from Assignment 3 - the same 1 000 000 training sentences and the same
length-stratified 1000-sentence validation and test sets. If they are not on disk (this
notebook run on its own), they are rebuilt from IndicCorp v2 with the same seed and the same
procedure, so the result is identical either way.

### 1.1 Tokenization and a streaming corpus

The pre-tokenizer is the same one used in Assignments 3 and 4, so the vocabulary here matches
the vocabulary the n-gram models saw: runs of Gujarati characters stay together (which keeps a
consonant attached to its matras), Latin runs and digit runs form words, and every other
non-space character is its own token.

One million tokenized sentences held in a Python list costs a couple of gigabytes, so the
training corpus is exposed as a **restartable iterator** that re-reads the file on every pass -
the `MyCorpus` pattern from the gensim word2vec tutorial. gensim needs several passes over the
data, which a plain generator could not provide.

## 2. Task 1a - word2vec

Skip-gram with negative sampling. Skip-gram is the right default here: Gujarati is
morphologically rich, so the vocabulary has a very long tail of rare inflected forms, and
skip-gram learns rare words better than CBOW (CBOW averages the context and effectively
smooths them away).

Each model is trained once and cached - re-running the cell reloads it from `models/`.

## 3. Task 1b - fastText

Same objective as word2vec, but every word is additionally represented as a bag of character
n-grams, and its vector is the sum of the n-gram vectors. Two consequences matter a lot for
Gujarati:

* **morphology is shared.** Inflected forms of the same stem overlap in their character
  n-grams, so they end up near each other even if some of them are individually rare.
* **there is no OOV.** A word never seen in training still has character n-grams, so fastText
  can produce a vector for it. word2vec cannot - this is demonstrated below, and it is what
  makes fastText the better sentence encoder in section 7.

## 4. Task 1c - Doc2Vec

Doc2Vec (paragraph vectors) learns a vector per *document* alongside the word vectors. Every
training sentence is one document, tagged by its line number.

The linked tutorial uses PV-DM (`dm=1`) with 40 epochs over a 300-document corpus. Forty
epochs over 1 000 000 documents is not a sensible trade, so the epoch count follows
`CONFIG["epochs"]` like the other two models; with three orders of magnitude more data, far
fewer passes are needed.

## 5. The three models side by side

## 6. Task 2 - K-means clustering of the words

### 6.1 The algorithm

K-means from scratch, with **k-means++** initialisation (seeding centroids far apart, which
avoids the bad local optima that uniform random seeding regularly falls into) and several
restarts, keeping the lowest-inertia run.

The word vectors are **L2-normalised** before clustering. On unit vectors squared Euclidean
distance is $2-2\cos\theta$, a monotone function of cosine similarity, so Euclidean k-means on
normalised vectors is *spherical* k-means - clustering by direction, which is what carries the
meaning in an embedding space, rather than by vector magnitude, which mostly tracks word
frequency.

### 6.2 Clustering the training vocabulary

The vectors come from **word2vec**. (`CONFIG["cluster_vocab_limit"]` can restrict clustering to
the N most frequent words; by default the whole vocabulary is clustered, and gensim orders
`index_to_key` by descending frequency so a limit would simply take the head.)

### 6.3 Task 2a - the 20 words closest to each centroid

For every cluster, the 20 vocabulary words whose vectors are closest to the centroid. Because
the vectors are unit length, "closest" is by cosine similarity to the centroid direction.

Note these are the words closest to the centroid **among the members of that cluster** - which
is what makes them a description *of the cluster*; the nearest word overall could belong to a
neighbouring cluster.

## 7. Task 2b - sentence vectors and the closest sentence

### 7.1 Four ways to turn a sentence into a vector

| encoder | how |
|---|---|
| **w2v-mean** | mean of the word2vec vectors of the in-vocabulary words |
| **ft-mean** | mean of the fastText vectors - no word is ever skipped, since OOV words are built from character n-grams |
| **ft-SIF** | *smooth inverse frequency* weighted average of the fastText vectors, with the top principal component removed |
| **doc2vec** | `infer_vector`, i.e. the Doc2Vec model's own sentence representation |

The SIF encoder is included because a plain mean is dominated by high-frequency words, which
carry almost no information about what a sentence is *about*. SIF weights each word by
$\frac{a}{a+p(w)}$ - down-weighting frequent words smoothly instead of using a stop-word
list - and then removes the projection on the first principal component of the sentence set,
which is the direction that encodes "generic sentence" rather than content.

Every vector is L2-normalised at the end, so the nearest neighbour under cosine similarity is
just the largest dot product.

### 7.2 Finding the closest sentence

For every sentence in the pool, the most similar *other* sentence by cosine similarity. The
whole pool is 2000 sentences, so the similarity matrix is computed in one go and the diagonal
is masked out so that a sentence cannot retrieve itself.

### 7.3 Do the encoders agree with each other?

There is no gold standard for "the closest sentence", but if two encoders pick the same
neighbour for the same query, that is evidence both found something real rather than noise.

## 8. A quantitative check on the sentence encoders

Agreement is suggestive but not decisive - four bad encoders could agree with each other. So
here is a retrieval task with a **known** right answer: perturb each sentence (drop a few
words and lightly shuffle the rest), then ask each encoder to retrieve the original from the
2000-sentence pool. A good sentence representation should be robust to that kind of edit, so
**Recall@1** is directly meaningful and directly comparable across encoders.

## 9. Discussion

**word2vec vs fastText.** The two models see exactly the same corpus and the same objective,
and differ only in how a word is represented. For Gujarati that difference is large. A word
like a noun with a case suffix appears far less often than its stem, so word2vec either misses
it (below `min_count`) or estimates it from a handful of occurrences. fastText shares the
character n-grams across every inflected form of the stem, so the whole paradigm is learned
together and rare forms inherit the stem's neighbourhood. The nearest-neighbour lists in
sections 2 and 3 make this visible: fastText's neighbours tend to be *morphological* variants
of the query, word2vec's tend to be *distributional* associates. Neither is strictly better -
fastText's bias towards surface-similar words is a real cost when you want semantic rather
than morphological similarity - but the OOV property is decisive: word2vec simply has no answer
for words it never saw, and the dev/test OOV count in section 3 shows how often that happens.

**Doc2Vec.** Learns sentence vectors directly rather than composing them from words, so it can
in principle capture word order and sentence-level topic that any averaging encoder discards.
The cost is that inference for a new sentence is itself a small optimisation (`infer_vector`
runs gradient steps), which makes it stochastic and slow, and it needs far more data per
document than these one-sentence documents provide. The self-retrieval check in section 4 is
the honest test of whether the document vectors mean anything at all.

**Clustering.** With unit-normalised vectors, k-means groups words by direction, and the
clusters that come out are the ones a distributional model can see: numbers and dates, place
names, verb inflections, function words, and a large diffuse cluster of rare words that the
model never had enough evidence about. The tightness plot in section 6 separates these - a
high mean cosine to the centroid means a genuinely coherent group, a low one means the cluster
is a catch-all. Cluster *size* alone is misleading, which is why both are plotted.

**Sentence encoders.** The perturbation probe in section 8 is the one number that is not a
matter of taste. Expect the frequency-weighted fastText encoder to lead: a plain mean is
dominated by the handful of very frequent words that every sentence contains, so plain-mean
vectors all point in roughly the same direction and the "closest sentence" is often just the
one with the most similar function-word mix. Down-weighting frequent words and removing the
first principal component strips exactly that shared component out. Doc2Vec's inferred vectors
are the least stable, because inference is a stochastic optimisation from a random start over
a short document.

**On "the closest sentence".** These are 2000 unrelated sentences sampled from a web corpus,
so for most queries there is no genuinely related sentence in the pool to find - the
distribution of nearest-neighbour cosines in section 7.3 shows this directly. The pairs worth
reading are the high-cosine ones at the top; the rest are the nearest of 1999 unrelated
options, which is a different thing entirely.

## 10. Summary

* Reused the Assignment 3 Gujarati corpus unchanged - 1 000 000 training sentences, and the
  length-stratified 1000-sentence validation and test sets - exposed as a restartable
  streaming iterator so the full corpus never has to be held in memory.
* Trained **word2vec**, **fastText** and **Doc2Vec** with gensim, following the linked
  tutorials, and cached all three under `models/`.
* Implemented **k-means** from scratch (k-means++ seeding, multiple restarts, empty-cluster
  re-seeding) and clustered the whole word2vec vocabulary, reporting the **20 words closest to
  each centroid** for every cluster.
* Encoded every validation and test sentence four ways - mean word2vec, mean fastText,
  SIF-weighted fastText, and Doc2Vec `infer_vector` - and found **the closest sentence** for
  each of the 2000 queries under every encoder, saving all pairs to `data/`.
* Compared the encoders three ways: nearest-neighbour agreement, the distribution of
  nearest-neighbour similarities, and a perturbed-sentence retrieval probe with a known
  correct answer, which gives a directly comparable Recall@1.
