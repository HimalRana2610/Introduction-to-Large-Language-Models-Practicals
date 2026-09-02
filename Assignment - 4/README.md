## Notebooks

Implementation and results live in [`main.ipynb`](main.ipynb), [`main-hindi.ipynb`](main-hindi.ipynb).

# Assignment 4 - N-gram language models for Gujarati

**Course:** Introduction to Large Language Models
**Language:** Gujarati (`gu`)

## Problem statement

Using **the same 1 000 000-sentence corpus and the same train / dev / test splits built in
Assignment 3**, build unigram, bigram, trigram and quadgram language models ($N = 1,2,3,4$)
under five settings, and compare their perplexities:

1. **Unsmoothed** - raw maximum-likelihood counts, no smoothing
2. **Add-one (Laplace)** smoothing
3. **Add-K** smoothing, with $0 < K < 1$
4. **Interpolated** smoothing
5. **Kneser-Ney** smoothing with $d = 0.75$

For unigrams the prescribed form is

$$P(w)=\frac{c(w)+\lambda}{N+\lambda V}$$

where $\lambda$ is the interpolation constant and $V$ the vocabulary size. This is the unigram
row of the table and also the base case that the interpolated and Kneser-Ney recursions back
off to.

Everything is implemented from scratch.

> ### A note on memory
>
> Counting every 1-, 2-, 3- and 4-gram of a 1 000 000-sentence corpus, plus the context and
> continuation statistics Kneser-Ney needs, is the memory-hungry part of this notebook -
> roughly **8-12 GB of RAM** at the full corpus size. If the kernel runs out of memory, lower
> `CONFIG["lm_train_sentences"]` to e.g. 300 000 (about 2-3 GB): every qualitative conclusion
> below is unchanged, only the absolute perplexities shift. The cell that builds the counts
> reports its own memory use as it goes.

## 1. The data

The splits come straight from Assignment 3 - same corpus, same length-stratified 1000-sentence
dev and test sets. If they are not on disk (this notebook run on its own), they are rebuilt
from IndicCorp v2 with the same seed and the same procedure, so the result is identical.

## 2. Vocabulary and tokenization

Word-level models need a closed vocabulary. Any training word occurring fewer than `min_count`
times becomes `<unk>`, and every dev/test word outside the vocabulary maps to `<unk>` as well.
That is what makes the perplexities comparable at all: without a shared, closed vocabulary a
model could "win" simply by refusing to assign probability mass to rare words.

Sentences are padded with $N-1$ copies of `<s>` and terminated by `</s>`. The `</s>` token is
**predicted** (so the model has to learn where sentences end) while `<s>` only ever appears in
a context, so it is not part of $V$.

## 3. Counting

One pass over the training data produces every statistic the five smoothing methods need:

| structure | meaning | used by |
|---|---|---|
| `ngram[n]` | $c(w_{1..n})$ | all methods |
| `ctx_total[n]` | $c(h)=\sum_w c(h,w)$ | all methods |
| `ctx_types[n]` | $N_{1+}(h\,\bullet)$, distinct continuations of $h$ | Kneser-Ney |
| `cont[n]` | $N_{1+}(\bullet\,w_{1..n})$, distinct left contexts | Kneser-Ney (lower orders) |
| `cont_total[n]`, `cont_types[n]` | the same two sums over continuation counts | Kneser-Ney |

`ctx_total` is kept separately rather than reusing the $(n-1)$-gram counts, because sentence
padding makes the two differ at sentence boundaries and the difference would quietly corrupt
every conditional probability.

## 4. The five language models

Every model exposes the same interface: `logprob(context, word)` in natural log, and
`perplexity(token_lists)`.

$$\mathrm{PP}=\exp\!\left(-\frac{1}{M}\sum_{i=1}^{M}\ln P(w_i\mid h_i)\right)$$

where $M$ counts every predicted token, `</s>` included.

### 4.1 Unsmoothed (maximum likelihood)

$$P(w\mid h)=\frac{c(h,w)}{c(h)}$$

Any n-gram never seen in training gets probability exactly zero, so a single unseen n-gram in
the evaluation set makes the whole corpus probability zero and the perplexity infinite. That
is the point of the exercise, not a bug.

### 4.2 Add-one (Laplace) and 4.3 Add-K

$$P_{\text{add-}k}(w\mid h)=\frac{c(h,w)+k}{c(h)+kV}$$

with $k = 1$ for Laplace and $0<k<1$ tuned on the development set. For $N = 1$ this is exactly
the prescribed unigram formula with $\lambda = k$.

### 4.4 Interpolated smoothing

Simple linear interpolation, recursive from the highest order down:

$$P_n(w\mid h)=\lambda_n\,\frac{c(h,w)}{c(h)}+(1-\lambda_n)\,P_{n-1}(w\mid h')$$

where $h'$ drops the oldest word, and the recursion bottoms out at the prescribed unigram
$P_1(w)=\frac{c(w)+\lambda}{N+\lambda V}$. Every $\lambda_n$ is tuned on the development set
by coordinate descent; that guarantees the mixture is at least as good as the best single
order it contains.

### 4.5 Kneser-Ney smoothing ($d = 0.75$)

Interpolated Kneser-Ney. The highest order uses ordinary counts,

$$P_N(w\mid h)=\frac{\max(c(h,w)-d,\;0)}{c(h)}
  +\frac{d\,N_{1+}(h\,\bullet)}{c(h)}\;P_{N-1}(w\mid h')$$

and every lower order uses **continuation counts** $N_{1+}(\bullet\,h'w)$ - *how many distinct
contexts a string appears in*, not how often it appears:

$$P_n(w\mid h')=\frac{\max(N_{1+}(\bullet\,h'w)-d,\;0)}{N_{1+}(\bullet\,h'\bullet)}
  +\frac{d\,N_{1+}(h'\,\bullet)}{N_{1+}(\bullet\,h'\bullet)}\;P_{n-1}(w\mid h'')$$

That is the whole idea behind Kneser-Ney: a word like the second half of a fixed phrase may be
*frequent* yet appear after only one context, so it is a poor bet in a new context. Raw
frequency cannot express that; continuation counts can. The recursion bottoms out at the
prescribed unigram formula. When a context was never seen, that level contributes nothing and
the model falls through to the next one down.

## 5. Tuning on the development set

$K$ for add-K and the interpolation weights are hyper-parameters, so they are chosen on **dev**
and then reported on **test**. $d$ is fixed at 0.75 by the assignment and nothing else is tuned.

## 6. Results - perplexity of all 20 models

## 7. Sanity checks

A perplexity number is easy to compute and easy to get subtly wrong, so two properties are
checked directly rather than assumed.

## 8. Discussion

**Unsmoothed.** Infinite at every order above 1, and effectively infinite at order 1 too as
soon as a dev/test token was never seen in training (which the `<unk>` mapping mostly prevents,
so the unigram case usually survives). The "zero-prob %" column is the interesting number: it
rises steeply with $N$, because the chance that a specific 4-word sequence occurred in training
is far lower than the chance that a specific 2-word sequence did. This is the sparsity problem
in one column, and it is why every practical n-gram model is smoothed.

**Add-one.** Well defined everywhere, but badly calibrated. With $V$ in the hundreds of
thousands, adding 1 to every one of the $V$ possible continuations of a context moves an
enormous amount of probability mass away from the events that were actually observed - the
denominator $c(h)+V$ is dominated by $V$ for all but the most frequent contexts. So add-one
gets *worse* as the order rises, which is exactly the opposite of what a higher-order model
should do.

**Add-K.** The same estimator with the mass transfer turned down. Tuning $K$ on dev typically
lands well below 0.1 for the higher orders, and the perplexity improves by a large factor over
add-one. It is still a crude fix - a flat prior over all $V$ continuations regardless of
context - but it shows how much of add-one's damage is simply a badly chosen constant.

**Interpolation.** The first method that actually uses the lower-order models rather than a
uniform prior. A context seen only twice contributes a noisy estimate, but the bigram and
unigram distributions behind it are estimated from far more data, and mixing them in recovers
most of what the sparse high-order estimate loses. It should be the first setting where going
from bigram to trigram to quadgram *helps* instead of hurting.

**Kneser-Ney.** Usually the best of the five, and for a good reason. Absolute discounting takes
a fixed $d = 0.75$ off every observed count - which matches the empirical fact that held-out
counts are roughly a constant below training counts - and redistributes exactly that mass to
the lower order. The lower orders then use continuation counts instead of raw frequency, which
fixes the failure that motivated the method: a word can be common overall and still be a bad
guess in a new context, because it only ever appears in one fixed phrase. Interpolation cannot
express that distinction; Kneser-Ney can.

**Order.** With good smoothing, perplexity falls from unigram to bigram to trigram. The step
from trigram to quadgram is much smaller, and at this corpus size can even reverse: a 4-gram
context is so rarely seen that the model spends nearly all its time backing off anyway, so the
extra order buys little and costs a lot of memory.

**Gujarati specifics.** Gujarati is morphologically rich and written without the analytic
function words English relies on, so a word-level vocabulary is large and the token
distribution has a very long tail - the `min_count` cutoff and the `<unk>` rate reported in
section 2 make that concrete. Both effects push perplexity up relative to what the same setup
would give on English of the same size, and they make the choice of smoothing matter *more*,
not less: there is simply more probability mass sitting on events seen once or never.

## 9. Summary

* Reused the Assignment 3 Gujarati corpus and its train / dev / test splits unchanged.
* Built a closed vocabulary with an `<unk>` class and counted all 1- to 4-grams, together with
  the context and continuation statistics Kneser-Ney needs, in one pass.
* Implemented five smoothing settings from scratch - unsmoothed, add-one, add-K, interpolated
  and Kneser-Ney with $d = 0.75$ - all sharing the prescribed unigram
  $\frac{c(w)+\lambda}{N+\lambda V}$ as their base case.
* Tuned $K$ and the interpolation weights on dev only, and reported perplexity for all
  **4 orders x 5 settings = 20 models** on both dev and test.
* Verified that every smoothed model is a proper distribution over the vocabulary, and that
  the perplexity loop agrees with a brute-force recomputation.


---


# Hindi variant (`main-hindi.ipynb`)


# Assignment 4 - N-gram language models for Hindi

**Course:** Introduction to Large Language Models
**Language:** Hindi (`hi`) from
[IndicCorp v2](https://huggingface.co/datasets/ai4bharat/IndicCorpV2)

## Problem statement

This is the Assignment 4 experiment repeated for **Hindi**. The corpus is built with **the
same procedure as Assignment 3** - IndicCorp v2 streamed, sentence-tokenized, then split
into 1 000 000 training sentences and length-stratified dev and test sets of 1000 sentences
each, same seed - only the language file changes. On that corpus, build unigram, bigram, trigram and quadgram language models ($N = 1,2,3,4$)
under five settings, and compare their perplexities:

1. **Unsmoothed** - raw maximum-likelihood counts, no smoothing
2. **Add-one (Laplace)** smoothing
3. **Add-K** smoothing, with $0 < K < 1$
4. **Interpolated** smoothing
5. **Kneser-Ney** smoothing with $d = 0.75$

For unigrams the prescribed form is

$$P(w)=\frac{c(w)+\lambda}{N+\lambda V}$$

where $\lambda$ is the interpolation constant and $V$ the vocabulary size. This is the unigram
row of the table and also the base case that the interpolated and Kneser-Ney recursions back
off to.

Everything is implemented from scratch.

## 1. The data

The corpus is built exactly the way Assignment 3 builds it - IndicCorp v2 streamed until enough
sentences have been collected, deduplicated, sentence-tokenized on danda / double danda / `.`,
filtered for length and script, then split with a length-stratified 1000-sentence dev and test
set - with `hi` in place of `gu` and the Devanagari block in place of the Gujarati one. If the
Hindi splits are already on disk (from an earlier run of this notebook, or an Assignment 3 run
on Hindi) they are reused, so the slow download happens only once.

## Hindi-specific observations

**Hindi specifics.** Hindi is more analytic than Gujarati - case and much of the grammatical
relation is carried by separate postpositions (`का`, `के`, `की`, `में`, `से`) rather than by suffixes glued
onto the noun. Those postpositions are extremely frequent word types, so the bigram and trigram
distributions are sharper than the Gujarati equivalents and the higher orders should pay off a
little more. Working the other way, verbs still inflect heavily and compound verbs are written
as separate words, and the Hindi slice of IndicCorp is the largest and the most orthographically
varied (nukta present or absent, `ॉ` versus `ो`, inconsistent anusvara-versus-conjunct spelling), all
of which inflates the type count. The net effect is a large vocabulary with a long tail, which
the `min_count` cutoff and the `<unk>` rate reported in section 2 make concrete: a lot of
probability mass sits on events seen once or never, which is exactly the regime where the choice
of smoothing decides the result.

## 9. Summary (Hindi)

* Built a Hindi corpus and its train / dev / test splits with the Assignment 3 procedure,
  unchanged apart from the language file and the script range.
* Built a closed vocabulary with an `<unk>` class and counted all 1- to 4-grams, together with
  the context and continuation statistics Kneser-Ney needs, in one pass.
* Implemented five smoothing settings from scratch - unsmoothed, add-one, add-K, interpolated
  and Kneser-Ney with $d = 0.75$ - all sharing the prescribed unigram
  $\frac{c(w)+\lambda}{N+\lambda V}$ as their base case.
* Tuned $K$ and the interpolation weights on dev only, and reported perplexity for all
  **4 orders x 5 settings = 20 models** on both dev and test.
* Verified that every smoothed model is a proper distribution over the vocabulary, and that
  the perplexity loop agrees with a brute-force recomputation.
