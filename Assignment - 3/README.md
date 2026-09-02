## Notebook

Implementation and results live in [`main.ipynb`](main.ipynb).

# Assignment 3 - Subword tokenization for Gujarati (BPE and WordPiece)

**Course:** Introduction to Large Language Models
**Language:** Gujarati (`gu`) from
[IndicCorp v2](https://huggingface.co/datasets/ai4bharat/IndicCorpV2)

## Problem statement

Each row of IndicCorp is a paragraph, so the data must first be sentence-tokenized. Then:

1. Create **training / development / test** splits. Development and test hold **1000 sentences
   each**, the rest is training data. Dev and test must **not** consist of sentences of
   similar lengths - pick sentences of different lengths by maintaining a length distribution.
2. Implement **BPE** and **WordPiece** on the training data and tokenize dev and test.
3. Test **4 training-data sizes** - 100 000 / 300 000 / 500 000 / 1 000 000 sentences - and
   show how the tokenization changes with size.
4. Test **3 vocabulary sizes** - 20 000 / 30 000 / 50 000 - and show the impact of vocabulary
   on tokenization.

Both algorithms are implemented from scratch; no tokenizer library is used.

## How the 4 x 3 grid is obtained from only 8 training runs

Both BPE and WordPiece build their vocabulary by **greedily appending one merge at a time**.
The vocabulary of size 20 000 is therefore an exact *prefix* of the vocabulary of size 50 000
trained on the same data. So for each training size we train **once** up to the largest
vocabulary (50 000) and snapshot the merge list at 20 000 and 30 000. That gives the complete
4 (sizes) x 3 (vocabularies) grid from 4 runs per algorithm instead of 12.

## 1. Getting the Gujarati data

### 1.1 Streaming the corpus

The Gujarati file in IndicCorp v2 is far larger than we need, so it is streamed over HTTP and
reading stops as soon as enough sentences have been collected. The result is cached on disk so
the (slow) download happens only once.

### 1.2 Sentence tokenization

Gujarati is written in the Gujarati script and, like the other Brahmic scripts, ends sentences
with the **danda** `U+0964`. Latin punctuation also appears in web text, so the tokenizer
handles `।`, `॥`, `?`, `!` and `.` - the full stop only when it is not a decimal point or an
abbreviation.

## 2. Train / dev / test splits with a controlled length distribution

The requirement is that dev and test must **not** be made of sentences of similar length. The
raw corpus length distribution is heavily skewed towards short sentences, so drawing 1000
sentences at random would give an evaluation set dominated by short ones.

Instead we **stratify by length**:

1. measure every sentence's length in words;
2. cut the corpus into `n_length_bins` equal-frequency bins (length quantiles);
3. draw the same number of sentences from **every** bin for dev and again for test.

The result is a dev/test length distribution that is approximately **uniform across the whole
length range** - short, medium and long sentences are equally represented - while the training
set keeps the corpus's natural distribution.

## 3. Pre-tokenization

Both BPE and WordPiece operate on a **word-frequency table**, not on raw text: a word type that
occurs 5000 times is processed once with weight 5000. Pre-tokenization splits a sentence into
those word units.

For Gujarati the rules are:

* runs of Gujarati characters (`U+0A80`-`U+0AFF`) form one word - this deliberately keeps a
  consonant together with its matras and the virama, which a naive `\w+` regex would split;
* runs of Latin letters and runs of digits form words;
* every other non-space character (punctuation, danda, symbols) becomes its own unit.

## 4. Byte Pair Encoding from scratch

### The algorithm

1. Split every word type into characters; mark the end of the word by attaching `</w>` to the
   last character (so `bank</w>` and `bank`ing are not confused).
2. Count every adjacent symbol pair, weighted by word frequency.
3. Repeatedly merge the **most frequent** pair into a new symbol, until the vocabulary reaches
   the target size.

### Making it fast enough

A naive implementation rescans every word after every merge, which is hopeless at 50 000
merges over ~1M word types. Two things make it tractable:

* **an inverted index** `pair -> {word ids containing it}`, so a merge only touches the words
  that actually contain the merged pair;
* **a lazy max-heap** over pair frequencies, so finding the best pair is $O(\log P)$ rather
  than a linear scan. Heap entries go stale as frequencies change, so each popped entry is
  validated against the live count and discarded if out of date.

Because merges are appended one at a time, truncating the merge list to the first $k$ entries
gives exactly the tokenizer that would have been trained with a vocabulary of $k$ + alphabet
size. That is what makes the snapshots in section 6 valid.

## 5. WordPiece from scratch

WordPiece uses the same merge loop but a different **selection criterion**. Instead of picking
the most frequent pair it picks the pair that most increases the likelihood of the training
data under a unigram model, which reduces to maximising

$$\mathrm{score}(a,b)=\frac{\mathrm{freq}(ab)}{\mathrm{freq}(a)\times\mathrm{freq}(b)} .$$

The denominator is what makes WordPiece different: a pair of two very common symbols has to be
*much* more frequent than chance before it is merged, so WordPiece prefers merges that are
genuinely informative rather than merely frequent. Continuation pieces are marked with `##`.

Because the score of a pair changes whenever either of its symbols changes frequency, the heap
entries go stale far more often than in BPE. A popped entry is therefore re-scored and pushed
back if it is out of date, and only accepted when its stored score still matches the live one.

## 6. The experiment grid

4 training sizes x 3 vocabulary sizes x 2 algorithms = 24 tokenizers, obtained from 8 training
runs thanks to the snapshotting described at the top.

> **Runtime.** This is the expensive cell. Training BPE and WordPiece to a 50 000-token
> vocabulary over 1 000 000 sentences in pure Python takes a while (tens of minutes per run on
> a typical machine). Trained models are cached under `models/`, so re-running the notebook
> reuses them. `CONFIG["min_word_freq"] = 2` drops singleton word types, which roughly halves
> the work with a negligible effect on the learned vocabulary.

### 6.1 Tokenizing dev and test

For every tokenizer we measure, on both dev and test:

| metric | meaning |
|---|---|
| **tokens/sentence** | average sequence length the model would see |
| **fertility** | tokens per pre-tokenized word - 1.0 means "every word is one token" |
| **chars/token** | how much text one token carries (compression) |
| **continuation %** | share of tokens that are word-internal pieces |
| **UNK %** | share of tokens the vocabulary could not cover |
| **vocab used %** | how much of the trained vocabulary the eval set actually exercises |

## 7. Task 3 - the effect of training-data size

Vocabulary size is held fixed while the amount of training data varies.

## 8. Task 4 - the effect of vocabulary size

Training-data size is held fixed while the vocabulary varies.

## 9. How the tokenization itself changes

The numbers above summarise the effect; these cells show it directly on real sentences.

## 10. Summary and observations

**Splits.** The corpus was sentence-tokenized from IndicCorp v2 paragraphs and split into
1 000 000 training sentences plus 1000 dev and 1000 test sentences. Dev and test were drawn by
**length stratification** - equal-frequency length bins, equal numbers sampled from each - so
both evaluation sets span the full length range instead of being dominated by the short
sentences that make up most of the corpus. The two histograms in section 2 show the
difference between the natural training distribution and the flattened evaluation one.

**What to expect from the numbers.**

* *More training data* mainly improves **coverage**: with more text the merge statistics are
  estimated from a much larger sample, so the learned vocabulary contains the genuinely common
  Gujarati morphemes rather than the accidents of a small sample. The visible effects are a
  falling UNK rate and a modest fall in fertility. The returns diminish quickly - the step from
  100k to 300k sentences moves the numbers far more than the step from 500k to 1M, because the
  frequent-morpheme inventory of a language is not that large.
* *A larger vocabulary* has a much stronger and more direct effect: every extra merge glues two
  pieces together, so tokens get longer, sequences get shorter, and fertility falls steadily
  from 20k to 50k. The cost is a larger embedding matrix and more rare tokens that are seen
  too seldom to be learned well - which is why the "vocab used %" column falls as the
  vocabulary grows.
* *BPE vs WordPiece.* BPE merges by raw frequency and therefore builds long, common strings
  fast. WordPiece divides by the frequency of the two parts, so it resists merging two already
  common symbols and tends to produce pieces that align better with morpheme boundaries. At the
  same vocabulary size the two usually land at similar fertility but on noticeably different
  vocabularies, as the overlap table shows. WordPiece can also emit `[UNK]` for a whole word
  when no prefix matches, whereas BPE always falls back to characters - visible in the UNK
  column.

**Gujarati specifics.** Gujarati is moderately agglutinative and written in an abugida, so a
single orthographic word packs a root plus several inflectional suffixes, and a single
"character" is often a consonant plus matras plus virama. The pre-tokenizer therefore keeps
whole Gujarati character runs together and lets the subword algorithm discover the morpheme
boundaries itself - which is exactly what shows up in the sample tokenizations in section 9,
where common case and postposition endings are split off as their own pieces.
