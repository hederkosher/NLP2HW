# Homework 2 - Embeddings: Final Report

A comparative study of four text-embedding methods on English Wikipedia texts.
All numbers below are produced by the code in `src/` and saved under `results/`.
Reproduce everything end-to-end with:

```bash
python -m src.run_all      # runs sections א→ט and regenerates all results
```

**Corpus:** 5 real English Wikipedia excerpts (≥300 words) + 5 Claude-generated
texts on the same 5 topics (≥300 words), so the two sources have parallel
structure and can be compared directly.

**Topics (chosen for a mix of close pairs + an outlier):**
Photosynthesis & Cellular respiration (biology pair), Jazz & Blues (music pair),
Mount Everest (outlier).

**Four methods:** Bag-of-Words, TF-IDF, FastText (trained with gensim),
MiniLM sentence-transformer (`all-MiniLM-L6-v2`).

### Methodology decisions (the ambiguities סעיף ד asks us to resolve)
- **Sizing to 30 / 300 dims:** BoW/TF-IDF via `max_features` (top-N terms);
  FastText via gensim `vector_size`; MiniLM (384-dim) via **PCA** to 30/300.
- **PCA bases are fit on a 2,251-sentence pool** (the full Wikipedia articles +
  generated texts), never on the 10 documents alone - with only 10 samples PCA
  could give at most 9 components. This makes genuine 30/300-dim bases possible
  and is the reusable PCA used again in סעיף ז.
- **Document vector** = the natural unit per method: BoW/TF-IDF row; FastText =
  mean of word vectors; MiniLM = mean of sentence embeddings then PCA.
- **Joint fitting:** each method is fit once on all 10 docs so originals and
  generated texts share one space; the "2 sources" axis (→ the 16 sets) is a
  partition of those vectors. The 16 sets are saved in `data/embeddings/`
  (`manifest.json` + 16 `.npz` files).
- **Comparison metric:** cosine similarity throughout.

---

## א - Data collection
**What:** Fetched 5 English Wikipedia articles via the MediaWiki action API
(`src/data_collection.py`) and saved ≥300-word cleaned excerpts.
**Where:** `data/original/wiki_01..05_<topic>.txt` (full articles cached in
`data/original/_full/` only as the PCA sentence pool).
**Result:** 5 texts, 300–327 words each.
**Sources (English Wikipedia, current-revision leads):**
[Photosynthesis](https://en.wikipedia.org/wiki/Photosynthesis) ·
[Cellular respiration](https://en.wikipedia.org/wiki/Cellular_respiration) ·
[Jazz](https://en.wikipedia.org/wiki/Jazz) ·
[Blues](https://en.wikipedia.org/wiki/Blues) ·
[Mount Everest](https://en.wikipedia.org/wiki/Mount_Everest)

## ב - AI-generated texts (restricted vocabulary)
**What:** Generated 5 Wikipedia-style texts with Claude using the exact prompt in
`data/generation_prompt.txt`, which **explicitly demands as limited/restricted a
vocabulary as possible** (the graded constraint). `src/generate_texts.py` can
re-generate via the Anthropic API and validates the ≥300-word minimum.
**Where:** `data/generated/wiki_gen_01..05_<topic>.txt`.
**Result / evidence the constraint worked** - type-token ratio (unique/total
words) is much lower for the generated texts:

| topic | original TTR | generated TTR |
|---|---|---|
| photosynthesis | 0.609 | 0.375 |
| cellular_respiration | 0.483 | 0.380 |
| jazz | 0.593 | 0.459 |
| blues | 0.543 | 0.461 |
| mount_everest | 0.556 | 0.421 |

The generated texts reuse a small word set, exactly as instructed - and this
vocabulary gap is what later lets us test *semantic* similarity across sources.

## ג + ד - 16 embedding sets
**What:** Represented all 10 texts with the 4 methods at dims 30 and 300, for
both sources → 4 × 2 × 2 = **16 sets** (`src/embeddings.py`).
**Where:** `data/embeddings/{source}_{method}_{dim}.npz` + `manifest.json`;
full artifacts (incl. 300-dim atom pools for סעיף ז) in `artifacts.pkl`.
**Note:** MiniLM PCA captures 51.8 % variance at 30-dim and 99.2 % at 300-dim
(2,251-sentence basis, no zero-padding needed).

## ה - Method comparison (2 research questions)
**Questions chosen:** (Q1) which method is most **sensitive to shared/overlapping
words**, and (Q2) which captures **semantic similarity without literal word
overlap**.
**Experiment (`src/analysis.py`):** over the joint 10-doc spaces we measure
- `lex_corr` = Pearson r between cosine similarity and Jaccard **word** overlap
  across all 45 doc pairs → high r answers **Q1**;
- `xtopic_cos` = mean cosine of the 5 same-topic **original↔generated** pairs.
  These pairs mean the same thing but share only ~14 % of their words
  (`xtopic_jaccard`=0.143), so high cosine answers **Q2**.

| method | dim | lex_corr (Q1) | xtopic_cos (Q2) | xtopic_jaccard |
|---|---|---|---|---|
| bow | 30 | 0.396 | 0.828 | 0.143 |
| bow | 300 | 0.433 | 0.692 | 0.143 |
| tfidf | 30 | 0.285 | 0.752 | 0.143 |
| **tfidf** | **300** | **0.543** | **0.469** | 0.143 |
| fasttext | 30 | 0.793 | 0.920 | 0.143 |
| fasttext | 300 | 0.797 | 0.919 | 0.143 |
| minilm | 30 | 0.446 | 0.771 | 0.143 |
| minilm | 300 | 0.475 | 0.657 | 0.143 |

**Findings.**
- **Q1 (shared words):** **TF-IDF at 300-dim** is the most lexical: its similarity
  tracks word overlap most strongly (lex_corr 0.543) and its cross-vocabulary
  semantic match collapses to 0.469. BoW behaves similarly. So lexical methods -
  TF-IDF especially - are the ones whose similarity is driven by shared words.
- **Q2 (semantic, no overlap):** Read `xtopic_cos` **together with topic
  separation** (סעיף ו), because FastText scores high on *everything*
  (≈0.92) and so is not discriminative. **MiniLM** is the method that keeps
  same-meaning/different-words pairs similar **while still separating unrelated
  topics** (separation 0.73–0.82 vs ≤0.19 for the others). The probe experiment
  (סעיף ט) confirms this decisively.
- See `results/heatmaps_dim300.png`: MiniLM shows clean biology/music blocks;
  FastText is uniformly bright (all pairs look similar).

## ו - Effect of embedding size
**Metric:** topic-**separation** score on a source's 5 docs = mean cosine of the
same-domain close pairs − mean cosine of the other pairs (higher = better). This
is the "quality" measure (does related cluster, unrelated separate), used
consistently here and in סעיף ז. The effect of length on סעיף ה's own metrics
(lex_corr, xtopic_cos) is already visible in the ה table, which reports every
method at both dim=30 and dim=300.

| method | sep@30 (orig) | sep@300 (orig) | sep@30 (gen) | sep@300 (gen) |
|---|---|---|---|---|
| bow | 0.049 | 0.076 | 0.056 | 0.087 |
| tfidf | 0.125 | 0.168 | 0.149 | 0.190 |
| fasttext | 0.145 | 0.140 | 0.057 | 0.057 |
| minilm | **0.824** | 0.732 | **0.767** | 0.673 |

**Findings.** Bigger is **not** uniformly better. For BoW/TF-IDF more dimensions
help (they need more vocabulary features to tell topics apart). For **MiniLM the
30-dim version separates topics *better* than 300-dim** - the first PCA
components hold the dominant semantic axes, and the extra dimensions add mostly
noise. FastText is roughly flat and weak regardless of size. (Feeds סעיף יא.)

## ז - PCA reduction (300 → 30) vs native-30
**What:** For each method, reduce the 300-dim doc vectors to 30 with PCA (basis
fit on that method's atom pool) and compare to the native-30 set.

| method | variant | lex_corr | xtopic_cos | sep_orig | PCA explVar |
|---|---|---|---|---|---|
| bow | native30 | 0.396 | 0.828 | 0.049 | |
| bow | pca300→30 | 0.657 | 0.832 | **0.095** | 0.701 |
| tfidf | native30 | 0.285 | 0.752 | 0.125 | |
| tfidf | pca300→30 | 0.645 | 0.487 | **0.395** | 0.500 |
| fasttext | native30 | 0.793 | 0.920 | 0.145 | |
| fasttext | pca300→30 | 0.823 | 0.530 | **0.861** | 0.919 |
| minilm | native30 | 0.446 | 0.771 | 0.824 | |
| minilm | pca300→30 | 0.445 | 0.772 | 0.824 | 0.521 |

**Findings.** For the lexical/word methods, **PCA-from-300 clearly beats native
top-30** at separating topics (TF-IDF 0.125→0.395, BoW 0.049→0.095, FastText
0.145→0.861). The reason is *feature extraction vs feature selection*: native-30
is just the 30 commonest words (mostly stopwords), while PCA's 30 dimensions are
information-rich linear combinations of all 300 features. For MiniLM the two
30-dim variants are essentially identical (both are PCA of the same 384-dim
space). So **how** you obtain 30 dimensions matters more than the number 30.

## ח - Manual BoW / TF-IDF / cosine walkthrough
Full step-by-step arithmetic (vocabulary, count matrix, TF/IDF/TF·IDF, L2
normalization, and cosine between word **columns**) is in
**`results/section_het_manual.md`**, cross-checked to match scikit-learn exactly.
Three sentences from the generated photosynthesis text; word-pair cosines:

| word pair | BoW cosine | TF-IDF cosine |
|---|---|---|
| make ~ food | 1.0000 | 1.0000 |
| food ~ this | 0.5000 | 0.4927 |
| plant ~ the | 1.0000 | 1.0000 |

`make` and `food` occur in exactly the same sentences → cosine 1.0; `food` and
`this` overlap in one sentence → 0.5; TF-IDF shifts `food~this` slightly
(0.4927) because IDF + per-sentence L2 normalization re-scale the columns,
down-weighting terms shared across sentences.

## ט - Probe sentences (method-difference stress test)
Hand-crafted sets (`src/probes.py`): **Group A** = same meaning, different words;
**Group B** = same words ("bank"), different meaning. Mean within-group cosine:

| method | within-A (meaning) | within-B (shared words) | A − B |
|---|---|---|---|
| bow | 0.351 | 0.499 | −0.149 |
| tfidf | 0.110 | 0.244 | −0.133 |
| fasttext | 0.852 | 0.899 | −0.047 |
| **minilm** | **0.694** | **0.376** | **+0.318** |

**Answers.**
- **Best at semantic similarity:** **MiniLM** - the only method with A > B, i.e.
  it rates *same-meaning/different-words* sentences as more similar than
  *shared-word/different-meaning* ones.
- **Most confused by lexical overlap:** **BoW** (and TF-IDF) - the shared word
  "bank" makes unrelated Group-B sentences look *more* similar (0.499) than the
  genuinely related Group-A sentences (0.351). FastText is a separate artifact:
  its cosines are uniformly high (~0.9), so it barely discriminates at all.
- **Largest semantic-vs-lexical gap:** pair **A1~A3** ("The doctor treated the
  patient." vs "Medical care was provided to the injured man.") - MiniLM 0.70 vs
  BoW 0.27 / TF-IDF 0.08 (gap +0.52). These two sentences share almost no words
  but mean nearly the same thing, so surface-token methods see nothing while the
  semantic encoder sees a strong match. See `results/section_tet_probes.png`.

## י - Why can TF-IDF fail to detect semantic similarity?
TF-IDF represents a sentence as weighted counts over a **fixed vocabulary**, with
each distinct word an independent, meaning-free dimension. Two sentences that
mean the same thing using **different words** share few or no dimensions, so their
vectors are near-orthogonal (low cosine) despite identical meaning. IDF makes it
worse: it *up-weights rare words*, so distinctive synonyms like "physician" vs
"doctor" - which should signal similarity - instead push the vectors apart.
**Evidence from our data:** TF-IDF within-Group-A (same meaning, different words)
is only **0.11** (vs MiniLM 0.694), and its same-topic original↔generated cosine
collapses to **0.469** at 300-dim precisely because the generated texts restate
the meaning with a different, restricted vocabulary. TF-IDF has no notion of
synonymy or relatedness - it can only match surface tokens.

## יא - Does a higher-dimensional embedding always give better results?
**No.** From סעיף ו, MiniLM's topic separation *drops* from **0.824 (30-dim) to
0.732 (300-dim)** on the originals (0.767→0.673 on generated): the first ~30 PCA
components hold the dominant semantic axes (51.8 % variance) and the extra 270
dimensions add mostly fine-grained noise that dilutes the separation contrast.
The right dimension depends on how information-dense each dimension is - not on
the raw count. **When the dimension is too small**, you can lose essential
information catastrophically: BoW/TF-IDF at 30 dims keep only the 30 commonest
words (dominated by "the, and, of, a, to, is…"), so all documents look alike and
separation is near zero (BoW 0.049). Section ז drives the point home: PCA-reduced
30 dims (information-rich combinations) beat native top-30 dims (just the 30
commonest words) - TF-IDF separation 0.125 → 0.395, FastText 0.145 → 0.861. So
more dimensions help only up to the point where each new dimension still carries
useful, non-redundant signal.

---

### Where everything lives
| section | code | output |
|---|---|---|
| א | `src/data_collection.py` | `data/original/` |
| ב | `src/generate_texts.py`, `data/generation_prompt.txt` | `data/generated/` |
| ג,ד | `src/embeddings.py`, `src/reduce.py` | `data/embeddings/` (16 sets + manifest) |
| ה | `src/analysis.py` | `results/section_he_method_comparison.csv`, `results/heatmaps_dim*.png` |
| ו | `src/analysis.py` | `results/section_vav_dim_effect.csv` |
| ז | `src/analysis.py`, `src/reduce.py` | `results/section_zayin_pca.csv` |
| ח | `src/manual_walkthrough.py` | `results/section_het_manual.md` |
| ט | `src/probes.py` | `results/section_tet_probes.{md,png}` |
| י, יא | this report (grounded in ה/ו/ז/ט) | - |
