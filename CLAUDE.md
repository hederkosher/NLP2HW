# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

University NLP assignment ("Homework 2: Embeddings"), now **implemented end to
end** in Python under `src/`. Spec: `exercise2.pdf` (Hebrew).

Deliverables:
- **`notebooks/HW2_Embeddings.ipynb`** - the graded submission (executed, with
  embedded outputs), organized by the Hebrew section letters א–יא.
- **`REPORT.md`** - the same narrative in Markdown; per-section outputs in `results/`.

Concrete choices already made (don't re-ask): genre = **English Wikipedia**;
5 topics = photosynthesis, cellular_respiration, jazz, blues, mount_everest
(two same-domain pairs + an outlier); the ה research questions are "sensitive to
shared words" and "semantic similarity without lexical overlap". All such
constants live in `src/config.py`.

## Commands

The base anaconda env is broken (numpy 2.x vs packages built for 1.x), so the
project uses an **isolated venv** with pinned versions. Always use `.venv`:

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m src.run_all        # full pipeline: sections א→ט, regenerates results/

# Submission notebook (built from src/, then executed to embed outputs):
.venv/bin/python -m src.build_notebook
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 notebooks/HW2_Embeddings.ipynb
# Static HTML export (for submission/sharing without a kernel):
.venv/bin/jupyter nbconvert --to html notebooks/HW2_Embeddings.ipynb
```

The heavy embedding artifacts (`artifacts.pkl`, the FastText `.model` and its
~2.2 GB `.wv.vectors_ngrams.npy`) are **gitignored and regenerable** via
`python -m src.embeddings`; only the 16 small `.npz` sets and `manifest.json` are
committed. After a fresh clone, run the pipeline (or the notebook) to rebuild them.

`notebooks/HW2_Embeddings.ipynb` is the graded deliverable; it is a thin narrative
that **calls the `src/` functions** (never duplicates logic), so editing a metric
in `src/` and re-running `build_notebook` keeps the notebook in sync.

Each phase is independently runnable (and that's the fastest way to iterate):
`python -m src.data_collection` (א, network), `src.generate_texts` (ב, validates),
`src.embeddings` (ג+ד, ~1 min: trains FastText + encodes MiniLM), `src.analysis`
(ה/ו/ז), `src.manual_walkthrough` (ח), `src.probes` (ט). Run modules with
`-m src.<name>` from the repo root (package imports assume this).

## Architecture (big picture)

The pipeline turns 10 texts into 16 embedding sets, then compares them. Data flow:

- `src/config.py` - single source of truth: the 5 topics, `CLOSE_PAIRS`/`OUTLIER`,
  methods/dims/sources, MiniLM model name, probe sentences. Change experiments here.
- `src/corpus.py` - fixed document order (5 original then 5 generated, topic-aligned)
  and the **sentence pool** (`load_sentence_pool`) drawn from the FULL articles.
  That large pool is what makes PCA to 30/300 dims possible - see below.
- `src/embeddings.py` → `data/embeddings/artifacts.pkl` (+ 16 `.npz` + manifest).
  Each method is fit **jointly on all 10 docs** so originals/generated share one
  space; the "2 sources" axis is just a partition. `atoms300[method]` (sentence/
  word vectors in 300-d) are persisted so `src.analysis` can fit a well-conditioned
  PCA(300→30) for section ז.
- `src/reduce.py` - the reusable PCA, **always fit on the pool, never on the 10
  docs** (10 samples → ≤9 components). Used for MiniLM sizing (ד) and for ז.
- `src/analysis.py` - sections ה/ו/ז metrics (lexical-overlap correlation,
  cross-source same-topic cosine, topic-separation score) + heatmaps.
- `src/probes.py`, `src/manual_walkthrough.py` - sections ט and ח (self-contained
  Markdown in `results/`, cross-checked against scikit-learn).

### Non-obvious gotchas
- **Wikipedia fetching** uses the MediaWiki action API directly with a real
  User-Agent (`src/data_collection.py`); the legacy `wikipedia` pip package now
  gets empty responses without one.
- **FastText** is trained on this tiny corpus, so its document cosines are
  globally inflated (~0.9 for everything) - it barely discriminates. This is a
  documented finding, not a bug; don't "fix" it by reading it as a strong method.
- Determinism: FastText uses `seed=0, workers=1`; reruns reproduce identical
  numbers, so the tables in `REPORT.md` are stable.

## The assignment (what the deliverable must do)

The goal is a **comparative study of text-embedding methods**. The core pipeline:

1. **Corpus (10 texts total).** Pick ONE text genre (WhatsApp/social, Wikipedia,
   speeches, news e.g. YNET, or short story). Collect **5 human-written texts**
   (≥300 words each), then use an LLM (e.g. Claude) to generate **5 more texts**
   of the same genre, prompting for **as limited/restricted a vocabulary as
   possible** (אוצר מילים מצומצם - this constraint is graded). Keep the two sets
   (human vs. AI-generated) distinguishable - comparisons depend on this split.

2. **Embed every text with 4 methods:**
   - Bag of Words
   - TF-IDF
   - Word2Vec / FastText / GloVe (pretrained, or a small model trained with `gensim`)
   - A sentence transformer: `sentence-transformers/all-MiniLM-L6-v2` / BERT /
     MiniLM / DistilBERT

3. **Two embedding dimensions: 30 and 300.** Combined with 4 methods × 2 text
   sources (human/AI), the spec frames this as **16 embedding configurations** -
   keep results organized along these three axes (method × dimension × source).

4. **Research questions (ה–יא).** Pick and investigate research questions such as:
   which method best identifies *similar* texts, best *separates* different texts,
   is most sensitive to *shared words*, or best captures *semantic similarity
   without lexical overlap*. Report results in tables.

5. **Dimensionality reduction.** Reduce the 300-dim embeddings to ~30 with **PCA**
   and compare against the natively-30-dim embeddings.

6. **Worked examples.** Manually compute and show the steps for BOW vectors,
   TF-IDF vectors, and **cosine similarity** for chosen sentence pairs. The spec
   provides two illustrative sentence groups to reason about:
   - Similar meaning, different words (doctor/physician/medical care).
   - Same words, different meaning ("bank": loan / fisherman / river bank).
   These probe where lexical methods (BOW/TF-IDF) diverge from semantic ones.

## Working notes

- The assignment text is **Hebrew**; section labels use Hebrew letters
  (א, ב, ג … יא). When asked about "section ז" etc., map back to the list above.
- The intellectual core is the **comparison and analysis**, not just generating
  embeddings - keep cosine-similarity comparisons, tables, and written
  conclusions as first-class outputs alongside any code.
