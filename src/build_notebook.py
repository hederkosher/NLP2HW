"""Build the submission notebook notebooks/HW2_Embeddings.ipynb.

The notebook is a thin, readable narrative organized by the Hebrew section letters
(א–יא). It REUSES the verified functions in src/ rather than duplicating logic, so
the notebook and the scripts can never drift apart. Run:

    python -m src.build_notebook        # writes the .ipynb (no outputs yet)
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
        notebooks/HW2_Embeddings.ipynb  # fills in outputs
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "HW2_Embeddings.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str):
    return nbf.v4.new_code_cell(text.strip("\n"))


def build() -> nbf.NotebookNode:
    cells = []

    cells.append(md(r"""
# Homework 2 - Embeddings

A comparative study of four text-embedding methods on English Wikipedia texts.

**Corpus:** 5 real English Wikipedia excerpts (≥300 words) + 5 Claude-generated
texts on the same 5 topics (≥300 words, deliberately **restricted vocabulary**).
**Topics:** Photosynthesis & Cellular respiration (biology pair), Jazz & Blues
(music pair), Mount Everest (outlier).
**Methods:** Bag-of-Words, TF-IDF, FastText (gensim), MiniLM (`all-MiniLM-L6-v2`).

### Methodology decisions (resolving the ambiguity in סעיף ד)
- **Sizing to 30 / 300 dims:** BoW/TF-IDF via `max_features`; FastText via gensim
  `vector_size`; MiniLM (384-d) via **PCA** to 30/300.
- **PCA bases are fit on a ~2,250-sentence pool** (full articles + generated
  texts), never on the 10 documents alone (10 samples → ≤9 components). This is
  the reusable PCA used again in סעיף ז.
- **Document vector** = BoW/TF-IDF row · FastText mean-of-word-vectors ·
  MiniLM mean-of-sentence-embeddings then PCA.
- **Joint fitting**: each method is fit once on all 10 docs (shared space); the
  "2 sources" axis is a partition → the 16 sets.
- **Metric:** cosine similarity throughout.

> This notebook reuses the functions in `src/`. The same results are produced by
> `python -m src.run_all`. A fuller write-up is in `REPORT.md`.
"""))

    cells.append(md("## Setup"))
    cells.append(code(r"""
import sys, os, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# Make `import src...` work regardless of where Jupyter started.
ROOT = Path.cwd()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
from IPython.display import Image, Markdown, display

from src import config, analysis, embeddings, manual_walkthrough, probes
from src.corpus import load_documents
from src.text_utils import word_count, tokenize_words
print("repo root:", ROOT)
"""))

    # --- א ---
    cells.append(md(r"""
## א - Data collection

5 English Wikipedia articles fetched via the MediaWiki action API
(`src/data_collection.py`), cleaned, and cut to ≥300-word excerpts. Each saved
text is the current-revision article lead.

**Sources (English Wikipedia):**
- Photosynthesis - https://en.wikipedia.org/wiki/Photosynthesis
- Cellular respiration - https://en.wikipedia.org/wiki/Cellular_respiration
- Jazz - https://en.wikipedia.org/wiki/Jazz
- Blues - https://en.wikipedia.org/wiki/Blues
- Mount Everest - https://en.wikipedia.org/wiki/Mount_Everest
"""))
    cells.append(code(r"""
from src import data_collection
need = [config.ORIGINAL_DIR / f"wiki_{i:02d}_{s}.txt"
        for i, (s, _, _) in enumerate(config.TOPICS, 1)]
if not all(p.exists() for p in need):
    data_collection.main()           # fetches from Wikipedia (needs network)

docs = load_documents()
display(pd.DataFrame([{"idx": d.idx, "source": d.source, "topic": d.slug,
                       "domain": d.domain, "words": word_count(d.text)}
                      for d in docs]))
print("\\nSample excerpt (original photosynthesis):\\n")
print(docs[0].text[:400], "...")
"""))

    # --- ב ---
    cells.append(md(r"""
## ב - AI-generated texts (restricted vocabulary)

Generated with Claude using the exact prompt below, which **explicitly demands as
limited a vocabulary as possible** (the graded constraint). Evidence that it
worked: the generated texts have a much lower type-token ratio (unique/total
words) than the originals.
"""))
    cells.append(code(r"""
print(config.GEN_PROMPT_FILE.read_text())
"""))
    cells.append(code(r"""
def ttr(text):
    w = tokenize_words(text); return len(set(w)) / len(w)

rows = []
for i, (slug, _, _) in enumerate(config.TOPICS, 1):
    o = (config.ORIGINAL_DIR / f"wiki_{i:02d}_{slug}.txt").read_text()
    g = (config.GENERATED_DIR / f"wiki_gen_{i:02d}_{slug}.txt").read_text()
    rows.append({"topic": slug, "original_TTR": ttr(o), "generated_TTR": ttr(g)})
pd.DataFrame(rows).round(3)
"""))

    # --- ג+ד ---
    cells.append(md(r"""
## ג + ד - The 16 embedding sets

4 methods × 2 sizes (30, 300) × 2 sources = **16 sets**
(`src/embeddings.py` → `data/embeddings/`). Building trains FastText and encodes
the sentence pool with MiniLM (~1 minute the first time).
"""))
    cells.append(code(r"""
if not embeddings.ARTIFACTS_PATH.exists():
    embeddings.main()
art = analysis.load_artifacts()

manifest = json.loads((config.EMB_DIR / "manifest.json").read_text())
print(f"{len(manifest)} embedding sets in manifest")
print("MiniLM PCA explained variance:", art["meta"]["pca"])
display(pd.DataFrame(manifest).pivot_table(
    index=["method", "dim"], columns="source", values="shape", aggfunc="first"))
"""))

    # --- ה ---
    cells.append(md(r"""
## ה - Method comparison (2 research questions)

**Q1 - sensitive to shared/overlapping words?**  `lex_corr` = Pearson r between
cosine similarity and Jaccard *word* overlap over all 45 document pairs (high ⇒
similarity is driven by shared words).

**Q2 - semantic similarity without literal overlap?**  `xtopic_cos` = mean cosine
of the 5 same-topic **original↔generated** pairs, which share only ~14 % of words
(`xtopic_jaccard`) yet mean the same thing (high ⇒ method sees meaning).
"""))
    cells.append(code(r"""
tok = analysis.doc_token_sets(load_documents())
he = analysis.section_he(art, tok)
display(he)
analysis.make_heatmaps(art, dim=300)
display(Image(str(config.RESULTS_DIR / "heatmaps_dim300.png")))
"""))
    cells.append(md(r"""
**Findings.**
- **Q1:** **TF-IDF at 300-dim** is the most lexical (lex_corr 0.54; its
  cross-vocabulary semantic match collapses to 0.47). BoW behaves similarly.
- **Q2:** read `xtopic_cos` *together with* topic separation (סעיף ו) because
  FastText scores ≈0.92 on *everything* and so is not discriminative. **MiniLM**
  is the method that keeps same-meaning/different-words pairs similar **while
  still separating unrelated topics** (see the clean biology/music blocks in the
  MiniLM heatmap vs the uniformly bright FastText panel). Confirmed in סעיף ט.
"""))

    # --- ו ---
    cells.append(md(r"""
## ו - Effect of embedding size

**Topic-separation score** on a source's 5 docs = mean cosine of same-domain
close pairs − mean cosine of the other pairs (higher = better). This is the
"quality" metric (it directly measures whether related texts cluster and
unrelated ones separate), used consistently here and in סעיף ז. The effect of
length on סעיף ה's own metrics (lex_corr, xtopic_cos) is also visible directly in
the ה table above, which lists every method at both dim=30 and dim=300.
"""))
    cells.append(code(r"""
display(analysis.section_vav(art))
"""))
    cells.append(md(r"""
**Finding.** Bigger is **not** uniformly better. BoW/TF-IDF improve with more
dimensions (they need vocabulary features to tell topics apart), but **MiniLM
separates topics *better* at 30-dim than 300-dim** (0.824 → 0.732): the first PCA
components hold the dominant semantic axes, extra dims add noise.
"""))

    # --- ז ---
    cells.append(md(r"""
## ז - PCA reduction (300 → 30) vs native-30

For each method, reduce the 300-dim doc vectors to 30 with PCA (basis fit on the
method's atom pool) and compare to native-30.
"""))
    cells.append(code(r"""
display(analysis.section_zayin(art, tok))
"""))
    cells.append(md(r"""
**Finding.** For lexical/word methods, **PCA-from-300 clearly beats native top-30**
at separating topics (TF-IDF 0.13→0.40, FastText 0.15→0.86): *feature extraction*
(rich linear combinations) beats *feature selection* (the 30 commonest words,
mostly stopwords). For MiniLM the two 30-dim variants are essentially identical
(both PCA of the same 384-dim space). So **how** you get 30 dims matters more than
the number 30.
"""))

    # --- ח ---
    cells.append(md(r"""
## ח - Manual BoW / TF-IDF / cosine walkthrough

Three sentences from the generated photosynthesis text, worked **by hand**:
vocabulary, count matrix, TF/IDF/TF·IDF, L2 normalization, and cosine between word
*column* vectors - each step shown, then cross-checked against scikit-learn.
"""))
    cells.append(code(r"""
manual_walkthrough.main()
display(Markdown((config.RESULTS_DIR / "section_het_manual.md").read_text()))
"""))

    # --- ט ---
    cells.append(md(r"""
## ט - Probe sentences (method-difference stress test)

**Group A** = same meaning, different words · **Group B** = same words ("bank"),
different meaning. We embed all 6 with each method and inspect the 6×6 cosines.
"""))
    cells.append(code(r"""
probes.main()
display(Image(str(config.RESULTS_DIR / "section_tet_probes.png")))
display(Markdown((config.RESULTS_DIR / "section_tet_probes.md").read_text()))
"""))

    # --- י / יא ---
    cells.append(md(r"""
## י - Why can TF-IDF fail to detect semantic similarity?

TF-IDF represents a sentence as weighted counts over a **fixed vocabulary**, each
word an independent, meaning-free dimension. Two sentences that mean the same
thing with **different words** share few/no dimensions → near-orthogonal vectors
→ low cosine. IDF makes it worse by *up-weighting rare words*, so distinctive
synonyms ("physician" vs "doctor") push the vectors apart instead of together.
**Evidence:** TF-IDF within-Group-A (same meaning, different words) is only
**0.11** (vs MiniLM 0.69), and its same-topic original↔generated cosine collapses
to **0.47** at 300-dim because the generated texts restate meaning with a
different, restricted vocabulary. TF-IDF has no notion of synonymy - it only
matches surface tokens.

## יא - Does a higher-dimensional embedding always give better results?

**No.** From סעיף ו, MiniLM's topic separation *drops* from **0.824 (30-d) to
0.732 (300-d)**: the first ~30 PCA components hold the dominant semantic axes
(51.8 % variance) and the extra 270 dims add mostly noise. **When the dimension is
too small**, information loss is catastrophic: BoW/TF-IDF at 30 dims keep only the
30 commonest words ("the, and, of, a, …"), so all documents look alike and
separation is ≈0 (BoW 0.049). Section ז reinforces it: PCA-reduced 30 dims
(information-rich) beat native top-30 dims (TF-IDF 0.13→0.40, FastText 0.15→0.86).
So more dimensions help only while each new dimension still carries useful,
non-redundant signal.
"""))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build(), OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
