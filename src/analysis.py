"""Phase 4 (sections 5, 6, 7): comparative experiments over the 16 embedding sets.

Chosen section-5 research questions (2 of 4):
  Q1  Which method is most SENSITIVE TO SHARED/OVERLAPPING WORDS?
  Q2  Which method captures SEMANTIC SIMILARITY WITHOUT literal word overlap?

Experimental design
-------------------
We use the joint 10-document embedding spaces plus two lexical-overlap signals:
  * lex_corr  = Pearson r between cosine similarity and Jaccard WORD overlap over
                all 45 document pairs. HIGH r  => similarity is driven by shared
                words (answers Q1).
  * xtopic_cos = mean cosine of the 5 same-topic ORIGINAL<->GENERATED pairs.
                These pairs share MEANING but (by construction, restricted vocab)
                have LOW word overlap, so HIGH cosine here => the method sees
                meaning beyond words (answers Q2). We report their mean Jaccard
                too, to show the overlap really is low.

section 6 : quality of results at 30 vs 300 dims, via a topic-SEPARATION score.
section 7 : reduce 300-dim -> 30 with PCA (basis fit on each method's atom pool) and
         re-run the same metrics; compare native-30 vs PCA-30.

Run:  python -m src.analysis   -> writes results/*.csv and results/*.png
"""
from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import CLOSE_PAIRS, EMB_DIR, METHODS, OUTLIER_IDX, RESULTS_DIR
from src.corpus import load_documents
from src.reduce import fit_pca
from src.text_utils import tokenize_words

ARTIFACTS_PATH = EMB_DIR / "artifacts.pkl"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_artifacts() -> dict:
    with open(ARTIFACTS_PATH, "rb") as f:
        return pickle.load(f)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def doc_token_sets(docs) -> list[set]:
    return [set(tokenize_words(d.text)) for d in docs]


def cos(mat: np.ndarray) -> np.ndarray:
    return cosine_similarity(mat)


def lex_corr_and_xtopic(mat: np.ndarray, tok_sets, labels) -> tuple[float, float, float]:
    """Return (lex_corr r, mean xtopic cosine, mean xtopic jaccard) for a 10xdim mat."""
    C = cos(mat)
    n = len(labels)
    cs, js = [], []
    for i in range(n):
        for j in range(i + 1, n):
            cs.append(C[i, j])
            js.append(jaccard(tok_sets[i], tok_sets[j]))
    cs, js = np.array(cs), np.array(js)
    r = float(np.corrcoef(cs, js)[0, 1]) if cs.std() > 1e-9 else float("nan")

    # same-topic original<->generated pairs
    by = {}
    for k, l in enumerate(labels):
        by.setdefault((l["topic_idx"],), {})[l["source"]] = k
    xcos, xjac = [], []
    for d in by.values():
        if "original" in d and "generated" in d:
            i, j = d["original"], d["generated"]
            xcos.append(C[i, j])
            xjac.append(jaccard(tok_sets[i], tok_sets[j]))
    return r, float(np.mean(xcos)), float(np.mean(xjac))


def separation(mat5: np.ndarray) -> float:
    """Topic-separation score on a single source's 5 docs:
    mean cosine of same-domain CLOSE pairs minus mean cosine of all other pairs.
    Higher = better at pulling similar topics together / pushing others apart.
    """
    C = cos(mat5)
    close = [C[i, j] for (i, j) in CLOSE_PAIRS]
    other = [C[i, j] for i in range(5) for j in range(i + 1, 5)
             if (i, j) not in CLOSE_PAIRS]
    return float(np.mean(close) - np.mean(other))


def src_rows(labels, source):
    return [l["idx"] for l in labels if l["source"] == source]


# --------------------------------------------------------------------------- #
# section 5 : method comparison
# --------------------------------------------------------------------------- #
def section_he(art, tok_sets) -> pd.DataFrame:
    labels = art["labels"]
    rows = []
    for m in METHODS:
        for dim in (30, 300):
            r, xcos, xjac = lex_corr_and_xtopic(art["doc"][m][dim], tok_sets, labels)
            rows.append({"method": m, "dim": dim,
                         "lex_corr(Q1)": round(r, 3),
                         "xtopic_cos(Q2)": round(xcos, 3),
                         "xtopic_jaccard": round(xjac, 3)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# section 6 : effect of embedding size
# --------------------------------------------------------------------------- #
def section_vav(art) -> pd.DataFrame:
    labels = art["labels"]
    rows = []
    for m in METHODS:
        rec = {"method": m}
        for source in ("original", "generated"):
            r = src_rows(labels, source)
            for dim in (30, 300):
                rec[f"sep@{dim}_{source[:4]}"] = round(separation(art["doc"][m][dim][r]), 3)
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# section 7 : PCA 300 -> 30 vs native 30
# --------------------------------------------------------------------------- #
def pca_reduce_docs(art, method) -> np.ndarray:
    """PCA-reduce a method's 10x300 doc vectors to 30, basis fit on its atom pool."""
    fp = fit_pca(art["atoms300"][method], 30)
    return fp.transform(art["doc"][method][300]), fp


def section_zayin(art, tok_sets) -> pd.DataFrame:
    labels = art["labels"]
    rows = []
    for m in METHODS:
        pca30, fp = pca_reduce_docs(art, m)
        # metrics for native-30, pca-from-300, and (reference) native-300.
        # MiniLM is natively 384-d, so its "30"/"300" sets are themselves PCA
        # projections of the 384-d space; label them as such, not "native".
        n30, n300 = ("pca384→30", "pca384→300") if m == "minilm" else ("native30", "native300")
        variants = {
            n30: art["doc"][m][30],
            "pca300to30": pca30,
            n300: art["doc"][m][300],
        }
        for name, mat in variants.items():
            r, xcos, _ = lex_corr_and_xtopic(mat, tok_sets, labels)
            orig = src_rows(labels, "original")
            rows.append({"method": m, "variant": name,
                         "lex_corr": round(r, 3),
                         "xtopic_cos": round(xcos, 3),
                         "sep_orig": round(separation(mat[orig]), 3),
                         "pca_explVar": round(fp.explained, 3) if name == "pca300to30" else ""})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# heatmaps
# --------------------------------------------------------------------------- #
def make_heatmaps(art, dim=300) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [l["label"] for l in art["labels"]]
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    for ax, m in zip(axes.ravel(), METHODS):
        C = cos(art["doc"][m][dim])
        im = ax.imshow(C, vmin=-0.2, vmax=1.0, cmap="viridis")
        ax.set_title(f"{m}  (dim={dim})  cosine similarity", fontsize=11)
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                        color="white" if C[i, j] < 0.6 else "black", fontsize=5.5)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Document cosine-similarity by method (10 docs, dim={dim})", fontsize=13)
    fig.tight_layout()
    out = RESULTS_DIR / f"heatmaps_dim{dim}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
def main() -> int:
    art = load_artifacts()
    docs = load_documents()
    tok_sets = doc_token_sets(docs)

    he = section_he(art, tok_sets)
    vav = section_vav(art)
    zay = section_zayin(art, tok_sets)

    he.to_csv(RESULTS_DIR / "section_he_method_comparison.csv", index=False)
    vav.to_csv(RESULTS_DIR / "section_vav_dim_effect.csv", index=False)
    zay.to_csv(RESULTS_DIR / "section_zayin_pca.csv", index=False)

    pd.set_option("display.width", 120)
    print("\n=== section 5  Method comparison (Q1 shared-words / Q2 semantic-no-overlap) ===")
    print(he.to_string(index=False))
    print("\n=== section 6  Effect of embedding size (topic-separation score) ===")
    print(vav.to_string(index=False))
    print("\n=== section 7  PCA 300->30 vs native-30 ===")
    print(zay.to_string(index=False))
    print("\nFigures:")
    make_heatmaps(art, dim=300)
    make_heatmaps(art, dim=30)
    print("\nCSVs written to results/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
