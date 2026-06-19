"""Phase 5 (section 8): by-hand BoW / TF-IDF / cosine walkthrough.

Picks 3 sentences from the GENERATED data and shows, step by step:
  1. the vocabulary and the BoW count matrix (manual counting);
  2. the TF, IDF and TF*IDF tables, then L2 row-normalization (standard TF-IDF);
  3. cosine similarity between 3 WORD pairs, where each word's vector is its
     term-document COLUMN across the 3 sentences (the agreed definition), with
     the dot-product / norm arithmetic written out.

Everything is rendered to results/section_het_manual.md so a grader can follow it
without running code. The numbers are cross-checked against sklearn at the bottom.

Run:  python -m src.manual_walkthrough
"""
from __future__ import annotations

import sys
from math import log, sqrt

import numpy as np

from src.config import RESULTS_DIR
from src.text_utils import tokenize_words

# 3 sentences taken verbatim from data/generated/wiki_gen_01_photosynthesis.txt
SENTENCES = [
    "Plants make their own food.",
    "They make this food from sunlight, water, and air.",
    "The green parts of a plant, like the leaves, do most of this work.",
]
WORD_PAIRS = [("make", "food"), ("food", "this"), ("plant", "the")]

OUT = RESULTS_DIR / "section_het_manual.md"


def build_vocab(tok_docs):
    return sorted({t for toks in tok_docs for t in toks})


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def cosine_steps(name_a, vec_a, name_b, vec_b):
    """Return a markdown block showing the cosine computation step by step."""
    dot = float(np.dot(vec_a, vec_b))
    na = sqrt(float(np.dot(vec_a, vec_a)))
    nb = sqrt(float(np.dot(vec_b, vec_b)))
    cos = dot / (na * nb) if na > 0 and nb > 0 else 0.0
    va = "[" + ", ".join(f"{x:.4f}" for x in vec_a) + "]"
    vb = "[" + ", ".join(f"{x:.4f}" for x in vec_b) + "]"
    terms = " + ".join(f"({a:.4f})({b:.4f})" for a, b in zip(vec_a, vec_b))
    return (
        f"**cos('{name_a}', '{name_b}')**\n\n"
        f"- column('{name_a}') = {va}\n"
        f"- column('{name_b}') = {vb}\n"
        f"- dot = {terms} = **{dot:.4f}**\n"
        f"- ‖{name_a}‖ = √{np.dot(vec_a, vec_a):.4f} = {na:.4f},  "
        f"‖{name_b}‖ = √{np.dot(vec_b, vec_b):.4f} = {nb:.4f}\n"
        f"- cosine = {dot:.4f} / ({na:.4f} × {nb:.4f}) = **{cos:.4f}**\n"
    ), cos


def main() -> int:
    tok_docs = [tokenize_words(s) for s in SENTENCES]
    vocab = build_vocab(tok_docs)
    vi = {w: i for i, w in enumerate(vocab)}
    n_docs = len(SENTENCES)

    # ---- 1. BoW counts -----------------------------------------------------
    bow = np.zeros((n_docs, len(vocab)), dtype=int)
    for d, toks in enumerate(tok_docs):
        for t in toks:
            bow[d, vi[t]] += 1

    # ---- 2. TF-IDF (sklearn-style: smooth_idf, then L2 row-normalize) ------
    tf = bow.astype(float)
    df = (bow > 0).sum(axis=0)
    idf = np.log((1 + n_docs) / (1 + df)) + 1.0          # smoothed idf
    tfidf = tf * idf
    row_norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    tfidf_norm = tfidf / np.where(row_norms == 0, 1, row_norms)

    # ---- render ------------------------------------------------------------
    L = []
    L.append("# Section 8 - Manual BoW / TF-IDF / Cosine Walkthrough\n")
    L.append("Three sentences taken verbatim from "
             "`data/generated/wiki_gen_01_photosynthesis.txt`:\n")
    for i, s in enumerate(SENTENCES, 1):
        L.append(f"- **S{i}:** \"{s}\"")
    L.append("\n**Tokenization** (lowercase, alphabetic tokens only):\n")
    for i, toks in enumerate(tok_docs, 1):
        L.append(f"- S{i} → {toks}")
    L.append(f"\n**Vocabulary** (sorted, {len(vocab)} terms): `{vocab}`\n")

    # BoW table
    L.append("## 1. Bag-of-Words (counting by hand)\n")
    L.append("Each cell = number of times the term occurs in the sentence.\n")
    rows = [["S" + str(d + 1)] + [int(bow[d, vi[w]]) for w in vocab] for d in range(n_docs)]
    L.append(md_table(["doc"] + vocab, rows) + "\n")

    # IDF table
    L.append("## 2. TF-IDF (step by step)\n")
    L.append("**TF** = raw count above. **DF** = #sentences containing the term. "
             f"**IDF** = ln((1+N)/(1+DF)) + 1, with N = {n_docs} "
             "(smoothed, as in scikit-learn).\n")
    rows = [[w, int(df[vi[w]]),
             f"ln((1+{n_docs})/(1+{int(df[vi[w]])}))+1 = {idf[vi[w]]:.4f}"]
            for w in vocab]
    L.append(md_table(["term", "DF", "IDF"], rows) + "\n")

    L.append("**TF·IDF** (raw), then each row is L2-normalized "
             "(divide the row by its Euclidean norm - scikit-learn default):\n")
    rows = []
    for d in range(n_docs):
        rows.append([f"S{d+1} raw"] + [f"{tfidf[d, vi[w]]:.3f}" for w in vocab])
        rows.append([f"S{d+1} norm"] + [f"{tfidf_norm[d, vi[w]]:.3f}" for w in vocab])
    L.append(md_table(["doc"] + vocab, rows) + "\n")
    norms_str = ", ".join(f"‖S{d+1}‖={row_norms[d,0]:.4f}" for d in range(n_docs))
    L.append(f"Row norms used for normalization: {norms_str}\n")

    # ---- 3. word-pair cosines ---------------------------------------------
    L.append("## 3. Cosine similarity between word pairs\n")
    L.append("Per the chosen definition, a **word's vector is its column** across "
             "the 3 sentences. We compute cosine on (a) the BoW count columns and "
             "(b) the normalized TF-IDF columns.\n")

    summary = []
    for kind, M in (("BoW", bow.astype(float)), ("TF-IDF (L2-normalized)", tfidf_norm)):
        L.append(f"### {kind}\n")
        for wa, wb in WORD_PAIRS:
            ca, cb = M[:, vi[wa]], M[:, vi[wb]]
            block, c = cosine_steps(wa, ca, wb, cb)
            L.append(block)
            summary.append((kind, f"{wa}~{wb}", c))

    L.append("## Summary of word-pair cosines\n")
    rows = []
    for wa, wb in WORD_PAIRS:
        b = next(c for k, p, c in summary if k == "BoW" and p == f"{wa}~{wb}")
        t = next(c for k, p, c in summary if k.startswith("TF-IDF") and p == f"{wa}~{wb}")
        rows.append([f"{wa} ~ {wb}", f"{b:.4f}", f"{t:.4f}"])
    L.append(md_table(["word pair", "BoW cosine", "TF-IDF cosine"], rows) + "\n")
    L.append("> Note: BoW weights every shared sentence equally, so columns with the "
             "same occurrence pattern have cosine 1.0. After TF-IDF's IDF weighting "
             "and per-sentence L2 normalization, the same columns are re-scaled "
             "differently per sentence, which changes the cosines - showing how "
             "TF-IDF down-weights terms that appear in many sentences.\n")

    # ---- cross-check against sklearn --------------------------------------
    from sklearn.feature_extraction.text import (CountVectorizer,
                                                  TfidfVectorizer)
    cv = CountVectorizer(vocabulary=vocab, token_pattern=r"[a-z]+(?:'[a-z]+)?")
    tv = TfidfVectorizer(vocabulary=vocab, token_pattern=r"[a-z]+(?:'[a-z]+)?")
    bow_ok = np.array_equal(cv.fit_transform([s.lower() for s in SENTENCES]).toarray(), bow)
    tfidf_ok = np.allclose(tv.fit_transform([s.lower() for s in SENTENCES]).toarray(),
                           tfidf_norm, atol=1e-6)
    L.append("---\n_Cross-check: the by-hand BoW matches scikit-learn "
             f"CountVectorizer = **{bow_ok}**; the by-hand normalized TF-IDF matches "
             f"scikit-learn TfidfVectorizer = **{tfidf_ok}**._\n")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"sklearn cross-check: BoW={bow_ok}  TF-IDF={tfidf_ok}")
    return 0 if (bow_ok and tfidf_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
