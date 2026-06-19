"""Phase 6 (section 9): hand-crafted probe sentences that expose method differences.

Group A: similar MEANING, different WORDS   -> tests semantic similarity.
Group B: similar WORDS, different MEANING    -> tests lexical-overlap confusion.

For each of the 4 methods we embed the 6 sentences and look at the 6x6 cosine
matrix, then answer the section-9 questions. Results -> results/section_tet_*.

Run:  python -m src.probes
"""
from __future__ import annotations

import sys
from itertools import combinations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (EMB_DIR, PROBE_SIMILAR_MEANING, PROBE_SIMILAR_WORDS,
                        RESULTS_DIR)
from src.text_utils import tokenize_words

SENTS = PROBE_SIMILAR_MEANING + PROBE_SIMILAR_WORDS   # 0,1,2 = A ; 3,4,5 = B
LABELS = [f"A{i+1}" for i in range(3)] + [f"B{i+1}" for i in range(3)]
GROUP_A, GROUP_B = [0, 1, 2], [3, 4, 5]


def embed_all() -> dict[str, np.ndarray]:
    """Return {method: 6xD embedding matrix} for the 6 probe sentences."""
    out: dict[str, np.ndarray] = {}
    pat = r"[a-z]+(?:'[a-z]+)?"
    out["bow"] = CountVectorizer(token_pattern=pat).fit_transform(SENTS).toarray().astype(float)
    out["tfidf"] = TfidfVectorizer(token_pattern=pat).fit_transform(SENTS).toarray()

    from gensim.models import FastText
    ft = FastText.load(str(EMB_DIR / "fasttext_300.model"))
    ftm = np.zeros((6, ft.wv.vector_size))
    for i, s in enumerate(SENTS):
        toks = tokenize_words(s)
        ftm[i] = np.mean([ft.wv[t] for t in toks], axis=0) if toks else 0
    out["fasttext"] = ftm

    from sentence_transformers import SentenceTransformer

    from src.config import ST_MODEL_NAME
    st = SentenceTransformer(ST_MODEL_NAME)
    out["minilm"] = st.encode(SENTS, show_progress_bar=False).astype(float)
    return out


def within(C, idx):
    return float(np.mean([C[i, j] for i, j in combinations(idx, 2)]))


def main() -> int:
    emb = embed_all()
    methods = ["bow", "tfidf", "fasttext", "minilm"]
    cosmats = {m: cosine_similarity(emb[m]) for m in methods}

    # per-method within-group means
    summary = []
    for m in methods:
        C = cosmats[m]
        summary.append({
            "method": m,
            "within_A_meaning": round(within(C, GROUP_A), 3),
            "within_B_words": round(within(C, GROUP_B), 3),
            "A_minus_B": round(within(C, GROUP_A) - within(C, GROUP_B), 3),
        })

    # Where do SEMANTIC (MiniLM) and LEXICAL (mean of BoW/TF-IDF) views disagree
    # most? Positive = MiniLM sees similarity the lexical methods miss
    # (same meaning, different words); negative = lexical methods see similarity
    # MiniLM rejects (shared words, different meaning).
    pairs = list(combinations(range(6), 2))
    gaps = []
    for (i, j) in pairs:
        vals = {m: cosmats[m][i, j] for m in methods}
        lex_mean = 0.5 * (vals["bow"] + vals["tfidf"])
        disagree = vals["minilm"] - lex_mean
        gaps.append((f"{LABELS[i]}~{LABELS[j]}", disagree, vals))
    gaps.sort(key=lambda x: -abs(x[1]))

    # ---- render markdown ---------------------------------------------------
    L = ["# Section 9 - Probe Sentences (method-difference stress test)\n"]
    L.append("**Group A - similar meaning, different words:**")
    for i in GROUP_A:
        L.append(f"- {LABELS[i]}: \"{SENTS[i]}\"")
    L.append("\n**Group B - similar words, different meaning:**")
    for i in GROUP_B:
        L.append(f"- {LABELS[i]}: \"{SENTS[i]}\"")

    L.append("\n## Within-group mean cosine by method\n")
    L.append("| method | within-A (meaning) | within-B (shared words) | A − B |")
    L.append("| --- | --- | --- | --- |")
    for r in summary:
        L.append(f"| {r['method']} | {r['within_A_meaning']} | "
                 f"{r['within_B_words']} | {r['A_minus_B']} |")

    best_sem = max(summary, key=lambda r: r["A_minus_B"])
    # "Most confused by lexical overlap" = lexical overlap beats meaning the most,
    # i.e. the most NEGATIVE A−B (rates shared-word sentences higher than
    # same-meaning ones). FastText is excluded from this pick because it inflates
    # ALL cosines (see caveat), which is a different artifact.
    most_lex = min((r for r in summary if r["method"] != "fasttext"),
                   key=lambda r: r["A_minus_B"])
    L.append("\n## Answers\n")
    L.append(f"- **Best at semantic similarity** (high A, and the only positive A−B): "
             f"**{best_sem['method']}** "
             f"(within-A={best_sem['within_A_meaning']}, A−B={best_sem['A_minus_B']}). "
             "It is the only method that rates the different-words/same-meaning "
             "Group A as MORE similar than the shared-word Group B.")
    L.append(f"- **Most confused by lexical overlap**: **{most_lex['method']}** "
             f"(A−B={most_lex['A_minus_B']}, within-B={most_lex['within_B_words']} > "
             f"within-A={most_lex['within_A_meaning']}). The shared token \"bank\" "
             "makes unrelated Group B sentences look MORE similar than the "
             "genuinely-related Group A sentences.")
    L.append("- **Caveat - FastText**: its cosines are uniformly high "
             f"(within-A={summary[2]['within_A_meaning']}, "
             f"within-B={summary[2]['within_B_words']}); mean-pooled subword "
             "vectors push every short sentence into a similar direction, so it "
             "barely discriminates either way.")
    top = gaps[0]
    direction = ("MiniLM sees meaning the lexical methods miss"
                 if top[1] > 0 else
                 "the lexical methods see shared words but MiniLM rejects the match")
    L.append(f"- **Largest semantic-vs-lexical gap** is for pair **{top[0]}** "
             f"(MiniLM − lexical = {top[1]:+.2f}): "
             + ", ".join(f"{m}={top[2][m]:.2f}" for m in methods) + ".")
    L.append(f"  - Why: for {top[0]}, {direction}. This is exactly where word "
             "overlap and meaning point in opposite directions, so lexical and "
             "semantic encoders disagree the most.")

    # per-method 6x6 tables
    L.append("\n## Full 6×6 cosine matrices\n")
    for m in methods:
        L.append(f"### {m}\n")
        L.append("| | " + " | ".join(LABELS) + " |")
        L.append("| --- |" + " --- |" * 6)
        for i in range(6):
            L.append(f"| {LABELS[i]} | " +
                     " | ".join(f"{cosmats[m][i, j]:.2f}" for j in range(6)) + " |")
        L.append("")

    (RESULTS_DIR / "section_tet_probes.md").write_text("\n".join(L), encoding="utf-8")

    # ---- figure ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, m in zip(axes, methods):
        C = cosmats[m]
        im = ax.imshow(C, vmin=-0.2, vmax=1.0, cmap="magma")
        ax.set_title(m)
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(LABELS); ax.set_yticklabels(LABELS)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                        color="white" if C[i, j] < 0.6 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Probe sentences: A=same meaning/diff words, B=same words/diff meaning")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "section_tet_probes.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print("=== Probe within-group cosines ===")
    for r in summary:
        print(r)
    print("Largest semantic-vs-lexical gap:", top[0],
          {m: round(top[2][m], 2) for m in methods})
    print("wrote results/section_tet_probes.md and .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
