"""Phase 3 (סעיפים ג + ד): build the 16 embedding sets.

4 methods x 2 sizes (30, 300) x 2 sources (original / generated) = 16 sets.

Methodology (documented; this is the ambiguity סעיף ד asks us to resolve):
  * Each method is FIT JOINTLY on all 10 documents so originals and generated
    texts live in the SAME space and are directly comparable; the "2 sources"
    axis is a partition of the 10 document vectors, sliced out for the 16 sets.
  * Sizing to 30 / 300:
      - BoW / TF-IDF : max_features = 30 / 300 (top-N terms).
      - FastText     : gensim vector_size = 30 / 300 (trained on the sentence pool).
      - MiniLM (384d): PCA down to 30 / 300, basis fit on the large sentence pool.
  * Document vector = the natural unit per method:
      - BoW / TF-IDF : the document's row vector.
      - FastText     : mean of its word vectors (mean pooling).
      - MiniLM       : mean of its sentence embeddings, then PCA.
  * `atoms300` per method (sentence/word vectors in the 300-dim space) are kept so
    section ז can fit a well-conditioned PCA(300->30) basis on many samples.

Run:  python -m src.embeddings   ->  writes data/embeddings/artifacts.pkl,
      16 per-set .npz files, and manifest.json.
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.config import DIMS, EMB_DIR, SOURCES
from src.corpus import load_documents, load_sentence_pool
from src.reduce import fit_pca
from src.text_utils import split_sentences, tokenize_words

ARTIFACTS_PATH = EMB_DIR / "artifacts.pkl"
MANIFEST_PATH = EMB_DIR / "manifest.json"


def _pad(mat: np.ndarray, dim: int) -> np.ndarray:
    """Right-pad columns with zeros so width == dim (if vocab < dim)."""
    if mat.shape[1] >= dim:
        return mat[:, :dim]
    return np.hstack([mat, np.zeros((mat.shape[0], dim - mat.shape[1]))])


def _build_count_based(docs, pool, dim, tfidf: bool):
    """BoW or TF-IDF: doc vectors (10 x dim) + 300-dim sentence atom pool."""
    texts = [d.text for d in docs]
    Vec = TfidfVectorizer if tfidf else CountVectorizer
    vec = Vec(max_features=dim, token_pattern=r"[a-z]+(?:'[a-z]+)?", lowercase=True)
    doc_vecs = _pad(vec.fit_transform(texts).toarray().astype(np.float64), dim)
    atoms = None
    if dim == 300:  # atoms only needed in the 300-dim space (for section ז)
        atoms = _pad(vec.transform(pool).toarray().astype(np.float64), dim)
    return doc_vecs, atoms, len(vec.get_feature_names_out())


def _build_fasttext(docs, pool, dim):
    """Train FastText at vector_size=dim; doc vector = mean of token vectors."""
    from gensim.models import FastText  # local import keeps startup light

    tok_sents = [tokenize_words(s) for s in pool]
    tok_sents = [t for t in tok_sents if t]
    model = FastText(
        sentences=tok_sents,
        vector_size=dim,
        window=5,
        min_count=1,
        sg=1,            # skip-gram: better on small corpora
        epochs=30,
        workers=1,
        seed=0,
    )
    doc_vecs = np.zeros((len(docs), dim))
    for i, d in enumerate(docs):
        toks = tokenize_words(d.text)
        vecs = [model.wv[t] for t in toks]  # FastText handles OOV via subwords
        doc_vecs[i] = np.mean(vecs, axis=0) if vecs else np.zeros(dim)
    atoms = None
    if dim == 300:
        atoms = np.array([model.wv[w] for w in model.wv.index_to_key], dtype=np.float64)
        model.save(str(EMB_DIR / "fasttext_300.model"))  # reused by probe analysis (ט)
    return doc_vecs, atoms


def _encode_minilm(pool, docs):
    """Encode the sentence pool and each doc's sentences with MiniLM (384-d)."""
    from sentence_transformers import SentenceTransformer

    from src.config import ST_MODEL_NAME

    model = SentenceTransformer(ST_MODEL_NAME)
    pool_emb = model.encode(pool, batch_size=64, show_progress_bar=False,
                            normalize_embeddings=False)
    doc384 = np.zeros((len(docs), pool_emb.shape[1]))
    for i, d in enumerate(docs):
        sents = split_sentences(d.text) or [d.text]
        emb = model.encode(sents, show_progress_bar=False)
        doc384[i] = emb.mean(axis=0)  # mean-pool sentences -> document vector
    return np.asarray(pool_emb, dtype=np.float64), doc384


def build_artifacts() -> dict:
    docs = load_documents()
    pool = load_sentence_pool()
    print(f"corpus: {len(docs)} docs | PCA sentence pool: {len(pool)} sentences\n")

    doc: dict = {m: {} for m in ("bow", "tfidf", "fasttext", "minilm")}
    atoms300: dict = {}
    meta: dict = {"pca": {}, "vocab": {}}

    # --- BoW / TF-IDF ---
    for tfidf, name in ((False, "bow"), (True, "tfidf")):
        for dim in DIMS:
            dv, atoms, vocab = _build_count_based(docs, pool, dim, tfidf)
            doc[name][dim] = dv
            meta["vocab"][f"{name}_{dim}"] = vocab
            if atoms is not None:
                atoms300[name] = atoms
            print(f"  {name:9s} dim={dim:3d}  docs={dv.shape}  (vocab kept={vocab})")

    # --- FastText ---
    for dim in DIMS:
        dv, atoms = _build_fasttext(docs, pool, dim)
        doc["fasttext"][dim] = dv
        if atoms is not None:
            atoms300["fasttext"] = atoms
        print(f"  fasttext  dim={dim:3d}  docs={dv.shape}")

    # --- MiniLM (encode once, PCA per dim) ---
    pool_emb, doc384 = _encode_minilm(pool, docs)
    print(f"  minilm    encoded {pool_emb.shape[0]} pool sents (native {pool_emb.shape[1]}d)")
    for dim in DIMS:
        fp = fit_pca(pool_emb, dim)
        doc["minilm"][dim] = fp.transform(doc384)
        meta["pca"][f"minilm_{dim}"] = {
            "explained_var": round(fp.explained, 4),
            "padded": fp.padded,
            "n_pool": fp.n_pool,
        }
        if dim == 300:
            atoms300["minilm"] = fp.transform(pool_emb)
        print(f"  minilm    dim={dim:3d}  docs={doc['minilm'][dim].shape}"
              f"  explained_var={fp.explained:.3f}")

    labels = [
        {"idx": d.idx, "source": d.source, "slug": d.slug,
         "domain": d.domain, "topic_idx": d.topic_idx, "label": d.label}
        for d in docs
    ]
    return {"labels": labels, "doc": doc, "atoms300": atoms300, "meta": meta}


def save(artifacts: dict) -> None:
    with open(ARTIFACTS_PATH, "wb") as f:
        pickle.dump(artifacts, f)

    labels = artifacts["labels"]
    src_idx = {s: [l["idx"] for l in labels if l["source"] == s] for s in SOURCES}
    manifest = []
    for method, dims in artifacts["doc"].items():
        for dim, mat in dims.items():
            for source in SOURCES:
                rows = src_idx[source]
                sub = mat[rows]
                fname = f"{source}_{method}_{dim}.npz"
                np.savez(
                    EMB_DIR / fname,
                    vectors=sub,
                    labels=np.array([labels[i]["label"] for i in rows]),
                    slugs=np.array([labels[i]["slug"] for i in rows]),
                )
                manifest.append({
                    "source": source, "method": method, "dim": dim,
                    "shape": list(sub.shape), "file": fname,
                })
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved artifacts.pkl, {len(manifest)} .npz sets, and manifest.json")


def main() -> int:
    save(build_artifacts())
    return 0


if __name__ == "__main__":
    sys.exit(main())
