"""Reusable PCA reduction (used both for transformer sizing in סעיף ד and for
the explicit 300->30 reduction experiment in סעיף ז).

The basis is ALWAYS fit on a large `pool` of vectors (sentences / words), never on
the 10 document vectors alone -- with only 10 samples PCA could yield at most 9
components, so fitting on the pool is what makes genuine 30/300-dim bases possible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class FittedPCA:
    pca: PCA
    n_components: int
    n_pool: int
    explained: float  # cumulative explained variance ratio

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self.pca.transform(X)
        # If the pool couldn't supply n_components, pad with zeros to keep the
        # requested width (documented limitation, flagged via .padded).
        if Z.shape[1] < self.n_components:
            pad = np.zeros((Z.shape[0], self.n_components - Z.shape[1]))
            Z = np.hstack([Z, pad])
        return Z

    @property
    def padded(self) -> bool:
        return self.pca.n_components_ < self.n_components


def fit_pca(pool: np.ndarray, n_components: int, seed: int = 0) -> FittedPCA:
    """Fit PCA to reduce vectors to `n_components`, on the large pool."""
    pool = np.asarray(pool, dtype=np.float64)
    max_comp = min(n_components, pool.shape[0], pool.shape[1])
    pca = PCA(n_components=max_comp, random_state=seed)
    pca.fit(pool)
    return FittedPCA(
        pca=pca,
        n_components=n_components,
        n_pool=pool.shape[0],
        explained=float(pca.explained_variance_ratio_.sum()),
    )
