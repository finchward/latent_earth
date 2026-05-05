"""
pca_engine.py
PCA fitting over the database embeddings.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from config import N_COMPONENTS
from state import app_state, log


def fit_pca(db_vectors: np.ndarray) -> None:
    """
    Fit PCA on db_vectors and store components + eigenvalues in app_state.
    Prints a variance breakdown table to stdout.
    """
    log(f"  Fitting PCA ({N_COMPONENTS} components) on {len(db_vectors)} vectors…")
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    pca.fit(db_vectors)

    evr = pca.explained_variance_ratio_
    print()
    print("  ┌─ Principal Component Variance ─────────────────────────┐")
    for i, ev in enumerate(evr):
        bar = "█" * int(ev * 50)
        print(f"  │  PC{i+1}  {ev*100:5.2f}%  {bar}")
    print(f"  │  {'─' * 53}")
    print(f"  │  Total {evr.sum()*100:.2f}% of variance captured")
    print("  └────────────────────────────────────────────────────────┘")
    print()

    app_state.pca_components = pca.components_.astype(np.float32)
    app_state.eigenvalues    = evr.tolist()
