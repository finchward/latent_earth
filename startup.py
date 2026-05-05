"""
startup.py
Blocking startup sequence, intended to be run in the thread executor so the
FastAPI event loop stays free while models and data are loading.
"""

from __future__ import annotations

from config import N_COMPONENTS, PORT
from db import init_qdrant, scroll_all_vectors
from embedder import init_clip
from gee import init_gee
from pca_engine import fit_pca
from state import app_state, log


def startup_sync() -> None:
    """Run the full initialisation pipeline (blocking, call from executor)."""
    print()
    print("━" * 60)
    print("  Sentinel Atlas — startup")
    print("━" * 60)

    log("1/4  Connecting to Google Earth Engine…")
    init_gee()

    log("2/4  Loading DINOv2-small…")
    init_clip()

    log("3/4  Connecting to Qdrant…")
    init_qdrant()

    log("4/4  Scrolling vectors and fitting PCA…")
    vecs, ids, pays = scroll_all_vectors()
    app_state.db_vectors  = vecs
    app_state.db_ids      = ids
    app_state.db_payloads = pays
    app_state.n_vectors   = len(ids)

    if len(ids) >= N_COMPONENTS:
        fit_pca(vecs)
    else:
        log(
            f"  ⚠  Only {len(ids)} vectors in DB — need ≥{N_COMPONENTS} for PCA.\n"
            "      Seed the database first (run the notebook collection cell)."
        )

    app_state.ready      = True
    app_state.status_msg = "Ready"

    print("━" * 60)
    print(f"  ✓ Sentinel Atlas ready  —  {len(ids)} patches indexed")
    print(f"  Open: http://localhost:{PORT}")
    print("━" * 60)
    print()
