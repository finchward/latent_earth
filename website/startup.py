"""
startup.py
Blocking startup sequence, intended to be run in the thread executor so the
FastAPI event loop stays free while Qdrant is loading.
"""

from __future__ import annotations

import os

from qdrant_client import QdrantClient

from config import (
    PORT,
    QDRANT_COLLECTION,
    QDRANT_LOCAL_PATH,
    QDRANT_PREFER_SERVER,
    QDRANT_URL,
)
from state import app_state, log


def startup_sync() -> None:
    """Run the full initialisation pipeline (blocking, call from executor)."""
    print()
    print("━" * 60)
    print("  Sentinel Atlas — startup")
    print("━" * 60)

    # ── Open Qdrant ───────────────────────────────────────────────────────────
    log("1/2  Connecting to Qdrant…")

    if QDRANT_PREFER_SERVER:
        log(f"  → Server mode: {QDRANT_URL}")
        client = QdrantClient(url=QDRANT_URL)
    else:
        log(f"  → Local mode: {QDRANT_LOCAL_PATH}")
        os.makedirs(QDRANT_LOCAL_PATH, exist_ok=True)
        client = QdrantClient(path=QDRANT_LOCAL_PATH)

    existing = [c.name for c in client.get_collections().collections]

    if QDRANT_COLLECTION not in existing:
        log(f"  ⚠  Collection '{QDRANT_COLLECTION}' not found — DB is empty.")
        app_state.n_vectors = 0
    else:
        info = client.get_collection(QDRANT_COLLECTION)
        app_state.n_vectors = info.points_count
        log(f"  ✓ Qdrant: {info.points_count} patches in '{QDRANT_COLLECTION}'")

    app_state.qdrant = client

    # ── Done ──────────────────────────────────────────────────────────────────
    log("2/2  Ready.")
    app_state.ready = True
    app_state.status_msg = "Ready"

    print("━" * 60)
    print(f"  ✓ Sentinel Atlas ready  —  {app_state.n_vectors} patches indexed")
    print(f"  Open: http://localhost:{PORT}")
    print("━" * 60)
    print()
