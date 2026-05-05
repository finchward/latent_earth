"""
db.py
Qdrant connection, database sync from Google Drive, and vector scrolling.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
from qdrant_client import QdrantClient

from config import (
    MAX_PCA_VECTORS,
    QDRANT_COLLECTION,
    QDRANT_DRIVE_PATH,
    QDRANT_LOCAL_PATH,
    VECTOR_SIZE,
)
from state import app_state, log


def init_qdrant() -> None:
    """
    Copy Qdrant storage from Google Drive to a local temp path, then
    open the client and store it in app_state.
    """
    if QDRANT_DRIVE_PATH and os.path.isdir(QDRANT_DRIVE_PATH):
        log("  Copying Qdrant storage from Drive…")
        shutil.copytree(QDRANT_DRIVE_PATH, QDRANT_LOCAL_PATH, dirs_exist_ok=True)
        log(f"  ✓ Restored from {QDRANT_DRIVE_PATH}")
    else:
        log(
            f"  ⚠  Drive path not found: {QDRANT_DRIVE_PATH!r}\n"
            "      Set QDRANT_DRIVE_PATH env var, or check _DRIVE_GLOBS in config."
        )
        os.makedirs(QDRANT_LOCAL_PATH, exist_ok=True)

    # Remove stale lock that may prevent QdrantClient from opening
    lock_file = os.path.join(QDRANT_LOCAL_PATH, ".lock")
    if os.path.exists(lock_file):
        os.remove(lock_file)

    client = QdrantClient(path=QDRANT_LOCAL_PATH)
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        log(f"  ⚠  Collection '{QDRANT_COLLECTION}' not found — DB is empty.")
    else:
        info = client.get_collection(QDRANT_COLLECTION)
        log(f"  ✓ Qdrant: {info.points_count} patches in '{QDRANT_COLLECTION}'")

    app_state.qdrant = client


def scroll_all_vectors() -> tuple[np.ndarray, list, list[dict]]:
    """
    Scroll up to MAX_PCA_VECTORS points from Qdrant.
    Returns (vectors ndarray, id list, payload list).
    """
    client = app_state.qdrant
    vectors: list = []
    ids:     list = []
    payloads: list[dict] = []
    offset = None

    while len(vectors) < MAX_PCA_VECTORS:
        try:
            results, next_offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=500,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )
        except Exception as exc:
            log(f"  ⚠  Scroll error: {exc}")
            break

        if not results:
            break
        for p in results:
            vectors.append(p.vector)
            ids.append(p.id)
            payloads.append(p.payload or {})

        offset = next_offset
        if next_offset is None:
            break

    log(f"  ✓ Scrolled {len(vectors)} vectors")
    if not vectors:
        return np.empty((0, VECTOR_SIZE), dtype=np.float32), [], []
    return np.array(vectors, dtype=np.float32), ids, payloads
