"""
camera.py
Real-time camera mosaic backend logic.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image
from qdrant_client.models import QueryRequest, Prefetch, FusionQuery, Fusion

from config import QDRANT_COLLECTION, EMBEDDING_METHOD
from embedder import compute_features
from state import app_state


def process_camera_frame(b64_str: str) -> list[list[str]]:
    """
    Decode the 800×800 base64 camera frame, slice into 8×8 (100px patches),
    compute vectors based on configuration, and batch-search Qdrant.

    Returns an 8×8 grid of image URLs (e.g. "/images/abc123.png").
    """
    raw_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    # 800 / 8 = 100
    patch_size = 100

    # Slice the image into 64 patches
    patches = []
    for r in range(8):
        for c in range(8):
            left = c * patch_size
            upper = r * patch_size
            right = left + patch_size
            lower = upper + patch_size
            patches.append(img.crop((left, upper, right, lower)))

    print(f"[Camera] Sliced into {len(patches)} patches. Computing {EMBEDDING_METHOD.upper()}...", flush=True)
    vectors = compute_features(patches)

    print(f"[Camera] Computed {len(vectors)} {EMBEDDING_METHOD.upper()} vectors. Batch searching Qdrant...", flush=True)

    TOP_K = 25

    search_requests: list[QueryRequest] = []
    if EMBEDDING_METHOD == "hog_hybrid":
        for vec in vectors:
            if np.linalg.norm(vec["hog"]) < 50.0:
                # Flat/featureless patch — colour descriptor only
                search_requests.append(QueryRequest(
                    query=vec["colour"],
                    using="colour",
                    limit=TOP_K,
                    with_payload=True,
                ))
            else:
                search_requests.append(QueryRequest(
                    prefetch=[
                        Prefetch(query=vec["hog"],    using="hog",    limit=500),
                        Prefetch(query=vec["colour"], using="colour", limit=500),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=TOP_K,
                    with_payload=True,
                ))
    else:
        # fused_hybrid, hog, dinov2 — all return an ndarray
        for vec in vectors:
            search_requests.append(QueryRequest(
                query=vec.tolist(),
                limit=TOP_K,
                with_payload=True,
            ))

    batch_results = app_state.qdrant.query_batch_points(
        collection_name=QDRANT_COLLECTION,
        requests=search_requests,
    )

    print(f"[Camera] Batch search complete. Applying greedy uniqueness + building URL grid...", flush=True)

    # Client-side greedy: prefer the best-match that hasn't been used yet.
    # This keeps the mosaic visually diverse across the 8×8 grid.
    used_ids: set = set()
    chosen_points = []
    for query_result in batch_results:
        candidates = query_result.points
        chosen = None
        for candidate in candidates:
            if candidate.id not in used_ids:
                chosen = candidate
                break
        if chosen is None and candidates:
            chosen = candidates[0]   # duplicate rather than leave blank
        if chosen is not None:
            used_ids.add(chosen.id)
        chosen_points.append(chosen)

    # Build the 8×8 grid of image URLs directly from Qdrant payloads.
    # Each payload contains "filename" — no enrichment step needed.
    grid_urls: list[list[str]] = []
    idx = 0
    for r in range(8):
        row_urls: list[str] = []
        for c in range(8):
            point = chosen_points[idx]
            if point:
                filename = point.payload.get("filename", "")
                row_urls.append(f"/images/{filename}" if filename else "")
            else:
                row_urls.append("")
            idx += 1
        grid_urls.append(row_urls)

    return grid_urls