"""
camera.py
Real-time camera mosaic backend logic.
"""

from __future__ import annotations

import base64
import io

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

    if EMBEDDING_METHOD == "hog_hybrid":
        # Fusion query: search both HOG and colour vector spaces, combine with RRF
        # If the HOG vector is very small (e.g. flat patch with just noise, std < 5.0), fallback to colour only
        search_requests = []
        for vec in vectors:
            import numpy as np
            if np.linalg.norm(vec["hog"]) < 50.0:
                search_requests.append(
                    QueryRequest(
                        query=vec["colour"],
                        using="colour",
                        limit=1,
                        with_payload=True
                    )
                )
            else:
                search_requests.append(
                    QueryRequest(
                        prefetch=[
                            Prefetch(query=vec["hog"],    using="hog",    limit=500),
                            Prefetch(query=vec["colour"], using="colour", limit=500),
                        ],
                        query=FusionQuery(fusion=Fusion.RRF),
                        limit=1,
                        with_payload=True,
                    )
                )
    else:
        # Single-vector query (hog or dinov2)
        search_requests = [
            QueryRequest(query=vec, limit=1, with_payload=True)
            for vec in vectors.tolist()
        ]

    batch_results = app_state.qdrant.query_batch_points(
        collection_name=QDRANT_COLLECTION,
        requests=search_requests,
    )

    print(f"[Camera] Batch search complete. Building URL grid...", flush=True)

    # Build the 8×8 grid of image URLs directly from Qdrant payloads.
    # Each payload contains "filename" — no enrichment step needed.
    grid_urls: list[list[str]] = []
    idx = 0
    for r in range(8):
        row_urls: list[str] = []
        for c in range(8):
            points = batch_results[idx].points
            if points:
                filename = points[0].payload.get("filename", "")
                row_urls.append(f"/images/{filename}" if filename else "")
            else:
                row_urls.append("")
            idx += 1
        grid_urls.append(row_urls)

    return grid_urls