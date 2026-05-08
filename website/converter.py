"""
converter.py
Image upload → satellite mosaic conversion.
"""

from __future__ import annotations

import base64
import io
import os

from PIL import Image, ImageOps
import random
import numpy as np
from qdrant_client.models import Prefetch, FusionQuery, Fusion, Filter, HasIdCondition

from config import DATA_DIR, QDRANT_COLLECTION, EMBEDDING_METHOD
from embedder import compute_features
from state import app_state

MAX_RESOLUTION = 2000


def process_uploaded_image(
    image_bytes: bytes,
    pixels_per_patch: int = 100,
    output_scale: int = 1,
) -> dict:
    """
    Process an uploaded image into a satellite mosaic.

    1. Decode & center-crop to square (capped at MAX_RESOLUTION)
    2. Slice into grid based on pixels_per_patch
    3. Compute feature vectors for each patch
    4. Batch-search Qdrant for closest satellite matches
    5. Assemble output image from matched satellite patches

    Returns dict with:
        image_b64:  base64-encoded PNG of the mosaic
        grid_rows:  number of rows
        grid_cols:  number of columns
    """
    # Clamp settings
    pixels_per_patch = max(25, min(200, pixels_per_patch))
    output_scale = max(1, min(3, output_scale))

    # Decode image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Center-crop to square
    side = min(img.size)
    img = ImageOps.fit(img, (side, side), centering=(0.5, 0.5))

    # Cap resolution
    if side > MAX_RESOLUTION:
        img = img.resize((MAX_RESOLUTION, MAX_RESOLUTION), Image.LANCZOS)
        side = MAX_RESOLUTION

    # Determine grid
    n = side // pixels_per_patch
    if n < 1:
        n = 1

    # Resize to exact multiple
    exact_side = n * pixels_per_patch
    if exact_side != side:
        img = img.resize((exact_side, exact_side), Image.LANCZOS)

    # Slice into patches
    patches = []
    for r in range(n):
        for c in range(n):
            left = c * pixels_per_patch
            upper = r * pixels_per_patch
            right = left + pixels_per_patch
            lower = upper + pixels_per_patch
            patches.append(img.crop((left, upper, right, lower)))

    print(f"[Convert] Sliced into {len(patches)} patches ({n}×{n}). "
          f"Computing {EMBEDDING_METHOD.upper()}...", flush=True)
    vectors = compute_features(patches)

    print(f"[Convert] Computed {len(vectors)} vectors. Batch searching Qdrant...",
          flush=True)

    # Pair each vector with its original index, then shuffle
    # This allows us to search in random order (to distribute diversity)
    # while keeping track of where each result belongs in the grid.
    indexed_vectors = list(enumerate(vectors if EMBEDDING_METHOD == "hog_hybrid" else vectors.tolist()))
    random.shuffle(indexed_vectors)

    excluded_ids = []
    results_by_index = {}  # original index → point

    for original_idx, vec in indexed_vectors:
        exclusion_filter = Filter(
            must_not=[HasIdCondition(has_id=excluded_ids)]
        ) if excluded_ids else None

        if EMBEDDING_METHOD == "hog_hybrid":
            if np.linalg.norm(vec["hog"]) < 50.0:
                result = app_state.qdrant.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=vec["colour"],
                    using="colour",
                    limit=1,
                    with_payload=True,
                    query_filter=exclusion_filter,
                )
            else:
                result = app_state.qdrant.query_points(
                    collection_name=QDRANT_COLLECTION,
                    prefetch=[
                        Prefetch(query=vec["hog"],    using="hog",    limit=1000),
                        Prefetch(query=vec["colour"], using="colour", limit=1000),
                    ],
                    query=FusionQuery(fusion=Fusion.DBSF),
                    limit=1,
                    with_payload=True,
                    query_filter=exclusion_filter,
                )
        else:
            result = app_state.qdrant.query_points(
                collection_name=QDRANT_COLLECTION,
                query=vec,
                limit=1,
                with_payload=True,
                query_filter=exclusion_filter,
            )

        if result.points:
            point = result.points[0]
            excluded_ids.append(point.id)
            results_by_index[original_idx] = point
        else:
            results_by_index[original_idx] = None  # exhausted the collection

    # Restore original ordering
    final_results = [results_by_index[i] for i in range(len(vectors))]

    print(f"[Convert] Search complete. Assembling output image...", flush=True)

    # Assemble output
    out_patch = pixels_per_patch * output_scale
    output = Image.new("RGB", (n * out_patch, n * out_patch))
    images_dir = os.path.join(DATA_DIR, "images")

    idx = 0
    for r in range(n):
        for c in range(n):
            point = final_results[idx]
            if point:
                filename = point.payload.get("filename", "")
                if filename:
                    img_path = os.path.join(images_dir, filename)
                    try:
                        sat_img = Image.open(img_path).convert("RGB")
                        sat_img = sat_img.resize((out_patch, out_patch), Image.LANCZOS)
                        output.paste(sat_img, (c * out_patch, r * out_patch))
                    except Exception as e:
                        print(f"[Convert] Failed to load {filename}: {e}", flush=True)
            idx += 1

    # Encode to PNG
    buf = io.BytesIO()
    output.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    print(f"[Convert] Done. Output: {output.size[0]}×{output.size[1]}px", flush=True)

    return {
        "image_b64": b64,
        "grid_rows": n,
        "grid_cols": n,
    }
