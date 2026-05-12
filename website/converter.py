"""
converter.py
Image upload → satellite mosaic conversion.
"""

from __future__ import annotations

import base64
import io
import os

from PIL import Image, ImageOps
import numpy as np
from qdrant_client.models import Prefetch, FusionQuery, Fusion, QueryRequest

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

    # Cap resolution maintaining aspect ratio
    w, h = img.size
    if w > MAX_RESOLUTION or h > MAX_RESOLUTION:
        scale = MAX_RESOLUTION / max(w, h)
        w = int(w * scale)
        h = int(h * scale)
        img = img.resize((w, h), Image.LANCZOS)

    # Determine grid dimensions
    cols = max(1, w // pixels_per_patch)
    rows = max(1, h // pixels_per_patch)

    # Center-crop to exact grid multiples (the "center boxes and crop" logic)
    grid_w = cols * pixels_per_patch
    grid_h = rows * pixels_per_patch
    left = (w - grid_w) // 2
    top = (h - grid_h) // 2
    img = img.crop((left, top, left + grid_w, top + grid_h))

    # Slice into patches
    patches = []
    for r in range(rows):
        for c in range(cols):
            l = c * pixels_per_patch
            u = r * pixels_per_patch
            r_edge = l + pixels_per_patch
            b_edge = u + pixels_per_patch
            patches.append(img.crop((l, u, r_edge, b_edge)))

    print(f"[Convert] Sliced into {len(patches)} patches ({cols}×{rows}). "
          f"Computing {EMBEDDING_METHOD.upper()}...", flush=True)
    vectors = compute_features(patches)

    print(f"[Convert] Computed {len(vectors)} vectors. Batch searching Qdrant...",
          flush=True)

    TOP_K = 25
    MAX_ROUNDS = 15  # Search up to Round 4 (offset 100)

    final_results = [None] * len(vectors)
    fallbacks = [None] * len(vectors)
    used_ids: set = set()
    pending_indices = list(range(len(vectors)))

    for round_idx in range(MAX_ROUNDS):
        if not pending_indices:
            break

        offset = round_idx * TOP_K
        if round_idx > 0:
            print(f"[Convert] Round {round_idx}: Searching for {len(pending_indices)} pending patches (offset={offset})...", flush=True)

        # ── Build QueryRequests for pending patches ───────────────────────────
        search_requests: list[QueryRequest] = []
        for idx in pending_indices:
            vec = vectors[idx]
            if EMBEDDING_METHOD == "hog_hybrid":
                if np.linalg.norm(vec["hog"]) < 50.0:
                    search_requests.append(QueryRequest(
                        query=vec["colour"],
                        using="colour",
                        limit=TOP_K,
                        offset=offset,
                        with_payload=True,
                    ))
                else:
                    # Use a larger prefetch limit (500) to ensure RRF fusion
                    # has enough candidates to rank for deeper rounds.
                    search_requests.append(QueryRequest(
                        prefetch=[
                            Prefetch(query=vec["hog"],    using="hog",    limit=500),
                            Prefetch(query=vec["colour"], using="colour", limit=500),
                        ],
                        query=FusionQuery(fusion=Fusion.RRF),
                        limit=TOP_K,
                        offset=offset,
                        with_payload=True,
                    ))
            else:
                search_requests.append(QueryRequest(
                    query=vec.tolist(),
                    limit=TOP_K,
                    offset=offset,
                    with_payload=True,
                ))

        # ── Batch round-trip ──────────────────────────────────────────────────
        batch_results = app_state.qdrant.query_batch_points(
            collection_name=QDRANT_COLLECTION,
            requests=search_requests,
        )

        # ── Greedy Assignment ─────────────────────────────────────────────────
        newly_resolved = []
        for i, query_result in enumerate(batch_results):
            orig_idx = pending_indices[i]
            candidates = query_result.points
            
            # Save fallback from Round 0 (absolute best match)
            if round_idx == 0 and candidates:
                fallbacks[orig_idx] = candidates[0]

            chosen = None
            for candidate in candidates:
                if candidate.id not in used_ids:
                    chosen = candidate
                    break
            
            if chosen:
                final_results[orig_idx] = chosen
                used_ids.add(chosen.id)
                newly_resolved.append(orig_idx)

        # Update pending list
        resolved_count = len(newly_resolved)
        resolved_set = set(newly_resolved)
        pending_indices = [idx for idx in pending_indices if idx not in resolved_set]

        if round_idx > 0 or resolved_count < len(batch_results):
            print(f"[Convert] Round {round_idx} resolved {resolved_count} unique patches. {len(pending_indices)} still pending.", flush=True)

        if not newly_resolved:
            break

    # ── Final Fallback ────────────────────────────────────────────────────────
    # For any patches that still didn't find a unique match after all rounds,
    # use their original best match (duplicate).
    for idx in pending_indices:
        final_results[idx] = fallbacks[idx]


    print(f"[Convert] Search complete. Assembling output image...", flush=True)

    # Assemble output
    out_patch = pixels_per_patch * output_scale
    output = Image.new("RGB", (cols * out_patch, rows * out_patch))
    images_dir = os.path.join(DATA_DIR, "images")

    idx = 0
    for r in range(rows):
        for c in range(cols):
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
        "grid_rows": rows,
        "grid_cols": cols,
    }
