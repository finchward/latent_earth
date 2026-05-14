"""
camera.py
Real-time camera mosaic backend logic.
"""

from __future__ import annotations

import base64
import io
import time
import cv2

import numpy as np
from PIL import Image
from qdrant_client.models import QueryRequest, Prefetch, FusionQuery, Fusion, SearchParams

from config import QDRANT_COLLECTION, EMBEDDING_METHOD, CAM_PATCH_DIM
from embedder import compute_features
from state import app_state


def process_camera_frame(image_bytes: bytes) -> tuple[list[list[str]], list[list[dict]]]:
    """
    Decode the camera frame bytes, slice into CAM_PATCH_DIM×CAM_PATCH_DIM patches,
    compute vectors based on configuration, and batch-search Qdrant.

    Returns an CAM_PATCH_DIM×CAM_PATCH_DIM grid of image URLs (e.g. "/images/abc123.png").
    """
    t0 = time.perf_counter()
    print(f"[Camera] Got {len(image_bytes)} bytes, magic: {image_bytes[:4].hex()}", flush=True)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # # Boost saturation by 30%
    # arr = np.array(img)

    # hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    # hsv[..., 1] *= 1.3
    # hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)

    # arr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # img = Image.fromarray(arr)

    t_load = time.perf_counter() - t0

    # Dynamically determine patch size (e.g. 400px / 8 -> 50px)
    w, h = img.size
    patch_size = w // CAM_PATCH_DIM

    # Slice the image into patches
    patches = []
    for r in range(CAM_PATCH_DIM):
        for c in range(CAM_PATCH_DIM):
            left = c * patch_size
            upper = r * patch_size
            right = left + patch_size
            lower = upper + patch_size
            patches.append(img.crop((left, upper, right, lower)))

    t_slice = time.perf_counter() - t0 - t_load
    print(f"[Timer] Load: {t_load*1000:.1f}ms, Slice: {t_slice*1000:.1f}ms", flush=True)

    t1 = time.perf_counter()
    vectors = compute_features(patches)
    t_feat = time.perf_counter() - t1

    print(f"[Camera] Computed {len(vectors)} {EMBEDDING_METHOD.upper()} vectors ({t_feat*1000:.1f}ms). Batch searching Qdrant...", flush=True)

    TOP_K = 20  # Reduced from 10 to speed up response size
    HNSW_EF = 32
    search_params = SearchParams(hnsw_ef=HNSW_EF)
    WITH_PAYLOAD = ["filename", "lat", "lon", "date"]  # Fetch metadata for tooltip

    search_requests: list[QueryRequest] = []
    if EMBEDDING_METHOD == "hog_hybrid":
        for vec in vectors:
            if np.linalg.norm(vec["hog"]) < 5.0:
                # Flat/featureless patch — colour descriptor only
                search_requests.append(QueryRequest(
                    query=vec["colour"],
                    using="colour",
                    limit=TOP_K,
                    with_payload=WITH_PAYLOAD,
                    params=search_params,
                ))
            else:
                search_requests.append(QueryRequest(
                    prefetch=[
                        Prefetch(query=vec["hog"],    using="hog",    limit=100), # Reduced from 500
                        Prefetch(query=vec["colour"], using="colour", limit=100), # Reduced from 500
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=TOP_K,
                    with_payload=WITH_PAYLOAD,
                    params=search_params,
                ))
    else:
        # fused_hybrid, hog, dinov2 — all return an ndarray
        for vec in vectors:
            search_requests.append(QueryRequest(
                query=vec.tolist(),
                limit=TOP_K,
                with_payload=WITH_PAYLOAD,
                params=search_params,
            ))

    t2 = time.perf_counter()
    batch_results = app_state.qdrant.query_batch_points(
        collection_name=QDRANT_COLLECTION,
        requests=search_requests,
    )
    t_search = time.perf_counter() - t2

    t3 = time.perf_counter()
    # Client-side greedy: prefer the best-match that hasn't been used yet.
    # This keeps the mosaic visually diverse across the grid.
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

    # Build the grid of image URLs directly from Qdrant payloads.
    grid_urls: list[list[str]] = []
    grid_meta: list[list[dict]] = []
    idx = 0
    for r in range(CAM_PATCH_DIM):
        row_urls: list[str] = []
        row_meta: list[dict] = []
        for c in range(CAM_PATCH_DIM):
            point = chosen_points[idx]
            if point:
                filename = point.payload.get("filename", "")
                row_urls.append(f"/api/thumb/{filename}" if filename else "")
                row_meta.append({
                    "lat": point.payload.get("lat"),
                    "lon": point.payload.get("lon"),
                    "date": point.payload.get("date")
                })
            else:
                row_urls.append("")
                row_meta.append(None)
            idx += 1
        grid_urls.append(row_urls)
        grid_meta.append(row_meta)
    
    t_grid = time.perf_counter() - t3
    print(f"[Timer] Search: {t_search*1000:.1f}ms, Grid: {t_grid*1000:.1f}ms, Total: {(time.perf_counter()-t0)*1000:.1f}ms", flush=True)

    return grid_urls, grid_meta