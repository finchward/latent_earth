#!/usr/bin/env python3
"""
satellite_explorer.py
=====================
Standalone FastAPI + localhost web app for exploring a Qdrant satellite
imagery database through the lens of principal components.

At startup it:
  1. Syncs Qdrant from Google Drive
  2. Loads CLIP ViT-B/32
  3. Scrolls all vectors (up to 10 000) and fits 4-component PCA
  4. Prints eigenvalues to console
  5. Launches a localhost web server and opens your browser

Usage:
    python satellite_explorer.py

Key env vars (override config defaults):
    GEE_PROJECT_ID    – Google Cloud / Earth Engine project
    QDRANT_DRIVE_PATH – full path to qdrant_satellite_storage on Drive
    PORT              – HTTP port (default 7432)
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IMPORTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import asyncio
import base64
import datetime
import glob
import io
import json
import math
import os
import random
import re
import shutil
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Optional

import numpy as np
import uvicorn
import webbrowser
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import ee
import requests as http_requests
from PIL import Image

import torch
from transformers import AutoImageProcessor, AutoModel

from qdrant_client import QdrantClient
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG  ← edit these values for your environment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GEE_PROJECT_ID = os.environ.get("GEE_PROJECT_ID", "mercari-agent")

# Google Drive path to the qdrant_satellite_storage folder.
# The app will auto-detect common locations; set QDRANT_DRIVE_PATH to override.
_DRIVE_GLOBS = [
    # macOS – Google Drive for Desktop
    "~/Library/CloudStorage/GoogleDrive-*/My Drive/qdrant_satellite_storage",
    # macOS – older mount
    "~/Google Drive/My Drive/qdrant_satellite_storage",
    # Linux
    "~/GoogleDrive/My Drive/qdrant_satellite_storage",
    "~/google-drive/qdrant_satellite_storage",
]
_auto_drive = next(
    (m for p in _DRIVE_GLOBS for m in glob.glob(os.path.expanduser(p))), None
)
QDRANT_DRIVE_PATH: str = r"G:\My Drive\qdrant_satellite_storage"
QDRANT_LOCAL_PATH: str = "/tmp/satellite_explorer_qdrant"
QDRANT_COLLECTION: str = "satellite_patches"

# Imagery
PATCH_SIZE_M      = 5_000
IMAGE_SIZE_PX     = 500
GEE_COLLECTION    = "COPERNICUS/S2_SR_HARMONIZED"
BANDS             = ["B4", "B3", "B2"]
MAX_CLOUD_PCT     = 20
MIN_BRIGHTNESS    = 30.0
SENTINEL_START    = "2017-03-28"
WINDOW_DAYS       = 600

# Embedding
VECTOR_SIZE       = 384
EMBED_BATCH_SIZE  = 16

# PCA
MAX_PCA_VECTORS   = 10_000
N_COMPONENTS      = 4

# Display
GRID_THUMB_PX     = 60    # pixels per grid cell
QUERY_PREVIEW_PX  = 240   # pixels for sidebar preview

PORT = int(os.environ.get("PORT", 7432))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GLOBAL STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_state: dict[str, Any] = {
    "ready":          False,
    "status_msg":     "Initialising…",
    "n_vectors":      0,
    "eigenvalues":    [],        # list[float] – % variance explained, len 4
    "pca_components": None,      # np.ndarray (4, 512)
    "db_vectors":     None,      # np.ndarray (N, 512)
    "db_ids":         None,      # list[int | str]
    "db_payloads":    None,      # list[dict]
    "qdrant":         None,
    "clip_model":     None,
    "clip_proc":      None,
    "device":         None,
    # Keyed by (lat, lon) rounded to 6dp.
    # Value: {"embedding": np.ndarray, "date": str, "thumb_b64": str}
    "patch_cache":    {},
}

_executor = ThreadPoolExecutor(max_workers=6)


def _log(msg: str) -> None:
    print(msg, flush=True)
    _state["status_msg"] = msg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GOOGLE EARTH ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SENTINEL_START_DATE = datetime.date.fromisoformat(SENTINEL_START)
_TOTAL_DAYS = (datetime.date.today() - _SENTINEL_START_DATE).days
_LAND_IMAGE: Any = None   # ee.Image, set after GEE init


def _init_gee() -> None:
    global _LAND_IMAGE
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT_ID)
    _LAND_IMAGE = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        .select("occurrence")
        .unmask(100)
        .lt(50)
    )
    _log(f"  ✓ GEE initialised  ({GEE_PROJECT_ID})")


def is_land(lat: float, lon: float) -> bool:
    point = ee.Geometry.Point([lon, lat])
    val   = _LAND_IMAGE.sample(region=point, scale=500).first().getInfo()
    return val is not None and val["properties"].get("occurrence", 0) == 1


def random_land_coord() -> tuple[float, float]:
    """Area-proportional random point on land (−60° to +75° lat)."""
    sin_min = math.sin(math.radians(-60))
    sin_max = math.sin(math.radians(75))
    while True:
        lon = random.uniform(-180, 180)
        lat = math.degrees(math.asin(random.uniform(sin_min, sin_max)))
        if is_land(lat, lon):
            return lat, lon


def _random_date_window() -> tuple[str, str]:
    max_offset = max(_TOTAL_DAYS - WINDOW_DAYS, 0)
    offset = random.randint(0, max_offset)
    start  = _SENTINEL_START_DATE + datetime.timedelta(days=offset)
    end    = start + datetime.timedelta(days=WINDOW_DAYS)
    return str(start), str(end)


def fetch_patch(lat: float, lon: float) -> Optional[tuple[Image.Image, str]]:
    """
    Fetch one Sentinel-2 patch at (lat, lon) from a random time window.
    Returns (PIL Image, scene_date) or None on failure / too dark.
    """
    try:
        point  = ee.Geometry.Point([lon, lat])
        region = point.buffer(PATCH_SIZE_M / 2).bounds()
        ws, we = _random_date_window()

        col = (
            ee.ImageCollection(GEE_COLLECTION)
            .filterBounds(region)
            .filterDate(ws, we)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PCT))
            .select(BANDS)
        )
        if col.size().getInfo() == 0:
            return None

        scenes   = col.toList(col.size())
        n_scenes = scenes.size().getInfo()
        scene    = ee.Image(scenes.get(random.randint(0, n_scenes - 1)))
        ts_ms    = scene.date().millis().getInfo()
        date_str = str(datetime.date.fromtimestamp(ts_ms / 1_000))

        img_ee = scene.divide(10_000).multiply(255).toByte()
        url = img_ee.getThumbURL({
            "region":     region,
            "dimensions": f"{IMAGE_SIZE_PX}x{IMAGE_SIZE_PX}",
            "format":     "png",
            "bands":      BANDS,
        })
        resp = http_requests.get(url, timeout=45)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")

        mean_brightness = float(np.array(img, dtype=np.float32).mean())
        if mean_brightness < MIN_BRIGHTNESS:
            return None
        return img, date_str

    except Exception:
        return None


def fetch_patch_with_retry(
    lat: float, lon: float, max_attempts: int = 30
) -> tuple[Image.Image, str]:
    """Retry fetch_patch up to max_attempts times."""
    for _ in range(max_attempts):
        result = fetch_patch(lat, lon)
        if result is not None:
            return result
    raise ValueError(
        f"No usable Sentinel-2 imagery found at ({lat:.5f}, {lon:.5f}) "
        f"after {max_attempts} attempts. "
        "The location may be persistently cloudy or have no archive coverage."
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLIP EMBEDDING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _init_clip() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"  Loading DINOv2-small on {device}…")
    model     = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model.eval()
    _state["clip_model"] = model
    _state["clip_proc"]  = processor
    _state["device"]     = device
    _log(f"  ✓ DINOv2 ready")


def embed_images(images: list[Image.Image]) -> np.ndarray:
    """Return L2-normalised DINOv2 visual embeddings, shape (N, 384)."""
    model, proc, device = (
        _state["clip_model"], _state["clip_proc"], _state["device"]
    )
    all_emb = []
    for i in range(0, len(images), EMBED_BATCH_SIZE):
        batch  = images[i: i + EMBED_BATCH_SIZE]
        batch  = [img.convert("RGB") for img in batch]
        inputs = proc(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs  = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]
        features = features / features.norm(dim=-1, keepdim=True)
        all_emb.append(features.cpu().numpy())
    return np.vstack(all_emb).astype(np.float32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QDRANT — connect and scroll
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _init_qdrant() -> None:
    if QDRANT_DRIVE_PATH and os.path.isdir(QDRANT_DRIVE_PATH):
        _log(f"  Copying Qdrant storage from Drive…")
        shutil.copytree(QDRANT_DRIVE_PATH, QDRANT_LOCAL_PATH, dirs_exist_ok=True)
        _log(f"  ✓ Restored from {QDRANT_DRIVE_PATH}")
    else:
        _log(
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
        _log(f"  ⚠  Collection '{QDRANT_COLLECTION}' not found — DB is empty.")
    else:
        info = client.get_collection(QDRANT_COLLECTION)
        _log(f"  ✓ Qdrant: {info.points_count} patches in '{QDRANT_COLLECTION}'")
    _state["qdrant"] = client


def _scroll_all_vectors() -> tuple[np.ndarray, list, list]:
    """
    Scroll up to MAX_PCA_VECTORS points from Qdrant.
    Returns (vectors ndarray, id list, payload list).
    """
    client   = _state["qdrant"]
    vectors, ids, payloads = [], [], []
    offset   = None

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
            _log(f"  ⚠  Scroll error: {exc}")
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

    _log(f"  ✓ Scrolled {len(vectors)} vectors")
    if not vectors:
        return np.empty((0, VECTOR_SIZE), dtype=np.float32), [], []
    return np.array(vectors, dtype=np.float32), ids, payloads


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PCA ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fit_pca(db_vectors: np.ndarray) -> None:
    """Fit PCA on db_vectors, store components + eigenvalues in _state."""
    _log(f"  Fitting PCA ({N_COMPONENTS} components) on {len(db_vectors)} vectors…")
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    pca.fit(db_vectors)

    evr = pca.explained_variance_ratio_
    print()
    print("  ┌─ Principal Component Variance ─────────────────────────┐")
    for i, ev in enumerate(evr):
        bar = "█" * int(ev * 50)
        print(f"  │  PC{i+1}  {ev*100:5.2f}%  {bar}")
    print(f"  │  ─────────────────────────────────────────────────────")
    print(f"  │  Total {evr.sum()*100:.2f}% of variance captured")
    print("  └────────────────────────────────────────────────────────┘")
    print()

    _state["pca_components"] = pca.components_.astype(np.float32)  # (4, 512)
    _state["eigenvalues"]    = evr.tolist()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRID BUILDER — 17 × 17 patch assignment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_thumb_from_payload(payload: dict, size: int = GRID_THUMB_PX) -> str:
    """Decode payload image_b64, resize to size×size, return base64 PNG."""
    raw = base64.b64decode(payload["image_b64"])
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_grid(
    query_vec: np.ndarray,
) -> tuple[list[list[Optional[dict]]], list[float]]:
    """
    Assign database patches to a 17×17 grid centred on the query.

    Grid orientation:
      Right  (+col) ← PC1 closest
      Up     (−row) ← PC2 closest
      Left   (−col) ← PC3 closest
      Down   (+row) ← PC4 closest
      Diagonals     ← blended eigenvectors, greedy centre-first

    Returns:
        grid       – 17×17 list-of-lists; each cell is a dict or None
        pc_weights – [w1,w2,w3,w4] summing to 1.0 (absolute PC projections)
    """
    db_vec  = _state["db_vectors"]      # (N, 512)
    db_ids  = _state["db_ids"]          # list[int]
    db_pay  = _state["db_payloads"]     # list[dict]
    comps   = _state["pca_components"]  # (4, 512)

    N = len(db_ids) if db_ids else 0

    empty_grid = [[None] * 17 for _ in range(17)]
    if N == 0 or comps is None:
        return empty_grid, [0.25, 0.25, 0.25, 0.25]

    e1, e2, e3, e4 = comps  # eigenvectors, each (512,)

    # ── Projections ──────────────────────────────────────────────────────────
    # proj[i, k] = dot(db_vec[i], e_k)
    proj   = db_vec @ comps.T          # (N, 4)
    q_proj = (query_vec @ comps.T).astype(np.float64)  # (4,)
    q1, q2, q3, q4 = q_proj

    # PC weights for glow: |q_k| / Σ|q_j|
    abs_q      = np.abs(q_proj)
    total_abs  = abs_q.sum()
    pc_weights = (abs_q / total_abs).tolist() if total_abs > 0 else [0.25] * 4

    # ── Helpers ───────────────────────────────────────────────────────────────
    grid     = [[None] * 17 for _ in range(17)]
    assigned = set()   # set of DB integer indices already placed

    def make_cell(db_idx: int, axis: str) -> dict:
        pay = db_pay[db_idx]
        return {
            "type":      "patch",
            "thumb_b64": _make_thumb_from_payload(pay),
            "lat":       pay.get("lat"),
            "lon":       pay.get("lon"),
            "date":      pay.get("date", ""),
            "axis":      axis,
        }

    def pick_closest_axis(axis_k: int, n: int = 8) -> list[int]:
        """Return up to n unassigned DB indices closest to query along PC_k."""
        diffs = np.abs(proj[:, axis_k] - q_proj[axis_k])
        order = np.argsort(diffs)
        result = []
        for idx in order:
            if int(idx) not in assigned and len(result) < n:
                result.append(int(idx))
        return result

    # ── Axis cells (highest priority) ─────────────────────────────────────────
    # RIGHT (PC1): row=8, col 9..16  (col 9 = most similar)
    for i, idx in enumerate(pick_closest_axis(0)):
        grid[8][9 + i] = make_cell(idx, "PC1")
        assigned.add(idx)

    # UP (PC2): col=8, row 7..0  (row 7 = most similar)
    for i, idx in enumerate(pick_closest_axis(1)):
        grid[7 - i][8] = make_cell(idx, "PC2")
        assigned.add(idx)

    # LEFT (PC3): row=8, col 7..0  (col 7 = most similar)
    for i, idx in enumerate(pick_closest_axis(2)):
        grid[8][7 - i] = make_cell(idx, "PC3")
        assigned.add(idx)

    # DOWN (PC4): col=8, row 9..16  (row 9 = most similar)
    for i, idx in enumerate(pick_closest_axis(3)):
        grid[9 + i][8] = make_cell(idx, "PC4")
        assigned.add(idx)

    # ── Diagonal cells (greedy, centre-outward) ────────────────────────────────
    diag_offsets = sorted(
        [(dr, dc)
         for dr in range(-8, 9)
         for dc in range(-8, 9)
         if dr != 0 and dc != 0],
        key=lambda rc: rc[0] ** 2 + rc[1] ** 2,
    )

    avail = np.ones(N, dtype=bool)
    for idx in assigned:
        avail[idx] = False

    for (dr, dc) in diag_offsets:
        row, col = 8 + dr, 8 + dc

        # Blend the two adjacent axis eigenvectors, weighted by grid distance
        if   dc > 0 and dr < 0:   d = float(dc) * e1 + float(-dr) * e2   # top-right
        elif dc > 0 and dr > 0:   d = float(dc) * e1 + float(dr)  * e4   # bottom-right
        elif dc < 0 and dr > 0:   d = float(-dc) * e3 + float(dr) * e4   # bottom-left
        else:                      d = float(-dc) * e3 + float(-dr) * e2  # top-left

        norm = float(np.linalg.norm(d))
        if norm < 1e-9:
            continue
        d = (d / norm).astype(np.float32)

        q_score  = float(query_vec @ d)
        scores   = db_vec @ d              # (N,)
        diffs_d  = np.abs(scores - q_score)
        diffs_d[~avail] = np.inf

        best = int(np.argmin(diffs_d))
        if not np.isinf(diffs_d[best]):
            grid[row][col] = make_cell(best, "diag")
            avail[best]    = False
            assigned.add(best)

    return grid, pc_weights


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IMAGE HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def pil_to_b64(img: Image.Image, max_px: Optional[int] = None) -> str:
    """Encode PIL image as base64 PNG, optionally scaling to fit max_px."""
    if max_px:
        img = img.copy()
        img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def parse_coordinate(text: str) -> tuple[float, float]:
    """
    Parse latitude and longitude from a string.
    Supports:
      • Decimal:  -33.8688, 151.2093
      • DMS:      33°52′12″S, 151°12′36″E
    """
    text = text.strip()

    # Decimal degrees (with comma or space separator)
    m = re.match(r'^([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)$', text)
    if m:
        return float(m.group(1)), float(m.group(2))

    # DMS  —  accepts ° ′ ″ or plain ' " with optional spaces
    dms = re.search(
        r'(\d+)\s*[°]\s*(\d+)\s*[\'′]\s*(\d+(?:\.\d+)?)\s*[\"″]?\s*([NSns])'
        r'[,\s]+'
        r'(\d+)\s*[°]\s*(\d+)\s*[\'′]\s*(\d+(?:\.\d+)?)\s*[\"″]?\s*([EWew])',
        text,
    )
    if dms:
        lat = (float(dms.group(1))
               + float(dms.group(2)) / 60
               + float(dms.group(3)) / 3600)
        lon = (float(dms.group(5))
               + float(dms.group(6)) / 60
               + float(dms.group(7)) / 3600)
        if dms.group(4).upper() == 'S':
            lat = -lat
        if dms.group(8).upper() == 'W':
            lon = -lon
        return lat, lon

    raise ValueError(
        f"Could not parse {text!r}. "
        "Use decimal (−33.8688, 151.2093) or DMS (33°52′12″S, 151°12′36″E)."
    )


def _cache_key(lat: float, lon: float) -> tuple[float, float]:
    return (round(lat, 6), round(lon, 6))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HTML TEMPLATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Sentinel Atlas</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=DM+Mono:wght@300;400&display=swap" rel="stylesheet" />
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:          #eae6dd;
  --surface:     #f2efe7;
  --sidebar:     #e2ddd5;
  --border:      #b4aca0;
  --border-lt:   #c8c1b8;
  --text:        #1b1815;
  --muted:       #7a7168;
  --empty-bg:    #d8d3ca;
  --font-serif:  'Cormorant Garamond', Georgia, serif;
  --font-mono:   'DM Mono', 'Courier New', monospace;
  --glow-dur:    0.9s;
}

html, body { height: 100%; overflow: hidden; }

body {
  font-family: var(--font-mono);
  background: var(--bg);
  color: var(--text);
  display: flex;
  flex-direction: column;
  font-size: 13px;
}

/* ══ HEADER ══════════════════════════════════════════════════════════════════ */
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 28px;
  background: var(--sidebar);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
  gap: 16px;
}

h1 {
  font-family: var(--font-serif);
  font-weight: 300;
  font-size: 1.55rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  white-space: nowrap;
}
h1 em { font-style: italic; font-weight: 300; }

#status-pill {
  font-size: 0.62rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  border: 1px solid var(--border);
  padding: 4px 12px;
  border-radius: 20px;
  white-space: nowrap;
  transition: all 0.3s;
}
#status-pill.ready {
  color: var(--text);
  border-color: var(--text);
}

/* ══ LAYOUT ═══════════════════════════════════════════════════════════════════ */
.layout {
  display: flex;
  flex: 1;
  min-height: 0;
}

/* ══ SIDEBAR ══════════════════════════════════════════════════════════════════ */
aside {
  width: 288px;
  flex-shrink: 0;
  background: var(--sidebar);
  border-right: 1px solid var(--border);
  overflow-y: auto;
  padding: 22px 18px 32px;
  display: flex;
  flex-direction: column;
  gap: 26px;
}

.s-label {
  font-size: 0.57rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 10px;
  display: block;
}

/* Coord input */
#coord-input {
  width: 100%;
  padding: 9px 11px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 2px;
  font-family: var(--font-mono);
  font-size: 0.72rem;
  color: var(--text);
  outline: none;
  transition: border-color 0.15s;
  letter-spacing: 0.03em;
}
#coord-input:focus { border-color: var(--text); }
#coord-input::placeholder { color: var(--muted); }

.coord-hint {
  font-size: 0.57rem;
  color: var(--muted);
  margin-top: 6px;
  line-height: 1.6;
}

.btn-row { display: flex; gap: 8px; margin-top: 11px; }

button {
  font-family: var(--font-mono);
  font-size: 0.62rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  border: 1px solid var(--border);
  border-radius: 2px;
  background: var(--surface);
  color: var(--text);
  padding: 8px 14px;
  cursor: pointer;
  transition: background 0.12s, border-color 0.12s, color 0.12s;
  white-space: nowrap;
}
button:hover:not(:disabled) { background: var(--bg); border-color: var(--text); }
button:disabled { opacity: 0.35; cursor: not-allowed; }

button.primary {
  background: var(--text);
  color: var(--surface);
  border-color: var(--text);
}
button.primary:hover:not(:disabled) { background: #35302a; }

/* Preview */
#preview-section { display: none; }

#preview-img {
  width: 100%;
  aspect-ratio: 1;
  object-fit: cover;
  border: 1px solid var(--border);
  display: block;
  background: var(--empty-bg);
}

.meta-grid {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 3px 10px;
  margin-top: 10px;
  font-size: 0.6rem;
}
.meta-grid .mk { color: var(--muted); }
.meta-grid .mv { color: var(--text); }

/* PC decomposition bars */
#pc-section { display: none; }

.pc-row {
  display: flex;
  align-items: center;
  gap: 9px;
  margin-bottom: 9px;
}
.pc-lbl {
  font-size: 0.6rem;
  width: 42px;
  flex-shrink: 0;
  color: var(--muted);
}
.pc-track {
  flex: 1;
  height: 3px;
  background: var(--border-lt);
  border-radius: 2px;
  overflow: hidden;
  position: relative;
}
.pc-fill {
  position: absolute;
  inset: 0 auto 0 0;
  width: 0%;
  background: var(--text);
  border-radius: 2px;
  transition: width 0.7s cubic-bezier(0.4, 0, 0.2, 1);
}
.pc-pct {
  font-size: 0.58rem;
  color: var(--muted);
  width: 38px;
  text-align: right;
}

/* Error message */
#error-msg {
  font-size: 0.62rem;
  color: #7a2c2c;
  display: none;
  margin-top: 8px;
  line-height: 1.6;
  padding: 7px 9px;
  background: rgba(122,44,44,0.07);
  border: 1px solid rgba(122,44,44,0.18);
  border-radius: 2px;
}

/* Index info */
.info-line {
  font-size: 0.6rem;
  color: var(--muted);
  line-height: 1.8;
}
.info-line span { color: var(--text); }

/* Legend */
.legend-entry {
  display: flex;
  align-items: center;
  gap: 9px;
  font-size: 0.58rem;
  color: var(--muted);
  margin-bottom: 6px;
}
.legend-arrow {
  font-size: 0.9rem;
  width: 16px;
  text-align: center;
  color: var(--text);
}
.legend-swatch {
  width: 12px; height: 12px;
  border: 1px solid var(--border);
  flex-shrink: 0;
}
.sw-empty { background: var(--empty-bg);
  background-image: repeating-linear-gradient(45deg,transparent,transparent 3px,rgba(0,0,0,.07) 3px,rgba(0,0,0,.07) 4px); }
.sw-query { background: var(--surface); outline: 2px solid var(--text); outline-offset: 1px; }

/* ══ MAIN / GRID AREA ════════════════════════════════════════════════════════ */
main {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: auto;
  padding: 44px 40px;
  position: relative;
}

/* Compass label wrapper */
.compass-wrap {
  position: relative;
  display: inline-block;
}
.clabel {
  position: absolute;
  font-size: 0.57rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  pointer-events: none;
}
.clabel.top    { bottom: calc(100% + 10px); left: 50%; transform: translateX(-50%); }
.clabel.bottom { top: calc(100% + 10px);    left: 50%; transform: translateX(-50%); }
.clabel.left   { right: calc(100% + 14px);  top: 50%;  transform: translateY(-50%); }
.clabel.right  { left: calc(100% + 14px);   top: 50%;  transform: translateY(-50%); }

/* The actual grid */
#grid-wrapper {
  display: grid;
  grid-template-columns: repeat(17, 60px);
  grid-template-rows:    repeat(17, 60px);
  gap: 1px;
  background: var(--border-lt);
  border: 1px solid var(--border);
  /* box-shadow set dynamically by JS */
  transition: box-shadow var(--glow-dur) cubic-bezier(0.25, 0.1, 0.25, 1);
}

/* Each cell */
.cell {
  width: 60px;
  height: 60px;
  position: relative;
  overflow: visible;          /* allow tooltip bleed */
  background: var(--surface);
}
.cell-inner {
  width: 100%;
  height: 100%;
  overflow: hidden;
  position: relative;
}
.cell img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  pointer-events: none;
}
.cell.empty .cell-inner {
  background: var(--empty-bg);
  background-image: repeating-linear-gradient(
    45deg,
    transparent, transparent 6px,
    rgba(0,0,0,.05) 6px, rgba(0,0,0,.05) 7px
  );
}
.cell.query-cell .cell-inner {
  outline: 2px solid var(--text);
  outline-offset: -2px;
  z-index: 2;
}
/* Subtle axis highlight */
.cell.ax-PC1 .cell-inner,
.cell.ax-PC2 .cell-inner,
.cell.ax-PC3 .cell-inner,
.cell.ax-PC4 .cell-inner {
  outline: 1px solid rgba(27,24,21,0.22);
  outline-offset: -1px;
}

/* Tooltip */
.tooltip {
  display: none;
  position: fixed;
  background: var(--text);
  color: var(--surface);
  font-family: var(--font-mono);
  font-size: 0.57rem;
  line-height: 1.55;
  padding: 6px 9px;
  white-space: nowrap;
  border-radius: 2px;
  z-index: 999;
  pointer-events: none;
  transform: translate(12px, -50%);
}

/* ══ LOADING OVERLAY ═════════════════════════════════════════════════════════ */
#loading-overlay {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(234, 230, 221, 0.72);
  backdrop-filter: blur(4px);
  -webkit-backdrop-filter: blur(4px);
  z-index: 200;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 14px;
}
#loading-overlay.vis { display: flex; }

.spinner {
  width: 26px; height: 26px;
  border: 2px solid var(--border);
  border-top-color: var(--text);
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

#loading-msg {
  font-size: 0.62rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
}

/* Idle placeholder grid */
#idle-grid {
  display: grid;
  grid-template-columns: repeat(17, 60px);
  grid-template-rows:    repeat(17, 60px);
  gap: 1px;
  background: var(--border-lt);
  border: 1px solid var(--border);
  opacity: 0.4;
}
.idle-cell { background: var(--empty-bg); }
.idle-cell.ic-center { background: var(--border); }

/* ══ ANIMATE IN ══════════════════════════════════════════════════════════════ */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeIn 0.35s ease both; }
</style>
</head>
<body>

<header>
  <h1>Sentinel <em>Atlas</em></h1>
  <div id="status-pill">Initialising…</div>
</header>

<div class="layout">

  <!-- ── SIDEBAR ────────────────────────────────────────────────────────── -->
  <aside>

    <!-- Coordinates -->
    <div id="coord-section">
      <span class="s-label">Coordinates</span>
      <input id="coord-input" type="text"
             placeholder="−33.8688, 151.2093"
             autocomplete="off" spellcheck="false" />
      <div class="coord-hint">
        Decimal  or  DMS — e.g.&nbsp;48°51′N&thinsp;2°21′E
      </div>
      <div id="error-msg"></div>
      <div class="btn-row">
        <button id="btn-random" disabled>⟳ Random</button>
        <button id="btn-fetch" class="primary" disabled>Fetch Patch</button>
      </div>
    </div>

    <!-- Preview -->
    <div id="preview-section">
      <span class="s-label">Query Patch</span>
      <img id="preview-img" alt="query patch" />
      <div class="meta-grid">
        <span class="mk">lat</span>  <span class="mv" id="meta-lat">—</span>
        <span class="mk">lon</span>  <span class="mv" id="meta-lon">—</span>
        <span class="mk">date</span> <span class="mv" id="meta-date">—</span>
      </div>
      <div class="btn-row" style="margin-top:14px">
        <button id="btn-build" class="primary">Build Grid →</button>
      </div>
    </div>

    <!-- PC bars -->
    <div id="pc-section">
      <span class="s-label">PC Decomposition of Query</span>
      <div class="pc-row">
        <span class="pc-lbl">PC1 →</span>
        <div class="pc-track"><div class="pc-fill" id="bar1"></div></div>
        <span class="pc-pct" id="pct1">—</span>
      </div>
      <div class="pc-row">
        <span class="pc-lbl">PC2 ↑</span>
        <div class="pc-track"><div class="pc-fill" id="bar2"></div></div>
        <span class="pc-pct" id="pct2">—</span>
      </div>
      <div class="pc-row">
        <span class="pc-lbl">PC3 ←</span>
        <div class="pc-track"><div class="pc-fill" id="bar3"></div></div>
        <span class="pc-pct" id="pct3">—</span>
      </div>
      <div class="pc-row">
        <span class="pc-lbl">PC4 ↓</span>
        <div class="pc-track"><div class="pc-fill" id="bar4"></div></div>
        <span class="pc-pct" id="pct4">—</span>
      </div>
    </div>

    <!-- Index -->
    <div>
      <span class="s-label">Index</span>
      <div class="info-line" id="index-info">
        <span id="n-vectors">—</span> patches indexed
      </div>
      <div class="info-line" id="eigen-info" style="display:none">
        PC1–4 capture <span id="total-ev">—</span>% of variance
      </div>
    </div>

    <!-- Legend -->
    <div>
      <span class="s-label">Grid Key</span>
      <div class="legend-entry"><span class="legend-arrow">→</span> PC1 axis (right)</div>
      <div class="legend-entry"><span class="legend-arrow">↑</span> PC2 axis (up)</div>
      <div class="legend-entry"><span class="legend-arrow">←</span> PC3 axis (left)</div>
      <div class="legend-entry"><span class="legend-arrow">↓</span> PC4 axis (down)</div>
      <div class="legend-entry" style="margin-top:4px">
        <div class="legend-swatch sw-query"></div> query patch (centre)
      </div>
      <div class="legend-entry">
        <div class="legend-swatch sw-empty"></div> no patch available
      </div>
      <div class="legend-entry" style="margin-top:6px; font-size:0.55rem; color:var(--muted);">
        Border glow ∝ query's PC decomposition
      </div>
    </div>

  </aside>

  <!-- ── MAIN ──────────────────────────────────────────────────────────── -->
  <main id="main-area">

    <!-- Idle placeholder -->
    <div id="idle-wrapper">
      <div id="idle-grid"></div>
    </div>

    <!-- Live grid (compass + wrapper) — hidden until first build -->
    <div id="grid-outer" style="display:none">
      <div class="compass-wrap">
        <div class="clabel top">PC2 &uarr;</div>
        <div class="clabel bottom">PC4 &darr;</div>
        <div class="clabel left">&larr; PC3</div>
        <div class="clabel right">PC1 &rarr;</div>
        <div id="grid-wrapper"></div>
      </div>
    </div>

  </main>
</div>

<!-- Loading overlay -->
<div id="loading-overlay">
  <div class="spinner"></div>
  <div id="loading-msg">Working…</div>
</div>

<!-- Global floating tooltip -->
<div class="tooltip" id="gtooltip"></div>

<script>
// ── GLOBALS ────────────────────────────────────────────────────────────────
let appReady = false;
let lastCoord = null;   // { lat, lon }
const tooltip = document.getElementById('gtooltip');

// ── $ SHORTHAND ───────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ── LOADING ────────────────────────────────────────────────────────────────
function showLoading(msg) {
  $('loading-msg').textContent = msg;
  $('loading-overlay').classList.add('vis');
}
function hideLoading() {
  $('loading-overlay').classList.remove('vis');
}

// ── ERROR ──────────────────────────────────────────────────────────────────
function showError(msg) {
  const el = $('error-msg');
  el.textContent = msg;
  el.style.display = 'block';
}
function clearError() {
  $('error-msg').style.display = 'none';
}

// ── API ────────────────────────────────────────────────────────────────────
async function api(path, body) {
  const opts = {
    method: body !== undefined ? 'POST' : 'GET',
    headers: {},
  };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({ detail: r.statusText }));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}

// ── IDLE PLACEHOLDER ────────────────────────────────────────────────────────
function buildIdleGrid() {
  const g = $('idle-grid');
  for (let r = 0; r < 17; r++) {
    for (let c = 0; c < 17; c++) {
      const div = document.createElement('div');
      div.className = 'idle-cell' + (r === 8 && c === 8 ? ' ic-center' : '');
      g.appendChild(div);
    }
  }
}

// ── STATUS POLL ────────────────────────────────────────────────────────────
async function pollStatus() {
  try {
    const s = await api('/api/status');
    if (s.ready) {
      $('status-pill').textContent = `Ready · ${s.n_vectors.toLocaleString()} vectors`;
      $('status-pill').classList.add('ready');
      $('n-vectors').textContent = s.n_vectors.toLocaleString();
      if (s.eigenvalues && s.eigenvalues.length === 4) {
        const total = s.eigenvalues.reduce((a, b) => a + b, 0);
        $('total-ev').textContent = (total * 100).toFixed(1);
        $('eigen-info').style.display = 'block';
      }
      setButtons(true);
      appReady = true;
    } else {
      $('status-pill').textContent = s.message || 'Initialising…';
      setTimeout(pollStatus, 1800);
    }
  } catch (_) {
    setTimeout(pollStatus, 2500);
  }
}

// ── BUTTONS ─────────────────────────────────────────────────────────────────
function setButtons(on) {
  $('btn-random').disabled = !on;
  $('btn-fetch').disabled  = !on;
}

// ── RANDOM COORD ────────────────────────────────────────────────────────────
$('btn-random').addEventListener('click', async () => {
  clearError();
  showLoading('Sampling random land coordinate…');
  setButtons(false);
  try {
    const { lat, lon } = await api('/api/random-coords', {});
    $('coord-input').value = `${lat.toFixed(6)}, ${lon.toFixed(6)}`;
    lastCoord = { lat, lon };
    // Reset preview whenever a new coord is picked
    $('preview-section').style.display = 'none';
    $('pc-section').style.display = 'none';
  } catch (e) {
    showError(e.message);
  } finally {
    setButtons(true);
    hideLoading();
  }
});

// ── FETCH PATCH ─────────────────────────────────────────────────────────────
$('btn-fetch').addEventListener('click', async () => {
  clearError();
  const raw = $('coord-input').value.trim();
  if (!raw) { showError('Enter coordinates first.'); return; }

  showLoading('Querying Sentinel-2 archive…');
  setButtons(false);
  $('preview-section').style.display = 'none';
  $('pc-section').style.display = 'none';

  try {
    const d = await api('/api/fetch-patch', { coord_str: raw });
    lastCoord = { lat: d.lat, lon: d.lon };

    $('preview-img').src = 'data:image/png;base64,' + d.preview_b64;
    $('meta-lat').textContent  = d.lat.toFixed(5);
    $('meta-lon').textContent  = d.lon.toFixed(5);
    $('meta-date').textContent = d.date;

    $('preview-section').style.display = 'block';
    $('preview-section').classList.add('fade-in');
  } catch (e) {
    showError(e.message);
  } finally {
    setButtons(true);
    hideLoading();
  }
});

// ── BUILD GRID ───────────────────────────────────────────────────────────────
$('btn-build').addEventListener('click', async () => {
  if (!lastCoord) return;
  clearError();
  showLoading('Projecting embeddings onto principal components…');
  setButtons(false);
  $('btn-build').disabled = true;

  try {
    const data = await api('/api/build-grid', lastCoord);
    renderGrid(data);
    applyGlow(data.pc_weights);
    showPCBars(data.pc_weights);
  } catch (e) {
    showError(e.message);
  } finally {
    setButtons(true);
    $('btn-build').disabled = false;
    hideLoading();
  }
});

// ── RENDER GRID ──────────────────────────────────────────────────────────────
function renderGrid(data) {
  const wrapper = $('grid-wrapper');
  wrapper.innerHTML = '';
  const { grid, query } = data;

  for (let r = 0; r < 17; r++) {
    for (let c = 0; c < 17; c++) {
      const cell  = document.createElement('div');
      const inner = document.createElement('div');
      cell.className  = 'cell';
      inner.className = 'cell-inner';

      if (r === 8 && c === 8) {
        // ── Query cell ──
        cell.classList.add('query-cell');
        if (query.thumb_b64) {
          const img = document.createElement('img');
          img.src = 'data:image/png;base64,' + query.thumb_b64;
          inner.appendChild(img);
        }
        attachTooltip(cell,
          `QUERY\n${fmtCoord(query.lat, query.lon)}\n${query.date || ''}`
        );
      } else {
        const patch = grid[r][c];
        if (patch) {
          if (patch.axis && patch.axis !== 'diag') {
            cell.classList.add('ax-' + patch.axis);
          }
          const img = document.createElement('img');
          img.src = 'data:image/png;base64,' + patch.thumb_b64;
          img.loading = 'lazy';
          inner.appendChild(img);
          attachTooltip(cell,
            `${fmtCoord(patch.lat, patch.lon)}\n${patch.date}\n${patch.axis}`
          );
        } else {
          cell.classList.add('empty');
        }
      }

      cell.appendChild(inner);
      wrapper.appendChild(cell);
    }
  }

  $('idle-wrapper').style.display = 'none';
  $('grid-outer').style.display   = 'block';
  $('grid-outer').classList.add('fade-in');
}

function fmtCoord(lat, lon) {
  if (lat == null || lon == null) return '—';
  return `${lat.toFixed(4)}, ${lon.toFixed(4)}`;
}

// ── TOOLTIP ──────────────────────────────────────────────────────────────────
function attachTooltip(el, text) {
  el.addEventListener('mouseenter', () => {
    tooltip.style.display = 'block';
    tooltip.innerHTML = text.replace(/\n/g, '<br>');
  });
  el.addEventListener('mousemove', e => {
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top  = (e.clientY - 18) + 'px';
  });
  el.addEventListener('mouseleave', () => {
    tooltip.style.display = 'none';
  });
}

// ── DIRECTIONAL GLOW ─────────────────────────────────────────────────────────
// The grid receives 4 directional box-shadows:
//   PC1 weight → shadow extends rightward
//   PC2 weight → shadow extends upward
//   PC3 weight → shadow extends leftward
//   PC4 weight → shadow extends downward
// The weight is |query projection onto PC_k| / sum of all |projections|,
// so the dominant component creates the strongest visible surge.

function applyGlow(weights) {
  const [w1, w2, w3, w4] = weights;
  const maxOff  = 100;   // max pixel offset
  const maxBlur = 60;    // max blur radius

  function shadow(xOff, yOff, w) {
    const off  = (w * maxOff).toFixed(1);
    const blur = (w * maxBlur).toFixed(1);
    const alpha= (w * 0.78).toFixed(2);
    return `${xOff(off)}px ${yOff(off)}px ${blur}px rgba(0,0,0,${alpha})`;
  }

  const shadows = [
    shadow(v => v,   _ => '0', w1),    // PC1 → right
    shadow(_ => '0', v => `-${v}`, w2), // PC2 → up
    shadow(v => `-${v}`, _ => '0', w3), // PC3 → left
    shadow(_ => '0', v => v, w4),       // PC4 → down
  ].join(', ');

  $('grid-wrapper').style.boxShadow = shadows;
}

// ── PC BARS ───────────────────────────────────────────────────────────────────
function showPCBars(weights) {
  $('pc-section').style.display = 'block';
  weights.forEach((w, i) => {
    $('bar' + (i+1)).style.width = (w * 100).toFixed(1) + '%';
    $('pct' + (i+1)).textContent = (w * 100).toFixed(1) + '%';
  });
}

// ── INIT ──────────────────────────────────────────────────────────────────────
buildIdleGrid();
pollStatus();
</script>
</body>
</html>"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FASTAPI  — Pydantic models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CoordStringRequest(BaseModel):
    """Used by /api/fetch-patch — accepts a raw coordinate string."""
    coord_str: str


class LatLonRequest(BaseModel):
    """Used by /api/build-grid — expects already-parsed lat/lon."""
    lat: float
    lon: float


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STARTUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _startup_sync() -> None:
    """Blocking startup sequence — runs in the thread executor."""
    print()
    print("━" * 60)
    print("  Sentinel Atlas — startup")
    print("━" * 60)

    _log("1/4  Connecting to Google Earth Engine…")
    _init_gee()

    _log("2/4  Loading DINOv2-small…")
    _init_clip()

    _log("3/4  Connecting to Qdrant…")
    _init_qdrant()

    _log("4/4  Scrolling vectors and fitting PCA…")
    vecs, ids, pays = _scroll_all_vectors()
    _state["db_vectors"]  = vecs
    _state["db_ids"]      = ids
    _state["db_payloads"] = pays
    _state["n_vectors"]   = len(ids)

    if len(ids) >= N_COMPONENTS:
        _fit_pca(vecs)
    else:
        _log(
            f"  ⚠  Only {len(ids)} vectors in DB — need ≥{N_COMPONENTS} for PCA.\n"
            "      Seed the database first (run the notebook collection cell)."
        )

    _state["ready"] = True
    _state["status_msg"] = "Ready"

    print("━" * 60)
    print(f"  ✓ Sentinel Atlas ready  —  {len(ids)} patches indexed")
    print(f"  Open: http://localhost:{PORT}")
    print("━" * 60)
    print()


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _startup_sync)
    yield


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FASTAPI APP + ROUTES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app = FastAPI(title="Sentinel Atlas", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTMLResponse(HTML_TEMPLATE)


@app.get("/api/status")
async def get_status():
    return {
        "ready":       _state["ready"],
        "message":     _state["status_msg"],
        "n_vectors":   _state["n_vectors"],
        "eigenvalues": _state["eigenvalues"],
    }


@app.post("/api/random-coords")
async def random_coords_endpoint():
    """Return a random land coordinate (blocks on GEE — runs in thread)."""
    if not _state["ready"]:
        raise HTTPException(503, "Server still initialising — please wait.")
    loop = asyncio.get_event_loop()
    try:
        lat, lon = await loop.run_in_executor(_executor, random_land_coord)
    except Exception as exc:
        raise HTTPException(500, f"GEE error: {exc}")
    return {"lat": round(lat, 6), "lon": round(lon, 6)}


@app.post("/api/fetch-patch")
async def fetch_patch_endpoint(body: CoordStringRequest):
    """
    Parse coordinates, fetch a Sentinel-2 patch via GEE, embed with CLIP.
    Returns preview image (base64) + metadata. Caches embedding for /build-grid.
    """
    if not _state["ready"]:
        raise HTTPException(503, "Server still initialising — please wait.")

    # Parse coordinates
    try:
        lat, lon = parse_coordinate(body.coord_str)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise HTTPException(400, f"Coordinates out of range: ({lat}, {lon})")

    # Fetch patch (blocking → thread)
    loop = asyncio.get_event_loop()

    def _do_fetch():
        return fetch_patch_with_retry(lat, lon)

    try:
        img, date = await loop.run_in_executor(_executor, _do_fetch)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Unexpected error fetching patch: {exc}")

    # Embed (blocking → thread)
    def _do_embed():
        return embed_images([img])[0]

    emb = await loop.run_in_executor(_executor, _do_embed)

    # Cache
    key = _cache_key(lat, lon)
    _state["patch_cache"][key] = {
        "embedding": emb,
        "date":      date,
        "thumb_b64": pil_to_b64(img, max_px=GRID_THUMB_PX),
    }

    return {
        "lat":         round(lat, 6),
        "lon":         round(lon, 6),
        "date":        date,
        "preview_b64": pil_to_b64(img, max_px=QUERY_PREVIEW_PX),
    }


@app.post("/api/build-grid")
async def build_grid_endpoint(body: LatLonRequest):
    """
    Build the full 17×17 grid for the cached query embedding.
    Returns grid cell data + PC weights for the directional glow.
    """
    if not _state["ready"]:
        raise HTTPException(503, "Server still initialising — please wait.")
    if _state["pca_components"] is None:
        raise HTTPException(422, "PCA not available — database may be too small.")

    key = _cache_key(body.lat, body.lon)
    cached = _state["patch_cache"].get(key)
    if cached is None:
        raise HTTPException(
            400,
            "No cached embedding for these coordinates. "
            "Fetch the patch first using 'Fetch Patch'."
        )

    q_emb    = cached["embedding"]
    date     = cached["date"]
    thumb_b64 = cached["thumb_b64"]

    loop = asyncio.get_event_loop()
    grid, pc_weights = await loop.run_in_executor(
        _executor, lambda: build_grid(q_emb)
    )

    return {
        "grid":       grid,
        "query": {
            "lat":      round(body.lat, 6),
            "lon":      round(body.lon, 6),
            "date":     date,
            "thumb_b64": thumb_b64,
        },
        "pc_weights":  pc_weights,
        "eigenvalues": _state["eigenvalues"],
        "n_db":        _state["n_vectors"],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTRYPOINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    # Brief delay lets uvicorn start before the browser opens
    def _open_browser():
        time.sleep(1.8)
        webbrowser.open(f"http://localhost:{PORT}")

    threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="warning",   # keep console clean — startup logs go via _log()
    )
