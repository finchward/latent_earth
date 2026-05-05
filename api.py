"""
api.py
FastAPI application: lifespan, Pydantic request models, and all route handlers.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import GRID_THUMB_PX, QUERY_PREVIEW_PX
from embedder import embed_images
from gee import fetch_patch_with_retry, random_land_coord
from grid import build_grid
from startup import startup_sync
from state import app_state, executor
from utils import cache_key, parse_coordinate, pil_to_b64

# ── Pydantic models ───────────────────────────────────────────────────────────


class CoordStringRequest(BaseModel):
    """Used by /api/fetch-patch — accepts a raw coordinate string."""
    coord_str: str


class LatLonRequest(BaseModel):
    """Used by /api/build-grid — expects already-parsed lat/lon."""
    lat: float
    lon: float


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, startup_sync)
    yield


# ── App ───────────────────────────────────────────────────────────────────────

_FRONTEND_DIR = Path(__file__).parent / "frontend"

app = FastAPI(title="Sentinel Atlas", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/")
async def serve_ui():
    return FileResponse(_FRONTEND_DIR / "index.html")


@app.get("/api/status")
async def get_status():
    return {
        "ready":       app_state.ready,
        "message":     app_state.status_msg,
        "n_vectors":   app_state.n_vectors,
        "eigenvalues": app_state.eigenvalues,
    }


@app.post("/api/random-coords")
async def random_coords_endpoint():
    """Return a random land coordinate (blocks on GEE — runs in executor)."""
    if not app_state.ready:
        raise HTTPException(503, "Server still initialising — please wait.")
    loop = asyncio.get_event_loop()
    try:
        lat, lon = await loop.run_in_executor(executor, random_land_coord)
    except Exception as exc:
        raise HTTPException(500, f"GEE error: {exc}")
    return {"lat": round(lat, 6), "lon": round(lon, 6)}


@app.post("/api/fetch-patch")
async def fetch_patch_endpoint(body: CoordStringRequest):
    """
    Parse coordinates, fetch a Sentinel-2 patch via GEE, embed with DINOv2.
    Returns preview image (base64) + metadata. Caches embedding for /build-grid.
    """
    if not app_state.ready:
        raise HTTPException(503, "Server still initialising — please wait.")

    try:
        lat, lon = parse_coordinate(body.coord_str)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise HTTPException(400, f"Coordinates out of range: ({lat}, {lon})")

    loop = asyncio.get_event_loop()

    try:
        img, date = await loop.run_in_executor(
            executor, lambda: fetch_patch_with_retry(lat, lon)
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Unexpected error fetching patch: {exc}")

    emb = await loop.run_in_executor(executor, lambda: embed_images([img])[0])

    key = cache_key(lat, lon)
    app_state.patch_cache[key] = {
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
    Build the 17×17 grid for the cached query embedding.
    Returns grid cell data + PC weights for the directional glow.
    """
    if not app_state.ready:
        raise HTTPException(503, "Server still initialising — please wait.")
    if app_state.pca_components is None:
        raise HTTPException(422, "PCA not available — database may be too small.")

    key    = cache_key(body.lat, body.lon)
    cached = app_state.patch_cache.get(key)
    if cached is None:
        raise HTTPException(
            400,
            "No cached embedding for these coordinates. "
            "Fetch the patch first using 'Fetch Patch'."
        )

    q_emb     = cached["embedding"]
    date      = cached["date"]
    thumb_b64 = cached["thumb_b64"]

    loop = asyncio.get_event_loop()
    grid, pc_weights = await loop.run_in_executor(
        executor, lambda: build_grid(q_emb)
    )

    return {
        "grid":       grid,
        "query": {
            "lat":       round(body.lat, 6),
            "lon":       round(body.lon, 6),
            "date":      date,
            "thumb_b64": thumb_b64,
        },
        "pc_weights":  pc_weights,
        "eigenvalues": app_state.eigenvalues,
        "n_db":        app_state.n_vectors,
    }
