"""
api.py
FastAPI application: lifespan, routes, and static file serving.
"""

from __future__ import annotations

import asyncio
import base64
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client.models import QueryRequest
from PIL import Image
import io
from functools import lru_cache

from camera import process_camera_frame
from converter import process_uploaded_image
from config import DATA_DIR, QDRANT_COLLECTION, CAM_PATCH_DIM, CAM_UPDATE_MS
from startup import startup_sync
from state import app_state, executor
import rff

# ── Pydantic models ───────────────────────────────────────────────────────────

class CameraRequest(BaseModel):
    frame_b64: str

class ConvertRequest(BaseModel):
    image_b64: str
    pixels_per_patch: int = 100
    output_scale: int = 1

class Coordinate(BaseModel):
    x: int
    y: int

class GlobeBatchRequest(BaseModel):
    coords: list[Coordinate]

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
app.mount("/images", StaticFiles(directory=os.path.join(DATA_DIR, "images")), name="images")

# ── Page Routes ───────────────────────────────────────────────────────────────

@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/cam")

@app.get("/cam")
async def serve_cam():
    return FileResponse(_FRONTEND_DIR / "cam.html")

@app.get("/convert")
async def serve_convert():
    return FileResponse(_FRONTEND_DIR / "convert.html")

@app.get("/explore")
async def serve_explore():
    return FileResponse(_FRONTEND_DIR / "explore.html")

@app.get("/globe")
async def serve_globe():
    return FileResponse(_FRONTEND_DIR / "globe.html")

# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    return {
        "ready":         app_state.ready,
        "message":       app_state.status_msg,
        "n_vectors":     app_state.n_vectors,
        "cam_patch_dim": CAM_PATCH_DIM,
        "cam_update_ms": CAM_UPDATE_MS,
    }

@app.post("/api/camera-frame")
async def camera_frame_endpoint(request: Request):
    if not app_state.ready:
        raise HTTPException(503, "Server still initialising — please wait.")

    image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(400, "Empty request body.")

    loop = asyncio.get_event_loop()
    try:
        urls, meta = await loop.run_in_executor(
            executor, lambda: process_camera_frame(image_bytes)
        )
    except Exception as exc:
        raise HTTPException(500, f"Error processing frame: {exc}")

    return {"urls": urls, "meta": meta}

@app.post("/api/convert-image")
async def convert_image_endpoint(body: ConvertRequest):
    if not app_state.ready:
        raise HTTPException(503, "Server still initialising — please wait.")

    try:
        image_bytes = base64.b64decode(body.image_b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 image data.")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            executor,
            lambda: process_uploaded_image(
                image_bytes, body.pixels_per_patch, body.output_scale
            )
        )
    except Exception as exc:
        raise HTTPException(500, f"Error converting image: {exc}")

    return result

@app.post("/api/globe-patches")
async def globe_patches_endpoint(body: GlobeBatchRequest):
    """
    Receives an array of (x,y) coordinates, computes RFF mapped vectors,
    and returns a mapping of coordinate strings "x_y" to image URLs.
    """
    if not app_state.ready:
        raise HTTPException(503, "Server still initialising — please wait.")

    if not body.coords:
        return {}

    mapper = rff.get_mapper()
    points = [(c.x, c.y) for c in body.coords]
    
    # 1. Map Coordinates -> R^N Latent Vectors
    vectors = mapper.map_points(points)
    
    # 2. Build Batch Query
    search_requests = []
    for vec in vectors:
        search_requests.append(QueryRequest(
            query=vec.tolist(),
            limit=1,
            with_payload=True,
        ))

    # 3. Search Qdrant
    loop = asyncio.get_event_loop()
    try:
        batch_results = await loop.run_in_executor(
            executor,
            lambda: app_state.qdrant.query_batch_points(
                collection_name=QDRANT_COLLECTION,
                requests=search_requests
            )
        )
    except Exception as exc:
        raise HTTPException(500, f"Error querying Qdrant: {exc}")

    # 4. Map Results
    res = {}
    for i, coord in enumerate(body.coords):
        key = f"{coord.x}_{coord.y}"
        pts = batch_results[i].points
        if pts:
            filename = pts[0].payload.get("filename", "")
            res[key] = f"/images/{filename}" if filename else None
        else:
            res[key] = None

    return res

# ── Thumbnail API ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=2048)
def _get_cached_thumb(img_path: str, size: int = 64) -> bytes | None:
    """Reads, resizes, and encodes an image to JPEG."""
    if not os.path.exists(img_path):
        return None
    try:
        with Image.open(img_path) as img:
            img.thumbnail((size, size))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
    except Exception as e:
        print(f"Error generating thumbnail for {img_path}: {e}")
        return None

@app.get("/api/thumb/{filename}")
async def get_thumbnail(filename: str):
    """Returns a small JPEG version of the requested satellite image."""
    img_path = os.path.join(DATA_DIR, "images", filename)
    
    # We use run_in_executor for the blocking disk I/O and PIL operations
    loop = asyncio.get_event_loop()
    thumb_bytes = await loop.run_in_executor(
        executor, lambda: _get_cached_thumb(img_path)
    )

    if thumb_bytes is None:
        raise HTTPException(404, "Image not found or could not be processed.")

    return Response(content=thumb_bytes, media_type="image/jpeg")