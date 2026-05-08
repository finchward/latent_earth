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

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from camera import process_camera_frame
from converter import process_uploaded_image
from config import DATA_DIR
from startup import startup_sync
from state import app_state, executor

# ── Pydantic models ───────────────────────────────────────────────────────────


class CameraRequest(BaseModel):
    """Used by /api/camera-frame to process webcam segments."""
    frame_b64: str


class ConvertRequest(BaseModel):
    """Used by /api/convert-image to process uploaded images."""
    image_b64: str
    pixels_per_patch: int = 100
    output_scale: int = 1


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


# ── API Routes ────────────────────────────────────────────────────────────────


@app.get("/api/status")
async def get_status():
    return {
        "ready":     app_state.ready,
        "message":   app_state.status_msg,
        "n_vectors": app_state.n_vectors,
    }


@app.post("/api/camera-frame")
async def camera_frame_endpoint(body: CameraRequest):
    """
    Receive an 800×800 base64 frame, slice into 8×8 patches,
    compute vectors, and batch-search Qdrant for matches.
    """
    if not app_state.ready:
        raise HTTPException(503, "Server still initialising — please wait.")

    print(f"[API] Received camera frame ({len(body.frame_b64)} chars)", flush=True)

    loop = asyncio.get_event_loop()
    try:
        urls = await loop.run_in_executor(
            executor, lambda: process_camera_frame(body.frame_b64)
        )
        print(f"[API] Frame processed, returning {len(urls)}×{len(urls[0])} grid.", flush=True)
    except Exception as exc:
        print(f"[API] Error processing frame: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Error processing frame: {exc}")

    return {"urls": urls}


@app.post("/api/convert-image")
async def convert_image_endpoint(body: ConvertRequest):
    """
    Receive a base64 image + settings, produce a satellite mosaic PNG.
    """
    if not app_state.ready:
        raise HTTPException(503, "Server still initialising — please wait.")

    try:
        image_bytes = base64.b64decode(body.image_b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 image data.")

    print(f"[API] Convert request: ppp={body.pixels_per_patch}, "
          f"scale={body.output_scale}", flush=True)

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            executor,
            lambda: process_uploaded_image(
                image_bytes, body.pixels_per_patch, body.output_scale
            )
        )
    except Exception as exc:
        print(f"[API] Error converting image: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Error converting image: {exc}")

    return result
