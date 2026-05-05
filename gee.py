"""
gee.py
Google Earth Engine helpers: initialisation, land-mask, and Sentinel-2 patch fetching.
"""

from __future__ import annotations

import datetime
import io
import math
import random
from typing import Optional

import ee
import numpy as np
import requests as http_requests
from PIL import Image

from config import (
    BANDS,
    GEE_COLLECTION,
    GEE_PROJECT_ID,
    IMAGE_SIZE_PX,
    MAX_CLOUD_PCT,
    MIN_BRIGHTNESS,
    PATCH_SIZE_M,
    SENTINEL_START,
    WINDOW_DAYS,
)
from state import log

# ── Module-level derived constants ────────────────────────────────────────────

_SENTINEL_START_DATE = datetime.date.fromisoformat(SENTINEL_START)
_TOTAL_DAYS = (datetime.date.today() - _SENTINEL_START_DATE).days

# Loaded once after GEE is initialised
_land_image = None


# ── Initialisation ────────────────────────────────────────────────────────────

def init_gee() -> None:
    global _land_image
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT_ID)

    _land_image = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        .select("occurrence")
        .unmask(100)
        .lt(50)
    )
    log(f"  ✓ GEE initialised  ({GEE_PROJECT_ID})")


# ── Land mask ─────────────────────────────────────────────────────────────────

def is_land(lat: float, lon: float) -> bool:
    point = ee.Geometry.Point([lon, lat])
    val   = _land_image.sample(region=point, scale=500).first().getInfo()
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


# ── Patch fetching ────────────────────────────────────────────────────────────

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

        if float(np.array(img, dtype=np.float32).mean()) < MIN_BRIGHTNESS:
            return None
        return img, date_str

    except Exception:
        return None


def fetch_patch_with_retry(
    lat: float, lon: float, max_attempts: int = 30
) -> tuple[Image.Image, str]:
    """Retry fetch_patch up to max_attempts times, raising ValueError on exhaustion."""
    for _ in range(max_attempts):
        result = fetch_patch(lat, lon)
        if result is not None:
            return result
    raise ValueError(
        f"No usable Sentinel-2 imagery found at ({lat:.5f}, {lon:.5f}) "
        f"after {max_attempts} attempts. "
        "The location may be persistently cloudy or have no archive coverage."
    )
