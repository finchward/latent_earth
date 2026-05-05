"""
config.py
All tunable constants for Sentinel Atlas.
Override via environment variables where noted.
"""

import glob
import os

# ── Google Earth Engine ───────────────────────────────────────────────────────

GEE_PROJECT_ID: str = os.environ.get("GEE_PROJECT_ID", "mercari-agent")

# ── Qdrant / Google Drive ─────────────────────────────────────────────────────

_DRIVE_GLOBS = [
    "~/Library/CloudStorage/GoogleDrive-*/My Drive/qdrant_satellite_storage",
    "~/Google Drive/My Drive/qdrant_satellite_storage",
    "~/GoogleDrive/My Drive/qdrant_satellite_storage",
    "~/google-drive/qdrant_satellite_storage",
]
_auto_drive = next(
    (m for p in _DRIVE_GLOBS for m in glob.glob(os.path.expanduser(p))), None
)

QDRANT_DRIVE_PATH: str = os.environ.get(
    "QDRANT_DRIVE_PATH",
    _auto_drive or r"G:\My Drive\qdrant_satellite_storage",
)
QDRANT_LOCAL_PATH: str = "/tmp/satellite_explorer_qdrant"
QDRANT_COLLECTION: str = "satellite_patches"

# ── Sentinel-2 imagery ────────────────────────────────────────────────────────

PATCH_SIZE_M: int      = 5_000
IMAGE_SIZE_PX: int     = 500
GEE_COLLECTION: str    = "COPERNICUS/S2_SR_HARMONIZED"
BANDS: list[str]       = ["B4", "B3", "B2"]
MAX_CLOUD_PCT: int     = 20
MIN_BRIGHTNESS: float  = 30.0
SENTINEL_START: str    = "2017-03-28"
WINDOW_DAYS: int       = 600

# ── Embedding ─────────────────────────────────────────────────────────────────

VECTOR_SIZE: int      = 384
EMBED_BATCH_SIZE: int = 16

# ── PCA ───────────────────────────────────────────────────────────────────────

MAX_PCA_VECTORS: int = 10_000
N_COMPONENTS: int    = 4

# ── Display ───────────────────────────────────────────────────────────────────

GRID_THUMB_PX: int    = 60
QUERY_PREVIEW_PX: int = 240

# ── Server ────────────────────────────────────────────────────────────────────

PORT: int = int(os.environ.get("PORT", 7432))
