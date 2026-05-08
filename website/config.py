"""
config.py
All tunable constants for Sentinel Atlas.
Override via environment variables where noted.
"""

import os

# ── Data Architecture ─────────────────────────────────────────────────────────

DATA_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_collection", "data"))

EMBEDDING_METHOD: str = os.environ.get("EMBEDDING_METHOD", "hog_hybrid")  # Set to "hog" or "dinov2"

if EMBEDDING_METHOD == "hog":
    QDRANT_LOCAL_PATH: str = os.path.join(DATA_DIR, "qdrant_hog")
    QDRANT_COLLECTION: str = "satellite_patches_hog"
    VECTOR_SIZE: int = 1568  # 7x7 blocks * 2x2 cells * 8 orientations
elif EMBEDDING_METHOD == "dinov2":
    QDRANT_LOCAL_PATH: str = os.path.join(DATA_DIR, "qdrant_dinov2")
    QDRANT_COLLECTION: str = "satellite_patches_dinov2"
    VECTOR_SIZE: int = 384
elif EMBEDDING_METHOD == "hog_hybrid":
    QDRANT_LOCAL_PATH: str = os.path.join(DATA_DIR, "qdrant_hog_hybrid")
    QDRANT_COLLECTION: str = "satellite_patches_hog_hybrid"
    HOG_VECTOR_SIZE: int = 1568
    COLOUR_VECTOR_SIZE: int = 192  # 8x8 grid * 3 channels (R, G, B)
    VECTOR_SIZE: int = 0  # Not used directly — named vectors instead
else:
    raise ValueError(f"Unknown EMBEDDING_METHOD: {EMBEDDING_METHOD}")

# ── Server ────────────────────────────────────────────────────────────────────

PORT: int = int(os.environ.get("PORT", 7432))
