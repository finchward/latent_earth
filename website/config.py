"""
config.py
All tunable constants for Sentinel Atlas.
Override via environment variables where noted.
"""

import os

# ── Data Architecture ─────────────────────────────────────────────────────────

QDRANT_HOST: str = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_URL: str = os.environ.get("QDRANT_URL", f"http://{QDRANT_HOST}:{QDRANT_PORT}")

# Set this to True to use Docker/Server mode instead of Local Path mode
QDRANT_PREFER_SERVER: bool = os.environ.get("QDRANT_PREFER_SERVER", "true").lower() == "true"

DATA_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_collection", "data"))

EMBEDDING_METHOD: str = os.environ.get("EMBEDDING_METHOD", "fused_hybrid_1_3")  # Set to "hog" or "dinov2" or "fused_hybrid"

if EMBEDDING_METHOD == "hog":
    QDRANT_LOCAL_PATH: str = os.path.join(DATA_DIR, "qdrant_hog")
    QDRANT_COLLECTION: str = "satellite_patches_hog"
    VECTOR_SIZE: int = 1568  # 7x7 blocks * 2x2 cells * 8 orientations
    PCA_ENABLED: bool = False
    PCA_MODEL_PATH: str = ""
elif EMBEDDING_METHOD == "dinov2":
    QDRANT_LOCAL_PATH: str = os.path.join(DATA_DIR, "qdrant_dinov2")
    QDRANT_COLLECTION: str = "satellite_patches_dinov2"
    VECTOR_SIZE: int = 384
    PCA_ENABLED: bool = False
    PCA_MODEL_PATH: str = ""
elif EMBEDDING_METHOD == "hog_hybrid":
    QDRANT_LOCAL_PATH: str = os.path.join(DATA_DIR, "qdrant_hog_hybrid")
    QDRANT_COLLECTION: str = "satellite_patches_hog_hybrid"
    HOG_VECTOR_SIZE: int = 1568
    COLOUR_VECTOR_SIZE: int = 192  # 8x8 grid * 3 channels (R, G, B)
    VECTOR_SIZE: int = 0  # Not used directly — named vectors instead
elif EMBEDDING_METHOD == "fused_hybrid":
    QDRANT_LOCAL_PATH: str = os.path.join(DATA_DIR, "qdrant_fused_hybrid")
    QDRANT_COLLECTION: str = "satellite_patches_fused_hybrid"
    HOG_VECTOR_SIZE: int = 1568
    COLOUR_VECTOR_SIZE: int = 192
    # Fusion weights applied to HOG and Colour vectors.
    # 0.5 and 0.866 (sqrt(0.75)) gives a 1:3 ratio of squared influence.
    FUSION_HOG_WEIGHT: float = float(os.environ.get("FUSION_HOG_WEIGHT", 0.577))
    FUSION_COLOUR_WEIGHT: float = float(os.environ.get("FUSION_COLOUR_WEIGHT", 0.816))
    # PCA dimensionality reduction (optional)
    PCA_ENABLED: bool = os.environ.get("PCA_ENABLED", "true").lower() == "true"
    PCA_DIM: int = int(os.environ.get("PCA_DIM", 256))
    PCA_MODEL_PATH: str = os.environ.get(
        "PCA_MODEL_PATH", os.path.join(DATA_DIR, "pca_fused_hybrid.joblib")
    )
    # Final Qdrant vector size depends on whether PCA is applied
    VECTOR_SIZE: int = PCA_DIM if PCA_ENABLED else (HOG_VECTOR_SIZE + COLOUR_VECTOR_SIZE)
elif EMBEDDING_METHOD == "fused_hybrid_1_3":
    QDRANT_LOCAL_PATH: str = os.path.join(DATA_DIR, "qdrant_fused_hybrid_1_3")
    QDRANT_COLLECTION: str = "satellite_patches_fused_hybrid_1_3"
    HOG_VECTOR_SIZE: int = 1568
    COLOUR_VECTOR_SIZE: int = 192
    # Fusion weights applied to HOG and Colour vectors.
    # 0.5 and 0.866 (sqrt(0.75)) gives a 1:3 ratio of squared influence.
    FUSION_HOG_WEIGHT: float = float(os.environ.get("FUSION_HOG_WEIGHT", 0.5))
    FUSION_COLOUR_WEIGHT: float = float(os.environ.get("FUSION_COLOUR_WEIGHT", 0.866))
    # PCA dimensionality reduction (optional)
    PCA_ENABLED: bool = os.environ.get("PCA_ENABLED", "true").lower() == "true"
    PCA_DIM: int = int(os.environ.get("PCA_DIM", 256))
    PCA_MODEL_PATH: str = os.environ.get(
        "PCA_MODEL_PATH", os.path.join(DATA_DIR, "pca_fused_hybrid_1_3.joblib")
    )
    # Final Qdrant vector size depends on whether PCA is applied
    VECTOR_SIZE: int = PCA_DIM if PCA_ENABLED else (HOG_VECTOR_SIZE + COLOUR_VECTOR_SIZE)


else:
    raise ValueError(f"Unknown EMBEDDING_METHOD: {EMBEDDING_METHOD}")

# ── Server ────────────────────────────────────────────────────────────────────

PORT: int = int(os.environ.get("PORT", 7432))

# ── Camera ────────────────────────────────────────────────────────────────────

# Number of patches per side for the camera mosaic (e.g. 8 for an 8x8 grid)
CAM_PATCH_DIM: int = int(os.environ.get("CAM_PATCH_DIM", 12))

# Maximum update frequency for camera processing in milliseconds
CAM_UPDATE_MS: int = int(os.environ.get("CAM_UPDATE_MS", 1000))
