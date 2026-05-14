"""
embedder.py
Feature computation for camera frame patches (HOG or DINOv2).
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from skimage.feature import hog
from skimage.color import rgb2lab
import joblib

import config
from config import EMBEDDING_METHOD

# Globals for DINOv2 lazy loading
_PROCESSOR = None
_MODEL = None
_DEVICE = "cpu"

# Global PCA model cache (fused_hybrid only)
_PCA_MODEL = None

def _init_dinov2():
    global _PROCESSOR, _MODEL, _DEVICE
    if _MODEL is not None:
        return
        
    import torch
    from transformers import AutoImageProcessor, AutoModel
    
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "facebook/dinov2-small"
    
    print(f"[Embedder] Loading {MODEL_NAME} on {_DEVICE}...")
    _PROCESSOR = AutoImageProcessor.from_pretrained(MODEL_NAME)
    _MODEL = AutoModel.from_pretrained(MODEL_NAME).to(_DEVICE)
    _MODEL.eval()

def _compute_single_hog(img: Image.Image) -> np.ndarray:
    """Helper for parallel HOG computation."""
    img_gray = img.convert("L")
    img_resized = img_gray.resize((64, 64))
    img_arr = np.array(img_resized)

    features = hog(
        img_arr,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )

    # Compute overall standard deviation to capture true global edge strength/contrast
    grad_mag = np.std(img_arr.astype(np.float32))
    
    # Scale HOG vector by the gradient magnitude
    return (features * grad_mag).astype(np.float32)

def compute_hog_features(images: list[Image.Image]) -> np.ndarray:
    """
    Compute normalised HOG feature vectors for a list of images in parallel.
    """
    with ThreadPoolExecutor() as executor:
        all_features = list(executor.map(_compute_single_hog, images))
        
    return np.array(all_features)

def compute_dinov2_features(images: list[Image.Image]) -> np.ndarray:
    """
    Compute normalised DINOv2 feature vectors for a list of images.
    Returns an array of shape (N, 384).
    """
    _init_dinov2()
    import torch
    
    inputs = _PROCESSOR(images=images, return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        outputs = _MODEL(**inputs)
        features = outputs.last_hidden_state[:, 0, :]
        features = features / features.norm(dim=-1, keepdim=True)
        embeddings = features.cpu().numpy().astype(np.float32)
    return embeddings

def compute_colour_features(images: list[Image.Image]) -> np.ndarray:
    """
    Compute spatial colour descriptors in LAB space by resizing to 8x8 and flattening.
    Uses vectorized conversion for speed.
    """
    # 1. Resize all images to 8x8 using fast Bilinear filtering
    # and stack into a single (N, 8, 8, 3) array.
    batch_rgb = np.zeros((len(images), 8, 8, 3), dtype=np.float32)
    for i, img in enumerate(images):
        img_small = img.convert("RGB").resize((8, 8), Image.BILINEAR)
        batch_rgb[i] = np.array(img_small, dtype=np.float32) / 255.0
    
    # 2. Vectorized LAB conversion (much faster than looping)
    batch_lab = rgb2lab(batch_rgb)
    
    # 3. Flatten to (N, 192) vectors
    return batch_lab.reshape(len(images), -1).astype(np.float32)

def compute_hog_hybrid_features(images: list[Image.Image]) -> list[dict]:
    """
    Compute both HOG and colour features for each image.
    Returns a list of dicts: [{"hog": [...], "colour": [...]}, ...]
    """
    hog_vecs = compute_hog_features(images)
    colour_vecs = compute_colour_features(images)
    
    return [
        {"hog": hog_vecs[i].tolist(), "colour": colour_vecs[i].tolist()}
        for i in range(len(images))
    ]

def _load_pca_model():
    """Lazily load and cache the PCA model from disk (fused_hybrid only)."""
    global _PCA_MODEL
    if _PCA_MODEL is not None:
        return _PCA_MODEL
    
    print(f"[Embedder] Loading PCA model from {config.PCA_MODEL_PATH}...", flush=True)
    _PCA_MODEL = joblib.load(config.PCA_MODEL_PATH)
    print(f"[Embedder] PCA model loaded (→ {config.PCA_DIM} dims).", flush=True)
    return _PCA_MODEL


def compute_fused_hybrid_features(images: list[Image.Image]) -> np.ndarray:
    """
    Early Fusion: concatenate HOG and Colour vectors with configured weights
    (0.5 and 0.866 for 1:3 influence), then optionally reduce to PCA_DIM.
    No independent normalization is applied to preserve edge strength, 
    allowing Euclidean distance to reflect combined descriptor differences.
    Returns a 2-D np.ndarray of shape (N, VECTOR_SIZE).
    """
    import config

    hog_vecs = compute_hog_features(images)      # (N, 1568) float32
    colour_vecs = compute_colour_features(images) # (N, 192)  float32

    # --- Weighted fusion & concatenation -------------------------------------
    # We NO LONGER L2-normalise here, as we want to preserve the 'edge strength'
    # (magnitude) of the HOG features to distinguish flat vs textured patches.
    fused = np.concatenate(
        [
            config.FUSION_HOG_WEIGHT * hog_vecs,
            config.FUSION_COLOUR_WEIGHT * colour_vecs,
        ],
        axis=1,
    ).astype(np.float32)  # (N, 1760)

    # --- Optional PCA reduction ----------------------------------------------
    if config.PCA_ENABLED:
        pca = _load_pca_model()
        fused = pca.transform(fused).astype(np.float32)  # (N, PCA_DIM)

    return fused


def compute_features(images: list[Image.Image]):
    """
    Unified entry point that computes features based on config.EMBEDDING_METHOD.
    Returns np.ndarray for single-vector methods, or list[dict] for hog_hybrid.
    """
    if EMBEDDING_METHOD == "hog":
        return compute_hog_features(images)
    elif EMBEDDING_METHOD == "dinov2":
        return compute_dinov2_features(images)
    elif EMBEDDING_METHOD == "hog_hybrid":
        return compute_hog_hybrid_features(images)
    elif EMBEDDING_METHOD == "fused_hybrid" or EMBEDDING_METHOD == "fused_hybrid_1_3":
        return compute_fused_hybrid_features(images)
    else:
        raise ValueError(f"Unknown EMBEDDING_METHOD: {EMBEDDING_METHOD}")
