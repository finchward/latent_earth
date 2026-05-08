"""
embedder.py
Feature computation for camera frame patches (HOG or DINOv2).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from config import EMBEDDING_METHOD

# Globals for DINOv2 lazy loading
_PROCESSOR = None
_MODEL = None
_DEVICE = "cpu"

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

def compute_hog_features(images: list[Image.Image]) -> np.ndarray:
    """
    Compute normalised HOG feature vectors for a list of images.
    """
    from skimage.feature import hog
    
    all_features = []
    for img in images:
        img_gray = img.convert("L")
        img_resized = img_gray.resize((128, 128))
        img_arr = np.array(img_resized)

        features = hog(
            img_arr,
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            feature_vector=True
        )

        # Compute overall standard deviation to capture true global edge strength/contrast
        grad_mag = np.std(img_arr.astype(np.float32))
        
        # Scale HOG vector by the gradient magnitude
        features = features * grad_mag
            
        all_features.append(features.astype(np.float32))
        
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
    
    Using LAB space ensures that Euclidean distance corresponds to perceptual 
    colour difference (Delta E).
    """
    from skimage.color import rgb2lab
    all_features = []
    for img in images:
        img_rgb = img.convert("RGB")
        img_small = img_rgb.resize((8, 8), Image.LANCZOS)
        arr_rgb = np.array(img_small, dtype=np.float32) / 255.0
        
        # Convert to LAB space
        arr_lab = rgb2lab(arr_rgb)
        
        # Flatten to (192,) vector
        features = arr_lab.flatten().astype(np.float32)
        all_features.append(features)
    
    return np.array(all_features, dtype=np.float32)

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

def compute_features(images: list[Image.Image]):
    """
    Unified entry point that computes features based on config.EMBEDDING_METHOD.
    Returns np.ndarray for single-vector methods, or list[dict] for hybrid.
    """
    if EMBEDDING_METHOD == "hog":
        return compute_hog_features(images)
    elif EMBEDDING_METHOD == "dinov2":
        return compute_dinov2_features(images)
    elif EMBEDDING_METHOD == "hog_hybrid":
        return compute_hog_hybrid_features(images)
    else:
        raise ValueError(f"Unknown EMBEDDING_METHOD: {EMBEDDING_METHOD}")
