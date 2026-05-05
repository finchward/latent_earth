"""
embedder.py
DINOv2-small model initialisation and image embedding.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from config import EMBED_BATCH_SIZE
from state import app_state, log


def init_clip() -> None:
    """Load DINOv2-small onto the best available device and store in app_state."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  Loading DINOv2-small on {device}…")
    model     = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model.eval()
    app_state.clip_model = model
    app_state.clip_proc  = processor
    app_state.device     = device
    log("  ✓ DINOv2 ready")


def embed_images(images: list[Image.Image]) -> np.ndarray:
    """
    Return L2-normalised DINOv2 CLS-token embeddings.
    Shape: (N, 384) float32.
    """
    model  = app_state.clip_model
    proc   = app_state.clip_proc
    device = app_state.device

    all_emb: list[np.ndarray] = []
    for i in range(0, len(images), EMBED_BATCH_SIZE):
        batch  = [img.convert("RGB") for img in images[i : i + EMBED_BATCH_SIZE]]
        inputs = proc(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs  = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]
        features = features / features.norm(dim=-1, keepdim=True)
        all_emb.append(features.cpu().numpy())
    return np.vstack(all_emb).astype(np.float32)
