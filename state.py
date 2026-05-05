"""
state.py
Global application state as a typed dataclass.

All modules import `app_state` and `executor` from here rather than
maintaining their own globals, which makes the coupling explicit and testable.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class AppState:
    ready: bool = False
    status_msg: str = "Initialising…"
    n_vectors: int = 0

    # PCA outputs
    eigenvalues: list[float] = field(default_factory=list)
    pca_components: Optional[np.ndarray] = None   # shape (N_COMPONENTS, VECTOR_SIZE)

    # Database mirror (loaded once at startup)
    db_vectors: Optional[np.ndarray] = None        # shape (N, VECTOR_SIZE)
    db_ids: Optional[list] = None
    db_payloads: Optional[list[dict]] = None

    # ML models (set during startup)
    qdrant: Any = None
    clip_model: Any = None
    clip_proc: Any = None
    device: Optional[str] = None

    # Per-request embedding cache, keyed by (lat, lon) rounded to 6 dp
    patch_cache: dict = field(default_factory=dict)


# Singletons used across the whole application
app_state = AppState()
executor = ThreadPoolExecutor(max_workers=6)


def log(msg: str) -> None:
    """Print a status message and update the status pill text."""
    print(msg, flush=True)
    app_state.status_msg = msg
