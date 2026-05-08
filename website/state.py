"""
state.py
Global application state as a typed dataclass.

All modules import `app_state` and `executor` from here rather than
maintaining their own globals, which makes the coupling explicit and testable.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


@dataclass
class AppState:
    ready: bool = False
    status_msg: str = "Initialising…"
    n_vectors: int = 0

    # Qdrant client (set during startup)
    qdrant: Any = None


# Singletons used across the whole application
app_state = AppState()
executor = ThreadPoolExecutor(max_workers=6)


def log(msg: str) -> None:
    """Print a status message and update the status pill text."""
    print(msg, flush=True)
    app_state.status_msg = msg
