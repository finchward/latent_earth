"""
grid.py
Build the 17×17 PCA-structured patch grid for a given query embedding.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
from PIL import Image

from config import GRID_THUMB_PX
from state import app_state


# ── Internal helpers ──────────────────────────────────────────────────────────

def _thumb_from_payload(payload: dict, size: int = GRID_THUMB_PX) -> str:
    """Decode payload image_b64, resize to size×size, return base64 PNG."""
    raw = base64.b64decode(payload["image_b64"])
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Public API ────────────────────────────────────────────────────────────────

def build_grid(
    query_vec: np.ndarray,
) -> tuple[list[list[Optional[dict]]], list[float]]:
    """
    Assign database patches to a 17×17 grid centred on the query vector.

    Grid orientation:
      Right  (+col) ← PC1 closest
      Up     (−row) ← PC2 closest
      Left   (−col) ← PC3 closest
      Down   (+row) ← PC4 closest
      Diagonals     ← blended eigenvectors, greedy centre-first

    Returns:
        grid       – 17×17 list-of-lists; each cell is a dict or None
        pc_weights – [w1,w2,w3,w4] summing to 1.0 (absolute PC projections)
    """
    db_vec  = app_state.db_vectors
    db_ids  = app_state.db_ids
    db_pay  = app_state.db_payloads
    comps   = app_state.pca_components

    N = len(db_ids) if db_ids else 0
    empty_grid: list[list[Optional[dict]]] = [[None] * 17 for _ in range(17)]

    if N == 0 or comps is None:
        return empty_grid, [0.25, 0.25, 0.25, 0.25]

    e1, e2, e3, e4 = comps

    # ── Projections ────────────────────────────────────────────────────────────
    proj   = db_vec @ comps.T                            # (N, 4)
    q_proj = (query_vec @ comps.T).astype(np.float64)   # (4,)

    abs_q     = np.abs(q_proj)
    total_abs = abs_q.sum()
    pc_weights = (abs_q / total_abs).tolist() if total_abs > 0 else [0.25] * 4

    # ── Helpers ────────────────────────────────────────────────────────────────
    grid     = [[None] * 17 for _ in range(17)]
    assigned: set[int] = set()

    def make_cell(db_idx: int, axis: str) -> dict:
        pay = db_pay[db_idx]
        return {
            "type":      "patch",
            "thumb_b64": _thumb_from_payload(pay),
            "lat":       pay.get("lat"),
            "lon":       pay.get("lon"),
            "date":      pay.get("date", ""),
            "axis":      axis,
        }

    def pick_closest_axis(axis_k: int, n: int = 8) -> list[int]:
        diffs = np.abs(proj[:, axis_k] - q_proj[axis_k])
        order = np.argsort(diffs)
        result = []
        for idx in order:
            if int(idx) not in assigned and len(result) < n:
                result.append(int(idx))
        return result

    # ── Axis cells (highest priority) ─────────────────────────────────────────
    for i, idx in enumerate(pick_closest_axis(0)):   # RIGHT — PC1
        grid[8][9 + i] = make_cell(idx, "PC1")
        assigned.add(idx)

    for i, idx in enumerate(pick_closest_axis(1)):   # UP — PC2
        grid[7 - i][8] = make_cell(idx, "PC2")
        assigned.add(idx)

    for i, idx in enumerate(pick_closest_axis(2)):   # LEFT — PC3
        grid[8][7 - i] = make_cell(idx, "PC3")
        assigned.add(idx)

    for i, idx in enumerate(pick_closest_axis(3)):   # DOWN — PC4
        grid[9 + i][8] = make_cell(idx, "PC4")
        assigned.add(idx)

    # ── Diagonal cells (greedy, centre-outward) ────────────────────────────────
    diag_offsets = sorted(
        [(dr, dc)
         for dr in range(-8, 9)
         for dc in range(-8, 9)
         if dr != 0 and dc != 0],
        key=lambda rc: rc[0] ** 2 + rc[1] ** 2,
    )

    avail = np.ones(N, dtype=bool)
    for idx in assigned:
        avail[idx] = False

    for dr, dc in diag_offsets:
        row, col = 8 + dr, 8 + dc

        # Blend the two adjacent axis eigenvectors weighted by grid distance
        if   dc > 0 and dr < 0:  d = float(dc)  * e1 + float(-dr) * e2
        elif dc > 0 and dr > 0:  d = float(dc)  * e1 + float(dr)  * e4
        elif dc < 0 and dr > 0:  d = float(-dc) * e3 + float(dr)  * e4
        else:                    d = float(-dc)  * e3 + float(-dr) * e2

        norm = float(np.linalg.norm(d))
        if norm < 1e-9:
            continue
        d = (d / norm).astype(np.float32)

        q_score = float(query_vec @ d)
        scores  = db_vec @ d
        diffs_d = np.abs(scores - q_score)
        diffs_d[~avail] = np.inf

        best = int(np.argmin(diffs_d))
        if not np.isinf(diffs_d[best]):
            grid[row][col] = make_cell(best, "diag")
            avail[best]    = False
            assigned.add(best)

    return grid, pc_weights
