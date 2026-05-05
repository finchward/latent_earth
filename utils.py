"""
utils.py
Pure helper functions — no imports from other internal modules.
"""

from __future__ import annotations

import base64
import io
import re
from typing import Optional

from PIL import Image


def pil_to_b64(img: Image.Image, max_px: Optional[int] = None) -> str:
    """Encode a PIL image as a base64 PNG string, optionally scaling it first."""
    if max_px:
        img = img.copy()
        img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def parse_coordinate(text: str) -> tuple[float, float]:
    """
    Parse latitude and longitude from a string.

    Supported formats:
      • Decimal:  -33.8688, 151.2093
      • DMS:      33°52′12″S, 151°12′36″E
    """
    text = text.strip()

    # Decimal degrees (comma or space separator)
    m = re.match(r'^([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)$', text)
    if m:
        return float(m.group(1)), float(m.group(2))

    # DMS — accepts °′″ or plain '" with optional spaces
    dms = re.search(
        r'(\d+)\s*[°]\s*(\d+)\s*[\'′]\s*(\d+(?:\.\d+)?)\s*[\"″]?\s*([NSns])'
        r'[,\s]+'
        r'(\d+)\s*[°]\s*(\d+)\s*[\'′]\s*(\d+(?:\.\d+)?)\s*[\"″]?\s*([EWew])',
        text,
    )
    if dms:
        lat = (float(dms.group(1))
               + float(dms.group(2)) / 60
               + float(dms.group(3)) / 3600)
        lon = (float(dms.group(5))
               + float(dms.group(6)) / 60
               + float(dms.group(7)) / 3600)
        if dms.group(4).upper() == 'S':
            lat = -lat
        if dms.group(8).upper() == 'W':
            lon = -lon
        return lat, lon

    raise ValueError(
        f"Could not parse {text!r}. "
        "Use decimal (−33.8688, 151.2093) or DMS (33°52′12″S, 151°12′36″E)."
    )


def cache_key(lat: float, lon: float) -> tuple[float, float]:
    """Normalised cache key for a coordinate pair."""
    return (round(lat, 6), round(lon, 6))
