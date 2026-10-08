"""Planner entry point: delegates to the artist-style planner (see artist.py)."""
from __future__ import annotations

import numpy as np

from . import artist


def plan(img: np.ndarray, mode: str = "balanced", k: int | None = None) -> dict:
    return artist.plan(img, mode=mode)


def op_count(project: dict) -> int:
    return sum(len(l["operations"]) for l in project["layers"])
