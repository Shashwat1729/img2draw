"""Temporal/animation architecture skeleton.

The drawing program is frame-agnostic: a Project can embed per-frame
overrides. Objects/layers shared across frames keep identity; only the
delta is stored per frame. Reconstruction of frame t uses shared ops plus
frame-specific ops, so objects are tracked rather than rebuilt.
"""
from __future__ import annotations

from typing import Any


def project_for_frame(project: dict, frame: int) -> dict:
    """Return a shallow view of the project with frame `frame` overrides applied."""
    frames = project.get("frames")
    if not frames or frame == 0:
        return project
    import copy
    p = copy.deepcopy(project)
    overrides = frames.get(str(frame), {})
    for layer_id, changes in overrides.get("layers", {}).items():
        for layer in p["layers"]:
            if layer["id"] == layer_id:
                layer.update({k: v for k, v in changes.items() if k != "operations"})
                if "operations" in changes:
                    layer["operations"] = changes["operations"]
    return p


def temporal_consistency(frames_a: list, frames_b: list) -> dict | None:
    if len(frames_a) != len(frames_b) or not frames_a:
        return None
    import numpy as np
    diffs = [float(np.mean(np.abs(a.astype(int) - b.astype(int)))) for a, b in zip(frames_a, frames_b)]
    return {"mean_pixel_drift": float(np.mean(diffs)), "max_pixel_drift": float(max(diffs))}
