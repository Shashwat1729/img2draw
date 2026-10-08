"""Canonical versioned drawing-program schema.

A drawing program is a JSON-serializable dict: canvas, metadata, ordered
layers, each with ordered operations. Every operation is deterministic and
carries enough fields to reproduce itself.
"""
from __future__ import annotations

import base64
import json
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

VERSION = "1.0"

OP_TYPES = [
    "PATH", "BRUSH_STROKE", "FILL", "GRADIENT", "ERASE", "MASK",
    "TRANSFORM", "IMAGE_PATCH", "TEXTURE", "FILTER", "GROUP",
]


def new_canvas(width: int, height: int, color_space: str = "sRGB") -> dict:
    return {"width": int(width), "height": int(height), "color_space": color_space}


def new_project(width: int, height: int, seed: int = 0, mode: str = "balanced") -> dict:
    return {
        "version": VERSION,
        "canvas": new_canvas(width, height),
        "renderer_version": "1.0",
        "representation_version": VERSION,
        "color_profile": "sRGB",
        "seed": int(seed),
        "mode": mode,
        "metadata": {},
        "layers": [],
    }


def new_layer(layer_id: str, name: str, type_: str = "vector", **kw) -> dict:
    layer = {
        "id": layer_id, "name": name, "type": type_, "visible": True,
        "opacity": 1.0, "blend_mode": "normal", "locked": False, "operations": [],
    }
    layer.update(kw)
    return layer


def _encode_image(arr: np.ndarray) -> str:
    """arr uint8 HxWxC or HxWx4 -> base64 PNG."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_image(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))


def op_image_patch(x: int, y: int, image: np.ndarray, mask: np.ndarray | None = None) -> dict:
    op: dict[str, Any] = {
        "type": "IMAGE_PATCH", "x": int(x), "y": int(y),
        "image": _encode_image(image),
    }
    if mask is not None:
        m = mask if mask.ndim == 2 else mask[:, :, 0]
        op["mask"] = _encode_image(np.stack([m] * 3, -1).astype(np.uint8))
    return op


def op_texture(x: int, y: int, image: np.ndarray) -> dict:
    return {"type": "TEXTURE", "x": int(x), "y": int(y), "image": _encode_image(image)}


def decode_op_image(op: dict, key: str = "image") -> np.ndarray:
    return _decode_image(op[key])


def validate(project: dict) -> list[str]:
    errors = []
    if project.get("version") != VERSION:
        errors.append(f"unsupported version {project.get('version')}")
    for k in ("canvas", "layers"):
        if k not in project:
            errors.append(f"missing {k}")
    for i, layer in enumerate(project.get("layers", [])):
        for k in ("id", "name", "type", "operations"):
            if k not in layer:
                errors.append(f"layer {i} missing {k}")
        for j, op in enumerate(layer.get("operations", [])):
            t = op.get("type")
            if t not in OP_TYPES:
                errors.append(f"layer {i} op {j} bad type {t}")
    return errors


def save_project(project: dict, path: str | Path) -> None:
    errs = validate(project)
    if errs:
        raise ValueError(f"invalid project: {errs}")
    Path(path).write_text(json.dumps(project), encoding="utf-8")


def load_project(path: str | Path) -> dict:
    project = json.loads(Path(path).read_text(encoding="utf-8"))
    errs = validate(project)
    if errs:
        raise ValueError(f"invalid project file: {errs}")
    return project


def flatten_ops(project: dict) -> list[tuple[dict, dict]]:
    """Yield (layer, op) in render order, skipping invisible layers."""
    for layer in project["layers"]:
        if layer.get("visible", True):
            for op in layer["operations"]:
                yield layer, op
