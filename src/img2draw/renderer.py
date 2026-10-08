"""Deterministic numpy/OpenCV compositor for drawing programs.

Renders to float RGBA internally, composites layers in order with
opacity/blend modes, and returns uint8 RGB. Same input -> same pixels.
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import schema


def _decode(op: dict, key: str) -> np.ndarray:
    return schema.decode_op_image(op, key)


def _blend(base: np.ndarray, top: np.ndarray, mode: str) -> np.ndarray:
    """base/top: float RGBA in [0,1], top already alpha-weighted."""
    a = top[..., 3:4]
    if mode == "normal":
        rgb = top[..., :3] * a + base[..., :3] * (1 - a)
    elif mode == "multiply":
        rgb = top[..., :3] * base[..., :3] * a + base[..., :3] * (1 - a)
    elif mode == "screen":
        rgb = (1 - (1 - top[..., :3]) * (1 - base[..., :3])) * a + base[..., :3] * (1 - a)
    elif mode == "overlay":
        b = base[..., :3]
        o = np.where(b < 0.5, 2 * b * top[..., :3], 1 - 2 * (1 - b) * (1 - top[..., :3]))
        rgb = o * a + b * (1 - a)
    elif mode == "add":
        rgb = np.clip(base[..., :3] + top[..., :3] * a, 0, 1)
    elif mode == "add_signed":
        # detail/correction layer: top holds (residual + 128); exact base+residual
        rgb = np.clip(base[..., :3] + (top[..., :3] - 128.0 / 255.0) * a, 0, 1)
    else:  # unsupported modes fall back to normal rather than failing
        rgb = top[..., :3] * a + base[..., :3] * (1 - a)
    alpha = np.clip(base[..., 3:4] + a * (1 - base[..., 3:4]), 0, 1)
    return np.concatenate([rgb, alpha], -1)


def _apply_transform(canvas: np.ndarray, op: dict) -> None:
    m = op.get("matrix")
    if m and len(m) == 6:
        M = np.array(m, dtype=np.float32).reshape(2, 3)
        canvas[:] = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)


def _op_layer(op: dict, w: int, h: int) -> np.ndarray:
    """Rasterize one operation to a float RGBA layer."""
    t = op["type"]
    layer = np.zeros((h, w, 4), np.float32)
    if t == "IMAGE_PATCH" or t == "TEXTURE":
        img = _decode(op, "image").astype(np.float32) / 255.0
        x, y = int(op.get("x", 0)), int(op.get("y", 0))
        ih, iw = img.shape[:2]
        region = layer[y:y + ih, x:x + iw]
        rgb = img[..., :3]
        alpha = np.ones((ih, iw, 1), np.float32)
        if "mask" in op:
            m = _decode(op, "mask")[..., 0].astype(np.float32) / 255.0
            alpha = m[..., None]
        elif img.shape[2] == 4:
            alpha = img[..., 3:4]
        region[:] = np.concatenate([rgb, alpha], -1)
    elif t == "FILL":
        pts = np.array(op["points"], dtype=np.int32)
        color = op.get("color", [0, 0, 0])
        rgba = np.array(color[:3] + [255], np.uint8)
        cv2.fillPoly(layer, [pts], color=[float(color[0]) / 255, float(color[1]) / 255, float(color[2]) / 255, 1.0])
        layer[..., 3] = layer[..., 3]  # alpha set by fill
        # cv2 fillPoly on float writes the 4-tuple; ensure alpha is 1 where filled
        filled = layer[..., 3] > 0
        layer[..., 3] = np.where(filled, op.get("opacity", 1.0), 0.0)
        # normalize color channels where filled
        for c, v in enumerate(color[:3]):
            layer[..., c] = np.where(filled, float(v) / 255.0, 0.0)
    elif t == "PATH":
        pts = np.array(op["points"], dtype=np.int32)
        color = op.get("color", [0, 0, 0])
        cv2.polylines(layer, [pts], isClosed=bool(op.get("closed", False)),
                      color=[color[0] / 255, color[1] / 255, color[2] / 255, 1.0],
                      thickness=int(op.get("width", 2)), lineType=cv2.LINE_AA)
        filled = layer[..., 3] > 0
        layer[..., 3] = np.where(filled, op.get("opacity", 1.0), 0.0)
        for c, v in enumerate(color[:3]):
            layer[..., c] = np.where(filled, float(v) / 255.0, 0.0)
    elif t == "BRUSH_STROKE":
        pts = np.array(op["points"], dtype=np.float32)
        color = op.get("color", [0, 0, 0])
        base_w = float(op.get("width", 8))
        pressures = op.get("pressure") or [1.0] * len(pts)
        rgba = [color[0] / 255, color[1] / 255, color[2] / 255, 1.0]
        for i in range(len(pts) - 1):
            w = max(1, int(base_w * float(pressures[i])))
            cv2.line(layer, tuple(pts[i].astype(int)), tuple(pts[i + 1].astype(int)),
                     color=rgba, thickness=w, lineType=cv2.LINE_AA)
            cv2.circle(layer, tuple(pts[i].astype(int)), w // 2, color=rgba, lineType=cv2.LINE_AA)
        alpha = op.get("opacity", 1.0)
        layer[..., 3] *= alpha
    elif t == "GRADIENT":
        g = op.get("stops") or [[0.0, [0, 0, 0]], [1.0, [255, 255, 255]]]
        direction = op.get("direction", "vertical")
        n = h if direction == "vertical" else w
        tt = np.linspace(0, 1, n, dtype=np.float32)
        stops = sorted(g, key=lambda s: s[0])
        pos = np.array([s[0] for s in stops], np.float32)
        cols = np.array([s[1] for s in stops], np.float32) / 255.0
        ramp = np.stack([np.interp(tt, pos, cols[:, c]) for c in range(3)], -1)
        grad = np.repeat(ramp[:, None, :], w, axis=1) if direction == "vertical" else np.repeat(ramp[None, :, :], h, axis=0)
        layer[..., :3] = grad
        layer[..., 3] = float(op.get("opacity", 1.0))
    elif t == "ERASE":
        pts = np.array(op["points"], dtype=np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        layer[..., 3] = mask.astype(np.float32) / 255.0
        layer[..., :3] = 1.0  # color irrelevant; alpha used for destination-out
    elif t == "MASK":
        m = _decode(op, "image") if "image" in op else np.zeros((h, w, 3), np.uint8)
        layer[..., 3] = m[..., 0].astype(np.float32) / 255.0
        layer[..., :3] = 1.0
    elif t == "FILTER":
        pass  # filters apply to a source patch carried in the op
        src = _decode(op, "image") if "image" in op else None
        if src is not None:
            kind = op.get("kind", "blur")
            if kind == "blur":
                src = cv2.GaussianBlur(src, (0, 0), float(op.get("sigma", 2)))
            elif kind == "sharpen":
                k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                src = cv2.filter2D(src, -1, k)
            x, y = int(op.get("x", 0)), int(op.get("y", 0))
            ih, iw = src.shape[:2]
            layer[y:y + ih, x:x + iw, :3] = src[..., :3].astype(np.float32) / 255.0
            layer[y:y + ih, x:x + iw, 3] = 1.0
    elif t == "TRANSFORM":
        src = _decode(op, "image") if "image" in op else None
        if src is not None:
            inter = src.astype(np.float32) / 255.0
            m = op.get("matrix")
            if m and len(m) == 6:
                M = np.array(m, np.float32).reshape(2, 3)
                inter = cv2.warpAffine(inter, M, (w, h), flags=cv2.INTER_LINEAR)
            layer[:] = np.dstack([inter[..., :3], inter[..., 3:4] if inter.shape[2] == 4 else np.ones((h, w, 1), np.float32)])
    elif t == "GROUP":
        for sub in op.get("operations", []):
            sub_layer = _op_layer(sub, w, h)
            layer = _blend(layer, sub_layer, "normal")
    return layer


def render_project(project: dict, max_ops: int | None = None) -> np.ndarray:
    """Render the project. max_ops limits total ops drawn (timeline scrubbing)."""
    cw, ch = project["canvas"]["width"], project["canvas"]["height"]
    canvas = np.zeros((ch, cw, 4), np.float32)
    count = 0
    for layer in project["layers"]:
        if not layer.get("visible", True):
            continue
        layer_buf = np.zeros((ch, cw, 4), np.float32)
        for op in layer["operations"]:
            if max_ops is not None and count >= max_ops:
                break
            if op["type"] == "ERASE":
                m = _op_layer(op, cw, ch)[..., 3:4]
                layer_buf[..., 3:4] *= (1.0 - m)
            else:
                layer_buf = _blend(layer_buf, _op_layer(op, cw, ch), "normal")
            count += 1
        layer_buf[..., 3] *= float(layer.get("opacity", 1.0))
        canvas = _blend(canvas, layer_buf, layer.get("blend_mode", "normal"))
    # composite over white background
    rgb = canvas[..., :3]
    a = canvas[..., 3:4]
    out = rgb * a + (1 - a)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def render_to_array(project: dict) -> np.ndarray:
    return render_project(project)


def export_png(project: dict, path: str | Path) -> None:
    Image.fromarray(render_project(project)).save(path)


def export_webp(project: dict, path: str | Path) -> None:
    Image.fromarray(render_project(project)).save(path, format="WEBP")


def export_svg(project: dict, path: str | Path) -> None:
    """SVG export: vector ops as real SVG; raster patches embedded as PNG."""
    cw, ch = project["canvas"]["width"], project["canvas"]["height"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{cw}" height="{ch}">']
    for layer in project["layers"]:
        if not layer.get("visible", True):
            continue
        parts.append(f'<g opacity="{layer.get("opacity", 1.0)}">')
        for op in layer["operations"]:
            t = op["type"]
            if t == "FILL":
                pts = " ".join(f'{x},{y}' for x, y in op["points"])
                c = op.get("color", [0, 0, 0])
                parts.append(f'<polygon points="{pts}" fill="rgb({c[0]},{c[1]},{c[2]})" opacity="{op.get("opacity", 1.0)}"/>')
            elif t == "PATH":
                pts = " ".join(f'{x},{y}' for x, y in op["points"])
                c = op.get("color", [0, 0, 0])
                parts.append(f'<polyline points="{pts}" fill="none" stroke="rgb({c[0]},{c[1]},{c[2]})" stroke-width="{op.get("width", 2)}"/>')
            elif t in ("IMAGE_PATCH", "TEXTURE"):
                parts.append(f'<image x="{op.get("x", 0)}" y="{op.get("y", 0)}" href="data:image/png;base64,{op["image"]}"/>')
            elif t == "GRADIENT":
                stops = op.get("stops", [])
                gid = f'g{abs(hash(str(stops))) % 10**8}'
                parts.append(f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">')
                for pos, col in stops:
                    parts.append(f'<stop offset="{pos}" stop-color="rgb({col[0]},{col[1]},{col[2]})"/>')
                parts.append(f'</linearGradient></defs><rect width="{cw}" height="{ch}" fill="url(#{gid})" opacity="{op.get("opacity", 1.0)}"/>')
            # BRUSH_STROKE etc exported as embedded raster for fidelity
            elif t == "BRUSH_STROKE":
                raster = _op_layer(op, cw, ch)
                buf = io.BytesIO()
                Image.fromarray((raster[..., :3] * raster[..., 3:4] * 255).astype(np.uint8), "RGBA").save(buf, format="PNG")
                import base64
                parts.append(f'<image x="0" y="0" href="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"/>')
        parts.append("</g>")
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")
