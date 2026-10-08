"""Image analysis + visual decomposition.

Decomposes a raster image into an editable layered drawing program:
  Background  - fitted gradient (or dominant color)
  Structure   - quantized color regions as filled vector-ish polys (from contours)
  Highlights/Shadows - luminance-split raster patches
  Correction  - residual IMAGE_PATCH guaranteeing fidelity
"""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans

from . import schema


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img[..., ::-1].copy()


def kmeans_regions(img: np.ndarray, k: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels HxW, palette Kx3 uint8)."""
    h, w = img.shape[:2]
    pixels = img.reshape(-1, 3).astype(np.float32)
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(pixels)
    labels = km.labels_.reshape(h, w).astype(np.int32)
    palette = km.cluster_centers_.astype(np.uint8)
    return labels, palette


def region_polygons(labels: np.ndarray, palette: np.ndarray, min_area: int = 64):
    """Per-cluster filled polygons via contour extraction (OpenCV, deterministic)."""
    out = []
    for c in range(len(palette)):
        mask = (labels == c).astype(np.uint8) * 255
        # morphological close to reduce speckle
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            eps = 0.002 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
            if len(approx) >= 3:
                out.append((approx.tolist(), [int(v) for v in palette[c]]))
    return out


def fit_background_gradient(img: np.ndarray) -> dict:
    """Fit vertical luminance/color ramp to the image border pixels."""
    border = np.concatenate([img[0].reshape(-1, 3), img[-1].reshape(-1, 3),
                             img[:, 0].reshape(-1, 3), img[:, -1].reshape(-1, 3)])
    top = img[0].reshape(-1, 3).mean(0)
    bottom = img[-1].reshape(-1, 3).mean(0)
    return {"type": "GRADIENT", "direction": "vertical", "opacity": 1.0,
            "stops": [[0.0, [int(v) for v in top]], [1.0, [int(v) for v in bottom]]]}


def split_shading(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (highlights_patch_rgba, shadows_patch_rgba) as raster patches."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hi_mask = np.clip((gray - 160) / 95.0, 0, 1)
    lo_mask = np.clip((96 - gray) / 96.0, 0, 1)
    hi = np.dstack([img.astype(np.float32), hi_mask * 255]).astype(np.uint8)
    lo = np.dstack([np.zeros_like(img) + 10, lo_mask * 200]).astype(np.uint8)
    return hi, lo


def edges(img: np.ndarray) -> np.ndarray:
    return cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 50, 150)


def vtracer_available() -> bool:
    try:
        import vtracer  # noqa
        return True
    except Exception:
        return False


def trace_with_vtracer(path: str, out_svg: str) -> str | None:
    try:
        import vtracer
        vtracer.convert_image_to_svg_py(path, out_svg)
        return out_svg
    except Exception:
        return None
