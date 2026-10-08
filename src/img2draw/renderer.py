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
    """Legacy full-canvas rasterizer (straight-alpha RGBA): rare ops + SVG export."""
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


# ---------------------------------------------------------------- fast path
_SS = 4   # supersampling factor for stroke coverage
_FP = 4   # sub-pixel bits for polygon fills (1/16 px)


def _rings(op: dict) -> list[np.ndarray]:
    rings = op.get("rings") or [op["points"]]
    return [np.asarray(r, dtype=np.float64).reshape(-1, 2) for r in rings]


def _roi_box(w: int, h: int, pts_list, pad: int):
    allp = np.concatenate(pts_list)
    x0 = max(0, int(np.floor(allp[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(allp[:, 1].min())) - pad)
    x1 = min(w, int(np.ceil(allp[:, 0].max())) + pad + 1)
    y1 = min(h, int(np.ceil(allp[:, 1].max())) + pad + 1)
    return x0, y0, x1, y1


def _stroke_arrays(op: dict):
    pts = np.asarray(op["points"], np.float64).reshape(-1, 2)
    ws = op.get("widths")
    if ws:
        ws = np.asarray(ws, np.float64)
    else:
        ws = np.asarray(op.get("pressure") or [1.0] * len(pts), np.float64) * float(op.get("width", 8))
    if op["type"] == "PATH":
        ws = np.full(len(pts), float(op.get("width", 2)))
    return pts, ws


def stroke_prefix(op: dict, progress: float):
    """Points/widths of the first `progress` fraction of a stroke's length,
    plus the pen tip. progress>=1 returns the whole stroke."""
    pts, ws = _stroke_arrays(op)
    if progress >= 1 or len(pts) < 2:
        return pts, ws, pts[-1]
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.r_[0.0, np.cumsum(seg)]
    target = max(progress, 0.0) * cum[-1]
    k = min(int(np.searchsorted(cum, target, "right")) - 1, len(pts) - 2)
    f = (target - cum[k]) / max(seg[k], 1e-9)
    tip = pts[k] + f * (pts[k + 1] - pts[k])
    wt = ws[k] + f * (ws[k + 1] - ws[k])
    return np.vstack([pts[:k + 1], tip]), np.r_[ws[:k + 1], wt], tip


def _stroke_mask(op: dict, w: int, h: int, progress: float = 1.0):
    pts, ws, _ = stroke_prefix(op, progress)
    pad = int(ws.max() / 2) + 2
    x0, y0, x1, y1 = _roi_box(w, h, [pts], pad)
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros(((y1 - y0) * _SS, (x1 - x0) * _SS), np.uint8)
    q = np.rint((pts - [x0, y0] + 0.5) * _SS).astype(np.int32)
    for i in range(len(q)):
        r = max(1, int(round(ws[i] * _SS / 2)))
        cv2.circle(mask, (int(q[i, 0]), int(q[i, 1])), r, 255, -1)
        if i + 1 < len(q):
            t = max(1, int(round((ws[i] + ws[i + 1]) * _SS / 2)))
            cv2.line(mask, (int(q[i, 0]), int(q[i, 1])), (int(q[i + 1, 0]), int(q[i + 1, 1])), 255, t)
    m = cv2.resize(mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    return x0, y0, m.astype(np.float32) / 255.0


def _ease(p: float) -> float:
    return p * p * (3 - 2 * p)


def _empty(color):
    return 0, 0, color, np.zeros((0, 0), np.float32)


def _roi_op(op: dict, w: int, h: int, progress: float = 1.0):
    """Rasterize one op (first `progress` of it) to (x0, y0, rgb, alpha) on a
    tight box; rgb is (3,) or (bh,bw,3) float in [0,1], alpha (bh,bw) float.
    None => not a fast-path op."""
    t = op["type"]
    op_a = float(op.get("opacity", 1.0))
    color = np.asarray(op.get("color", [0, 0, 0])[:3], np.float32) / 255.0
    if t == "FILL":
        rings = _rings(op)
        x0, y0, x1, y1 = _roi_box(w, h, rings, 1)
        if x1 <= x0 or y1 <= y0:
            return _empty(color)
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        k = 1 << _FP
        poly = [np.rint((r - [x0, y0]) * k).astype(np.int32) for r in rings]
        cv2.fillPoly(mask, poly, 255, lineType=cv2.LINE_AA, shift=_FP)
        a = mask.astype(np.float32) / 255.0 * op_a
        if progress < 1:  # paint spreads from a seed point
            sx, sy = op.get("seed") or ((x0 + x1) / 2, (y0 + y1) / 2)
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            d = np.hypot(xx - sx, yy - sy)
            R = float(d[a > 0].max()) if (a > 0).any() else 1.0
            a = a * np.clip((_ease(max(progress, 0)) * (R + 2) - d) + 0.5, 0, 1)
        return x0, y0, color, a
    if t in ("PATH", "BRUSH_STROKE"):
        r = _stroke_mask(op, w, h, progress)
        if r is None:
            return _empty(color)
        return r[0], r[1], color, r[2] * op_a
    if t in ("IMAGE_PATCH", "TEXTURE"):
        img = _decode(op, "image").astype(np.float32) / 255.0
        x, y = int(op.get("x", 0)), int(op.get("y", 0))
        ih, iw = img.shape[:2]
        x1, y1 = min(w, x + iw), min(h, y + ih)
        if x1 <= x or y1 <= y:
            return _empty(color)
        img = img[:y1 - y, :x1 - x]
        if "mask" in op:
            a = _decode(op, "mask")[..., 0].astype(np.float32)[:y1 - y, :x1 - x] / 255.0
        elif img.shape[2] == 4:
            a = img[..., 3]
        else:
            a = np.ones(img.shape[:2], np.float32)
        if progress < 1:  # soft left-to-right wipe
            cols = np.arange(a.shape[1], dtype=np.float32)
            a = a * np.clip(progress * (a.shape[1] + 1) - cols, 0, 1)[None, :]
        return x, y, img[..., :3], a * op_a
    return None


class _LayerBuf:
    """Premultiplied-RGBA accumulator for one layer (signed delta for add_signed),
    plus an owner map: which op last painted each pixel (>50% coverage)."""

    def __init__(self, layer: dict, w: int, h: int):
        self.layer, self.w, self.h = layer, w, h
        self.signed = layer.get("blend_mode") == "add_signed"
        self.buf = np.zeros((h, w, 3 if self.signed else 4), np.float32)
        self.owner = None if self.signed else np.full((h, w), -1, np.int32)

    def draw(self, op: dict, progress: float = 1.0, clip: np.ndarray | None = None,
             owner_id: int = -1):
        """Draw (part of) op. clip (HxW in [0,1]) protects pixels from this op.
        Returns (undo token | None, footprint (x0, y0, alpha_before_clip) | None)."""
        r = _roi_op(op, self.w, self.h, progress)
        if r is None:  # rare ops via legacy full-canvas rasterizer
            leg = _op_layer(op, self.w, self.h)
            r = (0, 0, leg[..., :3], leg[..., 3] * progress)
        x0, y0, rgb, a = r
        bh, bw = a.shape
        if bh == 0 or bw == 0:
            return None, None
        foot = (x0, y0, a)
        if clip is not None:
            a = a * (1.0 - clip[y0:y0 + bh, x0:x0 + bw])
        roi = self.buf[y0:y0 + bh, x0:x0 + bw]
        undo = None
        if progress < 1:
            undo = (y0, x0, roi.copy(), None if self.owner is None else self.owner[y0:y0 + bh, x0:x0 + bw].copy())
        a3 = a[..., None]
        if self.signed:
            roi += (rgb - 128.0 / 255.0) * a3
        else:
            roi[..., :3] = rgb * a3 + roi[..., :3] * (1 - a3)
            roi[..., 3:4] = a3 + roi[..., 3:4] * (1 - a3)
            if owner_id >= 0:
                self.owner[y0:y0 + bh, x0:x0 + bw][a > 0.5] = owner_id
        return undo, foot

    def undo(self, token):
        if token is not None:
            y0, x0, saved, own = token
            self.buf[y0:y0 + saved.shape[0], x0:x0 + saved.shape[1]] = saved
            if own is not None:
                self.owner[y0:y0 + own.shape[0], x0:x0 + own.shape[1]] = own


def _composite(canvas: np.ndarray, lb: _LayerBuf) -> np.ndarray:
    """canvas: opaque float RGB in [0,1]. Returns the new canvas."""
    op = float(lb.layer.get("opacity", 1.0))
    mode = lb.layer.get("blend_mode", "normal")
    if lb.signed:
        # quantize first so signed residuals reproduce 8-bit sources exactly
        canvas = np.rint(canvas * 255.0) / 255.0
        return np.clip(canvas + lb.buf * op, 0, 1)
    pm, a = lb.buf[..., :3], lb.buf[..., 3:4]
    if mode == "multiply":
        return canvas * (1 - a * op) + canvas * pm * op
    if mode == "screen":
        return canvas + op * pm * (1 - canvas)
    if mode == "add":
        return np.clip(canvas + pm * op, 0, 1)
    if mode == "overlay":
        s = pm / np.maximum(a, 1e-6)
        o = np.where(canvas < 0.5, 2 * canvas * s, 1 - 2 * (1 - canvas) * (1 - s))
        return canvas * (1 - a * op) + o * a * op
    return pm * op + canvas * (1 - a * op)


GLOBAL_OPS = ("DELETE", "RECOLOR")  # edits that rewrite history from their frame on


def timeline(project: dict) -> list[tuple[int, dict]]:
    """(layer_index, op) in drawing order. Ops carry `t` (float ok: edits are
    slotted between steps); untagged ops keep layer order. Hidden layers skipped.
    Layer order is z-order (bottom first); `t` is the artist's time order."""
    items = []
    seq = 0
    for li, layer in enumerate(project["layers"]):
        if not layer.get("visible", True):
            continue
        for op in layer["operations"]:
            items.append((op.get("t", seq), seq, li, op))
            seq += 1
    items.sort(key=lambda it: (it[0], it[1]))
    return [(li, op) for _, _, li, op in items]


def _globals(tl, n):
    """(dead ids, recolor map) from DELETE/RECOLOR items among the first n steps."""
    dead, recolor = set(), {}
    for _, op in tl[:n]:
        if op["type"] == "DELETE":
            dead.update(op["targets"])
        elif op["type"] == "RECOLOR":
            recolor[op["target"]] = op["color"]
    return dead, recolor


class State:
    """Replay state. Everything later steps do respects earlier edits:
    deleted objects (and their detail patches) never draw, recolored objects keep
    their detail, and user paint/erase freezes its footprint against later ops."""

    PROTECT = True  # research ablation switch: False = later ops may repaint user edits

    def __init__(self, project: dict, dead=frozenset(), recolor=None):
        self.project = project
        self.w, self.h = project["canvas"]["width"], project["canvas"]["height"]
        self.dead, self.recolor = set(dead), dict(recolor or {})
        self.parent = {op["id"]: op["parent"] for l in project["layers"] for op in l["operations"]
                       if "parent" in op and "id" in op}
        if self.dead:  # deleting an object deletes its whole part subtree (shading, highlights, detail)
            self.dead |= {i for i in self.parent if self.root(i) in self.dead}
        if self.recolor:  # recoloring an object shifts its parts by the same colour delta
            col = {op["id"]: op["color"] for l in project["layers"] for op in l["operations"]
                   if "id" in op and "color" in op and op["type"] == "FILL"}
            for tgt, new in list(self.recolor.items()):
                if tgt not in col:
                    continue
                delta = np.asarray(new, int) - np.asarray(col[tgt], int)
                for i in self.parent:
                    if self.root(i) == tgt and i in col and i not in self.recolor:
                        self.recolor[i] = np.clip(np.asarray(col[i], int) + delta, 0, 255).tolist()
        self.bufs: dict[int, _LayerBuf] = {}
        self.protect = np.zeros((self.h, self.w), np.float32)
        self.ids: list[str] = []
        self._idn: dict[str, int] = {}

    def root(self, i: str) -> str:
        while i in self.parent:
            i = self.parent[i]
        return i

    def select(self, x: int, y: int):
        """Click selection: the whole object (root + its parts) under (x, y) -> (root id, mask)."""
        i = self.pick(x, y)
        if i is None:
            return None, None
        r = self.root(i)
        om = self.owner_map()
        ks = [k for k, o in enumerate(self.ids) if self.root(o) == r]
        return r, np.isin(om, ks)

    def _buf(self, li: int) -> _LayerBuf:
        if li not in self.bufs:
            self.bufs[li] = _LayerBuf(self.project["layers"][li], self.w, self.h)
        return self.bufs[li]

    def _num(self, oid: str) -> int:
        if oid not in self._idn:
            self._idn[oid] = len(self.ids)
            self.ids.append(oid)
        return self._idn[oid]

    def draw(self, li: int, op: dict, progress: float = 1.0):
        """-> undo token (list) for partial draws, else None."""
        t = op["type"]
        if t in GLOBAL_OPS:
            return None
        oid = op.get("id")
        if (oid is not None and oid in self.dead) or op.get("target") in self.dead:
            return None
        if oid in self.recolor:
            op = dict(op, color=self.recolor[oid])
        if t == "ERASER":  # user eraser: wipes every layer, then freezes the area
            r = _stroke_mask(op, self.w, self.h)
            if r is None:
                return None
            x0, y0, m = r
            sl = (slice(y0, y0 + m.shape[0]), slice(x0, x0 + m.shape[1]))
            for lb in self.bufs.values():
                lb.buf[sl] *= (1.0 - m)[..., None]
                if lb.owner is not None:
                    lb.owner[sl][m > 0.5] = -1
            self.protect[sl] = np.maximum(self.protect[sl], m)
            return None
        is_edit = bool(op.get("edit"))
        clip = None if (is_edit or not self.PROTECT) else self.protect
        undo, foot = self._buf(li).draw(op, progress, clip, self._num(oid) if oid else -1)
        if foot is not None and is_edit and op.get("protect", True) and progress >= 1:
            x0, y0, a = foot
            sl = (slice(y0, y0 + a.shape[0]), slice(x0, x0 + a.shape[1]))
            self.protect[sl] = np.maximum(self.protect[sl], np.clip(a * 4, 0, 1))
        return None if undo is None else [(li, undo)]

    def undo(self, token):
        for li, tk in token or []:
            self.bufs[li].undo(tk)

    def compose(self) -> np.ndarray:
        canvas = np.ones((self.h, self.w, 3), np.float32)
        for li in sorted(self.bufs):
            canvas = _composite(canvas, self.bufs[li])
        return np.clip(np.rint(canvas * 255.0), 0, 255).astype(np.uint8)

    def owner_map(self) -> np.ndarray:
        """Topmost object id (index into self.ids) per pixel, -1 if none."""
        out = np.full((self.h, self.w), -1, np.int32)
        for li in sorted(self.bufs):
            lb = self.bufs[li]
            if lb.owner is None:
                continue
            vis = (lb.buf[..., 3] > 0.5) & (lb.owner >= 0)
            out[vis] = lb.owner[vis]
        return out

    def pick(self, x: int, y: int) -> str | None:
        if not (0 <= x < self.w and 0 <= y < self.h):
            return None
        o = int(self.owner_map()[y, x])
        return self.ids[o] if o >= 0 else None


def state_at(project: dict, n: int | None = None) -> State:
    """State after the first n timeline steps (all if None), from scratch."""
    tl = timeline(project)
    n = len(tl) if n is None else min(max(int(n), 0), len(tl))
    dead, rec = _globals(tl, n)
    st = State(project, dead, rec)
    for li, op in tl[:n]:
        st.draw(li, op)
    return st


def render_frames(project: dict, steps: list[int]):
    """Yield (step, uint8 RGB) per requested step count. Incremental unless a
    DELETE/RECOLOR edit appears, which rewrites history and forces a replay."""
    tl = timeline(project)
    st, done = State(project), 0
    cur = (set(), {})
    for s in sorted({min(max(int(s), 0), len(tl)) for s in steps}):
        g = _globals(tl, s)
        if g != cur:  # history rewritten: replay from scratch with the new globals
            st, done, cur = State(project, *g), 0, g
        for li, op in tl[done:s]:
            st.draw(li, op)
        done = s
        yield s, st.compose()


def render_project(project: dict, max_ops: int | None = None) -> np.ndarray:
    """Render the project. max_ops limits steps drawn, in timeline order."""
    n = len(timeline(project)) if max_ops is None else max_ops
    return next(render_frames(project, [n]))[1]


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
