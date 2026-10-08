"""Artist-style planner: image -> drawing program that is drawn the way a
person would draw it, step by step.

Stages (timeline order; `t` on every op), z-order differs: line art sits on top.
  1 Outline      pen strokes traced from the thin dark lines (skeleton + width)
  2 Flat colors  prime canvas, then big color blocks, large -> small
  3 Shadows      darker tone shapes over the flats
  4 Highlights   lighter tone shapes
  5 Details      small color shapes/accents, error-guided
  6 Polish       small signed touch-up tiles where pixels still differ,
                 until the canvas equals the source (tol=0) or within tol
"""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans

from . import schema
from .renderer import render_project
from .schema import new_layer, new_project, op_image_patch

# base K, tonal K, detail K, polish tolerance (max abs error left, 0 = exact)
MODES = {
    "fast": (6, 12, 24, 6),
    "balanced": (8, 20, 48, 0),
    "high_fidelity": (10, 28, 80, 0),
    "research": (10, 28, 80, 0),
}
MAX_SIDE = 1024


# ------------------------------------------------------------------ lines
def line_mask(img: np.ndarray) -> np.ndarray:
    """Thin dark structures (ink lines). Broad dark areas are left to fills."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    r = max(2, round(max(h, w) / 170))  # widest line we call a line
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    m = ((bh > 25) & (gray < 150)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 8
    return keep[lab].astype(np.uint8)


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning, vectorized."""
    im = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p = im
            p2, p3, p4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            p5, p6, p7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            p8, p9 = p[1:-1, :-2], p[:-2, :-2]
            c = p[1:-1, 1:-1]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(x.astype(np.int16) for x in nb)
            A = sum(((nb[i] == 0) & (nb[(i + 1) % 8] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = (c == 1) & (B >= 2) & (B <= 6) & (A == 1) & cond
            if rm.any():
                im[1:-1, 1:-1][rm] = 0
                changed = True
        if not changed:
            break
    return im[1:-1, 1:-1]


_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def trace_paths(skel: np.ndarray) -> list[list[tuple[int, int]]]:
    """Skeleton -> pen paths of (x, y) pixels. Greedy walk that keeps going
    straight through junctions (like a pen); side branches start new paths
    anchored at the pixel they leave from."""
    ys, xs = np.nonzero(skel)
    S = set(zip(xs.tolist(), ys.tolist()))
    nbr = {p: [(p[0] + dx, p[1] + dy) for dy, dx in _N8 if (p[0] + dx, p[1] + dy) in S] for p in S}
    visited: set = set()

    def grow(cur, d=None):
        out = []
        while True:
            cands = [q for q in nbr[cur] if q not in visited]
            if not cands:
                return out
            if d is None:
                q = cands[0]
            else:
                q = max(cands, key=lambda c: ((c[0] - cur[0]) * d[0] + (c[1] - cur[1]) * d[1])
                        / np.hypot(c[0] - cur[0], c[1] - cur[1]))
            d = (q[0] - cur[0], q[1] - cur[1])
            visited.add(q)
            out.append(q)
            cur = q

    paths = []
    starts = sorted(S, key=lambda p: (len(nbr[p]) != 1, p))  # endpoints first
    for p in starts:
        if p in visited:
            continue
        visited.add(p)
        fwd = grow(p)
        back = grow(p) if len(nbr[p]) > 1 else []
        path = back[::-1] + [p] + fwd
        anchor = [q for q in nbr[path[0]] if q in visited and q not in path]
        if anchor:
            path = [anchor[0]] + path
        paths.append(path)
    return paths


def _rdp_idx(pts: np.ndarray, eps: float) -> list[int]:
    keep = [0, len(pts) - 1]
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        d = b - a
        L = np.hypot(*d)
        seg = pts[i + 1:j] - a
        dist = np.hypot(*(seg.T)) if L == 0 else np.abs(d[0] * seg[:, 1] - d[1] * seg[:, 0]) / L
        k = int(dist.argmax())
        if dist[k] > eps:
            m = i + 1 + k
            keep.append(m)
            stack += [(i, m), (m, j)]
    return sorted(keep)


def outline_ops(img: np.ndarray, lines: np.ndarray) -> list[dict]:
    """Pen strokes (variable width) that reproduce the line mask. Width comes
    from ink coverage (sub-pixel), color from the darkest pixels on the path."""
    if not lines.any():
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    r = max(2, round(max(h, w) / 170))
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    dist = cv2.distanceTransform(lines, cv2.DIST_L2, 3)
    skel = thin(lines)
    strokes = []
    for path in trace_paths(skel):
        if len(path) < 3:
            continue
        px = np.asarray(path)
        xs, ys = px[:, 0], px[:, 1]
        g = gray[ys, xs]
        ink = float(np.percentile(g, 10))
        cover = np.clip((paper - gray) / np.maximum(paper - ink, 25.0), 0, 1)
        cs = cv2.boxFilter(cover, -1, (5, 5), normalize=False) / 5.0   # ~ ink width in px
        wid = np.minimum(cs[ys, xs], 2 * dist[ys, xs] + 1.0)
        if len(wid) >= 5:
            wid = np.convolve(np.pad(wid, 2, mode="edge"), np.ones(5) / 5, mode="valid")
        wid = np.clip(wid, 0.6, 12.0)
        pts = px.astype(np.float32)
        idx = _rdp_idx(pts, 0.7)
        cols = img[ys, xs].astype(np.float32)
        dark = g <= np.percentile(g, 35)
        col = np.median(cols[dark], axis=0)
        strokes.append({"type": "BRUSH_STROKE",
                        "points": [[float(x), float(y)] for x, y in pts[idx]],
                        "widths": [round(float(v), 2) for v in wid[idx]],
                        "color": [int(v) for v in col], "opacity": 1.0, "_len": len(path)})
    strokes.sort(key=lambda s: -s["_len"])  # main contours first, small marks last
    for s in strokes:
        del s["_len"]
    return strokes


# ------------------------------------------------------------------ regions
def smooth_underlying(img: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Image with ink lines painted out (color continues under the line)."""
    hole = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    clean = cv2.inpaint(img, hole * 255, 3, cv2.INPAINT_TELEA)
    for _ in range(2):
        clean = cv2.bilateralFilter(clean, 7, 20, 7)
    return clean


def quantize(clean: np.ndarray, k: int) -> np.ndarray:
    lab = cv2.cvtColor(clean, cv2.COLOR_RGB2LAB).astype(np.float32)
    flat = lab.reshape(-1, 3)
    rng = np.random.default_rng(0)
    sample = flat[rng.choice(len(flat), min(len(flat), 30000), replace=False)]
    km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(sample)
    labels = km.predict(flat).reshape(clean.shape[:2]).astype(np.int32)
    return mode_filter(labels, len(km.cluster_centers_))


def mode_filter(labels: np.ndarray, k: int, size: int = 3) -> np.ndarray:
    best = np.full(labels.shape, -1.0, np.float32)
    out = labels.copy()
    for c in range(k):
        score = cv2.blur((labels == c).astype(np.float32), (size, size))
        better = score > best
        out[better] = c
        best[better] = score[better]
    return out


UP = 4  # contour up-sampling: boundaries become sub-pixel accurate


def _subpixel_rings(mask: np.ndarray, min_hole: float, eps: float = 0.3):
    """Binary crop -> list of (rings) groups [outer, holes...] in crop coords
    with sub-pixel accuracy (soft mask, up-sampled, re-contoured)."""
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.7)
    big = cv2.resize(soft, None, fx=UP, fy=UP, interpolation=cv2.INTER_LINEAR)
    bm = (big > 0.5).astype(np.uint8)
    cs, hier = cv2.findContours(bm, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []
    hier = hier[0]
    holes: dict[int, list] = {}
    for i, (_, _, _, par) in enumerate(hier):
        if par >= 0 and cv2.contourArea(cs[i]) / UP ** 2 >= min_hole:
            holes.setdefault(par, []).append(i)
    groups = []
    for i, (_, _, _, par) in enumerate(hier):
        if par >= 0:
            continue
        rings = []
        for j in [i] + holes.get(i, []):
            ap = cv2.approxPolyDP(cs[j], eps * UP, True).reshape(-1, 2).astype(np.float64)
            if len(ap) >= 3:
                rings.append(np.round((ap + 0.5) / UP - 0.5, 2))
        if rings:
            groups.append(rings)
    return groups


def components(labels: np.ndarray, ref: np.ndarray, min_area: float, eps: float = 0.3):
    """Per connected component of the label map: (area, rings, mean_color, seed).
    Boundaries are sub-pixel. Holes smaller than min_area are dropped so the
    parent covers them."""
    H, W = labels.shape
    out = []
    for c in np.unique(labels):
        n, cc, st, _ = cv2.connectedComponentsWithStats((labels == c).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            area = float(st[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x, y, bw, bh = (int(st[i, k]) for k in (0, 1, 2, 3))
            pad = 2
            x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + bw + pad, W), min(y + bh + pad, H)
            crop = (cc[y0:y1, x0:x1] == i).astype(np.uint8)
            # replicate at image borders so shapes touching the edge stay closed past it
            padw = ((pad if y0 == 0 else 0, pad if y1 == H else 0), (pad if x0 == 0 else 0, pad if x1 == W else 0))
            mk = np.pad(crop, padw, mode="edge")
            ox, oy = x0 - padw[1][0], y0 - padw[0][0]
            color = ref[y0:y1, x0:x1][crop.astype(bool)].mean(0)
            dist = cv2.distanceTransform(crop, cv2.DIST_L2, 3)
            sy, sx = np.unravel_index(int(dist.argmax()), dist.shape)
            for rings in _subpixel_rings(mk, min_area, eps):
                out.append((area, [(r + [ox, oy]).tolist() for r in rings],
                            [int(round(v)) for v in color], [int(x0 + sx), int(y0 + sy)]))
    out.sort(key=lambda r: -r[0])
    return out


def fill_op(rings, color, seed=None) -> dict:
    op = {"type": "FILL", "rings": rings, "points": rings[0], "color": color, "opacity": 1.0}
    if seed is not None:
        op["seed"] = seed
    return op


def _lab(rgb) -> np.ndarray:
    return cv2.cvtColor(np.asarray(rgb, np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)


def _mean_color_in(rings, canvas: np.ndarray) -> np.ndarray:
    x0 = max(int(np.floor(min(p[0] for r in rings for p in r))), 0); x1 = int(np.ceil(max(p[0] for r in rings for p in r))) + 1
    y0 = max(int(np.floor(min(p[1] for r in rings for p in r))), 0); y1 = int(np.ceil(max(p[1] for r in rings for p in r))) + 1
    mk = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mk, [np.rint((np.asarray(r) - [x0, y0])).astype(np.int32) for r in rings], 1)
    sel = mk[:canvas.shape[0] - y0, :canvas.shape[1] - x0].astype(bool)
    return canvas[y0:y0 + sel.shape[0], x0:x0 + sel.shape[1]][sel].mean(0)


# ------------------------------------------------------------------ polish
def polish(project: dict, img: np.ndarray, tol: int = 0, passes: int = 4, start_t: int | None = None) -> int:
    """Signed touch-up tiles (serpentine order, like retouching top to bottom)
    until max |error| <= tol. Returns the next free t."""
    h, w = img.shape[:2]
    if start_t is None:
        start_t = 1 + max((op.get("t", 0) for l in project["layers"] for op in l["operations"]), default=-1)
    layer = next((l for l in project["layers"] if l["id"] == "layer_polish"), None)
    if layer is None:
        layer = new_layer("layer_polish", "Polish", "raster")
        layer["blend_mode"] = "add_signed"
        project["layers"].append(layer)
    t = start_t
    tile = 16 if max(h, w) <= 600 else 32
    for _ in range(passes):
        err = img.astype(np.int16) - render_project(project).astype(np.int16)
        mag = np.abs(err).max(-1)
        if mag.max() <= tol:
            break
        for row, y in enumerate(range(0, h, tile)):
            xs = list(range(0, w, tile))
            for x in (xs if row % 2 == 0 else xs[::-1]):
                if mag[y:y + tile, x:x + tile].max() <= tol:
                    continue
                e = err[y:y + tile, x:x + tile]
                e = np.where(np.abs(e).max(-1, keepdims=True) > tol, e, 0)
                op = op_image_patch(x, y, np.clip(e + 128, 0, 255).astype(np.uint8))
                op["t"] = t
                t += 1
                layer["operations"].append(op)
    if not layer["operations"]:
        project["layers"].remove(layer)
    return t


# ------------------------------------------------------------------ plan
def plan(img: np.ndarray, mode: str = "balanced", tol: int | None = None) -> dict:
    kb, kt, kd, mtol = MODES.get(mode, MODES["balanced"])
    tol = mtol if tol is None else tol
    h, w = img.shape[:2]
    project = new_project(w, h, mode=mode)
    project["metadata"]["source_size"] = [w, h]
    area = h * w
    t = [0]
    stages: list[dict] = []

    layers = {n: new_layer(f"layer_{n.lower().replace(' ', '_')}", n, "vector")
              for n in ("Flat colors", "Shadows", "Highlights", "Details", "Outline")}
    layers["Shadows"]["blend_mode"] = "normal"
    # z-order bottom -> top
    for n in ("Flat colors", "Shadows", "Highlights", "Details", "Outline"):
        project["layers"].append(layers[n])

    def add(layer, op):
        op["t"] = t[0]
        t[0] += 1
        layer["operations"].append(op)

    def stage(name, start):
        if t[0] > start:
            stages.append({"name": name, "start": start, "end": t[0]})

    # 1 outline ---------------------------------------------------------
    lines = line_mask(img)
    s0 = t[0]
    for op in outline_ops(img, lines):
        add(layers["Outline"], op)
    stage("Outline", s0)

    # 2 flat colors -----------------------------------------------------
    clean = smooth_underlying(img, lines)
    s0 = t[0]
    add(layers["Flat colors"], fill_op([[[0, 0], [w, 0], [w, h], [0, h]]],
                                       [int(v) for v in clean.reshape(-1, 3).mean(0)]))
    for _, rings, col, seed in components(quantize(clean, kb), clean, max(12, area * 0.0006)):
        add(layers["Flat colors"], fill_op(rings, col, seed))
    stage("Flat colors", s0)

    # 3-5 error-guided tone/detail passes --------------------------------
    def tone_pass(k, min_frac, thr, names):
        current = render_project(project)
        cur_lab_img = None
        found = {n: [] for n in set(names.values())}
        for a, rings, col, seed in components(quantize(clean, k), clean, max(8, area * min_frac)):
            cur = _mean_color_in(rings, current)
            dl = _lab(col)[0] - _lab(np.rint(cur))[0]
            if np.linalg.norm(dl) < thr:
                continue
            cls = "dark" if dl[0] < -3 else "light" if dl[0] > 3 else "neutral"
            found[names[cls]].append((rings, col, seed))
        return found

    s0 = t[0]
    found = tone_pass(kt, 0.0004, 5.0, {"dark": "Shadows", "light": "Highlights", "neutral": "Details"})
    for name, stage_name in (("Shadows", "Shadows"), ("Highlights", "Highlights")):
        s0 = t[0]
        for rings, col, seed in found[name]:
            add(layers[name], fill_op(rings, col, seed))
        stage(stage_name, s0)
    s0 = t[0]
    for rings, col, seed in found["Details"]:
        add(layers["Details"], fill_op(rings, col, seed))
    found = tone_pass(kd, 0.00012, 4.0, {"dark": "Details", "light": "Details", "neutral": "Details"})
    for rings, col, seed in found["Details"]:
        add(layers["Details"], fill_op(rings, col, seed))
    stage("Details", s0)

    # drop empty layers (keeps z-order)
    project["layers"] = [l for l in project["layers"] if l["operations"]]

    # 6 polish ----------------------------------------------------------
    project["metadata"]["structure_ops"] = t[0]
    s0 = t[0]
    t[0] = polish(project, img, tol, start_t=t[0])
    stage("Polish", s0)

    project["metadata"]["stages"] = stages
    project["metadata"]["polish_tol"] = tol
    project["metadata"]["progression"] = [s["name"] for s in stages]
    return project
