"""In-browser backend (Pyodide / GitHub Pages): same routes as editor_server,
no web framework. `call(route, query_json, body_json) -> json str`."""
from __future__ import annotations

import base64
import copy
import json

import cv2
import numpy as np

from . import artist, schema
from .renderer import render_project, state_at, timeline

S: dict = {"project": None, "history": [], "future": []}


def _b64(arr: np.ndarray) -> str:
    ok, png = cv2.imencode(".png", np.ascontiguousarray(arr[..., ::-1]) if arr.shape[-1] == 3 else arr)
    return base64.b64encode(png.tobytes()).decode()


def load(data, max_side: int = 480) -> dict:
    data = bytes(data.to_py()) if hasattr(data, "to_py") else data
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    img = np.ascontiguousarray(img)
    S.update(project=artist.plan(img, "balanced"), history=[], future=[])
    return {"original": _b64(img)}


def _edit(op: dict, frame):
    tl = timeline(S["project"])
    n = len(tl) if frame is None else min(max(int(frame), 0), len(tl))
    prev = tl[n - 1][1].get("t", n - 1) if n else -1
    nxt = tl[n][1].get("t", n) if n < len(tl) else prev + 1
    op["t"] = (prev + nxt) / 2 if n < len(tl) else prev + 1
    S["history"].append(copy.deepcopy(S["project"]))
    S["future"].clear()
    layer = next((l for l in S["project"]["layers"] if l["name"] == "Edits"), None)
    if layer is None:
        layer = schema.new_layer("layer_edits", "Edits", type_="raster")
        S["project"]["layers"].append(layer)
    layer["operations"].append(op)


def call(route: str, query: str = "{}", body: str = "null") -> str:
    q, b = json.loads(query), json.loads(body)
    p = S["project"]
    fr = q.get("frame")
    if route == "/api/info":
        tl = timeline(p)
        ts = [op.get("t", i) for i, (_, op) in enumerate(tl)]
        stages = [{"name": s["name"], "start": sum(t < s["start"] for t in ts),
                   "end": sum(t < s["end"] for t in ts)} for s in p["metadata"].get("stages", [])]
        out = {"frames": len(tl), "stages": stages, "width": p["canvas"]["width"],
               "height": p["canvas"]["height"], "can_undo": bool(S["history"]), "can_redo": bool(S["future"])}
    elif route == "/api/render":
        out = {"reconstruction": _b64(render_project(p, int(q["max_ops"])))}
    elif route == "/api/pick":
        st = state_at(p, int(fr))
        x, y = int(q["x"]), int(q["y"])
        oid = st.pick(x, y)
        if oid is None:
            out = {"id": None}
        else:
            m = st.owner_map() == st._idn[oid]
            rgba = np.zeros((*m.shape, 4), np.uint8)
            rgba[m] = (255, 64, 160, 140)
            out = {"id": oid, "area": int(m.sum()), "mask": _b64(rgba[..., [2, 1, 0, 3]])}
    elif route == "/api/edit/brush":
        _edit({"type": "BRUSH_STROKE", "points": b["points"], "color": b["color"], "width": float(q.get("width", 8)),
               "opacity": 1.0, "edit": True}, fr); out = {"ok": True}
    elif route == "/api/edit/erase":
        _edit({"type": "ERASER", "points": b["points"], "width": float(q.get("width", 12))}, fr); out = {"ok": True}
    elif route == "/api/edit/delete":
        _edit({"type": "DELETE", "targets": [q["target"]]}, fr); out = {"ok": True}
    elif route == "/api/edit/recolor":
        _edit({"type": "RECOLOR", "target": q["target"], "color": b["color"]}, fr); out = {"ok": True}
    elif route == "/api/undo":
        if S["history"]:
            S["future"].append(copy.deepcopy(p)); S["project"] = S["history"].pop()
        out = {"ok": True}
    elif route == "/api/redo":
        if S["future"]:
            S["history"].append(copy.deepcopy(p)); S["project"] = S["future"].pop()
        out = {"ok": True}
    else:
        out = {"error": "unknown route " + route}
    return json.dumps(out)
