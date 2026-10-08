"""FastAPI editor backend: load/reconstruct, render, edit ops, layers,
undo/redo, timeline, metrics, project save/load, export."""
from __future__ import annotations

import base64
import copy
import io
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image

from . import metrics, planner, schema
from .decompose import load_image
from .renderer import render_project, export_svg, export_png, export_webp

app = FastAPI(title="img2draw editor")
STATE: dict = {"project": None, "original": None, "history": [], "future": [], "max_ops": None}

FRONTEND = Path(__file__).parent.parent.parent / "frontend" / "index.html"


def _png_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _snapshot():
    STATE["history"].append(copy.deepcopy(STATE["project"]))
    STATE["future"].clear()
    if len(STATE["history"]) > 50:
        STATE["history"].pop(0)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(FRONTEND.read_text(encoding="utf-8"))


@app.post("/api/reconstruct")
async def reconstruct(file: UploadFile = File(...), mode: str = "balanced"):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
    if STATE.get("busy"):
        return JSONResponse({"error": "Already processing an image. Wait for it to finish."}, status_code=409)
    STATE["busy"] = True
    try:
        STATE["original"] = np.ascontiguousarray(img)
        from starlette.concurrency import run_in_threadpool
        STATE["project"] = await run_in_threadpool(planner.plan, STATE["original"], mode)
    finally:
        STATE["busy"] = False
    STATE["video"] = None
    STATE["history"].clear(); STATE["future"].clear()
    rendered = render_project(STATE["project"])
    return {"metrics": metrics.full_report(STATE["original"], rendered),
            "reconstruction": _png_b64(rendered),
            "original": _png_b64(STATE["original"]),
            "layers": [{"id": l["id"], "name": l["name"], "visible": l["visible"],
                        "opacity": l["opacity"], "blend_mode": l["blend_mode"],
                        "ops": len(l["operations"])} for l in STATE["project"]["layers"]],
            "op_count": planner.op_count(STATE["project"])}


@app.get("/api/replay.mp4")
def replay(seconds: float = 30.0):
    """Continuous live-drawing video of the current project (cached until edited)."""
    if STATE["project"] is None:
        return JSONResponse({"error": "no project"}, status_code=400)
    from .replay import save_video
    path = Path(tempfile.gettempdir()) / "img2draw_replay.mp4"
    save_video(STATE["project"], path, seconds=seconds, original=STATE["original"])
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/render")
def render(max_ops: int | None = None):
    if STATE["project"] is None:
        return JSONResponse({"error": "no project"}, status_code=400)
    rendered = render_project(STATE["project"], max_ops=max_ops)
    out = {"reconstruction": _png_b64(rendered)}
    if max_ops is not None and STATE["original"] is not None:
        out["difference"] = _png_b64(metrics.error_map(STATE["original"], rendered))
    return out


@app.get("/api/compare")
def compare():
    proj, orig = STATE["project"], STATE["original"]
    if proj is None or orig is None:
        return JSONResponse({"error": "no project"}, status_code=400)
    r = render_project(proj)
    return {"reconstruction": _png_b64(r),
            "difference": _png_b64(metrics.error_map(orig, r)),
            "heatmap": _png_b64(metrics.error_heatmap(orig, r)),
            "overlay": _png_b64((orig.astype(float) * .5 + r.astype(float) * .5).astype(np.uint8)),
            "metrics": metrics.full_report(orig, r)}


@app.post("/api/layer/{layer_id}")
def update_layer(layer_id: str, visible: bool | None = None, opacity: float | None = None,
                 blend_mode: str | None = None, name: str | None = None):
    _snapshot()
    for l in STATE["project"]["layers"]:
        if l["id"] == layer_id:
            if visible is not None: l["visible"] = visible
            if opacity is not None: l["opacity"] = opacity
            if blend_mode is not None: l["blend_mode"] = blend_mode
            if name is not None: l["name"] = name
            return {"ok": True}
    return JSONResponse({"error": "not found"}, status_code=404)


@app.post("/api/layer/{layer_id}/delete")
def delete_layer(layer_id: str):
    _snapshot()
    STATE["project"]["layers"] = [l for l in STATE["project"]["layers"] if l["id"] != layer_id]
    return {"ok": True}


@app.post("/api/layers/reorder")
def reorder(ids: list[str]):
    _snapshot()
    by_id = {l["id"]: l for l in STATE["project"]["layers"]}
    STATE["project"]["layers"] = [by_id[i] for i in ids if i in by_id]
    return {"ok": True}


def _edit_layer():
    layer = next((l for l in STATE["project"]["layers"] if l["name"] == "Edits"), None)
    if layer is None:
        layer = schema.new_layer("layer_edits", "Edits", type_="raster")
        STATE["project"]["layers"].append(layer)
    return layer


def _add_edit(op: dict, frame: int | None):
    """Slot an edit into the timeline right after `frame` steps (end if None).
    Later frames then respect it: see renderer.State."""
    from .renderer import timeline
    tl = timeline(STATE["project"])
    n = len(tl) if frame is None else min(max(frame, 0), len(tl))
    prev = tl[n - 1][1].get("t", n - 1) if n else -1
    nxt = tl[n][1].get("t", n) if n < len(tl) else prev + 1
    op["t"] = (prev + nxt) / 2 if n < len(tl) else prev + 1
    _snapshot()
    _edit_layer()["operations"].append(op)


@app.post("/api/edit/brush")
def add_brush(points: list[list[float]], color: list[int], width: float = 8, opacity: float = 1.0,
              frame: int | None = None):
    _add_edit({"type": "BRUSH_STROKE", "points": points, "color": color, "width": width,
               "opacity": opacity, "edit": True}, frame)
    return {"ok": True}


@app.post("/api/edit/erase")
def add_erase(points: list[list[float]] = Body(embed=True), width: float = 12, frame: int | None = None):
    _add_edit({"type": "ERASER", "points": points, "width": width}, frame)
    return {"ok": True}


@app.get("/api/progress")
def progress():
    from .artist import PROGRESS
    return {**PROGRESS, "busy": bool(STATE.get("busy"))}


@app.get("/api/project")
def project_json():
    return JSONResponse(STATE["project"], headers={"Content-Disposition": "attachment; filename=drawing.project.json"})


@app.get("/api/info")
def info():
    """Frame count, stage boundaries (as frame numbers), undo/redo availability."""
    from .renderer import timeline
    if STATE["project"] is None:
        return JSONResponse({"error": "no project"}, status_code=400)
    tl = timeline(STATE["project"])
    ts = [op.get("t", i) for i, (_, op) in enumerate(tl)]
    stages = [{"name": st["name"], "start": sum(t < st["start"] for t in ts),
               "end": sum(t < st["end"] for t in ts)}
              for st in STATE["project"]["metadata"].get("stages", [])]
    return {"frames": len(tl), "stages": stages, "width": STATE["project"]["canvas"]["width"],
            "height": STATE["project"]["canvas"]["height"],
            "can_undo": bool(STATE["history"]), "can_redo": bool(STATE["future"])}


@app.get("/api/pick")
def pick(x: int, y: int, frame: int | None = None):
    """Object under (x, y) at `frame`: id, bbox and a tint mask (PNG, alpha)."""
    from .renderer import state_at
    st = state_at(STATE["project"], frame)
    om = st.owner_map()
    oid = st.pick(x, y)
    if oid is None:
        return {"id": None}
    m = om == st._idn[oid]
    ys, xs = np.nonzero(m)
    rgba = np.zeros((*m.shape, 4), np.uint8)
    rgba[m] = (255, 64, 160, 140)
    ok, png = cv2.imencode(".png", rgba[..., [2, 1, 0, 3]])
    return {"id": oid, "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "area": int(m.sum()), "mask": base64.b64encode(png.tobytes()).decode()}


@app.post("/api/edit/delete")
def delete_object(target: str, frame: int | None = None):
    _add_edit({"type": "DELETE", "targets": [target]}, frame)
    return {"ok": True}


@app.post("/api/edit/recolor")
def recolor_object(target: str, color: list[int] = Body(embed=True), frame: int | None = None):
    _add_edit({"type": "RECOLOR", "target": target, "color": color}, frame)
    return {"ok": True}


@app.post("/api/undo")
def undo():
    if STATE["history"]:
        STATE["future"].append(copy.deepcopy(STATE["project"]))
        STATE["project"] = STATE["history"].pop()
    return {"ok": True}


@app.post("/api/redo")
def redo():
    if STATE["future"]:
        STATE["history"].append(copy.deepcopy(STATE["project"]))
        STATE["project"] = STATE["future"].pop()
    return {"ok": True}


@app.post("/api/project/save")
def save(path: str = "project.json"):
    schema.save_project(STATE["project"], path)
    return {"ok": True, "path": path}


@app.post("/api/project/load")
def load(path: str):
    STATE["project"] = schema.load_project(path)
    STATE["history"].clear(); STATE["future"].clear()
    return {"ok": True}


@app.post("/api/export/{fmt}")
def export(fmt: str, path: str = "export"):
    if fmt == "png": export_png(STATE["project"], path + ".png")
    elif fmt == "webp": export_webp(STATE["project"], path + ".webp")
    elif fmt == "svg": export_svg(STATE["project"], path + ".svg")
    elif fmt == "project": schema.save_project(STATE["project"], path + ".project.json")
    else: return JSONResponse({"error": "bad format"}, status_code=400)
    return {"ok": True}


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
