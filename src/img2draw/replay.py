"""Continuous "watch it being drawn" replay.

Every op gets a duration; the video clock runs straight through all ops, so
at any instant the op in progress is drawn *partially*: pen strokes grow along
their path (with a pen-tip cursor), color shapes spread from a seed point,
polish tiles wipe in. No cuts, no jumps between stages.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from . import renderer as R

# share of the video each stage gets (renormalized over stages present)
STAGE_SHARE = {"Outline": 0.30, "Flat colors": 0.20, "Shadows": 0.07,
               "Highlights": 0.08, "Details": 0.17, "Polish": 0.18}


def _weight(op: dict) -> float:
    t = op["type"]
    if t in ("BRUSH_STROKE", "PATH"):
        pts = np.asarray(op["points"], float)
        length = float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 1.0
        return 0.4 + length / 60.0
    if t == "FILL":
        pts = np.asarray(op.get("rings", [op["points"]])[0], float)
        area = abs(cv2.contourArea(pts.astype(np.float32)))
        return 0.8 + np.sqrt(area) / 25.0
    return 0.5


def schedule(project: dict, seconds: float) -> tuple[list, np.ndarray, np.ndarray]:
    """-> (timeline, starts, ends) in seconds, stage-balanced."""
    tl = R.timeline(project)
    w = np.array([_weight(op) for _, op in tl], float)
    stages = project.get("metadata", {}).get("stages") or []
    t_of = np.array([op.get("t", i) for i, (_, op) in enumerate(tl)])
    dur = w.copy()
    if stages:
        shares = {s["name"]: STAGE_SHARE.get(s["name"], 0.1) for s in stages}
        tot = sum(shares.values())
        covered = np.zeros(len(tl), bool)
        for s in stages:
            m = (t_of >= s["start"]) & (t_of < s["end"])
            if m.any():
                dur[m] = w[m] / w[m].sum() * seconds * shares[s["name"]] / tot
                covered |= m
        if (~covered).any():
            dur[~covered] = w[~covered] / w.sum() * seconds * 0.05
    else:
        dur = w / w.sum() * seconds
    ends = np.cumsum(dur)
    return tl, ends - dur, ends


def _pen(frame: np.ndarray, tip, scale: float = 1.0) -> None:
    c = (int(round(tip[0])), int(round(tip[1])))
    cv2.circle(frame, c, 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, c, 4, (30, 30, 30), 1, cv2.LINE_AA)


def frames(project: dict, fps: int = 30, seconds: float = 30.0, hold: float = 1.5) -> Iterator[np.ndarray]:
    cw, ch = project["canvas"]["width"], project["canvas"]["height"]
    tl, starts, ends = schedule(project, seconds)
    bufs = {li: R._LayerBuf(project["layers"][li], cw, ch) for li in sorted({li for li, _ in tl})}
    n, done = len(tl), 0

    def compose() -> np.ndarray:
        canvas = np.ones((ch, cw, 3), np.float32)
        for li in sorted(bufs):
            canvas = R._composite(canvas, bufs[li])
        return np.clip(np.rint(canvas * 255.0), 0, 255).astype(np.uint8)

    total = int(round(seconds * fps))
    for f in range(total + 1):
        now = f / fps
        while done < n and ends[done] <= now:
            bufs[tl[done][0]].draw(tl[done][1])
            done += 1
        token, tip, li = None, None, None
        if done < n and starts[done] < now:
            li, op = tl[done]
            p = float((now - starts[done]) / max(ends[done] - starts[done], 1e-9))
            token = bufs[li].draw(op, p)
            if op["type"] in ("BRUSH_STROKE", "PATH"):
                tip = R.stroke_prefix(op, p)[2]
        frame = compose()
        if tip is not None:
            frame = frame.copy()
            _pen(frame, tip)
        if token is not None:
            bufs[li].undo(token)
        yield frame
    for op_li, op in tl[done:]:
        bufs[op_li].draw(op)
    last = compose()
    for _ in range(int(round(hold * fps))):
        yield last


def save_video(project: dict, path: str | Path, fps: int = 30, seconds: float = 30.0,
               hold: float = 1.5, original: np.ndarray | None = None, min_side: int = 720) -> Path:
    """H.264 mp4 via imageio-ffmpeg when available, else OpenCV mp4v.
    If `original` is given it is shown to the left of the canvas."""
    path = Path(path)
    cw, ch = project["canvas"]["width"], project["canvas"]["height"]
    k = max(1.0, min_side / min(cw, ch))
    pw = cw * (2 if original is not None else 1)
    W, H = int(round(pw * k)) // 2 * 2, int(round(ch * k)) // 2 * 2

    def prep(fr):
        if original is not None:
            fr = np.hstack([original, fr])
        return cv2.resize(fr, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)

    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        proc = subprocess.Popen(
            [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
             "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
        for fr in frames(project, fps, seconds, hold):
            proc.stdin.write(np.ascontiguousarray(prep(fr)).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    except ImportError:
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        for fr in frames(project, fps, seconds, hold):
            vw.write(np.ascontiguousarray(prep(fr)[..., ::-1]))
        vw.release()
    return path
