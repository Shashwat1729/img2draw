"""End-to-end pipeline: image -> drawing program -> render -> metrics -> refine."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import artist, metrics, schema
from .renderer import render_frames, render_project


def refine(project: dict, img: np.ndarray, iterations: int = 4, tol: float = 0) -> dict:
    """Append signed correction (polish) tiles until the render is within tol."""
    artist.polish(project, img, tol=int(tol), passes=iterations)
    return project


def reconstruct(path: str, mode: str = "balanced", out_dir: str = "out", refine_iters: int = 3,
                video: bool = False, seconds: float = 30.0, tol: int | None = None) -> dict:
    from .decompose import load_image
    from PIL import Image
    img = load_image(path)
    project = artist.plan(img, mode=mode, tol=tol)
    rendered = render_project(project)
    structure = render_project(project, project["metadata"]["structure_ops"])
    report = metrics.full_report(img, rendered)
    report["structure_psnr"] = metrics.psnr(img, structure)
    report["structure_ssim"] = metrics.ssim(img, structure)
    report["stages"] = project["metadata"]["stages"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    schema.save_project(project, out / f"{stem}.project.json")
    Image.fromarray(rendered).save(out / f"{stem}.reconstruction.png")
    Image.fromarray(structure).save(out / f"{stem}.structure.png")
    Image.fromarray(metrics.error_map(img, rendered)).save(out / f"{stem}.difference.png")
    Image.fromarray(metrics.error_heatmap(img, rendered)).save(out / f"{stem}.heatmap.png")
    overlay = (img.astype(float) * 0.5 + rendered.astype(float) * 0.5).astype(np.uint8)
    Image.fromarray(overlay).save(out / f"{stem}.overlay.png")
    steps = [s["end"] for s in project["metadata"]["stages"]]
    snaps = dict(render_frames(project, steps))
    sheet = np.hstack([img] + [snaps[e] for e in steps])
    Image.fromarray(sheet).save(out / f"{stem}.stages.png")
    (out / f"{stem}.metrics.json").write_text(json.dumps(report, indent=2))
    if video:
        from .replay import save_video
        save_video(project, out / f"{stem}.replay.mp4", seconds=seconds, original=img)
    return {"project": project, "rendered": rendered, "metrics": report}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="img2draw")
    ap.add_argument("image")
    ap.add_argument("--mode", default="balanced", choices=["fast", "balanced", "high_fidelity", "research"])
    ap.add_argument("--out", default="out")
    ap.add_argument("--video", action="store_true", help="write the live-drawing replay mp4")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--tol", type=int, default=None, help="allowed max pixel error (0 = exact copy of source)")
    args = ap.parse_args(argv)
    result = reconstruct(args.image, mode=args.mode, out_dir=args.out, video=args.video, seconds=args.seconds, tol=args.tol)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
