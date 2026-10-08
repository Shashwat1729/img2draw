"""End-to-end pipeline: image -> drawing program -> render -> metrics -> refine."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import metrics, planner, schema
from .renderer import render_project


def refine(project: dict, img: np.ndarray, iterations: int = 4, tol: float = 1.5) -> dict:
    """Cascade of signed correction layers fitted to the remaining error.

    Each layer corrects up to +/-127; stacking few layers covers the full
    +/-255 range so a converged cascade reproduces the source exactly
    (within renderer quantization).
    """
    from .schema import new_layer, op_image_patch
    for it in range(iterations):
        rendered = render_project(project)
        err = img.astype(np.int16) - rendered.astype(np.int16)
        if int(np.abs(err).max()) <= tol:
            break
        corr = new_layer(f"layer_correction_{it + 1}", f"Correction {it + 1}", type_="raster")
        corr["blend_mode"] = "add_signed"
        patch = np.clip(err + 128, 0, 255).astype(np.uint8)
        corr["operations"].append(op_image_patch(0, 0, patch))
        project["layers"].append(corr)
    return project


def reconstruct(path: str, mode: str = "balanced", out_dir: str = "out", refine_iters: int = 3) -> dict:
    from .decompose import load_image
    img = load_image(path)
    project = planner.plan(img, mode=mode)
    project = refine(project, img, iterations=refine_iters)
    rendered = render_project(project)
    report = metrics.full_report(img, rendered)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    schema.save_project(project, out / f"{stem}.project.json")
    from PIL import Image
    Image.fromarray(rendered).save(out / f"{stem}.reconstruction.png")
    Image.fromarray(metrics.error_map(img, rendered)).save(out / f"{stem}.difference.png")
    Image.fromarray(metrics.error_heatmap(img, rendered)).save(out / f"{stem}.heatmap.png")
    overlay = (img.astype(float) * 0.5 + rendered.astype(float) * 0.5).astype(np.uint8)
    Image.fromarray(overlay).save(out / f"{stem}.overlay.png")
    (out / f"{stem}.metrics.json").write_text(json.dumps(report, indent=2))
    return {"project": project, "rendered": rendered, "metrics": report}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="img2draw")
    ap.add_argument("image")
    ap.add_argument("--mode", default="balanced", choices=["fast", "balanced", "high_fidelity", "research"])
    ap.add_argument("--out", default="out")
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args(argv)
    result = reconstruct(args.image, mode=args.mode, out_dir=args.out, refine_iters=args.iters)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
