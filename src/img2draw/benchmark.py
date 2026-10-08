"""Benchmark: run reconstruction over a directory of images, write CSV + HTML."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .decompose import load_image
from .pipeline import reconstruct
from .planner import op_count
from .renderer import render_project, export_svg


def run(images_dir: str, out_dir: str, mode: str = "balanced") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(Path(images_dir).glob("*")):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        t0 = time.time()
        res = reconstruct(str(p), mode=mode, out_dir=str(out / p.stem))
        dt = time.time() - t0
        layer_count = len(res["project"]["layers"])
        rows.append({"image": p.name, "mode": mode, "seconds": round(dt, 2),
                     "operations": op_count(res["project"]), "layers": layer_count,
                     **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["metrics"].items()}})
    csv_path = out / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    html = ["<html><body><h1>Benchmark</h1><table border=1><tr>" +
            "".join(f"<th>{k}</th>" for k in rows[0]) + "</tr>"]
    for r in rows:
        html.append("<tr>" + "".join(f"<td>{v}</td>" for v in r.values()) + "</tr>")
    html.append("</table></body></html>")
    (out / "report.html").write_text("\n".join(html))
    return csv_path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="img2draw-bench")
    ap.add_argument("images")
    ap.add_argument("--out", default="benchmarks/results")
    ap.add_argument("--mode", default="balanced")
    a = ap.parse_args(argv)
    print(run(a.images, a.out, a.mode))


if __name__ == "__main__":
    main()
