"""Drawing planner: image -> structured drawing program.

Mode controls the fidelity/complexity trade-off:
  fast         : few regions, no correction layer at low weight, quick
  balanced     : regions + correction layer, good fidelity
  high_fidelity: more regions, correction layer, finer residual tiles

Fidelity always wins: the Correction layer carries the residual so the
rendered program reconstructs the source. Vector/region layers give the
editable semantic structure on top of (and removable from) that base.
"""
from __future__ import annotations

import numpy as np

from . import decompose, schema
from .schema import new_layer, new_project, op_image_patch

MODE_K = {"fast": 4, "balanced": 6, "high_fidelity": 10, "research": 10}


def plan(img: np.ndarray, mode: str = "balanced", k: int | None = None) -> dict:
    h, w = img.shape[:2]
    project = new_project(w, h, mode=mode)
    project["metadata"]["source_size"] = [w, h]
    k = k or MODE_K.get(mode, 6)

    # 1. Background gradient
    bg_layer = new_layer("layer_bg", "Background", type_="vector")
    bg_layer["operations"].append(decompose.fit_background_gradient(img))
    project["layers"].append(bg_layer)

    # 2. Region structure
    labels, palette = decompose.kmeans_regions(img, k=k)
    region_layer = new_layer("layer_regions", "Structure", type_="vector")
    for pts, color in decompose.region_polygons(labels, palette):
        region_layer["operations"].append({"type": "FILL", "points": pts, "color": color, "opacity": 1.0})
    project["layers"].append(region_layer)

    # 3. Shading splits (editable raster shading layers)
    hi, lo = decompose.split_shading(img)
    sh_layer = new_layer("layer_shadows", "Shadows", type_="raster")
    hl_layer = new_layer("layer_highlights", "Highlights", type_="raster")
    if lo is not None:
        sh_layer["operations"].append(op_image_patch(0, 0, lo))
    if hi is not None:
        hl_layer["operations"].append(op_image_patch(0, 0, hi))
    project["layers"].append(sh_layer)
    project["layers"].append(hl_layer)

    # 4. Correction cascade is fitted by pipeline.refine (adds signed-residual
    # layers). It exists as a difference signal, NOT a copy of the source;
    # removing it exposes the pure structured approximation.

    project["metadata"]["progression"] = [
        "Background", "Structure", "Shadows", "Highlights", "Correction",
    ]
    return project


def op_count(project: dict) -> int:
    return sum(len(l["operations"]) for l in project["layers"])
