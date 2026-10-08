"""Optional DiffVG refinement hook (requires diffvg build).

DiffVG is a differentiable renderer for vector paths. Where it is built we
could backprop path parameters against the source. When unavailable, the
iterative correction-layer refinement in pipeline.refine already provides
the render->compare->adjust loop, so the system is fully functional without
DiffVG. This adapter documents and isolates the optional dependency.
"""
from __future__ import annotations


def available() -> bool:
    try:
        import diffvg  # noqa
        return True
    except Exception:
        pass
    try:
        import pydiffvg  # noqa
        return True
    except Exception:
        return False


def optimize_path_colors(points, target_colors_means):  # pragma: no cover
    raise NotImplementedError("DiffVG refinement is optional; build diffvg to enable")
