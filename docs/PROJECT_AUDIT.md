# Inverse Drawing — Project Audit

Date: 2026-10-08

## Current state (pre-build)

The repository `D:\personal\projects\img2draw` was **empty** (no architecture,
language, dependencies, README, tests, or UI). Starting point: greenfield.

## What existed

- Nothing tracked; not a git repo.

## Missing components identified

- No drawing representation, renderer, metrics, editor, timeline, projects,
  benchmarks, docs, or licensing/attribution.

## Hardware/environment assumptions inspected

- Python 3.12 on Windows; numpy/opencv/PIL/sklearn/scipy/torch-cpu/fastapi
  installed. No vtracer, diffvg, lpips, scikit-image installed.
- torch preload prints a sitecustomize warning on every interpreter start
  (cosmetic; left as-is).
- `pip install` blocked in this environment → package is used via
  `PYTHONPATH=src` rather than `pip install -e .`.

## Research opportunities mapping

- LayerTracer → layer planning / sequential operations baseline.
- COVec → illumination decomposition concepts (shadow/highlight split).
- DiffVG → differentiable refinement hook (optional adapter).
- VTracer → vectorization baseline (optional adapter; default uses OpenCV).
- Video vectorization → animation architecture skeleton.

## Proposed architecture (implemented)

See README + docs/ARCHITECTURE.md. Key decision: hybrid representation with an
explicit signed-residual correction cascade to guarantee fidelity without
embedding the source image verbatim; vector/region layers remain editable and
independently removable.

## Technical risks

- Pure-vector fidelity ceiling → mitigated by correction cascade.
- SVG cannot express raster detail → export embeds raster layers (documented).
- Deterministic renderer tolerance → renderer is pure numpy/OpenCV, bit-exact
  for identical ops (tested).

## Migration plan

Greenfield: built `src/img2draw` package, editor, tests, benchmarks, docs,
vendored third_party submodules, git initialized, benchmark results recorded.
