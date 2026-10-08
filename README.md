# img2draw — Inverse Drawing / Editable Drawing Reconstruction

Given a raster image, img2draw infers an **editable, replayable drawing
program** (layers, regions, gradients, strokes, correction detail layers)
whose rendering reproduces the source image with very high fidelity.

The representation is *the means*; faithful reconstruction is the primary
objective, with editability/replayability/semantics making it research-worthy.

## Architecture

```
INPUT IMAGE
  -> decompose (gradient bg, KMeans region fills, shadow/highlight splits)
  -> drawing program (JSON project)
  -> renderer (deterministic numpy/OpenCV compositor)
  -> compare with source (MAE/MSE/PSNR/SSIM/MS-SSIM/edge/color/structural/LPIPS)
  -> error-guided refinement: cascade of signed-residual correction layers
  -> editable project -> editor UI -> timeline replay -> export (PNG/WebP/SVG/project)
```

Correction layers store `clip(source - render + 128)` and are composited
with `add_signed` blending. Each layer corrects ±127, so a converged
cascade reproduces the source within renderer quantization; removing them
exposes the pure structured (mostly vector) approximation.

## Layout

```
src/img2draw/        core package
  schema.py          versioned drawing-program schema + project save/load
  renderer.py        compositor + SVG/PNG/WebP export
  metrics.py         fidelity metrics + error maps
  decompose.py       image analysis/decomposition
  planner.py         drawing-plan generation (fast/balanced/high_fidelity)
  pipeline.py        end-to-end pipeline + refinement + CLI
  editor_server.py   FastAPI editor backend
  benchmark.py       benchmark runner -> CSV + HTML report
  animation.py       temporal/animation architecture skeleton
frontend/index.html  editor UI
tests/               pytest suite
examples/            synthetic benchmark images + generator
benchmarks/results/  measured results
third_party/         vendored research repos (submodules)
docs/                documentation
```

## Install & run

```bash
pip install -e .            # or: set PYTHONPATH=src
python examples/make_examples.py examples
python -m img2draw.pipeline examples/logo.png --mode balanced --out out
python -m img2draw.benchmark examples --out benchmarks/results
python -m img2draw.editor_server     # serves UI at http://127.0.0.1:8000
python -m pytest tests -q
```

## Modes

- `fast` — few regions, quick good-enough reconstruction
- `balanced` — regions + correction cascade (default)
- `high_fidelity` — more regions, finer correction
- `research` — alias of high_fidelity

The active mode is stored in the project metadata and displayed in the UI.

## Export

`PNG`, `WebP`, `SVG` (vector ops as real SVG; raster layers embedded as PNG —
SVG cannot faithfully express arbitrary raster detail), and the `.project.json`
format containing canvas, layers, operations, metadata, seed, renderer and
representation versions.

## Third-party research

Vendored for adapter/baseline use (see `THIRD_PARTY_NOTICES.md`):
- **vtracer** (MIT) — raster-to-vector baseline/candidate generator
- **diffvg** (Apache-2.0) — differentiable rendering (optional refinement)
- **LayerTracer** (MIT) — layered sequential design decomposition
- **COVec** (Apache-2.0) — illumination decomposition concepts

These are *wrapped, not copied*. The core system works without building them;
the installed OpenCV contour tracer is the default region vectorizer, and the
iterative correction cascade is the default refinement mechanism.

## Research contribution

Automatic recovery of a **structured sequence of editable drawing operations**
from a finished raster image — jointly considering reconstruction fidelity,
semantic structure, editability, drawing progression, and representation
complexity, with fidelity as the hard constraint.

## Limitations

- Vector layer reconstruction is approximate for noisy/photographic content;
  the correction cascade carries the difference (an honest detail layer).
- DiffVG-based gradient refinement and full LayerTracer/COVec integration are
  adapter/baseline points, not enabled by default.
- Animation is an architectural skeleton, not a finished temporal pipeline.

## License

MIT for this project's code. Vendored repos keep their own licenses (see
`THIRD_PARTY_NOTICES.md`).
