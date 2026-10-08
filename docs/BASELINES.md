# Baselines

| Baseline | What it is | Status in this repo |
|---|---|---|
| VTracer | raster→SVG tracer | vendored submodule; adapter hook in `decompose.vtracer_available/trace_with_vtracer`; optional extra |
| DiffVG optimization | differentiable vector renderer | vendored submodule; adapter stub `diffvg_adapter.py`; optional extra |
| LayerTracer | layered SVG synthesis | vendored for reference; ideas inform layer planner |
| COVec | illumination-decomposed vectorization | vendored for reference; shadow/highlight split inspired by it |
| naive raster→SVG | direct tracing | covered by VTracer path when enabled |
| raster baseline | identity "reconstruction" | correction cascade reduces to it if vector layers are removed |

We do not claim superiority — measured comparisons should be run via
`img2draw.benchmark` on the same image set.
