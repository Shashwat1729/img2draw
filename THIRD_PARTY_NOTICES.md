# Third-Party Notices

Vendored under `third_party/` as git submodules. Used as adapters/baselines;
not modified. Build them only if you enable the optional DiffVG/vtracer paths.

| Repository | Authors | License | Purpose | Modifications |
|---|---|---|---|---|
| vtracer (`visioncortex/vtracer`) | VisionCortex / TSANG Hao Fung et al. | MIT | raster-to-vector baseline, candidate generator, region vectorization | none |
| diffvg (`BachiLi/diffvg`) | Bachi Li, Tzu-Mao Li, Aaron Hertzmann, Frédo Durand | Apache-2.0 | differentiable renderer for vector paths (optional refinement) | none |
| LayerTracer (`showlab/LayerTracer`) | ShowLab group | MIT | layered/sequential design decomposition reference | none |
| COVec (`decade-de/COVec`) | decade-de | Apache-2.0 | illumination-aware decomposition (albedo/shade/light) reference | none |

Default pipeline does **not** require building any of these; it uses OpenCV
contour tracing + an iterative signed-residual correction cascade. Install the
optional dependencies to enable the adapters (`pip install -e .[vtracer,diffvg]`).
