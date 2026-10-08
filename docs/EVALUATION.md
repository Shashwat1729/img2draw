# Evaluation

For every reconstruction report: MAE, MSE, PSNR, SSIM, MS-SSIM,
color difference, edge difference, structural difference, LPIPS (if
installed), operation count, layer count, runtime.

Commands:

```bash
python -m img2draw.pipeline examples/logo.png --mode balanced --out out
python -m img2draw.benchmark examples --out benchmarks/results --mode balanced
# outputs benchmarks/results/results.csv and report.html
```

Editing-evaluation harness (planned example edits): recolor an object,
remove a shadow, delete a stroke, change opacity, move an object, modify a
layer, undo, redo — each mutating the project and re-rendering.

Deterministic replay test: `render(project)` twice → identical pixels
(`tests/test_core.py::test_renderer_deterministic`).
