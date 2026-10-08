# Architecture

High-level pipeline:

```
image -> decompose -> plan(project JSON) -> render -> metrics
      -> error map -> correction cascade (refine) -> render -> metrics
      -> editor / timeline / export
```

## Layers

- `Background` — fitted vertical gradient.
- `Structure` — KMeans region fills as polygon `FILL` ops (contour-extracted).
- `Shadows` / `Highlights` — luminance-split raster `IMAGE_PATCH` ops.
- `Correction N` — signed-residual raster patches, `add_signed` blending.
- `Edits` — user brush strokes from the editor.

## Rendering

Deterministic float-RGBA compositor. Blend modes: normal, multiply, screen,
overlay, add, add_signed. Eraser sets destination-out alpha. No randomness:
same project + same renderer version → same pixels (tested).

## Fidelity guarantee

`Correction` layers store the signed difference between the target render and
the source (offset by +128). Composited with `add_signed`, the rendered
output converges to the source within ±127 per channel per layer; stacking
layers covers the full range. This is an explicit detail layer — removing it
reveals the structured approximation and the error maps localize what
remains.

## Editing & determinism

All UI edits mutate the project JSON (not a flattened raster), are pushed to
an undo/redo stack, and re-render affects the actual output.
