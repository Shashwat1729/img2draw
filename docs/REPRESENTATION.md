# Representation

Versioned JSON drawing program (`version: "1.0"`):

```json
{
  "version": "1.0",
  "canvas": {"width": 128, "height": 128, "color_space": "sRGB"},
  "renderer_version": "1.0",
  "representation_version": "1.0",
  "color_profile": "sRGB",
  "seed": 0,
  "mode": "balanced",
  "metadata": {"progression": ["Background", "Structure", "Shadows", "Highlights"]},
  "layers": [
    {"id": "layer_bg", "name": "Background", "type": "vector",
     "visible": true, "opacity": 1.0, "blend_mode": "normal", "locked": false,
     "operations": [{"type": "GRADIENT", "direction": "vertical",
                      "stops": [[0.0,[10,10,10]],[1.0,[200,200,200]]], "opacity": 1.0}]}
  ]
}
```

Operation types: `PATH`, `BRUSH_STROKE`, `FILL`, `GRADIENT`, `ERASE`, `MASK`,
`TRANSFORM`, `IMAGE_PATCH`, `TEXTURE`, `FILTER`, `GROUP`.

- `FILL` / `PATH`: `{points: [[x,y]...], color: [r,g,b], width, closed, opacity}`
- `BRUSH_STROKE`: `{points, pressure[], width, opacity, color, brush}` — point list
  with per-point pressure, width, spacing, opacity; deterministic line+circle joins.
- `IMAGE_PATCH` / `TEXTURE`: base64-PNG payload + x/y (+ optional mask).
- `GRADIENT`: `{stops: [[pos,[r,g,b]]], direction, opacity}`.
- `ERASE`: polygon mask, applied destination-out.
- `TRANSFORM`: affine matrix on a raster patch.
- `MASK`: luminance alpha.
- `FILTER`: blur/sharpen on a raster patch.
- `GROUP`: nested operations.

Every operation carries enough fields to reproduce itself deterministically.
