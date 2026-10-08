# API

## CLI

```bash
python -m img2draw.pipeline <image> [--mode fast|balanced|high_fidelity|research] [--out DIR] [--iters N]
python -m img2draw.benchmark <images_dir> [--out DIR] [--mode MODE]
python -m img2draw.editor_server   # FastAPI on :8000
```

## HTTP (editor server)

- `POST /api/reconstruct?mode=...` (multipart file) → metrics + first render
- `GET /api/render?max_ops=N` → timeline state
- `GET /api/compare` → original/reconstruction/difference/heatmap/overlay + metrics
- `POST /api/layer/{id}?visible=&opacity=&blend_mode=&name=`
- `POST /api/layer/{id}/delete`
- `POST /api/layers/reorder` (JSON body: list of ids)
- `POST /api/edit/brush` (JSON: points, color, width, opacity)
- `POST /api/undo` / `/api/redo`
- `POST /api/project/save?path=` / `POST /api/project/load?path=`
- `POST /api/export/{png|webp|svg|project}?path=`

## Python

```python
from img2draw.pipeline import reconstruct
from img2draw.renderer import render_project, export_svg
from img2draw import schema, metrics

res = reconstruct("examples/logo.png", mode="balanced", out_dir="out")
print(res["metrics"])
```
