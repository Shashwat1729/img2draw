# Development

- `PYTHONPATH=src` (or `pip install -e .`) then `python -m img2draw.pipeline`.
- Tests: `python -m pytest tests -q`.
- Editor: `python -m img2draw.editor_server` → http://127.0.0.1:8000.
- Benchmark: `python -m img2draw.benchmark examples --out benchmarks/results`.
- Deterministic render, no network calls in the core path.
- No fake functionality: every UI control mutates the project JSON and
  re-renders; metrics are computed, never hard-coded.
