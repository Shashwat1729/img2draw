# Installation

## From source (recommended)

```bash
git clone --recurse-submodules <repo-url>
cd img2draw
python -m pip install -e .
# or without install:
#   set PYTHONPATH=src        (Windows PowerShell: $env:PYTHONPATH='src')
```

Python 3.10+. Required: numpy, pillow, opencv-python-headless, scikit-learn,
scipy, fastapi, uvicorn, pyyaml.

## Docker

```bash
docker build -t img2draw .
docker run --rm -p 8000:8000 -v ${PWD}/examples:/app/examples img2draw
```

## Optional extras

```bash
pip install -e .[vtracer]   # VTracer vectorization baseline
pip install -e .[diffvg]    # differentiable refinement
pip install -e .[lpips]     # learned perceptual metric
pip install -e .[dev]       # pytest
```

Models: none required by default. VTracer/DiffVG build steps are only needed
for their optional adapter paths; see third_party/*/README.md.
