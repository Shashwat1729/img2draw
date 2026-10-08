"""Build the static GitHub Pages site into site/ (Python runs in-browser via Pyodide)."""
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / "site"
shutil.rmtree(out, ignore_errors=True)
(out / "py" / "img2draw").mkdir(parents=True)
MODULES = ["schema", "renderer", "artist", "web_api"]
for m in MODULES:
    shutil.copy(root / "src/img2draw" / f"{m}.py", out / "py/img2draw" / f"{m}.py")
(out / "py/img2draw/__init__.py").write_text("")
(out / "modules.json").write_text(str(MODULES).replace("'", '"'))
html = (root / "frontend/index.html").read_text(encoding="utf-8")
inject = ('<script>window.STATIC_MODE=true</script>\n'
          '<script src="https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js"></script>\n'
          '<script src="static.js"></script>\n')
html = html.replace("<script>\nconst $=", inject + "<script>\nconst $=", 1)
(out / "index.html").write_text(html, encoding="utf-8")
shutil.copy(root / "frontend/static.js", out / "static.js")
for f in ("luffy", "tanjiro"):
    pass
(out / ".nojekyll").write_text("")
print("built", out)
