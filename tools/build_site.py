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
          '<script src="static.js"></script>\n')
html = html.replace("<script>\nconst $=", inject + "<script>\nconst $=", 1)
(out / "index.html").write_text(html, encoding="utf-8")
for f in ("static.js", "worker.js"):
    shutil.copy(root / "frontend" / f, out / f)
(out / "docs").mkdir()
shutil.copy(root / "docs/naruto_drawing.gif", out / "docs/naruto_drawing.gif")
(out / ".nojekyll").write_text("")
print("built", out)
