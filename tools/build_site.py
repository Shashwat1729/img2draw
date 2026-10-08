"""Build the static GitHub Pages site into site/ (Python runs in-browser via Pyodide)."""
import hashlib
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
ver = hashlib.sha1(b"".join((root / f).read_bytes() for f in ["frontend/index.html", "frontend/static.js", "frontend/worker.js"] + [f"src/img2draw/{m}.py" for m in MODULES])).hexdigest()[:10]
inject = ('<script>window.STATIC_MODE=true</script>\n'
          '<script src="static.js"></script>\n')
html = html.replace("<script>\nconst $=", inject + "<script>\nconst $=", 1)
(out / "index.html").write_text(html, encoding="utf-8")
(out / "static.js").write_text((root / "frontend/static.js").read_text(encoding="utf-8").replace('"worker.js"', f'"worker.js?v={ver}"'), encoding="utf-8")
(out / "worker.js").write_text((root / "frontend/worker.js").read_text(encoding="utf-8")
                               .replace('"modules.json"', f'"modules.json?v={ver}"').replace(".py`)", f".py?v={ver}`)"), encoding="utf-8")
(out / "docs").mkdir()
shutil.copy(root / "docs/naruto_drawing.gif", out / "docs/naruto_drawing.gif")
(out / ".nojekyll").write_text("")
print("built", out)
