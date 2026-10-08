// Runs the Python backend (src/img2draw/web_api.py) in the browser with Pyodide.
let _py = null, _api = null;
async function initPy() {
  if (_api) return _api;
  _py = await loadPyodide();
  await _py.loadPackage(["numpy", "opencv-python", "scikit-learn", "pillow"]);
  const mods = await (await fetch("modules.json")).json();
  _py.FS.mkdirTree("/home/pyodide/img2draw");
  for (const m of ["__init__", ...mods]) {
    const src = await (await fetch(`py/img2draw/${m}.py`)).text();
    _py.FS.writeFile(`/home/pyodide/img2draw/${m}.py`, src);
  }
  _py.runPython("import sys; sys.path.insert(0,'/home/pyodide'); from img2draw import web_api");
  _api = _py.pyimport("img2draw.web_api");
  return _api;
}
window.pyLoad = async (file) => {
  const api = await initPy();
  const buf = new Uint8Array(await file.arrayBuffer());
  await new Promise(r => setTimeout(r, 30));          // let the "working" message paint
  return JSON.parse(JSON.stringify(api.load(buf).toJs({dict_converter: Object.fromEntries})));
};
window.pyCall = async (route, query, body) => {
  const api = await initPy();
  return JSON.parse(api.call(route, JSON.stringify(query || {}), JSON.stringify(body ?? null)));
};
