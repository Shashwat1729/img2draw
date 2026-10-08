// Pyodide in a Web Worker so the page stays responsive while the drawing is planned.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js");
let api = null;
self.pyProgress = (i, label) => postMessage({type: "progress", i, label});
async function init() {
  if (api) return api;
  postMessage({type: "progress", i: -1, label: "Loading the drawing engine (first visit only)"});
  const py = await loadPyodide();
  await py.loadPackage(["numpy", "opencv-python", "scikit-learn", "pillow"]);
  const mods = await (await fetch("modules.json")).json();
  py.FS.mkdirTree("/home/pyodide/img2draw");
  for (const m of ["__init__", ...mods])
    py.FS.writeFile(`/home/pyodide/img2draw/${m}.py`, await (await fetch(`py/img2draw/${m}.py`)).text());
  py.runPython("import sys; sys.path.insert(0,'/home/pyodide'); from img2draw import web_api");
  api = py.pyimport("img2draw.web_api");
  return api;
}
onmessage = async (e) => {
  const {id, kind, args} = e.data;
  try {
    const a = await init();
    let out;
    if (kind === "load") out = a.load(new Uint8Array(args[0])).toJs({dict_converter: Object.fromEntries});
    else out = JSON.parse(a.call(args[0], JSON.stringify(args[1] || {}), JSON.stringify(args[2] ?? null)));
    postMessage({id, ok: true, out});
  } catch (err) { postMessage({id, ok: false, error: String(err)}); }
};
