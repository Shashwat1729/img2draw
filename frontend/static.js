// GitHub Pages backend: the Python engine runs in worker.js (Pyodide). Same API as the local server.
const worker = new Worker("worker.js");
let nid = 0;
const waiting = {};
worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "progress") return window.onPlanProgress && window.onPlanProgress(m.i, m.label);
  const w = waiting[m.id]; delete waiting[m.id];
  m.ok ? w.res(m.out) : w.rej(new Error(m.error));
};
const ask = (kind, args, transfer) => new Promise((res, rej) => { const id = ++nid; waiting[id] = {res, rej}; worker.postMessage({id, kind, args}, transfer || []); });
window.pyLoad = async (file) => { const buf = await file.arrayBuffer(); return ask("load", [buf], [buf]); };
window.pyCall = (route, query, body) => ask("call", [route, query, body]);
