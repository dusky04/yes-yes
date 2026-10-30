"""
build_report.py — render results/ into one self-contained HTML page: results/index.html.

Every training run with its per-epoch curves, per-class scores and confusion matrix,
plus the comparison charts and tables of Experiments 1-7. All data is embedded, so
the file opens offline. fetch_results.py calls this at the end; to rebuild alone:

    uv run --no-project --with matplotlib python benchmark/build_report.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

from fetch_results import ANALYSIS, ROOT, SOURCES, config_specs, headline_acc, row_for

LOG_EPOCH = re.compile(r"epoch (\d+)/(\d+) \| train loss ([\d.]+) acc ([\d.]+) \| "
                       r"val loss ([\d.]+) acc ([\d.]+) F1 ([\d.]+) \| (\d+)s")


def history(folder: Path) -> list[dict]:
    """Per-epoch history: history.csv once a run finished, else parsed from its log."""
    f = folder / "history.csv"
    if f.exists():
        rows = list(csv.DictReader(open(f)))
        return [{k: (int(v) if k == "epoch" else round(float(v), 5)) for k, v in r.items()} for r in rows]
    log = folder / "train.log"
    if not log.exists():
        return []
    out = {}
    for m in LOG_EPOCH.finditer(log.read_text(errors="replace")):
        e = int(m[1])
        out[e] = {"epoch": e, "train_loss": float(m[3]), "train_acc": float(m[4]),
                  "val_loss": float(m[5]), "val_acc": float(m[6]), "val_macro_f1": float(m[7]),
                  "seconds": float(m[8])}
    return [out[e] for e in sorted(out)]


def confusion(folder: Path):
    f = folder / "test_confusion_matrix.csv"
    if not f.exists():
        return None
    rows = list(csv.reader(open(f)))
    return {"classes": rows[0][1:], "matrix": [[int(x) for x in r[1:]] for r in rows[1:]]}


def settings(folder: Path) -> dict:
    f = folder / "results.json"
    if not f.exists():
        return {}
    r = json.loads(f.read_text())
    c, a = r["config"], r["args"]
    return {
        "batch_size": c["batch_size"], "hidden_dim": c["hidden_dim"], "epochs": c["epochs"],
        "lr": c["lr"], "weight_decay": c["weight_decay"],
        "augment": a.get("augment", False), "temporal_jitter": a.get("temporal_jitter", False),
        "label_smoothing": a.get("label_smoothing", 0.0), "schedule": a.get("schedule", "constant"),
        "ema": a.get("ema_decay", 0.0), "tta": a.get("tta", False),
        "dataset": Path(a.get("data_root", "")).name or "CricketEC",
        "per_class": r.get("test_per_class", {}),
        "per_class_source": r.get("test_per_class_source", "test"),
        "test_sha": r["test_set"]["sha256"][:12],
    }


def collect_runs(out: Path) -> list[dict]:
    specs = config_specs()
    runs = []
    for remote_dir, section, sub, title, desc in SOURCES:
        base = out / section / sub if sub else out / section
        if not base.exists():
            continue
        exp = int(section[:2])
        for folder in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("exp")):
            cfg = folder.name.split("_", 1)[0]
            row = row_for(cfg, folder, specs)
            token = (f"r{sub[3]}-{cfg}" if sub else cfg)
            runs.append({
                "token": token, "experiment": exp, "group": title, "group_desc": desc,
                "run": sub or "", "config": cfg, "row": row,
                "headline": headline_acc(row) if row["status"] == "done" else None,
                "history": history(folder), "confusion": confusion(folder),
                "settings": settings(folder),
                "source": str(folder.relative_to(out)),
            })
    return runs


def collect_analysis(out: Path) -> list[dict]:
    summary = out / "01_exp1_benchmark_A-E" / "exp1_test_eval" / "test_summary.csv"
    models = [r["model"] for r in csv.DictReader(open(summary))] if summary.exists() else []
    items = []
    for remote_dir, local, title, tables in ANALYSIS:
        d = out / local
        found = {t: list(csv.DictReader(open(d / t))) for t in tables if (d / t).exists()}
        status = "pending" if not found else ("trial" if len(models) < 5 else "done")
        items.append({"key": local.split("/")[-1], "title": title, "status": status,
                      "models": models, "tables": found, "source": local})
    return items


def build(out: Path, artifact: Path | None = None) -> Path:
    """Write out/index.html (a complete document that opens from disk). With `artifact`,
    also write the bare page body that the Artifact publisher wraps itself."""
    runs = collect_runs(out)
    data = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "runs": runs,
        "analysis": collect_analysis(out),
    }
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.replace("/*__DATA__*/", blob)
    head, body = page.split("</style>\n", 1)
    doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
           f'{head}</style>\n</head>\n<body>\n{body}</body>\n</html>\n')
    path = out / "index.html"
    path.write_text(doc)
    if artifact:
        artifact.write_text(page)
    return path


TEMPLATE = r"""<title>CricketEC Experiments</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a single reading column of numbered experiment sections, led by a run index and a run explorer. */
:root {
  --bg: #f6f6f3; --surface: #fcfcfb; --raise: #ffffff;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #75736d;
  --grid: #e1e0d9; --axis: #c3c2b7; --line: rgba(11,11,11,.10);
  --s1: #2a78d6; --s2: #eb6834; --accent: #2a78d6;
  --good: #0c7a0c; --warn: #b07800; --pend: #75736d;
  --font: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0d0d0d; --surface: #1a1a19; --raise: #212120;
  --ink: #f0efec; --ink-2: #c3c2b7; --muted: #9a988f;
  --grid: #2c2c2a; --axis: #383835; --line: rgba(255,255,255,.10);
  --s1: #3987e5; --s2: #d95926; --accent: #5598e7;
  --good: #3fbf3f; --warn: #e0a520; --pend: #9a988f; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #0d0d0d; --surface: #1a1a19; --raise: #212120;
  --ink: #f0efec; --ink-2: #c3c2b7; --muted: #9a988f;
  --grid: #2c2c2a; --axis: #383835; --line: rgba(255,255,255,.10);
  --s1: #3987e5; --s2: #d95926; --accent: #5598e7;
  --good: #3fbf3f; --warn: #e0a520; --pend: #9a988f; color-scheme: dark }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.55 var(--font); }
[hidden] { display: none !important; }
.wrap { max-width: 1120px; margin: 0 auto; padding-block: 28px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
h1 { font-size: 30px; font-weight: 600; letter-spacing: -.01em; margin: 0; text-wrap: balance; }
h2 { font-size: 21px; font-weight: 600; margin: 0; text-wrap: balance; }
h3 { font-size: 15px; font-weight: 600; margin: 0; }
p { margin: 0; max-width: 72ch; }
a { color: var(--accent); }
.mono, td.num, .tick { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.lede { color: var(--ink-2); margin-top: 6px; }
.meta { color: var(--muted); font-size: 13px; margin-top: 10px; display: flex; flex-wrap: wrap; gap: 4px 16px; }
nav { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 18px; }
nav a { font-size: 13px; text-decoration: none; color: var(--ink-2); border: 1px solid var(--line);
  padding: 4px 10px; border-radius: 999px; background: var(--surface); }
nav a:hover, nav a:focus-visible { color: var(--ink); border-color: var(--axis); }
section { margin-top: 44px; display: grid; gap: 14px; }
.eyebrow { font: 500 12px/1 var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
.head { display: grid; gap: 6px; }
.panel { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 16px; min-width: 0; }
.grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
@media (max-width: 760px) { .grid2 { grid-template-columns: 1fr; } }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { font-weight: 500; color: var(--ink-2); font-size: 12px; }
td.num, th.num { text-align: right; }
tr.sel td { background: color-mix(in srgb, var(--accent) 9%, transparent); }
button.linkish { font: inherit; color: var(--accent); background: none; border: 0; padding: 0; cursor: pointer; text-decoration: underline; text-underline-offset: 2px; }
button:focus-visible, select:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.pill { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--ink-2); }
.pill::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--pend); }
.pill.done::before { background: var(--good); }
.pill.running::before { background: var(--warn); }
.source { font-size: 12px; color: var(--muted); }
.source code { font-family: var(--mono); font-size: 11.5px; }
.picker { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
select { font: inherit; font-size: 14px; color: var(--ink); background: var(--raise); border: 1px solid var(--axis);
  border-radius: 8px; padding: 6px 10px; max-width: 100%; }
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; }
.fact { border-left: 2px solid var(--grid); padding-left: 10px; }
.fact b { display: block; font: 500 20px/1.2 var(--font); font-variant-numeric: tabular-nums; }
.fact span { font-size: 12px; color: var(--muted); }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { font: 12px/1 var(--mono); color: var(--ink-2); border: 1px solid var(--line); border-radius: 6px; padding: 5px 7px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12px; color: var(--ink-2); margin-bottom: 6px; }
.legend i { display: inline-block; width: 14px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 6px; }
.legend i.sq { width: 10px; height: 10px; border-radius: 2px; }
.chart { width: 100%; position: relative; }
.chart svg { display: block; width: 100%; height: auto; overflow: visible; }
.chart text { fill: var(--muted); font: 11px var(--mono); }
.chart .lbl { fill: var(--ink); font: 11px var(--mono); }
.chart .axis { stroke: var(--axis); stroke-width: 1; }
.chart .gridl { stroke: var(--grid); stroke-width: 1; }
.chart .hair { stroke: var(--ink-2); stroke-width: 1; opacity: .5; }
.note { font-size: 13px; color: var(--ink-2); border-left: 2px solid var(--warn); padding-left: 10px; }
details summary { cursor: pointer; font-size: 13px; color: var(--ink-2); }
details[open] summary { margin-bottom: 8px; }
#tip { position: fixed; z-index: 20; pointer-events: none; background: var(--raise); color: var(--ink);
  border: 1px solid var(--line); border-radius: 8px; padding: 7px 9px; font: 12px/1.45 var(--mono);
  box-shadow: 0 4px 14px rgba(0,0,0,.12); max-width: 260px; }
#tip .k { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.empty { color: var(--muted); font-size: 13px; padding: 18px 0; }
@media (prefers-reduced-motion: no-preference) { nav a, button { transition: color .15s, border-color .15s; } }
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">CricketEC · 15-class stroke &amp; outcome recognition</div>
    <h1>CricketEC Experiments</h1>
    <p class="lede">Training runs and results for the CNN–RNN benchmark (configs A–E) and Experiments 2–7. Test scores are on the fixed 375-clip test set, 25 clips per class; the best epoch is always chosen on validation.</p>
    <div class="meta" id="meta"></div>
    <nav id="nav"></nav>
  </header>

  <section id="runs">
    <div class="head"><div class="eyebrow">Index</div><h2>All training runs</h2>
      <p class="lede">Pick a run to see its per-epoch curves below. Test accuracy is with test-time augmentation where the run used it.</p></div>
    <div class="panel scroll"><table id="runs-table"></table></div>
  </section>

  <section id="curves">
    <div class="head"><div class="eyebrow">Run explorer</div><h2>Per-epoch training curves</h2></div>
    <div class="picker"><label for="run-select" class="source">Run</label><select id="run-select"></select><span id="run-status"></span></div>
    <div id="run-detail" style="display:grid;gap:14px"></div>
  </section>

  <div id="experiments"></div>
</div>
<div id="tip" hidden></div>

<script type="application/json" id="data">/*__DATA__*/</script>
<script>
(() => {
const D = JSON.parse(document.getElementById("data").textContent);
const $ = (s, r = document) => r.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const pct = (v, d = 2) => v == null || v === "" ? "–" : (100 * v).toFixed(d);
const fix = (v, d = 4) => v == null || v === "" ? "–" : Number(v).toFixed(d);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const statusPill = s => s === "done" ? '<span class="pill done">done</span>'
  : s.startsWith("running") ? `<span class="pill running">${esc(s.replace("running/stopped at", "at"))}</span>`
  : `<span class="pill">${esc(s)}</span>`;
const EXP_TITLES = {1: "Benchmark configurations A–E", 2: "Frame-sampling ablation", 3: "Frame-count sensitivity",
  4: "Camera-perspective robustness", 5: "Computational efficiency", 6: "Semantic-layer error analysis", 7: "Statistical reliability"};

// ---------- tooltip ----------
const tip = $("#tip");
function showTip(ev, html) {
  tip.innerHTML = html; tip.hidden = false;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
  if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => { tip.hidden = true; };
const key = c => `<span class="k" style="background:${c}"></span>`;

// ---------- svg helpers ----------
function svg(w, h) { const s = document.createElementNS(NS, "svg"); s.setAttribute("viewBox", `0 0 ${w} ${h}`); s.setAttribute("role", "img"); return s; }
function el(tag, attrs, parent, text) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || mag * 10;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
// bar with a 4px rounded data end, square at the baseline
function barPath(x, y0, w, y1, r = 4) {
  const h = y0 - y1; if (h <= 0) return ""; r = Math.min(r, w / 2, h);
  return `M${x},${y0} V${y1 + r} Q${x},${y1} ${x + r},${y1} H${x + w - r} Q${x + w},${y1} ${x + w},${y1 + r} V${y0} Z`;
}
function hbarPath(x0, y, x1, h, r = 4) {
  const w = x1 - x0; if (w <= 0) return ""; r = Math.min(r, h / 2, w);
  return `M${x0},${y} H${x1 - r} Q${x1},${y} ${x1},${y + r} V${y + h - r} Q${x1},${y + h} ${x1 - r},${y + h} H${x0} Z`;
}
const charts = [];
function mount(container, draw) { const c = { container, draw }; charts.push(c); draw(container); return c; }
let rsz; addEventListener("resize", () => { clearTimeout(rsz); rsz = setTimeout(() => charts.forEach(c => c.container.isConnected && c.draw(c.container)), 120); });

// line chart: series [{name, color, pts:[[x,y]]}], optional marker {x,label}
function lineChart(box, o) {
  box.innerHTML = "";
  const legend = document.createElement("div"); legend.className = "legend";
  if (o.series.length > 1) legend.innerHTML = o.series.map(s => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("");
  box.appendChild(legend);
  const W = Math.max(280, box.clientWidth), H = o.height || 230, m = { l: 46, r: 14, t: 10, b: 34 };
  const s = svg(W, H); box.appendChild(s);
  const all = o.series.flatMap(x => x.pts);
  if (!all.length) { box.insertAdjacentHTML("beforeend", '<div class="empty">No epochs logged yet.</div>'); return; }
  const xs = all.map(p => p[0]), ys = all.map(p => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let [y0, y1] = o.yDomain || [Math.min(...ys), Math.max(...ys)];
  if (!o.yDomain) { const pad = (y1 - y0) * .08 || .05; y0 = Math.max(o.yMin ?? -Infinity, y0 - pad); y1 = y1 + pad; }
  const X = v => m.l + (x1 === x0 ? .5 : (v - x0) / (x1 - x0)) * (W - m.l - m.r);
  const Y = v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
  for (const t of niceTicks(y0, y1, 4)) {
    el("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "gridl" }, s);
    el("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end", class: "tick" }, s, o.yFmt ? o.yFmt(t) : t);
  }
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }, s);
  const xt = o.xTicks || niceTicks(x0, x1, Math.min(8, x1 - x0 || 1)).filter(Number.isInteger);
  for (const t of xt) el("text", { x: X(t), y: H - m.b + 16, "text-anchor": "middle", class: "tick" }, s, t);
  if (o.xLabel) el("text", { x: W - m.r, y: H - 4, "text-anchor": "end" }, s, o.xLabel);
  if (o.marker) {
    el("line", { x1: X(o.marker.x), x2: X(o.marker.x), y1: m.t, y2: H - m.b, stroke: css("--axis"), "stroke-width": 1 }, s);
    el("text", { x: X(o.marker.x) + 4, y: m.t + 10, class: "tick" }, s, o.marker.label);
  }
  for (const se of o.series) {
    if (!se.pts.length) continue;
    el("path", { d: se.pts.map((p, i) => `${i ? "L" : "M"}${X(p[0])},${Y(p[1])}`).join(""), fill: "none",
      stroke: se.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, s);
    const showDots = o.dots || se.pts.length <= 8;
    const pts = showDots ? se.pts : [se.pts[se.pts.length - 1]];
    for (const p of pts) el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 4, fill: se.color, stroke: css("--surface"), "stroke-width": 2 }, s);
    if (o.endLabels) { const p = se.pts[se.pts.length - 1]; el("text", { x: X(p[0]), y: Y(p[1]) - 10, "text-anchor": "middle", class: "lbl" }, s, o.yFmt ? o.yFmt(p[1]) : p[1]); }
  }
  // hover: crosshair at the nearest x, values of every series
  const xsU = [...new Set(xs)].sort((a, b) => a - b);
  const hair = el("line", { y1: m.t, y2: H - m.b, class: "hair", visibility: "hidden" }, s);
  const dots = o.series.map(se => el("circle", { r: 4.5, fill: se.color, stroke: css("--surface"), "stroke-width": 2, visibility: "hidden" }, s));
  const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, s);
  hit.addEventListener("mousemove", ev => {
    const r = s.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width;
    const xv = xsU.reduce((a, b) => Math.abs(X(b) - px) < Math.abs(X(a) - px) ? b : a);
    hair.setAttribute("x1", X(xv)); hair.setAttribute("x2", X(xv)); hair.setAttribute("visibility", "visible");
    const lines = [`<b>${esc(o.xName || "x")} ${xv}</b>`];
    o.series.forEach((se, i) => {
      const p = se.pts.find(q => q[0] === xv);
      if (!p) { dots[i].setAttribute("visibility", "hidden"); return; }
      dots[i].setAttribute("cx", X(p[0])); dots[i].setAttribute("cy", Y(p[1])); dots[i].setAttribute("visibility", "visible");
      lines.push(`${key(se.color)}${esc(se.name)}: ${o.yFmt ? o.yFmt(p[1]) : p[1]}`);
    });
    showTip(ev, lines.join("<br>"));
  });
  hit.addEventListener("mouseleave", () => { hair.setAttribute("visibility", "hidden"); dots.forEach(d => d.setAttribute("visibility", "hidden")); hideTip(); });
}

// grouped columns: cats [names], series [{name,color,vals:[v|null]}]
function columnChart(box, o) {
  box.innerHTML = "";
  const legend = document.createElement("div"); legend.className = "legend";
  if (o.series.length > 1) legend.innerHTML = o.series.map(s => `<span><i class="sq" style="background:${s.color}"></i>${esc(s.name)}</span>`).join("");
  box.appendChild(legend);
  const W = Math.max(280, box.clientWidth), H = o.height || 240, m = { l: 46, r: 10, t: 18, b: 40 };
  const s = svg(W, H); box.appendChild(s);
  const [y0, y1] = o.yDomain;
  const Y = v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
  for (const t of niceTicks(y0, y1, 4)) {
    el("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "gridl" }, s);
    el("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end", class: "tick" }, s, o.yFmt(t));
  }
  const band = (W - m.l - m.r) / o.cats.length, n = o.series.length;
  const bw = Math.min(24, (band * .7 - 2 * (n - 1)) / n);
  o.cats.forEach((cat, ci) => {
    const cx = m.l + band * (ci + .5), start = cx - (n * bw + (n - 1) * 2) / 2;
    el("text", { x: cx, y: H - m.b + 18, "text-anchor": "middle", class: "lbl" }, s, cat);
    o.series.forEach((se, si) => {
      const v = se.vals[ci]; if (v == null) return;
      const x = start + si * (bw + 2);
      const p = el("path", { d: barPath(x, Y(y0), bw, Y(v)), fill: se.color }, s);
      el("text", { x: x + bw / 2, y: Y(v) - 5, "text-anchor": "middle", class: "lbl" }, s, o.yFmt(v));
      const hitR = el("rect", { x: x - 2, y: m.t, width: bw + 4, height: H - m.t - m.b, fill: "transparent" }, s);
      hitR.addEventListener("mousemove", ev => showTip(ev, `<b>${esc(cat)}</b><br>${key(se.color)}${esc(se.name)}: ${o.yFmt(v)}${o.extra ? "<br>" + o.extra(ci, si) : ""}`));
      hitR.addEventListener("mouseleave", hideTip);
    });
  });
  el("line", { x1: m.l, x2: W - m.r, y1: Y(y0), y2: Y(y0), class: "axis" }, s);
}

// horizontal bars, one row per item: items [{label, v, color, group}]
function rowBars(box, o) {
  box.innerHTML = "";
  if (o.legend) { const lg = document.createElement("div"); lg.className = "legend";
    lg.innerHTML = o.legend.map(([n, c]) => `<span><i class="sq" style="background:${c}"></i>${esc(n)}</span>`).join(""); box.appendChild(lg); }
  const rowH = 20, W = Math.max(280, box.clientWidth), m = { l: 86, r: 52, t: 4, b: 22 };
  const H = m.t + m.b + rowH * o.items.length, s = svg(W, H); box.appendChild(s);
  const X = v => m.l + v * (W - m.l - m.r);
  for (const t of [0, .25, .5, .75, 1]) {
    el("line", { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b, class: "gridl" }, s);
    el("text", { x: X(t), y: H - 6, "text-anchor": "middle", class: "tick" }, s, t.toFixed(2));
  }
  o.items.forEach((it, i) => {
    const y = m.t + i * rowH + 4, h = rowH - 8;
    el("text", { x: m.l - 8, y: y + h - 1, "text-anchor": "end", class: "lbl" }, s, it.label);
    el("path", { d: hbarPath(X(0), y, X(it.v), h), fill: it.color }, s);
    el("text", { x: X(it.v) + 6, y: y + h - 1, class: "lbl" }, s, it.v.toFixed(3));
    const hr = el("rect", { x: 0, y: y - 4, width: W, height: rowH, fill: "transparent" }, s);
    hr.addEventListener("mousemove", ev => showTip(ev, `<b>${esc(it.label)}</b> (${esc(it.group)})<br>${it.tip}`));
    hr.addEventListener("mouseleave", hideTip);
  });
}

// confusion matrix heatmap
function heatmap(box, cm) {
  box.innerHTML = "";
  const n = cm.classes.length, W = Math.max(300, box.clientWidth), m = { l: 74, t: 74, r: 4, b: 4 };
  const cell = Math.max(12, Math.min(30, (W - m.l - m.r) / n)), H = m.t + m.b + cell * n;
  const s = svg(m.l + m.r + cell * n, H); s.style.maxWidth = (m.l + m.r + cell * n) + "px"; box.appendChild(s);
  const max = Math.max(1, ...cm.matrix.flat()), c1 = css("--s1");
  cm.classes.forEach((c, i) => {
    el("text", { x: m.l - 6, y: m.t + i * cell + cell / 2 + 4, "text-anchor": "end", class: "tick" }, s, c);
    el("text", { transform: `translate(${m.l + i * cell + cell / 2 + 4},${m.t - 6}) rotate(-60)`, class: "tick" }, s, c);
  });
  cm.matrix.forEach((row, i) => row.forEach((v, j) => {
    const x = m.l + j * cell, y = m.t + i * cell;
    el("rect", { x: x + 1, y: y + 1, width: cell - 2, height: cell - 2, rx: 2, fill: v ? c1 : css("--grid"), "fill-opacity": v ? (.12 + .88 * v / max).toFixed(3) : .35 }, s);
    if (v && cell >= 16) el("text", { x: x + cell / 2, y: y + cell / 2 + 4, "text-anchor": "middle",
      style: `fill:${v / max > .55 ? "#ffffff" : css("--ink")};font:${cell >= 22 ? 11 : 9}px var(--mono)` }, s, v);
    const hr = el("rect", { x, y, width: cell, height: cell, fill: "transparent" }, s);
    const total = row.reduce((a, b) => a + b, 0);
    hr.addEventListener("mousemove", ev => showTip(ev, `true <b>${esc(cm.classes[i])}</b><br>predicted <b>${esc(cm.classes[j])}</b><br>${v} of ${total} clips`));
    hr.addEventListener("mouseleave", hideTip);
  }));
}

// interval chart: items [{label, v, lo, hi, color}]
function intervalChart(box, o) {
  box.innerHTML = "";
  const rowH = 30, W = Math.max(280, box.clientWidth), m = { l: 120, r: 60, t: 6, b: 24 };
  const H = m.t + m.b + rowH * o.items.length, s = svg(W, H); box.appendChild(s);
  const lo = Math.min(...o.items.map(i => i.lo)), hi = Math.max(...o.items.map(i => i.hi));
  const [d0, d1] = [Math.max(0, lo - .03), Math.min(1, hi + .03)];
  const X = v => m.l + (v - d0) / (d1 - d0) * (W - m.l - m.r);
  for (const t of niceTicks(d0, d1, 5)) {
    el("line", { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b, class: "gridl" }, s);
    el("text", { x: X(t), y: H - 6, "text-anchor": "middle", class: "tick" }, s, pct(t, 0));
  }
  o.items.forEach((it, i) => {
    const y = m.t + i * rowH + rowH / 2;
    el("text", { x: m.l - 10, y: y + 4, "text-anchor": "end", class: "lbl" }, s, it.label);
    el("line", { x1: X(it.lo), x2: X(it.hi), y1: y, y2: y, stroke: it.color, "stroke-width": 2, "stroke-linecap": "round" }, s);
    el("circle", { cx: X(it.v), cy: y, r: 5, fill: it.color, stroke: css("--surface"), "stroke-width": 2 }, s);
    el("text", { x: X(it.hi) + 8, y: y + 4, class: "lbl" }, s, pct(it.v) + "%");
    const hr = el("rect", { x: 0, y: y - rowH / 2, width: W, height: rowH, fill: "transparent" }, s);
    hr.addEventListener("mousemove", ev => showTip(ev, `<b>${esc(it.label)}</b><br>accuracy ${pct(it.v)}%<br>95% CI ${pct(it.lo)}–${pct(it.hi)}%`));
    hr.addEventListener("mouseleave", hideTip);
  });
}

// ---------- page ----------
const runs = D.runs, byToken = Object.fromEntries(runs.map(r => [r.token, r]));
const done = runs.filter(r => r.row.status === "done");
const sha = [...new Set(done.filter(r => r.settings.dataset === "CricketEC_split").map(r => r.settings.test_sha))];
$("#meta").innerHTML = [`Generated ${esc(D.generated)}`, `${done.length} of ${runs.length} training runs finished`,
  sha.length === 1 ? `Test set fingerprint <span class="mono">${esc(sha[0])}</span> (all runs on the 2,638-clip split)` : ""].filter(Boolean).map(t => `<span>${t}</span>`).join("");
$("#nav").innerHTML = '<a href="#runs">All runs</a><a href="#curves">Training curves</a>' +
  [1, 2, 3, 4, 5, 6, 7].map(n => `<a href="#exp${n}">Exp ${n}</a>`).join("");

const runLabel = r => (r.experiment === 1 ? `${r.group.split(":")[0]} · ` : "") + r.config;
const model = r => `${r.row.backbone === "effnet_b0" ? "EfficientNet-B0" : r.row.backbone === "resnet34" ? "ResNet-34" : r.row.backbone} + ${r.row.rnn}`;
function runsTable() {
  const head = `<tr><th>Exp</th><th>Run</th><th>Model</th><th>Sampling</th><th class="num">Frames</th><th class="num">Best epoch</th>
    <th class="num">Val acc %</th><th class="num">Test acc %</th><th class="num">Macro F1</th><th>Status</th></tr>`;
  const body = runs.map(r => `<tr data-token="${r.token}"><td class="mono">${r.experiment}</td>
    <td><button class="linkish" data-run="${r.token}">${esc(runLabel(r))}</button></td><td>${esc(model(r))}</td><td>${esc(r.row.sampling)}</td>
    <td class="num">${r.row.frames}</td><td class="num">${r.row.best_epoch ? r.row.best_epoch + "/" + r.row.epochs : "–"}</td>
    <td class="num">${pct(r.row.val_acc)}</td><td class="num">${pct(r.headline)}</td><td class="num">${fix(r.row.test_macro_f1)}</td>
    <td>${statusPill(r.row.status)}</td></tr>`).join("");
  $("#runs-table").innerHTML = `<thead>${head}</thead><tbody>${body}</tbody>`;
  $("#runs-table").addEventListener("click", e => { const b = e.target.closest("[data-run]"); if (b) { selectRun(b.dataset.run); $("#curves").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" }); } });
}

function selectRun(token) {
  const r = byToken[token]; if (!r) return;
  $("#run-select").value = token;
  document.querySelectorAll("#runs-table tr[data-token]").forEach(tr => tr.classList.toggle("sel", tr.dataset.token === token));
  $("#run-status").innerHTML = statusPill(r.row.status);
  const st = r.settings, h = r.history, det = $("#run-detail");
  const flags = st.lr ? [`batch ${st.batch_size}`, `hidden ${st.hidden_dim}`, `lr ${st.lr}`, `wd ${st.weight_decay}`, `${st.epochs} epochs`,
    st.augment ? "augmentation" : "no augmentation", st.temporal_jitter ? "temporal jitter" : null,
    st.label_smoothing ? `label smoothing ${st.label_smoothing}` : null, st.schedule === "cosine" ? "cosine LR" : "constant LR",
    st.ema ? `EMA ${st.ema}` : null, st.tta ? "TTA" : null, `data: ${st.dataset}`].filter(Boolean) : [];
  const last = h[h.length - 1];
  const facts = [
    [pct(r.row.val_acc) + (r.row.val_acc != null ? "%" : ""), "best val accuracy"],
    [pct(r.headline) + (r.headline != null ? "%" : ""), r.row.test_acc_tta != null ? "test accuracy (TTA)" : "test accuracy"],
    [fix(r.row.test_macro_f1), "test macro F1"],
    [r.row.best_epoch ? `${r.row.best_epoch} / ${r.row.epochs}` : (last ? `${last.epoch} so far` : "–"), r.row.best_epoch ? "best epoch" : "epochs logged"],
    [r.row.train_minutes != null && r.row.train_minutes !== "" ? r.row.train_minutes + " min" : (h.length ? (h.reduce((a, b) => a + b.seconds, 0) / 60).toFixed(1) + " min" : "–"), "training time"],
  ];
  det.innerHTML = `<div class="panel" style="display:grid;gap:12px">
      <div><h3>${esc(r.group)} · ${esc(r.config)}</h3><p class="source">${esc(model(r))}, ${esc(r.row.sampling)} sampling, ${r.row.frames} frames</p></div>
      <div class="facts">${facts.map(([v, l]) => `<div class="fact"><b>${v}</b><span>${l}</span></div>`).join("")}</div>
      ${flags.length ? `<div class="chips">${flags.map(f => `<span class="chip">${esc(f)}</span>`).join("")}</div>` : ""}
    </div>
    <div class="grid2"><div class="panel"><h3>Loss</h3><div class="chart" id="c-loss"></div></div>
      <div class="panel"><h3>Accuracy</h3><div class="chart" id="c-acc"></div></div></div>
    ${r.confusion ? `<div class="grid2"><div class="panel"><h3>Per-class test F1</h3><div class="chart" id="c-f1"></div></div>
      <div class="panel scroll"><h3>Test confusion matrix</h3><p class="source">Rows: true class · columns: predicted</p><div class="chart" id="c-cm"></div></div></div>` : ""}
    <details class="panel"><summary>Per-epoch table (${h.length} epochs)</summary><div class="scroll"><table><thead><tr><th class="num">Epoch</th>
      <th class="num">Train loss</th><th class="num">Train acc %</th><th class="num">Val loss</th><th class="num">Val acc %</th><th class="num">Val macro F1</th><th class="num">Seconds</th></tr></thead>
      <tbody>${h.map(e => `<tr><td class="num">${e.epoch}</td><td class="num">${fix(e.train_loss)}</td><td class="num">${pct(e.train_acc)}</td><td class="num">${fix(e.val_loss)}</td>
      <td class="num">${pct(e.val_acc)}</td><td class="num">${fix(e.val_macro_f1)}</td><td class="num">${e.seconds}</td></tr>`).join("")}</tbody></table></div></details>
    <p class="source">From <code>results/${esc(r.source)}/</code> (history.csv or train.log, results.json, test_confusion_matrix.csv)</p>`;
  const marker = r.row.best_epoch ? { x: r.row.best_epoch, label: "best" } : null;
  const s1 = css("--s1"), s2 = css("--s2");
  mount($("#c-loss"), b => lineChart(b, { xName: "epoch", xLabel: "epoch", yFmt: v => v.toFixed(2), marker, yMin: 0,
    series: [{ name: "Train loss", color: s1, pts: h.map(e => [e.epoch, e.train_loss]) }, { name: "Val loss", color: s2, pts: h.map(e => [e.epoch, e.val_loss]) }] }));
  mount($("#c-acc"), b => lineChart(b, { xName: "epoch", xLabel: "epoch", yFmt: v => (100 * v).toFixed(1) + "%", marker, yDomain: [0, 1],
    series: [{ name: "Train accuracy", color: s1, pts: h.map(e => [e.epoch, e.train_acc]) }, { name: "Val accuracy", color: s2, pts: h.map(e => [e.epoch, e.val_acc]) }] }));
  if (r.confusion) {
    const STROKE = ["cover", "defense", "flick", "hook", "late_cut", "lofted", "pull", "square_cut", "straight", "sweep"];
    const pc = st.per_class || {};
    const items = Object.entries(pc).map(([c, v]) => { const g = STROKE.includes(c) ? "stroke" : "outcome";
      return { label: c, v: v["f1-score"], group: g, color: g === "stroke" ? s1 : s2,
        tip: `F1 ${v["f1-score"].toFixed(3)}<br>precision ${v.precision.toFixed(3)} · recall ${v.recall.toFixed(3)}<br>${v.support} test clips` }; })
      .sort((a, b) => (a.group === b.group ? b.v - a.v : a.group === "stroke" ? -1 : 1));
    mount($("#c-f1"), b => rowBars(b, { items, legend: [["Batting stroke", s1], ["Match outcome", s2]] }));
    mount($("#c-cm"), b => heatmap(b, r.confusion));
  }
}

function runSelect() {
  const groups = {};
  runs.forEach(r => { (groups[r.group] ||= []).push(r); });
  $("#run-select").innerHTML = Object.entries(groups).map(([g, rs]) => `<optgroup label="${esc(g)}">${rs.map(r =>
    `<option value="${r.token}">${esc(r.config)} · ${esc(model(r))} · ${esc(r.row.sampling)} ${r.row.frames}f${r.row.status === "done" ? "" : " (in progress)"}</option>`).join("")}</optgroup>`).join("");
  $("#run-select").addEventListener("change", e => selectRun(e.target.value));
}

// ---------- experiment sections ----------
function section(n, intro) {
  const s = document.createElement("section"); s.id = `exp${n}`;
  s.innerHTML = `<div class="head"><div class="eyebrow">Experiment ${n}</div><h2>${EXP_TITLES[n]}</h2>${intro ? `<p class="lede">${intro}</p>` : ""}</div>`;
  $("#experiments").appendChild(s); return s;
}
function simpleTable(rows, cols) {
  if (!rows || !rows.length) return '<div class="empty">No rows.</div>';
  cols = cols || Object.keys(rows[0]);
  const isNum = c => rows.every(r => r[c] === "" || r[c] == null || !isNaN(Number(r[c])));
  return `<div class="scroll"><table><thead><tr>${cols.map(c => `<th class="${isNum(c) ? "num" : ""}">${esc(c.replace(/_/g, " "))}</th>`).join("")}</tr></thead>
    <tbody>${rows.map(r => `<tr>${cols.map(c => `<td class="${isNum(c) ? "num" : ""}">${esc(r[c])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}
const runRows = list => list.map(r => ({ run: runLabel(r), model: model(r), sampling: r.row.sampling, frames: r.row.frames,
  best_epoch: r.row.best_epoch ? `${r.row.best_epoch}/${r.row.epochs}` : "", val_acc_pct: pct(r.row.val_acc), test_acc_pct: pct(r.row.test_acc),
  test_acc_tta_pct: pct(r.row.test_acc_tta), macro_precision: fix(r.row.test_macro_precision), macro_recall: fix(r.row.test_macro_recall),
  macro_f1: fix(r.row.test_macro_f1), status: r.row.status.replace("running/stopped at ", "at ") }));

function exp1() {
  const s = section(1, "The paper's five CNN–RNN configurations (Table 7). Each run folder is one training recipe; config B was the reference while tuning the recipe, and the current recipe (run 3) is used for A–E.");
  const bench = runs.filter(r => r.experiment === 1);
  const prog = bench.filter(r => r.config === "B" && r.row.status === "done");
  const p = document.createElement("div"); p.className = "grid2"; s.appendChild(p);
  const a = document.createElement("div"); a.className = "panel"; a.innerHTML = '<h3>Config B across training recipes</h3><p class="source">Test accuracy, ResNet-34 + GRU, 24 frames</p><div class="chart"></div>'; p.appendChild(a);
  mount($(".chart", a), b => columnChart(b, { cats: prog.map(r => r.group.split(":")[0]), yDomain: [0, 1], yFmt: v => (100 * v).toFixed(1) + "%",
    series: [{ name: "Test accuracy", color: css("--s1"), vals: prog.map(r => r.headline) }],
    extra: ci => esc(prog[ci].group.split(": ")[1] || "") }));
  const r3 = bench.filter(r => r.run.startsWith("run3"));
  const c = document.createElement("div"); c.className = "panel"; p.appendChild(c);
  c.innerHTML = `<h3>A–E with the current recipe</h3><p class="source">Run 3 · ${r3.filter(r => r.row.status === "done").length} of 5 finished</p><div class="chart"></div>`;
  const order = ["A", "B", "C", "D", "E"], by = Object.fromEntries(r3.map(r => [r.config, r]));
  mount($(".chart", c), b => columnChart(b, { cats: order, yDomain: [0, 1], yFmt: v => (100 * v).toFixed(1) + "%",
    series: [{ name: "Test accuracy", color: css("--s1"), vals: order.map(k => by[k]?.headline ?? null) }],
    extra: ci => by[order[ci]] ? esc(model(by[order[ci]]) + ", " + by[order[ci]].row.sampling + " " + by[order[ci]].row.frames + "f") : "" }));
  const t = document.createElement("div"); t.className = "panel"; t.innerHTML = simpleTable(runRows(bench)); s.appendChild(t);
  const ex1 = D.analysis.find(x => x.key === "exp1_test_eval");
  if (ex1 && ex1.tables["test_summary.csv"]) {
    const e = document.createElement("div"); e.className = "panel";
    e.innerHTML = `<h3>Experiment 1 test evaluation summary</h3>${ex1.status === "trial" ? `<p class="note">Trial run on ${esc(ex1.models.join(", "))} only; the final run covers all five once they finish.</p>` : ""}${simpleTable(ex1.tables["test_summary.csv"])}<p class="source">From <code>results/${esc(ex1.source)}/test_summary.csv</code></p>`;
    s.appendChild(e);
  }
  s.insertAdjacentHTML("beforeend", '<p class="source">From <code>results/01_exp1_benchmark_A-E/run*/*/results.json</code></p>');
}

function exp2() {
  const s = section(2, "ResNet-34 at 16 frames, GRU and LSTM, each with uniform and pixel-intensity sampling. Every other setting is identical (batch 16, hidden 64, 40 epochs, lr 1e-3, wd 3e-3, seed 42).");
  const rs = runs.filter(r => r.experiment === 2), by = {};
  rs.forEach(r => { by[r.row.rnn + "|" + r.row.sampling] = r; });
  const g = document.createElement("div"); g.className = "grid2"; s.appendChild(g);
  for (const [k, title, fmt, dom] of [["headline", "Test accuracy", v => (100 * v).toFixed(1) + "%", [0, 1]], ["f1", "Test macro F1", v => v.toFixed(3), [0, 1]]]) {
    const p = document.createElement("div"); p.className = "panel"; p.innerHTML = `<h3>${title}</h3><div class="chart"></div>`; g.appendChild(p);
    const val = (rnn, smp) => { const r = by[rnn + "|" + smp]; if (!r || r.row.status !== "done") return null; return k === "f1" ? r.row.test_macro_f1 : r.headline; };
    mount($(".chart", p), b => columnChart(b, { cats: ["GRU", "LSTM"], yDomain: dom, yFmt: fmt,
      series: [{ name: "Uniform", color: css("--s1"), vals: ["GRU", "LSTM"].map(x => val(x, "uniform")) },
               { name: "Pixel-intensity", color: css("--s2"), vals: ["GRU", "LSTM"].map(x => val(x, "pixel")) }] }));
  }
  const t = document.createElement("div"); t.className = "panel"; t.innerHTML = simpleTable(runRows(rs)); s.appendChild(t);
  s.insertAdjacentHTML("beforeend", '<p class="source">From <code>results/02_exp2_sampling_ablation/*/results.json</code></p>');
}

function exp3() {
  const s = section(3, "ResNet-34 + GRU with uniform sampling at 8, 16, 24 and 32 frames; everything else as in Experiment 2. Latency and memory are single-clip inference on the A30 GPU.");
  const rs = runs.filter(r => r.experiment === 3 && r.row.status === "done").sort((a, b) => a.row.frames - b.row.frames);
  const g = document.createElement("div"); g.className = "panel"; s.appendChild(g);
  g.innerHTML = '<h3>Frame count vs test accuracy</h3><div class="chart"></div>';
  const fr = rs.map(r => r.row.frames);
  mount($(".chart", g), b => lineChart(b, { xName: "frames", xLabel: "sampled frames per clip", xTicks: fr, dots: true, endLabels: false,
    yFmt: v => (100 * v).toFixed(1) + "%",
    series: [{ name: "Test accuracy", color: css("--s1"), pts: rs.map(r => [r.row.frames, r.headline]) },
             { name: "Test macro F1", color: css("--s2"), pts: rs.map(r => [r.row.frames, r.row.test_macro_f1]) }] }));
  const costs = rs.filter(r => r.row.latency_ms != null && r.row.latency_ms !== "");
  if (costs.length) {
    const c = document.createElement("div"); c.className = "grid2"; s.appendChild(c);
    for (const [k, title, unit] of [["latency_ms", "Inference latency", " ms"], ["peak_mem_mb", "Peak GPU memory", " MB"]]) {
      const p = document.createElement("div"); p.className = "panel"; p.innerHTML = `<h3>${title}</h3><p class="source">per clip, batch size 1</p><div class="chart"></div>`; c.appendChild(p);
      const mx = Math.max(...costs.map(r => r.row[k]));
      mount($(".chart", p), b => lineChart(b, { xName: "frames", xLabel: "sampled frames per clip", xTicks: costs.map(r => r.row.frames), dots: true, endLabels: true,
        yDomain: [0, mx * 1.2], height: 200, yFmt: v => (k === "latency_ms" ? v.toFixed(1) : v.toFixed(0)) + unit,
        series: [{ name: title, color: css("--s1"), pts: costs.map(r => [r.row.frames, r.row[k]]) }] }));
    }
  }
  const t = document.createElement("div"); t.className = "panel"; t.innerHTML = simpleTable(runRows(runs.filter(r => r.experiment === 3)).map((row, i) => {
    const r = runs.filter(x => x.experiment === 3)[i]; return { ...row, latency_ms: r.row.latency_ms ?? "", peak_mem_mb: r.row.peak_mem_mb ?? "" }; }));
  s.appendChild(t);
  s.insertAdjacentHTML("beforeend", '<p class="source">From <code>results/03_exp3_frame_sensitivity/*/results.json</code></p>');
}

function analysisSection(n, key, intro, extra) {
  const a = D.analysis.find(x => x.key.includes(key + "_")), s = section(n, intro);
  if (!a || a.status === "pending") { s.insertAdjacentHTML("beforeend", '<div class="panel empty">Not run yet. It runs on the server once all five A–E models have finished training.</div>'); return; }
  if (a.status === "trial") s.insertAdjacentHTML("beforeend", `<p class="note">Trial run on model ${esc(a.models.join(", "))} only, to check the pipeline. The final run on all five A–E models starts automatically once they finish training.</p>`);
  if (extra) extra(s, a);
  for (const [name, rows] of Object.entries(a.tables)) {
    const p = document.createElement("div"); p.className = "panel"; p.innerHTML = `<h3 class="mono" style="font-weight:500">${esc(name)}</h3>${simpleTable(rows)}`; s.appendChild(p);
  }
  s.insertAdjacentHTML("beforeend", `<p class="source">From <code>results/${esc(a.source)}/</code></p>`);
}

runsTable(); runSelect();
exp1(); exp2(); exp3();
analysisSection(4, "exp4", "Test accuracy of A–E grouped by camera perspective, on the same 375-clip test set they were trained and scored with. Perspective labels come from the final annotated release (CricketEC_release_final), matched to our clips by file content: all 375 test clips are labelled, and every perspective appears in our train, val and test.");
analysisSection(5, "exp5", "Parameters, GFLOPs, single-clip latency, throughput and peak GPU memory of each model, measured on the idle A30 GPU.");
analysisSection(6, "exp6", "Accuracy on the 10 batting-stroke classes vs the 5 match-outcome classes, and whether errors stay within a group or cross between them.", (s, a) => {
  const rows = a.tables["semantic_layer_results.csv"]; if (!rows) return;
  const p = document.createElement("div"); p.className = "panel"; p.innerHTML = "<h3>Stroke vs outcome accuracy</h3><div class=\"chart\"></div>"; s.appendChild(p);
  mount($(".chart", p), b => columnChart(b, { cats: rows.map(r => r.model), yDomain: [0, 1], yFmt: v => (100 * v).toFixed(1) + "%",
    series: [{ name: "Batting strokes (10)", color: css("--s1"), vals: rows.map(r => +r.stroke_accuracy) },
             { name: "Match outcomes (5)", color: css("--s2"), vals: rows.map(r => +r.outcome_accuracy) }] }));
});
analysisSection(7, "exp7", "95% bootstrap confidence intervals for accuracy, and McNemar's exact test of the best model against each other model (Holm-corrected).", (s, a) => {
  const rows = a.tables["statistical_analysis_results.csv"]; if (!rows) return;
  const p = document.createElement("div"); p.className = "panel"; p.innerHTML = "<h3>Test accuracy with 95% confidence intervals</h3><div class=\"chart\"></div>"; s.appendChild(p);
  mount($(".chart", p), b => intervalChart(b, { items: rows.map(r => ({ label: `${r.model} · ${r.arch}`, v: +r.accuracy, lo: +r.acc_ci95_low, hi: +r.acc_ci95_high, color: css("--s1") })) }));
});

const h = location.hash.slice(1);
const first = byToken[h] ? h : (runs.find(r => r.token === "r3-B") || done[done.length - 1] || runs[0])?.token;
if (first) selectRun(first);
if (byToken[h]) $("#curves").scrollIntoView();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => charts.forEach(c => c.container.isConnected && c.draw(c.container)));
new MutationObserver(() => charts.forEach(c => c.container.isConnected && c.draw(c.container))).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--artifact", type=Path, help="also write the bare page for publishing")
    args = ap.parse_args()
    print(f"Wrote {build(args.out, args.artifact)}")


if __name__ == "__main__":
    main()
