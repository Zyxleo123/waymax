"""Turn per-iteration probe dumps into a slider over optimization steps. CPU only.

`train_safety.py --probe_per_split N` writes, every probe point, one rollout of a
FIXED set of named scenes into ``<out_dir>/probe/``::

    static.npz          roadgraph + human log + goal + the FROZEN-BASE rollout,
                        plus each scene's split label and whether it is in the
                        training pool. Written once; it is ~99% of the bytes.
    iter_%06d.npz       that iteration's paths / offroad marks / outcome flags
    probe_log.jsonl     per-split solved/offroad/overlap/reached, one row per point

This reads those and produces:

    probe_slider.html   2x4 grid -- rows are the arms (e.g. no-anchor vs anchor),
                        columns are failure/success x trained-on/held-out -- with
                        one slider that scrubs every panel through training. Frame
                        0 is the frozen base, i.e. the scene as it was ORIGINALLY.
    probe_curves.png    the same thing as rates: solved / offroad / overlap per
                        split per arm against cumulative Adam steps, with the
                        base as a dashed reference line.

The question it is built to answer is narrow and worth stating: *does applying
the loss fix a trajectory that was off-road or in collision?* A rate can rise
because a different scene started working; the slider cannot, because the scenes
are pinned and the eval seed is pinned, so a panel that changes changed because
the weights did.

No model, no corpus, no GPU -- it only reads the dumps, so the figures re-render
in seconds whenever the question changes.

Usage::

    python -m rl.diffusion_ft.tools.viz_probe_slider \\
        --arm no-anchor=/.../probe_pair/noanchor \\
        --arm anchor=/.../probe_pair/anchor \\
        --out_dir /.../probe_pair/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # headless compute node
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito. The base keeps ONE identity everywhere (it is the same rollout in
# every panel of every arm) and each arm gets its own.
_BASE_COLOR = "#8a929b"
_ARM_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
_SPLIT_ORDER = [
    # 8 transition buckets from --probe_offline: base-outcome x now-outcome x pool
    "recovered_train", "recovered_heldout", "stillfail_train", "stillfail_heldout",
    "regressed_train", "regressed_heldout", "stillok_train", "stillok_heldout",
    # legacy in-run probe (base-outcome x pool only)
    "failure_train", "success_train", "failure_heldout", "success_heldout",
]
_SPLIT_TITLE = {
    "recovered_train":  "FAIL → solved · trained on",
    "recovered_heldout": "FAIL → solved · held out",
    "stillfail_train":  "FAIL → still fails · trained on",
    "stillfail_heldout": "FAIL → still fails · held out",
    "regressed_train":  "solved → FAILS · trained on",
    "regressed_heldout": "solved → FAILS · held out",
    "stillok_train":    "solved → solved · trained on",
    "stillok_heldout":  "solved → solved · held out",
    "failure_train": "originally FAILED · trained on",
    "success_train": "originally SOLVED · trained on",
    "failure_heldout": "originally FAILED · held out",
    "success_heldout": "originally SOLVED · held out",
}
# Roadgraph point types (waymax): 1-3 lane centres, 14-16 road edges.
_LANE_TYPES = (1, 2, 3)
_EDGE_TYPES = (14, 15, 16)


# --------------------------------------------------------------------------- #
def load_arm(out_dir: Path) -> dict:
    """Read one run's probe directory. Frame 0 is always the frozen base."""
    d = out_dir / "probe"
    static_f = d / "static.npz"
    if not static_f.exists():
        raise SystemExit(f"[probe] {static_f} missing -- was the run launched "
                         f"with DFT_PROBE_PER_SPLIT > 0 and DFT_SPLITS_JSON set?")
    st = np.load(static_f, allow_pickle=False)
    frames = []
    for f in sorted(d.glob("iter_*.npz")):
        z = np.load(f, allow_pickle=False)
        frames.append({"iter": int(z["iter"]), "steps": int(z["adam_steps"]),
                       "path": z["path_ft"], "off": z["off_ft"],
                       "done": z["done_ft"],
                       **{k: z[f"{k}_ft"] for k in
                          ("solved", "offroad", "overlap", "reached")}})
    frames.sort(key=lambda r: r["iter"])
    base = {"iter": -1, "steps": 0, "path": st["path_base"], "off": st["off_base"],
            "done": st["done_base"],
            **{k: st[f"{k}_base"] for k in
               ("solved", "offroad", "overlap", "reached")}}
    return {"static": st, "frames": [base] + frames, "dir": out_dir}


def _outcome(fr: dict, i: int) -> str:
    """Compact per-scene outcome: S solved, R reached, O offroad, C collided."""
    if int(fr["solved"][i]):
        return "S"
    tag = "R" if int(fr["reached"][i]) else "-"
    if int(fr["offroad"][i]):
        tag += "O"
    if int(fr["overlap"][i]):
        tag += "C"
    return tag


def _decimate(xy: np.ndarray, cap: int) -> np.ndarray:
    """Keep at most ``cap`` points, evenly. Roadgraph clutter is the page weight."""
    if xy.shape[0] <= cap:
        return xy
    idx = np.linspace(0, xy.shape[0] - 1, cap).astype(int)
    return xy[idx]


# --------------------------------------------------------------------------- #
def build_scene_payload(arms: list[tuple[str, dict]], k: int, args) -> dict:
    """Everything one panel needs, in world metres, cropped to its own view.

    The view is computed over EVERY arm and EVERY frame, so the frame does not
    jump while the slider moves -- a panel that jitters cannot be read as "the
    trajectory changed".
    """
    st = arms[0][1]["static"]
    start = int(st["start_step"])
    goal = np.asarray(st["goal_xy"][k], np.float64)
    log = np.asarray(st["log_path"][k][start:, :2], np.float64)

    pts = [goal[None, :], log]
    for _, arm in arms:
        for fr in arm["frames"]:
            end = int(fr["done"][k])
            seg = np.asarray(fr["path"][k][start:end + 1, :2], np.float64)
            if seg.size:
                pts.append(seg)
    allp = np.concatenate([p for p in pts if p.size], axis=0)
    cx, cy = float(allp[:, 0].mean()), float(allp[:, 1].mean())
    half = float(max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])) * 0.65 + 14.0)

    v = np.asarray(st["rg_valid"][k]).astype(bool)
    rx, ry = np.asarray(st["rg_x"][k]), np.asarray(st["rg_y"][k])
    tp = np.asarray(st["rg_types"][k])
    box = (np.abs(rx - cx) <= half * 1.15) & (np.abs(ry - cy) <= half * 1.15)
    def rg(types):
        m = v & box & np.isin(tp, types)
        return _decimate(np.stack([rx[m], ry[m]], axis=-1).astype(np.float64),
                         args.rg_cap)

    def enc(a):
        return [[round(float(x), 2), round(float(y), 2)] for x, y in a]

    out = {"id": int(st["scene_ids"][k]),
           "label": str(st["labels"][k]),
           "trained": bool(int(st["trained"][k])),
           "view": [round(cx, 2), round(cy, 2), round(half, 2)],
           "edges": enc(rg(_EDGE_TYPES)), "lanes": enc(rg(_LANE_TYPES)),
           "log": enc(_decimate(log, 400)), "goal": enc(goal[None, :])[0],
           "arms": {}}
    for name, arm in arms:
        seq = []
        for fr in arm["frames"]:
            end = int(fr["done"][k])
            seg = np.asarray(fr["path"][k][start:end + 1, :2], np.float64)
            offm = np.asarray(fr["off"][k][start:end + 1]).astype(bool)
            seq.append({"it": int(fr["iter"]), "st": int(fr["steps"]),
                        "xy": enc(seg),
                        "off": [int(i) for i in np.flatnonzero(offm)],
                        "o": _outcome(fr, k)})
        out["arms"][name] = seq
    return out


# --------------------------------------------------------------------------- #
_HTML = r"""<title>__TITLE__</title>
<style>
:root{
  --bg:#fbfbfa; --panel:#ffffff; --ink:#1f2328; --ink2:#5b636b; --line:#e3e7ea;
  --road:#98a3ad; --lane:#dfe3e7; --logc:#8a929b; --bad:#c0392b; --good:#1e8449;
}
@media (prefers-color-scheme:dark){ :root:not([data-theme="light"]){
  --bg:#16181c; --panel:#1d2025; --ink:#e8eaed; --ink2:#9aa2ab; --line:#2c3138;
  --road:#7d8894; --lane:#333941; --logc:#6d757e;
}}
:root[data-theme="dark"]{
  --bg:#16181c; --panel:#1d2025; --ink:#e8eaed; --ink2:#9aa2ab; --line:#2c3138;
  --road:#7d8894; --lane:#333941; --logc:#6d757e;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;margin:0;padding:20px 22px 40px}
h1{font-size:19px;margin:0 0 4px;letter-spacing:-0.01em}
.sub{color:var(--ink2);font-size:13px;margin:0 0 16px;max-width:118ch}
.bar{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 0 12px;margin-bottom:14px}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
input[type=range]{flex:1 1 380px;min-width:240px;accent-color:#0072B2}
button{font:inherit;padding:4px 12px;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;cursor:pointer}
button:hover{border-color:var(--road)}
.step{font-variant-numeric:tabular-nums;font-size:13px;color:var(--ink2);white-space:nowrap}
.step b{color:var(--ink);font-size:15px}
.grid{display:grid;gap:12px;grid-template-columns:repeat(4,minmax(0,1fr))}
@media(max-width:1080px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:600px){.grid{grid-template-columns:1fr}}
.colhead{font-size:12px;font-weight:600;letter-spacing:.02em;color:var(--ink);padding:2px 0}
.colhead .tag{display:block;font-weight:400;color:var(--ink2);font-size:11px}
.cell{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:8px 8px 6px;overflow:hidden}
.cell canvas{width:100%;height:auto;display:block;border-radius:5px}
.cap{display:flex;justify-content:space-between;gap:8px;font-size:11.5px;color:var(--ink2);margin-top:5px;font-variant-numeric:tabular-nums}
.cap .o{font-weight:700}
.o.ok{color:var(--good)} .o.bad{color:var(--bad)}
.armname{font-size:12.5px;font-weight:600;margin:18px 0 7px;padding-bottom:5px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.sw{width:22px;height:3px;border-radius:2px;display:inline-block}
.tally{font-weight:400;font-size:11.5px;color:var(--ink2);font-variant-numeric:tabular-nums;margin-left:auto}
.tally b{font-weight:700}
.tally .rec{color:var(--good)} .tally .reg{color:var(--bad)}
label.tog{display:inline-flex;align-items:center;gap:5px;font-size:12.5px;cursor:pointer;user-select:none}
label.tog input{accent-color:#0072B2}
label.tog .sw{width:18px;height:3px}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin:2px 0 4px}
select{font:inherit;font-size:12px;padding:2px 5px;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:5px}
.note{margin-top:26px;padding:12px 14px;border:1px solid var(--line);border-radius:9px;background:var(--panel);color:var(--ink2);font-size:12.5px;max-width:118ch}
.note b{color:var(--ink)}
</style>

<h1>__TITLE__</h1>
<p class="sub">__SUB__</p>

<div class="bar">
  <div class="row">
    <button id="play">▶︎ play</button>
    <input type="range" id="slider" min="0" max="0" value="0" step="1">
    <span class="step">frame <b id="fi">0</b> · iter <b id="it">base</b> · <b id="st">0</b> Adam steps</span>
  </div>
  <div class="row" style="margin-top:9px">
    <label class="tog"><input type="checkbox" id="showBase" checked>
      <span class="sw" style="background:#8a929b"></span> original trajectory (base)</label>
    <label class="tog"><input type="checkbox" id="showNew" checked>
      <span class="sw" style="background:#0072B2"></span> new trajectory (trained)</label>
  </div>
  <div class="legend" style="margin-top:8px">
    <span><span class="sw" style="background:#8a929b;height:0;border-top:2px dashed #8a929b"></span> human log</span>
    <span>✕ off-road step</span><span>★ goal</span>
    <span>S solved · R reached · O off-road · C collision</span>
    <span>per-arm tally is over all 16 probe scenes at the current frame</span>
  </div>
</div>

<div id="cols" class="grid"></div>
<div id="arms"></div>

<div class="note">
__NOTE__
</div>

<script>
const DATA = __DATA__;
const ARMS = __ARMS__;
const COLS = __COLS__;
const NF   = __NFRAMES__;
const SIZE = 340;

const sel = {};                       // column -> chosen scene index into DATA
COLS.forEach(c => { sel[c.key] = 0; });

function T(view, w){                   // world -> pixel, square canvas
  const [cx,cy,half] = view, s = w/(2*half);
  return (x,y) => [ (x-cx)*s + w/2, w/2 - (y-cy)*s ];
}
function dots(ctx, pts, f, color, r){
  ctx.fillStyle = color;
  for (const p of pts){ const [X,Y] = f(p[0],p[1]);
    ctx.beginPath(); ctx.arc(X,Y,r,0,6.2832); ctx.fill(); }
}
function poly(ctx, pts, f, color, w, dash){
  if (!pts.length) return;
  ctx.save(); ctx.strokeStyle=color; ctx.lineWidth=w; ctx.lineJoin="round";
  ctx.lineCap="round"; ctx.setLineDash(dash||[]);
  ctx.beginPath(); let [X,Y]=f(pts[0][0],pts[0][1]); ctx.moveTo(X,Y);
  for (let i=1;i<pts.length;i++){ [X,Y]=f(pts[i][0],pts[i][1]); ctx.lineTo(X,Y); }
  ctx.stroke(); ctx.restore();
}
function cross(ctx, f, x, y, color){
  const [X,Y]=f(x,y), r=4;
  ctx.save(); ctx.strokeStyle=color; ctx.lineWidth=1.6;
  ctx.beginPath(); ctx.moveTo(X-r,Y-r); ctx.lineTo(X+r,Y+r);
  ctx.moveTo(X+r,Y-r); ctx.lineTo(X-r,Y+r); ctx.stroke(); ctx.restore();
}
function star(ctx, f, x, y){
  const [X,Y]=f(x,y);
  ctx.save(); ctx.fillStyle="#009E73"; ctx.font="17px serif";
  ctx.textAlign="center"; ctx.textBaseline="middle"; ctx.fillText("★",X,Y); ctx.restore();
}
function css(n){ return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }

function drawPanel(cv, sc, armName, color, fi){
  const dpr = window.devicePixelRatio || 1;
  cv.width = SIZE*dpr; cv.height = SIZE*dpr;
  const ctx = cv.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,SIZE,SIZE);
  const f = T(sc.view, SIZE);
  dots(ctx, sc.lanes, f, css("--lane"), 0.7);
  dots(ctx, sc.edges, f, css("--road"), 1.1);
  poly(ctx, sc.log, f, css("--logc"), 1.4, [5,4]);
  const seq = sc.arms[armName], base = seq[0], cur = seq[Math.min(fi, seq.length-1)];
  if (window.SHOW_BASE){                                      // where it started
    poly(ctx, base.xy, f, "#8a929b", 2.0, []);
    for (const i of base.off) if (base.xy[i]) cross(ctx, f, base.xy[i][0], base.xy[i][1], "#8a929b");
  }
  if (window.SHOW_NEW){                                        // where it is now
    poly(ctx, cur.xy,  f, color, 2.6, []);
    for (const i of cur.off)  if (cur.xy[i])  cross(ctx, f, cur.xy[i][0],  cur.xy[i][1],  "#c0392b");
    if (cur.xy.length){ const p=cur.xy[cur.xy.length-1], [X,Y]=f(p[0],p[1]);
      ctx.fillStyle=color; ctx.beginPath(); ctx.arc(X,Y,4.5,0,6.2832); ctx.fill(); }
  }
  star(ctx, f, sc.goal[0], sc.goal[1]);
  return cur;
}

// ---- build the DOM ------------------------------------------------------- //
const colsEl = document.getElementById("cols");
COLS.forEach(c => {
  const d = document.createElement("div"); d.className = "colhead";
  const opts = c.scenes.map((si,i) =>
      `<option value="${i}">scene ${DATA[si].id}</option>`).join("");
  d.innerHTML = `${c.title}<span class="tag">${c.tag}</span>` +
      `<select data-col="${c.key}">${opts}</select>`;
  colsEl.appendChild(d);
});

const armsEl = document.getElementById("arms");
const cells = [];
ARMS.forEach((a, ai) => {
  const h = document.createElement("div"); h.className = "armname";
  h.innerHTML = `<span class="sw" style="background:${a.color}"></span>${a.name}` +
                `<span style="font-weight:400;color:var(--ink2)">${a.desc}</span>` +
                `<span class="tally" data-arm="${a.name}"></span>`;
  armsEl.appendChild(h);
  const g = document.createElement("div"); g.className = "grid";
  COLS.forEach(c => {
    const cell = document.createElement("div"); cell.className = "cell";
    cell.innerHTML = `<canvas></canvas><div class="cap"><span class="was"></span>` +
                     `<span class="o"></span></div>`;
    g.appendChild(cell);
    cells.push({el: cell, cv: cell.querySelector("canvas"), col: c, arm: a});
  });
  armsEl.appendChild(g);
});

function render(fi){
  window.SHOW_BASE = document.getElementById("showBase").checked;
  window.SHOW_NEW  = document.getElementById("showNew").checked;
  for (const c of cells){
    const sc = DATA[c.col.scenes[sel[c.col.key]]];
    const cur = drawPanel(c.cv, sc, c.arm.name, c.arm.color, fi);
    const was = sc.arms[c.arm.name][0].o;
    c.el.querySelector(".was").textContent = `was ${was}`;
    const o = c.el.querySelector(".o");
    o.textContent = `now ${cur.o}`;
    o.className = "o " + (cur.o === "S" ? "ok" : "bad");
  }
  // Per-arm transition tally over ALL probe scenes at this frame. Base outcome
  // is the frame-0 (frozen base) rollout; "now" is the current frame -- so the
  // tally re-derives recovered/regressed live and works for either label scheme.
  ARMS.forEach(a => {
    let rec=0, sf=0, ss=0, reg=0;                 // recovered / still-fail / still-succ / regressed
    DATA.forEach(sc => {
      const seq = sc.arms[a.name];
      const baseS = seq[0].o === "S";
      const nowS = seq[Math.min(fi, seq.length-1)].o === "S";
      if (!baseS) (nowS ? rec++ : sf++);
      else        (nowS ? ss++  : reg++);
    });
    const el = document.querySelector(`.tally[data-arm="${a.name}"]`);
    if (el) el.innerHTML =
      `base failed → <b class="rec">recovered ${rec}</b> · still failing ${sf}` +
      `  |  base solved → still success ${ss} · <b class="reg">regressed ${reg}</b>`;
  });
  const any = DATA[COLS[0].scenes[sel[COLS[0].key]]].arms[ARMS[0].name];
  const fr = any[Math.min(fi, any.length-1)];
  document.getElementById("fi").textContent = fi;
  document.getElementById("it").textContent = fr.it < 0 ? "base" : fr.it;
  document.getElementById("st").textContent = fr.st;
}

const slider = document.getElementById("slider");
slider.max = NF - 1;
slider.addEventListener("input", () => render(+slider.value));
document.querySelectorAll("select[data-col]").forEach(s => {
  s.addEventListener("change", () => { sel[s.dataset.col] = +s.value; render(+slider.value); });
});
document.getElementById("showBase").addEventListener("change", () => render(+slider.value));
document.getElementById("showNew").addEventListener("change", () => render(+slider.value));
let timer = null;
document.getElementById("play").addEventListener("click", (e) => {
  if (timer){ clearInterval(timer); timer = null; e.target.textContent = "▶︎ play"; return; }
  e.target.textContent = "❚❚ pause";
  timer = setInterval(() => {
    let v = (+slider.value + 1) % NF;
    slider.value = v; render(v);
    if (v === NF - 1){ clearInterval(timer); timer = null;
                       document.getElementById("play").textContent = "▶︎ play"; }
  }, 320);
});
window.addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight"){ slider.value = Math.min(NF-1, +slider.value+1); render(+slider.value); }
  if (e.key === "ArrowLeft"){ slider.value = Math.max(0, +slider.value-1); render(+slider.value); }
});
render(0);
</script>
"""


def column_order(scenes: list[dict], idx: list[int], arm0: str) -> list[int]:
    """Order a column's scenes so the DEFAULT one is the interesting one.

    ~15% of the failure pool is ``never_reached`` -- a failure with no off-road
    and no contact in it, which the safety barrier has nothing to say about. A
    panel showing one of those answers a different question than the one asked,
    so the failure columns lead with a scene the base actually drove off the road
    (then one it collided in, then whatever is left). The base outcome is the same
    in every arm, so ``arm0`` is enough to sort by.
    """
    def rank(i):
        o = scenes[i]["arms"][arm0][0]["o"]    # BASE outcome (frame 0)
        if o != "S":                           # base failed: lead with a real off-road
            return (0 if "O" in o else 1 if "C" in o else 2, i)
        return (0, i)                          # base solved: any order
    return sorted(idx, key=rank)


def write_html(scenes: list[dict], arms: list[tuple[str, dict]], out: Path,
               n_frames: int, note: str, title: str, sub: str) -> None:
    cols = []
    for key in _SPLIT_ORDER:
        idx = [i for i, s in enumerate(scenes) if s["label"] == key]
        if not idx:
            continue
        idx = column_order(scenes, idx, arms[0][0])
        trained = scenes[idx[0]]["trained"]
        tag = ("in the training pool" if trained
               else "never in the training pool")
        cols.append({"key": key, "title": _SPLIT_TITLE.get(key, key),
                     "tag": tag, "scenes": idx})
    arm_meta = [{"name": n, "color": _ARM_COLORS[i % len(_ARM_COLORS)],
                 "desc": a.get("desc", "")}
                for i, (n, a) in enumerate(arms)]
    html = (_HTML
            .replace("__DATA__", json.dumps(scenes, separators=(",", ":")))
            .replace("__ARMS__", json.dumps(arm_meta))
            .replace("__COLS__", json.dumps(cols))
            .replace("__NFRAMES__", str(int(n_frames)))
            .replace("__NOTE__", note)
            .replace("__TITLE__", title)
            .replace("__SUB__", sub))
    out.write_text(html)
    print(f"[probe] wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")


# --------------------------------------------------------------------------- #
def fig_curves(arms: list[tuple[str, dict]], out: Path) -> None:
    """The same experiment as rates: per-split solved / offroad / overlap.

    Read against the dashed base line, not against the arm's own first point --
    the first eval already sits several Adam steps downstream of the base, and on
    the 09-02 sweep that gap was most of the movement anyone attributed to
    training.
    """
    rows = {}
    for name, arm in arms:
        f = arm["dir"] / "probe" / "probe_log.jsonl"
        rows[name] = ([json.loads(l) for l in f.open() if l.strip()]
                      if f.exists() else [])
    labels = np.asarray(arms[0][1]["static"]["labels"])
    splits = [s for s in _SPLIT_ORDER if (labels == s).any()]
    metrics = [("solved", "solved  ↑ better"), ("offroad", "off-road  ↓ better"),
               ("overlap", "collision  ↓ better")]

    fig, axes = plt.subplots(len(metrics), len(splits),
                             figsize=(3.5 * len(splits), 2.7 * len(metrics)),
                             squeeze=False, sharex=True)
    for c, sp in enumerate(splits):
        sel = labels == sp
        for r, (mk, mlab) in enumerate(metrics):
            ax = axes[r][c]
            for i, (name, arm) in enumerate(arms):
                rr = rows[name]
                if not rr:
                    continue
                x = [q["adam_steps"] for q in rr]
                y = [q.get(f"{sp}/{mk}", np.nan) for q in rr]
                base_v = float(arm["static"][f"{mk}_base"][sel].mean())
                ax.plot([0] + x, [base_v] + y, "-o", ms=2.6, lw=1.5,
                        color=_ARM_COLORS[i % len(_ARM_COLORS)], label=name)
                ax.axhline(base_v, ls="--", lw=1.0,
                           color=_ARM_COLORS[i % len(_ARM_COLORS)], alpha=0.45)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(color="#eceff1")
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if r == 0:
                ax.set_title(f"{_SPLIT_TITLE.get(sp, sp)}\n({int(sel.sum())} probe scenes)",
                             fontsize=9.5)
            if c == 0:
                ax.set_ylabel(mlab, fontsize=9.5)
            if r == len(metrics) - 1:
                ax.set_xlabel("cumulative Adam steps", fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Probe scenes through training — dashed line is the frozen base "
                 "on the same scenes and the same eval seed", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[probe] wrote {out}")


# --------------------------------------------------------------------------- #
def _draw_axes(ax, sc: dict, arm_name: str, color: str, fi: int) -> str:
    """One slider panel, drawn with matplotlib. Same content as the canvas."""
    e = np.asarray(sc["lanes"], float).reshape(-1, 2)
    if e.size:
        ax.plot(e[:, 0], e[:, 1], ".", ms=0.6, color="#dfe3e7", mec="none", zorder=0)
    e = np.asarray(sc["edges"], float).reshape(-1, 2)
    if e.size:
        ax.plot(e[:, 0], e[:, 1], ".", ms=1.2, color="#98a3ad", mec="none", zorder=1)
    lg = np.asarray(sc["log"], float).reshape(-1, 2)
    if lg.size:
        ax.plot(lg[:, 0], lg[:, 1], "--", lw=1.1, color="#8a929b", zorder=2)
    seq = sc["arms"][arm_name]
    for fr, col, lw, z in ((seq[0], "#8a929b", 1.8, 3),
                           (seq[min(fi, len(seq) - 1)], color, 2.4, 4)):
        xy = np.asarray(fr["xy"], float).reshape(-1, 2)
        if not xy.size:
            continue
        ax.plot(xy[:, 0], xy[:, 1], "-", lw=lw, color=col, zorder=z)
        off = [i for i in fr["off"] if i < xy.shape[0]]
        if off:
            ax.plot(xy[off, 0], xy[off, 1], "x", ms=4, mew=1.2,
                    color=("#8a929b" if z == 3 else "#c0392b"), zorder=z + 2)
        # Where it stopped -- without it a panel does not say which end is which.
        ax.plot(xy[-1, 0], xy[-1, 1], "o", ms=4.5, color=col, mec="white",
                mew=0.9, zorder=z + 1)
    ax.plot(sc["goal"][0], sc["goal"][1], "*", ms=11, color="#009E73",
            mec="white", mew=0.8, zorder=7)
    cx, cy, half = sc["view"]
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#e3e7ea")
    return seq[min(fi, len(seq) - 1)]["o"]


def fig_filmstrip(scenes: list[dict], arms: list[tuple[str, dict]], out: Path,
                  n_frames: int, n_cols: int) -> None:
    """The slider, flattened to a still: one row per (arm, split), frames across.

    Exists because an HTML file is only as good as the browser that opens it, and
    because a figure is what goes in a slide. One scene per split -- the first the
    probe picked -- so the rows are comparable across arms.
    """
    picks = []
    for sp in _SPLIT_ORDER:
        idx = [i for i, s in enumerate(scenes) if s["label"] == sp]
        if idx:
            picks.append((sp, column_order(scenes, idx, arms[0][0])[0]))
    if not picks:
        return
    cols = sorted(set(int(round(x)) for x in
                      np.linspace(0, n_frames - 1, min(n_cols, n_frames))))
    rows = [(an, ac, sp, si) for (an, ac) in
            [(n, _ARM_COLORS[i % len(_ARM_COLORS)]) for i, (n, _) in enumerate(arms)]
            for sp, si in picks]
    fig, axes = plt.subplots(len(rows), len(cols),
                             figsize=(2.05 * len(cols), 2.05 * len(rows)),
                             squeeze=False)
    for r, (an, ac, sp, si) in enumerate(rows):
        for c, fi in enumerate(cols):
            ax = axes[r][c]
            o = _draw_axes(ax, scenes[si], an, ac, fi)
            seq = scenes[si]["arms"][an]
            fr = seq[min(fi, len(seq) - 1)]
            ax.set_title(("base" if fr["it"] < 0 else f"iter {fr['it']}")
                         + f" · {fr['st']} steps · {o}", fontsize=7.5,
                         color=("#1e8449" if o == "S" else "#c0392b"))
            if c == 0:
                ax.set_ylabel(f"{an}\n{_SPLIT_TITLE.get(sp, sp)}\n"
                              f"scene {scenes[si]['id']}", fontsize=7.5)
    fig.suptitle("Grey = frozen base (the trajectory as it was originally); "
                 "colour = current weights; ✕ = off-road step", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[probe] wrote {out}")


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", default=[], metavar="NAME=OUT_DIR",
                   help="one training run to show as a row; repeatable")
    p.add_argument("--out_dir", default="", help="where the figures go "
                                                 "(default: the first arm's dir)")
    p.add_argument("--rg_cap", type=int, default=1400,
                   help="max roadgraph points of each kind per panel -- this is "
                        "what keeps the HTML small")
    p.add_argument("--title", default="Does the safety loss fix the trajectory?")
    p.add_argument("--strip_cols", type=int, default=6,
                   help="frames across the static filmstrip PNG")
    args = p.parse_args()

    if not args.arm:
        raise SystemExit("give at least one --arm NAME=OUT_DIR")
    arms = []
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm wants NAME=OUT_DIR, got {spec!r}")
        name, d = spec.split("=", 1)
        arm = load_arm(Path(d))
        arm["desc"] = f"{len(arm['frames']) - 1} probe points"
        arms.append((name, arm))

    ids0 = np.asarray(arms[0][1]["static"]["scene_ids"])
    for name, arm in arms[1:]:
        ids = np.asarray(arm["static"]["scene_ids"])
        if ids.shape != ids0.shape or (ids != ids0).any():
            raise SystemExit(
                f"[probe] arm '{name}' followed different scenes than "
                f"'{arms[0][0]}' -- the panels would not be comparable. Use the "
                f"same --probe_seed / --probe_splits / --splits_json for both.")
    n_frames = min(len(a["frames"]) for _, a in arms)
    if n_frames < 2:
        print("[probe] only the base frame is present; the slider will not move. "
              "That means no iter_*.npz was written yet -- check the run is past "
              "its first probe point.")

    out = Path(args.out_dir) if args.out_dir else arms[0][1]["dir"] / "figures"
    out.mkdir(parents=True, exist_ok=True)

    scenes = [build_scene_payload(arms, k, args) for k in range(len(ids0))]
    n_trained = sum(1 for s in scenes if s["trained"])
    sub = (f"{len(scenes)} pinned scenes, {n_trained} of them actually in the "
           f"training pool, rolled out at every probe point under one fixed eval "
           f"seed. Grey is the frozen base — the trajectory as it was originally; "
           f"colour is the same scene under the current weights. Nothing but the "
           f"weights changes between frames.")
    note = (
        "<b>How to read it.</b> A panel where a grey ✕-marked excursion is "
        "replaced by a clean coloured line is the loss doing the thing it was "
        "built for. A <i>held-out</i> panel doing that is generalization; a "
        "<i>trained-on</i> panel failing to do it means the objective cannot fix "
        "the violation even with the gradient pointed straight at it, and no "
        "amount of extra data will change that.<br><br>"
        "<b>What it cannot tell you.</b> These are a handful of scenes chosen by "
        "<code>--probe_seed</code>; the rate curves in <code>probe_curves.png</code> "
        "and the <code>&lt;split&gt;/*</code> keys in <code>train_log.jsonl</code> "
        "are the population claim. Read the two together — a slider is evidence "
        "about a mechanism, not about a rate.")
    write_html(scenes, arms, out / "probe_slider.html", n_frames, note,
               args.title, sub)
    fig_curves(arms, out / "probe_curves.png")
    fig_filmstrip(scenes, arms, out / "probe_filmstrip.png", n_frames,
                  args.strip_cols)


if __name__ == "__main__":
    main()
