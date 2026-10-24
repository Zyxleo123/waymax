"""Turn dumped scene geometry into a page where you drag the SDC. CPU only.

Reads ``safety_scenes.json`` from
:mod:`rl.diffusion_ft.tools.dump_safety_scenes` and writes ``safety_drag.html``:
one self-contained page, no model, no corpus, no GPU, that **recomputes the
safety loss in the browser** as you move the ego box.

Why recompute rather than precompute a field: the loss depends on the ego's yaw
as much as its position (the road-edge term is a sum over four corners, and it is
precisely the *rotation* of the box that swings a rear corner over the kerb in a
sharp turn), so a stored grid at one heading would answer a question nobody has.
Dragging and rotating covers the pose space that matters.

The JS is a transcription of :mod:`rl.diffusion_ft.safety_loss`:

* nearest road-edge point by the same stretched-z argmin, the same
  neighbour-compared half-plane sign, the same planar distance from the anchor;
* the same exact box-box gap -- SAT penetration inside, vertex-edge polygon
  distance outside;
* the same ``tau * softplus((margin - c) / tau)`` barrier, summed per corner and
  per agent rather than over an aggregated minimum.

Transcriptions rot, so the page checks itself: the dump ships pose offsets scored
by the Python loss, the page rescores them and reports the largest disagreement in
the header. Anything above a millimetre-scale residual means the two have parted
and the page should not be believed.

The one deliberate departure is a **bucket grid** over the edge points. Python
scans all 30k points per corner, which is fine on an accelerator and hopeless in
a mousemove handler; the grid expands rings until the nearest unvisited ring is
farther than the best hit so far, which returns the same argmin (planar distance
lower-bounds the stretched one, so the ring bound stays valid).

Usage::

    python -m rl.diffusion_ft.tools.viz_safety_drag \\
        --scenes OUT/safety_scenes.json --out_dir OUT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_HTML = r"""<title>__NAME__</title>
<style>
:root{
  --bg:#fbfbfa; --panel:#ffffff; --ink:#1f2328; --ink2:#5b636b; --line:#e3e7ea;
  --road:#98a3ad; --lane:#dfe3e7; --logc:#8a929b; --bad:#c0392b; --good:#1e8449;
  --warn:#b7791f; --ego:#0072B2; --other:#6d757e;
}
@media (prefers-color-scheme:dark){ :root:not([data-theme="light"]){
  --bg:#16181c; --panel:#1d2025; --ink:#e8eaed; --ink2:#9aa2ab; --line:#2c3138;
  --road:#8b96a2; --lane:#333941; --logc:#6d757e; --other:#8b939c;
}}
:root[data-theme="dark"]{
  --bg:#16181c; --panel:#1d2025; --ink:#e8eaed; --ink2:#9aa2ab; --line:#2c3138;
  --road:#8b96a2; --lane:#333941; --logc:#6d757e; --other:#8b939c;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;margin:0;padding:20px 22px 44px}
h1{font-size:19px;margin:0 0 4px;letter-spacing:-0.01em}
.sub{color:var(--ink2);font-size:13px;margin:0 0 14px;max-width:110ch}
.bar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;border-bottom:1px solid var(--line);padding-bottom:12px;margin-bottom:14px}
button,select{font:inherit;font-size:13px;padding:4px 11px;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;cursor:pointer}
button:hover{border-color:var(--road)}
label.chk{font-size:12.5px;color:var(--ink2);display:inline-flex;align-items:center;gap:5px;cursor:pointer}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:16px;align-items:start}
@media(max-width:900px){.wrap{grid-template-columns:1fr}}
.stage{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px;position:relative}
.stage canvas{width:100%;height:auto;display:block;border-radius:6px;touch-action:none;cursor:grab}
.stage canvas.drag{cursor:grabbing}
.hint{position:absolute;left:16px;bottom:16px;font-size:11.5px;color:var(--ink2);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 8px;opacity:.92}
.side{display:flex;flex-direction:column;gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:11px 13px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink2);margin:0 0 8px;font-weight:600}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
td{padding:3px 0;font-size:13px}
td.k{color:var(--ink2)} td.v{text-align:right;font-weight:600}
td.v small{font-weight:400;color:var(--ink2)}
.big{font-size:22px;font-weight:700;font-variant-numeric:tabular-nums}
.ok{color:var(--good)} .bad{color:var(--bad)} .warn{color:var(--warn)}
.flags{display:flex;gap:7px;margin-top:8px}
.flag{flex:1;text-align:center;font-size:11.5px;font-weight:600;border-radius:6px;padding:4px 0;border:1px solid var(--line);color:var(--ink2)}
.flag.on{background:var(--bad);border-color:var(--bad);color:#fff}
.bars{margin-top:8px}
.brow{display:flex;align-items:center;gap:7px;font-size:11.5px;color:var(--ink2);margin:3px 0}
.track{flex:1;height:7px;background:var(--lane);border-radius:4px;overflow:hidden}
.fill{height:100%;border-radius:4px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin-top:10px}
.sw{width:20px;height:3px;border-radius:2px;display:inline-block;vertical-align:middle}
.note{margin-top:22px;padding:12px 14px;border:1px solid var(--line);border-radius:10px;background:var(--panel);color:var(--ink2);font-size:12.5px;max-width:110ch}
.note b{color:var(--ink)} .note code{font-size:12px}
.chk-line{font-size:12px;color:var(--ink2);font-variant-numeric:tabular-nums}
.chk-line b.ok{color:var(--good)} .chk-line b.bad{color:var(--bad)}
</style>

<h1>__TITLE__</h1>
<p class="sub">__SUB__</p>

<div class="bar">
  <select id="scene"></select>
  <button id="reset">reset to log pose</button>
  <label class="chk"><input type="checkbox" id="field"> loss field (this heading)</label>
  <label class="chk"><input type="checkbox" id="anchors" checked> nearest-edge anchors</label>
  <label class="chk"><input type="checkbox" id="grad" checked> descent direction</label>
  <span class="chk-line" id="selfcheck"></span>
</div>

<div class="wrap">
  <div class="stage">
    <canvas id="cv"></canvas>
    <div class="hint">drag the blue box &middot; drag the nose handle to rotate &middot; arrows nudge, <b>[</b>/<b>]</b> rotate</div>
  </div>
  <div class="side">
    <div class="card">
      <h2>Safety loss at this pose</h2>
      <div class="big" id="total">0.000</div>
      <table>
        <tr><td class="k">edge barrier &Sigma;<sub>4 corners</sub></td><td class="v" id="el">0.000</td></tr>
        <tr><td class="k">agent barrier &Sigma;<sub>agents</sub></td><td class="v" id="al">0.000</td></tr>
      </table>
      <div class="flags">
        <div class="flag" id="fo">offroad</div>
        <div class="flag" id="fc">overlap</div>
      </div>
    </div>
    <div class="card">
      <h2>Hard clearances (what the metric thresholds)</h2>
      <table>
        <tr><td class="k">c_edge = min over corners</td><td class="v" id="ce">+0.00 <small>m</small></td></tr>
        <tr><td class="k">c_agent = min over agents</td><td class="v" id="ca">+0.00 <small>m</small></td></tr>
      </table>
      <div class="bars">
        <div class="brow"><span style="width:52px">edge</span><div class="track"><div class="fill" id="beI"></div></div><span id="beT" style="width:74px;text-align:right"></span></div>
        <div class="brow"><span style="width:52px">agent</span><div class="track"><div class="fill" id="baI"></div></div><span id="baT" style="width:74px;text-align:right"></span></div>
      </div>
      <div class="legend" style="margin-top:6px">
        <span>bar full = at the margin &middot; empty = on the plateau</span>
      </div>
    </div>
    <div class="card">
      <h2>Per-corner road-edge clearance</h2>
      <table id="corners"></table>
    </div>
    <div class="card">
      <h2>Constants</h2>
      <table>
        <tr><td class="k">edge margin / tau</td><td class="v" id="cfgE"></td></tr>
        <tr><td class="k">agent margin / tau</td><td class="v" id="cfgA"></td></tr>
        <tr><td class="k">agent geometry</td><td class="v" id="cfgG"></td></tr>
        <tr><td class="k">edge geometry</td><td class="v" id="cfgEG"></td></tr>
      </table>
    </div>
  </div>
</div>

<div class="note">__NOTE__</div>

<script>
const PAY = __DATA__;
const CFG = PAY.config, SCENES = PAY.scenes;
const SIZE = 660;
const SEGMENT = (CFG.edge_geometry || "point") === "segment";
let sc = null, pose = null, grid = null, fieldImg = null;

/* ---- the loss, transcribed from rl/diffusion_ft/safety_loss.py ---------- */
function softplus(z){ return z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)); }
function barrier(c, m, tau){ return tau * softplus((m - c) / tau); }

function corners(b){                       // b = [x, y, L, W, yaw]
  const [x,y,L,W,yaw] = b, c = Math.cos(yaw), s = Math.sin(yaw);
  const ux = c*L/2, uy = s*L/2, vx = -s*W/2, vy = c*W/2;
  return [[x+ux+vx, y+uy+vy], [x+ux-vx, y+uy-vy],
          [x-ux-vx, y-uy-vy], [x-ux+vx, y-uy+vy]];
}

/* Bucket grid over the edge points: the same argmin as the full scan, since a
   ring's minimum PLANAR distance lower-bounds the stretched distance we rank by. */
function buildGrid(E, cell){
  const n = E.xy.length;
  let x0=Infinity, y0=Infinity, x1=-Infinity, y1=-Infinity;
  for (const [x,y] of E.xy){ if(x<x0)x0=x; if(y<y0)y0=y; if(x>x1)x1=x; if(y>y1)y1=y; }
  const nx = Math.max(1, Math.ceil((x1-x0)/cell)+1), ny = Math.max(1, Math.ceil((y1-y0)/cell)+1);
  const buckets = new Array(nx*ny); for (let i=0;i<nx*ny;i++) buckets[i] = [];
  for (let i=0;i<n;i++){
    const cx = Math.min(nx-1, Math.max(0, Math.floor((E.xy[i][0]-x0)/cell)));
    const cy = Math.min(ny-1, Math.max(0, Math.floor((E.xy[i][1]-y0)/cell)));
    buckets[cy*nx+cx].push(i);
  }
  return {x0,y0,nx,ny,cell,buckets,E,n};
}

function nearestEdge(g, qx, qy){
  if (!g || g.n === 0) return -1;
  const cx = Math.floor((qx-g.x0)/g.cell), cy = Math.floor((qy-g.y0)/g.cell);
  let best = -1, bestD2 = Infinity;
  for (let r = 0; ; r++){
    // Everything outside ring r-1 is at least this far away in the plane.
    const ringLB = (r-1) * g.cell;
    if (best >= 0 && ringLB > 0 && ringLB*ringLB > bestD2) break;
    let touched = false;
    for (let j = cy-r; j <= cy+r; j++){
      if (j < 0 || j >= g.ny) continue;
      for (let i = cx-r; i <= cx+r; i++){
        if (i < 0 || i >= g.nx) continue;
        if (r > 0 && Math.abs(i-cx) !== r && Math.abs(j-cy) !== r) continue;  // shell only
        touched = true;
        for (const k of g.buckets[j*g.nx+i]){
          const dx = g.E.xy[k][0]-qx, dy = g.E.xy[k][1]-qy;
          const d2 = dx*dx + dy*dy + g.E.dz2[k];
          if (d2 < bestD2){ bestD2 = d2; best = k; }
        }
      }
    }
    if (!touched && r > g.nx + g.ny) break;
  }
  return best;
}

function segFoot(px, py, ax, ay, bx, by){
  const dx = bx-ax, dy = by-ay, den = Math.max(dx*dx+dy*dy, 1e-12);
  let t = ((px-ax)*dx + (py-ay)*dy) / den;
  t = t < 0 ? 0 : (t > 1 ? 1 : t);
  return [ax + t*dx, ay + t*dy];
}
function edgeClearance(g, px, py){
  const k = nearestEdge(g, px, py);
  if (k < 0) return {c: Infinity, ax: px, ay: py};
  const E = g.E, ax = E.xy[k][0], ay = E.xy[k][1];
  const tx = px - ax, ty = py - ay;
  // The SIGN is the same under both geometries -- that is the whole point of
  // the segment variant, and why the offroad boundary does not move.
  const cross  = tx*E.dir[k][1]  - ty*E.dir[k][0];
  const crossP = tx*E.pdir[k][1] - ty*E.pdir[k][0];
  const use = (E.same[k] && crossP < cross) ? crossP : cross;
  const sign = use > 0 ? 1 : (use < 0 ? -1 : 0);
  let dist = Math.sqrt(tx*tx + ty*ty + 1e-12), fx = ax, fy = ay;
  if (SEGMENT && E.pxy){
    // Nearer of the two incident chords; a segment's closest point is never
    // farther than its endpoint, so the sample distance is also the fallback
    // when neither neighbour is a real same-curve segment.
    for (const [ok, qx, qy, first] of [[E.pok[k], E.pxy[k][0], E.pxy[k][1], true],
                                       [E.nok[k], E.nxy[k][0], E.nxy[k][1], false]]){
      if (!ok) continue;
      const a = first ? [qx, qy] : [ax, ay], b = first ? [ax, ay] : [qx, qy];
      const f = segFoot(px, py, a[0], a[1], b[0], b[1]);
      const d = Math.sqrt((px-f[0])**2 + (py-f[1])**2 + 1e-12);
      if (d < dist){ dist = d; fx = f[0]; fy = f[1]; }
    }
  }
  return {c: -(dist*sign), ax: fx, ay: fy};   // positive == on road
}

/* Exact signed box-box gap: SAT penetration inside, polygon distance outside. */
function satPenetration(ba, bb, ca, cb){
  const axes = [];
  for (const b of [ba, bb]){
    const c = Math.cos(b[4]), s = Math.sin(b[4]);
    axes.push([c,s], [-s,c]);
  }
  let best = Infinity;
  for (const [axx, axy] of axes){
    let amin=Infinity, amax=-Infinity, bmin=Infinity, bmax=-Infinity;
    for (const [x,y] of ca){ const p = x*axx + y*axy; if(p<amin)amin=p; if(p>amax)amax=p; }
    for (const [x,y] of cb){ const p = x*axx + y*axy; if(p<bmin)bmin=p; if(p>bmax)bmax=p; }
    // How far must they move to separate along this axis -- NOT the intersection
    // length, whose gradient vanishes when one projection contains the other.
    const ov = Math.min(amax - bmin, bmax - amin);
    if (ov < best) best = ov;
  }
  return best;
}
function segDist(px, py, ax, ay, bx, by){
  const dx = bx-ax, dy = by-ay, den = Math.max(dx*dx+dy*dy, 1e-12);
  let t = ((px-ax)*dx + (py-ay)*dy) / den;
  t = t < 0 ? 0 : (t > 1 ? 1 : t);
  const ex = px - (ax + t*dx), ey = py - (ay + t*dy);
  return Math.sqrt(ex*ex + ey*ey + 1e-12);
}
function polyDistance(ca, cb){
  let best = Infinity;
  for (const [A, B] of [[ca, cb], [cb, ca]])
    for (const [px, py] of A)
      for (let i=0;i<B.length;i++){
        const a = B[i], b = B[(i+1)%B.length];
        const d = segDist(px, py, a[0], a[1], b[0], b[1]);
        if (d < best) best = d;
      }
  return best;
}
function boxGap(ba, bb){
  const ca = corners(ba), cb = corners(bb);
  const pen = satPenetration(ba, bb, ca, cb);
  return pen > 0 ? -pen : polyDistance(ca, cb);
}

/* The whole thing, at one pose. */
function evaluate(p, opts){
  const box = [p.x, p.y, sc.ego[2], sc.ego[3], p.yaw];
  const cs = corners(box);
  const per = [], anchors = [];
  let el = 0, ce = Infinity;
  for (const [x, y] of cs){
    const r = edgeClearance(grid, x, y);
    per.push(r.c); anchors.push([r.ax, r.ay]);
    el += barrier(r.c, CFG.edge_margin_m, CFG.edge_tau_m);
    if (r.c < ce) ce = r.c;
  }
  let al = 0, ca = Infinity, near = -1;
  if (!opts || !opts.edgeOnly){
    for (let i=0;i<sc.others.length;i++){
      const g = boxGap(box, sc.others[i]);
      al += barrier(g, CFG.agent_margin_m, CFG.agent_tau_m);
      if (g < ca){ ca = g; near = i; }
    }
  }
  return {el, al, total: el + al, ce, ca, per, anchors, near, corners: cs};
}

/* Descent direction by central differences -- the page shows where the analytic
   gradient would push this pose, not a separately-derived formula. */
function descent(p){
  const h = 0.02, hy = 0.01;
  const fx = evaluate({x:p.x+h, y:p.y, yaw:p.yaw}).total - evaluate({x:p.x-h, y:p.y, yaw:p.yaw}).total;
  const fy = evaluate({x:p.x, y:p.y+h, yaw:p.yaw}).total - evaluate({x:p.x, y:p.y-h, yaw:p.yaw}).total;
  const fr = evaluate({x:p.x, y:p.y, yaw:p.yaw+hy}).total - evaluate({x:p.x, y:p.y, yaw:p.yaw-hy}).total;
  return {gx: fx/(2*h), gy: fy/(2*h), gyaw: fr/(2*hy)};
}

/* ---- drawing ------------------------------------------------------------ */
const cv = document.getElementById("cv");
function view(){ return sc.view; }
function T(){ const [cx,cy,half] = view(), s = SIZE/(2*half);
  return (x,y) => [ (x-cx)*s + SIZE/2, SIZE/2 - (y-cy)*s ]; }
function Tinv(){ const [cx,cy,half] = view(), s = SIZE/(2*half);
  return (X,Y) => [ (X - SIZE/2)/s + cx, (SIZE/2 - Y)/s + cy ]; }
function css(n){ return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }

function drawBox(ctx, f, b, stroke, fill, w){
  const cs = corners(b);
  ctx.beginPath();
  cs.forEach(([x,y], i) => { const [X,Y] = f(x,y); i ? ctx.lineTo(X,Y) : ctx.moveTo(X,Y); });
  ctx.closePath();
  if (fill){ ctx.fillStyle = fill; ctx.fill(); }
  ctx.strokeStyle = stroke; ctx.lineWidth = w || 1.6; ctx.stroke();
}

function computeField(ctx){
  // Loss over ego CENTRE positions at the current heading. Coarse on purpose:
  // it is a map of where the barrier lives, and the readout is the exact value.
  const N = 132, [cx,cy,half] = view();
  const img = ctx.createImageData(N, N);
  let max = 1e-6; const vals = new Float64Array(N*N);
  for (let j=0;j<N;j++) for (let i=0;i<N;i++){
    const x = cx - half + (2*half)*(i+0.5)/N, y = cy + half - (2*half)*(j+0.5)/N;
    const v = evaluate({x, y, yaw: pose.yaw}).total;
    vals[j*N+i] = v; if (v > max) max = v;
  }
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  for (let k=0;k<N*N;k++){
    const t = Math.min(1, Math.log1p(vals[k]) / Math.log1p(max));
    img.data[4*k]   = dark ? 40 + 190*t : 255 - 60*t;
    img.data[4*k+1] = dark ? 44 +  40*t : 250 - 190*t;
    img.data[4*k+2] = dark ? 52 +  30*t : 245 - 195*t;
    img.data[4*k+3] = Math.round(210 * Math.min(1, t*1.6));
  }
  const off = document.createElement("canvas"); off.width = N; off.height = N;
  off.getContext("2d").putImageData(img, 0, 0);
  return off;
}

function render(){
  const dpr = window.devicePixelRatio || 1;
  cv.width = SIZE*dpr; cv.height = SIZE*dpr;
  const ctx = cv.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,SIZE,SIZE);
  const f = T();

  if (document.getElementById("field").checked){
    if (!fieldImg) fieldImg = computeField(ctx);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(fieldImg, 0, 0, SIZE, SIZE);
  }
  ctx.fillStyle = css("--lane");
  for (const [x,y] of sc.lanes){ const [X,Y] = f(x,y); ctx.fillRect(X-0.6, Y-0.6, 1.2, 1.2); }
  ctx.fillStyle = css("--road");
  for (const [x,y] of sc.edges.xy){ const [X,Y] = f(x,y); ctx.fillRect(X-1.1, Y-1.1, 2.2, 2.2); }

  ctx.save(); ctx.strokeStyle = css("--logc"); ctx.lineWidth = 1.4; ctx.setLineDash([5,4]);
  ctx.beginPath(); sc.path.forEach(([x,y], i) => { const [X,Y] = f(x,y); i ? ctx.lineTo(X,Y) : ctx.moveTo(X,Y); });
  ctx.stroke(); ctx.restore();

  const ev = evaluate(pose);
  for (let i=0;i<sc.others.length;i++)
    drawBox(ctx, f, sc.others[i], css("--other"),
            i === ev.near ? "rgba(192,57,43,0.16)" : "rgba(128,136,146,0.14)", 1.3);

  if (document.getElementById("anchors").checked){
    ctx.save(); ctx.setLineDash([3,3]); ctx.lineWidth = 1.2;
    ev.anchors.forEach(([ax, ay], i) => {
      const c = ev.per[i];
      ctx.strokeStyle = c < 0 ? css("--bad") : (c < CFG.edge_margin_m ? css("--warn") : css("--ink2"));
      const [X0,Y0] = f(ev.corners[i][0], ev.corners[i][1]), [X1,Y1] = f(ax, ay);
      ctx.beginPath(); ctx.moveTo(X0,Y0); ctx.lineTo(X1,Y1); ctx.stroke();
      ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.arc(X1,Y1,2.4,0,6.2832); ctx.fill();
    });
    ctx.restore();
  }

  const box = [pose.x, pose.y, sc.ego[2], sc.ego[3], pose.yaw];
  drawBox(ctx, f, box, ev.ce < 0 || ev.ca < 0 ? css("--bad") : css("--ego"),
          ev.ce < 0 || ev.ca < 0 ? "rgba(192,57,43,0.20)" : "rgba(0,114,178,0.18)", 2.2);
  ev.corners.forEach(([x,y], i) => {
    const c = ev.per[i], [X,Y] = f(x,y);
    ctx.fillStyle = c < 0 ? css("--bad") : (c < CFG.edge_margin_m ? css("--warn") : css("--good"));
    ctx.beginPath(); ctx.arc(X, Y, 3.4, 0, 6.2832); ctx.fill();
  });
  // Rotation handle, one metre off the nose.
  const [hx, hy] = handlePos();
  const [HX, HY] = f(hx, hy);
  ctx.strokeStyle = css("--ego"); ctx.lineWidth = 1.4;
  const [NX, NY] = f(pose.x + Math.cos(pose.yaw)*sc.ego[2]/2, pose.y + Math.sin(pose.yaw)*sc.ego[2]/2);
  ctx.beginPath(); ctx.moveTo(NX,NY); ctx.lineTo(HX,HY); ctx.stroke();
  ctx.fillStyle = css("--panel"); ctx.beginPath(); ctx.arc(HX,HY,5.5,0,6.2832); ctx.fill(); ctx.stroke();

  if (document.getElementById("grad").checked){
    const g = descent(pose), n = Math.hypot(g.gx, g.gy);
    if (n > 1e-4){
      const L = Math.min(9, 1.2 + 5*Math.log1p(n));           // metres, compressed
      const ux = -g.gx/n, uy = -g.gy/n;
      const [X0,Y0] = f(pose.x, pose.y), [X1,Y1] = f(pose.x + ux*L, pose.y + uy*L);
      ctx.save(); ctx.strokeStyle = css("--good"); ctx.fillStyle = css("--good"); ctx.lineWidth = 2.4;
      ctx.beginPath(); ctx.moveTo(X0,Y0); ctx.lineTo(X1,Y1); ctx.stroke();
      const a = Math.atan2(Y1-Y0, X1-X0);
      ctx.beginPath(); ctx.moveTo(X1,Y1);
      ctx.lineTo(X1 - 9*Math.cos(a-0.4), Y1 - 9*Math.sin(a-0.4));
      ctx.lineTo(X1 - 9*Math.cos(a+0.4), Y1 - 9*Math.sin(a+0.4));
      ctx.closePath(); ctx.fill(); ctx.restore();
    }
  }
  readout(ev);
}

function handlePos(){
  const d = sc.ego[2]/2 + 2.2;
  return [pose.x + Math.cos(pose.yaw)*d, pose.y + Math.sin(pose.yaw)*d];
}

function fmt(v, s){ return (v >= 0 ? "+" : "") + v.toFixed(2) + (s || ""); }
function readout(ev){
  const q = id => document.getElementById(id);
  q("total").textContent = ev.total.toFixed(3);
  q("total").className = "big " + (ev.ce < 0 || ev.ca < 0 ? "bad" : (ev.total > 0.05 ? "warn" : "ok"));
  q("el").textContent = ev.el.toFixed(3);
  q("al").textContent = ev.al.toFixed(3);
  q("ce").innerHTML = fmt(ev.ce) + ' <small>m</small>';
  q("ca").innerHTML = (isFinite(ev.ca) ? fmt(ev.ca) : "&mdash;") + ' <small>m</small>';
  q("ce").className = "v " + (ev.ce < 0 ? "bad" : ev.ce < CFG.edge_margin_m ? "warn" : "ok");
  q("ca").className = "v " + (ev.ca < 0 ? "bad" : ev.ca < CFG.agent_margin_m ? "warn" : "ok");
  q("fo").className = "flag" + (ev.ce < 0 ? " on" : "");
  q("fc").className = "flag" + (ev.ca < 0 ? " on" : "");
  // Bar: how far into the barrier's active band the clearance sits.
  const bandE = CFG.edge_margin_m + 2.7*CFG.edge_tau_m, bandA = CFG.agent_margin_m + 2.7*CFG.agent_tau_m;
  const setBar = (id, tid, c, band, m) => {
    const t = Math.max(0, Math.min(1, (band - c) / band));
    const e = q(id); e.style.width = (100*t).toFixed(1) + "%";
    e.style.background = c < 0 ? css("--bad") : (c < m ? css("--warn") : css("--good"));
    q(tid).textContent = isFinite(c) ? (c < m ? "charged" : "plateau") : "no agent";
  };
  setBar("beI", "beT", ev.ce, bandE, CFG.edge_margin_m);
  setBar("baI", "baT", isFinite(ev.ca) ? ev.ca : 99, bandA, CFG.agent_margin_m);
  q("corners").innerHTML = ["front-left","front-right","rear-right","rear-left"].map((n, i) => {
    const c = ev.per[i], cls = c < 0 ? "bad" : c < CFG.edge_margin_m ? "warn" : "ok";
    return `<tr><td class="k">${n}</td><td class="v ${cls}">${fmt(c)} <small>m</small></td>` +
           `<td class="v" style="width:70px"><small>${barrier(c, CFG.edge_margin_m, CFG.edge_tau_m).toFixed(3)}</small></td></tr>`;
  }).join("");
}

/* ---- self-check against the Python loss --------------------------------- */
function selfCheck(){
  let worst = 0, which = "", checked = 0;
  for (const s of SCENES){
    if (!s._grid) s._grid = buildGrid(s.edges, 4.0);
    grid = s._grid; sc = s;
    for (const pr of s.probes){
      const ev = evaluate({x: s.ego[0]+pr.d[0], y: s.ego[1]+pr.d[1], yaw: s.ego[4]+pr.d[2]});
      const d = [["c_edge", pr.ce, ev.ce], ["c_agent", pr.ca, ev.ca],
                 ["edge_loss", pr.el, ev.el], ["agent_loss", pr.al, ev.al]];
      for (const [n, py, js] of d){
        if (py === null || !isFinite(js)) continue;      // no agent in this scene
        const v = Math.abs(js - py);
        if (v > worst){ worst = v; which = n; }
        checked++;
      }
    }
  }
  const el = document.getElementById("selfcheck");
  const ok = worst < 5e-3;
  el.innerHTML = `self-check vs the Python loss: <b class="${ok ? "ok" : "bad"}">` +
    `max |&Delta;| = ${worst.toExponential(1)}</b> over ${checked} comparisons ` +
    `at ${SCENES.length * SCENES[0].probes.length} random poses` +
    `${which ? " (worst: " + which + ")" : ""}`;
}

/* ---- interaction -------------------------------------------------------- */
function loadScene(i){
  sc = SCENES[i];
  if (!sc._grid) sc._grid = buildGrid(sc.edges, 4.0);
  grid = sc._grid;
  pose = {x: sc.ego[0], y: sc.ego[1], yaw: sc.ego[4]};
  fieldImg = null; render();
}
let mode = null;
function localXY(e){
  const r = cv.getBoundingClientRect();
  return Tinv()((e.clientX - r.left) * SIZE / r.width, (e.clientY - r.top) * SIZE / r.height);
}
cv.addEventListener("pointerdown", e => {
  const [x, y] = localXY(e), [hx, hy] = handlePos();
  const scale = 2*view()[2]/SIZE;                       // metres per pixel
  if (Math.hypot(x-hx, y-hy) < 9*scale) mode = "rot";
  else mode = {dx: pose.x - x, dy: pose.y - y};
  cv.setPointerCapture(e.pointerId); cv.classList.add("drag");
});
cv.addEventListener("pointermove", e => {
  if (!mode) return;
  const [x, y] = localXY(e);
  if (mode === "rot") pose.yaw = Math.atan2(y - pose.y, x - pose.x);
  else { pose.x = x + mode.dx; pose.y = y + mode.dy; }
  render();
});
const stop = () => { mode = null; cv.classList.remove("drag"); };
cv.addEventListener("pointerup", stop); cv.addEventListener("pointercancel", stop);
window.addEventListener("keydown", e => {
  const s = e.shiftKey ? 0.05 : 0.25, r = e.shiftKey ? 0.005 : 0.03;
  const k = {ArrowLeft:[-s,0], ArrowRight:[s,0], ArrowUp:[0,s], ArrowDown:[0,-s]}[e.key];
  if (k){ pose.x += k[0]; pose.y += k[1]; render(); e.preventDefault(); }
  if (e.key === "["){ pose.yaw += r; render(); }
  if (e.key === "]"){ pose.yaw -= r; render(); }
});
document.getElementById("reset").addEventListener("click", () => {
  pose = {x: sc.ego[0], y: sc.ego[1], yaw: sc.ego[4]}; render();
});
document.getElementById("field").addEventListener("change", () => { fieldImg = null; render(); });
document.getElementById("anchors").addEventListener("change", render);
document.getElementById("grad").addEventListener("change", render);
// The field is drawn at one heading; rotating invalidates it.
const _r0 = render;
let lastYaw = null;
render = function(){ if (pose && lastYaw !== null && Math.abs(pose.yaw - lastYaw) > 1e-9) fieldImg = null;
                     if (pose) lastYaw = pose.yaw; _r0(); };

const selEl = document.getElementById("scene");
SCENES.forEach((s, i) => {
  const o = document.createElement("option"); o.value = i;
  o.textContent = `scene ${s.id} · t=${s.t} · log c_edge ${s.log.ce >= 0 ? "+" : ""}${s.log.ce.toFixed(2)} m`;
  selEl.appendChild(o);
});
selEl.addEventListener("change", () => loadScene(+selEl.value));
document.getElementById("cfgE").textContent = `${CFG.edge_margin_m} / ${CFG.edge_tau_m} m`;
document.getElementById("cfgA").textContent = `${CFG.agent_margin_m} / ${CFG.agent_tau_m} m`;
document.getElementById("cfgG").textContent = CFG.agent_geometry;
document.getElementById("cfgEG").textContent = CFG.edge_geometry || "point";
selfCheck();
loadScene(0);
</script>
"""

_NOTE = """
<b>What you are moving.</b> The blue box is the ego at one logged timestep; every
other agent and the roadgraph are frozen exactly as the simulator has them at that
step. The loss is recomputed in the browser at each pose, from the same geometry
the training objective differentiates: four corner clearances to the nearest road
edge and one exact box-box gap per agent, each pushed through
<code>tau*softplus((margin-c)/tau)</code> and <b>summed</b>, not minimised.
<br><br>
<b>Things worth trying.</b> Nudge a rear corner over the kerb with <b>[</b>/<b>]</b>
while the centre stays on the road -- the total rises before the offroad flag ever
flips, which is the whole point of the barrier. Park alongside a neighbour and
close the gap: nothing happens until 0.75 m, then the agent term turns on smoothly
and keeps rising after contact, because penetration depth is signed and unbounded.
Turn on the loss field to see the plateau: it is flat over most of the drivable
surface, so on ordinary driving this term contributes no gradient at all.
<br><br>
<b>The green arrow is the descent direction</b> of the total at this pose (central
differences over the same code path). Off the plateau it points across the nearest
edge or away from the nearest car; on the plateau it is absent because there is
nothing to descend. <b>Nothing here rewards progress</b> -- driving into an empty
field scores zero, which is exactly the degeneracy the training objective's anchor
term is meant to guard against.
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", required=True, help="safety_scenes.json from the dumper")
    p.add_argument("--out_dir", default="")
    p.add_argument("--title", default="What the safety loss charges, pose by pose")
    p.add_argument("--name", default="Safety Loss Sandbox",
                   help="short name for the browser tab; the title is the heading.")
    args = p.parse_args()

    src = Path(args.scenes)
    payload = json.loads(src.read_text())
    out_dir = Path(args.out_dir) if args.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = payload["config"]
    sub = (f"Drag the SDC. The road-edge and agent barriers are recomputed live from "
           f"the same geometry <code>rl/diffusion_ft/safety_loss.py</code> differentiates "
           f"— edge margin {cfg['edge_margin_m']} m (tau {cfg['edge_tau_m']}), agent margin "
           f"{cfg['agent_margin_m']} m (tau {cfg['agent_tau_m']}), exact box-box gap. "
           f"{len(payload['scenes'])} scenes, each opened at its tightest logged step.")

    html = (_HTML
            .replace("__DATA__", json.dumps(payload, separators=(",", ":")))
            .replace("__NAME__", args.name)
            .replace("__TITLE__", args.title)
            .replace("__SUB__", sub)
            .replace("__NOTE__", _NOTE))
    out = out_dir / "safety_drag.html"
    out.write_text(html)
    print(f"[drag] wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
