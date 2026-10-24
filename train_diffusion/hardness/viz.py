"""Per-scenario hardness diagnostics, drawn entirely from saved artefacts.

Answers, for one scenario, the four questions in one glance:

  * what did the logged expert do?     -- its path on the road graph
  * what do the features say?           -- the risk model's per-feature
                                          contribution, so the *alarming* ones
                                          (the features actually driving predicted
                                          risk up) are the long red bars
  * how hard did we predict it to be?   -- scenario risk + decile, anchor-local
                                          risk strip
  * did the policy actually fail?       -- the closed-loop label and its mode

Every input is a file on disk:

  risk.npz     scenario_risk, anchor_risk, coef, scaler_* -- from risk_model.py
  labels.npz   pol_success / pol_offroad / ... / ego_idx  -- from labels.py
  desc/*.npz   scenario_feat, anchor_feat                 -- from descriptors.py
  cache/*.npz  log_trajectory_*, roadgraph_*              -- the sim-state cache

so the figures re-render on CPU without touching the model or the simulator.
Geometry is reused from descriptors.py (same _prepare_scenario / _corners) so a
drawn box is the exact box the road-clearance feature was measured on.

Two population figures (overview.png) plus a gallery of per-scenario panels,
selected across the confusion matrix: top-decile-risk x {failed, passed} and
bottom-decile-risk x {failed, passed}, so a viewer sees hits, misses, and false
alarms rather than a curated best case.  Writes gallery_index.json listing what
was drawn, so a contact sheet or an HTML page can be rebuilt without recomputing.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon

from train_diffusion.hardness.descriptors import (
    CACHE_KEYS, FEATURE_NAMES, LANE_TYPES, ROAD_EDGE_TYPES,
    _corners, _prepare_scenario,
)

FAIL_COLORS = {
    "offroad": "#c1440e",
    "overlap": "#d98324",
    "no_goal": "#5b3b8c",
    "reached_goal_but_violated": "#b0306a",
}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def failure_mode(row: dict) -> str | None:
    """One label from the policy outcome, matching labels.py's failure_modes."""
    if row["pol_success"]:
        return None
    if not row["pol_goal_reached"]:
        return "no_goal"
    if row["pol_offroad"]:
        return "offroad"
    if row["pol_overlap"]:
        return "overlap"
    return "reached_goal_but_violated"


class Artefacts:
    """Lazily-joined view over risk / labels / descriptors / cache."""

    def __init__(self, risk_path, labels_path, desc_dir, cache_dir):
        self.desc_dir = desc_dir
        self.cache_dir = cache_dir

        r = np.load(risk_path, allow_pickle=True)
        self.risk_shard = r["shard"].astype(str)
        self.risk_scen = r["scenario_idx"].astype(np.int64)
        self.scenario_risk = r["scenario_risk"].astype(np.float64)
        self.anchor_risk = r["anchor_risk"].astype(np.float64)
        self.anchors = r["anchors"].astype(int).tolist()
        self.dirty = r["dirty"].astype(bool)
        self.decile_edges = r["decile_edges"].astype(np.float64)
        self.coef = r["coef"].astype(np.float64)
        self.scaler_median = r["scaler_median"].astype(np.float64)
        self.scaler_mean = r["scaler_mean"].astype(np.float64)
        self.scaler_std = r["scaler_std"].astype(np.float64)
        self._risk_row = {(s, int(i)): k for k, (s, i)
                          in enumerate(zip(self.risk_shard, self.risk_scen))}

        self.labels = dict(np.load(labels_path, allow_pickle=True))
        self.lab_shard = self.labels["shard"].astype(str)
        self.lab_scen = self.labels["scenario_idx"].astype(np.int64)
        self._lab_row = {(s, int(i)): k for k, (s, i)
                         in enumerate(zip(self.lab_shard, self.lab_scen))}

        self._desc_cache: dict[str, dict] = {}
        self._geom_cache: dict[str, dict] = {}

    # -- joins ---------------------------------------------------------------
    def risk_of(self, shard, scen):
        row = self._risk_row.get((shard, int(scen)))
        if row is None:
            return None
        return {
            "scenario_risk": float(self.scenario_risk[row]),
            "anchor_risk": self.anchor_risk[row],
            "dirty": bool(self.dirty[row]),
            "decile": int(np.searchsorted(self.decile_edges[1:-1], self.scenario_risk[row])),
        }

    def label_of(self, shard, scen):
        row = self._lab_row.get((shard, int(scen)))
        if row is None:
            return None
        out = {k: self.labels[k][row] for k in self.labels}
        out["pol_success"] = bool(out["pol_success"])
        for key in ("pol_offroad", "pol_overlap", "pol_goal_reached", "exp_labeled", "exp_success"):
            out[key] = bool(out[key])
        return out

    def features_of(self, shard, scen):
        desc = self._desc(shard)
        if desc is None:
            return None
        row = desc["row_of"].get(int(scen))
        if row is None:
            return None
        return desc["scenario_feat"][row], desc["anchor_feat"][row]

    def _desc(self, shard):
        if shard not in self._desc_cache:
            path = Path(self.desc_dir) / f"{shard}.desc.npz"
            if not path.exists():
                self._desc_cache[shard] = None
            else:
                with np.load(path, allow_pickle=True) as h:
                    idx = h["scenario_indices"].astype(np.int64)
                    self._desc_cache[shard] = {
                        "scenario_feat": h["scenario_feat"].astype(np.float64),
                        "anchor_feat": h["anchor_feat"].astype(np.float64),
                        "row_of": {int(s): k for k, s in enumerate(idx)},
                    }
        return self._desc_cache[shard]

    def geometry_of(self, shard, scen, ego_idx):
        """Expert path + road graph for one scenario, via descriptors' own prep."""
        path = Path(self.cache_dir) / f"{shard}.sim_state_cache.npz"
        if not path.exists():
            return None
        if shard not in self._geom_cache:
            with np.load(path, allow_pickle=True) as h:
                payload = {k: h[k] for k in CACHE_KEYS}
            scen_idx = payload["scenario_indices"].astype(np.int64)
            self._geom_cache[shard] = {
                "payload": payload,
                "row_of": {int(s): k for k, s in enumerate(scen_idx)},
            }
        block = self._geom_cache[shard]
        row = block["row_of"].get(int(scen))
        if row is None:
            return None
        return _prepare_scenario(block["payload"], row, int(ego_idx) if ego_idx is not None else None)

    def contributions(self, scen_feat):
        """Per-feature push on the failure logit: coef * z(feature)."""
        x = np.where(np.isfinite(scen_feat), scen_feat, self.scaler_median)
        z = (x - self.scaler_mean) / self.scaler_std
        contrib = self.coef[:-1] * z
        logit = float(contrib.sum() + self.coef[-1])
        return contrib, z, _sigmoid(logit)


# --------------------------------------------------------------------------
# Population overview
# --------------------------------------------------------------------------
def draw_overview(art: Artefacts, out_path: str):
    """Risk vs actual failure across the labelled set: does risk mean anything?"""
    shard, scen = art.lab_shard, art.lab_scen
    risk = np.full(shard.size, np.nan)
    for k in range(shard.size):
        r = art.risk_of(shard[k], scen[k])
        if r is not None:
            risk[k] = r["scenario_risk"]
    fail = ~art.labels["pol_success"].astype(bool)
    clean = art.labels["exp_labeled"].astype(bool) & art.labels["exp_success"].astype(bool)
    ok = np.isfinite(risk) & clean          # the honest slice, matching the fit
    risk, fail = risk[ok], fail[ok]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # (1) reliability: bin by predicted risk, plot empirical failure rate
    ax = axes[0]
    edges = np.quantile(risk, np.linspace(0, 1, 11))
    edges = np.unique(edges)
    mids, rates, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (risk >= lo) & (risk <= hi if hi == edges[-1] else risk < hi)
        if sel.sum() >= 20:
            mids.append(risk[sel].mean())
            rates.append(fail[sel].mean())
            ns.append(int(sel.sum()))
    lim = max(max(mids), max(rates)) * 1.1
    ax.plot([0, lim], [0, lim], "--", color="#888", lw=1, label="perfect calibration")
    ax.plot(mids, rates, "o-", color="#b0306a", lw=2)
    ax.set_xlabel("predicted risk (decile mean)")
    ax.set_ylabel("empirical failure rate")
    ax.set_title("Reliability (expert-clean)")
    ax.legend(fontsize=8, loc="upper left")

    # (2) risk distribution split by outcome
    ax = axes[1]
    bins = np.linspace(0, np.quantile(risk, 0.995), 40)
    ax.hist(risk[~fail], bins=bins, density=True, alpha=0.6, color="#2a7fb0", label="policy passed")
    ax.hist(risk[fail], bins=bins, density=True, alpha=0.6, color="#c1440e", label="policy failed")
    ax.set_xlabel("predicted risk")
    ax.set_ylabel("density")
    ax.set_title("Risk separates pass from fail")
    ax.legend(fontsize=8)

    # (3) decile lift
    ax = axes[2]
    order = np.argsort(risk)
    deciles = np.array_split(order, 10)
    lift = [fail[d].mean() / max(fail.mean(), 1e-9) for d in deciles]
    ax.bar(range(1, 11), lift, color="#d98324")
    ax.axhline(1.0, color="#555", lw=1, ls="--")
    ax.set_xlabel("predicted-risk decile (10 = riskiest)")
    ax.set_ylabel("failure rate / base rate")
    ax.set_title(f"Top-decile lift = {lift[-1]:.1f}x  (base {fail.mean():.2%})")

    fig.suptitle(f"Hardness overview  |  n={risk.size} expert-clean labelled scenarios",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Per-scenario panel
# --------------------------------------------------------------------------
def draw_scenario(art: Artefacts, shard, scen, out_path, top_k=10):
    lab = art.label_of(shard, scen)
    rk = art.risk_of(shard, scen)
    feats = art.features_of(shard, scen)
    if lab is None or rk is None or feats is None:
        return False
    scen_feat, anchor_feat = feats
    geom = art.geometry_of(shard, scen, lab.get("ego_idx", -1))
    if geom is None:
        return False
    contrib, z, recomputed = art.contributions(scen_feat)

    fig = plt.figure(figsize=(15, 7.5))
    gs = GridSpec(2, 2, width_ratios=[1.25, 1], height_ratios=[5, 1], figure=fig,
                  hspace=0.28, wspace=0.45)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[:, 1])
    ax_anc = fig.add_subplot(gs[1, 0])

    _draw_map(ax_map, geom, scen_feat)
    _draw_bars(ax_bar, contrib, scen_feat, z, top_k)
    _draw_anchor_strip(ax_anc, art.anchors, rk["anchor_risk"])

    mode = failure_mode(lab)
    outcome = "PASS" if lab["pol_success"] else f"FAIL: {mode}"
    oc_color = "#1a7f37" if lab["pol_success"] else FAIL_COLORS.get(mode, "#c1440e")
    dirty = " | DIRTY (expert offroad)" if rk["dirty"] else ""
    exp = ("expert clean" if (lab["exp_labeled"] and lab["exp_success"])
           else "expert DIRTY" if lab["exp_labeled"] else "expert unlabelled")
    fig.suptitle(
        f"{shard}  scenario {int(scen)}    "
        f"predicted risk {rk['scenario_risk']:.3f} (decile {rk['decile']+1}/10){dirty}",
        fontweight="bold", fontsize=12)
    ax_map.set_title(f"[{outcome}]   {exp}   recomputed risk {recomputed:.3f}",
                     color=oc_color, fontweight="bold", fontsize=11)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True


def _draw_map(ax, geom, scen_feat):
    edges = geom["edges"]
    if edges.idx.size:
        ep = edges.xy[edges.idx]
        ax.scatter(ep[:, 0], ep[:, 1], s=2, c="#444", marker=".", label="road edge")
    lanes = geom["lanes"]
    if lanes["n"]:
        ax.scatter(lanes["x"], lanes["y"], s=1, c="#bbb", marker=".", zorder=0)

    ego = geom["ego"]
    steps = np.flatnonzero(ego["valid"])
    x, y, yaw = ego["x"][steps], ego["y"][steps], ego["yaw"][steps]
    ax.plot(x, y, "-", color="#1a5fb4", lw=1.5, alpha=0.5, zorder=2)
    sc = ax.scatter(x, y, c=steps, cmap="viridis", s=10, zorder=3)
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label="timestep")

    # boxes at start / mid / end, plus every frame the box is within 0.5 m of an edge
    corners = _corners(x, y, yaw, ego["length"][steps], ego["width"][steps])
    signed = edges.signed_distance(corners.reshape(-1, 2)).reshape(-1, 4)
    clearance = -np.nanmax(signed, axis=-1)
    show = set(np.linspace(0, len(steps) - 1, 3).astype(int).tolist())
    show |= set(np.flatnonzero(np.isfinite(clearance) & (clearance < 0.5)).tolist())
    for i in sorted(show):
        near = np.isfinite(clearance[i]) and clearance[i] < 0.5
        ax.add_patch(Polygon(corners[i], closed=True, fill=False,
                             edgecolor="#c1440e" if near else "#1a5fb4",
                             lw=1.6 if near else 1.0, zorder=4))
    ax.plot(x[0], y[0], "o", color="#1a7f37", ms=9, label="start", zorder=5)
    ax.plot(x[-1], y[-1], "*", color="#b0306a", ms=15, label="goal (last logged)", zorder=5)

    ax.set_aspect("equal")
    pad = 8.0
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)
    ax.legend(fontsize=7, loc="best")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def _draw_bars(ax, contrib, scen_feat, z, top_k):
    order = np.argsort(-np.abs(contrib))[:top_k]
    order = order[np.argsort(contrib[order])]      # ascending for a tidy barh
    vals = contrib[order]
    colors = ["#c1440e" if v > 0 else "#2a7fb0" for v in vals]
    ax.barh(range(len(order)), vals, color=colors)
    labels = []
    for i in order:
        raw = scen_feat[i]
        raw_s = "n/a" if not np.isfinite(raw) else f"{raw:.2f}"
        arrow = "high" if z[i] > 0 else "low"
        labels.append(f"{FEATURE_NAMES[i]}  ({raw_s}, {arrow})")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.yaxis.tick_right()                 # clear of the map's colorbar
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("contribution to failure logit  (coef x z-score)")
    ax.set_title("Alarming features (red = pushes toward failure)", fontsize=10)


def _draw_anchor_strip(ax, anchors, anchor_risk):
    ax.bar(range(len(anchors)), anchor_risk, color="#d98324")
    ax.set_xticks(range(len(anchors)))
    ax.set_xticklabels([f"t={a}" for a in anchors], fontsize=8)
    ax.set_ylabel("anchor risk", fontsize=8)
    ax.set_title("Per-window (5 s) predicted risk -- where the danger sits", fontsize=9)


# --------------------------------------------------------------------------
# Gallery selection
# --------------------------------------------------------------------------
def select_gallery(art: Artefacts, n_per_cell: int, seed: int):
    """Sample n_per_cell scenarios for every (decile 1-10) x {failed, passed} cell,
    so every hardness range is populated for both outcomes.  Deciles come from the
    same decile_edges the per-scenario panels display, so a panel's badge and the
    dropdown bucket it falls under always agree."""
    shard, scen = art.lab_shard, art.lab_scen
    risk = np.full(shard.size, np.nan)
    decile = np.full(shard.size, -1, dtype=np.int64)
    for k in range(shard.size):
        r = art.risk_of(shard[k], scen[k])
        if r is not None:
            risk[k] = r["scenario_risk"]
            decile[k] = r["decile"]                 # 0..9
    fail = ~art.labels["pol_success"].astype(bool)
    clean = art.labels["exp_labeled"].astype(bool) & art.labels["exp_success"].astype(bool)
    ok = np.isfinite(risk) & clean & (decile >= 0)

    rng = np.random.default_rng(seed)
    picks = []
    for d in range(10):                             # decile 0..9  ->  bucket 1..10
        for failed, tag in ((True, "FAIL"), (False, "pass")):
            pool = np.flatnonzero(ok & (decile == d) & (fail == failed))
            if pool.size == 0:
                continue
            chosen = pool if pool.size <= n_per_cell else rng.choice(pool, n_per_cell, replace=False)
            for i in sorted(chosen.tolist()):
                picks.append({"cell": f"d{d + 1}_{tag}", "decile": d + 1,
                              "shard": str(shard[i]), "scenario_idx": int(scen[i]),
                              "risk": float(risk[i]), "failed": bool(failed)})
    return picks


def write_index_html(out_dir: Path, drawn: list[dict], decile_edges):
    """Interactive contact sheet: two columns (failed | passed) and a dropdown to
    pick the hardness decile.  All filtering is client-side over the embedded
    gallery data, so the page is a single static file that opens from disk."""
    edges = np.asarray(decile_edges, dtype=float)
    ranges = [f"decile {d + 1}  ({edges[d]:.3f}–{edges[d + 1]:.3f})"
              for d in range(len(edges) - 1)]
    data_json = json.dumps(drawn)
    ranges_json = json.dumps(ranges)
    html = """<!doctype html>
<html><head><meta charset="utf-8"><title>Hardness gallery</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:1.2rem;color:#1b1b1b}
 h1{font-size:1.25rem;margin:0 0 .3rem}
 .bar{display:flex;gap:1rem;align-items:center;margin:.6rem 0 1rem;flex-wrap:wrap}
 select{font-size:1rem;padding:.25rem .4rem}
 .count{color:#555;font-size:.9rem}
 .cols{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem}
 .col h2{font-size:1rem;margin:.2rem 0 .6rem;padding:.3rem .5rem;border-radius:4px}
 .col.fail h2{background:#f6dccd;color:#7a2a08}
 .col.pass h2{background:#d5ecda;color:#134b23}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:.7rem}
 figure{margin:0;border:1px solid #ddd;border-radius:6px;overflow:hidden;background:#fafafa}
 figure img{width:100%;display:block;cursor:zoom-in}
 figcaption{font-size:.78rem;padding:.35rem .5rem;color:#333}
 .empty{color:#999;font-style:italic;padding:1rem}
</style></head><body>
<h1>Hardness gallery &mdash; failed vs passed</h1>
<div class="bar">
  <label>Hardness range: <select id="sel"></select></label>
  <span class="count" id="count"></span>
</div>
<div class="cols">
  <div class="col fail"><h2>Policy FAILED</h2><div class="grid" id="gfail"></div></div>
  <div class="col pass"><h2>Policy passed</h2><div class="grid" id="gpass"></div></div>
</div>
<script>
const DATA = __DATA__;
const RANGES = __RANGES__;
const sel = document.getElementById('sel');
sel.add(new Option('All deciles', 'all'));
RANGES.forEach((r, i) => sel.add(new Option(r, String(i + 1))));
function card(d){
  const f = document.createElement('figure');
  const a = document.createElement('a'); a.href = d.file; a.target = '_blank';
  const img = document.createElement('img'); img.loading = 'lazy'; img.src = d.file;
  a.appendChild(img); f.appendChild(a);
  const c = document.createElement('figcaption');
  c.textContent = `${d.shard} · s${d.scenario_idx} · risk ${d.risk.toFixed(3)} · decile ${d.decile}`;
  f.appendChild(c); return f;
}
function render(){
  const v = sel.value;
  const rows = DATA.filter(d => v === 'all' || d.decile === Number(v));
  const gf = document.getElementById('gfail'), gp = document.getElementById('gpass');
  gf.innerHTML = ''; gp.innerHTML = '';
  let nf = 0, np = 0;
  rows.forEach(d => { if (d.failed){ gf.appendChild(card(d)); nf++; }
                      else { gp.appendChild(card(d)); np++; } });
  if (!nf) gf.innerHTML = '<div class="empty">none in this range</div>';
  if (!np) gp.innerHTML = '<div class="empty">none in this range</div>';
  document.getElementById('count').textContent =
    `${nf} failed · ${np} passed  (${rows.length} panels)`;
}
sel.addEventListener('change', render);
render();
</script></body></html>
"""
    html = html.replace("__DATA__", data_json).replace("__RANGES__", ranges_json)
    (out_dir / "index.html").write_text(html)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--risk", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--desc_dir", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_per_cell", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    art = Artefacts(args.risk, args.labels, args.desc_dir, args.cache_dir)
    out = Path(args.out_dir)
    (out / "scenarios").mkdir(parents=True, exist_ok=True)

    overview_path = out / "overview.png"
    if args.overwrite or not overview_path.exists():
        draw_overview(art, str(overview_path))
        print(f"wrote {overview_path}", flush=True)

    picks = select_gallery(art, args.n_per_cell, args.seed)
    drawn = []
    for pk in picks:
        name = f"{pk['cell']}__{pk['shard']}__s{pk['scenario_idx']}.png"
        path = out / "scenarios" / name
        if not args.overwrite and path.exists():
            drawn.append({**pk, "file": f"scenarios/{name}"})
            continue
        if draw_scenario(art, pk["shard"], pk["scenario_idx"], str(path)):
            drawn.append({**pk, "file": f"scenarios/{name}"})
            print(f"wrote {path}  (risk={pk['risk']:.3f} failed={pk['failed']})", flush=True)
        else:
            print(f"SKIP (missing inputs): {pk['shard']} s{pk['scenario_idx']}", flush=True)

    with open(out / "gallery_index.json", "w") as h:
        json.dump({"n_drawn": len(drawn), "scenarios": drawn}, h, indent=2)
    write_index_html(out, drawn, art.decile_edges)
    print(f"\n{len(drawn)} scenario panels + overview + index.html -> {out}", flush=True)


if __name__ == "__main__":
    main()
