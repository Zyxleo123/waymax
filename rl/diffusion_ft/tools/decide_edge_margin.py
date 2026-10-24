"""Set the road-edge margin for ``edge_geometry="segment"`` by the rule that set 0.3.

The 0.3 m ``edge_margin_m`` is the human's p01 road-edge clearance (0.314 m on
WOMD shard 0, scenes 0-63, job 40089) measured under ``edge_geometry="point"``,
i.e. to the nearest roadgraph SAMPLE. ``"segment"`` measures to the polyline
through that sample, which is never farther, so every clearance -- and every
percentile -- can only shrink. Keeping 0.3 under ``segment`` would therefore be a
stricter barrier, and an A/B against the point arm would confound the geometry
with that strictness. So the segment arm keeps the RULE (margin = expert p01,
floored to the cm) and re-measures the number.

Reads two ``calibrate_expert`` runs over the SAME scenes, one per geometry, and
writes ``edge_margin.env`` (``DFT_EDGE_MARGIN=<m>``) only if every gate passes.
``train_safety.sbatch`` sources that file via ``DFT_EDGE_MARGIN_FILE`` and
refuses to train if it is missing. ``decision.json`` is written either way.

Gates:
  G1  the point control reproduces the recorded p01 (same scenes, same rule),
      so the number is re-measured with the pipeline that produced 0.3;
  G2  both runs pooled the same expert-clean steps (the sign is geometry-
      independent, so the dirty/clean split must be too);
  G3  segment <= point step by step (the distance is a min that includes the
      point distance; a violation means the segment path is broken);
  G4  the segment p01 is at least ``--floor`` (below it the barrier would only
      act at contact, and the rule has stopped meaning anything).

    python -m rl.diffusion_ft.tools.decide_edge_margin \\
        --point DIR/point --segment DIR/segment --out_dir DIR
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def grad_strength(c: np.ndarray, margin: float, tau: float) -> float:
    """Mean softplus-barrier gradient magnitude over steps (calibrate_expert's)."""
    z = (margin - c) / tau
    g = np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))),
                 np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))
    return float(g.mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--point", required=True, help="calibrate_expert out_dir, point")
    p.add_argument("--segment", required=True, help="calibrate_expert out_dir, segment")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ref_p01", type=float, default=0.314,
                   help="the point p01 on record for these scenes (job 40089)")
    p.add_argument("--ref_tol", type=float, default=0.01)
    p.add_argument("--point_margin", type=float, default=0.3,
                   help="the margin every point run trained with")
    p.add_argument("--floor", type=float, default=0.10)
    a = p.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    js = {g: json.loads((Path(d) / "expert_calibration.json").read_text())
          for g, d in (("point", a.point), ("segment", a.segment))}
    cs = {g: np.load(Path(d) / "expert_clearances.npz")["c_edge_clean"].astype(np.float64)
          for g, d in (("point", a.point), ("segment", a.segment))}
    geo = {g: js[g]["config"].get("edge_geometry") for g in js}
    tau = float(js["point"]["config"]["edge_tau_m"])
    p01 = {g: float(np.percentile(cs[g], 1)) for g in cs}

    gates = {}
    gates["G0 each run used its geometry"] = (geo == {"point": "point", "segment": "segment"},
                                             f"configs say {geo}")
    gates["G1 point control reproduces"] = (
        abs(p01["point"] - a.ref_p01) <= a.ref_tol,
        f"point p01 {p01['point']:+.4f} vs recorded {a.ref_p01:+.3f} (tol {a.ref_tol})")
    same = (cs["point"].shape == cs["segment"].shape
            and js["point"]["n_dirty"] == js["segment"]["n_dirty"])
    gates["G2 same clean steps"] = (
        same, f"{cs['point'].size} vs {cs['segment'].size} steps, dirty "
              f"{js['point']['n_dirty']} vs {js['segment']['n_dirty']}")
    if same:
        worst = float(np.max(cs["segment"] - cs["point"]))
        gates["G3 segment <= point per step"] = (worst <= 1e-4,
                                                 f"max(segment - point) = {worst:+.2e} m")
    else:
        gates["G3 segment <= point per step"] = (False, "not paired (G2 failed)")
    gates["G4 above the floor"] = (p01["segment"] >= a.floor,
                                   f"segment p01 {p01['segment']:+.4f} vs floor {a.floor}")
    ok = all(v[0] for v in gates.values())

    margin = math.floor(p01["segment"] * 100.0 + 1e-9) / 100.0
    qs = (0.1, 1, 5, 10, 25, 50)
    dec = {
        "pass": ok,
        "gates": {k: {"pass": bool(v[0]), "detail": v[1]} for k, v in gates.items()},
        "rule": "margin = floor(expert p01 under the arm's geometry, 0.01 m)",
        "segment_margin": margin if ok else None,
        "percentiles": {g: {f"p{q:g}": float(np.percentile(cs[g], q)) for q in qs}
                        for g in cs},
        # What the barrier asks of the human, in the calibration's own unit:
        # equal numbers here are what "same rule" buys. seg@point_margin is the
        # unrecalibrated flip, for scale.
        "expert_grad_strength": {
            f"point@{a.point_margin:g}": grad_strength(cs["point"], a.point_margin, tau),
            f"segment@{margin:g}": grad_strength(cs["segment"], margin, tau),
            f"segment@{a.point_margin:g}": grad_strength(cs["segment"], a.point_margin, tau),
        },
        "tau": tau, "inputs": {"point": a.point, "segment": a.segment},
    }
    if same:
        near = cs["point"] < 1.0
        d = cs["point"][near] - cs["segment"][near]
        dec["shift_near_edge"] = {"n": int(near.sum()),
                                  "p50": float(np.median(d)) if d.size else None,
                                  "p90": float(np.percentile(d, 90)) if d.size else None}

    print("[edge-margin] gates:")
    for k, v in gates.items():
        print(f"  {'PASS' if v[0] else 'FAIL'}  {k}: {v[1]}")
    for g in ("point", "segment"):
        print(f"[edge-margin] {g:<8} " + "  ".join(
            f"{k}={v:+.3f}" for k, v in dec["percentiles"][g].items()))
    if "shift_near_edge" in dec:
        s = dec["shift_near_edge"]
        print(f"[edge-margin] point - segment where point < 1 m: n={s['n']} "
              f"p50={s['p50']} p90={s['p90']}")
    print("[edge-margin] expert grad strength: " + "  ".join(
        f"{k} {100 * v:.2f}%" for k, v in dec["expert_grad_strength"].items()))

    (out / "decision.json").write_text(json.dumps(dec, indent=2))
    env = out / "edge_margin.env"
    if ok:
        env.write_text(f"DFT_EDGE_MARGIN={margin:.2f}\n")
        print(f"[edge-margin] PASS -> {env}: DFT_EDGE_MARGIN={margin:.2f}")
    else:
        env.unlink(missing_ok=True)
        print(f"[edge-margin] GATE FAILED -> no {env.name}; the segment arm will not "
              f"train. See {out / 'decision.json'}.")


if __name__ == "__main__":
    main()
