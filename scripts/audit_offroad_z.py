"""Which scenes does the road-tracked offroad height (0ce9cab) affect?

Offroad is decided by the nearest road edge in 3D, so the box height matters. Before 0ce9cab:
- evaluation (Waymax's metric): the ego keeps its height from the rollout start (t=10), since
  StateDynamics never updates z;
- planner scorer: each 5 s plan is checked at the ego's height at the replan time.
After it, both follow the road surface (track_ground_z). This script replays the LOGGED ego
drive of every target scene under the three heights (frozen at t=10, road-tracked from t=10,
logged z) and, for the planner, compares frozen vs tracked height over every replan window.
The log is a proxy: the planner's executed path differs, but a scene where the height choice
changes the verdict on the human drive (or where the road height changes a lot) is one where
the change can matter.

    python scripts/audit_offroad_z.py --out offroad_z_audit.json \
        --run oct13=<sim dir> --run sweep_T1=<sim dir>
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import glob
import json
import os

import jax.numpy as jnp
import numpy as np
from waymax import config as waymax_config

from data.scenario_loader import load_scenario_state_batch_fast
from scores.scorer_lane_graph import _corner_signed_distances, ground_points, road_edge_points, track_ground_z

SUBSET_DIR = "/zfsauton/scratch/sbellad/Diffusion_ES_sim/es_failure_subset_n1500_seed0"
START_T, GOAL_T = 10, 90          # episode window of the closed-loop metric
REPLAN_EVERY, PLAN_STEPS = 10, 50
DZ_TOL_M = 0.5                    # "road height changes" threshold


def offroad_steps(boxes, edges, z):
    d = np.asarray(_corner_signed_distances(jnp.asarray(boxes[None]), *edges,
                                            jnp.asarray(z, jnp.float32)[None, :, None, None]))[0]
    return (d > 0.0).any(-1)


def tracked_z(boxes, z0, ground):
    return np.asarray(track_ground_z(jnp.asarray(boxes[None, :, :2]), jnp.asarray([z0], jnp.float32), ground))[0]


def audit_scene(st, w):
    log = st.log_trajectory
    e = int(np.argmax(np.asarray(st.object_metadata.is_sdc[w])))
    get = lambda a: np.asarray(a[w, e])  # noqa: E731
    boxes = np.stack([get(log.x), get(log.y), get(log.yaw), get(log.length), get(log.width)], -1).astype(np.float32)
    z_log, valid = get(log.z).astype(np.float32), get(log.valid).astype(bool)
    edges = road_edge_points(st.roadgraph_points, w)
    if edges is None:
        return None
    ground = ground_points(st.roadgraph_points, w)
    win = slice(START_T, GOAL_T + 1)
    b, zl, v = boxes[win], z_log[win], valid[win]
    z_frozen = np.full_like(zl, z_log[START_T])
    z_track = tracked_z(b, z_log[START_T], ground)
    off = {name: offroad_steps(b, edges, z) & v for name, z in
           (("frozen", z_frozen), ("tracked", z_track), ("logged", zl))}

    # Planner: each replan window, height frozen at the replan step vs tracked from it.
    planner_windows = 0
    for t in range(START_T, GOAL_T, REPLAN_EVERY):
        s = slice(t, min(t + PLAN_STEPS, GOAL_T) + 1)
        bw, vw = boxes[s], valid[s]
        if not vw.any():
            continue
        fz = offroad_steps(bw, edges, np.full((len(bw),), z_log[t], np.float32)) & vw
        tz = offroad_steps(bw, edges, tracked_z(bw, z_log[t], ground)) & vw
        planner_windows += int(fz.any() != tz.any())

    first = lambda f: int(np.argmax(f)) + START_T if f.any() else None  # noqa: E731
    return {
        "log_offroad_frozen": bool(off["frozen"].any()),
        "log_offroad_tracked": bool(off["tracked"].any()),
        "log_offroad_logged_z": bool(off["logged"].any()),
        "first_offroad_frozen": first(off["frozen"]),
        "first_offroad_tracked": first(off["tracked"]),
        "steps_differ": int((off["frozen"] != off["tracked"]).sum()),
        "planner_windows_differ": planner_windows,
        "max_dz_log_m": float(np.abs(zl[v] - z_log[START_T]).max()) if v.any() else 0.0,
        "max_track_err_m": float(np.abs(z_track[v] - zl[v]).max()) if v.any() else 0.0,
    }


def load_outcomes(sim_dir):
    out = {}
    for f in glob.glob(os.path.join(sim_dir, "*.scenario_*.json")):
        r = json.load(open(f))
        out[(os.path.basename(r["tfrecord"]), int(r["scenario_idx"]))] = r
    return out


def outcome(r):
    if r["success"]:
        return "success"
    parts = [k for k in ("overlap", "offroad") if r[k]] + ([] if r["goal_reached"] else ["no_goal"])
    return "+".join(parts) or "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset_dir", default=SUBSET_DIR)
    p.add_argument("--out", required=True)
    p.add_argument("--run", action="append", default=[], help="name=simulation dir; joined on the scene")
    p.add_argument("--max_tfrecords", type=int, default=None)
    a = p.parse_args()

    targets = json.load(open(os.path.join(a.subset_dir, "target_scenarios.json")))
    names = sorted(t for t, v in targets.items() if v)[: a.max_tfrecords]
    rows = []
    for i, name in enumerate(names):
        idx = sorted(int(x) for x in targets[name])
        cfg = dataclasses.replace(waymax_config.WOD_1_3_1_TRAINING,
                                  path=os.path.join(a.subset_dir, "tfrecords", "training", name),
                                  max_num_objects=128, batch_dims=(len(idx),), shuffle_seed=0)
        st, to_batch = load_scenario_state_batch_fast(cfg, idx)
        for s in idx:
            r = audit_scene(st, int(to_batch[s]))
            if r is not None:
                rows.append({"tfrecord": name, "scenario_idx": s, **r})
        print(f"[{i + 1}/{len(names)}] {name}: {len(rows)} scenes", flush=True)

    runs = {n: load_outcomes(d) for n, d in (r.split("=", 1) for r in a.run)}
    for row in rows:
        key = (row["tfrecord"], row["scenario_idx"])
        for n, res in runs.items():
            row[f"outcome_{n}"] = outcome(res[key]) if key in res else None
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)

    n = len(rows)
    metric = [r for r in rows if r["log_offroad_frozen"] != r["log_offroad_tracked"]]
    planner = [r for r in rows if r["planner_windows_differ"] > 0]
    sloped = [r for r in rows if r["max_dz_log_m"] > DZ_TOL_M]
    print(f"\n{n} scenes")
    print(f"  logged drive offroad: frozen z {sum(r['log_offroad_frozen'] for r in rows)}, "
          f"road-tracked z {sum(r['log_offroad_tracked'] for r in rows)}, "
          f"logged z {sum(r['log_offroad_logged_z'] for r in rows)}")
    print(f"  metric verdict on the logged drive changes: {len(metric)} "
          f"(frozen-only {sum(r['log_offroad_frozen'] for r in metric)}, "
          f"tracked-only {sum(r['log_offroad_tracked'] for r in metric)})")
    print(f"  planner verdict changes in >=1 replan window of the logged drive: {len(planner)}")
    print(f"  road height changes > {DZ_TOL_M} m over t={START_T}..{GOAL_T}: {len(sloped)}")
    print(f"  any of the above: {len({(r['tfrecord'], r['scenario_idx']) for r in metric + planner})} (metric or planner)")

    names = list(runs)
    for x, y in zip(names, names[1:]):
        print(f"\n{x} -> {y}, by group")
        for label, grp in (("metric-affected", metric), ("planner-affected", planner),
                           ("height > tol", sloped), ("all", rows)):
            pairs = [(r[f"outcome_{x}"], r[f"outcome_{y}"]) for r in grp
                     if r[f"outcome_{x}"] is not None and r[f"outcome_{y}"] is not None]
            if not pairs:
                continue
            fixed = collections.Counter(o for o, nn in pairs if nn == "success" and o != "success")
            broke = sum(o == "success" and nn != "success" for o, nn in pairs)
            print(f"  {label:17s} n={len(pairs):4d} success {sum(o == 'success' for o, _ in pairs):4d} -> "
                  f"{sum(nn == 'success' for _, nn in pairs):4d}; fixed {dict(fixed.most_common())}; broke {broke}")


if __name__ == "__main__":
    main()
