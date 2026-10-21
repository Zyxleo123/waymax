"""Compare two closed-loop runs' executed ego paths on scenes where their offroad verdicts differ.

For every selected scene, the executed path saved with --save_trajectory is checked against the road
edges exactly as the metric does (scripts/audit_offroad_z.py: Waymax's rule, ego box at its height
from the rollout start) and summarised per run:
- whether the re-run reproduces the original offroad verdict;
- first offroad step, the replan window it falls in, and the deepest violation (m past the edge);
- edge clearance: the smallest distance from the edge over the episode (negative = offroad) and over
  the last 2 s before the goal;
- lateral offset from the logged human path and mean speed.

    python scripts/offroad_debug_compare.py --scenes selected_scenes.json --out compare.json \
        --run ft=<run dir> --run a2margin=<run dir>
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os

import numpy as np
from waymax import config as waymax_config

from data.scenario_loader import load_scenario_state_batch_fast
from scores.scorer_lane_graph import _corner_signed_distances, road_edge_points

import jax.numpy as jnp

SUBSET_DIR = "/zfsauton/scratch/sbellad/Diffusion_ES_sim/es_failure_subset_n1500_seed0"
START_T, GOAL_T, REPLAN_EVERY = 10, 90, 10


def corner_dist(boxes, edges, z):
    """Per-step max over the 4 corners of Waymax's signed edge distance (> 0 offroad)."""
    d = np.asarray(_corner_signed_distances(jnp.asarray(boxes[None]), *edges,
                                            jnp.asarray(z, jnp.float32)[None, :, None, None]))[0]
    return d.max(-1)


def lateral_offset(path_xy, log_xy, log_valid):
    """Distance from each executed point to the nearest logged-path point."""
    ref = log_xy[log_valid]
    d = np.linalg.norm(path_xy[:, None, :] - ref[None, :, :], axis=-1)
    return d.min(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", required=True, help="json list of [tfrecord, scenario_idx]")
    p.add_argument("--run", action="append", required=True, help="name=run dir (chunk_*/ with trajectory npz)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    scenes = [tuple(s) for s in json.load(open(a.scenes))]
    runs = dict(r.split("=", 1) for r in a.run)

    rows = []
    for name in sorted({s[0] for s in scenes}):
        idx = sorted(int(s[1]) for s in scenes if s[0] == name)
        cfg = dataclasses.replace(waymax_config.WOD_1_3_1_TRAINING,
                                  path=os.path.join(SUBSET_DIR, "tfrecords", "training", name),
                                  max_num_objects=128, batch_dims=(len(idx),), shuffle_seed=0)
        st, to_batch = load_scenario_state_batch_fast(cfg, idx)
        for s in idx:
            w = int(to_batch[s])
            log = st.log_trajectory
            e = int(np.argmax(np.asarray(st.object_metadata.is_sdc[w])))
            g = lambda arr: np.asarray(arr[w, e])  # noqa: E731
            lw = np.array([g(log.length)[START_T], g(log.width)[START_T]], np.float32)
            z0 = float(g(log.z)[START_T])
            log_xy = np.stack([g(log.x), g(log.y)], -1)
            log_valid = g(log.valid).astype(bool)
            edges = road_edge_points(st.roadgraph_points, w)
            win = slice(START_T, GOAL_T + 1)
            log_boxes = np.concatenate([log_xy, g(log.yaw)[:, None], np.broadcast_to(lw, (len(log_xy), 2))], -1)[win]
            log_d = corner_dist(log_boxes.astype(np.float32), edges, np.full((len(log_boxes),), z0)) \
                if edges is not None else None
            row = {"tfrecord": name, "scenario_idx": s,
                   "log_offroad": bool(log_d is not None and (log_d[log_valid[win]] > 0).any()),
                   "log_min_clearance_m": float(-log_d[log_valid[win]].max()) if log_d is not None else None}
            for rn, rd in runs.items():
                out = glob.glob(os.path.join(rd, "chunk_*", f"{name}.scenario_{s:03d}.json"))
                trj = glob.glob(os.path.join(rd, "chunk_*", f"{name}.scenario_{s:03d}.trajectory.npz"))
                if not out or not trj or edges is None:
                    row[rn] = None
                    continue
                res = json.load(open(out[0]))
                tr = np.load(trj[0])["predicted_trajectory"]                       # [T, 5] x, y, yaw, vx, vy
                end = int(res["reached_timestep"]) if res.get("reached_timestep", -1) not in (None, -1) else GOAL_T
                sl = slice(START_T, min(end, GOAL_T) + 1)
                boxes = np.concatenate([tr[sl, :3], np.broadcast_to(lw, (len(tr[sl]), 2))], -1).astype(np.float32)
                d = corner_dist(boxes, edges, np.full((len(boxes),), z0))
                off = d > 0
                first = int(np.argmax(off)) + START_T if off.any() else None
                last2 = d[-21:]
                lat = lateral_offset(tr[sl, :2], log_xy, log_valid)
                row[rn] = {
                    "success": res["success"], "offroad": res["offroad"], "overlap": res["overlap"],
                    "goal_reached": res["goal_reached"], "final_goal_dist_m": res["final_goal_distance_m"],
                    "reached_t": res.get("reached_timestep"),
                    "first_offroad_t": first,
                    "first_offroad_replan": (first - START_T) // REPLAN_EVERY if first is not None else None,
                    "offroad_steps": int(off.sum()),
                    "max_depth_m": float(d.max()) if off.any() else 0.0,
                    "min_clearance_m": float(-d.max()),
                    "min_clearance_last2s_m": float(-last2.max()),
                    "mean_lateral_offset_m": float(lat.mean()), "max_lateral_offset_m": float(lat.max()),
                    "mean_speed_mps": float(np.linalg.norm(tr[sl, 3:5], axis=-1).mean()),
                }
            rows.append(row)
        print(f"{name}: {len(rows)} scenes", flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
