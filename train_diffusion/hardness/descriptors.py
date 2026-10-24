"""Per-scenario and per-anchor hardness descriptors for the WOMD cache shards.

Four candidate families, one per hypothesis about what makes the diffusion
policy fail (family A, expert-replay validity, is a label and lives in
labels.py):

  B  clearance  -- how much room the expert has before it would trip the metric.
                   The road-edge term reproduces waymax's own offroad criterion
                   (signed distance from each of the 4 bbox corners to the
                   nearest ROAD_EDGE_BOUNDARY/MEDIAN point, signed by the cross
                   product with the edge tangent), so it is the actual margin to
                   the violation rather than a proxy for it.  Targets the 89% of
                   failures that reach the goal and then violate.
  C  maneuver   -- curvature, heading change, braking, constant-velocity error.
                   Upstream cause of B: turns are where lateral error is largest
                   and road edges are closest.
  D  ambiguity  -- lane-polyline count and heading spread ahead of the ego.  The
                   polyline count is a density proxy, not a true branch count (a
                   real one needs the lane graph); the heading spread is the
                   sharper half of the pair.  A
                   diffusion policy can sample a valid-but-wrong branch at a
                   fork; neither margin nor length sees that.
  E  exposure   -- goal distance, path length, tortuosity.  Targets the 10.6% of
                   failures that never reach the goal.

Descriptors are computed on the *logged expert* trajectory, so they are all
available a priori for every scenario in the dataset -- including the 846 shards
that were never evaluated.  That is the point: the labels only cover 154 shards,
so the signal has to be carried by features that generalize beyond them.

The 2D road-edge distance drops waymax's z-stretch (the cache carries no z), so
overpasses can bind to the wrong deck.  Rare, and it makes the descriptor
conservative rather than optimistic.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# waymax datatypes.is_road_edge: ROAD_EDGE_BOUNDARY=15, ROAD_EDGE_MEDIAN=16.
ROAD_EDGE_TYPES = (15, 16)
# waymax MapElementIds lane centers.
LANE_TYPES = (1, 2, 3)

CACHE_KEYS = (
    "scenario_indices",
    "object_metadata_is_sdc",
    "log_trajectory_x",
    "log_trajectory_y",
    "log_trajectory_yaw",
    "log_trajectory_speed",
    "log_trajectory_valid",
    "log_trajectory_length",
    "log_trajectory_width",
    "roadgraph_points_x",
    "roadgraph_points_y",
    "roadgraph_points_ids",
    "roadgraph_points_types",
    "roadgraph_points_valid",
    "roadgraph_points_dir_x",
    "roadgraph_points_dir_y",
)

FEATURE_NAMES = (
    # --- B: clearance -------------------------------------------------------
    "road_clearance_min",       # metres of margin before offroad; <0 == already offroad
    "road_clearance_p10",
    "road_clearance_mean",
    "road_frames_below_0m5",    # sustained exposure, not just the worst instant
    "road_frames_below_1m",
    "agent_dist_min",           # centre-to-centre; no radius subtraction, never saturates
    "agent_dist_p10",
    "agent_clearance_min",      # centre distance minus both half-WIDTHS (inscribed circles)
    "agent_frames_below_5m",
    "agent_frames_below_10m",
    "n_agents_within_20m",
    # --- C: maneuver --------------------------------------------------------
    "heading_change_net",
    "heading_change_total",
    "curvature_max",
    "curvature_p90",
    "speed_mean",
    "speed_max",
    "decel_max",
    "accel_max",
    "stationary_frac",
    "cv_ade",                   # constant-velocity extrapolation error
    "cv_fde",
    # --- D: ambiguity -------------------------------------------------------
    "lane_polylines_ahead_50m",
    "lane_dir_spread_ahead",
    "lane_polylines_within_20m",
    # --- E: exposure --------------------------------------------------------
    "goal_dist",
    "path_len",
    "net_disp",
    "tortuosity",
    "n_steps",
)

FAMILY_OF = {name: fam for fam, names in {
    "B_clearance": FEATURE_NAMES[0:11],
    "C_maneuver": FEATURE_NAMES[11:22],
    "D_ambiguity": FEATURE_NAMES[22:25],
    "E_exposure": FEATURE_NAMES[25:30],
}.items() for name in names}


def _corners(x, y, yaw, length, width):
    """(T,) pose arrays -> (T, 4, 2) bounding-box corners."""
    half_l = length / 2.0
    half_w = width / 2.0
    off_x = np.stack([half_l, half_l, -half_l, -half_l], axis=-1)
    off_y = np.stack([half_w, -half_w, -half_w, half_w], axis=-1)
    cos_t = np.cos(yaw)[:, None]
    sin_t = np.sin(yaw)[:, None]
    return np.stack(
        [x[:, None] + off_x * cos_t - off_y * sin_t,
         y[:, None] + off_x * sin_t + off_y * cos_t],
        axis=-1,
    )


class _RoadEdges:
    """Signed distance to the nearest road edge, matching waymax in 2D.

    Indices are kept in the *full* roadgraph array so that ``nearest - 1`` is the
    preceding point of the same polyline exactly as waymax computes it; querying
    a tree built only on edge points would silently pick a different neighbour.
    """

    def __init__(self, xy, dir_xy, ids, types, valid):
        mask = valid & np.isin(types, ROAD_EDGE_TYPES)
        self.idx = np.flatnonzero(mask)
        self.xy = xy
        self.dir = dir_xy
        self.ids = ids
        self.tree = cKDTree(xy[self.idx]) if self.idx.size else None

    def signed_distance(self, query_xy):
        if self.tree is None:
            return np.full(query_xy.shape[0], np.nan, dtype=np.float64)
        _, nn = self.tree.query(query_xy)
        full = self.idx[nn]
        prior = np.maximum(full - 1, 0)
        near_xy = self.xy[full]
        near_dir = self.dir[full]
        prior_dir = self.dir[prior]
        to_edge = query_xy - near_xy
        cross = to_edge[:, 0] * near_dir[:, 1] - to_edge[:, 1] * near_dir[:, 0]
        cross_prior = to_edge[:, 0] * prior_dir[:, 1] - to_edge[:, 1] * prior_dir[:, 0]
        same_curve = self.ids[full] == self.ids[prior]
        sign = np.sign(np.where(same_curve & (cross_prior < cross), cross_prior, cross))
        return np.linalg.norm(near_xy - query_xy, axis=-1) * sign


def _nan_stats(values, reducer, default=np.nan):
    values = values[np.isfinite(values)]
    return float(reducer(values)) if values.size else float(default)


def _window_features(scn, t0, t1, goal_xy, dt):
    """Descriptors over world timesteps ``[t0, t1]`` inclusive."""
    ego = scn["ego"]
    steps = np.arange(t0, min(t1, scn["n_t"] - 1) + 1)
    valid = ego["valid"][steps]
    steps = steps[valid]
    out = {name: np.nan for name in FEATURE_NAMES}
    out["n_steps"] = float(steps.size)
    if steps.size < 3:
        return out

    x = ego["x"][steps]
    y = ego["y"][steps]
    yaw = ego["yaw"][steps]
    speed = ego["speed"][steps]

    # --- B: road-edge clearance -------------------------------------------
    corners = _corners(x, y, yaw, ego["length"][steps], ego["width"][steps])
    signed = scn["edges"].signed_distance(corners.reshape(-1, 2)).reshape(-1, 4)
    # waymax calls the box offroad when ANY corner has a positive signed
    # distance, so the worst corner is what matters; flip so >0 means margin.
    clearance = -np.nanmax(signed, axis=-1)
    out["road_clearance_min"] = _nan_stats(clearance, np.min)
    out["road_clearance_p10"] = _nan_stats(clearance, lambda v: np.percentile(v, 10))
    out["road_clearance_mean"] = _nan_stats(clearance, np.mean)
    finite = clearance[np.isfinite(clearance)]
    if finite.size:
        out["road_frames_below_0m5"] = float((finite < 0.5).sum())
        out["road_frames_below_1m"] = float((finite < 1.0).sum())

    # --- B: agent gaps ------------------------------------------------------
    others = scn["others"]
    if others["n"]:
        ox = others["x"][:, steps]
        oy = others["y"][:, steps]
        ovalid = others["valid"][:, steps]
        centre = np.where(ovalid, np.hypot(ox - x[None, :], oy - y[None, :]), np.inf)
        per_step = centre.min(axis=0)
        finite = per_step[np.isfinite(per_step)]
        if finite.size:
            out["agent_dist_min"] = float(finite.min())
            out["agent_dist_p10"] = float(np.percentile(finite, 10))
            out["agent_frames_below_5m"] = float((finite < 5.0).sum())
            out["agent_frames_below_10m"] = float((finite < 10.0).sum())
        # Inscribed circles (half-width), not half-diagonals: the bounding-circle
        # version reads as a collision for any two cars in adjacent lanes and so
        # carries no signal.
        clear = np.where(ovalid, centre - (others["half_width"][:, None] + ego["half_width"]), np.inf)
        finite_clear = clear.min(axis=0)
        finite_clear = finite_clear[np.isfinite(finite_clear)]
        if finite_clear.size:
            out["agent_clearance_min"] = float(finite_clear.min())
        out["n_agents_within_20m"] = float((centre.min(axis=1) < 20.0).sum())
    else:
        out["n_agents_within_20m"] = 0.0

    # --- C: maneuver --------------------------------------------------------
    unwrapped = np.unwrap(yaw)
    dyaw = np.diff(unwrapped)
    seg = np.hypot(np.diff(x), np.diff(y))
    out["heading_change_net"] = float(abs(unwrapped[-1] - unwrapped[0]))
    out["heading_change_total"] = float(np.abs(dyaw).sum())
    curvature = np.abs(dyaw) / np.maximum(seg, 1e-2)
    curvature = curvature[seg > 0.05]  # undefined while parked
    if curvature.size:
        out["curvature_max"] = float(curvature.max())
        out["curvature_p90"] = float(np.percentile(curvature, 90))
    out["speed_mean"] = float(speed.mean())
    out["speed_max"] = float(speed.max())
    accel = np.diff(speed) / dt
    if accel.size:
        out["decel_max"] = float(max(0.0, -accel.min()))
        out["accel_max"] = float(max(0.0, accel.max()))
    out["stationary_frac"] = float((speed < 0.5).mean())

    # Constant-velocity extrapolation from the first frame: the cheapest honest
    # measure of "this future was trivially predictable".
    span = (steps - steps[0]) * dt
    vx = np.cos(yaw[0]) * speed[0]
    vy = np.sin(yaw[0]) * speed[0]
    err = np.hypot(x - (x[0] + vx * span), y - (y[0] + vy * span))
    out["cv_ade"] = float(err.mean())
    out["cv_fde"] = float(err[-1])

    # --- D: ambiguity -------------------------------------------------------
    lanes = scn["lanes"]
    if lanes["n"]:
        rel_x = lanes["x"] - x[0]
        rel_y = lanes["y"] - y[0]
        cos_t, sin_t = np.cos(-yaw[0]), np.sin(-yaw[0])
        fwd = rel_x * cos_t - rel_y * sin_t
        lat = rel_x * sin_t + rel_y * cos_t
        dist = np.hypot(rel_x, rel_y)
        ahead = (fwd > 0.0) & (dist < 50.0) & (np.abs(lat) < 25.0)
        if ahead.any():
            out["lane_polylines_ahead_50m"] = float(np.unique(lanes["ids"][ahead]).size)
            heading = np.arctan2(lanes["dir_y"][ahead], lanes["dir_x"][ahead]) - yaw[0]
            out["lane_dir_spread_ahead"] = float(
                np.sqrt(-2.0 * np.log(max(np.hypot(np.cos(heading).mean(), np.sin(heading).mean()), 1e-6)))
            )
        else:
            out["lane_polylines_ahead_50m"] = 0.0
        out["lane_polylines_within_20m"] = float(np.unique(lanes["ids"][dist < 20.0]).size)

    # --- E: exposure --------------------------------------------------------
    out["path_len"] = float(seg.sum())
    out["net_disp"] = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    out["tortuosity"] = float(out["path_len"] / max(out["net_disp"], 1e-3))
    if goal_xy is not None and np.all(np.isfinite(goal_xy)):
        out["goal_dist"] = float(np.hypot(goal_xy[0] - x[0], goal_xy[1] - y[0]))
    return out


def _prepare_scenario(payload, row, ego_idx):
    is_sdc = payload["object_metadata_is_sdc"][row]
    if ego_idx is None or ego_idx < 0:
        ego_idx = int(np.argmax(is_sdc))
    valid = payload["log_trajectory_valid"][row]
    length = payload["log_trajectory_length"][row]
    width = payload["log_trajectory_width"][row]

    ego_len = float(np.nanmedian(length[ego_idx][valid[ego_idx]])) if valid[ego_idx].any() else 4.5
    ego_wid = float(np.nanmedian(width[ego_idx][valid[ego_idx]])) if valid[ego_idx].any() else 2.0
    ego = {
        "x": payload["log_trajectory_x"][row, ego_idx].astype(np.float64),
        "y": payload["log_trajectory_y"][row, ego_idx].astype(np.float64),
        "yaw": payload["log_trajectory_yaw"][row, ego_idx].astype(np.float64),
        "speed": payload["log_trajectory_speed"][row, ego_idx].astype(np.float64),
        "valid": valid[ego_idx].astype(bool),
        "length": np.where(valid[ego_idx], length[ego_idx], ego_len).astype(np.float64),
        "width": np.where(valid[ego_idx], width[ego_idx], ego_wid).astype(np.float64),
        "half_width": 0.5 * ego_wid,
    }

    keep = np.ones(valid.shape[0], dtype=bool)
    keep[ego_idx] = False
    keep &= valid.any(axis=1)
    others = {
        "n": int(keep.sum()),
        "x": payload["log_trajectory_x"][row][keep].astype(np.float64),
        "y": payload["log_trajectory_y"][row][keep].astype(np.float64),
        "valid": valid[keep],
        "half_width": 0.5 * np.nan_to_num(
            np.nanmax(np.where(valid[keep], width[keep], np.nan), axis=1), nan=2.0
        ),
    }

    rg_valid = payload["roadgraph_points_valid"][row].astype(bool)
    rg_xy = np.stack([payload["roadgraph_points_x"][row], payload["roadgraph_points_y"][row]], axis=-1).astype(np.float64)
    rg_dir = np.stack([payload["roadgraph_points_dir_x"][row], payload["roadgraph_points_dir_y"][row]], axis=-1).astype(np.float64)
    rg_ids = payload["roadgraph_points_ids"][row]
    rg_types = payload["roadgraph_points_types"][row]

    lane_mask = rg_valid & np.isin(rg_types, LANE_TYPES)
    lanes = {
        "n": int(lane_mask.sum()),
        "x": rg_xy[lane_mask, 0],
        "y": rg_xy[lane_mask, 1],
        "dir_x": rg_dir[lane_mask, 0],
        "dir_y": rg_dir[lane_mask, 1],
        "ids": rg_ids[lane_mask],
    }
    ego_valid_t = np.flatnonzero(ego["valid"])
    goal_t = int(ego_valid_t[-1]) if ego_valid_t.size else 0
    return {
        "ego": ego,
        "others": others,
        "lanes": lanes,
        # simulation/evaluation_utils.infer_goals_from_sim_state: the goal is the
        # ego's last valid logged position. Recomputing it here means unevaluated
        # shards get the same goal_dist feature as the labelled ones.
        "goal_fallback": np.array([ego["x"][goal_t], ego["y"][goal_t]]),
        "edges": _RoadEdges(rg_xy, rg_dir, rg_ids, rg_types, rg_valid),
        "n_t": ego["x"].shape[0],
        "ego_idx": ego_idx,
    }


def process_shard(cache_path, goal_lookup, ego_lookup, start_t, horizon_end, anchors, anchor_span, dt):
    payload = {key: None for key in CACHE_KEYS}
    with np.load(cache_path, allow_pickle=True) as handle:
        for key in CACHE_KEYS:
            payload[key] = handle[key]

    scenario_indices = payload["scenario_indices"].astype(np.int32)
    n = scenario_indices.size
    scenario_feat = np.full((n, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    anchor_feat = np.full((n, len(anchors), len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    ego_used = np.full(n, -1, dtype=np.int32)

    for row in range(n):
        scenario_idx = int(scenario_indices[row])
        scn = _prepare_scenario(payload, row, ego_lookup.get(scenario_idx))
        ego_used[row] = scn["ego_idx"]
        goal_xy = goal_lookup.get(scenario_idx)
        if goal_xy is None or not np.all(np.isfinite(goal_xy)):
            goal_xy = scn["goal_fallback"]
        end = horizon_end if horizon_end >= 0 else scn["n_t"] - 1
        feats = _window_features(scn, start_t, end, goal_xy, dt)
        scenario_feat[row] = [feats[name] for name in FEATURE_NAMES]
        for slot, anchor in enumerate(anchors):
            feats_a = _window_features(scn, anchor, anchor + anchor_span, goal_xy, dt)
            anchor_feat[row, slot] = [feats_a[name] for name in FEATURE_NAMES]
    return scenario_indices, scenario_feat, anchor_feat, ego_used


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--labels", default=None, help="labels.npz from labels.py")
    parser.add_argument("--shard_source", choices=("labels", "cache"), default="labels",
                        help="'labels' = the evaluated shards only; 'cache' = every shard in cache_dir")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--num_tasks", type=int, default=1)
    parser.add_argument("--start_timestep", type=int, default=10,
                        help="must match the eval's --start_timestep (runner default 10)")
    parser.add_argument("--horizon_end", type=int, default=-1, help="-1 == end of scenario")
    parser.add_argument("--anchors", type=int, nargs="*", default=[0, 10, 20, 30, 40])
    parser.add_argument("--anchor_span", type=int, default=50, help="world steps per anchor window")
    parser.add_argument("--world_dt", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    labels = np.load(args.labels, allow_pickle=True) if args.labels else None
    if args.shard_source == "cache":
        shards = sorted(
            os.path.basename(p)[: -len(".sim_state_cache.npz")]
            for p in glob.glob(os.path.join(args.cache_dir, "*.sim_state_cache.npz"))
        )
    else:
        if labels is None:
            raise SystemExit("--shard_source labels requires --labels")
        shards = sorted(np.unique(labels["shard"]).tolist())
    mine = shards[args.task_id :: args.num_tasks]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"task {args.task_id}/{args.num_tasks}: {len(mine)} of {len(shards)} shards", flush=True)

    for shard in mine:
        out_path = Path(args.out_dir) / f"{shard}.desc.npz"
        if out_path.exists() and not args.overwrite:
            print(f"skip (done): {shard}", flush=True)
            continue
        cache_path = Path(args.cache_dir) / f"{shard}.sim_state_cache.npz"
        if not cache_path.exists():
            print(f"MISSING cache, skipping: {cache_path}", flush=True)
            continue

        goal_lookup: dict[int, np.ndarray] = {}
        ego_lookup: dict[int, int] = {}
        if labels is not None:
            sel = labels["shard"] == shard
            goal_lookup = {
                int(s): np.array([gx, gy])
                for s, gx, gy in zip(labels["scenario_idx"][sel], labels["goal_x"][sel], labels["goal_y"][sel])
            }
            ego_lookup = {int(s): int(e) for s, e in zip(labels["scenario_idx"][sel], labels["ego_idx"][sel])}

        scenario_indices, scenario_feat, anchor_feat, ego_used = process_shard(
            cache_path, goal_lookup, ego_lookup,
            args.start_timestep, args.horizon_end,
            tuple(args.anchors), args.anchor_span, args.world_dt,
        )
        tmp_path = out_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp_path,
            shard=np.asarray(shard),
            scenario_indices=scenario_indices,
            ego_idx=ego_used,
            feature_names=np.asarray(FEATURE_NAMES),
            anchors=np.asarray(args.anchors, dtype=np.int32),
            scenario_feat=scenario_feat,
            anchor_feat=anchor_feat,
        )
        os.replace(tmp_path, out_path)  # atomic: a killed job never leaves a half-file
        print(f"wrote {out_path} ({scenario_indices.size} scenarios)", flush=True)

    with open(Path(args.out_dir) / f"_task_{args.task_id:04d}.done.json", "w") as handle:
        json.dump({"task_id": args.task_id, "num_shards": len(mine)}, handle)


if __name__ == "__main__":
    main()
