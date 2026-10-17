"""Characterize the fail1500 scenario subset vs. a control set of FM-success scenarios.

Computes per-scenario features from the original WOMD log (feasibility of the expert
log under the sim's offroad/overlap metrics, kinematics, interaction density, ...) and,
for the subset, features of the FM planner's failing rollout (failure step, collision
geometry, speed vs. log). Joins with diffusion / diffusion+IDM rerun outcomes.
"""
from __future__ import annotations

import glob
import json
import os
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import pandas as pd

SRC_EXP = Path("/zfsauton/scratch/mineuih/waymax_rs/exp/simulation_fm_training_collect_failure_20260903_030456")
SRC_CACHE = Path("/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz")
MANIFEST = Path("/zfsauton/scratch/sbellad/Diffusion_ES_sim/es_failure_subset_n1500_seed0/subset_manifest.json")
RUNS = {
    "diff": "/zfsauton/scratch/sbellad/diffusion_sim/diffusion_only_fail1500_seed0",
    "diffidm": "/zfsauton/scratch/sbellad/diffusion_sim/diffusion_only_IDM_fail1500_seed0",
}
OUT = Path("/zfsauton/scratch/sbellad/diffusion_sim/analysis")
START_T = 10
DT = 0.1
CONTROL_PER_TFRECORD = 12
ROAD_EDGE_TYPES = (15, 16)
VEH, PED, CYC = 1, 2, 3


def box_corners(x, y, yaw, length, width):
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = length / 2 * c, length / 2 * s
    wx, wy = width / 2 * s, width / 2 * c
    return np.stack(
        [
            np.stack([x + dx - wx, y + dy + wy], -1),
            np.stack([x + dx + wx, y + dy - wy], -1),
            np.stack([x - dx + wx, y - dy - wy], -1),
            np.stack([x - dx - wx, y - dy + wy], -1),
        ],
        -2,
    )


def signed_edge_distance(q_xy, rg):
    """Waymax-style signed distance (positive = offroad), 2D. q_xy: [Q,2]."""
    pts, dirs, ids = rg
    d2 = ((pts[None, :, :] - q_xy[:, None, :]) ** 2).sum(-1)
    nn = d2.argmin(-1)
    prior = np.maximum(nn - 1, 0)
    p2e = q_xy - pts[nn]
    cross = p2e[:, 0] * dirs[nn, 1] - p2e[:, 1] * dirs[nn, 0]
    cross_p = p2e[:, 0] * dirs[prior, 1] - p2e[:, 1] * dirs[prior, 0]
    same = ids[nn] == ids[prior]
    sign = np.sign(np.where(same & (cross_p < cross), cross_p, cross))
    return np.sqrt(d2[np.arange(len(nn)), nn]) * sign


def polys_overlap(a, b):
    """SAT test, a: [4,2], b: [K,4,2] -> [K] bool."""
    def axes(p):
        e = np.roll(p, -1, axis=-2) - p
        n = np.stack([-e[..., 1], e[..., 0]], -1)
        return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)

    ax = np.concatenate([np.broadcast_to(axes(a), (len(b), 4, 2)), axes(b)], axis=1)  # [K,8,2]
    pa = np.einsum("cd,kad->kac", a, ax)  # [K,8,4]
    pb = np.einsum("kcd,kad->kac", b, ax)
    sep = (pa.max(-1) < pb.min(-1)) | (pb.max(-1) < pa.min(-1))
    return ~sep.any(-1)


def traj_metrics(x, y, yaw, L, W, valid, types, ego, t0, t1, rg):
    """Offroad / overlap of ego over [t0, t1] given full trajectory arrays [N,T]."""
    out = {"offroad": False, "offroad_t": -1, "max_offroad_m": 0.0, "min_edge_clear_m": np.nan,
           "overlap": False, "overlap_t": -1, "overlap_obj": -1, "overlap_type": -1}
    ts = [t for t in range(t0, t1 + 1) if valid[ego, t]]
    if not ts:
        return out
    ts = np.array(ts)
    corners = box_corners(x[ego, ts], y[ego, ts], yaw[ego, ts], L[ego, ts], W[ego, ts])  # [T,4,2]
    if rg is not None:
        sd = signed_edge_distance(corners.reshape(-1, 2), rg).reshape(len(ts), 4).max(-1)
        off = sd > 0
        out["max_offroad_m"] = float(max(sd.max(), 0.0))
        out["min_edge_clear_m"] = float(-sd.max()) if not off.any() else float(-sd.max())
        if off.any():
            out["offroad"] = True
            out["offroad_t"] = int(ts[off.argmax()])
    others = np.array([i for i in range(x.shape[0]) if i != ego])
    for k, t in enumerate(ts):
        m = others[valid[others, t]]
        if len(m) == 0:
            continue
        near = np.hypot(x[m, t] - x[ego, t], y[m, t] - y[ego, t]) < 15
        m = m[near]
        if len(m) == 0:
            continue
        ob = box_corners(x[m, t], y[m, t], yaw[m, t], L[m, t], W[m, t])
        hit = polys_overlap(corners[k], ob)
        if hit.any():
            j = int(m[hit.argmax()])
            out.update(overlap=True, overlap_t=int(t), overlap_obj=j, overlap_type=int(types[j]))
            break
    return out


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def log_features(d, ego, rg):
    x, y, yaw = d["x"], d["y"], d["yaw"]
    vx, vy, valid = d["vx"], d["vy"], d["valid"]
    L, W, types = d["L"], d["W"], d["types"]
    vt = np.flatnonzero(valid[ego])
    goal_t = int(vt[-1]) if len(vt) else 0
    f = {"goal_t": goal_t}
    t0, t1 = START_T, goal_t
    seg = np.arange(t0, t1 + 1)
    seg = seg[valid[ego, seg]]
    f["ego_invalid_steps"] = int((t1 - t0 + 1) - len(seg))
    if len(seg) < 2:
        f["degenerate"] = True
        return f
    f["degenerate"] = False
    ex, ey, eyaw = x[ego, seg], y[ego, seg], yaw[ego, seg]
    spd = np.hypot(vx[ego, seg], vy[ego, seg])
    f["speed0"] = float(spd[0])
    f["speed_mean"] = float(spd.mean())
    f["speed_max"] = float(spd.max())
    f["path_len"] = float(np.hypot(np.diff(ex), np.diff(ey)).sum())
    f["disp"] = float(np.hypot(ex[-1] - ex[0], ey[-1] - ey[0]))
    f["stopped_frac"] = float((spd < 0.5).mean())
    acc = np.diff(spd) / DT
    acc = np.convolve(acc, np.ones(5) / 5, mode="valid") if len(acc) >= 5 else acc
    f["max_decel"] = float(-acc.min()) if len(acc) else 0.0
    f["max_accel"] = float(acc.max()) if len(acc) else 0.0
    f["heading_change_deg"] = float(np.degrees(np.abs(wrap(eyaw[-1] - eyaw[0]))))
    f["signed_heading_change_deg"] = float(np.degrees(wrap(eyaw[-1] - eyaw[0])))
    yr = np.abs(wrap(np.diff(eyaw))) / DT
    f["max_yaw_rate_dps"] = float(np.degrees(yr.max()))
    fwd = vx[ego, seg] * np.cos(eyaw) + vy[ego, seg] * np.sin(eyaw)
    f["reverse"] = bool((fwd < -0.5).any())

    others = np.array([i for i in range(x.shape[0]) if i != ego])
    v0 = others[valid[others, t0]]
    dist0 = np.hypot(x[v0, t0] - x[ego, t0], y[v0, t0] - y[ego, t0])
    for name, tid in (("veh", VEH), ("ped", PED), ("cyc", CYC)):
        f[f"n_{name}_30m"] = int(((types[v0] == tid) & (dist0 < 30)).sum())
    f["n_agents_valid"] = int(valid[:, t0].sum())

    # closest same-time approach of other agents to ego, and spatial path conflicts
    min_gap = {VEH: np.inf, PED: np.inf, CYC: np.inf}
    close_ids = set()
    for t in seg:
        m = others[valid[others, t]]
        if len(m) == 0:
            continue
        dd = np.hypot(x[m, t] - x[ego, t], y[m, t] - y[ego, t])
        for tid in min_gap:
            sel = types[m] == tid
            if sel.any():
                min_gap[tid] = min(min_gap[tid], float(dd[sel].min()))
        close_ids.update(m[dd < 6].tolist())
    f["min_dist_veh"] = min_gap[VEH]
    f["min_dist_vru"] = min(min_gap[PED], min_gap[CYC])
    f["n_close_agents_6m"] = len(close_ids)

    # agents whose path (any time after t0) passes within 2m of the ego's future path but not the same time
    ego_path = np.stack([ex, ey], -1)[::2]
    n_conflict = 0
    for i in others:
        vi = np.flatnonzero(valid[i, t0:]) + t0
        if len(vi) < 2 or types[i] not in (VEH, PED, CYC):
            continue
        p = np.stack([x[i, vi], y[i, vi]], -1)[::2]
        if np.hypot(p[:, 0].mean() - ex.mean(), p[:, 1].mean() - ey.mean()) > 80:
            continue
        dmin = np.sqrt(((p[:, None] - ego_path[None]) ** 2).sum(-1)).min()
        # exclude agents that just follow/lead along the same lane: require heading difference
        hd = np.abs(wrap(np.median(d["yaw"][i, vi]) - np.median(eyaw)))
        if dmin < 2.0 and hd > np.radians(30):
            n_conflict += 1
    f["n_crossing_conflicts"] = n_conflict

    tl_xy = np.stack([d["tl_x"][:, t0], d["tl_y"][:, t0]], -1)
    tl_ok = d["tl_valid"][:, t0]
    tl_d = np.hypot(tl_xy[:, 0] - x[ego, t0], tl_xy[:, 1] - y[ego, t0])
    f["tl_near"] = bool((tl_ok & (tl_d < 40)).any())

    m = traj_metrics(x, y, yaw, L, W, valid, types, ego, t0, goal_t, rg)
    f.update({f"log_{k}": v for k, v in m.items()})
    f["no_road_edges"] = rg is None
    return f


def rg_from(c, i=None):
    g = (lambda k: c[k][i]) if i is not None else (lambda k: c[k])
    types, ok = g("roadgraph_points_types"), g("roadgraph_points_valid").astype(bool)
    mask = ok & np.isin(types, ROAD_EDGE_TYPES)
    if not mask.any():
        return None
    pts = np.stack([g("roadgraph_points_x")[mask], g("roadgraph_points_y")[mask]], -1)
    dirs = np.stack([g("roadgraph_points_dir_x")[mask], g("roadgraph_points_dir_y")[mask]], -1)
    return pts, dirs, g("roadgraph_points_ids")[mask]


def traj_from(c, i=None):
    g = (lambda k: c[k][i]) if i is not None else (lambda k: c[k])
    return {
        "x": g("log_trajectory_x"), "y": g("log_trajectory_y"), "yaw": g("log_trajectory_yaw"),
        "vx": g("log_trajectory_vel_x"), "vy": g("log_trajectory_vel_y"),
        "valid": g("log_trajectory_valid").astype(bool),
        "L": g("log_trajectory_length"), "W": g("log_trajectory_width"),
        "types": g("object_metadata_object_types"),
        "is_sdc": g("object_metadata_is_sdc").astype(bool),
        "tl_x": g("log_traffic_light_x"), "tl_y": g("log_traffic_light_y"),
        "tl_valid": g("log_traffic_light_valid").astype(bool),
    }


def rollout_features(path, log_d, ego, goal_t, rg):
    """Features of FM failing rollout (ego track replaced by planner)."""
    if not os.path.exists(path):
        return {}
    with np.load(path) as c:
        d = traj_from(c)
    x, y, yaw = d["x"], d["y"], d["yaw"]
    m = traj_metrics(x, y, yaw, d["L"], d["W"], d["valid"], d["types"], ego, START_T, 90, rg)
    f = {f"fm_{k}": v for k, v in m.items()}
    fail_ts = [t for t in (m["offroad_t"], m["overlap_t"]) if t >= 0]
    ft = min(fail_ts) if fail_ts else -1
    f["fm_fail_t"] = ft
    lx, ly = log_d["x"][ego], log_d["y"][ego]
    if ft >= 0:
        f["fm_dev_at_fail_m"] = float(np.hypot(x[ego, ft] - lx[ft], y[ego, ft] - ly[ft]))
        # progress along log vs. log at same time: positive = FM ahead
        h = log_d["yaw"][ego, ft]
        f["fm_long_offset_m"] = float((x[ego, ft] - lx[ft]) * np.cos(h) + (y[ego, ft] - ly[ft]) * np.sin(h))
        f["fm_lat_offset_m"] = float(-(x[ego, ft] - lx[ft]) * np.sin(h) + (y[ego, ft] - ly[ft]) * np.cos(h))
    if m["overlap"]:
        j, t = m["overlap_obj"], m["overlap_t"]
        rel = np.array([x[j, t] - x[ego, t], y[j, t] - y[ego, t]])
        h = yaw[ego, t]
        lon = rel[0] * np.cos(h) + rel[1] * np.sin(h)
        lat = -rel[0] * np.sin(h) + rel[1] * np.cos(h)
        ang = np.degrees(np.arctan2(lat, lon))
        f["fm_coll_side"] = "front" if abs(ang) < 45 else ("rear" if abs(ang) > 135 else "side")
        ego_v = np.hypot(np.diff(x[ego, t - 2:t + 1]), np.diff(y[ego, t - 2:t + 1])).mean() / DT
        f["fm_ego_speed_at_coll"] = float(ego_v)
        log_v = np.hypot(log_d["vx"][ego, t], log_d["vy"][ego, t])
        f["log_ego_speed_at_coll_t"] = float(log_v)
    # time-averaged progress ratio over the first 3 s
    t2 = min(START_T + 30, 90)
    fm_prog = np.hypot(np.diff(x[ego, START_T:t2 + 1]), np.diff(y[ego, START_T:t2 + 1])).sum()
    log_prog = np.hypot(np.diff(lx[START_T:t2 + 1]), np.diff(ly[START_T:t2 + 1])).sum()
    f["fm_prog_ratio_3s"] = float(fm_prog / max(log_prog, 1e-3)) if log_prog > 1.0 else np.nan
    return f


def process_tfrecord(args):
    tfname, items = args
    rows = []
    path = SRC_CACHE / f"{tfname}.sim_state_cache.npz"
    with np.load(path) as c:
        idx_list = list(c["scenario_indices"])
        arrays = {k: c[k] for k in c.files if k.startswith(("log_", "object_", "roadgraph_"))}
    for scen_idx, group, meta in items:
        try:
            i = idx_list.index(scen_idx)
            d = traj_from(arrays, i)
            rg = rg_from(arrays, i)
            ego = int(np.argmax(d["is_sdc"]))
            f = {"tfrecord": tfname, "scenario_idx": scen_idx, "group": group, **meta}
            f.update(log_features(d, ego, rg))
            if group == "fail1500":
                fp = SRC_EXP / "sim_state_cache" / f"{tfname}.scenario_{scen_idx:03d}.sim_state_cache.npz"
                f.update(rollout_features(str(fp), d, ego, f["goal_t"], rg))
            rows.append(f)
        except Exception as e:  # noqa: BLE001
            rows.append({"tfrecord": tfname, "scenario_idx": scen_idx, "group": group, "error": repr(e)})
    return rows


def load_run(run_dir):
    res = {}
    for p in glob.glob(f"{run_dir}/*/*.scenario_*.json"):
        r = json.load(open(p))
        res[(os.path.basename(r["tfrecord"]), int(r["scenario_idx"]))] = r
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.load(open(MANIFEST))
    runs = {k: load_run(v) for k, v in RUNS.items()}
    jobs = defaultdict(list)
    for s in man["scenarios"]:
        key = (s["tfrecord"], int(s["scenario_idx"]))
        meta = {"fm_fail_type": s["fail_type"]}
        for rk, rr in runs.items():
            r = rr.get(key)
            if r is not None:
                meta[f"{rk}_success"] = r["success"]
                meta[f"{rk}_offroad"] = r["offroad"]
                meta[f"{rk}_overlap"] = r["overlap"]
                meta[f"{rk}_goal"] = r["goal_reached"]
        jobs[s["tfrecord"]].append((int(s["scenario_idx"]), "fail1500", meta))

    rng = random.Random(0)
    for tfname in list(jobs.keys()):
        succ = []
        for p in glob.glob(str(SRC_EXP / f"{tfname}.scenario_*.json")):
            r = json.load(open(p))
            if r.get("success"):
                succ.append(int(r["scenario_idx"]))
        for si in rng.sample(succ, min(CONTROL_PER_TFRECORD, len(succ))):
            jobs[tfname].append((si, "control_fm_success", {}))

    with Pool(10) as pool:
        rows = [r for rs in pool.imap_unordered(process_tfrecord, list(jobs.items())) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fail1500_features.csv", index=False)
    print(df.groupby("group").size())
    if "error" in df:
        print("errors:", df["error"].notna().sum())


if __name__ == "__main__":
    main()
