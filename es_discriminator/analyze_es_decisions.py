"""Label consistency, controllable failures and long-horizon decision quality for
recorded Diffusion-ES runs (es_candidate_recorder output).

Stages (all resumable; `all` runs them in order):

  relabel  Waymax-exact offroad labels for every candidate -> <batch>.labels_v2.npz.
           Query z is the logged ego z at the same timestep (matches Waymax
           OffroadMetric and Scorer.compute_offroad_score_v2, which uses the ego's
           logged z at the replan step). Collision labels are copied from
           labels.npz (SAT overlap with logged agents, as in Waymax OverlapMetric).
  scenes   Per-step Waymax overlap / offroad of the executed ego (exact metric
           functions), the scorer's offroad on the same poses (logged ego z),
           invalid-at-start flags, and a check that the runner's outcome is
           reproduced -> scenes.npz
  report   report.json + report.md: label confusion, controllable failure rates for
           every fail1500 run, analysis of all-unsafe replans, lead times and plan
           consistency.

  python -m es_discriminator.analyze_es_decisions all --data_dir .../es_candidates \
      --tfrecord_dir .../tfrecords/training --out_dir .../decision_analysis
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import re
import time
from collections import Counter, defaultdict

import jax
import jax.numpy as jnp
import numpy as np
from waymax import config as waymax_config
from waymax import datatypes
from waymax.metrics.roadgraph import compute_signed_distance_to_nearest_road_edge_point
from waymax.utils import geometry

from data.scenario_loader import load_scenario_state_batch_fast
from es_discriminator.label_bank import _to_world

START_T = 10
EPISODE_STEPS = 91
QUERY_CHUNK = 8192
SCRATCH = "/zfsauton/scratch/sbellad"


# ---- geometry (Waymax-exact) -------------------------------------------------------

def road_edges(sim_state, w: int) -> datatypes.RoadgraphPoints:
    """Valid road-edge points of world w, padded to a multiple of 1024 (padding invalid).

    Waymax masks non-edge points with inf before the argmin; dropping them keeps the
    nearest point, and the "prior point" check still compares ids within one polyline.
    """
    rg = jax.tree_util.tree_map(lambda a: np.asarray(a)[w], sim_state.roadgraph_points)
    keep = np.asarray(rg.valid).astype(bool) & np.asarray(datatypes.is_road_edge(rg.types))
    idx = np.flatnonzero(keep)
    p = max(1024, int(np.ceil(len(idx) / 1024) * 1024))
    def pad(a, fill=0):
        out = np.full((p,) + a.shape[1:], fill, dtype=a.dtype)
        out[:len(idx)] = a[idx]
        return out
    return datatypes.RoadgraphPoints(
        x=pad(rg.x), y=pad(rg.y), z=pad(rg.z), dir_x=pad(rg.dir_x), dir_y=pad(rg.dir_y),
        dir_z=pad(rg.dir_z), types=pad(rg.types), ids=pad(rg.ids, -1), valid=pad(rg.valid & keep, False),
    )


@jax.jit
def _corner_offroad(corners_q3: jax.Array, rg: datatypes.RoadgraphPoints) -> jax.Array:
    return compute_signed_distance_to_nearest_road_edge_point(corners_q3, rg) > 0.0


def offroad_poses(xy_yaw_lw: np.ndarray, z: np.ndarray, rg) -> np.ndarray:
    """Waymax OffroadMetric for poses [..., 5] = (x, y, yaw, length, width) at heights z [...]."""
    shape = xy_yaw_lw.shape[:-1]
    flat = xy_yaw_lw.reshape(-1, 5).astype(np.float32)
    zf = np.broadcast_to(z, shape).reshape(-1).astype(np.float32)
    boxes = np.stack([flat[:, 0], flat[:, 1], flat[:, 3], flat[:, 4], flat[:, 2]], -1)
    corners = np.asarray(geometry.corners_from_bboxes(jnp.asarray(boxes)))  # [Q,4,2]
    q = np.concatenate([corners, np.repeat(zf[:, None, None], 4, 1)], -1).reshape(-1, 3)
    out = np.zeros((len(q),), bool)
    for s in range(0, len(q), QUERY_CHUNK):
        part = q[s:s + QUERY_CHUNK]
        n = len(part)
        if n < QUERY_CHUNK:
            part = np.pad(part, ((0, QUERY_CHUNK - n), (0, 0)))
        out[s:s + n] = np.asarray(_corner_offroad(jnp.asarray(part), rg))[:n]
    return out.reshape(-1, 4).any(-1).reshape(shape)


@jax.jit
def _ego_overlap_steps(boxes_tn5: jax.Array, valid_tn: jax.Array, ego: jax.Array) -> jax.Array:
    """Waymax OverlapMetric for the ego at every step: any valid other box overlapping it."""
    def one(b, v):
        ov = geometry.compute_pairwise_overlaps(b)  # [N,N]
        return jnp.any(ov[:, ego] & v & (jnp.arange(b.shape[0]) != ego))
    return jax.vmap(one)(boxes_tn5, valid_tn)


def load_batch_state(d: dict, tfrecord_dir: str):
    cfg = dataclasses.replace(waymax_config.WOD_1_3_1_TRAINING,
                              path=os.path.join(tfrecord_dir, str(d["scen_tfrecord"][0])),
                              max_num_objects=128, batch_dims=(len(d["scen_index"]),), shuffle_seed=0)
    st, _, ids = load_scenario_state_batch_fast(cfg, [int(i) for i in d["scen_index"]], return_scenario_ids=True)
    ids = [s.decode() if isinstance(s, bytes) else str(s) for s in ids]
    if ids != [str(s) for s in d["scen_id"]]:
        raise RuntimeError(f"scenario id mismatch: {ids} vs {list(d['scen_id'])}")
    return st


def batch_files(data_dir: str) -> list[str]:
    return sorted(f for f in glob.glob(os.path.join(data_dir, "*.npz")) if re.search(r"\.scen_[\d_]+\.npz$", f))


def first_step(flags_kt: np.ndarray) -> np.ndarray:
    hit = flags_kt.any(axis=1)
    return np.where(hit, np.argmax(flags_kt, axis=1) + 1, 0).astype(np.int16)


# ---- stage 1: relabel offroad ------------------------------------------------------

def relabel_file(path: str, tfrecord_dir: str) -> None:
    out = path[:-4] + ".labels_v2.npz"
    if os.path.exists(out):
        return
    d = dict(np.load(path))
    lab = dict(np.load(path[:-4] + ".labels.npz"))
    st = load_batch_state(d, tfrecord_dir)
    log_z = np.asarray(st.log_trajectory.z)
    is_sdc = np.asarray(st.object_metadata.is_sdc).astype(bool)
    T = d["cand_traj"].shape[1]
    n = len(d["cand_id"])
    off_step = np.zeros((n,), np.int16)
    anchor_off = np.zeros((len(d["group_t"]),), bool)
    edges = {}
    for g in range(len(d["group_t"])):
        rows = np.flatnonzero(d["cand_group"] == g)
        w, t = int(d["group_scen"][g]), int(d["group_t"][g])
        if w not in edges:
            edges[w] = road_edges(st, w)
        e = int(np.argmax(is_sdc[w]))
        h = int(lab["label_horizon"][g])
        o, hd = d["group_origin"][g], float(d["group_heading"][g])
        L, W = d["group_ego_state"][g, 5], d["group_ego_state"][g, 6]
        world = _to_world(np.nan_to_num(d["cand_traj"][rows][:, :h].astype(np.float64)), o, hd)
        poses = np.concatenate([world, np.broadcast_to([L, W], world.shape[:2] + (2,))], -1)
        z = log_z[w, e, np.minimum(np.arange(t + 1, t + 1 + h), EPISODE_STEPS - 1)]
        flags = np.zeros((len(rows), T), bool)
        flags[:, :h] = offroad_poses(poses, z[None, :], edges[w])
        off_step[rows] = first_step(flags)
        now = np.array([o[0], o[1], hd, L, W])
        anchor_off[g] = bool(offroad_poses(now[None], np.array([log_z[w, e, t]]), edges[w])[0])
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, gt_offroad_step=off_step, anchor_offroad=anchor_off,
                        gt_collision_step=lab["gt_collision_step"], anchor_collision=lab["anchor_collision"],
                        label_horizon=lab["label_horizon"], label_version=np.asarray(2))
    os.replace(tmp, out)


# ---- stage 2: scene-level Waymax metrics ---------------------------------------------

def scene_file(path: str, tfrecord_dir: str) -> dict[str, np.ndarray]:
    d = dict(np.load(path))
    st = load_batch_state(d, tfrecord_dir)
    log = st.log_trajectory
    lx, ly, lyaw = np.asarray(log.x), np.asarray(log.y), np.asarray(log.yaw)
    llen, lwid, lz = np.asarray(log.length), np.asarray(log.width), np.asarray(log.z)
    lvalid = np.asarray(log.valid).astype(bool)
    is_sdc = np.asarray(st.object_metadata.is_sdc).astype(bool)
    S = len(d["scen_index"])
    out = defaultdict(list)
    for w in range(S):
        e = int(np.argmax(is_sdc[w]))
        ex = d["scen_executed_world"][w]  # [91,5]
        x, y, yaw = lx[w].copy(), ly[w].copy(), lyaw[w].copy()
        x[e], y[e], yaw[e] = ex[:, 0], ex[:, 1], ex[:, 2]
        boxes = np.stack([x, y, llen[w], lwid[w], yaw], -1).transpose(1, 0, 2)  # [T,N,5]
        overlap = np.asarray(_ego_overlap_steps(jnp.asarray(boxes, jnp.float32),
                                                jnp.asarray(lvalid[w].T), jnp.asarray(e)))
        pose = np.stack([ex[:, 0], ex[:, 1], ex[:, 2], llen[w, e], lwid[w, e]], -1)
        rg = road_edges(st, w)
        off_waymax = offroad_poses(pose, lz[w, e], rg)
        off_scorer = offroad_poses(pose, lz[w, e], rg)  # scorer uses ego logged z (same as Waymax)
        reached = bool(d["scen_goal_reached"][w])
        reached_step = int(d["scen_reached_step"][w])  # absolute timestep
        # Runner window: Waymax rollout metrics start at timestep init_steps - 1 = START_T, and
        # the runner slices [START_T:] again and keeps reached_step + 1 entries, so it checks
        # timesteps 2*START_T .. 2*START_T + reached_step (clamped to the episode).
        n_ep = reached_step + 1 if reached else EPISODE_STEPS - START_T
        runner_win = slice(2 * START_T, min(2 * START_T + n_ep, EPISODE_STEPS))
        end = reached_step + 1 if reached else EPISODE_STEPS  # corrected: START_T .. goal step
        out["scen_tfrecord"].append(str(d["scen_tfrecord"][w]))
        out["scen_index"].append(int(d["scen_index"][w]))
        out["overlap_t"].append(overlap)
        out["offroad_t"].append(off_waymax)
        out["offroad_scorer_t"].append(off_scorer)
        out["episode_end"].append(end)
        out["re_overlap"].append(bool(overlap[runner_win].any()))
        out["re_offroad"].append(bool(off_waymax[runner_win].any()))
        out["corrected_overlap"].append(bool(overlap[START_T:end].any()))
        out["corrected_offroad"].append(bool(off_waymax[START_T:end].any()))
        out["run_overlap"].append(bool(d["scen_overlap"][w]))
        out["run_offroad"].append(bool(d["scen_offroad"][w]))
        out["run_success"].append(bool(d["scen_success"][w]))
        out["goal_reached"].append(reached)
    return {k: np.asarray(v) for k, v in out.items()}


# ---- stage 3: report ---------------------------------------------------------------

def _rate(x) -> float:
    x = np.asarray(x)
    return float(x.mean()) if x.size else float("nan")


def label_consistency(scenes: dict) -> dict:
    ov, off, offs = scenes["overlap_t"], scenes["offroad_t"], scenes["offroad_scorer_t"]
    steps = np.zeros_like(off, dtype=bool)
    for i, end in enumerate(scenes["episode_end"]):
        steps[i, START_T:end] = True
    a, b = off[steps], offs[steps]
    at = off[:, START_T], offs[:, START_T]
    scene_w = np.array([o[START_T:e].any() for o, e in zip(off, scenes["episode_end"])])
    scene_s = np.array([o[START_T:e].any() for o, e in zip(offs, scenes["episode_end"])])
    corr_success = scenes["goal_reached"] & ~scenes["corrected_overlap"] & ~scenes["corrected_offroad"]
    return {
        "runner_outcome_reproduced": {
            "overlap": _rate(scenes["re_overlap"] == scenes["run_overlap"]),
            "offroad": _rate(scenes["re_offroad"] == scenes["run_offroad"]),
        },
        "recorded_run_success": {
            "runner_window": _rate(scenes["run_success"]),
            "corrected_window_t10_to_goal": _rate(corr_success),
            "runner_success_but_corrected_failure": int((scenes["run_success"] & ~corr_success).sum()),
            "runner_failure_but_corrected_success": int((~scenes["run_success"] & corr_success).sum()),
        },
        "executed_steps": int(a.size),
        "per_step_offroad": {"waymax": int(a.sum()), "scorer": int(b.sum()),
                             "both": int((a & b).sum()), "scorer_only": int((b & ~a).sum()),
                             "waymax_only": int((a & ~b).sum())},
        "at_t10_offroad": {"waymax": int(at[0].sum()), "scorer": int(at[1].sum()),
                           "scorer_only": int((at[1] & ~at[0]).sum()), "waymax_only": int((at[0] & ~at[1]).sum())},
        "per_scene_offroad": {"waymax": int(scene_w.sum()), "scorer": int(scene_s.sum()),
                              "scorer_only": int((scene_s & ~scene_w).sum()),
                              "waymax_only": int((scene_w & ~scene_s).sum())},
    }


def run_dirs() -> dict[str, str]:
    pats = ["world_model_sim/*fail1500*/simulation_*", "Diffusion_ES_sim/*fail1500*/simulation_*",
            "Flow_matching_sim/*fail1500*/simulation_*", "diffusion_sim/*fail1500*/simulation_*",
            "es_discriminator/es_candidate_data/fail1500*"]
    out = {}
    for p in pats:
        for d in sorted(glob.glob(os.path.join(SCRATCH, p))):
            name = d.split("/")[-2] if "simulation_" in d.split("/")[-1] else d.split("/")[-1] + " (recorded)"
            out[name] = d
    return out


def controllable_table(invalid: dict[tuple[str, int], str]) -> list[dict]:
    rx = re.compile(r"^(?P<tf>.+)\.scenario_(?P<i>\d+)\.json$")
    rows = []
    for name, d in run_dirs().items():
        res = {}
        for f in os.listdir(d):
            m = rx.match(f)
            if m:
                res[(m["tf"], int(m["i"]))] = json.load(open(os.path.join(d, f)))
        if len(res) < 1400:
            continue
        valid = [r for k, r in res.items() if k not in invalid]
        def pct(rs, key):
            return round(100 * _rate([bool(r[key]) for r in rs]), 1)
        rows.append({
            "run": name, "n": len(res), "n_controllable": len(valid),
            "success_all": pct(res.values(), "success"),
            "success_controllable": pct(valid, "success"),
            "safe_controllable": round(100 * _rate([not (r["overlap"] or r["offroad"]) for r in valid]), 1),
            "overlap_controllable": pct(valid, "overlap"),
            "offroad_controllable": pct(valid, "offroad"),
            "goal_controllable": pct(valid, "goal_reached"),
        })
    return rows


def decision_analysis(data_dir: str, scenes: dict, invalid: set) -> dict:
    """Per failed controllable scene: first bad replan (v2 labels) and what preceded it."""
    exec_fail_t = {}
    for i in range(len(scenes["scen_index"])):
        key = (scenes["scen_tfrecord"][i], int(scenes["scen_index"][i]))
        end = scenes["episode_end"][i]
        bad = scenes["overlap_t"][i][START_T:end] | scenes["offroad_t"][i][START_T:end]
        exec_fail_t[key] = START_T + int(np.argmax(bad)) if bad.any() else None

    first_bad = Counter()
    all_unsafe_cases = Counter()
    lead_times = []
    plan_shift = {"success": [], "failure": [], "before_point_of_no_return": []}
    pick_unsafe_mixed = []
    examples = defaultdict(list)
    n_fail = n_succ = 0
    for f in batch_files(data_dir):
        d = dict(np.load(f))
        lab = dict(np.load(f[:-4] + ".labels_v2.npz"))
        unsafe = (lab["gt_collision_step"] > 0) | (lab["gt_offroad_step"] > 0)
        ttv = np.where(unsafe, np.minimum(*[np.where(lab[k] > 0, lab[k], 999)
                                            for k in ("gt_collision_step", "gt_offroad_step")]), 0)
        flagged = lab["anchor_collision"] | lab["anchor_offroad"]
        g_of = d["cand_group"]
        for w in range(len(d["scen_index"])):
            key = (str(d["scen_tfrecord"][w]), int(d["scen_index"][w]))
            if key in invalid:
                continue
            success = bool(d["scen_success"][w])
            gs = np.flatnonzero(d["group_scen"] == w)
            gs = gs[np.argsort(d["group_t"][gs])]
            kinds, sel_rows, info = [], [], []
            for g in gs:
                rows = np.flatnonzero(g_of == g)
                sel = rows[d["cand_selected"][rows]][0]
                u = unsafe[rows]
                if flagged[g]:
                    k = "already_flagged"
                elif u.all():
                    k = "all_unsafe"
                elif not u.any():
                    k = "all_safe"
                else:
                    k = "mixed_ES_unsafe" if unsafe[sel] else "mixed_ES_safe"
                    pick_unsafe_mixed.append(bool(unsafe[sel]))
                kinds.append(k)
                sel_rows.append(sel)
                info.append(dict(g=int(g), t=int(d["group_t"][g]), h=int(lab["label_horizon"][g]),
                                 best_ttv=int(ttv[rows].max()) if u.all() else 0,
                                 n_safe=int((~u).sum()), sel_ttv=int(ttv[sel])))
            # Plan consistency: previous selected plan vs new selected plan over their overlap.
            shifts = []
            for j in range(1, len(gs)):
                a, b = sel_rows[j - 1], sel_rows[j]
                ga, gb = gs[j - 1], gs[j]
                pa = _to_world(d["cand_traj"][a].astype(np.float64), d["group_origin"][ga], float(d["group_heading"][ga]))
                pb = _to_world(d["cand_traj"][b].astype(np.float64), d["group_origin"][gb], float(d["group_heading"][gb]))
                dt = int(d["group_t"][gb] - d["group_t"][ga])
                m = min(len(pa) - dt, info[j]["h"], info[j - 1]["h"] - dt)
                if m > 0:
                    shifts.append(float(np.linalg.norm(pa[dt:dt + m, :2] - pb[:m, :2], axis=-1).mean()))
            if success:
                n_succ += 1
                plan_shift["success"].extend(shifts)
                continue
            n_fail += 1
            plan_shift["failure"].extend(shifts)
            bad_idx = [i for i, k in enumerate(kinds) if k in ("already_flagged", "all_unsafe", "mixed_ES_unsafe")]
            if not bad_idx:
                first_bad["none (goal not reached / beyond label horizon)"] += 1
                continue
            k0 = bad_idx[0]
            first_bad[kinds[k0]] += 1
            if k0 > 0:
                plan_shift["before_point_of_no_return"].append(shifts[k0 - 1] if k0 - 1 < len(shifts) else np.nan)
            ft = exec_fail_t.get(key)
            if ft is not None:
                lead_times.append((ft - info[k0]["t"]) / 10.0)
            if kinds[k0] == "all_unsafe":
                cur = info[k0]
                if k0 == 0:
                    cat = "at first replan (t=10): no earlier choice"
                else:
                    prev = info[k0 - 1]
                    dt = cur["t"] - prev["t"]
                    covered = prev["h"] - dt  # steps after cur["t"] that the previous plan was checked safe
                    if prev["sel_ttv"] == 0 and cur["best_ttv"] <= covered:
                        cat = "previous plan still safe beyond every new candidate's violation (pool missed it)"
                    elif prev["sel_ttv"] == 0:
                        cat = "hazard appears after the previous plan's horizon (myopic horizon)"
                    else:
                        cat = "previous pick already unsafe beyond the 5 s label horizon"
                all_unsafe_cases[cat] += 1
                if len(examples[cat]) < 5:
                    examples[cat].append({"tfrecord": key[0], "scenario": key[1], "replan_t": cur["t"],
                                          "best_ttv_s": cur["best_ttv"] / 10.0,
                                          "prev_safe_candidates": info[k0 - 1]["n_safe"] if k0 else None})

    def summ(v):
        v = np.asarray([x for x in v if np.isfinite(x)])
        return {"n": int(v.size), "median": float(np.median(v)) if v.size else None,
                "p25": float(np.percentile(v, 25)) if v.size else None,
                "p75": float(np.percentile(v, 75)) if v.size else None}
    lt = np.asarray(lead_times)
    return {
        "controllable_failed_scenes": n_fail, "controllable_successful_scenes": n_succ,
        "first_bad_replan": dict(first_bad),
        "all_unsafe_first": dict(all_unsafe_cases),
        "all_unsafe_examples": dict(examples),
        "mixed_replans_ES_pick_unsafe_rate": _rate(pick_unsafe_mixed),
        "lead_time_s_first_bad_to_executed_failure": {
            **summ(lt), "within_1s": _rate(lt <= 1.0) if lt.size else None,
            "over_3s": _rate(lt > 3.0) if lt.size else None},
        "plan_shift_m": {k: summ(v) for k, v in plan_shift.items()},
    }


def write_report(out_dir: str, rep: dict) -> None:
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=str)
    lc = rep["label_consistency"]
    L = ["# Diffusion-ES decision analysis (fail1500)", "",
         "## 1. Offroad label vs Waymax metric", "",
         f"Runner outcome reproduced from per-step Waymax metrics: overlap {lc['runner_outcome_reproduced']['overlap']:.3f}, "
         f"offroad {lc['runner_outcome_reproduced']['offroad']:.3f}", "",
         "Recorded run success, runner window (t = 20 .. goal + 10) vs corrected window (t = 10 .. goal): "
         + json.dumps(lc["recorded_run_success"]), "",
         "| | Waymax offroad | scorer offroad | scorer only | Waymax only |", "|---|---|---|---|---|"]
    for name, key in (("executed steps", "per_step_offroad"), ("t = 10 (logged pose)", "at_t10_offroad"),
                      ("scenes (any step)", "per_scene_offroad")):
        r = lc[key]
        L.append(f"| {name} | {r['waymax']} | {r['scorer']} | {r['scorer_only']} | {r['waymax_only']} |")
    inv = rep["invalid_at_start"]
    L += ["", "## 2. Scenes invalid before ES takes control (Waymax overlap/offroad at t = 10)", "",
          f"{inv['n']} of 1500 ({inv['by_type']})", "",
          "| run | n | controllable | success (all) | success (controllable) | safe (controllable) | overlap | offroad | goal |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rep["controllable_runs"]:
        L.append(f"| {r['run']} | {r['n']} | {r['n_controllable']} | {r['success_all']} | {r['success_controllable']} | "
                 f"{r['safe_controllable']} | {r['overlap_controllable']} | {r['offroad_controllable']} | {r['goal_controllable']} |")
    da = rep["decisions"]
    L += ["", "## 3-4. Failed controllable scenes (recorded run, v2 labels)", "",
          f"Failed {da['controllable_failed_scenes']}, succeeded {da['controllable_successful_scenes']}", "",
          "First bad replan: " + json.dumps(da["first_bad_replan"]), "",
          "First bad replan = all candidates unsafe: " + json.dumps(da["all_unsafe_first"], indent=1), "",
          f"Mixed replans, ES pick unsafe: {da['mixed_replans_ES_pick_unsafe_rate']:.3f}", "",
          "Lead time from first bad replan to executed failure (s): "
          + json.dumps(da["lead_time_s_first_bad_to_executed_failure"]), "",
          "Plan shift between consecutive selected plans (m): " + json.dumps(da["plan_shift_m"])]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["relabel", "scenes", "report", "all"])
    p.add_argument("--data_dir", required=True)
    p.add_argument("--tfrecord_dir", required=True)
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    files = batch_files(a.data_dir)
    scenes_path = os.path.join(a.out_dir, "scenes.npz")

    if a.stage in ("relabel", "all"):
        t0 = time.time()
        for i, f in enumerate(files):
            relabel_file(f, a.tfrecord_dir)
            if (i + 1) % 10 == 0:
                print(f"relabel {i + 1}/{len(files)} {(time.time() - t0) / (i + 1):.1f}s/batch", flush=True)
    if a.stage in ("scenes", "all") and not os.path.exists(scenes_path):
        t0 = time.time()
        parts = []
        for i, f in enumerate(files):
            parts.append(scene_file(f, a.tfrecord_dir))
            if (i + 1) % 10 == 0:
                print(f"scenes {i + 1}/{len(files)} {(time.time() - t0) / (i + 1):.1f}s/batch", flush=True)
        np.savez_compressed(scenes_path, **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
    if a.stage in ("report", "all"):
        scenes = dict(np.load(scenes_path))
        start_bad = scenes["overlap_t"][:, START_T] | scenes["offroad_t"][:, START_T]
        invalid = {}
        for i in np.flatnonzero(start_bad):
            kind = "+".join(k for k, v in (("overlap", scenes["overlap_t"][i, START_T]),
                                           ("offroad", scenes["offroad_t"][i, START_T])) if v)
            invalid[(str(scenes["scen_tfrecord"][i]), int(scenes["scen_index"][i]))] = kind
        rep = {
            "label_consistency": label_consistency(scenes),
            "invalid_at_start": {"n": len(invalid), "by_type": dict(Counter(invalid.values()))},
            "controllable_runs": controllable_table(invalid),
            "decisions": decision_analysis(a.data_dir, scenes, set(invalid)),
        }
        with open(os.path.join(a.out_dir, "invalid_at_start.json"), "w") as f:
            json.dump([{"tfrecord": k[0], "scenario_idx": k[1], "type": v} for k, v in sorted(invalid.items())], f, indent=1)
        write_report(a.out_dir, rep)


if __name__ == "__main__":
    main()
