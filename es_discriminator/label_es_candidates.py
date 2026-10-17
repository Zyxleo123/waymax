"""Offline GT-replay labels and progress metrics for recorded ES candidates.

For every `<batch>.npz` written by `es_candidate_recorder`, reloads the batch's
scenarios from the tfrecords and writes `<batch>.labels.npz` (row-aligned with
the `cand_*` arrays). Already-labelled batches are skipped, so re-running resumes.

Log replay makes these exact open-loop labels: other agents follow the log, so a
candidate's outcome over its horizon does not depend on what the ego did before.

  collision  per-step SAT box overlap with logged agents (same geometry as
             Scorer.compute_collision_score)
  offroad    Scorer.compute_offroad_score_v2 on each (candidate, step) pose

Per candidate:
  gt_collision_step / gt_offroad_step  first future step (1..T, 0.1 s) with a
      violation, 0 = none within the group's label horizon
  gt_path_len_m, gt_progress_m (arc length along the logged ego path reached),
  gt_final_goal_dist_m, gt_goal_gain_m (anchor-to-goal minus end-to-goal),
  gt_ade_m, gt_fde_m (vs logged ego future)
  (all over the label horizon, i.e. the steps that exist before the episode ends)
Per group: label_horizon (valid future steps), anchor_collision, anchor_offroad.

  python -m es_discriminator.label_es_candidates --data_dir .../es_candidates \
      --tfrecord_dir .../es_failure_subset_n1500_seed0/tfrecords/training
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import os
import re
import time

import jax
import jax.numpy as jnp
import numpy as np
from waymax import config as waymax_config

from data.scenario_loader import load_scenario_state_batch_fast
from scores.helpers import forecast_other_agents_log_futures
from scores.polygon import pairwise_intersections, polygon_corners
from scores.scorer_lane_graph import Scorer

from es_discriminator.data import to_ego_frame
from es_discriminator.label_bank import _to_world

EPISODE_STEPS = 91


@jax.jit
def collision_steps(ego_kt5: jax.Array, obj_ot5: jax.Array, mask_ot: jax.Array) -> jax.Array:
    """[K,T] True where candidate k overlaps any valid object at step t."""
    ego_poly = polygon_corners(ego_kt5)
    obj_poly = polygon_corners(obj_ot5)
    hits = jax.vmap(pairwise_intersections, in_axes=(1, 1), out_axes=0)(ego_poly, obj_poly)  # [T,K,O]
    hits = hits & jnp.swapaxes(mask_ot, 0, 1)[:, None, :]
    return jnp.swapaxes(jnp.any(hits, axis=2), 0, 1)


def offroad_steps(scorer: Scorer, ego_kt5: np.ndarray, sim_state, t: int, w: int, chunk: int) -> np.ndarray:
    """[K,T] offroad flag per pose, via compute_offroad_score_v2 on 1-step trajectories."""
    k, steps = ego_kt5.shape[:2]
    flat = ego_kt5.reshape(k * steps, 1, 5)
    out = np.empty((k * steps,), dtype=bool)
    for s in range(0, len(flat), chunk):
        part = jnp.asarray(flat[s:s + chunk], dtype=jnp.float32)
        out[s:s + chunk] = np.asarray(scorer.compute_offroad_score_v2(part, sim_state, t, w)) < 0.5
    return out.reshape(k, steps)


def first_step(flags_kt: np.ndarray) -> np.ndarray:
    """1-based first True step per row, 0 if none."""
    hit = flags_kt.any(axis=1)
    return np.where(hit, np.argmax(flags_kt, axis=1) + 1, 0).astype(np.int16)


def progress_metrics(cand: np.ndarray, expert: np.ndarray, expert_valid: np.ndarray,
                     goal_local: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    """Ego-frame metrics over the first `horizon` future steps (anchor at the origin)."""
    h = max(int(horizon), 1)
    xy = cand[:, :h, :2].astype(np.float64)
    seg = np.diff(np.concatenate([np.zeros_like(xy[:, :1]), xy], axis=1), axis=1)
    path_len = np.linalg.norm(seg, axis=-1).sum(1)
    end = xy[:, -1]

    ev = expert_valid[:h]
    exy = expert[:h, :2].astype(np.float64)
    if ev.any():
        pts = np.concatenate([np.zeros((1, 2)), exy[ev]], axis=0)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=-1))])
        nearest = np.argmin(np.linalg.norm(end[:, None] - pts[None], axis=-1), axis=1)
        progress = arc[nearest]
        d = np.linalg.norm(xy[:, ev] - exy[ev][None], axis=-1)
        ade = d.mean(1)
        last = np.flatnonzero(ev)[-1]
        fde = np.linalg.norm(xy[:, last] - exy[last][None], axis=-1)
    else:
        progress = ade = fde = np.full((len(cand),), np.nan)
    goal_end = np.linalg.norm(end - goal_local[None], axis=-1)
    return {
        "gt_path_len_m": path_len.astype(np.float32),
        "gt_progress_m": progress.astype(np.float32),
        "gt_final_goal_dist_m": goal_end.astype(np.float32),
        "gt_goal_gain_m": (np.linalg.norm(goal_local) - goal_end).astype(np.float32),
        "gt_ade_m": ade.astype(np.float32),
        "gt_fde_m": fde.astype(np.float32),
    }


def label_file(path: str, tfrecord_dir: str, max_num_objects: int, chunk: int) -> str:
    out_path = path[:-len(".npz")] + ".labels.npz"
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    tfrecord = str(d["scen_tfrecord"][0])
    indices = [int(i) for i in d["scen_index"]]
    wcfg = dataclasses.replace(waymax_config.WOD_1_3_1_TRAINING, path=os.path.join(tfrecord_dir, tfrecord),
                               max_num_objects=int(max_num_objects), batch_dims=(len(indices),), shuffle_seed=0)
    st, _, ids = load_scenario_state_batch_fast(wcfg, indices, return_scenario_ids=True)
    ids = [s.decode() if isinstance(s, bytes) else str(s) for s in ids]
    if list(ids) != [str(s) for s in d["scen_id"]]:
        raise RuntimeError(f"{path}: reloaded scenario ids {ids} != recorded {list(d['scen_id'])}")

    scorer = Scorer(metrics=["collision", "offroad"], other_agent_forecast="gt")
    n_groups = len(d["group_t"])
    T = d["cand_traj"].shape[1]
    n = len(d["cand_id"])
    coll_step = np.zeros((n,), np.int16)
    off_step = np.zeros((n,), np.int16)
    prog: dict[str, np.ndarray] = {}
    label_h = np.zeros((n_groups,), np.int16)
    anchor_coll = np.zeros((n_groups,), bool)
    anchor_off = np.zeros((n_groups,), bool)
    for g in range(n_groups):
        rows = np.flatnonzero(d["cand_group"] == g)
        w, t = int(d["group_scen"][g]), int(d["group_t"][g])
        h = min(T, EPISODE_STEPS - 1 - t)
        label_h[g] = h
        origin, heading = d["group_origin"][g], float(d["group_heading"][g])
        length, width = d["group_ego_state"][g, 5], d["group_ego_state"][g, 6]
        world = _to_world(d["cand_traj"][rows].astype(np.float64), origin, heading)  # [K,T,3]
        size = np.broadcast_to(np.array([length, width]), world.shape[:2] + (2,))
        ego = np.concatenate([world, size], axis=-1).astype(np.float32)
        ego = np.nan_to_num(ego)
        obj, mask, _, _ = forecast_other_agents_log_futures(st, t, T + 1, w)  # index 0 = t
        obj = np.asarray(obj)
        mask = np.asarray(mask)
        # Pad the episode tail to a fixed T so the jitted collision check compiles once.
        pad = T + 1 - obj.shape[1]
        if pad > 0:
            obj = np.pad(obj, ((0, 0), (0, pad), (0, 0)))
            mask = np.pad(mask, ((0, 0), (0, pad)))
        steps_ok = np.arange(1, T + 1) <= h
        coll = np.asarray(collision_steps(jnp.asarray(ego), jnp.asarray(obj[:, 1:]), jnp.asarray(mask[:, 1:])))
        coll = coll & steps_ok[None]
        off = np.zeros_like(coll)
        off[:, :h] = offroad_steps(scorer, ego[:, :h], st, t, w, chunk)
        coll_step[rows] = first_step(coll)
        off_step[rows] = first_step(off)

        now = np.concatenate([origin, [heading, length, width]]).astype(np.float32)[None, None]
        anchor_coll[g] = bool(np.asarray(collision_steps(jnp.asarray(now), jnp.asarray(obj[:, :1]),
                                                         jnp.asarray(mask[:, :1])))[0, 0])
        anchor_off[g] = bool(offroad_steps(scorer, now, st, t, w, chunk)[0, 0])

        goal_local = to_ego_frame(d["scen_goal_world"][w], origin, heading)
        m = progress_metrics(d["cand_traj"][rows], d["expert_future"][g], d["expert_future_valid"][g],
                             goal_local, h)
        for key, value in m.items():
            prog.setdefault(key, np.full((n,), np.nan, np.float32))[rows] = value

    tmp = out_path + ".tmp.npz"
    np.savez_compressed(tmp, gt_collision_step=coll_step, gt_offroad_step=off_step, label_horizon=label_h,
                        anchor_collision=anchor_coll, anchor_offroad=anchor_off, **prog)
    os.replace(tmp, out_path)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, help="es_candidates/ folder written by the runner")
    p.add_argument("--tfrecord_dir", required=True, help="Folder holding the batches' tfrecords")
    p.add_argument("--max_num_objects", type=int, default=128)
    p.add_argument("--offroad_chunk", type=int, default=2048, help="poses per offroad call (memory bound)")
    p.add_argument("--shard", type=int, default=0, help="this worker's index (for parallel labelling)")
    p.add_argument("--num_shards", type=int, default=1)
    a = p.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(a.data_dir, "*.npz")) if re.search(r"\.scen_[\d_]+\.npz$", f))
    files = files[a.shard::a.num_shards]
    todo = [f for f in files if not os.path.exists(f[:-len(".npz")] + ".labels.npz")]
    print(f"{len(files)} batch files, {len(files) - len(todo)} already labelled, {len(todo)} to do", flush=True)
    start = time.time()
    for i, f in enumerate(todo):
        label_file(f, a.tfrecord_dir, a.max_num_objects, a.offroad_chunk)
        el = time.time() - start
        print(f"[{i + 1}/{len(todo)}] {os.path.basename(f)}  {el / (i + 1):.1f}s/batch", flush=True)


if __name__ == "__main__":
    main()
