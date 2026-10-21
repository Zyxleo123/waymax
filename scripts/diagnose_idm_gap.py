"""Why the closed-loop IDM view scores so far below log replay: decompose it by replay.

An ego path is replayed through the runner's own evaluation rollout
(``_build_rollout_env_and_actor``) with the non-ego agents driven by ``log``, ``idm``
(waymax stock IDMRoutePolicy, the team's IDM view) or ``idm_brake``
(simulation/sim_agents.py), and scored with the runner's window (t10..goal step).
NPCs are deterministic given the ego's path, so replaying a closed-loop run's executed
ego trajectory under ``idm`` reproduces that run's NPCs exactly.

Ego sources:
  expert      the human's own log. Every overlap it gets under a reactive model is the
              NPC model's doing: the recorded traffic was collision-free around it.
  <arm dir>   an ES run's ``<view>/<arm>`` directory with ``*.trajectory.npz``
              (``predicted_trajectory`` = the executed ego path, t = 0..90).
  delay:D     the human's path run D steps late (the ego waits at its t=10 pose, then
              follows its log): the reactivity probe. Under log replay followers drive
              through the waiting ego; reactive NPCs must brake for it.

For every scored overlap the colliding object is identified (per-object box overlap,
the waymax metric's own test) at the first overlapping step t*, and described against
its own recording: was it still in the log at t* (or kept alive past it), how far
ahead/behind its logged schedule along its own path, lateral offset from that path,
sim vs logged speed, its type, the bearing from the ego, the largest single-step jump.
For a run that misses the goal, the NPC most often straight ahead of the ego over the
last 3 s is described the same way (``blocker``).

One JSON line per (scene, ego source, NPC model); resumable.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from collections import defaultdict
from glob import glob
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")  # TF only parses records; left alone it grabs the whole GPU
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from waymax import config as waymax_config  # noqa: E402
from waymax import dataloader  # noqa: E402
from waymax.dataloader import womd_factories  # noqa: E402
from waymax.utils import geometry  # noqa: E402

from simulation.evaluation_utils import (  # noqa: E402
    _build_rollout_env_and_actor,
    _run_batched_rollout,
    check_goal_reaching,
    episode_failure_flags,
    infer_goals_from_sim_state,
)
from simulation.planning_utils import apply_ego_replacements_to_expanded_state  # noqa: E402

START_T = 10
MIN_PATH_M = 2.0  # shorter logged paths carry no schedule (parked)
VAL_DIR = "/data/user_data/yixiz/autonlab/zfsauton/datasets/womd/tf_example/validation"
OBJ_TYPES = {0: "unset", 1: "vehicle", 2: "pedestrian", 3: "cyclist", 4: "other"}


def shard_path(k: int) -> str:
    return f"{VAL_DIR}/validation_tfexample.tfrecord-{k:05d}-of-00150"


def parse_batch(records, cfg):
    processed = dataloader.preprocess_serialized_womd_data(tf.stack(records, axis=0), cfg)
    return womd_factories.simulator_state_from_womd_dict(processed, include_sdc_paths=cfg.include_sdc_paths)


def make_roll(kind: str, n_obj: int, n_t: int):
    env, actor = _build_rollout_env_and_actor(("overlap", "offroad"), n_obj, kind)

    @jax.jit
    def roll(state):
        # The runner rolls n_t - 1 steps from t=10, i.e. to t=100: waymax clamps the
        # writes past the log's end into column n_t - 1, so the final sim trajectory's
        # last step would hold t=100. Stopping at t = n_t - 1 leaves it intact; the
        # metric rows inside the scoring window (t <= 90) are the same either way.
        ro = _run_batched_rollout(state, env, actor, jax.random.PRNGKey(0), n_t - 1 - START_T)
        sim = jax.tree_util.tree_map(lambda x: x[-1], ro.state).sim_trajectory
        out = {f: getattr(sim, f) for f in ("x", "y", "yaw", "vel_x", "vel_y", "valid", "length", "width")}
        out["overlap"] = ro.metrics["overlap"].value
        out["offroad"] = ro.metrics["offroad"].value
        t = ro.state.timestep
        out["timestep"] = t.reshape(t.shape[0], -1)[:, 0]
        return out

    return roll


@jax.jit
def per_object_overlap(x, y, yaw, length, width, valid, is_sdc):
    """[B, N, T] bool: object n's box overlaps the SDC's box at t (both valid, n != SDC)."""
    boxes = jnp.stack([x, y, length, width, yaw], -1)
    sdc = jnp.argmax(is_sdc, axis=1)
    ego = jnp.take_along_axis(boxes, sdc[:, None, None, None], axis=1)
    ov = geometry.has_overlap(jnp.broadcast_to(ego, boxes.shape), boxes)
    ego_valid = jnp.take_along_axis(valid, sdc[:, None, None], axis=1)
    return ov & valid & ego_valid & ~is_sdc[:, :, None]


def describe(b, j, t, tr, lx, ly, lv, lspeed, types, sdc):
    """An NPC at step t vs its own recording, and as seen from the ego."""
    sx, sy, sv = tr["x"][b, j], tr["y"][b, j], tr["valid"][b, j]
    t_idx = np.flatnonzero(lv[b, j])
    d = {"obj": int(j), "type": OBJ_TYPES.get(int(types[b, j]), str(int(types[b, j]))),
         "log_valid": bool(lv[b, j, t]), "first_log_t": int(t_idx[0]) if t_idx.size else -1,
         "last_log_t": int(t_idx[-1]) if t_idx.size else -1}
    v_sim = float(np.hypot(tr["vel_x"][b, j, t], tr["vel_y"][b, j, t]))
    d["v_sim"] = round(v_sim, 2)
    d["v_log"] = round(float(lspeed[b, j, t]), 2) if lv[b, j, t] else None
    if t_idx.size >= 2:
        path = np.stack([lx[b, j, t_idx], ly[b, j, t_idx]], -1)
        s_path = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=-1))])
        dist = np.hypot(path[:, 0] - sx[t], path[:, 1] - sy[t])
        jn = int(dist.argmin())
        d["lat_m"] = round(float(dist[jn]), 2)
        d["path_len_m"] = round(float(s_path[-1]), 1)
        if lv[b, j, t]:
            pos = int(np.searchsorted(t_idx, t))
            d["sched_m"] = round(float(s_path[jn] - s_path[pos]), 2) if s_path[-1] >= MIN_PATH_M else None
            d["dev_m"] = round(float(np.hypot(sx[t] - lx[b, j, t], sy[t] - ly[b, j, t])), 2)
        else:
            d["sched_m"] = None
            d["dev_m"] = None
            if t > t_idx[-1]:  # kept alive past the end of its log: how far past the last logged pose
                e = t_idx[-1]
                d["past_end_m"] = round(float(np.hypot(sx[t] - lx[b, j, e], sy[t] - ly[b, j, e])), 2)
    # largest single-step displacement while valid, reset .. t
    ts = np.arange(START_T + 1, t + 1)
    ok = sv[ts] & sv[ts - 1]
    jumps = np.hypot(sx[ts] - sx[ts - 1], sy[ts] - sy[ts - 1])[ok]
    d["max_jump_m"] = round(float(jumps.max()), 2) if jumps.size else 0.0
    ex, ey, eyaw = tr["x"][b, sdc, t], tr["y"][b, sdc, t], tr["yaw"][b, sdc, t]
    rx, ry = sx[t] - ex, sy[t] - ey
    c, s = np.cos(-eyaw), np.sin(-eyaw)
    d["bearing_deg"] = round(float(np.degrees(np.arctan2(s * rx + c * ry, c * rx - s * ry))), 1)
    d["range_m"] = round(float(np.hypot(rx, ry)), 2)
    d["ego_v"] = round(float(np.hypot(tr["vel_x"][b, sdc, t], tr["vel_y"][b, sdc, t])), 2)
    return d


def blocker(b, tr, sdc, is_sdc, lo=60, hi=90):
    """The NPC most often within 25 m and +-25 deg straight ahead of the ego over steps lo..hi."""
    counts = defaultdict(int)
    for t in range(lo, hi + 1):
        if not tr["valid"][b, sdc, t]:
            continue
        ex, ey, eyaw = tr["x"][b, sdc, t], tr["y"][b, sdc, t], tr["yaw"][b, sdc, t]
        rx, ry = tr["x"][b, :, t] - ex, tr["y"][b, :, t] - ey
        c, s = np.cos(-eyaw), np.sin(-eyaw)
        fx, fy = c * rx - s * ry, s * rx + c * ry
        ok = tr["valid"][b, :, t] & ~is_sdc[b] & (fx > 0) & (np.hypot(fx, fy) < 25.0) & (np.abs(np.arctan2(fy, fx)) < np.radians(25))
        for j in np.flatnonzero(ok):
            counts[int(j)] += 1
    if not counts:
        return None, 0
    j = max(counts, key=counts.get)
    return j, counts[j]


def stream_batches(path: str, wanted: list[int], batch: int):
    """Yield lists of (index, serialized record) for ``wanted`` in file order, ``batch`` at a time,
    without holding the shard (~800 MB) in memory: the session's cgroup is capped at 8 GB."""
    want = set(wanted)
    buf = []
    for i, rec in enumerate(tf.data.TFRecordDataset([path])):
        if i in want:
            buf.append((i, rec))
            want.discard(i)
            if len(buf) == batch:
                yield buf
                buf = []
        if not want:
            break
    if buf:
        yield buf


def count_records(path: str) -> int:
    return sum(1 for _ in tf.data.TFRecordDataset([path]))


def load_arm_index(arm_dir: str):
    idx = {}
    for p in glob(os.path.join(arm_dir, "**", "*.trajectory.npz"), recursive=True):
        name = os.path.basename(p)
        shard, rest = name.split(".scenario_")
        idx[(shard, int(rest.split(".")[0]))] = p
    return idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ego", default="expert", help="'expert' or an ES '<view>/<arm>' run directory")
    ap.add_argument("--ego_name", default=None, help="label for --ego in the output (default: last 2 path parts)")
    ap.add_argument("--npc", default="log,idm,idm_brake")
    ap.add_argument("--shards", default="0-9")
    ap.add_argument("--max_scenes", type=int, default=0, help="per shard, 0 = all")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    lo, hi = (int(v) for v in a.shards.split("-")) if "-" in a.shards else (int(a.shards), int(a.shards))
    kinds = [k for k in a.npc.split(",") if k]
    delay = int(a.ego.split(":", 1)[1]) if a.ego.startswith("delay:") else None
    builtin = a.ego == "expert" or delay is not None
    ego_name = a.ego_name or (a.ego if builtin else "/".join(Path(a.ego).parts[-2:]))
    arm_idx = None if builtin else load_arm_index(a.ego)
    if arm_idx is not None:
        print(f"[diag] {ego_name}: {len(arm_idx)} saved trajectories", flush=True)

    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["shard"], r["scenario_idx"], r["ego"], r["npc"]))
    fout = open(a.out, "a")
    rolls = {}
    B = int(a.batch)
    t0 = time.time()
    n_rows = 0

    for k in range(lo, hi + 1):
        path = shard_path(k)
        shard = os.path.basename(path)
        cfg = dataclasses.replace(waymax_config.WOD_1_3_1_TRAINING, path=path, max_num_objects=128,
                                  batch_dims=(B,), shuffle_seed=0)
        n_rec = count_records(path)
        n_sc = n_rec if not a.max_scenes else min(a.max_scenes, n_rec)
        todo = [i for i in range(n_sc) if any((shard, i, ego_name, kd) not in done for kd in kinds)]
        if arm_idx is not None:
            todo = [i for i in todo if (shard, i) in arm_idx]
        print(f"[diag] {shard}: {n_rec} records, {len(todo)} scenes to do", flush=True)
        for chunk in stream_batches(path, todo, B):
            ids = [i for i, _ in chunk]
            recs = [r for _, r in chunk]
            n_real = len(ids)
            ids_p = ids + [ids[-1]] * (B - n_real)  # pad to a fixed batch (one compile)
            state = parse_batch(recs + [recs[-1]] * (B - n_real), cfg)
            goal_xy, goal_t, _ = infer_goals_from_sim_state(state)
            log = state.log_trajectory
            lx, ly = np.asarray(log.x), np.asarray(log.y)
            lv = np.asarray(log.valid).astype(bool)
            lspeed = np.hypot(np.asarray(log.vel_x), np.asarray(log.vel_y))
            is_sdc = np.asarray(state.object_metadata.is_sdc).astype(bool)
            types = np.asarray(state.object_metadata.object_types)
            sdc_b = is_sdc.argmax(1)
            if arm_idx is not None:
                pred = np.stack([np.load(arm_idx[(shard, i)])["predicted_trajectory"] for i in ids_p]).astype(np.float32)
                state = apply_ego_replacements_to_expanded_state(state, start_t_b=START_T, traj_bt5=pred[:, START_T:])
            elif delay is not None:
                src = np.clip(np.arange(START_T, lx.shape[2]) - delay, START_T, None)  # log step shown at t
                sdc_rows = [np.stack([np.asarray(f)[w, sdc_b[w]] for f in (log.x, log.y, log.yaw, log.vel_x, log.vel_y)], -1)
                            for w in range(B)]
                pred = np.stack([r[src] for r in sdc_rows]).astype(np.float32)
                pred[:, : min(delay, pred.shape[1]), 3:] = 0.0  # at rest while it waits
                state = apply_ego_replacements_to_expanded_state(state, start_t_b=START_T, traj_bt5=pred)
            n_obj, n_t = lx.shape[1], lx.shape[2]
            geval = check_goal_reaching(state, goal_xy, start_timestep=START_T)
            reached = np.asarray(geval["reached"])
            rstep = np.asarray(geval["reached_step"])

            res, ovl_obj = {}, {}
            for kd in kinds:
                if kd not in rolls:
                    rolls[kd] = make_roll(kd, n_obj, n_t)
                out = jax.device_get(rolls[kd](state))
                res[kd] = out
                ovl_obj[kd] = np.asarray(per_object_overlap(
                    out["x"], out["y"], out["yaw"], out["length"], out["width"],
                    out["valid"].astype(bool), jnp.asarray(is_sdc)))
            for kd in kinds:
                out = res[kd]
                win = dict(start_timestep=START_T, goal_reached_b=reached, goal_reached_step_b=rstep, num_log_steps=n_t)
                overlap = episode_failure_flags(out["overlap"], out["timestep"], **win)
                offroad = episode_failure_flags(out["offroad"], out["timestep"], **win)
                end_b = np.where(reached, rstep, n_t - 1)
                tr = {f: np.asarray(out[f]) for f in ("x", "y", "yaw", "vel_x", "vel_y")}
                tr["valid"] = np.asarray(out["valid"]).astype(bool)
                for b in range(n_real):
                    key = (shard, ids[b], ego_name, kd)
                    if key in done:
                        continue
                    sdc = int(sdc_b[b])
                    row = {"shard": shard, "scenario_idx": int(ids[b]), "ego": ego_name, "npc": kd,
                           "goal_reached": bool(reached[b]), "reached_step": int(rstep[b]),
                           "overlap": bool(overlap[b]), "offroad": bool(offroad[b]),
                           "success": bool(reached[b] & ~overlap[b] & ~offroad[b])}
                    ob = ovl_obj[kd][b]  # [N, T]
                    win_t = np.zeros(n_t, bool)
                    win_t[START_T:end_b[b] + 1] = True
                    hit_t = np.flatnonzero(ob.any(0) & win_t)
                    # the metric and the per-object test must agree on which steps overlap
                    m_rows = np.asarray(out["timestep"])
                    m_t = m_rows[(np.asarray(out["overlap"])[:, b] > 0)]
                    m_t = m_t[(m_t >= START_T) & (m_t <= end_b[b])]
                    row["metric_agrees"] = bool(set(m_t.tolist()) == set(hit_t.tolist()))
                    if hit_t.size:
                        t_star = int(hit_t[0])
                        js = np.flatnonzero(ob[:, t_star])
                        row["first_overlap_t"] = t_star
                        row["n_colliders"] = int(js.size)
                        col = describe(b, int(js[0]), t_star, tr, lx, ly, lv, lspeed, types, sdc)
                        if "log" in ovl_obj:  # same ego path, recorded NPCs: did this object hit it too?
                            col["hits_under_log"] = bool((ovl_obj["log"][b, js[0]] & win_t).any())
                        row["collider"] = col
                    if not reached[b]:
                        j, cnt = blocker(b, tr, sdc, is_sdc)
                        row["ego_v_last3s"] = round(float(np.hypot(tr["vel_x"][b, sdc, 60:], tr["vel_y"][b, sdc, 60:]).mean()), 2)
                        row["final_goal_m"] = round(float(np.asarray(geval["final_goal_distance_m"])[b]), 1)
                        if j is not None:
                            bl = describe(b, j, n_t - 1 if tr["valid"][b, j, n_t - 1] else int(np.flatnonzero(tr["valid"][b, j])[-1]),
                                          tr, lx, ly, lv, lspeed, types, sdc)
                            bl["steps_ahead_of_ego"] = cnt
                            row["blocker"] = bl
                    # scene-level NPC realism: how far did the NPCs stray from their recording
                    npc = ~is_sdc[b]
                    keep = npc[:, None] & lv[b] & tr["valid"][b]
                    keep[:, :START_T + 1] = False
                    dev = np.hypot(tr["x"][b] - lx[b], tr["y"][b] - ly[b])[keep]
                    row["npc_dev_p99"] = round(float(np.percentile(dev, 99)), 2) if dev.size else 0.0
                    row["npc_alive_past_log"] = int((npc[:, None] & tr["valid"][b] & ~lv[b])[:, START_T + 1:].sum())
                    fout.write(json.dumps(row) + "\n")
                    n_rows += 1
            fout.flush()
            print(f"[diag] {shard} scenes {ids[0]}..{ids[-1]}: {n_rows} rows, {time.time() - t0:.0f}s", flush=True)
    fout.close()
    print(f"[diag] done: {n_rows} rows in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
