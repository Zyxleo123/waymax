"""Gate: es_WM's closed-loop IDM evaluator (simulation/idm_closed_loop.py) really closes the loop.

The evaluation-only ``--sim_agents idm`` of 0a45e5e was a silent no-op (IDM sat in a config
field PlanningAgentEnvironment never reads), so this checks the closed loop directly, with a
planner that needs no model: it re-plans the LOGGED ego future at every replan and records the
state it was shown. On real scenes:

  L1  ego: the returned trajectory follows the plans exactly (== its log, max |d| < 1e-3 m)
  L2  NPCs move under IDM: some non-ego agent leaves its logged path (max |d| > 0.5 m)
  L3  history: every agent equals its log through the first replan step (reset copies it)
  L4  the planner is shown the SIMULATED NPCs: at each replan t its observed NPC poses equal
      the returned closed-loop poses at t, and differ from the log for some agent at t >= 20
Reported: overlap / offroad counts of the returned rollout (the runner's metric path).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from glob import glob
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from waymax import config as waymax_config  # noqa: E402

from data.scenario_loader import load_scenario_state_batch_fast  # noqa: E402
from planner.abstract_planner import PlannerResult  # noqa: E402
from simulation.evaluation_utils import rollout_predicted_trajectories_with_metrics  # noqa: E402
from simulation.idm_closed_loop import predict_planner_trajectories_closed_loop_idm  # noqa: E402


class LogEgoPlanner:
    """Plans the logged ego future from the replan step; records what it was shown."""

    def __init__(self, log_traj_bt5: np.ndarray, is_sdc: np.ndarray):
        self.log = log_traj_bt5            # [B, T, 5] x, y, yaw, vx, vy of the ORIGINAL log
        self.is_sdc = is_sdc
        self.seen = {}                     # t -> (x [B, N], y [B, N], valid [B, N]) shown at t

    def plan_trajectory(self, sim_state, goal, *, rng, timestep: int, **kwargs):
        log = sim_state.log_trajectory
        t = int(timestep)
        self.seen[t] = (np.asarray(log.x)[..., t].copy(), np.asarray(log.y)[..., t].copy(),
                        np.asarray(log.valid)[..., t].astype(bool).copy())
        plan = jnp.asarray(self.log[:, t:], jnp.float32)
        B = plan.shape[0]
        return PlannerResult(start_t_b=jnp.full((B,), t, jnp.int32), trajectory_norm_btd=plan,
                             trajectory_world_bt5=plan, world_t_seconds_bt=jnp.zeros(plan.shape[:2]),
                             world_t_valid_bt=jnp.ones(plan.shape[:2], bool),
                             aux={"world_dt_seconds": jnp.full((B,), 0.1, jnp.float32)})


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tfrecord", default=sorted(glob(
        "/data/user_data/yixiz/autonlab/zfsauton/datasets/womd/tf_example/validation/validation_tfexample.tfrecord-00000-of-00150"))[0])
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--start", type=int, default=0)
    a = p.parse_args()
    ds_cfg = dataclasses.replace(waymax_config.WOD_1_3_1_TRAINING, path=a.tfrecord, max_num_objects=128)
    out = load_scenario_state_batch_fast(ds_cfg, list(range(a.start, a.start + a.n)))
    state = out[0] if isinstance(out, tuple) else out
    log = state.log_trajectory
    lx, ly = np.asarray(log.x), np.asarray(log.y)
    lv = np.asarray(log.valid).astype(bool)
    is_sdc = np.asarray(state.object_metadata.is_sdc).astype(bool)
    sdc = np.argmax(is_sdc, axis=1)
    w = np.arange(len(sdc))
    ego_log = np.stack([np.asarray(f)[w, sdc] for f in (log.x, log.y, log.yaw, log.vel_x, log.vel_y)], -1)
    planner = LogEgoPlanner(ego_log, is_sdc)
    cfg = SimpleNamespace(mask_goal=False, use_subgoal=False)
    goal = jnp.asarray(ego_log[:, -1, :2])
    _, final, _ = predict_planner_trajectories_closed_loop_idm(
        cfg, state, goal, planner, replan_interval_steps=10, instruction_interval_steps=10,
        start_timestep=10, use_subgoal=False, rng_key=jax.random.PRNGKey(0))
    fx, fy = np.asarray(final.log_trajectory.x), np.asarray(final.log_trajectory.y)
    fv = np.asarray(final.log_trajectory.valid).astype(bool)
    both = fv & lv
    d = np.hypot(fx - lx, fy - ly)
    npc = both & ~is_sdc[:, :, None]
    ego = both & is_sdc[:, :, None]
    fails = []
    ego_max = float(d[ego].max())
    npc_max = float(d[npc].max())
    hist_max = float(d[:, :, :11][both[:, :, :11]].max())
    dn = d[npc]
    print(f"[cl] ego max |closed-loop - log| {ego_max:.5f} m; history (t<=10) max {hist_max:.5f} m", flush=True)
    print(f"[cl] npc |closed-loop - log| p50 {np.percentile(dn, 50):.2f} p90 {np.percentile(dn, 90):.2f} "
          f"p99 {np.percentile(dn, 99):.2f} max {npc_max:.1f} m over {dn.size} agent-steps", flush=True)
    print(f"[cl] valid agent-steps: log {int(lv.sum())}, closed loop {int(fv.sum())}, both {int(both.sum())}", flush=True)
    if not ego_max < 1e-3:
        fails.append("L1 ego deviated from its plans")
    if not npc_max > 0.5:
        fails.append("L2 NPCs did not move under IDM")
    if not hist_max < 1e-3:
        fails.append("L3 history differs from the log")
    shown_dev, shown_err = 0.0, 0.0
    for t, (sx, sy, sv) in sorted(planner.seen.items()):
        m = sv & fv[..., t] & ~is_sdc
        if m.any():
            shown_err = max(shown_err, float(np.hypot(sx - fx[..., t], sy - fy[..., t])[m].max()))
            if t >= 20:
                shown_dev = max(shown_dev, float(np.hypot(sx - lx[..., t], sy - ly[..., t])[m & lv[..., t]].max()))
    print(f"[cl] replans at {sorted(planner.seen)}; shown NPC vs closed-loop max {shown_err:.5f} m; "
          f"shown NPC vs log (t>=20) max {shown_dev:.2f} m", flush=True)
    if not (shown_err < 1e-3 and shown_dev > 0.5):
        fails.append("L4 the planner was not shown the simulated NPCs")
    ro = rollout_predicted_trajectories_with_metrics(final, rng_key=jax.random.PRNGKey(0))
    ovl = (np.asarray(ro["metric_timeseries"]["overlap"]) > 0).any(0)
    off = (np.asarray(ro["metric_timeseries"]["offroad"]) > 0).any(0)
    print(f"[cl] returned rollout: scenes with overlap {int(ovl.sum())}/{a.n}, offroad {int(off.sum())}/{a.n}", flush=True)
    print("=== " + ("PASS" if not fails else "FAIL: " + "; ".join(fails)) + " ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
