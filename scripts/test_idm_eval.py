"""Gate: the evaluation rollout's ``sim_agents="idm"`` really moves the other agents.

0a45e5e put IDM in ``EnvironmentConfig.sim_agents``, which ``PlanningAgentEnvironment``
never reads, so "idm" silently replayed the log. With the fix the agents are passed
as actors. On real scenes with the logged ego (expert replay):

  I1  log rollout: every non-ego agent's executed xy == its logged xy (max |d| < 1e-3 m)
  I2  idm rollout: non-ego agents deviate from the log somewhere (max |d| > 0.5 m)
  I3  both rollouts: the ego follows its log exactly (it is the expert actor)
Reported: overlap / offroad counts per rollout.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from glob import glob
from pathlib import Path

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from waymax import config as waymax_config

from data.scenario_loader import load_scenario_state_batch_fast
from simulation.evaluation_utils import _build_rollout_env_and_actor, _run_batched_rollout


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tfrecord", default=sorted(glob(
        "/data/user_data/yixiz/autonlab/zfsauton/datasets/womd/tf_example/validation/validation_tfexample.tfrecord-00000-of-00150"))[0])
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--start", type=int, default=0)
    a = p.parse_args()
    ds_cfg = dataclasses.replace(waymax_config.WOD_1_3_1_TRAINING, path=a.tfrecord, max_num_objects=128)
    idx = list(range(a.start, a.start + a.n))
    out = load_scenario_state_batch_fast(ds_cfg, idx)
    state = out[0] if isinstance(out, tuple) else out
    T = int(np.asarray(state.log_trajectory.x).shape[-1])
    is_sdc = np.asarray(state.object_metadata.is_sdc).astype(bool)
    lx, ly = np.asarray(state.log_trajectory.x), np.asarray(state.log_trajectory.y)
    lv = np.asarray(state.log_trajectory.valid).astype(bool)
    fails = []
    res = {}
    for sa in ("log", "idm"):
        env, actor = _build_rollout_env_and_actor(("overlap", "offroad"), int(lx.shape[1]), sa)
        print(f"[idm] {sa}: {len(env._sim_agent_actors)} sim-agent actor(s)", flush=True)
        ro = _run_batched_rollout(state, env, actor, jax.random.PRNGKey(0), T - 1)
        sim = jax.tree_util.tree_map(lambda x: x[-1], ro.state).sim_trajectory  # final state's full trajectory
        sx, sy = np.asarray(sim.x), np.asarray(sim.y)
        sv = np.asarray(sim.valid).astype(bool) & lv
        d = np.hypot(sx - lx, sy - ly)
        npc = sv & ~is_sdc[:, :, None]
        ego = sv & is_sdc[:, :, None]
        if sa == "idm":
            dn = d[npc]
            print(f"[idm] npc |sim-log| quantiles m: p50 {np.percentile(dn,50):.2f} p90 {np.percentile(dn,90):.2f} "
                  f"p99 {np.percentile(dn,99):.2f} max {dn.max():.1f}; agent-steps >20 m: {(dn>20).sum()} of {dn.size}", flush=True)
            bad = np.argwhere(npc & (d > 20))
            agents = sorted({(int(b), int(n)) for b, n, _ in bad})
            print(f"[idm] agents with any step >20 m: {len(agents)}", flush=True)
            for b, n in agents[:6]:
                ts = np.flatnonzero(npc[b, n])
                worst = ts[np.argmax(d[b, n, ts])]
                lval = np.flatnonzero(lv[b, n])
                print(f"[idm]   scene {b} agent {n} type {int(np.asarray(state.object_metadata.object_types)[b, n])} "
                      f"log-valid t[{lval.min()}..{lval.max()}] ({lval.size}) worst t={worst} "
                      f"log ({lx[b,n,worst]:.1f},{ly[b,n,worst]:.1f}) sim ({sx[b,n,worst]:.1f},{sy[b,n,worst]:.1f}) "
                      f"d(t=11..20) {np.round(d[b,n,11:21],1).tolist()}", flush=True)
        res[sa] = dict(npc_max=float(d[npc].max()), ego_max=float(d[ego].max()),
                       overlap=int((np.asarray(ro.metrics["overlap"].value)[:, 10:] > 0).any(0).sum()),
                       offroad=int((np.asarray(ro.metrics["offroad"].value)[:, 10:] > 0).any(0).sum()))
        print(f"[idm] {sa}: npc max |sim-log| {res[sa]['npc_max']:.3f} m, ego {res[sa]['ego_max']:.4f} m, "
              f"scenes with overlap {res[sa]['overlap']}/{a.n}, offroad {res[sa]['offroad']}/{a.n}", flush=True)
    if not res["log"]["npc_max"] < 1e-3:
        fails.append("I1 log rollout moved non-ego agents")
    if not res["idm"]["npc_max"] > 0.5:
        fails.append("I2 idm rollout did not move non-ego agents")
    if not max(res["log"]["ego_max"], res["idm"]["ego_max"]) < 1e-3:
        fails.append("I3 ego deviated from its log")
    print("=== " + ("PASS" if not fails else "FAIL: " + "; ".join(fails)) + " ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
