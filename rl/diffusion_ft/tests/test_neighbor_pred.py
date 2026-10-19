"""Gates for the repair bank's non-oracle neighbour sources (``neighbor_pred``).

WHY. The bank's selector scores each candidate against the other cars' boxes
over the next ``n`` steps. With ``neighbor_pred="log"`` (the default, and every
bank result before 2026-09-25) those come from the recorded future -- under log
replay, exactly where every car WILL be: an oracle no deployed planner has.
``"cv"`` (constant-velocity prediction from each car's state at ``t0``) and
``"none"`` (road edges only) must not read past ``t0`` at all.

WHAT HAS TO HOLD:

* **N1 no peeking** -- scramble every object's future (log AND sim, all steps
  after ``t0``) and the ``cv`` / ``none`` boxes, validity and clearances are
  bit-identical, while the ``log`` oracle's DO change (so the scramble hits what
  the oracle reads, and the gate has teeth).
* **N2 the default is untouched** -- ``candidate_safety`` with no
  ``neighbor_pred`` equals ``neighbor_pred="log"`` bit for bit.
* **N3 cv is a sane predictor** -- one step ahead it is within 0.5 m of the log
  (p90, moving agents); the 2 s error is reported, not gated.
* **N4 none sees no agents; cv sees exactly the agents valid at t0.**

No policy, no env: waymax reset + log replay to ``--t0``. CPU, minutes.
Submitted, not run inline::

    bash rl/diffusion_ft/slurm/submit_bank_honest.sh     (its first link)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from waymax import config as waymax_config
from waymax import datatypes
from waymax import dynamics as waymax_dynamics
from waymax import env as waymax_env

from rl.diffusion_ft.safety_objective import (
    SafetyObjectiveConfig, candidate_safety, neighbor_window, predicted_neighbor_window)
from rl.diffusion_ft.scene_sampler import DEFAULT_INDEX_DIR, SceneSampler

FAILS: list[str] = []
NUMS: dict[str, float] = {}


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""),
          flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def scramble_future(state, t0: int):
    """Every object's log AND sim poses after ``t0`` moved by 50 m."""
    def mv(traj):
        T = traj.x.shape[-1]
        after = (jnp.arange(T) > t0)[None, None, :]
        return traj.replace(x=jnp.where(after, traj.x + 50.0, traj.x),
                            y=jnp.where(after, traj.y - 50.0, traj.y),
                            vel_x=jnp.where(after, traj.vel_x + 7.0, traj.vel_x))
    return state.replace(log_trajectory=mv(state.log_trajectory),
                         sim_trajectory=mv(state.sim_trajectory))


def ego_candidate(state, t0: int, n: int):
    """The ego's own log window as a candidate ``[B, n+1, 5]``, anchor first."""
    log = state.log_trajectory
    e = jnp.argmax(jnp.asarray(state.object_metadata.is_sdc).astype(jnp.int32), 1)
    def row(f):
        return jnp.take_along_axis(f, e[:, None, None], axis=1)[:, 0, t0:t0 + n + 1]
    return jnp.stack([row(log.x), row(log.y), row(log.yaw), row(log.vel_x), row(log.vel_y)], -1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits_json", required=True)
    p.add_argument("--shard_glob",
                   default="/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*")
    p.add_argument("--index_dir", default=DEFAULT_INDEX_DIR)
    p.add_argument("--n_scenes", type=int, default=8)
    p.add_argument("--t0", type=int, default=40)
    p.add_argument("--n_steps", type=int, default=20)
    p.add_argument("--verdict", default="")
    args = p.parse_args(argv)

    spec = json.loads(Path(args.splits_json).read_text())
    ids = [int(i) for i in spec["splits"]["failure_heldout"]["scene_ids"][:args.n_scenes]]
    sampler = SceneSampler(args.shard_glob, batch_size=1, eval_size=0,
                           include_sdc_paths=False, prefetch=False, index_dir=args.index_dir)
    scen = sampler.load_explicit(ids)
    n_obj = int(np.asarray(scen.log_trajectory.x).shape[1])
    env = waymax_env.PlanningAgentEnvironment(
        dynamics_model=waymax_dynamics.StateDynamics(),
        config=waymax_config.EnvironmentConfig(
            max_num_objects=n_obj, controlled_object=waymax_config.ObjectType.SDC))
    st0 = env.reset(jax.tree_util.tree_map(jnp.asarray, scen))
    state = datatypes.update_state_by_log(st0, args.t0 - int(st0.timestep))
    t0, n = int(state.timestep), args.n_steps
    assert t0 == args.t0, (t0, args.t0)
    print(f"[gate] {len(ids)} failure_heldout scenes, {n_obj} objects, t0={t0}, window {n}",
          flush=True)
    cfg = SafetyObjectiveConfig()
    cand = ego_candidate(state, t0, n)
    scr = scramble_future(state, t0)

    def score(st, mode):
        return candidate_safety(cfg, cand, state=st, t0=jnp.asarray(t0, jnp.int32),
                                n_steps=n, neighbor_pred=mode)

    # ---- N1 no peeking ------------------------------------------------------ #
    for mode in ("cv", "none"):
        b0, v0 = predicted_neighbor_window(state, jnp.asarray(t0), n, mode=mode)
        b1, v1 = predicted_neighbor_window(scr, jnp.asarray(t0), n, mode=mode)
        s0, s1 = score(state, mode), score(scr, mode)
        d_box = float(jnp.max(jnp.abs(jnp.where(v0[..., None], b0 - b1, 0.0))))
        d_val = bool(jnp.all(v0 == v1))
        ca0 = np.where(np.isfinite(np.asarray(s0["c_agent"])), np.asarray(s0["c_agent"]), 1e9)
        ca1 = np.where(np.isfinite(np.asarray(s1["c_agent"])), np.asarray(s1["c_agent"]), 1e9)
        d_c = float(np.max(np.abs(ca0 - ca1)))
        check(f"N1[{mode}] scrambling the future changes nothing it reads",
              d_box == 0.0 and d_val and d_c == 0.0,
              f"max |box diff| {d_box:.2e}, validity same {d_val}, max |c_agent diff| {d_c:.2e}")
    bl0, vl0 = neighbor_window(state, jnp.asarray(t0), n)
    bl1, _ = neighbor_window(scr, jnp.asarray(t0), n)
    d_log = float(jnp.max(jnp.abs(jnp.where(vl0[..., None], bl0 - bl1, 0.0))))
    check("N1 ...while the log oracle DOES see the scramble (the gate has teeth)",
          d_log > 10.0, f"max |box diff| {d_log:.1f} m")

    # ---- N2 default untouched ------------------------------------------------- #
    sd = candidate_safety(cfg, cand, state=state, t0=jnp.asarray(t0, jnp.int32), n_steps=n)
    sl = score(state, "log")
    d2 = max(float(jnp.max(jnp.abs(jnp.nan_to_num(sd[k] - sl[k], posinf=0.0, neginf=0.0))))
             for k in ("c_edge", "c_agent", "total"))
    check("N2 default neighbor_pred is the log path, bit for bit", d2 == 0.0, f"max diff {d2:.2e}")

    # ---- N3 cv accuracy --------------------------------------------------------- #
    bc, vc = predicted_neighbor_window(state, jnp.asarray(t0), n, mode="cv")
    both = np.asarray(vc & vl0)
    err = np.hypot(np.asarray(bc[..., 0] - bl0[..., 0]), np.asarray(bc[..., 1] - bl0[..., 1]))
    sp = np.hypot(np.asarray(scen.log_trajectory.vel_x)[..., t0],
                  np.asarray(scen.log_trajectory.vel_y)[..., t0])            # [B, N]
    moving = both & (sp[:, None, :] > 1.0)
    e1, en = err[:, 0][moving[:, 0]], err[:, -1][moving[:, -1]]
    NUMS.update({"cv_err_1step_p90": float(np.percentile(e1, 90)),
                 "cv_err_2s_p50": float(np.percentile(en, 50)),
                 "cv_err_2s_p90": float(np.percentile(en, 90))})
    print(f"[gate] cv vs the true future, moving agents: 1 step p50/p90 "
          f"{np.percentile(e1, 50):.3f}/{np.percentile(e1, 90):.3f} m; "
          f"{n} steps p50/p90 {np.percentile(en, 50):.2f}/{np.percentile(en, 90):.2f} m")
    check("N3 cv is within 0.5 m of the truth one step ahead (p90)",
          float(np.percentile(e1, 90)) < 0.5, f"{np.percentile(e1, 90):.3f} m")

    # ---- N4 who is seen ---------------------------------------------------------- #
    sn, sc = score(state, "none"), score(state, "cv")
    now_valid = np.asarray(state.sim_trajectory.valid)[..., t0] & ~np.asarray(state.object_metadata.is_sdc)
    check("N4 'none' sees no agents", float(sn["n_neighbors"]) == 0.0
          and bool(np.all(~np.isfinite(np.asarray(sn["c_agent"])) | (np.asarray(sn["c_agent"]) > 1e8))),
          f"n_neighbors {float(sn['n_neighbors']):.2f}")
    want = float(now_valid.sum(1).mean())
    check("N4 'cv' sees exactly the agents valid at t0", abs(float(sc["n_neighbors"]) - want) < 1e-4,
          f"n_neighbors {float(sc['n_neighbors']):.2f} vs {want:.2f}")

    print()
    head = "PASS" if not FAILS else "FAIL: " + "; ".join(FAILS)
    if args.verdict:
        Path(args.verdict).parent.mkdir(parents=True, exist_ok=True)
        Path(args.verdict).write_text(head + "\n" + json.dumps(NUMS) + "\n")
        print(f"[gate] verdict -> {args.verdict}")
    if FAILS:
        print(f"[gate] {len(FAILS)} FAILED")
        raise SystemExit(1)
    print("[gate] all neighbour-prediction gates passed")


if __name__ == "__main__":
    main()
