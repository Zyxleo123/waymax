"""Gates for ``ego_vel_source="realised"``: the observation stops lying about
the ego's speed, and nothing else moves.

THE DEFECT. The diffusion model's position and velocity heads are denormalized
by independent constants, so the velocity the policy WRITES overstates what its
positions DRIVE by 13-23% (2026-09-19, 753 episodes; the log's own two agree to
1.000). ``_splice_sim_into_log_batched`` copied that written velocity into the
scratch log, and ``data/preprocess.py`` builds the policy's ``ego_state`` from
it. ``"realised"`` replaces the ego's velocity on policy-executed steps with the
backward difference of the executed positions.

WHAT HAS TO HOLD, each asserted rather than argued:

* **G0 default untouched** -- ``"commanded"`` reproduces the pre-change splice
  (verbatim copy below) bit for bit, on a driven state, every object and field.
* **G1 calibration: the log column still reads 1.000** -- the estimator applied
  to the HUMAN's logged positions reproduces the human's logged velocity
  (pooled speed ratio within ``--calib_tol``). This is the property that makes
  ``"realised"`` a return to the pretraining manifold and not a new input scale:
  the model was trained on the log's velocity field, and on the log the two
  agree. Checked end to end too (G1b): under expert replay the ``ego_state``
  speed at each policy anchor matches the log's recorded speed there.
* **G2 no-op until the policy drives** -- at reset both splices are identical,
  and so is the first observation.
* **G3 only the ego's velocity moves** -- after driving, positions / yaw / every
  non-ego row are bit-identical between the two splices, and the ego's velocity
  differs only on steps ``first_policy_step..timestep``.
* **G4 the observation reads the truth** -- the ego drives the logged path at
  ``--warp`` x its schedule while writing the log's (faster) velocity, the
  policy's defect in miniature. ``"commanded"`` must show the lie
  (obs / realised ~ 1 / warp) and ``"realised"`` must show 1.000 at every
  anchor, on the jitted replan path and the eager one.
* **G5 jit == eager** under ``"realised"``.

G1 divides each step by its OWN timestamp gap, so it tests the log alone. Its
first run (job 62441, 2026-09-23) divided by ``compute_world_dt_seconds``
instead, read 1.1085, and so exposed the first-pair world-dt bug (see
``tests/test_world_dt.py``): the envs here therefore run ``--world_dt_method
median`` by default, without which ``"realised"`` divides by the wrong step.

No denoiser: every trajectory is supplied to ``env.step``. CPU, minutes. Scenes
come from the select run's own corpus (``--splits_json`` ids through the byte
offset index), so the calibration is on the population being evaluated.
Submitted, not run inline::

    bash rl/diffusion_ft/slurm/submit_test_ego_vel.sh

``--verdict PATH`` writes ``PASS`` or ``FAIL: <gates>`` as its first line (plus a
JSON line of the numbers) for a dependent job to read.
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

import jax.numpy as jnp
import numpy as np

from data.preprocess import compute_world_dt_seconds
from rl.diffusion_ft.checkpoint import DEFAULT_CHECKPOINT_PATH, load_diffusion_checkpoint
from rl.diffusion_ft.env import DiffusionWaymaxEnv
from rl.diffusion_ft.reward import RewardConfig
from rl.diffusion_ft.scene_sampler import DEFAULT_INDEX_DIR, SceneSampler

FAILS: list[str] = []
NUMS: dict[str, float] = {}
FIELDS = ("x", "y", "yaw", "vel_x", "vel_y")


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""),
          flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def _splice_reference(state):
    """The splice as it was before ``ego_vel_source`` existed (2026-09-22),
    verbatim. G0 holds the default to it."""
    sim = state.sim_trajectory
    log = state.log_trajectory
    step_idx = jnp.arange(log.x.shape[-1])
    use_bnt = (step_idx[None, None, :] <= state.timestep) & jnp.asarray(sim.valid)

    def blend(sim_f, log_f):
        return jnp.where(use_bnt, sim_f, log_f)

    return state.replace(log_trajectory=log.replace(
        x=blend(sim.x, log.x), y=blend(sim.y, log.y), yaw=blend(sim.yaw, log.yaw),
        vel_x=blend(sim.vel_x, log.vel_x), vel_y=blend(sim.vel_y, log.vel_y)))


def max_diff(a, b, fields=FIELDS, mask=None) -> float:
    out = 0.0
    for f in fields:
        d = np.abs(np.asarray(getattr(a.log_trajectory, f), np.float64)
                   - np.asarray(getattr(b.log_trajectory, f), np.float64))
        if mask is not None:
            d = d[mask]
        out = max(out, float(d.max()) if d.size else 0.0)
    return out


def ego_rows(arr_bnt, ego_b):
    return np.asarray(arr_bnt)[np.arange(len(ego_b)), ego_b]


def warped_override(env, warp: float) -> np.ndarray:
    """The ego's logged path driven at ``warp`` x its schedule from the reset
    anchor, while the velocity field keeps the LOG's (faster) values -- positions
    under-travel and the written velocity does not know, as with the policy."""
    log = np.asarray(env.ego_log_world_trajectory(), np.float64)   # [B, T, 5]
    T = log.shape[1]
    t0 = env.timestep
    tau = np.arange(T, dtype=np.float64)
    tau[t0:] = t0 + warp * (tau[t0:] - t0)
    lo = np.clip(np.floor(tau).astype(int), 0, T - 1)
    hi = np.clip(lo + 1, 0, T - 1)
    f = (tau - lo)[None, :]
    yaw_u = np.unwrap(log[..., 2], axis=1)
    out = log.copy()
    for k, src in ((0, log[..., 0]), (1, log[..., 1]), (2, yaw_u)):
        out[..., k] = src[:, lo] * (1 - f) + src[:, hi] * f
    out[..., 2] = np.arctan2(np.sin(out[..., 2]), np.cos(out[..., 2]))
    return out.astype(np.float32)


def drive(env, state, override, replans: int, vmax: float):
    """Reset, then execute ``override`` for ``replans`` replans. Returns the
    first observation and, per anchor r >= 1, ``[R, B]`` arrays of: the ego speed
    ``ego_state`` shows, the speed the positions drove into that anchor, and the
    speed carried by ``aux["anchor_world_state"]``."""
    feats0 = env.reset(state)
    obs, real, auxv = [], [], []
    ego_b = np.argmax(np.asarray(env.sim_state.object_metadata.is_sdc), 1)
    for _ in range(replans):
        t = env.timestep
        out = env.step(trajectory_world_bt5=jnp.asarray(override[:, t:t + env._replan + 1]))
        t1 = env.timestep
        es = np.asarray(out.features["ego_state"], np.float64)
        obs.append(np.linalg.norm(es[:, 2:4], axis=-1) * vmax)
        sx = ego_rows(env.sim_state.sim_trajectory.x, ego_b)
        sy = ego_rows(env.sim_state.sim_trajectory.y, ego_b)
        dt = np.asarray(env.last_aux["world_dt_seconds"], np.float64)
        real.append(np.hypot(sx[:, t1] - sx[:, t1 - 1], sy[:, t1] - sy[:, t1 - 1]) / dt)
        aw = np.asarray(env.last_aux["anchor_world_state"], np.float64)   # x,y,vx,vy,yaw
        auxv.append(np.hypot(aw[:, 2], aw[:, 3]))
    return feats0, np.array(obs), np.array(real), np.array(auxv)


def calibration(state, moving_mps: float):
    """G1 on the logs: backward difference of the logged positions vs the logged
    velocity, for the ego (the gate) and every valid object (reported)."""
    log = state.log_trajectory
    # Each step's own timestamp gap where it is usable, else the scene's median
    # step: the estimator must be judged against the time that actually elapsed.
    med = np.asarray(compute_world_dt_seconds(state, 0.1, "median"), np.float64)[:, None, None]
    ts = np.asarray(log.timestamp_micros, np.float64) * 1e-6
    gap = np.diff(ts, axis=-1)
    dt = np.where(gap > 0.0, gap, med)
    dt2 = np.where(ts[..., 2:] - ts[..., :-2] > 0.0, ts[..., 2:] - ts[..., :-2], 2 * med)
    x, y = np.asarray(log.x, np.float64), np.asarray(log.y, np.float64)
    vx, vy = np.asarray(log.vel_x, np.float64), np.asarray(log.vel_y, np.float64)
    ok = np.asarray(log.valid)
    pair = ok[..., 1:] & ok[..., :-1]
    bx, by = np.diff(x, axis=-1) / dt, np.diff(y, axis=-1) / dt           # t = 1..T-1
    lx, ly = vx[..., 1:], vy[..., 1:]
    lsp = np.hypot(lx, ly)
    mov = pair & (lsp > moving_mps)
    # The backward estimate against the log averaged over the SAME step: if its
    # error is mostly the half-step lag, this comparison is much tighter.
    avg_x, avg_y = 0.5 * (vx[..., 1:] + vx[..., :-1]), 0.5 * (vy[..., 1:] + vy[..., :-1])
    cx = (x[..., 2:] - x[..., :-2]) / dt2
    cy = (y[..., 2:] - y[..., :-2]) / dt2
    csp_log = np.hypot(vx[..., 1:-1], vy[..., 1:-1])
    movc = ok[..., 2:] & ok[..., :-2] & ok[..., 1:-1] & (csp_log > moving_mps)
    ego_b = np.argmax(np.asarray(state.object_metadata.is_sdc), 1)
    sel = np.zeros(mov.shape[:2], bool)
    sel[np.arange(len(ego_b)), ego_b] = True

    def stats(m):
        err = np.hypot(bx - lx, by - ly)[m]
        err_avg = np.hypot(bx - avg_x, by - avg_y)[m]
        return {"n": int(m.sum()), "ratio": float(np.hypot(bx, by)[m].sum() / lsp[m].sum()),
                "err_p50": float(np.percentile(err, 50)),
                "err_p90": float(np.percentile(err, 90)),
                "err_p99": float(np.percentile(err, 99)),
                "err_vs_step_avg_p50": float(np.percentile(err_avg, 50)),
                "rel_err_p50": float(np.percentile(err / lsp[m], 50))}

    ego = stats(mov & sel[..., None])
    allobj = stats(mov)
    mc = movc & sel[..., None]
    ego["central_ratio"] = float(np.hypot(cx, cy)[mc].sum() / csp_log[mc].sum())
    return ego, allobj


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits_json", required=True)
    p.add_argument("--splits", default="failure_heldout,success_heldout")
    p.add_argument("--shard_glob",
                   default="/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*")
    p.add_argument("--index_dir", default=DEFAULT_INDEX_DIR)
    p.add_argument("--ckpt", default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--n_calib", type=int, default=32, help="scenes per split for G1")
    p.add_argument("--n_env", type=int, default=4, help="scenes per split for G0/G1b-G5")
    p.add_argument("--replans", type=int, default=4)
    p.add_argument("--replan_interval_steps", type=int, default=10)
    p.add_argument("--warp", type=float, default=0.9)
    p.add_argument("--calib_tol", type=float, default=0.01)
    p.add_argument("--moving_mps", type=float, default=0.5)
    p.add_argument("--world_dt_method", default="median", choices=("first", "median"))
    p.add_argument("--verdict", default="")
    args = p.parse_args(argv)

    spec = json.loads(Path(args.splits_json).read_text())
    names = [s for s in args.splits.split(",") if s]
    ids_cal = [int(i) for n in names for i in spec["splits"][n]["scene_ids"][:args.n_calib]]
    ids_env = [int(i) for n in names for i in spec["splits"][n]["scene_ids"][:args.n_env]]
    sampler = SceneSampler(args.shard_glob, batch_size=1, eval_size=0,
                           include_sdc_paths=False, prefetch=False, index_dir=args.index_dir)
    ckpt = load_diffusion_checkpoint(args.ckpt)
    vmax = float(ckpt.preprocess_cfg.max_velocity)

    # ---- G1: the estimator against the human's own log -------------------- #
    ego, allobj = calibration(sampler.load_explicit(ids_cal), args.moving_mps)
    NUMS.update({f"calib_ego_{k}": v for k, v in ego.items()})
    NUMS.update({f"calib_all_{k}": v for k, v in allobj.items()})
    print(f"[gate] calibration on {len(ids_cal)} scenes, human moving > {args.moving_mps} m/s")
    print(f"       ego   n={ego['n']} backward/log speed {ego['ratio']:.4f}  "
          f"central/log {ego['central_ratio']:.4f}  |err| p50/p90/p99 "
          f"{ego['err_p50']:.3f}/{ego['err_p90']:.3f}/{ego['err_p99']:.3f} m/s  "
          f"(vs the log averaged over the same step: p50 {ego['err_vs_step_avg_p50']:.3f})")
    print(f"       all   n={allobj['n']} backward/log speed {allobj['ratio']:.4f}  "
          f"|err| p50/p90 {allobj['err_p50']:.3f}/{allobj['err_p90']:.3f} m/s  (reported only)")
    check("G1 the log column still reads 1.000: backward difference of the human's "
          "positions reproduces the human's logged speed",
          abs(ego["ratio"] - 1.0) < args.calib_tol,
          f"ratio {ego['ratio']:.4f}, tol {args.calib_tol}")

    # ---- env gates --------------------------------------------------------- #
    state = sampler.load_explicit(ids_env)
    n_obj = int(np.asarray(state.log_trajectory.x).shape[1])

    def make(src, jit=True):
        return DiffusionWaymaxEnv(ckpt, reward_config=RewardConfig(), max_num_objects=n_obj,
                                  replan_interval_steps=args.replan_interval_steps, seed=0,
                                  jit_replan=jit, ego_vel_source=src,
                                  world_dt_method=args.world_dt_method)

    env_c, env_r, env_re = make("commanded"), make("realised"), make("realised", jit=False)
    ego_b = np.argmax(np.asarray(state.object_metadata.is_sdc), 1)

    # G2 at reset.
    f_c, f_r = env_c.reset(state), env_r.reset(state)
    d_reset = max_diff(env_c._splice(env_c.sim_state), env_r._splice(env_r.sim_state))
    d_feat0 = max(float(np.abs(np.asarray(f_c[k], np.float64)
                               - np.asarray(f_r[k], np.float64)).max())
                  for k in f_c if np.asarray(f_c[k]).dtype.kind == "f")
    check("G2 no-op until the policy drives: splice and first observation identical at reset",
          d_reset == 0.0 and d_feat0 == 0.0,
          f"max |splice diff| {d_reset:.2e}, max |feature diff| {d_feat0:.2e}")

    # G1b: expert replay (the log's own poses and velocities) under realised --
    # G1 end to end, at the anchors the policy would see.
    log_ov = np.asarray(env_r.ego_log_world_trajectory(), np.float32)
    _, obs_e, _, _ = drive(env_r, state, log_ov, args.replans, vmax)
    lvx = ego_rows(state.log_trajectory.vel_x, ego_b)
    lvy = ego_rows(state.log_trajectory.vel_y, ego_b)
    t_reset = env_r._first_policy_step - 1
    anchors = [t_reset + args.replan_interval_steps * (r + 1) for r in range(args.replans)]
    lsp = np.array([np.hypot(lvx[:, a], lvy[:, a]) for a in anchors])
    mv = lsp > args.moving_mps
    e2e = float(obs_e[mv].sum() / lsp[mv].sum()) if mv.any() else float("nan")
    NUMS["expert_obs_over_log"] = e2e
    check("G1b under expert replay the realised ego_state speed matches the log's "
          "recorded speed at every policy anchor",
          bool(np.isfinite(e2e)) and abs(e2e - 1.0) < args.calib_tol,
          f"pooled obs/log {e2e:.4f} over {int(mv.sum())} moving anchors")

    # The policy's defect in miniature, driven identically under all three envs.
    ov = warped_override(env_c, args.warp)
    res = {}
    for tag, env in (("commanded", env_c), ("realised", env_r), ("realised_eager", env_re)):
        res[tag] = drive(env, state, ov, args.replans, vmax)
        res[tag + "_splice"] = env._splice(env.sim_state)
        res[tag + "_ref"] = _splice_reference(env.sim_state)
        res[tag + "_t"] = env.timestep

    # G0: the default is the old splice.
    d0 = max_diff(res["commanded_splice"], res["commanded_ref"])
    check("G0 ego_vel_source='commanded' is the pre-change splice, bit for bit",
          d0 == 0.0, f"max |diff| {d0:.2e} over every object, step and field")

    # G3: only the ego's executed velocity moves.
    sc, sr = res["commanded_splice"], res["realised_splice"]
    shape = np.asarray(sc.log_trajectory.x).shape
    is_ego = np.zeros(shape[:2], bool)
    is_ego[np.arange(len(ego_b)), ego_b] = True
    t_end = res["commanded_t"]
    window = np.zeros(shape[-1], bool)
    window[env_r._first_policy_step:t_end + 1] = True
    ego_bnt = np.broadcast_to(is_ego[..., None], shape)
    win_bnt = np.broadcast_to(window[None, None], shape)
    d_pose = max_diff(sc, sr, ("x", "y", "yaw"))
    d_npc = max_diff(sc, sr, ("vel_x", "vel_y"), mask=~ego_bnt)
    d_hist = max_diff(sc, sr, ("vel_x", "vel_y"), mask=ego_bnt & ~win_bnt)
    d_ego = max_diff(sc, sr, ("vel_x", "vel_y"), mask=ego_bnt & win_bnt)
    check("G3 positions/yaw of every object identical between the two splices",
          d_pose == 0.0, f"max {d_pose:.2e}")
    check("G3 every non-ego velocity identical (the log column is untouched)",
          d_npc == 0.0, f"max {d_npc:.2e}")
    check("G3 the ego's replayed history keeps its logged velocity",
          d_hist == 0.0, f"max {d_hist:.2e} outside steps {env_r._first_policy_step}..{t_end}")
    check("G3 the ego's executed velocity DOES change (the flag is not inert)",
          d_ego > 0.1, f"max {d_ego:.3f} m/s inside the window")

    # G4: what the observation says against what the car did.
    for tag in ("commanded", "realised", "realised_eager"):
        _, obs, real, auxv = res[tag]
        m = real > args.moving_mps
        ratio = float(obs[m].sum() / real[m].sum())
        worst = float(np.max(np.abs(obs[m] / real[m] - 1.0)))
        aux_gap = float(np.max(np.abs(auxv - obs)))
        NUMS[f"{tag}_obs_over_real"] = ratio
        NUMS[f"{tag}_worst_rel"] = worst
        print(f"[gate] {tag:<15} obs/realised pooled {ratio:.4f}  worst anchor "
              f"|obs/real-1| {worst:.4f}  max |aux anchor - ego_state| {aux_gap:.2e} m/s")
        check(f"G4[{tag}] aux anchor velocity agrees with ego_state", aux_gap < 1e-3,
              f"{aux_gap:.2e} m/s")
        if tag == "commanded":
            # The override writes the log's speed over positions driven at `warp`,
            # so the lie is ~1/warp, divided by the log's own calibration.
            want = 1.0 / (args.warp * ego["ratio"])
            check("G4 'commanded' shows the lie", abs(ratio - want) < 0.03,
                  f"obs/realised {ratio:.4f}, expected ~{want:.4f}")
        else:
            check(f"G4[{tag}] 'realised' shows the truth at every anchor", worst < 1e-3,
                  f"worst |obs/real - 1| {worst:.2e}")

    # G5: the two replan paths agree.
    d_je = float(np.max(np.abs(res["realised"][1] - res["realised_eager"][1])))
    check("G5 realised: jitted and eager replan paths agree", d_je < 1e-3,
          f"max |ego speed diff| {d_je:.2e} m/s")

    print()
    head = "PASS" if not FAILS else "FAIL: " + "; ".join(FAILS)
    if args.verdict:
        Path(args.verdict).parent.mkdir(parents=True, exist_ok=True)
        Path(args.verdict).write_text(head + "\n" + json.dumps(NUMS) + "\n")
        print(f"[gate] verdict -> {args.verdict}")
    if FAILS:
        print(f"[gate] {len(FAILS)} FAILED")
        raise SystemExit(1)
    print("[gate] all ego-velocity splice gates passed")


if __name__ == "__main__":
    main()
