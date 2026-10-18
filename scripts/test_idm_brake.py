"""Gate: ``idm_brake`` (simulation/sim_agents.py) reacts to the ego without inventing traffic.

Expert replay (the ego drives its own log), so every difference below belongs to the NPCs.
Rollouts through the evaluation path (``_build_rollout_env_and_actor``) under log / idm /
idm_brake, plus the closed loop (``predict_planner_trajectories_closed_loop_idm``) with a
planner that re-plans the logged ego:

  B1  log rollout: every NPC's simulated pose == its log (the control)
  B2  the ego is identical under log / idm / idm_brake
  B3  idm_brake keeps NPCs on their own logged path (lateral p95 < 1 m)
  B4  idm_brake NPCs never run AHEAD of their recording (along-path p99 <= 1 m)
  B5  idm_brake invents no collision on the human's own log (no scene overlaps that log replay does not)
  B6  late entrants (invalid at reset, valid later) appear at their logged pose
  B7  the closed loop's NPCs == the evaluation-path idm_brake rollout (same ego, same model)
  B8  the closed loop shows the planner the simulated NPCs
Stock idm's numbers are printed alongside (B3-B6), so its failure stays visible; only
idm_brake is gated. The along-path / lateral measure is yixiz/waymax_rs's
rl/diffusion_ft/tests/test_reactive_npc.py gate 3, unchanged.
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from waymax import config as waymax_config  # noqa: E402

from data.scenario_loader import load_scenario_state_batch_fast  # noqa: E402
from simulation.evaluation_utils import _build_rollout_env_and_actor, _run_batched_rollout  # noqa: E402
from simulation.idm_closed_loop import predict_planner_trajectories_closed_loop_idm  # noqa: E402
from test_idm_closed_loop import LogEgoPlanner  # noqa: E402

RESET_T = 10          # both rollouts reset with init_steps=11: t<=10 is the log
MIN_PATH_M = 2.0      # shorter logged paths carry no schedule information (parked cars)
FAILS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def eval_rollout(state, kind: str):
    """(final sim trajectory as numpy dict, per-scene overlap) of the evaluation path."""
    n_obj, T = int(np.asarray(state.log_trajectory.x).shape[1]), int(np.asarray(state.log_trajectory.x).shape[-1])
    env, actor = _build_rollout_env_and_actor(("overlap", "offroad"), n_obj, kind)
    ro = _run_batched_rollout(state, env, actor, jax.random.PRNGKey(0), T - 1)
    sim = jax.tree_util.tree_map(lambda x: x[-1], ro.state).sim_trajectory
    traj = {f: np.asarray(getattr(sim, f)) for f in ("x", "y", "valid")}
    # metric rows are rollout steps (t=10..), columns scenes -- runner.py's convention
    overlap = (np.asarray(ro.metrics["overlap"].value) > 0).any(0)
    return traj, overlap


def path_offsets(lx, ly, lv, sx, sy, keep):
    """Lateral distance to the agent's own logged path, and the signed along-path offset in
    METRES of arclength (neg = behind the recording, pos = ahead), over moving agents."""
    lat, sched, n_static = [], [], 0
    for b in range(lx.shape[0]):
        for i in range(lx.shape[1]):
            steps = np.flatnonzero(keep[b, i])
            t_idx = np.flatnonzero(lv[b, i])
            if steps.size == 0 or t_idx.size < 2:
                continue
            path = np.stack([lx[b, i, t_idx], ly[b, i, t_idx]], -1)
            pts = np.stack([sx[b, i, steps], sy[b, i, steps]], -1)
            d = np.linalg.norm(pts[:, None] - path[None], axis=-1)
            j = d.argmin(1)
            lat.append(d[np.arange(len(j)), j])
            s_path = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=-1))])
            if s_path[-1] < MIN_PATH_M:
                n_static += 1
                continue
            pos = np.searchsorted(t_idx, steps).clip(0, len(t_idx) - 1)
            sched.append(s_path[j] - s_path[pos])
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(1)  # noqa: E731
    return cat(lat), cat(sched), n_static


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tfrecord", default=sorted(glob(
        "/zfsauton/datasets/womd/tf_example/validation/validation_tfexample.tfrecord-00000-of-00150"))[0])
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--start", type=int, default=0)
    a = p.parse_args()
    ds_cfg = dataclasses.replace(waymax_config.WOD_1_3_1_TRAINING, path=a.tfrecord, max_num_objects=128)
    out = load_scenario_state_batch_fast(ds_cfg, list(range(a.start, a.start + a.n)))
    state = out[0] if isinstance(out, tuple) else out
    log = state.log_trajectory
    lx, ly = np.asarray(log.x), np.asarray(log.y)
    lv = np.asarray(log.valid).astype(bool)
    is_sdc = np.asarray(state.object_metadata.is_sdc).astype(bool)
    npc = ~is_sdc[:, :, None]
    late = npc[..., 0] & ~lv[..., RESET_T] & lv[..., RESET_T + 1:].any(-1)
    print(f"[brake] {a.tfrecord} scenes {a.start}..{a.start + a.n - 1}: {int((npc[..., 0] & lv.any(-1)).sum())} NPCs, "
          f"{int(late.sum())} late entrants", flush=True)

    res = {k: eval_rollout(state, k) for k in ("log", "idm", "idm_brake")}
    for k, (tr, ovl) in res.items():
        print(f"[brake] {k:9s}: scenes with ego overlap {int(ovl.sum())}/{a.n}: {np.flatnonzero(ovl).tolist()}", flush=True)

    tr = res["log"][0]
    m = npc & lv & tr["valid"].astype(bool)
    dev = np.maximum(np.abs(tr["x"] - lx), np.abs(tr["y"] - ly))[m]
    check("B1 log rollout replays every NPC", float(dev.max()) < 1e-3,
          f"max |sim-log| {float(dev.max()):.2e} m over {int(m.sum())} poses")

    sdc = np.argmax(is_sdc, 1)
    w = np.arange(a.n)
    ego = {k: np.stack([r[0]["x"][w, sdc], r[0]["y"][w, sdc]], -1) for k, r in res.items()}
    ego_dev = max(float(np.abs(ego[k] - ego["log"]).max()) for k in ("idm", "idm_brake"))
    check("B2 ego identical under log / idm / idm_brake", ego_dev < 1e-3, f"max {ego_dev:.2e} m")

    for k in ("idm", "idm_brake"):
        tr = res[k][0]
        sv = tr["valid"].astype(bool)
        keep = npc & lv & sv
        d = np.hypot(tr["x"] - lx, tr["y"] - ly)[keep]
        lat, sched, n_static = path_offsets(lx, ly, lv, tr["x"], tr["y"], keep)
        ahead = float(np.mean(sched > 1.0))
        print(f"[brake] {k:9s}: |sim-log| p50 {np.percentile(d, 50):.2f} p99 {np.percentile(d, 99):.2f} "
              f"max {d.max():.1f} m; {np.mean(d > 0.5):.1%} of poses > 0.5 m off schedule; "
              f"valid NPC poses sim {int((npc & sv).sum())} vs log {int((npc & lv).sum())}", flush=True)
        lat_txt = f"lateral p50 {np.percentile(lat, 50):.3f} p95 {np.percentile(lat, 95):.3f} p99 {np.percentile(lat, 99):.3f} m"
        sched_txt = (f"along-path p1 {np.percentile(sched, 1):.2f} p50 {np.percentile(sched, 50):.2f} "
                     f"p99 {np.percentile(sched, 99):.2f} m; {ahead:.1%} of poses > 1 m ahead ({n_static} parked excluded)")
        # late entrants: valid in the sim at their first logged step after reset, at the logged pose
        bs, ns = np.nonzero(late)
        t_in = np.array([RESET_T + 1 + np.flatnonzero(lv[b, n, RESET_T + 1:])[0] for b, n in zip(bs, ns)], int)
        ok_in = sv[bs, ns, t_in] & (np.hypot(tr["x"][bs, ns, t_in] - lx[bs, ns, t_in],
                                             tr["y"][bs, ns, t_in] - ly[bs, ns, t_in]) < 1e-3)
        late_txt = f"{int(ok_in.sum())}/{len(t_in)} late entrants appear at their logged pose"
        if k == "idm":
            print(f"[brake] idm (stock, not gated): {lat_txt}; {sched_txt}; {late_txt}", flush=True)
            continue
        check("B3 idm_brake keeps NPCs on their own logged path", float(np.percentile(lat, 95)) < 1.0, lat_txt)
        check("B4 idm_brake NPCs never run ahead of their recording", float(np.percentile(sched, 99)) <= 1.0, sched_txt)
        invented = res["idm_brake"][1] & ~res["log"][1]
        check("B5 idm_brake invents no collision on the human's log", not invented.any(),
              f"scenes overlapping only under idm_brake: {np.flatnonzero(invented).tolist()}")
        check("B6 late entrants appear at their logged pose", bool(ok_in.all()), late_txt)

    # ---- the closed loop with the same model must reproduce the evaluation path ----
    sdc_log = np.stack([np.asarray(f)[w, sdc] for f in (log.x, log.y, log.yaw, log.vel_x, log.vel_y)], -1)
    planner = LogEgoPlanner(sdc_log, is_sdc)
    cfg = SimpleNamespace(mask_goal=False, use_subgoal=False, other_agents="idm_brake")
    _, final, _ = predict_planner_trajectories_closed_loop_idm(
        cfg, state, jnp.asarray(sdc_log[:, -1, :2]), planner, replan_interval_steps=10,
        instruction_interval_steps=10, start_timestep=RESET_T, use_subgoal=False, rng_key=jax.random.PRNGKey(0))
    fx, fy = np.asarray(final.log_trajectory.x), np.asarray(final.log_trajectory.y)
    fv = np.asarray(final.log_trajectory.valid).astype(bool)
    tr = res["idm_brake"][0]
    ev = tr["valid"].astype(bool)
    both = npc & fv & ev
    cl_dev = float(np.hypot(fx - tr["x"], fy - tr["y"])[both].max()) if both.any() else 0.0
    n_mis = int(((fv != ev) & npc).sum())
    check("B7 closed-loop NPCs == evaluation-path idm_brake rollout", cl_dev < 0.05 and n_mis == 0,
          f"max |closed loop - eval| {cl_dev:.2e} m; validity mismatches {n_mis}")
    shown_err = 0.0
    for t, (sx, sy, sv) in sorted(planner.seen.items()):
        mm = sv & fv[..., t] & ~is_sdc
        if mm.any():
            shown_err = max(shown_err, float(np.hypot(sx - fx[..., t], sy - fy[..., t])[mm].max()))
    check("B8 the planner is shown the simulated NPCs", shown_err < 1e-3,
          f"replans {sorted(planner.seen)}; max |shown - closed loop| {shown_err:.2e} m")

    print("=== " + ("PASS" if not FAILS else "FAIL: " + "; ".join(FAILS)) + " ===", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
