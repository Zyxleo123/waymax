"""Validate the differentiable safety loss on the scenes it is meant to fix.

Before spending a training run on a new objective, the objective itself has to
be shown to measure what it claims. This script rolls the *base* diffusion
policy over the curated failure set -- reproducing the very off-road clips and
near-collisions the loss is designed to price -- and checks the loss against the
simulator's own verdict, frame by frame.

Five gates, each falsifiable, in decreasing order of how badly a failure would
invalidate the design:

  A. **Edge agreement (exact).** ``c_edge < 0`` iff waymax's ``offroad`` flag is
     1, on every executed step. Both read the same signed corner distance, so
     anything short of 100% agreement is a bug in this module, not a modelling
     choice.
  B. **Agent agreement.** Under the default exact geometry this is two-sided
     like Gate A: ``c_agent < 0`` iff waymax's ``overlap`` is 1, because the
     sign of the exact gap *is* the separating-axis test ``has_overlap`` runs.
     Under ``--agent_geometry disc`` it falls back to the one-sided form the
     cover permits -- every overlap must be caught, but flagging a close
     non-overlap is the intended pessimism and only its rate is reported.
  C. **Separation.** The loss on violating frames must stand well clear of the
     loss on clean frames. Without this the term is a constant with noise on it.
  D. **Plateau.** Frames comfortably beyond the margin must sit at ~0 -- this is
     the "flat when safe" half of the requirement, and it is what keeps the loss
     from taxing ordinary driving. Reported as a fraction of the violation-frame
     loss and in absolute metres-equivalent.
  E. **Gradient usefulness.** The gradient w.r.t. the ego pose must be finite,
     concentrated on the frames that are actually close, and *descending it must
     increase real clearance* -- re-measured from scratch, with a fresh
     nearest-edge search, not by trusting the linearization.

The expert log is rolled as a control arm and scored by the same loss. It is the
calibration reference: the margins are too aggressive if the human demonstration
pays a large bill, and too slack if it pays exactly nothing on scenes it drives
tightly. That number is reported, not asserted on.

GPU job; writes artifacts only. Figures are drawn from them on CPU by
``rl.diffusion_ft.tests.viz_safety_loss``.

Usage:
    python -m rl.diffusion_ft.tests.probe_safety_loss \
        --tfrecord .../failures.tfrecord --indices 0,1,...,31 \
        --out_dir OUT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from rl.diffusion_ft.checkpoint import DEFAULT_CHECKPOINT_PATH, load_diffusion_checkpoint
from rl.diffusion_ft.env import DiffusionWaymaxEnv
from rl.diffusion_ft.reward import RewardConfig
from rl.diffusion_ft.rl_actor import DiffusionRLActor
from rl.diffusion_ft.safety_loss import (
    SafetyLossConfig,
    ego_box_5dof,
    expert_clean_mask,
    road_edge_clearance,
    safety_terms,
)
from rl.diffusion_ft.train_dppo import load_batch


# --------------------------------------------------------------------------- #
# Pulling geometry back out of a finished rollout
# --------------------------------------------------------------------------- #
def _ego_index_b(state) -> jax.Array:
    return jnp.argmax(jnp.asarray(state.object_metadata.is_sdc).astype(jnp.int32), axis=1)


def scene_geometry(state, *, use_log: bool = False) -> dict[str, jax.Array]:
    """Ego pose/shape and every other agent's box, laid out ``[B, T, ...]``.

    ``use_log`` reads the untouched human demonstration instead of the executed
    motion, which is how the expert control arm is scored through the identical
    loss. Note the *other* agents come from ``sim_trajectory`` either way: they
    are log-replayed, so the two agree, and reading them from one place keeps the
    ego the only thing that differs between arms.
    """
    ego_b = _ego_index_b(state)
    sim = state.sim_trajectory
    src = state.log_trajectory if use_log else sim
    n_obj = jnp.asarray(sim.x).shape[1]

    def ego_row(field):
        return jnp.take_along_axis(jnp.asarray(field), ego_b[:, None, None], axis=1)[:, 0]

    ego_x, ego_y, ego_yaw = ego_row(src.x), ego_row(src.y), ego_row(src.yaw)
    ego_pose = jnp.stack([ego_x, ego_y, ego_yaw], axis=-1)           # [B, T, 3]
    ego_z = ego_row(src.z)                                           # [B, T]
    # Extents are constant over time; take them at t=0 from the log, which is
    # populated for every step even where the sim has not been written yet.
    ego_len = ego_row(jnp.asarray(state.log_trajectory.length))[:, 0]
    ego_wid = ego_row(jnp.asarray(state.log_trajectory.width))[:, 0]

    def to_btn(field):
        return jnp.transpose(jnp.asarray(field), (0, 2, 1))          # [B,N,T] -> [B,T,N]

    other_box = jnp.stack(
        [to_btn(sim.x), to_btn(sim.y), to_btn(sim.length), to_btn(sim.width),
         to_btn(sim.yaw)],
        axis=-1,
    )                                                                # [B, T, N, 5]
    # Drop the ego from its own neighbour set, else it collides with itself at
    # gap = -2 * radius on every single step and swamps everything.
    not_ego_bn = ~jax.nn.one_hot(ego_b, n_obj, dtype=bool)           # [B, N]
    other_valid = to_btn(sim.valid) & not_ego_bn[:, None, :]         # [B, T, N]
    return {
        "ego_pose": ego_pose, "ego_z": ego_z,
        "ego_len": ego_len, "ego_wid": ego_wid,
        "other_box": other_box, "other_valid": other_valid,
    }


# --------------------------------------------------------------------------- #
def rollout(env, actor, state, *, seed: int, replan: int, deterministic: bool = True):
    """One batched fixed-seed episode, keeping the per-world-step waymax flags.

    Mirrors ``tests.diagnose_drive.rollout`` on the quantities they share; this
    version keeps only what the loss has to be checked against (the offroad /
    overlap indicator per executed world step, and which steps were live).
    """
    features = env.reset(state)
    start_step = env.timestep
    horizon = int(jnp.asarray(env.sim_state.log_trajectory.x).shape[-1])
    n_steps = (horizon - env.timestep) // replan
    rng = jax.random.PRNGKey(int(seed))
    B = env.batch_size

    off_step = np.zeros((B, horizon), dtype=bool)
    col_step = np.zeros((B, horizon), dtype=bool)
    live_step = np.zeros((B, horizon), dtype=bool)
    prev_done = np.zeros(B, dtype=bool)

    for _ in range(int(n_steps)):
        t_before = env.timestep
        cond1, cond2 = env.compute_condition(features)
        rng, krng = jax.random.split(rng)
        traj_ncT = actor.act(cond1, cond2, features["inst_valid"], rng=krng,
                             deterministic=deterministic)
        world = env.norm_to_world(jnp.transpose(traj_ncT, (0, 2, 1)))
        out = env.step(trajectory_world_bt5=world)

        live = ~prev_done
        ps = out.per_step_components
        if ps:
            off_k = np.asarray(ps["offroad_flag"]).astype(bool)
            col_k = np.asarray(ps["collision_flag"]).astype(bool)
            n_exec = int(off_k.shape[-1])
            sl = slice(t_before + 1, t_before + 1 + n_exec)
            off_step[:, sl] = off_k & live[:, None]
            col_step[:, sl] = col_k & live[:, None]
            live_step[:, sl] = live[:, None]
        prev_done = prev_done | np.asarray(out.terminated_b) | np.asarray(out.truncated_b)
        features = out.features
        if prev_done.all():
            break
    return {"off_step": off_step, "col_step": col_step, "live_step": live_step,
            "start_step": int(start_step), "horizon": horizon}


def expert_flags(env, state, replan: int):
    """The same per-world-step flags for the human log replay.

    ``expert_replay_episode`` runs until *every* world has absorbed, and its
    per-step flags are raw -- ``reward_step`` does not mask them by ``active``.
    So a world that crashed at step 12 still has 70 flags after it, describing a
    replay the reward already stopped scoring. Liveness is reconstructed here the
    same way the reward defines it: a step counts iff no terminating condition
    fired *strictly before* it. Without this the expert's calibration numbers
    would average in post-mortem frames, which on the dirty ground-truth scenes
    are exactly the ones sitting deep off-road.
    """
    env.reset(state)
    start_step = env.timestep
    cfg = env._reward_cfg
    ex = env.expert_replay_episode()
    horizon = int(jnp.asarray(env.sim_state.log_trajectory.x).shape[-1])
    B = env.batch_size
    off_step = np.zeros((B, horizon), dtype=bool)
    col_step = np.zeros((B, horizon), dtype=bool)
    live_step = np.zeros((B, horizon), dtype=bool)
    off_k = np.asarray(ex["per_step_offroad_bk"]).astype(bool)
    col_k = np.asarray(ex["per_step_collision_bk"]).astype(bool)
    rch_k = np.asarray(ex["per_step_reached_bk"]).astype(bool)
    n_exec = int(off_k.shape[-1])

    term_k = rch_k.copy()
    if cfg.terminate_on_collision:
        term_k = term_k | col_k
    if cfg.terminate_on_offroad:
        term_k = term_k | off_k
    fired_before = np.concatenate(
        [np.zeros((B, 1), bool), np.cumsum(term_k, axis=1)[:, :-1] > 0], axis=1
    )
    live_k = ~fired_before

    sl = slice(start_step + 1, start_step + 1 + n_exec)
    off_step[:, sl] = off_k & live_k
    col_step[:, sl] = col_k & live_k
    live_step[:, sl] = live_k
    return {"off_step": off_step, "col_step": col_step, "live_step": live_step,
            "start_step": start_step, "horizon": horizon}


# --------------------------------------------------------------------------- #
def evaluate_loss(cfg, geo, roadgraph, live_bt):
    """Run :func:`safety_terms` and return everything as numpy."""
    box = ego_box_5dof(geo["ego_pose"], geo["ego_len"][:, None], geo["ego_wid"][:, None])
    terms = safety_terms(
        cfg,
        ego_box_bt5=box,
        ego_z_bt=geo["ego_z"],
        roadgraph_points=roadgraph,
        other_box_btn5=geo["other_box"],
        other_valid_btn=geo["other_valid"],
        step_valid_bt=jnp.asarray(live_bt, jnp.float32),
    )
    return {k: np.asarray(v) for k, v in terms.items()}


def pose_gradient(cfg, geo, roadgraph, live_bt):
    """``dL/d(ego x, y, yaw)`` -- what an optimizer would actually descend."""
    length, width = geo["ego_len"][:, None], geo["ego_wid"][:, None]
    mask = jnp.asarray(live_bt, jnp.float32)

    def total(pose_bt3):
        return safety_terms(
            cfg,
            ego_box_bt5=ego_box_5dof(pose_bt3, length, width),
            ego_z_bt=geo["ego_z"],
            roadgraph_points=roadgraph,
            other_box_btn5=geo["other_box"],
            other_valid_btn=geo["other_valid"],
            step_valid_bt=mask,
        )["total"]

    return total(geo["ego_pose"]), jax.grad(total)(geo["ego_pose"])


def clearances_at(cfg, pose, geo, roadgraph, live_bt=None):
    """Recompute both hard clearances and the loss from scratch at a pose.

    Deliberately re-runs the nearest-edge search rather than reusing the anchors
    from the gradient step: the point of Gate E is to confirm the *simulator's*
    geometry improves, and reusing a stale anchor would let a linearization
    error pass as a real gain.
    """
    box = ego_box_5dof(pose, geo["ego_len"][:, None], geo["ego_wid"][:, None])
    terms = safety_terms(
        cfg, ego_box_bt5=box, ego_z_bt=geo["ego_z"], roadgraph_points=roadgraph,
        other_box_btn5=geo["other_box"], other_valid_btn=geo["other_valid"],
        step_valid_bt=(None if live_bt is None else jnp.asarray(live_bt, jnp.float32)),
    )
    return (np.asarray(terms["c_edge"]), np.asarray(terms["c_agent"]),
            float(terms["total"]))


# --------------------------------------------------------------------------- #
def gate_report(name, passed, detail):
    mark = "PASS" if passed else "FAIL"
    print(f"[probe] {mark}  Gate {name}: {detail}")
    return {"gate": name, "passed": bool(passed), "detail": detail}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=str, default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--tfrecord", type=str, required=True)
    p.add_argument("--indices", type=str, required=True)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--replan_interval_steps", type=int, default=10)
    p.add_argument("--k_trainable", type=int, default=10)
    p.add_argument("--prefix_len", type=int, default=5)
    p.add_argument("--sigma_floor", type=float, default=0.03)
    # --- loss config --- #
    p.add_argument("--edge_margin_m", type=float, default=0.3)
    p.add_argument("--edge_tau_m", type=float, default=0.15)
    p.add_argument("--agent_margin_m", type=float, default=0.75)
    p.add_argument("--agent_tau_m", type=float, default=0.25)
    p.add_argument("--edge_weight", type=float, default=1.0)
    p.add_argument("--agent_weight", type=float, default=1.0)
    p.add_argument("--agent_geometry", type=str, default="exact",
                   choices=("exact", "disc"))
    p.add_argument("--n_discs", type=int, default=5,
                   help="cap; the cover is sized per object from its L/W")
    p.add_argument("--gate_expert_dirty", action="store_true", default=True,
                   help="drop scenes whose human log is itself off-road")
    p.add_argument("--no-gate_expert_dirty", dest="gate_expert_dirty",
                   action="store_false")
    p.add_argument("--time_chunk", type=int, default=8)
    # --- gate E --- #
    p.add_argument("--grad_step", type=float, default=0.05,
                   help="metres-scale step along -grad for the descent check")
    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()

    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SafetyLossConfig(
        edge_margin_m=args.edge_margin_m, edge_tau_m=args.edge_tau_m,
        edge_weight=args.edge_weight, agent_margin_m=args.agent_margin_m,
        agent_tau_m=args.agent_tau_m, agent_weight=args.agent_weight,
        n_discs=args.n_discs, time_chunk=args.time_chunk,
        agent_geometry=args.agent_geometry,
    )
    print(f"[probe] out={out_dir} scenes={len(indices)} seeds={args.seeds}")
    print(f"[probe] cfg={cfg}")

    ckpt = load_diffusion_checkpoint(args.ckpt)
    state = load_batch(args.tfrecord, indices)
    # --- gate scenes whose ground truth is itself off-road ---------------- #
    # Measured over this failure set: 3.2% of scenes carry 56.5% of all expert
    # loss, the worst single scene 36.4%. Against an unbounded barrier they would
    # dominate every statistic below, so they are removed before anything is
    # measured rather than reported alongside.
    if args.gate_expert_dirty:
        clean_b, _, _ = expert_clean_mask(cfg, state, init_steps=11)
        keep = np.asarray(clean_b)
        if not keep.all():
            dropped = [indices[i] for i in np.where(~keep)[0]]
            print(f"[probe] gating {len(dropped)}/{len(indices)} expert-dirty "
                  f"scenes (log itself off-road): {dropped}")
            indices = [indices[i] for i in np.where(keep)[0]]
            if not indices:
                raise SystemExit("[probe] every scene is expert-dirty; nothing to measure.")
            state = load_batch(args.tfrecord, indices)
        else:
            print(f"[probe] all {len(indices)} scenes are expert-clean")

    n_obj = int(jnp.asarray(state.log_trajectory.x).shape[1])
    # The legacy reward config: this probe never trains, and the reward only
    # decides which steps get executed before a world absorbs. Keeping it at the
    # default keeps the rollouts comparable to the existing diagnose_drive
    # artifacts.
    env = DiffusionWaymaxEnv(
        ckpt, reward_config=RewardConfig(), max_num_objects=n_obj,
        replan_interval_steps=args.replan_interval_steps, seed=0,
    )
    actor = DiffusionRLActor(
        env.policy.diffusion, k_trainable=args.k_trainable,
        prefix_len=args.prefix_len, sigma_sample_floor=args.sigma_floor,
        sigma_logprob_floor=args.sigma_floor, seed=0,
    )
    actor.reset_trainable_to_frozen()

    rows: list[dict] = []
    arms: dict[str, dict] = {}
    npz: dict[str, np.ndarray] = {}

    # ---------------- policy arms ---------------------------------------- #
    for s in range(args.seeds):
        seed = 778 + 2 * s
        r = rollout(env, actor, state, seed=seed, replan=args.replan_interval_steps)
        geo = scene_geometry(env.sim_state)
        rg = env.sim_state.roadgraph_points
        terms = evaluate_loss(cfg, geo, rg, r["live_step"])
        arms[f"base/seed{seed}"] = {"flags": r, "terms": terms, "geo": geo}
        print(f"[probe] base seed={seed}: "
              f"offroad_steps={r['off_step'].sum()} overlap_steps={r['col_step'].sum()} "
              f"live={r['live_step'].sum()} mean_loss/step="
              f"{terms['loss'].sum() / max(r['live_step'].sum(), 1):.3f}")
        if s == 0:
            npz["c_edge_base"] = terms["c_edge"].astype(np.float32)
            npz["c_agent_base"] = terms["c_agent"].astype(np.float32)
            npz["loss_base"] = terms["loss"].astype(np.float32)
            npz["edge_loss_base"] = terms["edge_loss"].astype(np.float32)
            npz["agent_loss_base"] = terms["agent_loss"].astype(np.float32)
            npz["off_base"] = r["off_step"]
            npz["col_base"] = r["col_step"]
            npz["live_base"] = r["live_step"]
            npz["ego_pose_base"] = np.asarray(geo["ego_pose"], np.float32)
            npz["nearest_edge_base"] = terms["nearest_edge_xy"].astype(np.float32)

    # ---------------- expert control arm ---------------------------------- #
    ex = expert_flags(env, state, args.replan_interval_steps)
    geo_ex = scene_geometry(env.sim_state, use_log=True)
    terms_ex = evaluate_loss(cfg, geo_ex, env.sim_state.roadgraph_points, ex["live_step"])
    arms["expert"] = {"flags": ex, "terms": terms_ex, "geo": geo_ex}
    npz["c_edge_expert"] = terms_ex["c_edge"].astype(np.float32)
    npz["c_agent_expert"] = terms_ex["c_agent"].astype(np.float32)
    npz["loss_expert"] = terms_ex["loss"].astype(np.float32)
    npz["live_expert"] = ex["live_step"]
    npz["ego_pose_expert"] = np.asarray(geo_ex["ego_pose"], np.float32)
    print(f"[probe] expert: offroad_steps={ex['off_step'].sum()} "
          f"overlap_steps={ex['col_step'].sum()} live={ex['live_step'].sum()} "
          f"mean_loss/step={terms_ex['loss'].sum() / max(ex['live_step'].sum(), 1):.3f}")

    # ---------------- per-(arm, scene, step) rows ------------------------- #
    for arm, d in arms.items():
        f, t = d["flags"], d["terms"]
        live = f["live_step"]
        for i, sc in enumerate(indices):
            m = live[i]
            if not m.any():
                continue
            # c_agent is +inf on a step with no valid neighbour at all (an empty
            # scene, or every other object already gone); json cannot hold inf,
            # so those report as None rather than poisoning the column.
            ca = t["c_agent"][i][m]
            ca_fin = ca[np.isfinite(ca)]
            rows.append({
                "arm": arm, "scene": int(sc),
                "n_live_steps": int(m.sum()),
                "n_offroad_steps": int(f["off_step"][i].sum()),
                "n_overlap_steps": int(f["col_step"][i].sum()),
                "loss": float(t["loss"][i].sum()),
                "edge_loss": float(t["edge_loss"][i].sum()),
                "agent_loss": float(t["agent_loss"][i].sum()),
                "loss_per_step": float(t["loss"][i].sum() / m.sum()),
                "min_c_edge": float(t["c_edge"][i][m].min()),
                "min_c_agent": (float(ca_fin.min()) if ca_fin.size else None),
                "median_c_edge": float(np.median(t["c_edge"][i][m])),
            })

    # ===================== the gates ====================================== #
    verdicts = []
    # Pool every executed step of every policy arm; the expert is a control and
    # is reported separately rather than mixed into the agreement statistics.
    pol = [d for a, d in arms.items() if a != "expert"]
    live = np.concatenate([d["flags"]["live_step"].ravel() for d in pol])
    off = np.concatenate([d["flags"]["off_step"].ravel() for d in pol])[live]
    col = np.concatenate([d["flags"]["col_step"].ravel() for d in pol])[live]
    c_edge = np.concatenate([d["terms"]["c_edge"].ravel() for d in pol])[live]
    c_agent = np.concatenate([d["terms"]["c_agent"].ravel() for d in pol])[live]
    loss = np.concatenate([d["terms"]["loss"].ravel() for d in pol])[live]
    edge_l = np.concatenate([d["terms"]["edge_loss"].ravel() for d in pol])[live]
    agent_l = np.concatenate([d["terms"]["agent_loss"].ravel() for d in pol])[live]
    n = int(live.sum())
    print(f"\n[probe] pooled executed steps: {n} "
          f"({off.mean()*100:.1f}% offroad, {col.mean()*100:.1f}% overlap)")

    # --- Gate A: edge agreement is exact ---------------------------------- #
    pred_off = c_edge < 0.0
    agree = float((pred_off == off).mean())
    verdicts.append(gate_report(
        "A (edge agreement)", agree == 1.0,
        f"{agree*100:.4f}% of {n} steps agree with waymax offroad "
        f"(false pos {int((pred_off & ~off).sum())}, false neg {int((~pred_off & off).sum())})"))

    # --- Gate B: agent agreement ------------------------------------------ #
    exact_geom = cfg.agent_geometry == "exact"
    label = "B (agent agreement)" if exact_geom else "B (agent conservatism)"
    if col.any():
        missed = int((col & ~(c_agent < 0.0)).sum())
        extra = int(((c_agent < 0.0) & ~col).sum())
        if exact_geom:
            # Two-sided: the sign of the exact gap is has_overlap term for term,
            # so a disagreement is a bug here or a mismatch in which neighbours
            # are counted valid -- not a modelling choice.
            verdicts.append(gate_report(
                label, missed == 0 and extra == 0,
                f"{missed + extra} of {n} steps disagree with waymax overlap "
                f"(false neg {missed} of {int(col.sum())} overlaps, false pos {extra})"))
        else:
            verdicts.append(gate_report(
                label, missed == 0,
                f"{missed} of {int(col.sum())} overlap steps missed by the disc "
                f"cover; pessimism rate {extra / max(n, 1) * 100:.2f}% of steps "
                f"flagged close but not overlapping"))
    else:
        verdicts.append(gate_report(
            label, True,
            "no overlap steps in this batch -- vacuous, rerun on scenes with collisions"))

    # --- Gate C: separation ------------------------------------------------ #
    viol = off | col
    if viol.any() and (~viol).any():
        lo, hi = float(np.median(loss[~viol])), float(np.median(loss[viol]))
        p95_clean = float(np.percentile(loss[~viol], 95))
        ratio = hi / max(lo, 1e-9)
        verdicts.append(gate_report(
            "C (separation)", hi > p95_clean,
            f"median loss {lo:.4f} clean vs {hi:.4f} violating ({ratio:.0f}x); "
            f"violating median {'above' if hi > p95_clean else 'BELOW'} clean p95 {p95_clean:.4f}"))
    else:
        verdicts.append(gate_report("C (separation)", False,
                                    "batch has no violating or no clean steps"))

    # --- Gate D: plateau --------------------------------------------------- #
    safe = c_edge > (cfg.edge_margin_m + 3 * cfg.edge_tau_m)
    if safe.any() and viol.any():
        plateau = float(np.median(edge_l[safe]))
        at_viol = float(np.median(edge_l[viol])) if viol.any() else float("nan")
        frac = plateau / max(at_viol, 1e-9)
        verdicts.append(gate_report(
            "D (plateau)", frac < 0.01,
            f"median edge loss {plateau:.5f} when clear of margin+3tau, "
            f"{frac*100:.3f}% of the {at_viol:.4f} at violation"))
    else:
        verdicts.append(gate_report("D (plateau)", False, "no comfortably-clear steps"))

    # --- Gate E: gradient descends into real clearance --------------------- #
    d0 = pol[0]
    geo0, rg0 = d0["geo"], env.sim_state.roadgraph_points
    total0, grad = pose_gradient(cfg, geo0, rg0, d0["flags"]["live_step"])
    grad_np = np.asarray(grad)
    finite = bool(np.isfinite(grad_np).all())
    gnorm_xy = np.linalg.norm(grad_np[..., :2], axis=-1)             # [B, T]
    live0 = d0["flags"]["live_step"]
    # "Near" and "far" must be judged against BOTH hazards, because the gradient
    # sums both terms. Masking on `c_edge` alone -- which an earlier version did
    # -- puts every frame that is clear of the kerb but tucked in behind a car
    # into the "far" bucket, complete with its large agent gradient, and the
    # locality ratio collapses (measured 2x, against a true value orders higher).
    # This is the same single-term-criterion-on-a-two-term-loss mistake the
    # descent check above already had to be corrected for.
    ce0_, ca0_ = np.asarray(d0["terms"]["c_edge"]), np.asarray(d0["terms"]["c_agent"])
    near_edge = ce0_ < cfg.edge_margin_m
    near_agent = np.isfinite(ca0_) & (ca0_ < cfg.agent_margin_m)
    close = live0 & (near_edge | near_agent)
    far = live0 & (ce0_ > cfg.edge_margin_m + 3 * cfg.edge_tau_m) & (
        ~np.isfinite(ca0_) | (ca0_ > cfg.agent_margin_m + 3 * cfg.agent_tau_m))
    conc = (float(gnorm_xy[close].mean()) if close.any() else 0.0)
    conc_far = (float(gnorm_xy[far].mean()) if far.any() else 0.0)

    # Descend and re-measure from scratch, with a fresh nearest-edge search.
    #
    # The assertion is on the TOTAL loss, not on either clearance. Real scenes
    # routinely wedge the ego between a vehicle and a curb, and there the two
    # barriers pull in opposite directions -- requiring both clearances to
    # improve would fail on a correct loss facing a genuinely constrained frame.
    # The per-term deltas are still reported, and so is the wedge count, because
    # "how often is this objective self-conflicting on the failure set?" is a
    # real question about the design rather than a pass/fail.
    #
    # Only the positional components are stepped: `ego_pose` mixes metres and
    # radians, so one scalar step size would treat 1 rad like 1 m.
    grad_xy_only = grad.at[..., 2].set(0.0)
    step = args.grad_step / max(float(np.abs(grad_np[..., :2]).max()), 1e-9)
    pose_new = jnp.asarray(geo0["ego_pose"]) - step * grad_xy_only
    ce0, ca0 = np.asarray(d0["terms"]["c_edge"]), np.asarray(d0["terms"]["c_agent"])
    ce1, ca1, total1 = clearances_at(cfg, pose_new, geo0, rg0, live0)
    total0 = float(total0)

    edge_at = live0 & (ce0 < cfg.edge_margin_m)
    agent_at = live0 & np.isfinite(ca0) & np.isfinite(ca1) & (ca0 < cfg.agent_margin_m)
    wedged = edge_at & agent_at
    d_edge = float((ce1 - ce0)[edge_at].mean()) if edge_at.any() else 0.0
    d_agent = float((ca1 - ca0)[agent_at].mean()) if agent_at.any() else 0.0
    descended = total1 < total0
    verdicts.append(gate_report(
        "E (gradient)", finite and descended and conc > 10 * max(conc_far, 1e-12),
        f"finite={finite}; |grad_xy| {conc:.4f} inside a margin ({int(close.sum())} frames) "
        f"vs {conc_far:.2e} clear of both ({int(far.sum())}) "
        f"({conc / max(conc_far, 1e-12):.0f}x); one {args.grad_step} m step takes the "
        f"total loss {total0:.2f} -> {total1:.2f} ({total1 - total0:+.2f}), moving "
        f"c_edge {d_edge:+.4f} m on {int(edge_at.sum())} frames and c_agent "
        f"{d_agent:+.4f} m on {int(agent_at.sum())}; {int(wedged.sum())} frames are "
        f"wedged (both terms active, so the two pull against each other)"))
    summary_gradient = {
        "total_before": total0, "total_after": total1,
        "d_c_edge": d_edge, "d_c_agent": d_agent,
        "n_edge_at_risk": int(edge_at.sum()), "n_agent_at_risk": int(agent_at.sum()),
        "n_wedged": int(wedged.sum()),
        "grad_norm_near_margin": conc, "grad_norm_far": conc_far,
        "n_close": int(close.sum()), "n_far": int(far.sum()),
    }

    npz["grad_pose_base"] = grad_np.astype(np.float32)
    npz["c_edge_after_step"] = ce1.astype(np.float32)
    npz["c_agent_after_step"] = ca1.astype(np.float32)

    # --- expert calibration (reported, not asserted) ---------------------- #
    exl = ex["live_step"]
    ex_edge = terms_ex["c_edge"][exl]
    ex_loss = terms_ex["loss"][exl]
    ex_charged = float((ex_loss > 0.01).mean())
    print(f"\n[probe] expert calibration: median c_edge {np.median(ex_edge):+.3f} m, "
          f"p05 {np.percentile(ex_edge, 5):+.3f} m, "
          f"{ex_charged*100:.1f}% of expert steps charged >0.01, "
          f"mean loss/step {ex_loss.mean():.4f}")
    print(f"[probe] policy  comparison:  median c_edge {np.median(c_edge):+.3f} m, "
          f"p05 {np.percentile(c_edge, 5):+.3f} m, "
          f"{float((loss > 0.01).mean())*100:.1f}% of policy steps charged >0.01, "
          f"mean loss/step {loss.mean():.4f}")

    summary = {
        "config": cfg.__dict__,
        "n_scenes": len(indices), "n_seeds": args.seeds,
        "pooled_steps": n,
        "offroad_rate": float(off.mean()), "overlap_rate": float(col.mean()),
        "gates": verdicts,
        "all_passed": all(v["passed"] for v in verdicts),
        "gradient": summary_gradient,
        "expert": {
            "median_c_edge": float(np.median(ex_edge)),
            "p05_c_edge": float(np.percentile(ex_edge, 5)),
            "frac_steps_charged": ex_charged,
            "mean_loss_per_step": float(ex_loss.mean()),
        },
        "policy": {
            "median_c_edge": float(np.median(c_edge)),
            "p05_c_edge": float(np.percentile(c_edge, 5)),
            "frac_steps_charged": float((loss > 0.01).mean()),
            "mean_loss_per_step": float(loss.mean()),
            "mean_edge_loss_per_step": float(edge_l.mean()),
            "mean_agent_loss_per_step": float(agent_l.mean()),
        },
    }

    # Clearance histograms, so the viz job can redraw the distribution without
    # carrying every step's raw value.
    edges = np.linspace(-5, 10, 151)
    npz["hist_edges"] = edges.astype(np.float32)
    npz["hist_c_edge_policy"] = np.histogram(c_edge, bins=edges)[0]
    npz["hist_c_edge_expert"] = np.histogram(ex_edge, bins=edges)[0]
    npz["hist_c_agent_policy"] = np.histogram(c_agent[np.isfinite(c_agent)], bins=edges)[0]
    npz["indices"] = np.asarray(indices, dtype=np.int32)

    rg = env.sim_state.roadgraph_points
    npz["rg_x"] = np.asarray(rg.x, np.float32)
    npz["rg_y"] = np.asarray(rg.y, np.float32)
    npz["rg_types"] = np.asarray(rg.types, np.int16)
    npz["rg_valid"] = np.asarray(rg.valid)

    np.savez_compressed(out_dir / "safety_probe.npz", **npz)
    with open(out_dir / "safety_probe.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with open(out_dir / "safety_probe_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[probe] verdict: {'ALL GATES PASS' if summary['all_passed'] else 'GATES FAILED'}")
    print(f"[probe] wrote {out_dir}/safety_probe.npz, .jsonl and _summary.json")


if __name__ == "__main__":
    main()
