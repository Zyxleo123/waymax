"""Gate the analytic-gradient path before spending a training run on it.

Every stage of this package has been gated before use, and this is the stage with
the most new machinery: a gradient that flows from the geometric safety loss,
back through the world postprocess, back through the diffusion sampler, into the
trainable denoiser. Four falsifiable claims, ordered by how badly a failure would
invalidate the training run:

  1. **The gradient exists and reaches the parameters.** Finite and non-zero. If
     it were zero the trainer would run happily for hours and change nothing --
     the failure mode that already cost this project a 500-iteration DPPO run.
  2. **Gradient isolation.** The frozen base denoiser, tokenizer and instruction
     encoders must be untouched. Structural in ``DiffusionRLActor``, re-checked
     here because the safety path reaches the sampler by a new route.
  3. **It descends.** N steps on a fixed batch must actually reduce the safety
     loss of the *sampled* trajectory. A correct gradient on a badly-scaled
     objective still goes nowhere.
  4. **The anchor prevents collapse.** This is the one that matters. The safety
     loss has no term rewarding progress, so its unconstrained minimum is to stop
     in open space -- perfectly safe and completely useless. Trained with
     ``anchor_weight=0`` the proposed trajectory should visibly *shrink*; trained
     with the anchor it should not. Run as a two-arm comparison so the check has a
     control rather than a threshold pulled from the air.

GPU job, artifacts only.

Usage:
    python -m rl.diffusion_ft.tests.gate_safety_grad --tfrecord ... --out_dir OUT
"""

from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rl.diffusion_ft.checkpoint import DEFAULT_CHECKPOINT_PATH, load_diffusion_checkpoint
from rl.diffusion_ft.env import DiffusionWaymaxEnv
from rl.diffusion_ft.reward import RewardConfig
from rl.diffusion_ft.rl_actor import DiffusionRLActor
from rl.diffusion_ft.safety_loss import SafetyLossConfig, expert_clean_mask
from rl.diffusion_ft.safety_objective import SafetyObjectiveConfig, candidate_safety
from rl.diffusion_ft.train_dppo import load_batch


def report(name, ok, detail):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return {"gate": name, "passed": bool(ok), "detail": detail}


def arc_length(world_bw5, n_steps):
    """Path length of the proposed window, [B] -- the collapse detector."""
    xy = np.asarray(world_bw5)[:, : n_steps + 1, :2]
    return np.linalg.norm(np.diff(xy, axis=1), axis=-1).sum(axis=1)


def expert_drift(world_bw5, state, t0, n_steps):
    """Mean distance from the proposal to the human log over the window, [B].

    The competence proxy the anchor is supposed to protect: a policy that wanders
    off the demonstration is losing the driving ability the pretraining bought,
    however safe its geometry looks.
    """
    log = state.log_trajectory
    ego_b = jnp.argmax(jnp.asarray(state.object_metadata.is_sdc).astype(jnp.int32), axis=1)

    def row(f):
        return jnp.take_along_axis(jnp.asarray(f), ego_b[:, None, None], axis=1)[:, 0]

    ref = np.stack([np.asarray(row(log.x))[:, t0 + 1 : t0 + 1 + n_steps],
                    np.asarray(row(log.y))[:, t0 + 1 : t0 + 1 + n_steps]], axis=-1)
    got = np.asarray(world_bw5)[:, 1 : 1 + n_steps, :2]
    return np.linalg.norm(got - ref, axis=-1).mean(axis=1)


def make_loss(actor, env, objcfg, *, state, t0, n_exec, cond1, cond2, mask,
              expert, alive, rng, anchor_weight):
    """The joint objective as a pure function of the trainable params."""
    def loss_fn(params):
        traj_ncT = actor.sample_world_for(params, cond1, cond2, mask, rng)
        world = env.norm_to_world(jnp.transpose(traj_ncT, (0, 2, 1)))
        terms = candidate_safety(objcfg, world, state=state, t0=t0,
                                 n_steps=n_exec, alive_b=alive)
        safety = terms["total"]
        anchor = (actor._expert_loss(params, expert, alive)
                  if anchor_weight > 0 else jnp.asarray(0.0))
        total = objcfg.safety_weight * safety + anchor_weight * anchor
        return total, {"safety": safety, "anchor": anchor,
                       "edge": jnp.sum(terms["edge_loss"]) / max(n_exec, 1),
                       "agent": jnp.sum(terms["agent_loss"]) / max(n_exec, 1)}
    return loss_fn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=str, default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--tfrecord", type=str, required=True)
    p.add_argument("--indices", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--k_trainable", type=int, default=10)
    p.add_argument("--prefix_len", type=int, default=5)
    p.add_argument("--sigma_floor", type=float, default=0.03)
    p.add_argument("--replan_interval_steps", type=int, default=10)
    p.add_argument("--safety_weight", type=float, default=1.0)
    p.add_argument("--anchor_weight", type=float, default=1.0)
    p.add_argument("--edge_margin_m", type=float, default=0.3)
    p.add_argument("--agent_margin_m", type=float, default=0.75)
    p.add_argument("--n_discs", type=int, default=5)
    p.add_argument("--agent_geometry", type=str, default="exact",
                   choices=("exact", "disc"))
    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()

    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    sl_cfg = SafetyLossConfig(edge_margin_m=args.edge_margin_m,
                              agent_margin_m=args.agent_margin_m,
                              n_discs=args.n_discs,
                              agent_geometry=args.agent_geometry)
    objcfg = SafetyObjectiveConfig(safety=sl_cfg, safety_weight=args.safety_weight,
                                   anchor_weight=args.anchor_weight)
    print(f"[gate] scenes={indices} steps={args.steps} lr={args.lr}")
    print(f"[gate] {sl_cfg}")

    ckpt = load_diffusion_checkpoint(args.ckpt)
    state = load_batch(args.tfrecord, indices)
    clean_b, _, _ = expert_clean_mask(sl_cfg, state, init_steps=11)
    keep = np.asarray(clean_b)
    if not keep.all():
        indices = [indices[i] for i in np.where(keep)[0]]
        print(f"[gate] gating {int((~keep).sum())} expert-dirty scene(s); "
              f"training on {indices}")
        state = load_batch(args.tfrecord, indices)
    n_obj = int(jnp.asarray(state.log_trajectory.x).shape[1])

    env = DiffusionWaymaxEnv(ckpt, reward_config=RewardConfig(), max_num_objects=n_obj,
                             replan_interval_steps=args.replan_interval_steps, seed=0)
    features = env.reset(state)
    t0 = env.timestep
    n_exec = args.replan_interval_steps
    cond1, cond2 = env.compute_condition(features)
    mask = features["inst_valid"]
    B = env.batch_size
    alive = jnp.ones((B,), jnp.float32)
    expert_target = features["ego_trajectory"]
    rng = jax.random.PRNGKey(0)

    results, verdicts = {}, []
    for arm, aw in (("anchored", args.anchor_weight), ("safety_only", 0.0)):
        actor = DiffusionRLActor(
            env.policy.diffusion, k_trainable=args.k_trainable,
            prefix_len=args.prefix_len, sigma_sample_floor=args.sigma_floor,
            sigma_logprob_floor=args.sigma_floor, actor_lr=args.lr,
            expert_lr=args.lr, seed=0)
        actor.reset_trainable_to_frozen()
        frozen_ref = actor.frozen_state()
        # The expert target MUST be truncated to the model's own horizon. The
        # preprocessor emits a longer ego_trajectory, and the diffusion loss is
        # length-agnostic, so a mismatch does not raise -- it silently scores a
        # horizon the checkpoint never trained on. train_dppo measured expert_loss
        # ~1.1e8, flat from the first step, which held every BC arm of the
        # 20260830 sweep at solve 0.000. Assert rather than rediscover.
        h_model = int(actor.diffusion.denoise_fn.horizon)
        probe_traj = actor.sample_world_for(actor._tparams, cond1, cond2, mask, rng)
        h_sample = int(probe_traj.shape[-1])
        if h_sample != h_model:
            raise SystemExit(f"[gate] horizon mismatch: sampler T={h_sample} vs "
                             f"denoise_fn.horizon={h_model}")
        if expert_target.shape[1] < h_model:
            raise SystemExit(f"[gate] expert target too short: "
                             f"{expert_target.shape[1]} < {h_model}")
        et = expert_target[:, :h_model, :]
        expert = (et, cond1, cond2, mask, jax.random.PRNGKey(7))

        if arm == "anchored":
            # The anchor came back at 2e8 on the first run, which is not a loss
            # value -- a v-prediction diffusion loss on in-distribution data is
            # O(1). The horizon guard above already rules out the cause
            # train_dppo blamed (and that slice is a no-op anyway: the
            # preprocessor emits exactly predict_horizon rows). So measure the
            # target itself and the per-scene loss: a single bad scene and a
            # uniformly bad target are different bugs with different fixes.
            e = np.asarray(et)
            print(f"[gate] expert target {e.shape} dtype={e.dtype}  "
                  f"min={np.nanmin(e):+.3f} max={np.nanmax(e):+.3f} "
                  f"mean|x|={np.nanmean(np.abs(e)):.3f}  "
                  f"nan={int(np.isnan(e).sum())} inf={int(np.isinf(e).sum())}")
            print(f"[gate] sampled traj  {tuple(probe_traj.shape)}  "
                  f"min={float(jnp.min(probe_traj)):+.3f} "
                  f"max={float(jnp.max(probe_traj)):+.3f} "
                  f"mean|x|={float(jnp.mean(jnp.abs(probe_traj))):.3f}")
            fn0 = actor._trainable_fn(actor._tparams)
            x0 = jnp.transpose(et, (0, 2, 1))
            lpe = np.asarray(actor.diffusion.loss_per_example(
                x0, cond1, cond2, mask, rng=jax.random.PRNGKey(7), denoise_fn=fn0))
            print(f"[gate] per-scene anchor loss: "
                  + " ".join(f"{v:.3g}" for v in lpe))
            print(f"[gate]   median {np.median(lpe):.3g}  max {lpe.max():.3g} "
                  f"(scene {int(np.argmax(lpe))}); "
                  f"{'ONE OUTLIER SCENE' if lpe.max() > 100*max(np.median(lpe),1e-9) else 'UNIFORMLY LARGE'}")
            # Same loss on the *sampled* trajectory: if the model scores its own
            # output as badly as the expert's, the loss is broken, not the target.
            lps = np.asarray(actor.diffusion.loss_per_example(
                probe_traj, cond1, cond2, mask, rng=jax.random.PRNGKey(7),
                denoise_fn=fn0))
            print(f"[gate]   same loss on the model's OWN sample: "
                  f"median {np.median(lps):.3g} max {lps.max():.3g} "
                  f"-> {'target is the problem' if np.median(lps) < np.median(lpe)/100 else 'loss//model is the problem'}")
            # Gate 0: the BC target must live in the model's own output space.
            # It did not (mean|x| 8486 vs 0.156) because the trajectory resampler
            # extrapolated instead of interpolating -- see the clamp in
            # data/preprocess.py::resample_trajectory_timebase_jax. Asserted here
            # so a regression is caught in seconds instead of being diagnosed
            # again from a 2e8 loss value.
            # --- round-trip the BC target through the policy's own postprocess.
            # The target IS the human log, so reconstructing it in world
            # coordinates and comparing against the log must give ~0. Anything
            # else means the chain (target -> norm_to_world -> compare) is broken
            # somewhere, and the anchor is regressing toward something that is not
            # the demonstration. This is the test that separates "the anchor is
            # mis-weighted" from "the anchor points at the wrong thing" -- the
            # sweep showed drift getting monotonically WORSE as the anchor got
            # stronger, which no weighting can explain.
            tgt_world = env.norm_to_world(et)
            tgt_drift = expert_drift(tgt_world, state, t0, n_exec)
            base_world = env.norm_to_world(jnp.transpose(probe_traj, (0, 2, 1)))
            base_drift = expert_drift(base_world, state, t0, n_exec)
            tgt_arc = arc_length(tgt_world, n_exec)
            base_arc = arc_length(base_world, n_exec)
            dt = 0.1
            print(f"[gate] ROUND TRIP  BC target -> world -> vs log: "
                  f"drift {tgt_drift.mean():.3f} m  (MUST be ~0; the target is the log)")
            print(f"[gate]   per scene: " + " ".join(f"{v:.2f}" for v in tgt_drift))
            print(f"[gate]   base policy drift vs log: {base_drift.mean():.3f} m "
                  f"(this is the baseline the anchor has to beat)")
            print(f"[gate]   arc over {n_exec} world steps: target {tgt_arc.mean():.2f} m "
                  f"({tgt_arc.mean()/(n_exec*dt):.1f} m/s), policy {base_arc.mean():.2f} m "
                  f"({base_arc.mean()/(n_exec*dt):.1f} m/s) -- implausible speed means "
                  f"the window or the time base is wrong, not the policy")
            verdicts.append(report(
                "0b the BC target reconstructs to the human log", tgt_drift.mean() < 0.5,
                f"round-tripping the anchor target through norm_to_world lands "
                f"{tgt_drift.mean():.3f} m from the log; it should be ~0 because the "
                f"target IS the log"))

            scale_ok = float(np.nanmean(np.abs(e))) < 10.0
            verdicts.append(report(
                "0 the anchor target is in the model's output space", scale_ok,
                f"expert mean|x| = {np.nanmean(np.abs(e)):.3f} vs the model's own "
                f"sample at {float(jnp.mean(jnp.abs(probe_traj))):.3f}; anchor loss "
                f"median {np.median(lpe):.3g} vs {np.median(lps):.3g} on its own sample"))
        lf = make_loss(actor, env, objcfg, state=state, t0=t0, n_exec=n_exec,
                       cond1=cond1, cond2=cond2, mask=mask, expert=expert,
                       alive=alive, rng=rng, anchor_weight=aw)

        # --- claim 1 + 2, on the very first step ------------------------- #
        (l0, aux0), g0 = jax.value_and_grad(lf, has_aux=True)(actor._tparams)
        # Loss ratio is the wrong number to weight by -- what matters is how much
        # of the *gradient* each term contributes. Take each alone at the same
        # point so the two are directly comparable.
        lf_s = make_loss(actor, env, objcfg, state=state, t0=t0, n_exec=n_exec,
                         cond1=cond1, cond2=cond2, mask=mask, expert=expert,
                         alive=alive, rng=rng, anchor_weight=0.0)
        g_safety = jax.grad(lf_s, has_aux=True)(actor._tparams)[0]
        gn_s = float(np.asarray(optax.global_norm(g_safety)))
        gn_all = float(np.asarray(optax.global_norm(g0)))
        anchor_grad_share = abs(gn_all - gn_s) / max(gn_all, 1e-9)
        anchor_share = (aw * float(np.asarray(aux0["anchor"]))
                        / max(float(np.asarray(l0)), 1e-9))
        gn = float(np.asarray(optax.global_norm(g0)))
        leaves = jax.tree_util.tree_leaves(g0)
        finite = all(bool(np.isfinite(np.asarray(x)).all()) for x in leaves)
        if arm == "anchored":
            print(f"[gate] gradient share: safety-only |g| {gn_s:.4e} vs joint "
                  f"{gn_all:.4e} -> anchor is {anchor_grad_share*100:.3f}% of the "
                  f"gradient and {anchor_share*100:.2f}% of the loss")
            print(f"[gate] scale check: safety {float(np.asarray(aux0['safety'])):.4f} "
                  f"vs anchor {float(np.asarray(aux0['anchor'])):.4f} "
                  f"(ratio {float(np.asarray(aux0['anchor']))/max(float(np.asarray(aux0['safety'])),1e-9):.2f}x) "
                  f"-- wildly unequal scales make this a single-term objective")
            verdicts.append(report("1 gradient reaches the parameters",
                                   finite and gn > 0,
                                   f"|grad| = {gn:.4e} over {len(leaves)} leaves, finite={finite}"))

        base_len = arc_length(env.norm_to_world(jnp.transpose(
            actor.sample_world_for(actor._tparams, cond1, cond2, mask, rng), (0, 2, 1))), n_exec)

        hist = []
        for i in range(args.steps):
            m = actor.safety_update(lf)
            hist.append(m)
            if i % max(args.steps // 6, 1) == 0 or i == args.steps - 1:
                print(f"[gate] {arm:11s} step {i:>3}  loss {m['loss']:+.4f}  "
                      f"safety {m['safety']:.4f}  anchor {m['anchor']:.4f}  "
                      f"|g| {m['grad_norm']:.3e}")

        if arm == "anchored":
            verdicts.append(report("2 gradient isolation",
                                   actor.frozen_params_unchanged(frozen_ref),
                                   "frozen tokenizer / base denoiser unchanged after "
                                   f"{args.steps} safety steps"))
            s0, s1 = hist[0]["safety"], float(np.mean([h["safety"] for h in hist[-3:]]))
            verdicts.append(report("3 the safety loss descends", s1 < s0,
                                   f"safety {s0:.4f} -> {s1:.4f} "
                                   f"({100*(s1-s0)/max(abs(s0),1e-9):+.1f}%) over "
                                   f"{args.steps} steps"))

        final_len = arc_length(env.norm_to_world(jnp.transpose(
            actor.sample_world_for(actor._tparams, cond1, cond2, mask, rng), (0, 2, 1))), n_exec)
        final_world = env.norm_to_world(jnp.transpose(
            actor.sample_world_for(actor._tparams, cond1, cond2, mask, rng), (0, 2, 1)))
        drift = expert_drift(final_world, state, t0, n_exec)
        results[arm] = {
            "drift": float(drift.mean()),
            "anchor_share": anchor_share, "anchor_grad_share": anchor_grad_share,
            "grad_norm_safety_only": gn_s, "grad_norm_joint": gn_all,
            "safety_first": hist[0]["safety"], "safety_last": hist[-1]["safety"],
            "grad_norm_first": hist[0]["grad_norm"],
            "arc_base": float(base_len.mean()), "arc_final": float(final_len.mean()),
            "arc_ratio": float(final_len.mean() / max(base_len.mean(), 1e-9)),
            "history": hist,
        }
        print(f"[gate] {arm}: proposed path {base_len.mean():.2f} -> "
              f"{final_len.mean():.2f} m ({results[arm]['arc_ratio']*100:.1f}% of base); "
              f"drift from log {drift.mean():.3f} m")

    # --- claim 4: the anchor holds the policy near the demonstration ----- #
    # The first version asserted that the safety term alone *shortens* the
    # trajectory (stop-in-open-space) and that the anchor prevents it. Measured,
    # no such collapse happens at this budget -- safety-only came back at 101.1%
    # of the base path length, i.e. the optimizer found a STEERING solution, not
    # a stopping one. That is good news about the loss and a wrong prediction in
    # the test, so the claim is restated as what the anchor is actually for: it
    # is a behaviour-cloning term, so it must hold the proposal closer to the
    # human demonstration than the unanchored arm does. Arc length is kept as a
    # collapse floor, since stopping remains the failure mode to watch for.
    ra, rs = results["anchored"]["arc_ratio"], results["safety_only"]["arc_ratio"]
    da, ds = results["anchored"]["drift"], results["safety_only"]["drift"]
    verdicts.append(report(
        "4a no collapse to a standstill", ra > 0.8,
        f"anchored proposed path is {ra*100:.1f}% of base "
        f"(unanchored {rs*100:.1f}%) -- neither arm stops"))
    # Materiality, not sign. The first version asked only `da < ds` and passed on
    # a 1 mm difference out of 8.9 m -- the anchor was inert (0.66% of the
    # objective, 0.05% of the gradient norm) and the gate said it was working.
    # A term that cannot move the outcome by 1% is not constraining anything, and
    # calling that a pass is how the DPPO anchor's 1000:1 misweighting survived.
    rel = 100 * (ds - da) / max(ds, 1e-9)
    verdicts.append(report(
        "4b the anchor materially holds the demonstration", rel >= 1.0,
        f"mean deviation from the human log: {da:.3f} m anchored vs {ds:.3f} m "
        f"unanchored ({rel:+.2f}% closer; needs >= 1% to count as constraining). "
        f"Anchor is {results['anchored']['anchor_share']*100:.2f}% of the objective "
        f"and {results['anchored']['anchor_grad_share']*100:.2f}% of the gradient norm"))

    ok = all(v["passed"] for v in verdicts)
    (out / "gate_safety_grad.json").write_text(json.dumps(
        {"config": {"steps": args.steps, "lr": args.lr, "scenes": indices,
                    "safety_weight": args.safety_weight,
                    "anchor_weight": args.anchor_weight,
                    # The sampler's exploration floor is normalized; positions are
                    # divided by ego_range=100 m, so 0.03 here is 3 m of per-knot
                    # jitter and it dominates `drift`/`arc_*`. Two gate runs are
                    # not comparable without it -- record it.
                    "sigma_floor": args.sigma_floor,
                    "replan_interval_steps": args.replan_interval_steps,
                    "k_trainable": args.k_trainable, "prefix_len": args.prefix_len,
                    "tfrecord": args.tfrecord, **sl_cfg.__dict__},
         "results": results, "gates": verdicts, "all_passed": ok}, indent=2))
    print(f"\n[gate] verdict: {'ALL PASS' if ok else 'FAILED'}")
    print(f"[gate] wrote {out}/gate_safety_grad.json")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
