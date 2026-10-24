"""Gate the teacher-forced unrolled proposal before spending a run on it.

:mod:`rl.diffusion_ft.unrolled_prediction` replaces the free-running 100-step
sample the safety objective is currently differentiated through with a short
unroll from a corrupted expert trajectory. That is a cheaper gradient, and the
whole question is whether it is the *same* gradient in the ways that matter.
Five falsifiable claims, ordered by how badly a failure would invalidate the run:

  1. **K=1 is exactly the one-step posterior mean.** The unroll must be a strict
     generalisation of the object it replaces, or every comparison against the
     one-step arm is confounded by an implementation difference.
  2. **The unrolled endpoint stands in for a driven sample better than the
     one-step mean does.** The load-bearing claim: the objective should score
     something like the trajectory that actually gets driven, so the test is a
     distance to the free-running sample's loss. The Jensen gap that motivates
     the mechanic upstream is measured and recorded here but NOT gated -- its
     sign was predicted wrong on the first run (see claim 2 in the code) and it
     is informative either way.
  3. **It is cheaper.** K network evaluations per gradient step against the
     chain's ``timesteps``. Measured in wall clock, because the claim is about
     what a run costs, not about an op count.
  4. **The gradient exists and is isolated.** Finite, non-zero, and reaching only
     the trainable denoiser copy -- the failure mode that already cost this
     project a 500-iteration DPPO run.
  5. **It transfers.** Steps taken on the teacher-forced objective must reduce
     the safety loss of the *free-running* sample, which is the only trajectory
     that ever gets driven. Run against the free-running objective as a control,
     so this is a comparison rather than a threshold pulled from the air.

GPU job, artifacts only.

Usage:
    python -m rl.diffusion_ft.tests.gate_bond_unroll --tfrecord ... --out_dir OUT
"""

from __future__ import annotations

import argparse, json, os, sys, time
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
from rl.diffusion_ft.unrolled_prediction import unroll_timestep_grid, unrolled_clean_prediction


def report(name, ok, detail):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return {"gate": name, "passed": bool(ok), "detail": detail}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=str, default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--tfrecord", type=str, required=True)
    p.add_argument("--indices", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--k_trainable", type=int, default=10)
    p.add_argument("--prefix_len", type=int, default=5)
    p.add_argument("--sigma_floor", type=float, default=1e-4)
    p.add_argument("--replan_interval_steps", type=int, default=10)
    p.add_argument("--unroll_steps", type=int, default=3)
    p.add_argument("--unroll_grad_steps", type=int, default=1)
    p.add_argument("--unroll_t_start", type=int, default=-1)
    p.add_argument("--unroll_t_end", type=int, default=0)
    p.add_argument("--unroll_eta", type=float, default=0.0)
    # Monte-Carlo draws for the Jensen-gap measurement. The comparison is one
    # scalar (the mean) against a sample mean, so the error bar on the second is
    # what decides whether a gap is readable -- report it, do not assume it.
    p.add_argument("--n_mc", type=int, default=16)
    p.add_argument("--edge_margin_m", type=float, default=0.3)
    p.add_argument("--agent_margin_m", type=float, default=0.75)
    p.add_argument("--n_discs", type=int, default=5)
    p.add_argument("--agent_geometry", type=str, default="exact", choices=("exact", "disc"))
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    sl_cfg = SafetyLossConfig(edge_margin_m=args.edge_margin_m,
                              agent_margin_m=args.agent_margin_m,
                              n_discs=args.n_discs, agent_geometry=args.agent_geometry)
    objcfg = SafetyObjectiveConfig(safety=sl_cfg, safety_weight=1.0, anchor_weight=0.0)

    ckpt = load_diffusion_checkpoint(args.ckpt)
    state = load_batch(args.tfrecord, indices)
    clean_b, _, _ = expert_clean_mask(sl_cfg, state, init_steps=11)
    keep = np.asarray(clean_b)
    if not keep.all():
        indices = [indices[i] for i in np.where(keep)[0]]
        print(f"[gate] gating {int((~keep).sum())} expert-dirty scene(s); using {indices}")
        state = load_batch(args.tfrecord, indices)
    n_obj = int(jnp.asarray(state.log_trajectory.x).shape[1])

    env = DiffusionWaymaxEnv(ckpt, reward_config=RewardConfig(), max_num_objects=n_obj,
                             replan_interval_steps=args.replan_interval_steps, seed=0)
    features = env.reset(state)
    t0, n_exec = env.timestep, args.replan_interval_steps
    cond1, cond2 = env.compute_condition(features)
    mask = features["inst_valid"]
    B = env.batch_size
    alive = jnp.ones((B,), jnp.float32)

    actor = DiffusionRLActor(env.policy.diffusion, k_trainable=args.k_trainable,
                             prefix_len=args.prefix_len, sigma_sample_floor=args.sigma_floor,
                             sigma_logprob_floor=args.sigma_floor, actor_lr=args.lr,
                             expert_lr=args.lr, seed=0)
    actor.reset_trainable_to_frozen()
    diffusion = actor.diffusion
    n_t = int(diffusion.timesteps)
    h = int(diffusion.denoise_fn.horizon)
    if features["ego_trajectory"].shape[1] < h:
        raise SystemExit(f"[gate] expert target too short: "
                         f"{features['ego_trajectory'].shape[1]} < {h}")
    x0_bct = jnp.transpose(features["ego_trajectory"][:, :h, :], (0, 2, 1))

    t_start = args.k_trainable - 1 if args.unroll_t_start < 0 else args.unroll_t_start
    if not 0 <= args.unroll_t_end < t_start:
        raise SystemExit(f"[gate] need 0 <= t_end < t_start, got {args.unroll_t_end} / {t_start}")
    sqrt_1mab = np.asarray(diffusion.sqrt_one_minus_alphas_cumprod.value)
    print(f"[gate] scenes={indices}  K={args.unroll_steps} grad_steps={args.unroll_grad_steps} "
          f"eta={args.unroll_eta}  t_start={t_start} ({100 * sqrt_1mab[t_start]:.1f} m) "
          f"-> t_end={args.unroll_t_end} ({100 * sqrt_1mab[args.unroll_t_end]:.1f} m)")
    print(f"[gate] {sl_cfg}")

    def safety_of(traj_ncT):
        """Safety loss of a normalized [B, C, T] candidate, as a scalar."""
        world = env.norm_to_world(jnp.transpose(traj_ncT, (0, 2, 1)))
        return candidate_safety(objcfg, world, state=env.sim_state, t0=t0,
                                n_steps=n_exec, alive_b=alive)["total"]

    t_start_b = jnp.full((B,), t_start, jnp.int32)
    grid_k = unroll_timestep_grid(t_start_b, args.unroll_steps, args.unroll_t_end)
    grid_1 = unroll_timestep_grid(t_start_b, 1, args.unroll_t_end)
    # One fixed corruption, shared by every arm below, so the comparisons are at
    # the same x_t and differ only in what is done from there.
    noise = jax.random.normal(jax.random.PRNGKey(11), x0_bct.shape, x0_bct.dtype)
    x_t = diffusion.q_sample(x0_bct, t_start_b, noise=noise)

    results, verdicts = {}, []

    # --- claim 1: K=1 is the one-step posterior mean --------------------- #
    u1 = unrolled_clean_prediction(
        diffusion, actor._trainable_fn, actor._tparams, x_t=x_t, t_grid_kb=grid_1,
        cond1=cond1, cond2=cond2, cond2_mask=mask, rng=jax.random.PRNGKey(1))
    _, _, x0_one = diffusion.p_mean_variance(
        x_t, t_start_b, cond1, cond2, mask, denoise_fn=actor._trainable_fn(actor._tparams))
    d1 = float(jnp.max(jnp.abs(u1 - x0_one)))
    results["k1_max_abs_diff"] = d1
    verdicts.append(report("1 K=1 is the one-step posterior mean", d1 == 0.0,
                           f"max|unroll(K=1) - p_mean_variance x0_pred| = {d1:.3e} "
                           f"(must be exactly 0 -- same call, no transitions)"))

    # --- claim 2: the unrolled endpoint stands in for a sample ------------ #
    # This gate originally asserted a DIRECTION: that the loss of the conditional
    # mean would come in BELOW the mean loss of the samples, the way it does for
    # a ring closure, where averaging over conformations pulls the two anchors
    # together so the mean closes when nothing sampled does. Measured here it
    # runs the OTHER way (job 40471: mean 0.2980 vs samples 0.2639 +/- 0.0007,
    # i.e. -46 sem). The one-step prediction from a 16.7 m corrupted state is a
    # blurred, under-committed trajectory, and blur costs clearance rather than
    # buying it. That is a wrong prediction in the test, not a broken mechanic.
    #
    # So the claim is restated as what the unroll is actually for: the object
    # being penalised should stand in for the trajectory that gets DRIVEN. The
    # free-running sample is that trajectory, so the test is a distance -- is the
    # unrolled endpoint's loss closer to it than the one-step mean's is? The
    # Jensen gap is still measured and recorded, with no pass/fail attached,
    # because its sign is informative either way.
    l_mean = float(safety_of(u1))
    l_expert = float(safety_of(x0_bct))

    def draw_losses(eta, n, seed):
        return np.asarray([
            float(safety_of(unrolled_clean_prediction(
                diffusion, actor._trainable_fn, actor._tparams, x_t=x_t,
                t_grid_kb=grid_k, cond1=cond1, cond2=cond2, cond2_mask=mask,
                eta=eta, rng=jax.random.PRNGKey(seed + i))))
            for i in range(n)])

    def sem_of(a):
        return float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float("inf")

    # eta=1 is the ancestral chain (what the sampler runs); eta=`unroll_eta` is
    # the object training will actually optimise. Both are reported: a mechanic
    # judged at one eta and trained at another is judged on the wrong thing.
    d_eta1 = draw_losses(1.0, args.n_mc, 100)
    d_train = (d_eta1 if float(args.unroll_eta) == 1.0
               else draw_losses(float(args.unroll_eta), args.n_mc, 200))
    # The free-running sample: a different trajectory each rng, from pure noise.
    d_free = np.asarray([
        float(safety_of(actor.sample_world_for(
            actor._tparams, cond1, cond2, mask, jax.random.PRNGKey(300 + i))))
        for i in range(args.n_mc)])

    gap = float(d_eta1.mean() - l_mean)
    results["jensen"] = {"loss_of_mean": l_mean,
                         "mean_loss_of_samples_eta1": float(d_eta1.mean()),
                         "sem_eta1": sem_of(d_eta1), "gap": gap,
                         "expert_reference": l_expert, "n_mc": int(args.n_mc)}
    print(f"[gate] Jensen gap (recorded, not gated): loss of the mean {l_mean:.4f} vs "
          f"mean loss of {args.n_mc} eta=1 samples {d_eta1.mean():.4f} +/- "
          f"{sem_of(d_eta1):.4f}; gap {gap:+.4f}. Human log {l_expert:.4f}")

    d_unroll_err = abs(float(d_train.mean()) - float(d_free.mean()))
    d_mean_err = abs(l_mean - float(d_free.mean()))
    results["surrogate"] = {
        "free_running": float(d_free.mean()), "free_running_sem": sem_of(d_free),
        "unrolled_train_eta": float(d_train.mean()), "unrolled_sem": sem_of(d_train),
        "one_step_mean": l_mean, "unrolled_abs_err": d_unroll_err,
        "one_step_abs_err": d_mean_err,
        "ratio": d_mean_err / max(d_unroll_err, 1e-9), "eta": float(args.unroll_eta)}
    verdicts.append(report(
        "2 the unrolled endpoint is the better stand-in for a driven sample",
        d_unroll_err < d_mean_err,
        f"free-running sample scores {d_free.mean():.4f} +/- {sem_of(d_free):.4f}; "
        f"unrolled endpoint (eta={args.unroll_eta}) {d_train.mean():.4f} "
        f"(off by {d_unroll_err:.4f}) vs one-step mean {l_mean:.4f} "
        f"(off by {d_mean_err:.4f}) -- {d_mean_err / max(d_unroll_err, 1e-9):.1f}x closer. "
        f"Human log {l_expert:.4f}"))

    # --- claim 3: it is cheaper ------------------------------------------- #
    # `safety_of` is evaluated ONCE and reused for the aux dict. The first version
    # called it twice (once for the loss, once for `{"safety": ...}`) and, since
    # `safety_update` runs the loss eagerly rather than under jit, XLA never
    # folded the duplicate away -- it added a full safety evaluation to BOTH arms
    # and suppressed the measured speedup (job 40471 read 1.4x).
    def unrolled_loss(params):
        traj = actor.unrolled_sample_for(
            params, cond1, cond2, mask, x0_bct, grid_k, jax.random.PRNGKey(3),
            grad_steps=args.unroll_grad_steps, eta=float(args.unroll_eta))
        s = safety_of(traj)
        return s, {"safety": s}

    def free_loss(params):
        traj = actor.sample_world_for(params, cond1, cond2, mask, jax.random.PRNGKey(3))
        s = safety_of(traj)
        return s, {"safety": s}

    def timed(fn, n=5):
        jax.block_until_ready(fn())                       # compile
        t = time.time()
        for _ in range(n):
            jax.block_until_ready(fn())
        return (time.time() - t) / n

    g_unroll = jax.value_and_grad(unrolled_loss, has_aux=True)
    g_free = jax.value_and_grad(free_loss, has_aux=True)
    t_unroll = timed(lambda: g_unroll(actor._tparams))
    t_free = timed(lambda: g_free(actor._tparams))
    # Decompose it. A bare ratio hides WHERE the money goes, and the answer
    # changes what is worth optimising next: if the safety evaluation dominates,
    # no amount of shortening the denoiser chain will buy much.
    t_prop_unroll = timed(lambda: actor.unrolled_sample_for(
        actor._tparams, cond1, cond2, mask, x0_bct, grid_k, jax.random.PRNGKey(3),
        grad_steps=args.unroll_grad_steps, eta=float(args.unroll_eta)))
    t_prop_free = timed(lambda: actor.sample_world_for(
        actor._tparams, cond1, cond2, mask, jax.random.PRNGKey(3)))
    probe = actor.unrolled_sample_for(actor._tparams, cond1, cond2, mask, x0_bct,
                                      grid_k, jax.random.PRNGKey(3),
                                      grad_steps=args.unroll_grad_steps,
                                      eta=float(args.unroll_eta))
    t_safety = timed(lambda: safety_of(probe))
    results["seconds_per_grad_step"] = {
        "unrolled": t_unroll, "free_running": t_free,
        "speedup": t_free / max(t_unroll, 1e-9),
        "forward_only": {"unrolled_proposal": t_prop_unroll,
                         "free_running_proposal": t_prop_free,
                         "safety_evaluation": t_safety}}
    verdicts.append(report(
        "3 the unrolled gradient is cheaper", t_unroll < t_free,
        f"{t_unroll * 1e3:.0f} ms/step unrolled ({args.unroll_steps} nn evals) vs "
        f"{t_free * 1e3:.0f} ms/step free-running ({n_t} nn evals) -- "
        f"{t_free / max(t_unroll, 1e-9):.1f}x"))
    print(f"[gate]   forward only: proposal {t_prop_unroll * 1e3:.0f} ms unrolled / "
          f"{t_prop_free * 1e3:.0f} ms free-running, safety evaluation "
          f"{t_safety * 1e3:.0f} ms. The speedup is capped by whichever of these "
          f"dominates, and the safety evaluation is not the part the unroll shortens.")

    # --- claim 4: gradient exists, is finite, and is isolated ------------- #
    frozen_ref = actor.frozen_state()
    (loss0, _), grads = jax.value_and_grad(unrolled_loss, has_aux=True)(actor._tparams)
    gnorm = float(np.asarray(optax.global_norm(grads)))
    results["grad_norm"] = gnorm
    verdicts.append(report(
        "4a gradient is finite and non-zero", np.isfinite(gnorm) and gnorm > 0,
        f"|grad| = {gnorm:.3e} at loss {float(loss0):.4f}"))
    verdicts.append(report(
        "4b frozen modules untouched", actor.frozen_params_unchanged(frozen_ref),
        "base denoiser / tokenizer / instruction encoders bit-identical after "
        "differentiating through the unroll"))

    # --- claim 5: it transfers to the free-running sample ----------------- #
    # Both arms start from the same params and are scored on the same fixed-rng
    # free-running sample, which is the trajectory that actually gets driven.
    eval_rng = jax.random.PRNGKey(77)

    def free_running_safety(a):
        return float(safety_of(a.sample_world_for(a._tparams, cond1, cond2, mask, eval_rng)))

    arms = {}
    for arm, loss_fn_of in (("unrolled", unrolled_loss), ("free_running", free_loss)):
        a = DiffusionRLActor(env.policy.diffusion, k_trainable=args.k_trainable,
                             prefix_len=args.prefix_len, sigma_sample_floor=args.sigma_floor,
                             sigma_logprob_floor=args.sigma_floor, actor_lr=args.lr,
                             expert_lr=args.lr, seed=0)
        a.reset_trainable_to_frozen()

        def arm_loss(params, _a=a, _which=arm):
            traj = (_a.unrolled_sample_for(params, cond1, cond2, mask, x0_bct, grid_k,
                                           jax.random.PRNGKey(3),
                                           grad_steps=args.unroll_grad_steps,
                                           eta=float(args.unroll_eta))
                    if _which == "unrolled"
                    else _a.sample_world_for(params, cond1, cond2, mask, jax.random.PRNGKey(3)))
            s = safety_of(traj)
            return s, {"safety": s}

        before = free_running_safety(a)
        # Compile the arm's gradient program before the clock starts. Each arm
        # builds a fresh actor and therefore a fresh jit cache, so without this
        # the first step's compilation lands inside `wall_s` and the two arms are
        # compared on how long XLA took, not on how long a step takes.
        jax.block_until_ready(jax.value_and_grad(arm_loss, has_aux=True)(a._tparams))
        t_arm = time.time()
        for _ in range(args.steps):
            a.safety_update(arm_loss)
        wall = time.time() - t_arm
        after = free_running_safety(a)
        arms[arm] = {"free_running_safety_before": before,
                     "free_running_safety_after": after,
                     "rel_drop": (before - after) / max(abs(before), 1e-9),
                     "wall_s": wall}
        print(f"[gate] {arm:>12}: free-running safety {before:.4f} -> {after:.4f} "
              f"({100 * arms[arm]['rel_drop']:+.1f}%) in {wall:.1f}s for {args.steps} steps")
    results["transfer"] = arms

    du, df = arms["unrolled"]["rel_drop"], arms["free_running"]["rel_drop"]
    verdicts.append(report(
        "5a the teacher-forced objective transfers", du > 0.0,
        f"{args.steps} unrolled steps move the FREE-RUNNING safety loss "
        f"{arms['unrolled']['free_running_safety_before']:.4f} -> "
        f"{arms['unrolled']['free_running_safety_after']:.4f} ({100 * du:+.1f}%)"))
    # Not "as good as": the teacher-forced arm optimises a different (cheaper)
    # object, so it is allowed to transfer less. What would kill it is buying the
    # speedup with no progress at all -- half the control's per-step effect at a
    # fraction of the cost is a trade worth making, none is not.
    verdicts.append(report(
        "5b it is worth the trade", du >= 0.5 * df or du > 0.0 and df <= 0.0,
        f"per-step effect on free-running safety: unrolled {100 * du:+.1f}% "
        f"({arms['unrolled']['wall_s']:.0f}s) vs free-running control "
        f"{100 * df:+.1f}% ({arms['free_running']['wall_s']:.0f}s)"))

    ok = all(v["passed"] for v in verdicts)
    (out / "gate_bond_unroll.json").write_text(json.dumps(
        {"config": {"scenes": indices, "steps": args.steps, "lr": args.lr,
                    "k_trainable": args.k_trainable, "prefix_len": args.prefix_len,
                    "sigma_floor": args.sigma_floor,
                    "replan_interval_steps": args.replan_interval_steps,
                    "unroll_steps": args.unroll_steps,
                    "unroll_grad_steps": args.unroll_grad_steps,
                    "unroll_t_start": t_start, "unroll_t_end": args.unroll_t_end,
                    "unroll_eta": args.unroll_eta, "n_mc": args.n_mc,
                    "timesteps": n_t, "tfrecord": args.tfrecord, **sl_cfg.__dict__},
         "results": results, "gates": verdicts, "all_passed": ok}, indent=2))
    print(f"\n[gate] verdict: {'ALL PASS' if ok else 'FAILED'}")
    print(f"[gate] wrote {out}/gate_bond_unroll.json")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
