"""Analytic safety fine-tuning of the base diffusion policy.

Unlike DPPO (:mod:`rl.diffusion_ft.train_dppo`), nothing here is a policy
gradient: :mod:`rl.diffusion_ft.safety_objective` is differentiable in the
denoiser parameters, so the update is an ordinary gradient step on a geometric
loss. No rollouts-for-credit, no advantages, no critic, no trust region.

**Where the gradient enters the chain.** The loss is always evaluated on the
*finished* candidate (the sample at diffusion step ``t=0``), but it backpropagates
only through the last ``k_trainable`` reverse steps -- the denoiser's behaviour at
timesteps ``k_trainable-1 .. 0``. With the checkpoint's 100-step cosine schedule
that window is the low-noise end, which is what sets the precise geometry of a
turn; the high-noise steps choose which coarse manoeuvre to make and are left
frozen. Residual forward noise ``sqrt(1-alphabar_t)`` on the position channels,
in metres (they are normalized by ``ego_range=100``)::

    t      0     1     2     3     5     9    19    49    99
    m    2.5   4.2   5.8   7.4  10.5  16.7  31.8  71.1  100.0

So ``k_trainable=10`` already trains only ``t<=9`` (<=16.7 m); ``k_trainable=3``
tightens it to ``t<=2`` (<=5.8 m). Lowering it also makes each step cheaper.

**Which states.** Free-running, not teacher-forced: each iteration rolls the
*current* policy out closed-loop with no gradients, records the replan contexts it
actually reaches, and only then differentiates them. Scoring the log's own states
would only ever teach the policy to be safe from states a human visited -- it is
the policy's own drift that needs recovering from. Collecting first and
differentiating second keeps the batch stationary: the policy that generated
the states is the policy being differentiated at those states.

**Two ways to propose the trajectory that gets scored.** ``--unroll_steps 0``
(default) is the free-running path described above: denoise from pure noise
through the whole 100-step chain, differentiate the last ``k_trainable``.
``--unroll_steps K`` switches to the *teacher-forced unroll* ported from
Proteina-Complexa's ``bondunroll`` (see
:mod:`rl.diffusion_ft.unrolled_prediction`): corrupt the expert future to a
random low-noise timestep, unroll ``K`` reverse steps back down, and score that
endpoint. One gradient step then costs ``K`` network evaluations instead of 100,
which until 2026-09-14 did NOT make it faster: the unroll ran through
``_loss_for``'s un-jitted closure and re-traced the whole graph once per context
per iteration, measuring 34.5 s/iter against the jitted free-running path's 5.96.
It now shares ``_free_core``; ``--unroll_eager`` restores the old closure as the
equivalence reference (``tests/test_unroll_jit.py`` pins the key streams equal).
The arithmetic below was always right, but read any pre-09-14 timing as measuring
tracing overhead rather than the method,
and the state the denoiser sees is anchored to ground truth rather than to the
policy's own high-noise prefix. ``K=1`` degenerates to the one-step posterior
mean, which is the Jensen-gap control worth running as an ablation arm.

**A third proposal: replay the sampler's own transitions.** ``--tb_weight``
turns on :mod:`rl.diffusion_ft.transition_buffer`. The collection rollout already
computes, and until 2026-09-14 discarded, the ``(x_t, t)`` it passes through on
each of the ``k_trainable`` late steps; storing one per row per replan costs a
device-to-host copy and no denoiser work. The safety loss is then charged on the
clean prediction reached from a STORED state -- ``--tb_unroll_steps 1`` scores
``x0_hat(x_t, t)`` directly (one network evaluation), ``K > 1`` unrolls ``K-1``
DDIM steps first and scores the final clean prediction, and ``--tb_grad_steps 1``
keeps the gradient on exactly one transition either way. This is the teacher-forced
unroll with its one acknowledged defect removed: the state is not
``q_sample(x0_expert, t)`` but one the deployed sampler actually visited. What it
adds instead is staleness -- a replayed state was produced by an older policy --
bounded by ``--tb_capacity`` and ``--tb_max_uses`` and reported per iteration as
``tb_age``. ``--safety_weight 0 --tb_weight 1`` is the pure-replay arm; the
rollout still runs, because it is what fills the buffer.

**Where the states come from is a separate axis.** ``--anchor_states policy``
(default) collects contexts by rolling the *policy* out closed-loop, as described
above. ``--anchor_states log`` drives the collection rollout with the human's own
log window instead, which removes every sample from the iteration (with
``--unroll_steps`` on, nothing runs the 100-step chain at all) at the cost of only
ever supervising states a human visited.

**One term by default.** ``--anchor_weight`` defaults to 0. The denoising anchor
was measured (job 40324) at 0.06% of the gradient norm and flat under a 20x
increase in drift from the log, so including it costs compute and buys assurance
it does not provide. That leaves the objective single-term and therefore capable
of the degeneracy its own docstring warns about, so the held-out eval here
deliberately reports drift/arc/speed alongside offroad/overlap: if the policy buys
clearance by wandering off the demonstrated line, that shows up as falling
offroad with rising drift, and you should stop the run.

**Everything measured about the anchor before 2026-09-03 was measured on a bug.**
Its rng was the literal ``jax.random.PRNGKey(7)`` on every gradient step, and
that one key seeds both the diffusion timestep and the noise, so the term was not
the denoising loss but a single frozen Monte-Carlo slice of it, descended 1600
times. It explains the 09-02 sweep's central anomaly -- that the ``sw=0, aw=1``
control, a plain BC continuation that optimizes nothing new, was the *worst* of
six arms (success-heldout 0.995 -> 0.639, failure-heldout 0.276 -> 0.132, and its
own rollout off-road rate climbing 0.00 -> 0.25 over 139 iterations). It also
explains job 40324's "0.06% of the gradient norm": one memorized draw stops
producing gradient. Re-measure both before trusting either number, and read the
``anchor_states`` policy-vs-log question as still open rather than as answered by
that sweep. See ``_loss_for``.

**How far the policy is allowed to travel.** The update is plain Adam with a
``clip_by_global_norm`` in front of it, and that pairing does less than it looks:
Adam divides by its own running RMS, so a uniform pre-scale cancels and the update
is ``~lr`` per parameter *whatever the gradient norm is*. Gradient magnitude
chooses the direction, not the distance. Distance is set by the step COUNT, and
nothing else here bounds it -- ``safety_update`` deliberately skips the DPPO trust
region (correctly: there is no importance ratio to protect) but that region was
also the only leash on how far the weights moved.

Measured on the 09-02 ratio sweep, with 31.7M parameters at ``lr=1e-5``: 200
iterations x 8 replan contexts = 1600 steps, whose *aligned* displacement would be
``lr*N*sqrt(P) = 90`` and whose random-walk displacement is ``lr*sqrt(N*P) = 2.25``.
The arms' final weights differed pairwise by 1.35-5.02 -- the noise scale, a few
percent of the signal scale. In metres that walk is worth ~0.45m of trajectory
drift (the anchor-only control drifted 0.80 -> 1.25m while optimizing nothing),
which is the same order as ``edge_margin`` (0.3m) and the failure set's median
off-road excursion (0.77m). So the noise moved the car about as far as the
violation being fixed, which is why threshold metrics read as a lottery.

Two defaults follow, both changed on 2026-09-03:

* ``--grad_accum 0`` averages the gradient over every replan context and takes
  ONE step per iteration: 8x fewer steps (2.8x shorter walk) on 8x the data each.
  ``--grad_accum 1`` restores the old per-context schedule for comparison.
* evals draw from ``--eval_seed``, not from the training stream, and average
  ``--eval_repeats`` of them; the frozen base is scored once on the same seeds and
  reported as ``base_*`` so every row is a paired comparison.

Usage::

    python -m rl.diffusion_ft.train_safety --out_dir DIR \\
        --shard_glob '/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/*' \\
        --iters 500 --scenes_per_iter 16 --k_trainable 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from rl.diffusion_ft.checkpoint import DEFAULT_CHECKPOINT_PATH, load_diffusion_checkpoint
from data.preprocess import WORLD_DT_METHODS
from rl.diffusion_ft.env import EGO_VEL_SOURCES, DiffusionWaymaxEnv
from rl.diffusion_ft.reward import RewardConfig
from rl.diffusion_ft.rl_actor import DiffusionRLActor
from rl.diffusion_ft.safety_loss import SafetyLossConfig, expert_clean_mask
from rl.diffusion_ft.safety_objective import SafetyObjectiveConfig, candidate_safety
from rl.diffusion_ft.sim_agents import SIM_AGENT_CHOICES
from rl.diffusion_ft.scene_sampler import SceneSampler
from rl.diffusion_ft.transition_buffer import TransitionBuffer
from rl.diffusion_ft.unrolled_prediction import unroll_timestep_grid
from rl.diffusion_ft.train_dppo import load_batch


def _kv_ratio(spec: str) -> dict[str, float]:
    """``'failure_train=1,success_train=1'`` -> ordered ``{split: weight}``; '' -> {}."""
    out: dict[str, float] = {}
    for tok in [t.strip() for t in str(spec or "").split(",") if t.strip()]:
        if "=" not in tok:
            raise SystemExit(f"--repair_split_ratio: '{tok}' is not split=weight")
        k, v = tok.split("=", 1)
        if float(v) < 0:
            raise SystemExit(f"--repair_split_ratio: negative weight in '{tok}'")
        out[k.strip()] = float(v)
    if out and sum(out.values()) <= 0:
        raise SystemExit("--repair_split_ratio: weights sum to 0")
    return out


# --------------------------------------------------------------------------- #
class ReplanContext:
    """Everything needed to re-differentiate one replan after the env moved on.

    ``candidate_safety`` reads only ``log_trajectory``, ``object_metadata`` and
    ``roadgraph_points`` off the state -- all constant for the whole episode -- so
    the episode's state is held once by the caller and only ``t0`` varies here.
    """

    __slots__ = ("t0", "cond1", "cond2", "mask", "aux", "alive", "n_steps",
                 "n_exec", "expert_target", "x_prefix", "key_late",
                 "safety_alive", "preserve_target", "preserve_w")

    def __init__(self, t0, cond1, cond2, mask, aux, alive, n_steps, n_exec=None):
        self.t0, self.cond1, self.cond2 = int(t0), cond1, cond2
        self.mask, self.aux, self.alive, self.n_steps = mask, aux, alive, int(n_steps)
        # World steps the env will actually DRIVE, which is no longer the same as
        # `n_steps` (the scored window) once --safety_window widens the loss past
        # the replan interval. Kept for the viz/`done_at` bookkeeping, which must
        # track execution rather than scoring.
        self.n_exec = int(n_steps if n_exec is None else n_exec)
        # Attached after the rollout by whoever needs it (the anchor term, the
        # teacher-forced unroll). `__slots__` means an unset attribute raises
        # rather than reading None, so bind it here.
        self.expert_target = None
        # `--reuse_prefix`: the collection sample's frozen-prefix output x_{k-1}
        # and its late-chain key, so the gradient pass replays only the
        # k_trainable steps of the trajectory that was actually executed.
        self.x_prefix = None
        self.key_late = None
        # `--safety_rows unsolved`: [B] mask the SAFETY term is charged on, set
        # after the rollout once each row's outcome is known. None = `alive`
        # (every earlier run), so the default path is bit-for-bit unchanged.
        self.safety_alive = None
        # `--preserve_weight`: the frozen base's own world-coordinate
        # continuation from this context's prefix ([B, W, 5]) and the [B] row
        # weight (alive AND the base solved the episode). Preservation contexts
        # only; see `SafetyFineTuner.collect_preserve_contexts`.
        self.preserve_target = None
        self.preserve_w = None


def preserve_loss(world_bw5, target_bw5, w_b, *, n, s_pos=1.0, s_yaw=0.1, s_speed=1.0):
    """World-space distance between a candidate and the base's own continuation.

    Both are ``[B, W, 5] = (x, y, yaw, vx, vy)`` with index 0 the shared anchor, so
    steps ``1..n`` are scored. Per step: squared position error / ``s_pos**2`` +
    squared WRAPPED heading error / ``s_yaw**2`` + squared speed error /
    ``s_speed**2``, averaged over steps, then a ``w_b``-weighted mean over rows
    (rows with zero weight -- the base failed that episode, or the scene was
    already absorbed -- contribute nothing). Zero, with zero gradient, when the
    candidate equals the target. Returns ``(loss, aux)``; ``aux["preserve_rms_m"]`` is
    the weighted RMS position deviation in metres, the number to read.
    """
    w = world_bw5[:, 1:1 + n, :]
    g = jax.lax.stop_gradient(target_bw5[:, 1:1 + n, :])
    dpos2 = jnp.sum((w[..., :2] - g[..., :2]) ** 2, axis=-1)
    d = w[..., 2] - g[..., 2]
    dyaw = jnp.arctan2(jnp.sin(d), jnp.cos(d))
    # +eps inside the sqrt: a stopped car has zero speed and d|v|/dv is undefined.
    sp = jnp.sqrt(jnp.sum(w[..., 3:5] ** 2, axis=-1) + 1e-6)
    sg = jnp.sqrt(jnp.sum(g[..., 3:5] ** 2, axis=-1) + 1e-6)
    per_step = dpos2 / s_pos ** 2 + dyaw ** 2 / s_yaw ** 2 + (sp - sg) ** 2 / s_speed ** 2
    per_b = jnp.mean(per_step, axis=1)
    wb = jnp.asarray(w_b, per_b.dtype)
    den = jnp.maximum(jnp.sum(wb), 1.0)
    loss = jnp.sum(per_b * wb) / den
    rms = jnp.sqrt(jnp.sum(jnp.mean(dpos2, axis=1) * wb) / den)
    return loss, {"preserve": loss, "preserve_rms_m": rms, "preserve_rows": jnp.sum(wb)}


class SafetyFineTuner:
    def __init__(self, args):
        self.args = args
        self.ckpt = load_diffusion_checkpoint(args.ckpt)
        self.sl_cfg = SafetyLossConfig(
            edge_margin_m=args.edge_margin, agent_margin_m=args.agent_margin,
            edge_tau_m=args.edge_tau, agent_tau_m=args.agent_tau,
            n_discs=args.n_discs, agent_geometry=args.agent_geometry,
            barrier_shape=args.barrier_shape, edge_geometry=args.edge_geometry)
        self.objcfg = SafetyObjectiveConfig(
            safety=self.sl_cfg, safety_weight=args.safety_weight,
            anchor_weight=args.anchor_weight)

        # Read the resume point BEFORE the sampler exists: it seeds the batch
        # stream. Without this a resumed run re-draws the SAME sequence of
        # batches it already trained on -- invisible in the curve, and over a
        # multi-day run with several restarts it means the policy sees the first
        # few hundred batches many times and the rest never.
        self.start_iter = self._resume_iter()
        self.train_pool_ids = None
        self.sampler = None
        self.train_pool_groups = None
        self.train_pool_names = []
        self._success_groups = []
        # Per-scene [B] weight for the base anchor; None = anchor every row
        # (no groups configured, e.g. --indices mode or an unsplit corpus).
        self._anchor_mask = None
        if args.shard_glob:
            self.train_pool_ids = self._resolve_train_pool()
            groups, weights = self._resolve_group_ratio()
            self.sampler = SceneSampler(
                args.shard_glob, batch_size=args.scenes_per_iter,
                eval_size=args.eval_scenes,
                seed=int(args.data_seed) + 100003 * int(self.start_iter),
                prefetch=bool(args.prefetch), index_dir=args.index_dir,
                scene_ids=self.train_pool_ids,
                scene_groups=groups, group_weights=weights)
            print(f"[safety] {self.sampler.describe()}", flush=True)
            probe = self.sampler.eval_batch() if args.eval_scenes else self.sampler.next_train()
        else:
            self.fixed_indices = [int(i) for i in args.indices.split(",") if i.strip()]
            probe = load_batch(args.tfrecord, self.fixed_indices)
        n_obj = int(jnp.asarray(probe.log_trajectory.x).shape[1])

        self.env = DiffusionWaymaxEnv(
            self.ckpt, reward_config=RewardConfig(), max_num_objects=n_obj,
            replan_interval_steps=args.replan_interval_steps, seed=args.seed,
            sim_agents=args.sim_agents, ego_vel_source=args.ego_vel_source,
            world_dt_method=args.world_dt_method)
        # Read at TRACE time inside `_free_core`, so it must be settled before
        # the first gradient and never change afterwards -- one executable is
        # compiled against one neighbour source.
        self._neighbor_from_sim = (args.neighbor_source == "sim")
        if self._neighbor_from_sim and args.sim_agents == "none":
            print("[safety] NOTE --neighbor_source=sim under --sim_agents=none: "
                  "log-replayed NPCs put the same motion in both sources, so "
                  "this is a no-op that only costs a slice. It is allowed as an "
                  "equivalence control.", flush=True)
        if self._neighbor_from_sim and float(getattr(args, "tb_weight", 0.0)) > 0:
            # `buffer_grads` reloads its scenes by id at timestep 0, so their
            # `sim_trajectory` holds nothing -- the NPC motion the transition was
            # recorded under is not stored. The replay term would silently keep
            # reading the log while the on-policy term reads the sim, and the
            # objective would be two different objectives added together.
            raise SystemExit(
                "--neighbor_source=sim is incompatible with --tb_weight>0: the "
                "transition buffer reloads scenes at timestep 0 and cannot "
                "reconstruct the NPC motion its transitions were recorded "
                "under. Run one or the other.")
        print(f"[safety] sim_agents={self.env.sim_agents} "
              f"neighbor_source={args.neighbor_source} "
              f"ego_vel_source={self.env.ego_vel_source} "
              f"world_dt_method={self.env.world_dt_method}", flush=True)
        # DDIM striding of the frozen prefix, resolved once. `None` = full chain.
        # Two values, because the training passes and the eval measure different
        # things: collection/gradient may run an approximation of the policy,
        # but an eval that strides too would confound a curve movement between
        # the weights and the sampler. `sample_with_trace` requires
        # k_trainable <= S <= timesteps; check here so a bad flag fails at startup
        # instead of inside the first traced rollout.
        n_t = int(self.env.policy.diffusion.timesteps)
        def _steps(v, name):
            v = int(v)
            if v <= 0:
                return None
            if not (int(args.k_trainable) <= v <= n_t):
                raise SystemExit(
                    f"--{name}={v} must lie in [k_trainable={args.k_trainable}, "
                    f"timesteps={n_t}] (0 = full chain)")
            return v
        self._steps_train = _steps(getattr(args, "collect_steps", 0), "collect_steps")
        self._steps_eval = _steps(getattr(args, "eval_steps", 0), "eval_steps")

        self.actor = DiffusionRLActor(
            self.env.policy.diffusion, k_trainable=args.k_trainable,
            prefix_len=args.prefix_len, sigma_sample_floor=args.sigma_floor,
            sigma_logprob_floor=args.sigma_floor, actor_lr=args.lr,
            expert_lr=args.lr, seed=args.seed,
            num_sample_steps=self._steps_train)
        self.actor.reset_trainable_to_frozen()
        self.frozen_ref = self.actor.frozen_state()
        # The pretrained weights as a params pytree the samplers accept. `load`
        # and every Adam step REPLACE `actor._tparams` rather than mutating it,
        # so this reference stays the base for the life of the run.
        self._base_tparams = self.actor._tparams
        self._jit_preserve = None
        self.eval_state = (self.sampler.eval_batch()
                           if (self.sampler and args.eval_scenes) else probe)
        # Fixed NAMED eval splits (failure/success held-out + train pools),
        # loaded once and cached, so the training curve reports solve/offroad on
        # each split under its own `<split>/` prefix -- not just the one random
        # held-out batch. Empty unless --splits_json is given (and never in
        # --eval_only, which has its own dump path in main()).
        self.eval_splits = ({} if args.eval_only
                            else self._load_eval_splits())
        # A small fixed set of NAMED scenes whose rollout is dumped every probe
        # point, so the curve has pictures under it. See `_build_probe`.
        # Stratum weights for the corpus-level ("marginal") rate. Resolved once:
        # they are a property of the splits file, not of the run.
        self._marg_w = {} if args.eval_only else self._marginal_weights()
        self._marg_cov = 0.0
        if self._marg_w:
            pop = self._read_splits_spec(args.splits_json).get("population", {})
            self._marg_cov = float(pop.get("coverage", 0.0))
            print("[safety] MARGINAL rate on: "
                  + "  ".join(f"{k} w={v:.6f}" for k, v in self._marg_w.items())
                  + f"  (covers {self._marg_cov:.4f} of the corpus)", flush=True)
        self.probe_set = None if args.eval_only else self._build_probe()
        self.probe_every = int(args.probe_every) or int(args.eval_every)
        # Cumulative Adam steps, the x-axis that actually sets how far the
        # weights have travelled (an iteration is 1 step at --grad_accum 0 and
        # one per replan context at 1). Restored from the log on resume.
        self._adam_steps = 0
        self.rng = jax.random.PRNGKey(args.seed)
        # Cached jitted value_and_grad for the free-running loss (see
        # `_grads_for`). Built lazily so it captures the resolved actor/env; JAX
        # keeps one compiled variant per (t0, n_steps, branch-flags) signature.
        self._jit_free_grad = None
        # Eval draws from its OWN key, never from `self.rng`: an eval must not
        # depend on how many training rollouts preceded it, or arms are not
        # comparable and a curve is not a curve.
        self._eval_key = jax.random.PRNGKey(int(args.eval_seed))
        self._base_ref = None
        self.start_iter = 0

        n_t = int(self.actor.diffusion.timesteps)
        print(f"[safety] diffusion timesteps={n_t}  k_trainable={args.k_trainable} "
              f"-> gradient through reverse steps t={args.k_trainable - 1}..0 "
              f"({100.0 * args.k_trainable / n_t:.0f}% lowest-noise end)", flush=True)
        self._init_unroll()
        self._init_windows()
        self._init_preserve()
        self._init_repair()
        self._init_pref()
        self._init_transition_buffer()
        steps_per_iter = ("all contexts -> 1 step/iter" if args.grad_accum <= 0
                          else f"{args.grad_accum} ctx/step")
        print(f"[safety] optimizer: lr={args.lr} accum={steps_per_iter}  "
              f"(Adam moves ~lr per parameter per STEP whatever the gradient "
              f"norm, so step count sets the distance from base)", flush=True)
        print(f"[safety] eval: seed={args.eval_seed} repeats={args.eval_repeats} "
              f"base_reference={bool(args.eval_base)}", flush=True)
        if float(getattr(args, "base_anchor_weight", 0.0)) > 0:
            print(f"[safety] base anchor: success_only="
                  f"{bool(args.base_anchor_success_only)}"
                  f"{'  (groups: ' + ','.join(self.train_pool_names) + ')' if self.train_pool_names else '  (NO GROUPS -- anchoring every row)'}",
                  flush=True)
        print(f"[safety] anchor_weight={args.anchor_weight}"
              f" bc_weight={getattr(args, 'bc_weight', 0.0)}"
              f" base_anchor_weight={getattr(args, 'base_anchor_weight', 0.0)}"
              f"{'  (SAFETY ONLY -- watch eval drift/arc)' if args.anchor_weight == 0 and getattr(args, 'base_anchor_weight', 0.0) == 0 and getattr(args, 'bc_weight', 0.0) == 0 else ''}",
              flush=True)

    # ------------------------------------------------------------------ #
    def _init_windows(self):
        """Resolve the two scoring windows: the safety term's and the anchors'.

        **They live in different time bases, which is the whole trap here.** The
        candidate is ``predict_horizon`` model steps at ``model_dt`` (25 x 0.2 s =
        5.0 s); ``postprocess_predictions`` resamples it to ``world_dt`` (0.1 s),
        giving ~50 world steps plus the anchor at index 0; and the env executes
        only ``replan_interval_steps`` (10) of those before replanning. So:

        * ``--safety_window`` counts WORLD steps (the geometry is scored in world
          coordinates against log-replayed neighbours). 0 = the executed window,
          which is what every run before 2026-09-09 scored; -1 = the whole
          candidate; N = N world steps.
        * ``--anchor_window`` counts MODEL steps (the denoising losses reduce over
          the denoiser's own horizon axis). -1 = the whole horizon, which is what
          every run before 2026-09-09 anchored; 0 = auto, the executed window
          converted into model steps; N = N model steps.

        The auto/full conversions are done here so a sweep arm is a single flag
        value and not arithmetic in a shell script.

        Widening the safety window prices trajectory that will be REPLANNED before
        it is driven -- see the `safety_objective` module docstring, which argues
        against it. That is exactly why it is an ablation axis and not a default:
        the counter-argument is that a 1.0 s window cannot see a violation the
        candidate commits at 2.5 s, so the policy is free to aim at one.
        """
        a = self.args
        cfg = self.ckpt.preprocess_cfg
        model_dt = float(cfg.model_dt)
        world_dt = float(cfg.world_dt_fallback)
        h_model = int(self.actor.diffusion.denoise_fn.horizon)
        # Valid world steps the candidate resamples to (index 0 is the anchor, so
        # this counts steps 1..n). `preprocess` pads two extra steps whose time is
        # clamped to the horizon end -- they duplicate the last pose, so scoring
        # them would charge the same frame repeatedly. Excluded by construction.
        n_world = int(round(h_model * model_dt / world_dt))
        exec_world = int(a.replan_interval_steps)

        sw = int(a.safety_window)
        if sw == 0:
            self.safety_window = exec_world
        elif sw < 0:
            self.safety_window = n_world
        else:
            self.safety_window = min(sw, n_world)
        if self.safety_window < 1:
            raise SystemExit(f"--safety_window resolves to {self.safety_window} steps")
        # Same WORLD-step convention as --safety_window. The default (0) scores the
        # executed window: that segment is the behaviour that earned the success.
        pw = int(getattr(a, "preserve_window", 0))
        self.preserve_window = exec_world if pw == 0 else (n_world if pw < 0 else min(pw, n_world))

        aw = int(a.anchor_window)
        if aw < 0:
            self.anchor_window = h_model
        elif aw == 0:
            # The executed world window expressed in model steps, rounded UP: at
            # replan=10 / dt 0.1 vs 0.2 that is 5 of the 25 model steps.
            self.anchor_window = max(1, min(
                h_model, int(np.ceil(exec_world * world_dt / model_dt))))
        else:
            self.anchor_window = min(aw, h_model)
        # None (not an all-ones vector) when the window is the whole horizon, so
        # the unwindowed code path stays bit-for-bit what it was.
        self._anchor_tw = (
            None if self.anchor_window >= h_model
            else jnp.asarray(
                (np.arange(h_model) < self.anchor_window).astype(np.float32)))

        print(f"[safety] safety window: {self.safety_window} world steps "
              f"({self.safety_window * world_dt:.1f} s of the candidate's "
              f"{n_world * world_dt:.1f} s); env executes {exec_world} "
              f"({exec_world * world_dt:.1f} s)"
              + ("" if self.safety_window <= exec_world else
                 "  -- SCORING PAST THE REPLAN: steps beyond the executed window "
                 "are re-planned before they are ever driven"), flush=True)
        print(f"[safety] anchor window: {self.anchor_window}/{h_model} model steps "
              f"({self.anchor_window * model_dt:.1f} s)"
              + ("  (whole horizon)" if self._anchor_tw is None else
                 "  -- the anchors are charged on the EXECUTED part only; the "
                 "denoiser still predicts the full horizon"), flush=True)

    # ------------------------------------------------------------------ #
    def _init_unroll(self):
        """Resolve and validate the teacher-forced unroll window.

        The window must sit **inside the trainable one**. Every step of the unroll
        runs the trainable denoiser copy, but the sampler only ever invokes that
        copy at ``t <= k_trainable - 1``; a gradient taken at a noisier timestep
        trains behaviour that is never used at deployment except through weight
        sharing. So ``t_start`` defaults to ``k_trainable - 1`` and an explicit
        value above it is an error rather than a warning -- it would look like a
        working run and teach the wrong steps.
        """
        a = self.args
        self.unroll_steps = int(a.unroll_steps)
        self.reuse_prefix = False
        if self.unroll_steps <= 0:
            self.unroll_steps = 0
            # Log-driven collection samples nothing, so there is no prefix to keep.
            self.reuse_prefix = (bool(getattr(a, "reuse_prefix", False))
                                 and a.anchor_states != "log")
            if self.reuse_prefix:
                print(f"[safety] reuse_prefix: gradient replays the {a.k_trainable} "
                      "trainable steps of the EXECUTED sample from its stored "
                      "frozen prefix (was: a fresh full-chain sample)", flush=True)
            n_t = int(self.actor.diffusion.timesteps)
            tr = self._steps_train or n_t
            ev = self._steps_eval or n_t
            print(f"[safety] proposal=free-running  train={tr} nn evals/proposal "
                  f"eval={ev}"
                  + ("  (full reverse chain)" if tr == n_t and ev == n_t else
                     f"  [DDIM-strided prefix: {n_t} -> {tr} train / {ev} eval; "
                     f"the {a.k_trainable} trainable steps are unchanged]"),
                  flush=True)
            if self._steps_train and not self._steps_eval:
                print("[safety]   train strided, eval on the FULL chain: the curve "
                      "measures the deployable policy, so a gain there is not the "
                      "sampler's doing", flush=True)
            if self._steps_eval:
                print("[safety]   WARNING eval is strided too -- eval numbers are "
                      "NOT comparable to a full-chain run", flush=True)
            return

        n_t = int(self.actor.diffusion.timesteps)
        self.t_end = int(a.unroll_t_end)
        self.t_start_max = (a.k_trainable - 1 if a.unroll_t_start < 0
                            else int(a.unroll_t_start))
        if not 0 <= self.t_end < self.t_start_max:
            raise SystemExit(
                f"[safety] need 0 <= unroll_t_end < unroll_t_start, got "
                f"t_end={self.t_end} t_start={self.t_start_max}"
                + (f" (auto from k_trainable={a.k_trainable}; raise k_trainable "
                   f"to widen the window)" if a.unroll_t_start < 0 else ""))
        if self.t_start_max > a.k_trainable - 1:
            raise SystemExit(
                f"[safety] unroll_t_start={self.t_start_max} is outside the "
                f"trainable window t<={a.k_trainable - 1}: those steps run the "
                f"frozen denoiser at sampling time, so training them there is "
                f"wasted. Raise --k_trainable or lower --unroll_t_start.")
        if self.t_start_max >= n_t:
            raise SystemExit(f"[safety] unroll_t_start={self.t_start_max} >= timesteps={n_t}")
        self.unroll_grad_steps = max(1, min(int(a.unroll_grad_steps), self.unroll_steps))

        sqrt_1mab = np.asarray(self.actor.diffusion.sqrt_one_minus_alphas_cumprod.value)
        ego_range = 100.0
        print(f"[safety] proposal=teacher-forced unroll  K={self.unroll_steps} "
              f"grad_steps={self.unroll_grad_steps} eta={a.unroll_eta}  "
              f"t_start~U[{self.t_end + 1},{self.t_start_max}] -> t_end={self.t_end}",
              flush=True)
        print(f"[safety]   corruption noise on xy: "
              f"t={self.t_start_max} -> {ego_range * sqrt_1mab[self.t_start_max]:.1f} m, "
              f"t={self.t_end} -> {ego_range * sqrt_1mab[self.t_end]:.1f} m  "
              f"({self.unroll_steps} nn evals per gradient step vs {n_t} free-running)",
              flush=True)
        if a.anchor_states == "log":
            print("[safety]   anchor_states=log: contexts come from the human's own "
                  "states -- no exposure-bias correction in this run", flush=True)

    # ------------------------------------------------------------------ #
    def _next_batch(self):
        if self.sampler is not None:
            state = self.sampler.next_train()
            self._anchor_mask = self._success_mask_for(self.sampler.last_ids)
            return state
        return load_batch(self.args.tfrecord, self.fixed_indices)

    def _success_mask_for(self, ids):
        """[B] 1.0 on success-group scenes, 0.0 elsewhere; None if not applicable.

        None (rather than all-ones) so the loss can tell "no restriction wanted"
        from "restriction wanted and nothing matched" -- the second is a
        misconfiguration that would silently disable the anchor entirely.
        """
        if not (self.args.base_anchor_success_only and self._success_groups
                and self.sampler is not None and self.sampler.groups is not None):
            return None
        lab = self.sampler.group_labels(ids)
        return jnp.asarray(np.isin(lab, self._success_groups).astype(np.float32))

    def _gate_expert_clean(self, state):
        """Drop scenes whose own log is off-road (they dominate an unbounded barrier)."""
        # Jitted once: eager, the 91-step road-edge clearance ran op-by-op twice
        # per iteration (policy + BC pass) and once per eval chunk.
        fn = getattr(self, "_jit_clean", None)
        if fn is None:
            cfg = self.sl_cfg
            fn = self._jit_clean = jax.jit(
                lambda s: expert_clean_mask(cfg, s, init_steps=11)[0])
        return jnp.asarray(fn(state), jnp.float32)

    def _propose(self, params, ctx, rng):
        """Normalized -> world candidate for one context, differentiable in params."""
        traj_ncT = self.actor.sample_world_for(params, ctx.cond1, ctx.cond2, ctx.mask, rng)
        return self.env.norm_to_world_with(jnp.transpose(traj_ncT, (0, 2, 1)), ctx.aux)

    def _expert_x0_bct(self, ctx):
        """The expert future as the denoiser's own ``[B, C, T]`` layout.

        Truncated to the model's horizon for the same reason
        :meth:`DiffusionRLActor._expert_loss`'s caller does it: the preprocessor
        emits a longer ``ego_trajectory`` and the conv denoiser is length-agnostic,
        so a mismatch scores a horizon the checkpoint never trained on instead of
        raising.
        """
        h = int(self.actor.diffusion.denoise_fn.horizon)
        if ctx.expert_target is None:
            raise SystemExit("[safety] teacher forcing needs the expert target; "
                             "the rollout did not attach one")
        return jnp.transpose(ctx.expert_target[:, :h, :], (0, 2, 1))

    def _propose_unrolled(self, params, ctx, x0_bct, t_grid_kb, rng):
        """Teacher-forced unrolled candidate -> world, differentiable in params."""
        traj_ncT = self.actor.unrolled_sample_for(
            params, ctx.cond1, ctx.cond2, ctx.mask, x0_bct, t_grid_kb, rng,
            grad_steps=self.unroll_grad_steps, eta=float(self.args.unroll_eta))
        return self.env.norm_to_world_with(jnp.transpose(traj_ncT, (0, 2, 1)), ctx.aux)

    def _resume_iter(self) -> int:
        """The iteration a resume would start at, read from the log alone.

        Called in ``__init__`` (the sampler seed depends on it) and again in
        ``train``; it is a pure read of ``train_log.jsonl``, so calling it twice
        is free and it stays correct when --resume is off (returns 0).
        """
        if not int(getattr(self.args, "resume", 0)):
            return 0
        out = Path(self.args.out_dir)
        log_path, ckpt = out / "train_log.jsonl", out / "latest"
        if not (log_path.exists() and ckpt.exists()):
            return 0
        last = 0
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                last = max(last, int(json.loads(line)["iter"]) + 1)
            except Exception:
                continue        # torn last line from a killed job
        return last

    def collect_bc_contexts(self, state):
        """Contexts for the BC term: the ORIGINAL pretraining objective's data.

        WHY THIS IS NOT `--anchor_weight`. Both terms call the same function --
        `DiffusionRLActor._expert_loss` -> `diffusion.loss_per_example`, the
        denoising loss the checkpoint was pretrained with. What differs is the
        state distribution the conditioning comes from, and that is the whole
        question:

        * the **anchor** conditions on states the POLICY reached and takes the
          human's future re-expressed from that (drifted) pose. From a drifted
          state that target demands an instantaneous lateral correction, and a
          model trained on it learns to emit exactly that -- DAgger-shaped
          relabeling, not the pretraining loss.
        * the **BC term** conditions on the states the HUMAN visited, where the
          target is feasible by construction. That is what the original pipeline
          optimized, and it is the term that can hold the policy on the
          pretraining manifold rather than teaching it a new reflex.

        `--anchor_states log` already drives a rollout from the log, but it
        drives the *only* rollout, so the safety term is then scored on human
        states where there is nothing to fix. This is a SECOND, cheap pass over
        the same batch: no sampling at all (the log supplies the trajectory), so
        it costs 8 env steps against the policy pass's 8 x 100 denoiser evals.

        Same batch as the safety term on purpose. Half of it is failure scenes
        under `--train_split_ratio`, and on those the human log is exactly the
        trajectory the safety term is trying to find -- so the BC term is a
        do-no-harm leash on the successes and a demonstration on the failures.
        """
        contexts, _ = self.rollout(state, train=True, log_states=True)
        return contexts

    # ------------------------------------------------------------------ #
    # Matched-sample preservation (`--preserve_weight`, 2026-09-13)
    # ------------------------------------------------------------------ #
    # WHY NOT THE BASE ANCHOR. `base_anchor_loss` compares one-step posterior
    # means on NOISED HUMAN trajectories at t ~ U[0, 100), on the FINE-TUNE's
    # visited states. Deployment is none of those: the trainable copy only ever
    # runs the last k steps, from a prefix the frozen denoiser produced, on
    # states the base itself reaches when it succeeds. Its logged share of the
    # objective was ~1e-5 at every iteration of the stopping sweep -- present
    # enough to matter (removing it collapses success to 0.76) but measuring
    # something other than the behaviour it is meant to protect.
    #
    # This term protects the REALIZED behaviour instead. Each iteration rolls the
    # frozen BASE out on success-pool scenes (hard ones over-sampled, recently
    # broken ones replayed), keeps each replan's prefix x_{k-1} and late key, and
    # keeps only the rows the base actually SOLVED. The loss is the world-space
    # difference -- position, heading, speed over the executed window -- between
    # the trainable late chain and the base late chain from that same prefix and
    # noise. It is exactly zero, in value and gradient, at the base weights.
    def _init_preserve(self):
        a = self.args
        self.preserve_weight = float(getattr(a, "preserve_weight", 0.0))
        self._replay: dict[int, int] = {}          # scene id -> draws left
        self._preserve_hard = None
        self._preserve_pool = None
        if self.preserve_weight <= 0:
            return
        if self.sampler is None or not self._success_groups:
            raise SystemExit("--preserve_weight needs --train_splits with a success split "
                             "(it draws the scenes it protects from that pool)")
        if (int(a.grad_accum) != 0 or int(a.epochs) != 1 or self.unroll_steps
                or a.anchor_states != "policy"):
            raise SystemExit("--preserve_weight supports only --grad_accum 0 --epochs 1 "
                             "--anchor_states policy and free-running proposals (one Adam "
                             "step per iteration; the base rollout must drive itself)")
        held = set(int(i) for i in getattr(self.sampler, "_eval_set", set()))
        pool = np.unique(np.concatenate(
            [self.train_pool_groups[g] for g in self._success_groups]))
        pool = np.asarray([i for i in pool.tolist() if int(i) not in held], np.int64)
        self._preserve_pool = pool
        risk_path = str(getattr(a, "preserve_risk", "") or "")
        if risk_path and float(a.preserve_hard_frac) > 0:
            from rl.diffusion_ft.tools.gate_analysis import scene_risk
            spec = self._read_splits_spec(a.splits_json)
            spec = {**spec, "shard_glob": spec.get("shard_glob") or a.shard_glob,
                    "index_dir": spec.get("index_dir") or a.index_dir}
            risk = scene_risk(spec, risk_path, pool)
            thr = float(np.quantile(risk, float(a.preserve_hard_quantile)))
            self._preserve_hard = pool[risk >= thr]
        print(f"[safety] PRESERVE weight={self.preserve_weight} scenes/iter="
              f"{a.preserve_scenes} pool={pool.size} hard="
              f"{0 if self._preserve_hard is None else self._preserve_hard.size}"
              f" (q>={a.preserve_hard_quantile}, frac {a.preserve_hard_frac}) "
              f"replay cap={a.preserve_replay} per_iter={a.preserve_replay_per_iter} "
              f"uses={a.preserve_replay_uses}  window={self.preserve_window} world steps  "
              f"scales pos={a.preserve_scale_pos}m yaw={a.preserve_scale_yaw}rad "
              f"speed={a.preserve_scale_speed}m/s", flush=True)

    # ------------------------------------------------------------------ #
    # Replayed sampler transitions (`--tb_weight`, 2026-09-14)
    # ------------------------------------------------------------------ #
    def _init_transition_buffer(self):
        """Resolve the transition-replay term. See :mod:`transition_buffer`.

        Three things are checked rather than warned about, because each would
        produce a run that looks like it worked:

        * corpus mode, since a replayed datum is reloaded by scene id;
        * one Adam step per iteration, so the buffer term and the on-policy term
          land in the SAME step instead of the buffer chasing weights the
          on-policy term just moved;
        * ``--anchor_states policy`` and free-running collection, because the
          whole content of a stored transition is that the deployed sampler
          produced it. Under ``log`` there is no sampler and nothing to store.
        """
        a = self.args
        self.tb_weight = float(getattr(a, "tb_weight", 0.0))
        self.tb = None
        self._jit_tb = None
        self._tb_iter = 0
        self._tb_rng = np.random.default_rng(int(a.seed) * 6700417 + 11)
        if self.tb_weight <= 0:
            return
        if self.sampler is None:
            raise SystemExit("--tb_weight needs corpus mode (--shard_glob): a replayed "
                             "transition carries a scene id, and the geometry is "
                             "recovered by reloading that scene")
        if int(a.grad_accum) != 0 or int(a.epochs) != 1 or self.unroll_steps:
            raise SystemExit("--tb_weight supports only --grad_accum 0 --epochs 1 and "
                             "free-running collection (--unroll_steps 0): the stored "
                             "state IS the sampler's, so there is nothing to teacher-force")
        if a.anchor_states != "policy":
            raise SystemExit("--tb_weight needs --anchor_states policy -- under `log` "
                             "the collection rollout runs no sampler, so there are no "
                             "transitions to store")

        k = int(a.k_trainable)
        self.tb_t_min = max(0, int(a.tb_t_min))
        self.tb_t_max = (k - 1) if int(a.tb_t_max) < 0 else min(int(a.tb_t_max), k - 1)
        if self.tb_t_min > self.tb_t_max:
            raise SystemExit(f"--tb_t_min={self.tb_t_min} > --tb_t_max={self.tb_t_max} "
                             f"(the trace only holds t = {k - 1}..0)")
        self.tb_window = (self.safety_window if int(a.tb_window) <= 0
                          else min(int(a.tb_window), self.safety_window))
        self.tb_K = max(1, int(a.tb_unroll_steps))
        self.tb_t_end = int(a.tb_unroll_t_end)
        if self.tb_K > 1 and not 0 <= self.tb_t_end < self.tb_t_max:
            raise SystemExit(f"--tb_unroll_t_end={self.tb_t_end} must satisfy "
                             f"0 <= t_end < tb_t_max={self.tb_t_max}, or there is "
                             f"nothing to unroll down to")
        self.tb_grad_steps = max(1, min(int(a.tb_grad_steps), self.tb_K))
        self.tb_min_size = (int(a.tb_batch) if int(a.tb_min_size) <= 0
                            else int(a.tb_min_size))
        self.tb = TransitionBuffer(int(a.tb_capacity), max_uses=int(a.tb_max_uses),
                                   seed=int(a.seed) + 9176)
        sqrt_1mab = np.asarray(self.actor.diffusion.sqrt_one_minus_alphas_cumprod.value)
        print(f"[safety] TRANSITION REPLAY weight={self.tb_weight} "
              f"capacity={a.tb_capacity} batch={a.tb_batch} "
              f"store={a.tb_store_per_replan}/row/replan max_uses={a.tb_max_uses} "
              f"min_size={self.tb_min_size}", flush=True)
        print(f"[safety]   stored timesteps t in [{self.tb_t_min}, {self.tb_t_max}] "
              f"(residual xy noise {100.0 * sqrt_1mab[self.tb_t_min]:.1f}"
              f"-{100.0 * sqrt_1mab[self.tb_t_max]:.1f} m); scored window="
              f"{self.tb_window} world steps", flush=True)
        if self.tb_K <= 1:
            print("[safety]   proposal=ONE-STEP clean prediction x0_hat(x_t, t) from the "
                  "stored state -- 1 network eval, no unroll (the Jensen-gap control: "
                  "the penalised object is a posterior MEAN, not a sample)", flush=True)
        else:
            print(f"[safety]   proposal=UNROLL K={self.tb_K} (eta={a.tb_unroll_eta}) from "
                  f"the stored state down to t={self.tb_t_end}, scoring the final clean "
                  f"prediction; gradient through the last {self.tb_grad_steps} of "
                  f"{self.tb_K} evals", flush=True)
        if float(a.safety_weight) <= 0:
            print("[safety]   safety_weight=0: the ON-POLICY safety term is OFF, so this "
                  "arm's geometric gradient comes from the buffer alone (the rollout "
                  "still runs -- it is what fills the buffer)", flush=True)
        # The buffer is NOT checkpointed: it holds sampler states, and states
        # produced before a multi-day run's restart belong to the weights of a
        # policy several restarts old. It refills at
        # `scenes_per_iter * max_replans * tb_store_per_replan` per iteration.
        refill = (int(a.scenes_per_iter) * int(a.max_replans)
                  * int(a.tb_store_per_replan))
        print(f"[safety]   buffer is NOT checkpointed -- a resume starts it empty and "
              f"refills at ~{refill} transitions/iter "
              f"({max(1, int(a.tb_capacity) // max(refill, 1))} iters to capacity)",
              flush=True)

    def _store_transitions(self, ctx, x_t_kbct, t_kb, scene_ids, n_steps):
        """Push this replan's sampler transitions into the buffer, one per datum.

        ``x_t_kbct`` is ``[K, B, C, T]`` and ``t_kb`` ``[K, B]`` straight off
        ``sample_with_trace`` -- the states the chain passed through on its way to
        the trajectory the env is about to execute. Only ALIVE rows are stored (a
        terminated scene's replan never happened), and only when the scenario has
        the full ``tb_window`` left, since a short window at the end of an episode
        would score a different number of steps than the rest of the buffer and
        the loss is a per-step mean.
        """
        if self.tb is None or scene_ids is None:
            return 0
        if int(n_steps) < self.tb_window:
            return 0
        alive_np = np.asarray(ctx.alive, np.float32)
        rows = np.flatnonzero(alive_np > 0.5)
        if rows.size == 0:
            return 0
        t_np = np.asarray(t_kb)                      # [K, B]
        x_np = np.asarray(x_t_kbct)                  # [K, B, C, T]
        # Trace row j holds timestep k-1-j for every batch row, so the eligible
        # rows of the trace are the same for all scenes; pick them once.
        t_col = t_np[:, 0]
        eligible = np.flatnonzero((t_col >= self.tb_t_min) & (t_col <= self.tb_t_max))
        if eligible.size == 0:
            return 0
        # Distinct timesteps per ROW, drawn independently ACROSS rows. A flat
        # `choice(..., size=(rows, p), replace=False)` would be wrong twice over:
        # it draws rows*p distinct values from a pool of `eligible.size` (~10),
        # which raises as soon as the batch is bigger than the trainable window,
        # and it would couple the draws of different scenes. Argsorting a random
        # key per (row, eligible) is the vectorized per-row permutation.
        p = min(int(self.args.tb_store_per_replan), int(eligible.size))
        order = np.argsort(self._tb_rng.random((rows.size, eligible.size)), axis=1)
        pick = eligible[order[:, :p]]
        rr = np.repeat(rows, p)                      # batch row per datum
        jj = pick.reshape(-1)                        # trace row per datum
        out = {
            "scene_id": np.asarray(scene_ids, np.int64)[rr],
            "t0": np.full(rr.size, int(ctx.t0), np.int32),
            "t": t_np[jj, rr].astype(np.int32),
            "x_t": x_np[jj, rr].astype(np.float32),
            "cond1": np.asarray(ctx.cond1)[rr],
            "cond2": np.asarray(ctx.cond2)[rr],
            "cond2_mask": np.asarray(ctx.mask)[rr],
        }
        for k, v in ctx.aux.items():
            out[f"aux/{k}"] = np.asarray(v)[rr]
        return self.tb.add(out, iteration=self._tb_iter)

    def _tb_core(self, params, state, dyn, rng, *, n, grad_steps, eta):
        """Safety loss of the clean prediction reached from a STORED ``x_t``.

        Pure in ``params`` + arrays, same contract as :meth:`_free_core`. The one
        structural difference from every other path here is that ``t0`` is a
        ``[B]`` vector: the rows of a replayed minibatch were recorded at
        different replans of different episodes, which is what
        ``candidate_safety``'s per-row window slice exists for.
        """
        x0_bct = self.actor.clean_prediction_from_xt(
            params, dyn["cond1"], dyn["cond2"], dyn["cond2_mask"],
            dyn["x_t"], dyn["t_grid"], rng, grad_steps=grad_steps, eta=eta)
        world = self.env.norm_to_world_with(jnp.transpose(x0_bct, (0, 2, 1)), dyn["aux"])
        terms = candidate_safety(self.objcfg, world, state=state, t0=dyn["t0"],
                                 n_steps=n, alive_b=None)
        total = terms["total"]
        return total, {"tb_safety": total,
                       "tb_edge": jnp.sum(terms["edge_loss"]) / max(n, 1),
                       "tb_agent": jnp.sum(terms["agent_loss"]) / max(n, 1),
                       # Named apart from the policy term's `tail_share`: both
                       # land in the same row, and a replay arm's buffer states
                       # are not the states the on-policy term scored.
                       "tb_tail_share": (jnp.sum(terms["tail_loss"])
                                         / jnp.maximum(jnp.sum(terms["loss"]),
                                                       1e-8))}

    def _tb_grad_fn(self):
        if self._jit_tb is None:
            self._jit_tb = jax.jit(
                jax.value_and_grad(self._tb_core, has_aux=True),
                static_argnames=("n", "grad_steps", "eta"))
        return self._jit_tb

    def buffer_grads(self, it: int):
        """(grads, metrics) from one replayed minibatch, or (None, {}) if too few.

        The scenes are reloaded by id: ``candidate_safety`` reads only
        ``log_trajectory``, ``object_metadata`` and ``roadgraph_points``, which
        are constants of the episode, so a freshly loaded state at timestep 0
        scores a transition recorded at ``t0`` exactly as the live state would
        have. Nothing about the rollout that produced the transition needs to be
        reconstructed.
        """
        batch = self.tb.sample(int(self.args.tb_batch), iteration=it)
        if batch is None or len(self.tb) < self.tb_min_size:
            return None, {}
        ids = [int(i) for i in batch["scene_id"]]
        state = self.sampler.load_explicit(ids)
        t_b = jnp.asarray(batch["t"], jnp.int32)
        if self.tb_K <= 1:
            t_grid = t_b[None, :]
        else:
            # A stored state already at or below t_end has nothing to unroll; the
            # clamp turns its grid into the tied no-op `unroll_timestep_grid`
            # documents rather than an ASCENDING grid, which would re-noise it.
            t_grid = unroll_timestep_grid(
                jnp.maximum(t_b, self.tb_t_end), self.tb_K, self.tb_t_end)
        dyn = {"cond1": jnp.asarray(batch["cond1"]),
               "cond2": jnp.asarray(batch["cond2"]),
               "cond2_mask": jnp.asarray(batch["cond2_mask"]),
               "x_t": jnp.asarray(batch["x_t"]),
               "t_grid": t_grid,
               "t0": jnp.asarray(batch["t0"], jnp.int32),
               "aux": {k[4:]: jnp.asarray(v) for k, v in batch.items()
                       if k.startswith("aux/")}}
        self.rng, rk = jax.random.split(self.rng)
        (loss, aux), grads = self._tb_grad_fn()(
            self.actor._tparams, state, dyn, rk, n=self.tb_window,
            grad_steps=self.tb_grad_steps, eta=float(self.args.tb_unroll_eta))
        m = {k: float(np.asarray(v)) for k, v in aux.items()}
        m["tb_age"] = float(np.mean(batch["age"]))
        m["tb_t"] = float(np.mean(batch["t"]))
        return grads, m

    # ------------------------------------------------------------------ #
    # Correction from verified repairs (`--repair_weight`, 2026-09-13)
    # ------------------------------------------------------------------ #
    # `counterfactual --mode repair` rewinds base failures, searches timing /
    # lateral / re-noised candidates, and keeps a repair only if the WHOLE
    # continuation solves the scene. Each kept repair leaves (frozen condition,
    # executed candidate in normalized space) per replan it was applied on. This
    # term teaches the trainable late copy to emit those candidates from those
    # conditions -- `late_denoise_loss`, i.e. only at the t < k_trainable steps the
    # copy actually drives. The copy starts equal to the base, so the correction is
    # a zero residual at iteration 0 in weight space.
    def _init_repair(self):
        a = self.args
        self.repair_weight = float(getattr(a, "repair_weight", 0.0))
        self._repair = None
        self._jit_repair = None
        if self.repair_weight <= 0:
            return
        import glob as _glob
        paths = sorted(_glob.glob(str(a.repair_data)))
        if not paths:
            raise SystemExit(f"--repair_weight > 0 but --repair_data matches nothing: {a.repair_data}")
        if int(a.grad_accum) != 0 or int(a.epochs) != 1 or self.unroll_steps:
            raise SystemExit("--repair_weight supports only --grad_accum 0 --epochs 1 "
                             "and free-running proposals (one Adam step per iteration)")
        rows_mode = str(getattr(a, "repair_rows", "all"))
        ratio = _kv_ratio(getattr(a, "repair_split_ratio", ""))
        keys = ["cond1", "cond2", "mask", "target", "valid", "scene_id", "switched"]
        # `proposal_select --dump_bc` files carry the episode outcome and the
        # split per row; counterfactual repairs are all verified and carry neither.
        if rows_mode == "solved":
            keys.append("ep_solved")
        if ratio:
            keys.append("split")
        parts = {k: [] for k in keys}
        rt = []
        for f in paths:
            z = np.load(f)
            n_f = z["target"].shape[0]
            for k in parts:
                # `valid` marks the model steps the repair target is real on. A
                # file written before that mask existed carries a frozen tail past
                # the scenario end, so refuse it rather than train on it.
                if k == "switched" and k not in z.files:
                    parts[k].append(np.ones(n_f, bool))      # a verified repair IS an edit
                    continue
                if k not in z.files:
                    raise SystemExit(f"{f} has no '{k}' array -- rebuild the repair data "
                                     + ("(it predates the covered-steps mask)" if k == "valid" else
                                        f"(needed by --repair_rows {rows_mode} / "
                                        f"--repair_split_ratio; only proposal_select "
                                        f"--dump_bc writes it)"))
                parts[k].append(z[k])
            if "rt_err" in z.files:
                rt.append(np.asarray(z["rt_err"]))
        self._repair = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
        n_all = self._repair["target"].shape[0]
        if rows_mode == "solved":
            # Filtered BC: only the rollouts whose whole episode solved.
            keep = self._repair.pop("ep_solved").astype(bool)
            self._repair = {k: v[keep] for k, v in self._repair.items()}
        # Row pools per split, sampled at a fixed mix per batch -- otherwise the mix
        # is whatever the collection happened to produce.
        self._repair_pools = None
        # `--repair_switch_share S`: a fixed S of every batch from the rows where
        # the arm EDITED the plan, the rest from the kept rows. Uniform sampling
        # leaves the edits at ~7% of a batch, next to rows whose target is the
        # base's own sample (bank_bc 09-29: their loss was still falling at 3000
        # steps).
        share = float(getattr(a, "repair_switch_share", 0.0))
        if not 0.0 <= share < 1.0:
            raise SystemExit(f"--repair_switch_share must be in [0, 1), got {share}")
        B = int(a.repair_batch)
        n_sw = int(round(B * share))
        sw_all = self._repair["switched"].astype(bool)
        if n_sw and not sw_all.any():
            raise SystemExit("--repair_switch_share > 0 but no row is switched")
        rest = ~sw_all if n_sw else np.ones(len(sw_all), bool)
        if ratio:
            split = self._repair.pop("split")
            missing = set(ratio) - set(np.unique(split).tolist())
            if missing:
                raise SystemExit(f"--repair_split_ratio names {sorted(missing)} but the "
                                 f"data holds only {sorted(np.unique(split).tolist())}")
            tot = sum(ratio.values())
            counts = [int(round((B - n_sw) * w / tot)) for w in ratio.values()]
            counts[-1] = (B - n_sw) - sum(counts[:-1])
            self._repair_pools = [(s if not n_sw else f"{s}/kept",
                                   np.nonzero((split == s) & rest)[0], c)
                                  for s, c in zip(ratio, counts)]
        elif n_sw:
            self._repair_pools = [("kept", np.nonzero(rest)[0], B - n_sw)]
        if n_sw:
            self._repair_pools.insert(0, ("switched", np.nonzero(sw_all)[0], n_sw))
        for s, ix, c in self._repair_pools or []:
            if c > 0 and len(ix) == 0:
                raise SystemExit(f"repair pool '{s}' is asked for {c}/batch but holds no rows")
        # A training scene id must never be a held-out one: repairs are searched on
        # the training pools, and this re-checks it against the splits in use.
        if a.splits_json:
            spec = self._read_splits_spec(a.splits_json)
            held = set()
            for n, v in spec["splits"].items():
                if n.endswith("_heldout"):
                    held |= {int(i) for i in v["scene_ids"]}
            leak = held & set(int(i) for i in self._repair["scene_id"].tolist())
            if leak:
                raise SystemExit(f"--repair_data holds {len(leak)} held-out scene(s) -- refusing")
        rt_max = np.nanmax(np.stack(rt), axis=0).tolist() if rt else None
        n_kept = self._repair["target"].shape[0]
        if n_kept == 0:
            raise SystemExit(f"--repair_rows {rows_mode} kept 0 of {n_all} rows")
        print(f"[safety] REPAIR weight={self.repair_weight} targets={n_kept} "
              f"(of {n_all}, rows={rows_mode}) from "
              f"{len(np.unique(self._repair['scene_id']))} scenes, batch={a.repair_batch}"
              f" x {int(getattr(a, 'repair_steps', 1))} step(s)/iter, "
              f"switched {float(self._repair['switched'].mean()):.3f}, "
              f"covered steps {float(self._repair['valid'].mean()):.3f}; "
              f"world<->norm round-trip max err (pos, vel, yaw) = {rt_max}", flush=True)
        if self._repair_pools is not None:
            sw = self._repair["switched"]
            print("[safety] REPAIR per-batch mix: " + "  ".join(
                f"{s} {c}/batch from {len(ix)} rows (switched {float(sw[ix].mean()):.3f})"
                for s, ix, c in self._repair_pools), flush=True)

    def repair_grads(self, it: int, sub: int = 0):
        """Gradient of the late-step denoising loss on one batch of repair rows.

        ``sub`` indexes the extra steps of ``--repair_steps``; ``sub=0`` draws
        exactly the batch every earlier run drew at iteration ``it``."""
        a = self.args
        R = self._repair
        n = R["target"].shape[0]
        rng = np.random.default_rng(int(a.data_seed) * 15485863 + 7 * int(it)
                                    + 1000003 * int(sub))
        if self._repair_pools is None:
            idx = rng.choice(n, size=int(a.repair_batch), replace=n < int(a.repair_batch))
        else:
            idx = np.concatenate([rng.choice(ix, size=c, replace=len(ix) < c)
                                  for _, ix, c in self._repair_pools if c > 0])
        if self._jit_repair is None:
            self._jit_repair = jax.jit(jax.value_and_grad(self.actor.late_denoise_loss))
        self.rng, rk = jax.random.split(self.rng)
        x0 = jnp.transpose(jnp.asarray(R["target"][idx], jnp.float32), (0, 2, 1))
        loss, grads = self._jit_repair(
            self.actor._tparams, x0, jnp.asarray(R["cond1"][idx]), jnp.asarray(R["cond2"][idx]),
            jnp.asarray(R["mask"][idx]), jnp.ones(len(idx), jnp.float32), rk,
            jnp.asarray(R["valid"][idx], jnp.float32))
        return grads, float(np.asarray(loss))

    # ------------------------------------------------------------------ #
    # Diffusion-DPO on (condition, rejected base draw, accepted edit) pairs
    # (`--pref_weight`, 2026-10-01)
    # ------------------------------------------------------------------ #
    # Repaired-trajectory BC (`--repair_*`, bank_bc 09-28/30) teaches "go to the
    # edit" and never shows the student the draw the edit replaced: weak pressure
    # moved nothing, strong pressure shifted the whole conditional and broke
    # successes (docs/bank_bc_distillation.md, idea 2). A pair says "this edit
    # over THIS draw of yours", relative to the frozen base. Pairs are the rows
    # where the arm switched (`target` = the edit, `rejected` = the base draw,
    # same normalization path, see `proposal_select._bc_rows`). The tether is the
    # existing frozen-base KL (`base_anchor_loss`), on success_train rows only --
    # the success-only grouping -- so a failure scene's pair is never pulled back
    # toward the draw it is moving off. Both terms share every Adam step.
    def _init_pref(self):
        a = self.args
        self.pref_weight = float(getattr(a, "pref_weight", 0.0))
        self._pref = None
        self._jit_pref = self._jit_pref_anchor = None
        self._pref_init_checked = False
        if self.pref_weight <= 0:
            return
        import glob as _glob
        paths = sorted(_glob.glob(str(a.pref_data)))
        if not paths:
            raise SystemExit(f"--pref_weight > 0 but --pref_data matches nothing: {a.pref_data}")
        if int(a.grad_accum) != 0 or int(a.epochs) != 1 or self.unroll_steps:
            raise SystemExit("--pref_weight supports only --grad_accum 0 --epochs 1 "
                             "and free-running proposals (one Adam step per iteration)")
        keys = ["cond1", "cond2", "mask", "target", "rejected", "valid", "scene_id",
                "switched", "split", "ep_solved", "rt_err_m"]
        parts = {k: [] for k in keys}
        for f in paths:
            z = np.load(f)
            for k in keys:
                if k not in z.files:
                    raise SystemExit(
                        f"{f} has no '{k}' array -- preference pairs need "
                        + ("the rejected base draw, which `proposal_select --dump_bc` "
                           "stores since 2026-10-01; recollect" if k == "rejected"
                           else "`proposal_select --dump_bc` data"))
                parts[k].append(z[k])
        R = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
        n_all = R["target"].shape[0]
        sw = R["switched"].astype(bool)
        # ---- the one preflight: both sides are the same object --------------
        if R["target"].shape != R["rejected"].shape:
            raise SystemExit(f"target {R['target'].shape} != rejected {R['rejected'].shape}")
        diff = np.abs(R["target"] - R["rejected"])                       # [N, T, 5]
        kept_gap = float(diff[~sw].max()) if (~sw).any() else 0.0
        unc = ~R["valid"].astype(bool)                                  # [N, T]
        unc_gap = float(diff[sw][unc[sw]].max()) if (unc[sw]).any() else 0.0
        if kept_gap > 0 or unc_gap > 0:
            raise SystemExit(f"--pref_data: target != rejected where the arm did not edit "
                             f"(kept rows {kept_gap:.3g}, uncovered steps {unc_gap:.3g}); "
                             f"the pair would differ by more than the edit")
        ego_range = float(self.ckpt.preprocess_cfg.ego_range)
        d_m = np.linalg.norm((R["target"] - R["rejected"])[..., :2], axis=-1) * ego_range
        rows_mode = str(getattr(a, "pref_rows", "solved"))
        pair = sw & (R["ep_solved"].astype(bool) if rows_mode == "solved" else True)
        edit_peak = d_m.max(axis=1)
        if not pair.any():
            raise SystemExit(f"--pref_rows {rows_mode}: no switched row to make a pair from")
        if (edit_peak[pair] <= 0).any():
            raise SystemExit(f"{int((edit_peak[pair] <= 0).sum())} switched rows carry no "
                             f"edit on covered steps -- not a pair")
        # Step weights: the covered-steps mask, x (1 + lambda) where the pair
        # differs by more than delta (`--pref_edit_weight`, arm B). lambda = 0
        # is plain masking.
        lam = float(getattr(a, "pref_edit_weight", 0.0))
        dlt = float(getattr(a, "pref_edit_delta_m", 0.1))
        R["step_w"] = (R["valid"].astype(np.float32)
                       * (1.0 + lam * (d_m > dlt).astype(np.float32)))
        anchor = (R["split"] == "success_train") & ~sw
        self._pref_ix = np.nonzero(pair)[0]
        self._pref_anchor_ix = np.nonzero(anchor)[0]
        self.pref_anchor_weight = float(getattr(a, "pref_anchor_weight", 0.0))
        if self.pref_anchor_weight > 0 and not len(self._pref_anchor_ix):
            raise SystemExit("--pref_anchor_weight > 0 but the data holds no kept "
                             "success_train row to anchor on")
        if a.splits_json:
            spec = self._read_splits_spec(a.splits_json)
            held = set()
            for n, v in spec["splits"].items():
                if n.endswith("_heldout"):
                    held |= {int(i) for i in v["scene_ids"]}
            leak = held & set(int(i) for i in R["scene_id"].tolist())
            if leak:
                raise SystemExit(f"--pref_data holds {len(leak)} held-out scene(s) -- refusing")
        T_max = int(getattr(a, "pref_t_max", -1))
        self.pref_t = (int(getattr(a, "pref_t_min", 0)),
                       int(self.actor.k_trainable) if T_max < 0 else T_max)
        if not 0 <= self.pref_t[0] < self.pref_t[1] <= int(self.actor.diffusion.timesteps):
            raise SystemExit(f"--pref_t_min/--pref_t_max: empty or out-of-range t "
                             f"window {self.pref_t}")
        keep = ("cond1", "cond2", "mask", "target", "rejected", "step_w")
        self._pref = {k: R[k] for k in keep}
        pk = self._pref_ix
        q = lambda x: " / ".join(f"{v:.2f}" for v in np.quantile(x, [0.1, 0.5, 0.9]))
        w_sw = R["step_w"][pk]
        print(f"[safety] PREF weight={self.pref_weight} beta={a.pref_beta} "
              f"pairs={len(pk)} (switched {int(sw.sum())} of {n_all} rows, rows={rows_mode}; "
              f"{len(np.unique(R['scene_id'][pk]))} scenes; by split "
              + ", ".join(f"{s} {int((R['split'][pk] == s).sum())}"
                          for s in np.unique(R["split"][pk]))
              + f")  anchor rows={len(self._pref_anchor_ix)} (success_train kept) "
              f"x {self.pref_anchor_weight}", flush=True)
        print(f"[safety] PREF batch {a.pref_batch} pairs + {a.pref_anchor_batch} anchor rows "
              f"x {int(getattr(a, 'pref_steps', 1))} step(s)/iter; t ~ U[{self.pref_t[0]}, {self.pref_t[1]}) "
              f"(k_trainable {self.actor.k_trainable}); anchor t ~ U[0, "
              f"{self.actor.k_trainable})", flush=True)
        print(f"[safety] PREF pair check: target/rejected {R['target'].shape[1:]} both, "
              f"identical off the edit; edit peak |d| p10/50/90 = {q(edit_peak[pk])} m; "
              f"round trip of the rejected draw p50/p99/max = "
              f"{np.quantile(R['rt_err_m'][pk], 0.5):.3f} / "
              f"{np.quantile(R['rt_err_m'][pk], 0.99):.3f} / {R['rt_err_m'][pk].max():.3f} m",
              flush=True)
        print(f"[safety] PREF step weights: edit_weight {lam} above {dlt} m -> "
              f"{float((w_sw > 1).sum() / max((w_sw > 0).sum(), 1)):.3f} of covered pair "
              f"steps up-weighted", flush=True)

    def pref_grads(self, it: int, sub: int = 0):
        """Gradient of ``pref_loss + pref_anchor_weight * base_anchor`` on one
        batch of pairs and one batch of success rows, and its diagnostics."""
        import functools
        import optax
        a = self.args
        P = self._pref
        rng = np.random.default_rng(int(a.data_seed) * 32452843 + 11 * int(it)
                                    + 1000033 * int(sub))
        nb, na = int(a.pref_batch), int(a.pref_anchor_batch)
        ix = rng.choice(self._pref_ix, size=nb, replace=len(self._pref_ix) < nb)
        if self._jit_pref is None:
            self._jit_pref = jax.jit(jax.value_and_grad(functools.partial(
                self.actor.pref_loss, beta=float(a.pref_beta), t_min=self.pref_t[0],
                t_max=self.pref_t[1]), has_aux=True))
        self.rng, kp, ka = jax.random.split(self.rng, 3)
        tr = lambda x: jnp.transpose(jnp.asarray(x, jnp.float32), (0, 2, 1))
        (loss, aux), g = self._jit_pref(
            self.actor._tparams, tr(P["target"][ix]), tr(P["rejected"][ix]),
            jnp.asarray(P["cond1"][ix]), jnp.asarray(P["cond2"][ix]),
            jnp.asarray(P["mask"][ix]), jnp.asarray(P["step_w"][ix]), kp)
        m = {f"pref/{k}": float(np.asarray(v)) for k, v in aux.items()}
        m["pref/loss"] = float(np.asarray(loss))
        m["pref/grad_norm"] = float(np.asarray(optax.global_norm(g)))
        if not self._pref_init_checked:
            # The plan's one sanity check: on a fresh start the model IS the
            # reference, so z = 0 and the loss is log 2. Only fp noise may differ.
            self._pref_init_checked = True
            if self._adam_steps == 0:
                # What must hold is that the trainable copy IS the base: its params
                # bit-identical. The LOSS is only equal up to float noise -- the two
                # forwards trace differently (params traced vs module constants), and
                # job 71201 measured a 4.8e-6 gap on a 0.013 loss. beta multiplies
                # that into z (|z| 0.01 at beta 1e4, loss 0.696 vs log 2 = 0.693), so
                # "loss == log 2" is not a beta-free test. Check the params exactly
                # and the loss gap relative to the loss, which no beta touches.
                from flax import nnx
                _, fp, _ = nnx.split(self.actor.frozen_denoise, nnx.Param, ...)
                a_l, b_l = (jax.tree_util.tree_leaves(x) for x in (self.actor._tparams, fp))
                same = len(a_l) == len(b_l) and all(
                    np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a_l, b_l))
                rel = m["pref/init_gap"] / max(m["pref/good_ref_loss"], 1e-12)
                print(f"[safety] PREF init check: params identical to the base {same}; "
                      f"max |model - ref| loss {m['pref/init_gap']:.2e} = {rel:.1e} of the "
                      f"ref loss (float noise); at beta {a.pref_beta} that is |z| "
                      f"{m['pref/z_abs']:.2e}, loss {m['pref/loss']:.6f} vs log 2 "
                      f"{np.log(2.0):.6f}", flush=True)
                if not same or rel > 1e-2:
                    raise SystemExit("[safety] PREF init check FAILED: the trainable copy "
                                     "is not the reference at step 0")
        lam = self.pref_anchor_weight
        m["pref/anchor_loss"] = m["pref/anchor_grad_norm"] = 0.0
        if lam > 0 and na > 0:
            ia = rng.choice(self._pref_anchor_ix, size=na,
                            replace=len(self._pref_anchor_ix) < na)
            if self._jit_pref_anchor is None:
                self._jit_pref_anchor = jax.jit(jax.value_and_grad(functools.partial(
                    self.actor.base_anchor_loss, t_max=int(self.actor.k_trainable))))
            # x_t is formed from the base's own draw (`rejected` == `target` on a
            # kept row): the tether is charged where the base puts its mass.
            al, ga = self._jit_pref_anchor(
                self.actor._tparams, jnp.asarray(P["rejected"][ia], jnp.float32),
                jnp.asarray(P["cond1"][ia]), jnp.asarray(P["cond2"][ia]),
                jnp.asarray(P["mask"][ia]), jnp.ones(na, jnp.float32), ka)
            m["pref/anchor_loss"] = float(np.asarray(al))
            m["pref/anchor_grad_norm"] = float(np.asarray(optax.global_norm(ga)))
            g = jax.tree_util.tree_map(lambda x, y: x + lam * y, g, ga)
        return g, m

    def preserve_ids(self, it: int) -> tuple[list[int], int]:
        """Scene ids for this iteration's preservation batch, and how many are replay.

        Fixed size (one compiled shape). Replayed scenes first -- successes the
        CURRENT policy broke in a recent collection rollout -- then hard
        successes, then uniform ones. Seeded by (data_seed, it) so a resumed run
        draws the same batches it would have."""
        a = self.args
        P = int(a.preserve_scenes)
        rng = np.random.default_rng(int(a.data_seed) * 7919 + 104729 * int(it))
        rep = []
        if self._replay:
            cand = sorted(self._replay)
            k = min(int(a.preserve_replay_per_iter), len(cand), P)
            rep = [int(i) for i in rng.choice(cand, size=k, replace=False)]
            for i in rep:
                self._replay[i] -= 1
                if self._replay[i] <= 0:
                    del self._replay[i]
        rest = P - len(rep)
        n_hard = 0
        if self._preserve_hard is not None and self._preserve_hard.size:
            n_hard = min(int(round(float(a.preserve_hard_frac) * rest)), self._preserve_hard.size)
        hard = rng.choice(self._preserve_hard, size=n_hard, replace=False).tolist() if n_hard else []
        uni = rng.choice(self._preserve_pool, size=rest - n_hard,
                         replace=self._preserve_pool.size < rest - n_hard).tolist()
        return rep + [int(i) for i in hard] + [int(i) for i in uni], len(rep)

    def note_broken(self, solved_b, clean_b):
        """Add success-pool rows the policy just failed to the replay buffer.

        The success pool is not individually verified (~2% of it the base fails
        too); those rows are harmless here because preservation weights every row
        by whether the BASE solved it on that draw."""
        if self.preserve_weight <= 0:
            return 0
        a = self.args
        succ = np.isin(self.sampler.group_labels(self.sampler.last_ids), self._success_groups)
        broken = succ & (np.asarray(solved_b) < 0.5) & (np.asarray(clean_b) > 0.5)
        ids = np.asarray(self.sampler.last_ids)[broken]
        for i in ids.tolist():
            self._replay.pop(int(i), None)             # re-insert = refresh to newest
            self._replay[int(i)] = int(a.preserve_replay_uses)
        while len(self._replay) > int(a.preserve_replay):
            self._replay.pop(next(iter(self._replay)))  # oldest first
        return int(ids.size)

    def collect_preserve_contexts(self, it: int):
        """Base rollout on the preservation batch -> contexts with base targets."""
        ids, n_rep = self.preserve_ids(it)
        state = self.sampler.load_explicit(ids)
        self.rng, rk = jax.random.split(self.rng)
        contexts, _ = self.rollout(state, train=True, params=self._base_tparams,
                                   rng=rk, keep_prefix=True)
        solved = jnp.asarray(self._last_scene["solved"], jnp.float32)
        for c in contexts:
            x = self.actor.sample_late_for(self._base_tparams, c.cond1, c.cond2, c.mask,
                                           c.x_prefix, c.key_late)
            c.preserve_target = self.env.norm_to_world_with(jnp.transpose(x, (0, 2, 1)), c.aux)
            c.preserve_w = c.alive * solved
        return state, contexts, {"preserve_n": len(ids), "preserve_replayed": n_rep,
                                 "preserve_base_solved": float(jnp.mean(solved))}

    def _preserve_core(self, params, dyn, *, n):
        """Matched-continuation loss, pure in ``params`` (jitted by the caller)."""
        a = self.args
        x = self.actor.sample_late_for(params, dyn["cond1"], dyn["cond2"], dyn["mask"],
                                       dyn["x_prefix"], dyn["key_late"])
        world = self.env.norm_to_world_with(jnp.transpose(x, (0, 2, 1)), dyn["aux"])
        return preserve_loss(world, dyn["target"], dyn["w"], n=n,
                             s_pos=float(a.preserve_scale_pos),
                             s_yaw=float(a.preserve_scale_yaw),
                             s_speed=float(a.preserve_scale_speed))

    def preserve_grads(self, ctx):
        if self._jit_preserve is None:
            self._jit_preserve = jax.jit(
                jax.value_and_grad(self._preserve_core, has_aux=True),
                static_argnames=("n",))
        n = int(min(self.preserve_window, ctx.preserve_target.shape[1] - 1))
        dyn = {"cond1": ctx.cond1, "cond2": ctx.cond2, "mask": ctx.mask, "aux": ctx.aux,
               "x_prefix": ctx.x_prefix, "key_late": ctx.key_late,
               "target": ctx.preserve_target, "w": ctx.preserve_w}
        dyn = jax.tree_util.tree_map(lambda v: jnp.asarray(v, dtype=v.dtype), dyn)
        (loss, aux), grads = self._jit_preserve(self.actor._tparams, dyn, n=n)
        return grads, loss, aux

    def _neighbor_state(self, state):
        """The state the safety term reads the OTHER agents' boxes off.

        ``--neighbor_source log`` (the default) returns ``None``, which is what
        ``candidate_safety`` already meant by "read ``log_trajectory``" -- the
        gradient is then byte-for-byte what it was.

        ``sim`` splices the env's post-rollout ``sim_trajectory`` (and its
        timestep) into the caller's state and hands that back. The substitution
        is deliberate and cheap: the loaded state ALREADY carries a
        ``sim_trajectory`` of that exact shape through every jitted gradient, so
        this costs nothing extra, and ``candidate_safety`` keeps reading the ego
        extents, the ego z and the roadgraph off the untouched ``log_trajectory``
        while the neighbours come from the executed motion.

        Correctness rests on the gradient pass running AFTER the collection
        rollout: by then the window every context scores has been driven, so the
        NPCs' reactive poses are written. `neighbor_window` raises on a window
        past ``timestep`` rather than reading invalid placeholders and reporting
        a clear road -- see its precondition.

        Staleness is one policy update, exactly like the contexts themselves:
        the neighbours reacted to the collection-time weights, not to the
        candidate the gradient is currently proposing. Making them react to the
        live candidate would mean re-stepping the simulator inside the gradient
        at every step, and the IDM's lead selection is an argmin over an overlap
        indicator -- zero gradient almost everywhere -- so the extra fidelity
        would buy no extra signal, only cost.
        """
        if not self._neighbor_from_sim:
            return None
        end = self.env.sim_state
        # Canonicalize both spliced leaves to the dtypes the loaded state already
        # carries. `_free_core` is jitted on `state`; a leaf whose abstract
        # signature flips between iterations (weak-typed host int vs int32 device
        # array) recompiles the whole gradient, which is the bug bench job 51578
        # found in `dyn` and the reason that dict is canonicalized too.
        return state.replace(
            sim_trajectory=jax.tree_util.tree_map(
                lambda new, old: jnp.asarray(new, dtype=old.dtype),
                end.sim_trajectory, state.sim_trajectory),
            timestep=jnp.asarray(end.timestep, dtype=jnp.int32))

    def _loss_for(self, state, ctx, rng, bc_ctx=None):
        aw = self.args.anchor_weight
        bw = float(getattr(self.args, "bc_weight", 0.0))
        baw = float(getattr(self.args, "base_anchor_weight", 0.0))
        expert = None
        if aw > 0:
            # The anchor's key is split fresh per gradient step, NOT a literal
            # constant. `diffusion._loss_impl` splits this one key into both the
            # timestep `t` and the noise `eps`, so a fixed key does not compute
            # the denoising loss `E_{t,eps}[.]` at all -- it descends a single
            # Monte-Carlo slice of it, the same 16 `(t, eps)` draws for every
            # context of every iteration. That is what a 200-iteration run was
            # doing until 2026-09-03: 1600 Adam steps memorizing one noise draw,
            # which is why the `anchor_weight=1, safety_weight=0` control (a
            # plain BC continuation, optimizing nothing new) came out as the
            # WORST arm of the 09-02 sweep -- success-heldout solved 0.995 ->
            # 0.639. Drawn once per step and not inside `loss_fn`, for the same
            # reason the unroll's `t_start` is: `value_and_grad` re-traces, and
            # the gradient must belong to one fixed objective.
            rng, key_anchor = jax.random.split(rng)
            h = int(self.actor.diffusion.denoise_fn.horizon)
            expert = (ctx.expert_target[:, :h, :], ctx.cond1, ctx.cond2, ctx.mask,
                      key_anchor)

        bc = None
        if bw > 0 and bc_ctx is not None:
            # Split per gradient step, exactly as the anchor's key is: this key
            # is what `_loss_impl` turns into BOTH the timestep `t` and the noise
            # `eps`, so holding it fixed descends one Monte-Carlo slice of the
            # denoising loss instead of the loss. That bug (a literal PRNGKey(7))
            # is what made the 09-02 BC control the worst arm of six.
            rng, key_bc = jax.random.split(rng)
            h = int(self.actor.diffusion.denoise_fn.horizon)
            bc = (bc_ctx.expert_target[:, :h, :], bc_ctx.cond1, bc_ctx.cond2,
                  bc_ctx.mask, key_bc)

        base_anchor_args = None
        if baw > 0:
            # Split once per gradient step, exactly as the anchor/bc keys are:
            # `base_anchor_loss` turns this key into the (t, eps) the KL is a
            # Monte-Carlo estimate of, so a fixed key would descend one slice of
            # it. Evaluated on the POLICY's own visited states (`ctx`), seeded by
            # noising the log there -- the states where "did we drift from base?"
            # is the question. It needs no second collection pass, so it is a
            # do-no-harm leash available even at --bc_weight 0.
            rng, key_base = jax.random.split(rng)
            h = int(self.actor.diffusion.denoise_fn.horizon)
            # Success-only: zero the anchor's weight on failure rows so it never
            # opposes the safety term on a scene we are trying to change. The
            # batch composition is fixed for the whole iteration, so one [B] mask
            # applies to every replan context.
            ba_alive = ctx.alive
            if self._anchor_mask is not None:
                ba_alive = ba_alive * self._anchor_mask
            base_anchor_args = (ctx.expert_target[:, :h, :], ctx.cond1, ctx.cond2,
                                ctx.mask, ba_alive, key_base)

        if self.unroll_steps:
            # The corruption level is drawn once *per gradient step*, not inside
            # loss_fn: `value_and_grad` would otherwise differentiate a different
            # objective on every re-trace, and the reported t_start would not be
            # the one the gradient came from.
            rng, key_t = jax.random.split(rng)
            t_start_b = jax.random.randint(
                key_t, (int(ctx.cond1.shape[0]),),
                self.t_end + 1, self.t_start_max + 1, dtype=jnp.int32)
            t_grid = unroll_timestep_grid(t_start_b, self.unroll_steps, self.t_end)
            x0_bct = self._expert_x0_bct(ctx)
            t_start_mean = jnp.mean(t_start_b.astype(jnp.float32))

            def propose(params):
                return self._propose_unrolled(params, ctx, x0_bct, t_grid, rng)
        else:
            t_start_mean = jnp.asarray(float(self.actor.diffusion.timesteps - 1))

            def propose(params):
                return self._propose(params, ctx, rng)

        def loss_fn(params):
            world = propose(params)
            terms = candidate_safety(self.objcfg, world, state=state, t0=ctx.t0,
                                     n_steps=ctx.n_steps,
                                     alive_b=(ctx.alive if ctx.safety_alive is None
                                              else ctx.safety_alive),
                                     neighbor_state=(state if self._neighbor_from_sim
                                                     else None))
            safety = terms["total"]
            tw = self._anchor_tw
            anchor = (self.actor._expert_loss(params, expert, ctx.alive, tw)
                      if expert is not None else jnp.asarray(0.0))
            # The BC pass has its own alive mask (its own expert-clean gate and
            # its own terminations); using `ctx.alive` here would weight the
            # human's states by which scenes the POLICY was still driving.
            bc_loss = (self.actor._expert_loss(params, bc, bc_ctx.alive, tw)
                       if bc is not None else jnp.asarray(0.0))
            base_anchor = (self.actor.base_anchor_loss(params, *base_anchor_args,
                                                       time_weight=tw)
                           if base_anchor_args is not None else jnp.asarray(0.0))
            total = (self.objcfg.safety_weight * safety + aw * anchor
                     + bw * bc_loss + baw * base_anchor)
            return total, {"safety": safety, "anchor": anchor, "bc": bc_loss,
                           "base_anchor": base_anchor,
                           "edge": jnp.sum(terms["edge_loss"]) / max(ctx.n_steps, 1),
                           "agent": jnp.sum(terms["agent_loss"]) / max(ctx.n_steps, 1),
                           "tail_share": (jnp.sum(terms["tail_loss"])
                                          / jnp.maximum(jnp.sum(terms["loss"]),
                                                        1e-8)),
                           "n_neighbors": terms["n_neighbors"],
                           "t_start": t_start_mean}
        return loss_fn

    # ------------------------------------------------------------------ #
    def _free_core(self, params, state, dyn, keys, *, n_steps, h,
                   has_expert, has_bc, has_base, has_prefix=False,
                   has_unroll=False):
        """Free-running safety loss as a pure function of ``params`` + arrays.

        The traced twin of :meth:`_loss_for`'s ``loss_fn`` for ``unroll_steps==0``:
        every per-call value is an explicit argument (``dyn`` arrays, ``keys``),
        every per-run choice is a static kwarg (``t0``/``n_steps``/``h`` and the
        branch flags), so one ``jax.jit`` executable is reused across iterations
        instead of the old fresh-closure trace that ran the anchor's extra
        denoiser evals un-fused and eager every step. Branch flags are Python
        booleans (static), so an inactive term compiles to nothing and its
        placeholder arrays in ``dyn`` are pruned. Must stay bit-for-bit equivalent
        to ``loss_fn``; keys are drawn by the caller for the same fixed-objective
        reason ``_loss_for`` documents."""
        aw = float(self.args.anchor_weight)
        bw = float(getattr(self.args, "bc_weight", 0.0))
        baw = float(getattr(self.args, "base_anchor_weight", 0.0))
        # `t0` is traced (dyn), so one executable serves all replans; it used to
        # be static and compiled once per replan index (~6 min at startup).
        t0 = dyn["t0"]
        if has_unroll:
            # Teacher-forced unroll (`--unroll_steps K`). Routed through here
            # rather than through `_loss_for`'s eager closure since 2026-09-14:
            # that closure re-traced the whole unrolled sampler + geometry graph
            # once PER CONTEXT PER ITERATION, and measured on the first proposal
            # sweep it cost 34.5 s/iter against the jitted free-running path's
            # 5.96 -- i.e. the "cheap" proposal, whose whole claim is K network
            # evaluations instead of 100, ran 6x SLOWER than the thing it was
            # meant to undercut. The arithmetic was never wrong; the tracing
            # overhead simply dwarfed it.
            traj_ncT = self.actor.unrolled_sample_for(
                params, dyn["cond1"], dyn["cond2"], dyn["mask"],
                dyn["x0_bct"], dyn["t_grid"], keys["propose"],
                grad_steps=self.unroll_grad_steps,
                eta=float(self.args.unroll_eta))
        elif has_prefix:
            traj_ncT = self.actor.sample_late_for(
                params, dyn["cond1"], dyn["cond2"], dyn["mask"],
                dyn["x_prefix"], dyn["key_late"])
        else:
            traj_ncT = self.actor.sample_world_for(
                params, dyn["cond1"], dyn["cond2"], dyn["mask"], keys["propose"])
        world = self.env.norm_to_world_with(jnp.transpose(traj_ncT, (0, 2, 1)),
                                             dyn["aux"])
        # `self._neighbor_from_sim` is read at TRACE time and baked into the
        # executable, which is correct because it is set once at startup. The
        # caller has already spliced the executed NPC motion into `state`
        # (`_neighbor_state`), so the neighbours come off `state.sim_trajectory`
        # and the ego extents / z / roadgraph still come off `log_trajectory`.
        terms = candidate_safety(self.objcfg, world, state=state, t0=t0,
                                 n_steps=n_steps, alive_b=dyn["safety_alive"],
                                 neighbor_state=(state if self._neighbor_from_sim
                                                 else None))
        safety = terms["total"]
        tw = self._anchor_tw
        anchor = (self.actor._expert_loss(
            params, (dyn["expert_target"][:, :h, :], dyn["cond1"], dyn["cond2"],
                     dyn["mask"], keys["anchor"]), dyn["alive"], tw)
            if has_expert else jnp.asarray(0.0))
        bc_loss = (self.actor._expert_loss(
            params, (dyn["bc_target"][:, :h, :], dyn["bc_cond1"], dyn["bc_cond2"],
                     dyn["bc_mask"], keys["bc"]), dyn["bc_alive"], tw)
            if has_bc else jnp.asarray(0.0))
        base_anchor = (self.actor.base_anchor_loss(
            params, dyn["expert_target"][:, :h, :], dyn["cond1"], dyn["cond2"],
            dyn["mask"], dyn["ba_alive"], keys["base"], time_weight=tw)
            if has_base else jnp.asarray(0.0))
        total = (self.objcfg.safety_weight * safety + aw * anchor
                 + bw * bc_loss + baw * base_anchor)
        aux = {"safety": safety, "anchor": anchor, "bc": bc_loss,
               "base_anchor": base_anchor,
               "edge": jnp.sum(terms["edge_loss"]) / max(n_steps, 1),
               "agent": jnp.sum(terms["agent_loss"]) / max(n_steps, 1),
               # Fraction of the safety loss charged to corners/agents that are
               # already CLEAR of their margin -- the softplus barrier's tail,
               # which scales with how many agents and steps were summed over
               # rather than with anything the policy did. Exactly 0 under a
               # compact-support `--barrier_shape`; read it on the softplus arm
               # to know whether the shape can matter here at all.
               "tail_share": (jnp.sum(terms["tail_loss"])
                              / jnp.maximum(jnp.sum(terms["loss"]), 1e-8)),
               # Valid ego-neighbour pairs per scored step. The agent barrier
               # reads 0 both on an empty road and on a misaddressed neighbour
               # window, and under `--neighbor_source sim` the second is a real
               # possibility (`sim_trajectory` is invalid past `timestep`). This
               # is the number that separates them, so it belongs in the curve
               # rather than in an assertion that only fires in eager mode.
               "n_neighbors": terms["n_neighbors"],
               # The noise level the gradient was actually taken from. Under the
               # unroll that is the drawn corruption level (per-sample, so report
               # its mean); free-running always enters at the top of the chain.
               "t_start": (jnp.mean(dyn["t_grid"][0].astype(jnp.float32))
                           if has_unroll
                           else jnp.asarray(float(self.actor.diffusion.timesteps - 1)))}
        return total, aux

    def _free_grad_fn(self):
        """Cached ``jit(value_and_grad(_free_core))``; grad w.r.t. ``params``."""
        if self._jit_free_grad is None:
            self._jit_free_grad = jax.jit(
                jax.value_and_grad(self._free_core, has_aux=True),
                static_argnames=("n_steps", "h", "has_expert", "has_bc",
                                 "has_base", "has_prefix", "has_unroll"))
        return self._jit_free_grad

    def _grads_for(self, state, ctx, rng, bc_ctx=None, propose_key=None):
        """(grads, loss, aux) w.r.t. the trainable denoiser for one context.

        Free-running (``unroll_steps==0``) goes through the cached jitted
        :meth:`_free_core`; the teacher-forced unroll keeps the un-jitted
        closure. The key derivation below mirrors :meth:`_loss_for` exactly --
        anchor, then bc, then base, with ``propose`` taking the remainder -- so
        the two paths differentiate the same objective from the same seed."""
        # The gradient is a TRAINING pass, so it samples on the training chain --
        # restated here because an eval rollout in between will have left the eval
        # value on the actor. It matters only for the first call, which is the one
        # that traces `_jit_free_grad` and bakes the value in as a closure
        # constant; every later call reuses that executable regardless. Setting it
        # unconditionally keeps the traced program and the eager unroll path
        # agreeing on which chain the objective is defined over.
        self.actor.num_sample_steps = self._steps_train
        # `--unroll_eager` keeps the pre-2026-09-14 un-jitted closure. It is the
        # equivalence reference for the jitted branch below, not a mode to train
        # in: it re-traces once per context per iteration (34.5 s/iter measured,
        # vs 5.96 jitted).
        if self.unroll_steps and bool(getattr(self.args, "unroll_eager", False)):
            return self.actor.safety_grads(self._loss_for(state, ctx, rng, bc_ctx))

        aw = float(self.args.anchor_weight)
        bw = float(getattr(self.args, "bc_weight", 0.0))
        baw = float(getattr(self.args, "base_anchor_weight", 0.0))
        h = int(self.actor.diffusion.denoise_fn.horizon)
        has_expert = aw > 0
        has_bc = bw > 0 and bc_ctx is not None
        has_base = baw > 0

        # Same split order and stream as `_loss_for`, so the jitted and closure
        # paths are seed-identical. `propose` consumes what remains, exactly as
        # the free-running `else` branch there does.
        k_anchor = k_bc = k_base = None
        if has_expert:
            rng, k_anchor = jax.random.split(rng)
        if has_bc:
            rng, k_bc = jax.random.split(rng)
        if has_base:
            rng, k_base = jax.random.split(rng)
        # The unroll draws its corruption level from the stream NEXT, before the
        # proposal key -- the same order and the same reassignment `_loss_for`
        # uses, so the jitted and eager paths agree seed-for-seed and
        # `--unroll_eager` stays a usable reference. Drawn out here rather than
        # inside the traced core for the reason that docstring gives: a
        # `value_and_grad` that re-traces must not silently change its objective.
        x0_bct = t_grid = None
        if self.unroll_steps:
            rng, key_t = jax.random.split(rng)
            t_start_b = jax.random.randint(
                key_t, (int(ctx.cond1.shape[0]),),
                self.t_end + 1, self.t_start_max + 1, dtype=jnp.int32)
            t_grid = unroll_timestep_grid(t_start_b, self.unroll_steps, self.t_end)
            x0_bct = self._expert_x0_bct(ctx)
        # `propose_key` pins the full-chain sample's key (equivalence bench only).
        k_propose = rng if propose_key is None else propose_key
        # Inactive branches never execute (static flag), but their keys still
        # ride in the `keys` pytree; reuse the propose key as a placeholder.
        keys = {"propose": k_propose,
                "anchor": k_anchor if k_anchor is not None else k_propose,
                "bc": k_bc if k_bc is not None else k_propose,
                "base": k_base if k_base is not None else k_propose}

        # base-anchor alive mask: success-only zeros the anchor on failure rows.
        ba_alive = ctx.alive
        if self._anchor_mask is not None:
            ba_alive = ba_alive * self._anchor_mask
        # `expert_target` is None on a pure safety-only run (nothing attaches it);
        # it is only read inside static-false branches then, so a placeholder that
        # keeps the pytree structure stable across contexts is enough.
        expert_target = (ctx.expert_target if ctx.expert_target is not None
                         else jnp.zeros((ctx.cond1.shape[0], 1, 1), jnp.float32))
        # bc arrays: real when active, else placeholders (pruned by the flag).
        if has_bc:
            bc_target = bc_ctx.expert_target
            bc_cond1, bc_cond2 = bc_ctx.cond1, bc_ctx.cond2
            bc_mask, bc_alive = bc_ctx.mask, bc_ctx.alive
        else:
            bc_target, bc_cond1, bc_cond2 = expert_target, ctx.cond1, ctx.cond2
            bc_mask, bc_alive = ctx.mask, ctx.alive
        dyn = {"cond1": ctx.cond1, "cond2": ctx.cond2, "mask": ctx.mask,
               "aux": ctx.aux, "alive": ctx.alive, "ba_alive": ba_alive,
               "safety_alive": (ctx.alive if ctx.safety_alive is None
                                else ctx.safety_alive),
               "expert_target": expert_target,
               "bc_target": bc_target, "bc_cond1": bc_cond1, "bc_cond2": bc_cond2,
               "bc_mask": bc_mask, "bc_alive": bc_alive,
               "t0": jnp.asarray(ctx.t0, jnp.int32)}
        has_unroll = bool(self.unroll_steps)
        # The unrolled proposal replaces the sampler, so a stored frozen prefix
        # is meaningless there (nothing runs the 100-step chain under gradient).
        has_prefix = ctx.x_prefix is not None and not has_unroll
        if has_prefix:
            dyn["x_prefix"], dyn["key_late"] = ctx.x_prefix, ctx.key_late
        if has_unroll:
            dyn["x0_bct"], dyn["t_grid"] = x0_bct, t_grid

        # Canonicalize to device arrays of their own dtype. The first replan's
        # context differs from later ones in some leaf's abstract signature
        # (weak type / host array), which recompiled the gradient once more on
        # replan 1 (bench job 51578: 4 cache entries where 2 were expected).
        dyn = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype), dyn)
        keys = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype), keys)

        (loss, aux), grads = self._free_grad_fn()(
            self.actor._tparams, state, dyn, keys,
            n_steps=ctx.n_steps, h=h, has_expert=has_expert, has_bc=has_bc,
            has_base=has_base, has_prefix=has_prefix, has_unroll=has_unroll)
        return grads, loss, aux

    # ------------------------------------------------------------------ #
    def rollout(self, state, *, train: bool, params=None, rng=None,
                log_states=None, keep_prefix: bool = False,
                store_transitions: bool = False, scene_ids=None):
        """Free-running closed loop under ``params``; returns contexts + metrics.

        With ``train=True`` no gradient is taken here -- this is the collection
        pass, and the policy is held fixed for the whole episode so the contexts
        it produces belong to one stationary batch.

        ``rng`` overrides the sampler stream and, when given, is NOT written back
        to ``self.rng``. Evaluation passes one so an eval is reproducible from its
        own seed alone. Sharing ``self.rng`` made every eval's sampler noise a
        function of how much training had run before it, which is why arms with
        identical weights disagreed at ``iter 0`` (success 0.69 vs 1.00) and why a
        single arm's solved rate swung +-0.15 between adjacent eval points.
        """
        params = self.actor._tparams if params is None else params
        key = self.rng if rng is None else rng
        # Which reverse chain this rollout samples on. The actor caches one
        # compiled sampler per (rl_mode, num_sample_steps), so flipping this is a
        # cache lookup after the first use of each value -- but it only reaches
        # the *eager* sampler calls below. The gradient pass bakes the value into
        # `_jit_free_grad` at trace time, which is why `_grads_for` re-asserts the
        # training value rather than trusting whatever the last rollout left here.
        self.actor.num_sample_steps = (self._steps_train if train
                                       else self._steps_eval)
        features = self.env.reset(state, seed=self.args.seed)
        # Log-driven collection is a *training* option only: an eval that followed
        # the human would measure the demonstration, not the policy.
        # `log_states` overrides --anchor_states so ONE iteration can collect
        # both distributions: the safety term needs the policy's own states (the
        # violations only exist there) and the BC term needs the dataset's (see
        # `collect_bc_contexts`). The flag alone couples them.
        log_states = (bool(train and self.args.anchor_states == "log")
                      if log_states is None else bool(log_states and train))
        log_world = self.env.ego_log_world_trajectory() if log_states else None
        # Harvesting the sampler's own transitions is a property of THIS call,
        # not of the run: the preservation pass also rolls out with `train=True`
        # (under the FROZEN base params), and storing its states would fill the
        # buffer with transitions the trainable copy never produced.
        store_tx = bool(store_transitions and self.tb is not None
                        and train and not log_states)
        n_stored = 0
        clean_b = self._gate_expert_clean(self.env.sim_state)
        alive = clean_b                          # dirty scenes contribute nothing
        contexts, ctx_expert = [], []
        offroad = np.zeros(self.env.batch_size)
        overlap = np.zeros(self.env.batch_size)
        reached = np.zeros(self.env.batch_size)
        # Per-world-step offroad marks + their absolute timestep, concatenated
        # across replans, so a viz job can place a red dot at exactly the sim
        # positions that left the road (aligned by absolute timestep, not by a
        # fragile offset from the episode end).
        off_step_marks: list[np.ndarray] = []
        exec_step_idx: list[np.ndarray] = []
        # Absolute timestep at which each scene first absorbed (collision / offroad
        # terminal / goal / horizon). -1 until it does; a viz job draws the path
        # only up to here so a frozen post-absorption tail is not plotted.
        done_at = np.full(self.env.batch_size, -1, dtype=np.int32)
        n_replans = 0

        for _ in range(self.args.max_replans):
            if float(jnp.sum(alive)) <= 0.0:
                break
            cond1, cond2 = self.env.compute_condition(features)
            mask = features["inst_valid"]
            t0 = self.env.timestep
            horizon = int(jnp.asarray(self.env.sim_state.log_trajectory.x).shape[-1])
            n_exec = min(self.args.replan_interval_steps, horizon - 1 - t0)
            # The SCORED window, which is `--safety_window` (>= n_exec when the
            # loss is charged past the replan) clipped by what the log can supply:
            # `neighbor_window` slices log_trajectory at t0+1..t0+n_steps, so a
            # window running off the end of the scenario would silently return a
            # short slice and mismatch the ego boxes.
            n_steps = min(self.safety_window, horizon - 1 - t0)
            if n_exec <= 0 or n_steps <= 0:
                break
            ctx = ReplanContext(t0, cond1, cond2, mask, self.env.last_aux, alive,
                                n_steps, n_exec)
            if train:
                contexts.append(ctx)
                ctx_expert.append(features["ego_trajectory"])

            key, rk = jax.random.split(key)
            if log_states:
                # The window `_execute_prefix` expects: index 0 is the anchor
                # (current) state, 1.. are the steps to drive. Same slice
                # `env.expert_replay_episode` uses.
                world = log_world[:, t0 : t0 + self.args.replan_interval_steps + 1, :]
            elif train and (store_tx or self.reuse_prefix or keep_prefix):
                if store_tx:
                    traj_ncT, pre, x_t_kbct, t_kb = \
                        self.actor.sample_world_with_prefix_trace(
                            params, ctx.cond1, ctx.cond2, ctx.mask, rk)
                else:
                    traj_ncT, pre = self.actor.sample_world_with_prefix(
                        params, ctx.cond1, ctx.cond2, ctx.mask, rk)
                # Only attach the prefix when the GRADIENT is going to replay it.
                # Storing transitions needs the same sampler call but must not
                # flip `_grads_for`'s `has_prefix` branch, or `--no-reuse_prefix`
                # would silently stop meaning what it says.
                if self.reuse_prefix or keep_prefix:
                    ctx.x_prefix, ctx.key_late = pre
                world = self.env.norm_to_world_with(
                    jnp.transpose(traj_ncT, (0, 2, 1)), ctx.aux)
                if store_tx:
                    n_stored += self._store_transitions(
                        ctx, x_t_kbct, t_kb, scene_ids, n_steps)
            else:
                world = self._propose(params, ctx, rk)
            out = self.env.step(trajectory_world_bt5=world)
            # `offroad`/`collision`/`reached` are shaped reward TERMS; the 0/1
            # indicators are the `_flag` keys, OR'd over the window by env.step.
            # Taking the term would score the penalty, not the event, and would
            # read as 0 whenever the shaping is off.
            comp = out.components
            off_b = np.asarray(comp["offroad_flag"], np.float32) > 0
            offroad = np.maximum(offroad, off_b)
            overlap = np.maximum(overlap, np.asarray(comp["collision_flag"], np.float32) > 0)
            reached = np.maximum(reached, np.asarray(comp["reached_flag"], np.float32) > 0)
            # Per-step offroad marks, aligned to absolute world timesteps
            # t0+1..t0+k (the steps env.step actually executed this replan).
            pso = out.per_step_components.get("offroad_flag")
            if pso is not None:
                pso = np.asarray(pso, np.float32) > 0     # [B, k]
                off_step_marks.append(pso)
                exec_step_idx.append(
                    t0 + 1 + np.arange(pso.shape[-1], dtype=np.int32))
            done = jnp.asarray(out.terminated_b) | jnp.asarray(out.truncated_b)
            last_exec = int(t0 + (pso.shape[-1] if pso is not None else n_exec))
            newly_done = (done_at < 0) & (np.asarray(done))
            done_at[newly_done] = last_exec
            alive = alive * (1.0 - done.astype(jnp.float32))
            features = out.features
            n_replans += 1

        if rng is None:
            self.rng = key
        if (self.args.anchor_weight > 0 or self.unroll_steps
                or float(getattr(self.args, "bc_weight", 0.0)) > 0
                or float(getattr(self.args, "base_anchor_weight", 0.0)) > 0):
            for c, et in zip(contexts, ctx_expert):
                c.expert_target = et
        clean_np = np.asarray(clean_b) > 0.5
        # Solved == the DPPO `solved`: reached the goal AND never off-road AND
        # never collided. Reported over the whole eval batch, and separately over
        # the clean-expert denominator (a scene whose own log is dirty is not a
        # fair charge against the policy).
        solved = (reached > 0.5) & (offroad < 0.5) & (overlap < 0.5)
        n_clean = int(clean_np.sum())
        metrics = {"offroad_rate": float(offroad.mean()),
                   "overlap_rate": float(overlap.mean()),
                   "reached_rate": float(reached.mean()),
                   "solved_rate": float(solved.mean()),
                   "solved_rate_clean": float(solved[clean_np].mean()) if n_clean else 0.0,
                   "replans": n_replans,
                   "tb_stored": float(n_stored),
                   "clean_frac": float(jnp.mean(clean_b))}
        # Any scene that never absorbed ran to the last executed step.
        last_idx = int(exec_step_idx[-1][-1]) if exec_step_idx else 0
        done_at = np.where(done_at < 0, last_idx, done_at)
        # Stash per-scene / per-step arrays for a dump job (eval only; harmless
        # in training since it is just numpy already computed above).
        self._last_scene = {
            "offroad": offroad, "overlap": overlap, "reached": reached,
            "solved": solved.astype(np.float32), "clean": clean_np.astype(np.float32),
            "done_at": done_at,
            "off_step_marks": (np.concatenate(off_step_marks, axis=-1)
                               if off_step_marks else np.zeros((self.env.batch_size, 0))),
            "exec_step_idx": (np.concatenate(exec_step_idx, axis=-1)
                              if exec_step_idx else np.zeros((0,), np.int32)),
        }
        return contexts, metrics

    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_ids(spec, name):
        if name not in spec["splits"]:
            raise SystemExit(f"[safety] split '{name}' absent from the splits "
                             f"JSON (have: {', '.join(sorted(spec['splits']))})")
        return [int(i) for i in spec["splits"][name]["scene_ids"]]

    def _read_splits_spec(self, path):
        """Parse a ``tools/build_eval_splits.py`` JSON, refusing another corpus.

        The scene ids are positions in the scan's ``shard_glob`` order, so a JSON
        built against a different shard list names *different scenes* and nothing
        would error -- every split would just be mislabelled. ``...-*`` and
        ``...-*-of-01000`` are different strings that resolve to the same shards,
        so the RESOLVED file lists are compared, not the patterns.
        """
        spec = json.loads(Path(path).read_text())
        blob_glob = spec.get("shard_glob")
        if blob_glob:
            import glob as _glob
            scan_shards = sorted(_glob.glob(blob_glob))
            train_shards = sorted(_glob.glob(self.args.shard_glob))
            if scan_shards != train_shards:
                raise SystemExit(
                    f"splits.json was built against a different corpus:\n"
                    f"  splits: {blob_glob} ({len(scan_shards)} shards)\n"
                    f"  train : {self.args.shard_glob} ({len(train_shards)} shards)\n"
                    f"scene ids are glob-relative; scoring them here would mislabel.")
        return spec

    def _resolve_train_pool(self):
        """The TRAINING pool as a union of named splits, or ``None`` for the corpus.

        ``--train_splits failure_train`` is what makes "trained on the agent's own
        failures" true rather than nominal. Without it the trainer draws uniformly
        over ~482k WOMD scenes, so a 48-scene split *called* ``failure_train`` is
        touched about 0.3 times in a 200-iteration run -- statistically identical
        to the held-out split it is supposed to be contrasted against. The 09-02
        sweep is exactly that experiment: its ``failure_train`` and
        ``failure_heldout`` curves agree because NEITHER was trained on, and no
        run in this package has yet trained on a scene it then scored.

        ``--train_split_cap`` caps each split (one value for all, or one per
        split; ``-1`` keeps it whole) so the pool's composition is deliberate. A
        failure-only pool carries no do-no-harm pressure at all; equal caps on
        ``failure_train,success_train`` are what make the four-panel probe
        readable, since ``success_train`` is 23k scenes against 564 failures and
        an uncapped union is 98% successes.
        """
        a = self.args
        names = [s.strip() for s in a.train_splits.split(",") if s.strip()]
        if not names:
            return None
        if not a.splits_json:
            raise SystemExit("--train_splits needs --splits_json")
        spec = self._read_splits_spec(a.splits_json)
        caps = [c.strip() for c in str(a.train_split_cap).split(",") if c.strip()]
        if len(caps) == 1:
            caps = caps * len(names)
        if len(caps) != len(names):
            raise SystemExit(f"--train_split_cap has {len(caps)} values for "
                             f"{len(names)} --train_splits")
        rng = np.random.default_rng(int(a.data_seed))
        pool: list[int] = []
        groups: list[np.ndarray] = []
        for name, cap in zip(names, caps):
            ids = np.asarray(sorted(set(self._split_ids(spec, name))), dtype=np.int64)
            c = int(cap)
            if 0 <= c < ids.size:
                ids = np.sort(rng.choice(ids, size=c, replace=False))
            pool.extend(int(i) for i in ids.tolist())
            groups.append(ids)
            print(f"[safety] train pool += {name}: {ids.size} scenes"
                  f"{'' if c < 0 else f' (cap {c})'}", flush=True)
        # Overlapping splits would make a per-batch quota meaningless (a scene in
        # both groups is drawn under both quotas), and the union below would
        # silently shrink the pool relative to the printed sizes.
        seen: set[int] = set()
        for name, ids in zip(names, groups):
            dup = seen & set(int(i) for i in ids.tolist())
            if dup:
                raise SystemExit(f"--train_splits overlap: {len(dup)} scene(s) of "
                                 f"'{name}' are already in an earlier split")
            seen |= set(int(i) for i in ids.tolist())
        self.train_pool_groups = groups
        self.train_pool_names = names
        # Which group the base-anchor is allowed to act on. The anchor is a
        # do-no-harm leash: on a FAILURE scene it opposes the safety term that is
        # trying to change exactly that scene, so restricting it to successes is
        # the point, not an optimization. Matched by name so the order of
        # --train_splits does not silently decide it.
        self._success_groups = [i for i, n in enumerate(names) if "success" in n]
        arr = np.unique(np.asarray(pool, dtype=np.int64))
        print(f"[safety] TRAIN POOL: {arr.size} scenes from {'+'.join(names)} -- "
              f"a deliberate distribution shift; --train_splits '' is the "
              f"uniform-corpus default", flush=True)
        return arr

    def _resolve_group_ratio(self):
        """``(groups, weights)`` for balanced per-batch sampling, or ``(None, None)``.

        Balancing is what lets the pool be ALL of the data instead of a capped
        sample of it: `success_train` is ~23k scenes against a few hundred
        failures, so a uniform draw over the union spends ~98% of every gradient
        on scenes the base already solves. A quota fixes the mix without
        discarding the 22k successes the way equal `--train_split_cap`s do.
        """
        a = self.args
        raw = str(getattr(a, "train_split_ratio", "") or "").strip()
        if not raw:
            return None, None
        if not self.train_pool_groups:
            raise SystemExit("--train_split_ratio needs --train_splits")
        w = [float(x) for x in raw.split(",") if x.strip()]
        if len(w) == 1:
            w = w * len(self.train_pool_groups)
        if len(w) != len(self.train_pool_groups):
            raise SystemExit(f"--train_split_ratio has {len(w)} values for "
                             f"{len(self.train_pool_groups)} --train_splits")
        share = np.asarray(w, dtype=np.float64)
        share = share / share.sum()
        print("[safety] BALANCED batches: " + "  ".join(
            f"{n}={s:.2f}" for n, s in zip(self.train_pool_names, share)), flush=True)
        return self.train_pool_groups, share

    # ------------------------------------------------------------------ #
    def _build_probe(self):
        """A small fixed set of named scenes, rolled out and dumped every probe
        point so the training curve has pictures under it and not only rates.

        One batch, four labelled groups: ``failure_train`` / ``success_train``
        (in the training pool when ``--train_splits`` puts them there) and
        ``failure_heldout`` / ``success_heldout`` (never trained on). That is the
        2x2 the question needs -- does the loss fix a violation it *is* trained
        on, does it leave a success it is trained on alone, and does either
        transfer to a scene it has never seen.

        A scene is marked ``trained`` only if it is in the sampler's pool AND not
        in the sampler's own held-out ids: `SceneSampler` removes those from the
        training draw, so a probe scene that landed there would be labelled
        "trained on" and never trained on -- the exact mislabelling this whole
        probe exists to rule out.
        """
        a = self.args
        js = a.probe_json or a.splits_json
        if not js or not a.probe_splits or int(a.probe_per_split) <= 0:
            return None
        if self.sampler is None:
            raise SystemExit("--probe_splits needs --shard_glob (corpus mode)")
        spec = self._read_splits_spec(js)
        pool = set(int(i) for i in (self.train_pool_ids.tolist()
                                    if self.train_pool_ids is not None else []))
        held = set(int(i) for i in getattr(self.sampler, "_eval_set", set()))
        rng = np.random.default_rng(int(a.probe_seed))
        ids, labels, trained = [], [], []
        for name in [s.strip() for s in a.probe_splits.split(",") if s.strip()]:
            cand = np.asarray(sorted(set(self._split_ids(spec, name))), dtype=np.int64)
            want = bool(pool) and any(int(i) in pool and int(i) not in held
                                      for i in cand.tolist())
            if want:
                cand = np.asarray([i for i in cand.tolist()
                                   if int(i) in pool and int(i) not in held],
                                  dtype=np.int64)
            elif name.endswith("_train"):
                print(f"[safety] probe: no {name} scene is in the training pool "
                      f"-- that panel is NOT a trained-on panel", flush=True)
            n = min(int(a.probe_per_split), int(cand.size))
            if n <= 0:
                continue
            for i in np.sort(rng.choice(cand, size=n, replace=False)).tolist():
                ids.append(int(i)); labels.append(name); trained.append(bool(want))
        if not ids:
            return None
        counts = "  ".join(f"{n}={labels.count(n)}" for n in dict.fromkeys(labels))
        print(f"[safety] probe: {len(ids)} scenes  {counts}  "
              f"(trained-on: {sum(trained)})  every {self.args.probe_every or self.args.eval_every} iters",
              flush=True)
        return {"ids": ids, "labels": labels, "trained": trained,
                "state": self.sampler.load_explicit(ids)}

    def _probe_arm(self, tag):
        """One probe rollout under the weights currently loaded.

        Uses `_eval_arm`, i.e. the FIXED eval seed, so the sampler noise is
        identical at every probe point and a difference between two frames of the
        slider is the weights' doing and not the sampler's.
        """
        m, sim, sc = self._eval_arm(self.probe_set["state"])
        B, T = int(sim.shape[0]), int(sim.shape[1])
        off = np.zeros((B, T), dtype=bool)
        idx = sc["exec_step_idx"].astype(int)
        keep = (idx >= 0) & (idx < T)
        if keep.any():
            off[:, idx[keep]] = (sc["off_step_marks"] > 0.5)[:, keep]
        fields = {f"path_{tag}": sim[:, :, :3].astype(np.float32),
                  f"off_{tag}": off,
                  f"done_{tag}": sc["done_at"].astype(np.int32)}
        for k in ("solved", "offroad", "overlap", "reached"):
            fields[f"{k}_{tag}"] = (sc[k] > 0.5).astype(np.int8)
        return fields, m

    def probe_static(self, out_dir):
        """Roadgraph, human log, goal and the FROZEN-BASE rollout -- written once.

        The map is constant across probe points and is ~99% of the bytes, so it
        lives in one file and each per-iteration dump carries only paths and
        flags. The base arm belongs here for the same reason: it is the
        "originally off-road" frame the slider opens on, and it never changes.
        """
        f = Path(out_dir) / "probe" / "static.npz"
        if f.exists():
            return
        f.parent.mkdir(parents=True, exist_ok=True)
        # reset_trainable_to_frozen re-initializes BOTH optimizer states, so a
        # mid-run call without this snapshot silently wipes Adam's moments.
        snap = (self.actor._tparams, self.actor._opt_state,
                self.actor._opt_state_expert)
        self.actor.reset_trainable_to_frozen()
        try:
            fields, m = self._probe_arm("base")
        finally:
            (self.actor._tparams, self.actor._opt_state,
             self.actor._opt_state_expert) = snap
        rg = self.env.sim_state.roadgraph_points
        np.savez_compressed(
            f,
            scene_ids=np.asarray(self.probe_set["ids"], np.int64),
            labels=np.asarray(self.probe_set["labels"]),
            trained=np.asarray(self.probe_set["trained"], np.int8),
            start_step=np.int32(self.env._start_timestep),
            goal_xy=np.asarray(self.env._goal_xy_b2, np.float32),
            log_path=np.asarray(self.env.ego_log_world_trajectory(), np.float32),
            rg_x=np.asarray(rg.x, np.float32), rg_y=np.asarray(rg.y, np.float32),
            rg_types=np.asarray(rg.types, np.int16),
            rg_valid=np.asarray(rg.valid),
            **fields)
        print(f"[safety] probe static -> {f}   base over probe scenes: solved "
              f"{m['solved_rate']:.3f} offroad {m['offroad_rate']:.3f} "
              f"overlap {m['overlap_rate']:.3f}", flush=True)

    def probe_dump(self, out_dir, it):
        """Dump this iteration's probe rollout: one small npz + one jsonl row."""
        d = Path(out_dir) / "probe"
        self.probe_static(out_dir)
        fields, _ = self._probe_arm("ft")
        np.savez_compressed(d / f"iter_{int(it):06d}.npz",
                            iter=np.int32(it),
                            adam_steps=np.int32(self._adam_steps), **fields)
        labels = np.asarray(self.probe_set["labels"])
        rec = {"iter": int(it), "adam_steps": int(self._adam_steps)}
        for name in dict.fromkeys(labels.tolist()):
            sel = labels == name
            for k in ("solved", "offroad", "overlap", "reached"):
                rec[f"{name}/{k}"] = float(fields[f"{k}_ft"][sel].mean())
        with (d / "probe_log.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def _classify_now(self, ids, chunk):
        """Per-scene solved flag at the CURRENTLY loaded weights, in id order.

        Rolls the given scene ids through `_probe_arm` (the fixed eval seed the
        slider uses), so a scene's bucket here is the outcome its last slider
        frame shows -- not a differently-seeded eval that would disagree.
        """
        keep = self.probe_set
        solved = []
        try:
            for i in range(0, len(ids), chunk):
                sub = ids[i:i + chunk]
                self.probe_set = {"state": self.sampler.load_explicit(sub)}
                fields, _ = self._probe_arm("ft")
                solved.extend(int(v) for v in fields["solved_ft"].tolist())
        finally:
            self.probe_set = keep
        return np.asarray(solved, dtype=bool)

    def probe_offline(self, run_dir, out_dir, k):
        """Re-select probe scenes from a finished run so all 8 transition
        buckets are populated, then re-roll them across the run's checkpoints.

        base-outcome (the split: failure_* = base failed, success_* = base
        solved) x now-outcome (solved at the last checkpoint) x train/heldout is
        2x2x2 = 8 columns. The in-run probe cannot show `regressed` because it
        pins its scenes before training; this scans the split populations at the
        trained weights and picks scenes that actually landed in each bucket.
        """
        a = self.args
        run_dir, out_dir = Path(run_dir), Path(out_dir)
        spec = self._read_splits_spec(a.splits_json)
        chunk = max(1, int(a.eval_chunk))
        cap = max(1, int(a.split_cap))
        rng = np.random.default_rng(int(a.probe_seed))
        ckpts = sorted(run_dir.glob("iter_*"), key=lambda p: int(p.name.split("_")[1]))
        if not ckpts:
            raise SystemExit(f"[probe-offline] no iter_* checkpoints in {run_dir}")
        last = ckpts[-1]
        print(f"[probe-offline] classifying at {last.name} over {cap}/split", flush=True)
        self.load(last)

        # base-failed vs base-solved x train/heldout -> now-outcome -> bucket.
        cat_of = {("failure", True): "recovered", ("failure", False): "stillfail",
                  ("success", True): "stillok",   ("success", False): "regressed"}
        pinned_ids, pinned_lab, pinned_tr = [], [], []
        counts = {}
        for split in ("failure_train", "success_train",
                      "failure_heldout", "success_heldout"):
            if split not in spec["splits"]:
                continue
            base_fail = split.startswith("failure")
            pool = "train" if split.endswith("_train") else "heldout"
            ids = sorted(set(self._split_ids(spec, split)))[:cap]
            solved = self._classify_now(ids, chunk)
            for now_ok in (True, False):
                cat = f"{cat_of[('failure' if base_fail else 'success', now_ok)]}_{pool}"
                sel = np.asarray(ids)[solved == now_ok]
                counts[cat] = int(sel.size)
                if sel.size == 0:
                    print(f"[probe-offline]   {cat:>18}: 0 scenes (empty bucket)", flush=True)
                    continue
                take = sel if sel.size <= k else rng.choice(sel, size=k, replace=False)
                for i in np.sort(take).tolist():
                    pinned_ids.append(int(i)); pinned_lab.append(cat)
                    pinned_tr.append(pool == "train")
                print(f"[probe-offline]   {cat:>18}: {sel.size} available, took {len(take)}",
                      flush=True)
        if not pinned_ids:
            raise SystemExit("[probe-offline] no scenes selected")
        self.probe_set = {"ids": pinned_ids, "labels": pinned_lab,
                          "trained": pinned_tr,
                          "state": self.sampler.load_explicit(pinned_ids)}
        print(f"[probe-offline] pinned {len(pinned_ids)} scenes across "
              f"{len(set(pinned_lab))} buckets; counts={counts}", flush=True)

        # Base (frozen) + one frame per checkpoint, into out_dir/probe/.
        (out_dir / "probe").mkdir(parents=True, exist_ok=True)
        self.probe_static(out_dir)
        for c in ckpts:
            it = int(c.name.split("_")[1])
            self.load(c)
            self._adam_steps = it
            self.probe_dump(out_dir, it)
            print(f"[probe-offline] dumped {c.name}", flush=True)
        print(f"[probe-offline] wrote {out_dir/'probe'}  ({len(ckpts)} frames + base)",
              flush=True)

    # ------------------------------------------------------------------ #
    def _load_eval_splits(self):
        """Load the fixed named eval splits from ``--splits_json`` for the
        training curve. Returns ``{split_name: state}``.

        Reuses the same JSON and args the ``--eval_only`` dump path uses
        (``--splits_json`` / ``--splits`` / ``--split_cap``). The split scene ids
        are positions in the scan's ``shard_glob`` order, so this refuses a glob
        whose RESOLVED shard list differs (``...-*`` and ``...-*-of-01000`` are
        different strings that resolve to the same shards, so compare the files,
        not the pattern).
        """
        a = self.args
        if not a.splits_json:
            return {}
        if self.sampler is None:
            raise SystemExit("--splits_json needs --shard_glob (corpus mode)")
        spec = self._read_splits_spec(a.splits_json)
        cap = max(1, int(a.split_cap))
        chunk = max(1, int(a.eval_chunk))
        out = {}
        for name in [s for s in a.splits.split(",") if s.strip()]:
            if name not in spec["splits"]:
                print(f"[safety] eval split '{name}' absent from {a.splits_json}; "
                      f"skipping", flush=True)
                continue
            ids = [int(i) for i in spec["splits"][name]["scene_ids"]][:cap]
            if not ids:
                continue
            # CHUNKED, and parked on the HOST. A split of 1000 scenes is one
            # batched state of ~2 GB whose rollout intermediates do not fit on a
            # 24 GB a5000; and four such splits resident on the device would not
            # fit even unrolled. `jax.device_get` drops each chunk to numpy, and
            # jax re-uploads the one chunk being rolled. Ragged last chunk costs
            # exactly one extra jit compile, cached for every later eval point.
            parts = []
            for i in range(0, len(ids), chunk):
                sub = ids[i:i + chunk]
                parts.append((jax.device_get(self.sampler.load_explicit(sub)),
                              len(sub)))
            out[name] = parts
            print(f"[safety] eval split {name}: {len(ids)} scenes (cap {cap}) "
                  f"in {len(parts)} chunk(s) of <= {chunk}", flush=True)
        return out

    def _marginal_weights(self):
        """``{split: weight}`` for the corpus-level rate, from the splits JSON.

        The splits are a STRATIFIED sample and the strata are wildly
        disproportionate: verified failures are ~1.5% of the corpus and verified
        successes ~98.5%. So the per-split curves cannot answer "did the overall
        solve rate go up", and they mislead about its SIGN -- measured on the
        first arm, failure 0.040 -> 0.364 and success 0.998 -> 0.962 read as two
        wins and are a net loss of 0.031, because +0.324 on a 1.5% stratum buys
        +0.005 while -0.036 on a 98.5% stratum costs -0.035.

        Empty for a splits file with no ``population`` block (everything built
        before 2026-09-08); the marginal is then simply not logged.
        """
        if not self.args.splits_json:
            return {}
        spec = self._read_splits_spec(self.args.splits_json)
        pop = spec.get("population") or {}
        strata = pop.get("strata") or {}
        w = {k: float(v["weight"]) for k, v in strata.items()
             if k in self.eval_splits}
        if len(w) != len(strata):
            # A partial marginal is not a marginal: it would silently reweight
            # the corpus to whichever splits happen to be evaluated.
            missing = sorted(set(strata) - set(w))
            print(f"[safety] marginal rate DISABLED: --splits does not include "
                  f"{missing}, and a subset would not be the corpus", flush=True)
            return {}
        return w

    def _marginal(self, per_split, prefix=""):
        """Weighted corpus-level rates from this eval point's per-split rates.

        ``per_split`` is ``{split: {metric: value}}``. Also reports the
        break-even success rate: the value the success split must hold for the
        marginal to match the base, given the failure split's current rate --
        the number that says whether an arm is worth anything at deployment.
        """
        w = self._marg_w
        if not w or any(k not in per_split for k in w):
            return {}
        metrics = ("solved_rate", "offroad_rate", "overlap_rate", "reached_rate")
        out = {f"{prefix}marginal/{m}": sum(w[k] * per_split[k][m] for k in w)
               for m in metrics if all(m in per_split[k] for k in w)}
        out[f"{prefix}marginal/coverage"] = self._marg_cov
        return out

    def evaluate(self, state=None, *, rng=None):
        """Closed-loop eval on ``state`` (default: the held-out batch), with the
        degeneracy diagnostics attached."""
        _, m = self.rollout(self.eval_state if state is None else state,
                            train=False, rng=rng)
        sim = np.asarray(self.env.ego_sim_world_trajectory())
        log = np.asarray(self.env.ego_log_world_trajectory())
        n = min(sim.shape[1], log.shape[1])
        d = np.linalg.norm(sim[:, :n, :2] - log[:, :n, :2], axis=-1)
        arc = np.linalg.norm(np.diff(sim[:, :n, :2], axis=1), axis=-1).sum(-1)
        arc_log = np.linalg.norm(np.diff(log[:, :n, :2], axis=1), axis=-1).sum(-1)
        m = dict(m)
        m["drift_m"] = float(d.mean())
        m["arc_m"] = float(arc.mean())
        m["arc_ratio_vs_log"] = float(arc.mean() / max(arc_log.mean(), 1e-9))
        # Carried only so `_eval_chunks` can aggregate a chunked split exactly;
        # stripped before anything is logged.
        m["_n"] = float(sim.shape[0])
        m["_arc_log_m"] = float(arc_log.mean())
        return m

    def _eval_chunks(self, chunks, *, rng=None):
        """Aggregate ``evaluate`` over a chunked split into one scene-weighted row.

        Every metric here is a per-scene mean, so a mean weighted by chunk size
        reproduces the number a single un-chunked batch would have given --
        EXCEPT ``arc_ratio_vs_log``, which is a ratio of two means and is
        recomputed from the summed arcs rather than averaged.
        """
        if not isinstance(chunks, list):
            return self.evaluate(chunks, rng=rng)
        acc: dict[str, float] = {}
        total = 0
        for state, n in chunks:
            m = self.evaluate(state, rng=rng)
            n = int(m.get("_n", n))
            for k, v in m.items():
                acc[k] = acc.get(k, 0.0) + float(v) * n
            total += n
        out = {k: v / max(total, 1) for k, v in acc.items()}
        out["arc_ratio_vs_log"] = (out["arc_m"] / max(out["_arc_log_m"], 1e-9))
        out["_n"] = float(total)
        return out

    def evaluate_fixed(self, state=None):
        """Eval averaged over ``--eval_repeats`` FIXED sampler seeds.

        The seeds are ``fold_in(PRNGKey(--eval_seed), r)`` -- a function of the
        flag and nothing else, so the same policy scores the same number at every
        iteration, in every arm, on every rerun. One rollout of 48 scenes has a
        binomial floor of ~0.07 on a rate near 0.8 before any policy noise; the
        measured swing was twice that, so averaging a few seeds is what makes an
        arm-vs-arm difference readable at all. Cost is linear in ``eval_repeats``.
        """
        reps = max(1, int(self.args.eval_repeats))
        target = self.eval_state if state is None else state
        acc: dict[str, list[float]] = {}
        for r in range(reps):
            for k, v in self._eval_chunks(
                    target, rng=jax.random.fold_in(self._eval_key, r)).items():
                acc.setdefault(k, []).append(float(v))
        return {k: float(np.mean(v)) for k, v in acc.items()
                if not k.startswith("_")}

    # ------------------------------------------------------------------ #
    def verify_splits(self, out_path, *, names, repeats, seed, chunk, caps=None,
                      ckpt=""):
        """Re-measure the BASE policy per scene on named splits; append JSONL rows.

        WHY. The split labels came from `scan_womd_headroom`, which scored the
        base DETERMINISTICALLY at `sigma_floor=0.03` under its own reward config.
        The training curve scores it with `rollout(train=False)` -- a stochastic
        sampler at a different floor, 8 replans, and the expert-clean gate. Those
        are different policies on the same scenes, and the gap is not small: the
        base solves 0.328 of `failure_heldout`, a split whose definition is
        "the base does not solve it". A third of the failure split was therefore
        already solved at iteration 0, which both dilutes the headroom and makes
        any improvement partly a re-measurement of the same noise.

        This runs the ACTUAL eval path (same `rollout`, same gate, same flags) at
        ``--verify_repeats`` seeds and keeps the per-scene outcome, so
        `rebuild_verified_splits` can keep only the scenes the deployed policy
        genuinely fails. The seed is ``--verify_seed``, deliberately NOT
        ``--eval_seed``: selecting scenes on the very draws they are later scored
        on would bake this sample's noise into the split.

        Resumable -- rows already in ``out_path`` are skipped, so a task killed by
        the time limit is fixed by resubmitting the same job.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if out_path.exists():
            for line in out_path.read_text().splitlines():
                try:
                    done.add(int(json.loads(line)["scene_id"]))
                except Exception:
                    continue        # torn last line from a killed task
        spec = self._read_splits_spec(self.args.splits_json)
        key = jax.random.PRNGKey(int(seed))
        t0 = time.time()
        caps = list(caps or [])
        if len(caps) == 1:
            caps = caps * len(names)
        if caps and len(caps) != len(names):
            raise SystemExit(f"--verify_cap has {len(caps)} values for "
                             f"{len(names)} --splits")
        for k, name in enumerate(names):
            ids = [int(i) for i in self._split_ids(spec, name)]
            # A cap is what keeps `success_train` (23k scenes) affordable: only
            # the held-out success set needs verifying, the rest of that split is
            # a training pool. The slice is the split's own order, so it is a
            # fixed prefix and a resumed job verifies the same scenes.
            cap = int(caps[k]) if caps else -1
            if 0 <= cap < len(ids):
                ids = ids[:cap]
            todo = [i for i in ids if i not in done]
            print(f"[verify] {name}: {len(ids)} scenes, {len(todo)} to do",
                  flush=True)
            for start in range(0, len(todo), chunk):
                sub = todo[start:start + chunk]
                state = self.sampler.load_explicit(sub)
                acc = None
                for r in range(int(repeats)):
                    self.rollout(state, train=False, rng=jax.random.fold_in(key, r))
                    ls = {k: np.asarray(v, np.float64)
                          for k, v in self._last_scene.items()
                          if k in ("solved", "offroad", "overlap", "reached", "clean")}
                    acc = ls if acc is None else {k: acc[k] + v for k, v in ls.items()}
                with out_path.open("a") as fh:
                    for j, sid in enumerate(sub):
                        fh.write(json.dumps({
                            "scene_id": int(sid), "split": name,
                            "repeats": int(repeats), "verify_seed": int(seed),
                            # "" = frozen base. Recorded per row so two files can
                            # never be silently paired against the wrong policy.
                            "ckpt": str(ckpt),
                            **{f"{k}_frac": float(acc[k][j] / repeats) for k in acc},
                        }) + "\n")
                done.update(sub)
                n = start + len(sub)
                print(f"[verify] {name} {n}/{len(todo)}  "
                      f"solved={float(acc['solved'].mean()/repeats):.3f}  "
                      f"{time.time() - t0:.0f}s", flush=True)
        print(f"[verify] wrote -> {out_path}", flush=True)

    def base_reference(self):
        """The FROZEN base policy's own scores on the same batches and seeds.

        Computed once and cached: with a fixed eval seed the base is a constant,
        so this costs one eval pass for the whole run and turns every later row
        into a paired comparison. Without it a rate of 0.83 is unreadable -- the
        base scores somewhere in 0.67-1.00 depending on the seed, and most of the
        movement attributed to fine-tuning in the 09-02 sweep was that spread.

        Restores the trained weights AND both optimizer states afterwards:
        ``reset_trainable_to_frozen`` re-initializes them, so calling it mid-run
        without a snapshot would silently wipe Adam's moments.
        """
        if self._base_ref is not None:
            return self._base_ref
        if not self.args.eval_base:
            self._base_ref = {}
            return self._base_ref
        # Disk cache. The in-memory cache only spans ONE process, but a free run
        # is a chain of segments, so every restart was re-paying a full eval pass
        # (2346 scenes x eval_repeats, ~17-34 min) to recompute a constant. The
        # base is the frozen checkpoint scored at a fixed seed, so it is
        # reproducible; key the cache on everything that changes it and recompute
        # on any mismatch rather than silently reusing a stale reference.
        cache = Path(self.args.out_dir) / "base_reference.json"
        key = {"ckpt": str(self.args.ckpt), "eval_seed": int(self.args.eval_seed),
               "eval_repeats": int(self.args.eval_repeats),
               "split_cap": int(self.args.split_cap),
               "splits": str(self.args.splits),
               "splits_json": str(self.args.splits_json),
               "eval_scenes": int(self.args.eval_scenes),
               "data_seed": int(self.args.data_seed),
               # The simulator settings change the base's own scores too; a
               # reference taken under another clock or agent model is stale.
               "world_dt_method": self.env.world_dt_method,
               "ego_vel_source": self.env.ego_vel_source,
               "sim_agents": self.env.sim_agents}
        if cache.exists():
            try:
                blob = json.loads(cache.read_text())
                if blob.get("key") == key:
                    self._base_ref = blob["ref"]
                    print(f"[safety] base reference: reused {cache} "
                          f"(saved one full eval pass)", flush=True)
                    return self._base_ref
                print("[safety] base reference: cache key changed, recomputing",
                      flush=True)
            except Exception as exc:
                print(f"[safety] base reference: unreadable cache ({exc}), "
                      f"recomputing", flush=True)
        snap = (self.actor._tparams, self.actor._opt_state,
                self.actor._opt_state_expert)
        self.actor.reset_trainable_to_frozen()
        try:
            ref = {f"base_eval_{k}": v for k, v in self.evaluate_fixed().items()}
            per_split = {}
            for name, st in self.eval_splits.items():
                per_split[name] = self.evaluate_fixed(st)
                ref.update({f"base_{name}/{k}": v
                            for k, v in per_split[name].items()})
            ref.update(self._marginal(per_split, prefix="base_"))
        finally:
            (self.actor._tparams, self.actor._opt_state,
             self.actor._opt_state_expert) = snap
        self._base_ref = ref
        try:
            cache.write_text(json.dumps(
                {"key": key, "ref": {k: float(v) for k, v in ref.items()}}, indent=1))
        except Exception as exc:
            print(f"[safety] base reference: could not cache ({exc})", flush=True)
        print("[safety] base reference (fixed seeds): "
              + "  ".join(f"{k.split('/')[0].replace('base_', '')}"
                          f"={ref[k]:.3f}"
                          for k in ref if k.endswith("solved_rate")), flush=True)
        return ref

    # ------------------------------------------------------------------ #
    def _eval_arm(self, state):
        """One eval pass on ``state``; returns (metrics, sim_path, per-scene stash).

        Fixed seed, so the fine-tuned and base arms of a dump are scored on the
        same sampler noise and their difference is the policy's.
        """
        m = {k: v for k, v in
             self.evaluate(state, rng=jax.random.fold_in(self._eval_key, 0)).items()
             if not k.startswith("_")}
        sim = np.asarray(self.env.ego_sim_world_trajectory())    # [B, T, 5]
        return m, sim, {k: np.asarray(v) for k, v in self._last_scene.items()}

    def dump_eval(self, out_dir, state, load_ckpt, tag=""):
        """Score the fine-tuned AND the frozen-base policy on the SAME batch
        ``state`` and write a self-contained npz + summary the CPU viz job reads.

        ``tag`` names the artifacts (e.g. "" for held-out, "train" for a
        train-pool batch) so several batches land in one dump dir. ``load_ckpt``
        is reloaded before the fine-tuned arm because scoring the base arm calls
        ``reset_trainable_to_frozen``, which overwrites the trainable weights --
        so each call must restore them itself rather than assume caller order.

        The npz is written in the schema ``tests/viz_drive.fig_paths`` consumes
        (``path_<arm>``/``off_<arm>``/``done_<arm>``, ``log_path``, roadgraph,
        ``goal_xy``), so "how it drives" re-renders with the existing, tested
        path renderer and no recomputation. Called eval-only -- no gradient.
        """
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        # Fine-tuned first (reload the trained weights), then reset to the
        # pretrained base and re-score the identical batch. Order cannot leak:
        # `reset_trainable_to_frozen` restores the base from the frozen module.
        arms = {}
        self.load(load_ckpt)
        arms["ft"] = self._eval_arm(state)
        B = self.env.batch_size
        self.actor.reset_trainable_to_frozen()
        arms["base"] = self._eval_arm(state)

        st = self.env.sim_state
        rg = st.roadgraph_points
        T = int(arms["ft"][1].shape[1])
        log_path = np.asarray(self.env.ego_log_world_trajectory())   # [B, T, 5]
        npz = {
            "indices": np.arange(B, dtype=np.int32),
            "start_step": np.int32(self.env._start_timestep),
            "goal_xy": np.asarray(self.env._goal_xy_b2, np.float32),
            "log_path": log_path.astype(np.float32),
            "rg_x": np.asarray(rg.x, np.float32), "rg_y": np.asarray(rg.y, np.float32),
            "rg_types": np.asarray(rg.types, np.int16), "rg_valid": np.asarray(rg.valid),
        }
        rows, summary = [], {}
        for name, (m, sim, sc) in arms.items():
            npz[f"path_{name}"] = sim.astype(np.float32)
            # Scatter per-step offroad marks (over executed timesteps) into a
            # dense [B, T] boolean the renderer indexes by absolute timestep.
            off_full = np.zeros((B, T), dtype=bool)
            idx = sc["exec_step_idx"].astype(int)
            keep = (idx >= 0) & (idx < T)          # guard: never index past the path
            if keep.any():
                off_full[:, idx[keep]] = (sc["off_step_marks"] > 0.5)[:, keep]
            npz[f"off_{name}"] = off_full
            npz[f"done_{name}"] = sc["done_at"].astype(np.int32)
            summary[name] = {k: float(v) for k, v in m.items()}
            for b in range(B):
                rows.append({"scene": int(b), "seed": 778, "arm": name,
                             "solved": bool(sc["solved"][b] > 0.5),
                             "reached": bool(sc["reached"][b] > 0.5),
                             "offroad": bool(sc["offroad"][b] > 0.5),
                             "collided": bool(sc["overlap"][b] > 0.5),
                             "clean_expert": bool(sc["clean"][b] > 0.5)})
        np.savez_compressed(out / f"reach_rollouts{suffix}.npz", **npz)
        (out / f"reach_rows{suffix}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")
        (out / f"reach_summary{suffix}.json").write_text(json.dumps(summary, indent=2))
        label = tag or "heldout"
        for arm in ("base", "ft"):
            print(f"[eval:{label}] {arm:4} ({B} scenes): " + "  ".join(
                f"{k} {summary[arm][k]:.3f}" for k in
                ("solved_rate", "reached_rate", "offroad_rate", "overlap_rate")), flush=True)
        print(f"[eval:{label}] wrote {out/f'reach_rollouts{suffix}.npz'}", flush=True)
        return summary

    # ------------------------------------------------------------------ #
    def _init_wandb(self, out_dir):
        """Start a W&B run mirroring the JSONL rows, or return None if disabled.

        Unlike :mod:`rl.diffusion_ft.train_dppo`, which opens a fresh timestamped
        directory per launch, this trainer **resumes into a fixed ``out_dir``** --
        that is what makes a 12 h wall-clock limit a resubmit rather than a
        restart. So the run id is persisted beside the checkpoint and reused with
        ``resume="allow"``: a resubmitted job continues the same W&B run instead
        of starting a second one, which would split one training curve across two
        charts at whatever iteration the time limit happened to fall.
        """
        a = self.args
        if not a.wandb:
            return None
        import wandb

        id_path = Path(out_dir) / "wandb_run_id.txt"
        run_id = id_path.read_text().strip() if id_path.exists() else wandb.util.generate_id()
        name = a.wandb_name or (
            f"u{self.unroll_steps}_{a.anchor_states}_k{a.k_trainable}_aw{a.anchor_weight}")
        run = wandb.init(
            project=a.wandb_project, entity=a.wandb_entity,
            name=name, group=a.wandb_group, id=run_id, resume="allow",
            config={**vars(a), "out_dir": str(out_dir),
                    "data_regime": "corpus" if self.sampler is not None else "fixed",
                    "proposal": ("free_running" if not self.unroll_steps
                                 else f"unroll_K{self.unroll_steps}"),
                    "unroll_t_start": getattr(self, "t_start_max", None),
                    "unroll_t_end": getattr(self, "t_end", None)},
        )
        id_path.write_text(run_id)
        # One shared step axis = the training iteration. Eval keys land every
        # `eval_every` iterations and plot with gaps, which is the honest picture.
        wandb.define_metric("iter")
        wandb.define_metric("*", step_metric="iter")
        return run

    # ------------------------------------------------------------------ #
    def train(self):
        a = self.args
        out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
        log_path = out / "train_log.jsonl"
        if a.resume and (out / "latest").exists():
            self.load(str(out / "latest"))
            if log_path.exists():
                with log_path.open() as f:
                    rows = [json.loads(l) for l in f if l.strip()]
                # `_resume_iter` is what seeded the sampler in __init__; deriving
                # the loop start any other way would let the batch stream and the
                # iteration counter disagree after a restart.
                self.start_iter = self._resume_iter()
                # Distance from the base is set by the STEP count, not the
                # iteration count, so the probe's x-axis has to survive a resume.
                self._adam_steps = sum(int(r.get("n_steps", 0)) for r in rows)
            print(f"[safety] resumed at iter {self.start_iter} "
                  f"({self._adam_steps} Adam steps)", flush=True)
        wandb_run = self._init_wandb(out)
        if self.probe_set is not None:
            self.probe_static(out)

        for it in range(self.start_iter, a.iters):
            t_start = time.time()
            state = self._next_batch()
            self._tb_iter = it
            contexts, roll_m = self.rollout(
                state, train=True, store_transitions=self.tb is not None,
                scene_ids=(self.sampler.last_ids if self.sampler is not None else None))
            t_collect = time.time() - t_start
            # Row outcomes of THIS policy's rollout, captured before any later
            # rollout (bc, preservation) overwrites `_last_scene`.
            roll_scene = dict(self._last_scene)
            # Same hazard, same fix: `_neighbor_state` reads `self.env.sim_state`,
            # and the BC and preservation passes below each roll the env again --
            # the preservation pass on a DIFFERENT batch of scenes entirely. Taken
            # here, the neighbours are the ones THIS policy's contexts were
            # collected against. `state` unchanged under the default.
            _ns = self._neighbor_state(state)
            grad_state = state if _ns is None else _ns
            n_broken = self.note_broken(roll_scene["solved"], roll_scene["clean"])
            n_safety_rows = -1.0
            if getattr(a, "safety_rows", "all") == "unsolved":
                # Repair pressure only where this rollout FAILED. A solved row --
                # typically a tight success -- would otherwise still feel the
                # softplus barrier near its margin and be pushed off the line
                # that earned the success; there it is left to the anchors.
                keep = jnp.asarray(1.0 - roll_scene["solved"], jnp.float32)
                for c in contexts:
                    c.safety_alive = c.alive * keep
                n_safety_rows = float(np.sum((1.0 - roll_scene["solved"]) * roll_scene["clean"]))
            # The BC pass is a SECOND collection over the same batch, driven by
            # the log. It must run after the policy pass and be kept separate:
            # `roll_m` is the policy's own rollout metrics and would otherwise
            # report the human's.
            bc_contexts = []
            t_bc = 0.0
            if float(a.bc_weight) > 0:
                t0_bc = time.time()
                bc_contexts = self.collect_bc_contexts(state)
                t_bc = time.time() - t0_bc
            pres_ctx, pres_m, t_pres = [], {}, 0.0
            if self.preserve_weight > 0:
                t0_p = time.time()
                _, pres_ctx, pres_m = self.collect_preserve_contexts(it)
                t_pres = time.time() - t0_p

            order = np.random.default_rng(a.seed + it).permutation(len(contexts))
            # `grad_accum` contexts share one optimizer step (0 == the whole
            # iteration). See `DiffusionRLActor.safety_grads`: Adam's step size is
            # O(lr) per parameter regardless of the gradient norm, so the step
            # COUNT is what sets the distance travelled from the pretrained
            # weights. At the old one-step-per-context schedule a 200-iteration
            # run took 1600 steps and moved ~||lr*sqrt(N*P)|| -- the pure
            # random-walk distance -- while averaging each step over a single
            # replan window.
            accum = len(contexts) if a.grad_accum <= 0 else int(a.grad_accum)
            if pres_ctx or self.repair_weight > 0 or self.pref_weight > 0 or self.tb is not None:
                accum = 10 ** 9   # one step at the end, with the extra terms' grads added
            # A pure-replay arm (`--safety_weight 0 --tb_weight 1`) has no
            # on-policy term left to differentiate, and paying 100 denoiser evals
            # per context for a gradient that is multiplied by zero would make it
            # the most expensive arm in the sweep. The ROLLOUT still runs: it is
            # what fills the buffer.
            skip_policy_term = (float(a.safety_weight) <= 0
                                and float(a.anchor_weight) <= 0
                                and float(getattr(a, "bc_weight", 0.0)) <= 0
                                and float(getattr(a, "base_anchor_weight", 0.0)) <= 0)
            hist, step_hist = [], []
            buf, nbuf = None, 0
            for ep in range(a.epochs):
                for j in order:
                    ctx = contexts[int(j)]
                    if float(jnp.sum(ctx.alive)) <= 0.0:
                        continue
                    if skip_policy_term:
                        hist.append({k: 0.0 for k in
                                     ("loss", "safety", "anchor", "bc",
                                      "base_anchor", "edge", "agent",
                                      "tail_share", "n_neighbors", "t_start")})
                        continue
                    self.rng, rk = jax.random.split(self.rng)
                    # Paired by replan index so each step sees the human's
                    # states from the SAME window the safety term is scoring.
                    # Modular: the policy pass can end early (scenes absorb),
                    # so it may hold fewer contexts than the log pass.
                    bc_ctx = (bc_contexts[int(j) % len(bc_contexts)]
                              if bc_contexts else None)
                    grads, loss, aux = self._grads_for(grad_state, ctx, rk, bc_ctx)
                    hist.append({"loss": float(np.asarray(loss)),
                                 **{k: float(np.asarray(v)) for k, v in aux.items()}})
                    buf = (grads if buf is None else
                           jax.tree_util.tree_map(jnp.add, buf, grads))
                    nbuf += 1
                    if nbuf >= accum:
                        step_hist.append(self.actor.apply_safety_grads(
                            jax.tree_util.tree_map(lambda x: x / nbuf, buf)))
                        buf, nbuf = None, 0
            p_hist = []
            repair_loss = 0.0
            pref_hist: list[dict] = []
            tb_m: dict[str, float] = {}
            has_extra = (bool(pres_ctx) or self.repair_weight > 0 or self.pref_weight > 0
                         or self.tb is not None)
            if buf is not None or has_extra:      # partial group at the end
                if buf is None:
                    # Pure-replay arm: start the step from an explicit zero rather
                    # than skipping it, so the buffer term is the whole update.
                    g = jax.tree_util.tree_map(jnp.zeros_like, self.actor._tparams)
                else:
                    g = jax.tree_util.tree_map(lambda x: x / nbuf, buf)
                pbuf, pn = None, 0
                for c in pres_ctx:
                    if float(jnp.sum(c.preserve_w)) <= 0.0:
                        continue
                    pg, _, paux = self.preserve_grads(c)
                    p_hist.append({k: float(np.asarray(v)) for k, v in paux.items()})
                    pbuf = pg if pbuf is None else jax.tree_util.tree_map(jnp.add, pbuf, pg)
                    pn += 1
                if pn:
                    lam = self.preserve_weight
                    g = jax.tree_util.tree_map(lambda x, y: x + lam * y / pn, g, pbuf)
                if self.repair_weight > 0:
                    rg, repair_loss = self.repair_grads(it)
                    rw = self.repair_weight
                    g = jax.tree_util.tree_map(lambda x, y: x + rw * y, g, rg)
                if self.pref_weight > 0:
                    fg, fm = self.pref_grads(it)
                    pref_hist.append(fm)
                    fw = self.pref_weight
                    g = jax.tree_util.tree_map(lambda x, y: x + fw * y, g, fg)
                tbg = None
                if self.tb is not None:
                    tbg, tb_m = self.buffer_grads(it)
                    if tbg is not None:
                        tw = self.tb_weight
                        g = jax.tree_util.tree_map(lambda x, y: x + tw * y, g, tbg)
                # Skip the step when NOTHING contributed to it -- a pure-replay arm
                # whose buffer has not reached --tb_min_size yet would otherwise
                # take an Adam step on an all-zero gradient, which is not a no-op
                # once the moments are warm and would miscount `adam_steps`, the
                # x-axis that sets how far the weights have travelled.
                if (buf is not None or pn or self.repair_weight > 0 or self.pref_weight > 0
                        or tbg is not None):
                    step_hist.append(self.actor.apply_safety_grads(g))
                # `--repair_steps K`: K-1 further Adam steps on the repair term
                # alone, each on a fresh batch. Distilling a dataset of 10^4 rows at
                # one batch per iteration would need an iteration -- and a rollout --
                # per step.
                if self.repair_weight > 0 and int(getattr(a, "repair_steps", 1)) > 1:
                    r_losses = [repair_loss]
                    for sub in range(1, int(a.repair_steps)):
                        rg, rl = self.repair_grads(it, sub)
                        step_hist.append(self.actor.apply_safety_grads(
                            jax.tree_util.tree_map(lambda y: rw * y, rg)))
                        r_losses.append(rl)
                    repair_loss = float(np.mean(r_losses))
                # `--pref_steps K`: same, for the preference term (+ its anchor).
                if self.pref_weight > 0 and int(getattr(a, "pref_steps", 1)) > 1:
                    for sub in range(1, int(a.pref_steps)):
                        fg, fm = self.pref_grads(it, sub)
                        pref_hist.append(fm)
                        step_hist.append(self.actor.apply_safety_grads(
                            jax.tree_util.tree_map(lambda y: fw * y, fg)))
            if not hist:
                print(f"[safety] iter {it}: no live contexts, skipping", flush=True)
                continue

            self._adam_steps += len(step_hist)
            # The anchor's actual size next to the safety term, per iteration.
            # Without it the only way to know what --anchor_weight bought was a
            # separate gate run, and every such number taken before 2026-09-03
            # was taken on the frozen-rng bug.
            a_term = float(a.anchor_weight) * float(hist[-1].get("anchor", 0.0))
            s_term = float(a.safety_weight) * float(hist[-1]["safety"])
            # Same reason `anchor_share` exists: a weight is meaningless without
            # the size of the term it multiplies. Read `bc_share` at iter 0
            # before spending a run on a --bc_weight guess.
            b_term = float(a.bc_weight) * float(hist[-1].get("bc", 0.0))
            # The frozen-base anchor's size next to the safety term, same reason
            # anchor_share/bc_share exist: a weight (beta) is meaningless without
            # the term it multiplies, and the KL collapses fast so iter 0 lies.
            ba_term = float(getattr(a, "base_anchor_weight", 0.0)) * float(
                hist[-1].get("base_anchor", 0.0))
            pr_term = (self.preserve_weight * float(np.mean([h["preserve"] for h in p_hist]))
                       if p_hist else 0.0)
            s_mean = float(a.safety_weight) * float(np.mean([h["safety"] for h in hist]))
            row = {"iter": it, "collect_s": round(t_collect, 2),
                   # Matched-sample preservation. `preserve_rms_m` is the number
                   # to read: metres the fine-tune's executed segment has moved off
                   # the base's on base-SOLVED rows. -1 rows when the term is off.
                   "safety_rows": n_safety_rows,
                   # Denoising error of the trainable copy on verified repairs
                   # (normalized units, t < k_trainable). 0 when the term is off.
                   "repair_loss": repair_loss,
                   "repair_share": (self.repair_weight * repair_loss
                                    / (self.repair_weight * repair_loss + s_mean)
                                    if self.repair_weight * repair_loss + s_mean > 0 else 0.0),
                   # Diffusion-DPO (`--pref_weight`): means over this iteration's
                   # steps. `pref/logit_z` > 0 and `pref/accuracy` -> 1 mean the
                   # model now denoises the edit better than the rejected draw,
                   # RELATIVE to the base. Compare `pref/grad_norm` with
                   # `pref/anchor_grad_norm` x weight for the balance of the two
                   # terms -- the anchor's VALUE is ~0 by construction.
                   **({k: float(np.mean([h[k] for h in pref_hist])) for k in pref_hist[0]}
                      if pref_hist else {}),
                   "replay_added": n_broken, "replay_size": len(self._replay),
                   # Transition replay. `tb_age` is the number to watch: it is how
                   # many iterations old, on average, the sampler states the
                   # gradient was taken at have become.
                   **tb_m, **(self.tb.stats() if self.tb is not None else {}),
                   "tb_share": (self.tb_weight * tb_m.get("tb_safety", 0.0)
                                / (self.tb_weight * tb_m.get("tb_safety", 0.0) + s_mean)
                                if self.tb_weight * tb_m.get("tb_safety", 0.0) + s_mean > 0
                                else 0.0),
                   "preserve_s": round(t_pres, 2), "preserve_ctx": len(p_hist),
                   "preserve_last": p_hist[-1]["preserve"] if p_hist else 0.0,
                   "preserve_rms_m": (float(np.mean([h["preserve_rms_m"] for h in p_hist]))
                                      if p_hist else 0.0),
                   "preserve_rows": (float(np.mean([h["preserve_rows"] for h in p_hist]))
                                     if p_hist else 0.0),
                   "preserve_share": (pr_term / (pr_term + s_mean)
                                      if pr_term + s_mean > 0 else 0.0),
                   **pres_m,
                   # How many of the batch's scenes the anchor was allowed to act
                   # on. -1 = unrestricted. If this reads 0 while
                   # base_anchor_weight > 0 the anchor is silently doing nothing.
                   "base_anchor_n": (-1 if self._anchor_mask is None
                                     else float(jnp.sum(self._anchor_mask))),
                   "base_anchor_last": hist[-1].get("base_anchor", 0.0),
                   "base_anchor_share": (ba_term / (ba_term + s_term)
                                         if ba_term + s_term > 0 else 0.0),
                   "bc_collect_s": round(t_bc, 2),
                   "bc_last": hist[-1].get("bc", 0.0),
                   "bc_share": (b_term / (b_term + s_term + a_term)
                                if b_term + s_term + a_term > 0 else 0.0),
                   "n_bc_ctx": len(bc_contexts),
                   "iter_s": round(time.time() - t_start, 2),
                   "n_ctx": len(contexts), "n_steps": len(step_hist),
                   "adam_steps": self._adam_steps,
                   "n_grads": len(hist),
                   "safety_first": hist[0]["safety"], "safety_last": hist[-1]["safety"],
                   "edge_last": hist[-1]["edge"], "agent_last": hist[-1]["agent"],
                   # Mean over the iteration's grad steps, not the last one: the
                   # tail share moves with which scenes were in the minibatch
                   # (it scales with their traffic density), which is exactly
                   # the quantity in question, so a single step is noisy.
                   "tail_share": float(np.mean(
                       [float(h["tail_share"]) for h in hist])),
                   "anchor_last": hist[-1].get("anchor", 0.0),
                   "anchor_share": (a_term / (a_term + s_term)
                                    if a_term + s_term > 0 else 0.0),
                   # Valid ego-neighbour pairs per scored step. Watch it on any
                   # `--neighbor_source sim` run: 0 there means the agent barrier
                   # was charged against nothing, which the loss curve alone
                   # cannot distinguish from an empty road.
                   "n_neighbors": float(np.mean(
                       [float(h.get("n_neighbors", 0.0)) for h in hist])),
                   "t_start_mean": hist[-1]["t_start"],
                   "grad_norm": step_hist[-1]["grad_norm"] if step_hist else 0.0,
                   **{f"roll_{k}": v for k, v in roll_m.items()}}
            if a.eval_every and (it % a.eval_every == 0 or it == a.iters - 1):
                # The random held-out batch keeps its flat `eval_*` keys; each
                # named split reports the same metric family under its own
                # `<split>/` prefix, so the sweep's arms are comparable per split.
                row.update(self.base_reference())
                row.update({f"eval_{k}": v for k, v in self.evaluate_fixed().items()})
                per_split = {}
                for name, st in self.eval_splits.items():
                    per_split[name] = self.evaluate_fixed(st)
                    row.update({f"{name}/{k}": v
                                for k, v in per_split[name].items()})
                # The corpus-level rate, and the two numbers that make it
                # readable: how far it has moved from the frozen base, and what
                # the success split would have to hold for this arm's failure
                # gain to break even. Without these an arm that improves both
                # splits it is scored on can still be a net regression.
                marg = self._marginal(per_split)
                row.update(marg)
                if marg and "base_marginal/solved_rate" in row:
                    row["marginal/delta_vs_base"] = (
                        marg["marginal/solved_rate"] - row["base_marginal/solved_rate"])
                    wf = self._marg_w.get("failure_heldout", 0.0)
                    d_fail = (per_split["failure_heldout"]["solved_rate"]
                              - row["base_failure_heldout/solved_rate"])
                    row["marginal/breakeven_success_rate"] = (
                        row["base_success_heldout/solved_rate"]
                        - wf * d_fail / max(1.0 - wf, 1e-9))
                if not self.actor.frozen_params_unchanged(self.frozen_ref):
                    raise SystemExit("[safety] frozen modules changed -- aborting")
            # Rollout pictures for the same fixed scenes at every probe point.
            # Written as npz + jsonl so the slider re-renders on a CPU node
            # without touching the model again.
            if (self.probe_set is not None and self.probe_every
                    and (it % self.probe_every == 0 or it == a.iters - 1)):
                self.probe_dump(out, it)
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if wandb_run is not None:
                import wandb
                wandb.log(row, step=it)
            print(f"[safety] iter {it:>4} safety {row['safety_first']:.4f}->"
                  f"{row['safety_last']:.4f} |g| {row['grad_norm']:.2e} "
                  # Share of the safety loss charged to geometry ALREADY CLEAR
                  # of its margin -- the softplus barrier's infinite tail,
                  # summed over 4 corners x N agents x T steps, so it grows with
                  # the scene's traffic rather than with anything the policy
                  # did. Exactly 0 under a compact-support --barrier_shape. On
                  # the console because it is the number that says whether the
                  # shape can matter on this data at all.
                  f"tail {row['tail_share']:.3f} "
                  + (f"PREF loss {row['pref/loss']:.4f} z {row['pref/logit_z']:+.2e} "
                     f"acc {row['pref/accuracy']:.2f} |g| {row['pref/grad_norm']:.1e}"
                     f"/{row['pref/anchor_grad_norm']:.1e} "
                     if "pref/loss" in row else "")
                  + f"ctx {row['n_ctx']} ({row['iter_s']}s)"
                  + (f"  EVAL solved {row.get('eval_solved_rate', float('nan')):.3f}"
                     f" reached {row.get('eval_reached_rate', float('nan')):.3f}"
                     f" offroad {row.get('eval_offroad_rate', float('nan')):.3f}"
                     f" overlap {row.get('eval_overlap_rate', float('nan')):.3f}"
                     f" drift {row.get('eval_drift_m', float('nan')):.2f}m"
                     if "eval_offroad_rate" in row else "")
                  # The corpus-level number, on the console line and not only in
                  # train_log.jsonl. `eval_solved` above is the random 32-scene
                  # held-out batch; it is NOT the quantity that decides whether
                  # this arm is worth deploying, and reading a tailed .out
                  # without this means reading the wrong number. `d` negative =
                  # a net regression against the frozen base however good the
                  # per-split curves look; `be` is the success rate this arm's
                  # failure gain would need to break even.
                  + (f"  MARG {row['marginal/solved_rate']:.4f}"
                     f" (base {row.get('base_marginal/solved_rate', float('nan')):.4f}"
                     f" d {row.get('marginal/delta_vs_base', float('nan')):+.4f})"
                     f" be {row.get('marginal/breakeven_success_rate', float('nan')):.4f}"
                     if "marginal/solved_rate" in row else "")
                  + (f"  MARGINAL {row['marginal/solved_rate']:.4f} "
                     f"({row['marginal/delta_vs_base']:+.4f} vs base)"
                     if "marginal/delta_vs_base" in row else ""), flush=True)

            if a.save_every and (it % a.save_every == 0 or it == a.iters - 1):
                self.save(out / "latest")
            # `latest` is overwritten, so on its own it cannot answer "the
            # success split was best 2000 iterations ago, give me that model".
            # Over a multi-day run that question is the whole point: more Adam
            # steps means more displacement, and displacement rank-orders damage.
            if a.keep_every and it % a.keep_every == 0:
                self.save(out / f"iter_{it:06d}", with_opt=False)
        if wandb_run is not None:
            import wandb
            wandb.finish()
        print("[safety] done", flush=True)

    # ------------------------------------------------------------------ #
    def save(self, path, *, with_opt: bool = True):
        """Write a checkpoint. ``with_opt`` also persists the optimizer state.

        `latest` needs it (a resume without it is a new run -- see below). A
        kept `iter_%06d` snapshot does not: it exists to be SCORED later, and
        Adam's two moment trees are 4x the weights, so carrying them would turn
        a 121 MB snapshot into ~600 MB and a long run into hundreds of GB.
        """
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        flat = jax.tree_util.tree_flatten(self.actor._tparams)[0]
        # Zero-padded keys: np.savez's positional names sort lexically, so
        # `arr_10` would load before `arr_2` and silently permute the pytree.
        np.savez(path / "tparams.npz",
                 **{f"p{i:05d}": np.asarray(x) for i, x in enumerate(flat)})
        (path / "meta.json").write_text(json.dumps(
            {"base_ckpt": self.args.ckpt, "k_trainable": self.args.k_trainable,
             "anchor_weight": self.args.anchor_weight,
             "edge_margin_m": self.args.edge_margin,
             "edge_geometry": getattr(self.args, "edge_geometry", "point"),
             "agent_margin_m": self.args.agent_margin,
             "unroll_steps": self.unroll_steps,
             "unroll_grad_steps": (self.unroll_grad_steps if self.unroll_steps else None),
             "unroll_t_start": (self.t_start_max if self.unroll_steps else None),
             "unroll_t_end": (self.t_end if self.unroll_steps else None),
             "unroll_eta": (self.args.unroll_eta if self.unroll_steps else None),
             "anchor_states": self.args.anchor_states,
             "grad_accum": self.args.grad_accum,
             "lr": self.args.lr,
             "eval_seed": self.args.eval_seed,
             "eval_repeats": self.args.eval_repeats,
             "safety_rows": getattr(self.args, "safety_rows", "all"),
             "preserve_weight": float(getattr(self.args, "preserve_weight", 0.0)),
             "repair_weight": float(getattr(self.args, "repair_weight", 0.0)),
             "repair_data": str(getattr(self.args, "repair_data", "")),
             "repair_steps": int(getattr(self.args, "repair_steps", 1)),
             "repair_rows": str(getattr(self.args, "repair_rows", "all")),
             "repair_split_ratio": str(getattr(self.args, "repair_split_ratio", "")),
             "repair_switch_share": float(getattr(self.args, "repair_switch_share", 0.0)),
             **{k: getattr(self.args, k, None) for k in (
                 "pref_weight", "pref_data", "pref_beta", "pref_batch",
                 "pref_anchor_batch", "pref_anchor_weight", "pref_steps", "pref_rows",
                 "pref_t_min", "pref_t_max", "pref_edit_weight", "pref_edit_delta_m")},
             "tb_weight": float(getattr(self.args, "tb_weight", 0.0)),
             "tb_capacity": int(getattr(self.args, "tb_capacity", 0)),
             "tb_batch": int(getattr(self.args, "tb_batch", 0)),
             "tb_unroll_steps": int(getattr(self.args, "tb_unroll_steps", 1)),
             "tb_grad_steps": int(getattr(self.args, "tb_grad_steps", 1)),
             "tb_max_uses": int(getattr(self.args, "tb_max_uses", 0)),
             "tb_store_per_replan": int(getattr(self.args, "tb_store_per_replan", 1))},
            indent=2))

        # --- the rest of the training state -----------------------------------
        # A checkpoint of weights ALONE makes a resume a NEW run that happens to
        # start from these weights: Adam's moments and step counter reset, and
        # bias correction then makes the first steps after every restart
        # full-size whatever the gradient is. Measured on this model: the null
        # objective moves drift 0.412 -> 0.717 m in ONE iteration and only ->
        # 0.810 over the next 80. Over a multi-day run chained across the
        # partition's time limit that is one uncontrolled kick per restart, in a
        # project where displacement rank-orders damage (Spearman -1.0).
        #
        # Written as a SEPARATE npz: `tparams.npz` is read by eval_reach,
        # probe_param_distance and `load` via `sorted(z.files)`, so adding keys
        # to it would permute the pytree in every one of them.
        if not with_opt:
            return
        sd = self.actor.state_dict()
        flat = {}
        for tag, tree in (("a", sd["opt_state"]), ("e", sd["opt_state_expert"])):
            for i, x in enumerate(jax.tree_util.tree_flatten(tree)[0]):
                flat[f"{tag}{i:05d}"] = np.asarray(x)
        np.savez(path / "opt_state.npz", **flat)
        (path / "train_state.json").write_text(json.dumps({
            "adam_steps": int(self._adam_steps),
            "rng": [int(x) for x in np.asarray(self.rng).reshape(-1)],
            "actor_rng": [int(x) for x in np.asarray(sd["rng"]).reshape(-1)],
            # Insertion order is the eviction order, so keep it as a list.
            "preserve_replay": [[int(k), int(v)]
                                for k, v in getattr(self, "_replay", {}).items()],
        }))

    def load(self, path):
        path = Path(path)
        z = np.load(path / "tparams.npz")
        leaves, treedef = jax.tree_util.tree_flatten(self.actor._tparams)
        vals = [jnp.asarray(z[k]) for k in sorted(z.files)]
        if len(vals) != len(leaves):
            raise SystemExit(f"checkpoint has {len(vals)} leaves, model wants {len(leaves)}")
        self.actor._tparams = jax.tree_util.tree_unflatten(treedef, vals)

        # Optimizer state. The freshly initialized trees supply the structure --
        # they are `tx.init(tparams)` on the weights just loaded, so the treedef
        # matches by construction and only the leaf VALUES come from disk.
        opt_path = path / "opt_state.npz"
        if not opt_path.exists():
            print("[safety] WARNING: no opt_state.npz in this checkpoint -- Adam "
                  "moments and step count restart from zero. Bias correction "
                  "makes the next few steps full-size whatever the gradient is; "
                  "treat this resume as a new run, not a continuation.",
                  flush=True)
        else:
            zo = np.load(opt_path)
            for tag, attr in (("a", "_opt_state"), ("e", "_opt_state_expert")):
                cur = getattr(self.actor, attr)
                cur_leaves, cur_treedef = jax.tree_util.tree_flatten(cur)
                keys = sorted(k for k in zo.files if k.startswith(tag))
                if len(keys) != len(cur_leaves):
                    raise SystemExit(
                        f"opt_state.npz has {len(keys)} '{tag}' leaves, "
                        f"{attr} wants {len(cur_leaves)} -- refusing to resume "
                        f"onto a different optimizer")
                setattr(self.actor, attr, jax.tree_util.tree_unflatten(
                    cur_treedef,
                    [jnp.asarray(zo[k]).astype(leaf.dtype).reshape(leaf.shape)
                     for k, leaf in zip(keys, cur_leaves)]))
            print(f"[safety] restored optimizer state from {opt_path}", flush=True)

        ts_path = path / "train_state.json"
        if ts_path.exists():
            ts = json.loads(ts_path.read_text())
            self._adam_steps = int(ts.get("adam_steps", self._adam_steps))
            if ts.get("rng"):
                self.rng = jnp.asarray(ts["rng"], dtype=jnp.uint32)
            if ts.get("actor_rng"):
                self.actor._rng = jnp.asarray(ts["actor_rng"], dtype=jnp.uint32)
            if ts.get("preserve_replay") is not None:
                self._replay = {int(k): int(v) for k, v in ts["preserve_replay"]}


# --------------------------------------------------------------------------- #
def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tfrecord", default="")
    p.add_argument("--indices", default="0,1,2,3,4,5,6,7")
    p.add_argument("--shard_glob", default="")
    p.add_argument("--index_dir", default="/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets")
    p.add_argument("--scenes_per_iter", type=int, default=16)
    p.add_argument("--eval_scenes", type=int, default=32)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--prefetch", type=int, default=1)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_replans", type=int, default=8)
    p.add_argument("--replan_interval_steps", type=int, default=10)
    # ---- how the OTHER agents move ----------------------------------------- #
    # "none" log-replays them (every result before 2026-09-19): each non-ego
    # object repeats its recorded pose and cannot see the ego at all. "idm"
    # hands them to waymax's IDMRoutePolicy -- each stays on its OWN logged path
    # and modulates only its speed against whatever box is ahead of it, the ego
    # included. Longitudinal reactivity: a car brakes for the ego, it never
    # steers around it.
    #
    # This changes what `overlap` MEANS, so it changes every collision number a
    # run reports. Measured motivation: log-replay rails charge the ego 0.031 ->
    # 0.719 collision purely for running LATE (probe_logreplay_collision, the
    # delay sweep), which is the failure mode a slow policy actually has, while
    # 1 m of lateral offset costs one scene in 32.
    p.add_argument("--sim_agents", type=str, default="none",
                   choices=SIM_AGENT_CHOICES)
    # Where the SAFETY LOSS reads the other agents' boxes. "log" is the recorded
    # motion; "sim" is where the NPCs actually were when the collected episode
    # was driven (`sim_trajectory` of the state the rollout ended in). Under
    # `--sim_agents none` the two are the same motion and this is a no-op. Under
    # `--sim_agents idm` they are not, and reading "log" charges the candidate
    # for a hazard that braked its way out of existence.
    p.add_argument("--neighbor_source", type=str, default="log",
                   choices=("log", "sim"))
    # Which velocity the policy is SHOWN for its own executed steps. "commanded"
    # (every result before 2026-09-23) is what it wrote, which overstates what
    # its positions drive by 13-23%; "realised" is the executed positions'
    # backward difference, so `ego_state` stops reporting a speed the car is not
    # doing. See env._splice_sim_into_log_batched.
    p.add_argument("--ego_vel_source", type=str, default="commanded",
                   choices=EGO_VEL_SOURCES)
    # How each scene's world step is read off its log timestamps -- the grid the
    # plan is resampled onto, one point per executed log step. "first" (every
    # result before 2026-09-24) takes the FIRST pair, ~0.075 s in 44/173
    # failure_heldout scenes, which executes their plan at ~0.75x; 43 of the 44
    # never reach. "median" is the scene's median step, the default since the
    # A/B (jobs 62629-62634): those 44 go 0.01 -> 0.95 solved, nothing else moves.
    # RESUMING a run started before 2026-09-24? pass --world_dt_method first.
    # See data.preprocess.compute_world_dt_seconds.
    p.add_argument("--world_dt_method", type=str, default="median",
                   choices=WORLD_DT_METHODS)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--k_trainable", type=int, default=10)
    p.add_argument("--prefix_len", type=int, default=5)
    # The collection rollout is free-running; exploration noise is not needed and
    # 0.03 normalized is 3 m of positional jitter (ego_range=100). Keep it tiny.
    p.add_argument("--sigma_floor", type=float, default=1e-4)
    # ---- DDIM striding of the FROZEN prefix (the loss-speed knob) ------------ #
    # Free-running costs `timesteps` (100) denoiser evals per proposal, and the
    # iteration pays that TWICE per replan: once in the collection rollout and
    # again inside the gradient pass. Only the last `k_trainable` steps carry a
    # gradient; the other 90 are frozen forward compute. `num_sample_steps`
    # replaces that prefix with `S - k_trainable` strided DDIM steps over the same
    # span, leaving the trainable late chain untouched -- so S=20 at k=10 is 20
    # evals instead of 100 on BOTH passes.
    #
    # This is an APPROXIMATION, not a refactor: a strided prefix delivers a
    # slightly different x_{k-1}, so the policy is not the same policy. That is
    # why it is off by default and why eval has its own knob below. Gate it with
    # slurm/submit_bench_safety_ddim.sh before adopting a value.
    p.add_argument("--reuse_prefix", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="free-running only: keep each collected sample's frozen "
                        "prefix output and differentiate the executed trajectory "
                        "by replaying just the k_trainable late steps, instead "
                        "of re-running the whole chain on a fresh noise draw.")
    p.add_argument("--collect_steps", type=int, default=0,
                   help="0 = full reverse chain. S in [k_trainable, timesteps] "
                        "DDIM-subsamples the frozen prefix for the TRAINING "
                        "passes (collection rollout + gradient). S=20 at "
                        "k_trainable=10 is ~5x fewer denoiser evals.")
    p.add_argument("--eval_steps", type=int, default=0,
                   help="Same knob for eval/probe/verify rollouts. Default 0 "
                        "(full chain) deliberately: eval must measure the policy "
                        "you would deploy, and striding it too would make a "
                        "curve movement unattributable between the weights and "
                        "the sampler. Set it only to reproduce an old run.")
    # ---- teacher-forced unroll (rl.diffusion_ft.unrolled_prediction) --------- #
    # 0 = free-running: denoise from pure noise through the whole chain (the
    # original path). K >= 1 = corrupt the expert future to a random low-noise
    # timestep and unroll K reverse steps back down, scoring that endpoint --
    # K nn evals per gradient step instead of `timesteps`. K=1 degenerates to the
    # one-step posterior mean, which is the Jensen-gap control.
    p.add_argument("--unroll_steps", type=int, default=0)
    # Final steps that keep a gradient (truncated backprop). 1 is enough for the
    # endpoint gradient to reach the weights; each extra one holds another
    # forward's activations. Still cheaper than the free-running path, which
    # backprops through all `k_trainable` steps.
    p.add_argument("--unroll_grad_steps", type=int, default=1)
    # Noisiest timestep the unroll may start from. -1 = k_trainable - 1, the
    # noisiest step the trainable copy is actually used at when sampling; an
    # explicit value above that is rejected rather than silently training steps
    # the deployed sampler drives with the frozen denoiser.
    p.add_argument("--unroll_t_start", type=int, default=-1)
    # The low-noise timestep the unroll lands on. At 0 the scored endpoint is
    # exactly what the sampler emits (the t=0 DDIM update reduces to x0_pred).
    p.add_argument("--unroll_t_end", type=int, default=0)
    # DDIM noise scale inside the unroll. 0 = deterministic ODE unroll, matching
    # the mechanic this is ported from; 1.0 = the ancestral chain the deployed
    # sampler runs, at higher gradient variance.
    p.add_argument("--unroll_eta", type=float, default=0.0)
    # Where the replan contexts come from. `policy` rolls the policy out
    # closed-loop (its own states, exposure bias corrected); `log` drives the
    # rollout with the human's log window, which removes every 100-step sample
    # from the iteration but only ever supervises states a human visited.
    p.add_argument("--anchor_states", default="policy", choices=("policy", "log"))
    p.add_argument("--safety_weight", type=float, default=1.0)
    p.add_argument("--bc_weight", type=float, default=0.0,
                   help="weight on the ORIGINAL pretraining denoising loss, "
                        "computed on the HUMAN's states from a second, "
                        "sampling-free pass over the same batch (see "
                        "`collect_bc_contexts`). This is not --anchor_weight: "
                        "the anchor conditions on policy-visited states and "
                        "relabels the human future from a drifted pose, which "
                        "is DAgger, not the pretraining objective. 0 = off. "
                        "Size it from the MEDIAN `bc_share`, not the iter-0 "
                        "value: the term is minimized within a few iterations, "
                        "so iter 0 overstates it ~13x (0.95 vs 0.57 measured). "
                        "Weight 100 -> 0.30 of the objective, 300 -> 0.57.")
    p.add_argument("--anchor_weight", type=float, default=0.0)
    p.add_argument("--base_anchor_success_only",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="restrict --base_anchor_weight to the success group of a "
                        "--train_splits/--train_split_ratio batch. ON by default: "
                        "the anchor is a do-no-harm leash, and on a failure scene "
                        "it directly opposes the safety term trying to fix that "
                        "scene (the 2026-09-08 beta30/beta100 arms, which ran with "
                        "it OFF, reached failure ~0.17 against the unanchored "
                        "arm's ~0.33). No-op without groups.")
    p.add_argument("--base_anchor_weight", type=float, default=0.0,
                   help="weight (beta) on the FROZEN-BASE-POLICY anchor: the "
                        "Gaussian KL between the trainable and the pretrained "
                        "denoiser's one-step posterior means, on the policy's own "
                        "visited states (see `DiffusionRLActor.base_anchor_loss`). "
                        "Unlike --anchor_weight/--bc_weight, whose target is the "
                        "human log, its minimizer IS the base policy, so it is "
                        "zero where the policy still matches base and pulls only "
                        "against drift -- the regression guard on the successes. "
                        "0 = off. Size it from the MEDIAN `base_anchor_share`, not "
                        "iter 0 (the KL collapses within a few steps, like bc).")
    p.add_argument("--edge_margin", type=float, default=0.3)
    p.add_argument("--agent_margin", type=float, default=0.75)
    # tau is the WIDTH OF THE KNEE of `tau*softplus((margin-c)/tau)`, and it is a
    # separate knob from the margin: the margin says where the cost turns on, tau
    # says how sharply. It also sets the loss AT the margin (tau*ln2) and the
    # distance over which gradient exists past it (~2.7*tau), so a small tau is a
    # near-hinge that only pushes on frames already inside the margin, while a
    # large one is a soft field that pulls on clear frames too. Calibrated
    # defaults, not arbitrary -- see tests/calibrate_expert.py.
    p.add_argument("--edge_tau", type=float, default=0.15,
                   help="softplus knee width (m) on the road-edge barrier")
    p.add_argument("--agent_tau", type=float, default=0.25,
                   help="softplus knee width (m) on the agent-gap barrier")
    p.add_argument("--barrier_shape", type=str, default="softplus",
                   choices=("softplus", "hinge2", "hinge"),
                   help="shape of the one-sided barrier. `softplus` (default, "
                        "and what every run on record used) has INFINITE "
                        "support: a corner or an agent comfortably clear of the "
                        "margin still contributes, and since the barrier is "
                        "summed over 4 corners x N agents x T steps that "
                        "epsilon scales with how crowded and how long the scene "
                        "is rather than with anything the policy did. `hinge2` "
                        "is the quadratic-smoothed hinge: same knee width, same "
                        "unbounded ramp, still C^1, but exactly zero value AND "
                        "zero gradient at or beyond the margin. Measure the "
                        "difference first with tests/probe_barrier_tail.py -- "
                        "the margins were calibrated under softplus.")
    # See `SafetyLossConfig.edge_geometry`. The sign -- and so agreement with
    # waymax's `offroad` -- is identical under both; only the magnitude moves.
    p.add_argument("--edge_geometry", type=str, default="point",
                   choices=("point", "segment"),
                   help="how a corner's road-edge clearance is measured once the "
                        "nearest roadgraph sample is found. `point` (default, and "
                        "what every run on record used) is waymax's distance to "
                        "that SAMPLE: it overstates clearance by up to half the "
                        "~1 m sample spacing and jumps where the polyline is "
                        "crossed between samples. `segment` measures to the "
                        "incident polyline segment instead. --edge_margin was "
                        "calibrated under `point`; recalibrate it for `segment` "
                        "(slurm/calibrate_edge_geometry.sbatch) before comparing.")
    # See `SafetyFineTuner._init_windows` for the two time bases.
    p.add_argument("--safety_window", type=int, default=0,
                   help="WORLD steps of the candidate the safety term scores. "
                        "0 = the executed window (replan_interval_steps), which "
                        "is what every run before 2026-09-09 did; -1 = the whole "
                        "candidate (~50 steps / 5 s); N = N world steps. Scoring "
                        "past the replan prices trajectory that will be replanned "
                        "before it is driven -- but a 1 s window also cannot see a "
                        "violation the candidate commits at 2.5 s.")
    p.add_argument("--anchor_window", type=int, default=-1,
                   help="MODEL steps of the horizon the anchor/bc/base_anchor "
                        "terms are charged on. -1 = the whole horizon (25 steps / "
                        "5 s), the pre-2026-09-09 behaviour; 0 = auto, the "
                        "executed window in model steps (5 at replan=10); N = N "
                        "model steps. Windowing reweights the LOSS only -- the "
                        "denoiser still predicts the full horizon.")
    # ---- matched-sample preservation + where the safety term applies -------- #
    p.add_argument("--safety_rows", default="all", choices=("all", "unsolved"),
                   help="rows the SAFETY term is charged on. all = every live row "
                        "(every run before 2026-09-13). unsolved = only rows whose "
                        "collection rollout failed: a solved tight manoeuvre is left "
                        "to the anchors instead of being pushed off its line by the "
                        "barrier's pressure near the margin.")
    p.add_argument("--preserve_weight", type=float, default=0.0,
                   help="weight on the matched-continuation preservation term (see "
                        "`SafetyFineTuner.collect_preserve_contexts`): the trainable "
                        "late chain vs the frozen base's, from the SAME stored prefix "
                        "and noise, on states of base rollouts that SOLVED their "
                        "episode. 0 = off. Size from the median `preserve_share`.")
    p.add_argument("--preserve_scenes", type=int, default=16,
                   help="scenes in each iteration's preservation base rollout")
    p.add_argument("--preserve_window", type=int, default=0,
                   help="WORLD steps compared (0 = executed window, -1 = whole candidate)")
    p.add_argument("--preserve_scale_pos", type=float, default=1.0, help="metres")
    p.add_argument("--preserve_scale_yaw", type=float, default=0.1, help="radians")
    p.add_argument("--preserve_scale_speed", type=float, default=1.0, help="m/s")
    p.add_argument("--preserve_risk", default="",
                   help="risk.npz (scenario_risk) used to over-sample HARD successes; "
                        "empty = uniform over the success pool")
    p.add_argument("--preserve_hard_frac", type=float, default=0.5,
                   help="share of the non-replay preservation batch drawn from the hard set")
    p.add_argument("--preserve_hard_quantile", type=float, default=0.8,
                   help="hard set = success-pool scenes at or above this risk quantile")
    p.add_argument("--preserve_replay", type=int, default=256,
                   help="replay buffer capacity: success-pool scenes the policy recently broke")
    p.add_argument("--preserve_replay_per_iter", type=int, default=8)
    p.add_argument("--preserve_replay_uses", type=int, default=10,
                   help="draws before a replayed scene expires (re-breaking refreshes it)")
    p.add_argument("--repair_data", default="",
                   help="glob of counterfactual --mode repair repair_data/*.npz")
    p.add_argument("--repair_weight", type=float, default=0.0,
                   help="weight on the late-step denoising loss toward VERIFIED repairs "
                        "(see `SafetyFineTuner._init_repair`). 0 = off.")
    p.add_argument("--repair_batch", type=int, default=16)
    p.add_argument("--repair_steps", type=int, default=1,
                   help="Adam steps per iteration on the repair term, each on a fresh "
                        "batch (the first shares its step with every other term; the "
                        "rest are repair-only). 1 = every earlier run")
    p.add_argument("--repair_rows", choices=("all", "solved"), default="all",
                   help="'solved' keeps only rows whose EPISODE solved (filtered BC); "
                        "needs proposal_select --dump_bc data (ep_solved)")
    p.add_argument("--repair_split_ratio", default="",
                   help="fixed per-batch mix by collection split, e.g. "
                        "'failure_train=1,success_train=1'. '' = uniform over rows")
    p.add_argument("--repair_switch_share", type=float, default=0.0,
                   help="fixed share of every repair batch drawn from rows where the arm "
                        "EDITED the plan (`switched`); the rest follows --repair_split_ratio "
                        "over the kept rows. 0 = no switched pool (every earlier run)")
    # --- Diffusion-DPO on bank pairs (`SafetyFineTuner._init_pref`) ----------- #
    p.add_argument("--pref_data", default="",
                   help="glob of proposal_select --dump_bc bc_data/<arm>/*.npz carrying "
                        "`rejected` (collected 2026-10-01 or later)")
    p.add_argument("--pref_weight", type=float, default=0.0,
                   help="weight on the Diffusion-DPO objective (pairs + its success "
                        "anchor) next to every other term. 0 = off")
    p.add_argument("--pref_beta", type=float, default=1.0,
                   help="DPO beta: z = beta * [(l_bad - l_bad_ref) - (l_good - l_good_ref)]")
    p.add_argument("--pref_batch", type=int, default=16, help="pairs per step")
    p.add_argument("--pref_anchor_batch", type=int, default=16,
                   help="success_train kept rows per step for the anchor")
    p.add_argument("--pref_anchor_weight", type=float, default=100.0,
                   help="weight on base_anchor_loss over the success rows (100 = the "
                        "safety recipe's --base_anchor_weight). 0 = no anchor")
    p.add_argument("--pref_steps", type=int, default=1,
                   help="Adam steps per iteration on the preference objective")
    p.add_argument("--pref_rows", choices=("all", "solved"), default="solved",
                   help="pairs from switched rows of solved episodes only, or all")
    p.add_argument("--pref_t_min", type=int, default=0)
    p.add_argument("--pref_t_max", type=int, default=-1,
                   help="exclusive upper end of the pair's t draw; -1 = k_trainable "
                        "(the steps the trainable copy drives). Low-noise arm: smaller")
    p.add_argument("--pref_edit_weight", type=float, default=0.0,
                   help="lambda: model steps where |good - bad| > --pref_edit_delta_m "
                        "weigh 1 + lambda in both denoising losses. 0 = off")
    p.add_argument("--pref_edit_delta_m", type=float, default=0.1)
    # --- transition replay (rl/diffusion_ft/transition_buffer.py) ------------ #
    # The collection rollout's own reverse-diffusion transitions, stored one per
    # datum and replayed later. The safety loss is then charged on the CLEAN
    # PREDICTION reached from a stored (x_t, t) -- either directly (K=1, one
    # network eval) or after a short unroll (K>1) -- so the state the denoiser is
    # corrected at is one the deployed sampler actually visited, instead of the
    # `q_sample(x0_expert, t)` a teacher-forced unroll substitutes for it.
    p.add_argument("--unroll_eager", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="run the teacher-forced unroll through the old un-jitted "
                        "closure (the equivalence reference; ~6x slower per iter)")
    p.add_argument("--tb_weight", type=float, default=0.0,
                   help="weight on the replayed-transition safety term. 0 = off "
                        "(and nothing is stored, so the run is unchanged).")
    p.add_argument("--tb_capacity", type=int, default=8192,
                   help="buffer slots; FIFO. Capacity == tb_batch is on-policy.")
    p.add_argument("--tb_batch", type=int, default=32,
                   help="transitions per replayed minibatch (one gradient/iter)")
    p.add_argument("--tb_store_per_replan", type=int, default=1,
                   help="transitions stored per ALIVE row per replan, drawn "
                        "uniformly from the eligible trace timesteps")
    p.add_argument("--tb_max_uses", type=int, default=0,
                   help="draws before a slot is dropped (0 = unlimited)")
    p.add_argument("--tb_min_size", type=int, default=0,
                   help="buffer fill before the term switches on (0 = tb_batch)")
    p.add_argument("--tb_t_min", type=int, default=0)
    p.add_argument("--tb_t_max", type=int, default=-1,
                   help="stored timesteps are clipped to [tb_t_min, tb_t_max]; "
                        "-1 = k_trainable-1, i.e. the whole trainable window")
    p.add_argument("--tb_unroll_steps", type=int, default=1,
                   help="1 = score the one-step clean prediction from the stored "
                        "state (no unroll); K > 1 = unroll K-1 eta-DDIM steps down "
                        "to --tb_unroll_t_end and score the final clean prediction")
    p.add_argument("--tb_unroll_t_end", type=int, default=0)
    p.add_argument("--tb_unroll_eta", type=float, default=0.0)
    p.add_argument("--tb_grad_steps", type=int, default=1,
                   help="how many of the final evals carry a gradient; 1 means the "
                        "update differentiates exactly ONE transition")
    p.add_argument("--tb_window", type=int, default=0,
                   help="world steps of the replayed candidate to score "
                        "(0 = --safety_window)")
    p.add_argument("--n_discs", type=int, default=5)
    p.add_argument("--agent_geometry", default="exact", choices=("exact", "disc"))
    p.add_argument("--eval_every", type=int, default=10)
    # Eval reproducibility. `eval_seed` seeds a stream used ONLY by evals, so an
    # eval no longer depends on how much training preceded it; `eval_repeats`
    # averages that many fixed seeds (cost is linear); `--no-eval_base` drops the
    # once-computed frozen-base reference row.
    p.add_argument("--eval_seed", type=int, default=12345)
    p.add_argument("--eval_repeats", type=int, default=1)
    p.add_argument("--eval_base", action=argparse.BooleanOptionalAction, default=True)
    # Replan contexts averaged into one optimizer step. 0 == the whole iteration
    # (one step per iteration). `1` restores the pre-09-03 one-step-per-context
    # schedule, which is 8x the Adam steps on 1/8th the data each.
    p.add_argument("--grad_accum", type=int, default=0)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--resume", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    # Weights & Biases (off by default; the JSONL log is always written regardless,
    # and `tools/wandb_backfill.py` can push a finished JSONL up after the fact).
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False,
                   help="mirror every per-iter row to Weights & Biases (same keys as the JSONL)")
    p.add_argument("--wandb_project", type=str, default="diffusion_ft_safety")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None,
                   help="run name; defaults to a proposal/anchor-states/k/anchor-weight tag")
    p.add_argument("--wandb_group", type=str, default=None,
                   help="group name so a sweep's arms plot together")
    # ---- eval-only (post-hoc reach/solved scoring + driving dump) ------------ #
    p.add_argument("--eval_only", action="store_true",
                   help="load a checkpoint, score base+ft on the held-out eval "
                        "scenes, dump reach_rollouts.npz for viz, and exit (no train)")
    p.add_argument("--load_ckpt", default="",
                   help="checkpoint dir to load for --eval_only "
                        "(default: <out_dir>/latest)")
    p.add_argument("--dump_dir", default="",
                   help="where --eval_only writes its artifacts "
                        "(default: <out_dir>/eval)")
    p.add_argument("--eval_train", action="store_true",
                   help="with --eval_only, ALSO score a train-pool batch "
                        "(size = --scenes_per_iter, disjoint from the held-out "
                        "eval ids) -- the in-distribution number at a resolvable "
                        "scene count. Writes reach_*_train.{npz,jsonl,json}.")
    p.add_argument("--splits_json", default="",
                   help="named-split JSON (build_eval_splits.py output: "
                        "failure_heldout / success_heldout / failure_train / "
                        "success_train). During TRAINING, each split in --splits is "
                        "additionally evaluated every --eval_every and logged under "
                        "its own `<split>/` prefix; with --eval_only each is dumped "
                        "to reach_*_<split>.{npz,jsonl,json} instead.")
    p.add_argument("--splits", default="failure_heldout,success_heldout",
                   help="comma-separated split names in --splits_json to score")
    # ---- training pool: which scenes the gradient is actually taken on ------ #
    p.add_argument("--train_splits", default="",
                   help="comma-separated split names from --splits_json to TRAIN "
                        "on, e.g. 'failure_train' or "
                        "'failure_train,success_train'. Empty (the default) is a "
                        "uniform draw over the whole corpus -- under which a "
                        "48-scene split named `*_train` is touched ~0.3 times in "
                        "200 iterations and is NOT trained on in any useful "
                        "sense. A pool is a real distribution shift and is "
                        "printed at startup.")
    p.add_argument("--train_split_cap", default="-1",
                   help="per-split cap for --train_splits: one value for all, or "
                        "a comma list aligned to it; -1 keeps the split whole. "
                        "Equal caps are what balance a mixed pool -- "
                        "success_train is ~23k scenes against failure_train's "
                        "564, so an uncapped union is 98%% successes -- but "
                        "capping DISCARDS data; --train_split_ratio balances the "
                        "same pool without throwing any away.")
    p.add_argument("--train_split_ratio", default="",
                   help="per-batch quota for each of --train_splits, e.g. "
                        "'1,1' for a 50/50 failure/success batch. Empty = draw "
                        "uniformly over the union, under which a 400-scene "
                        "failure split inside a 23k-scene success pool is 1.7%% "
                        "of the gradient. Weights are normalized; every batch "
                        "gets each split's quota exactly, so batch composition "
                        "is constant and the trade-off in the curve is "
                        "attributable.")
    # ---- per-iteration rollout probe (the slider's data) -------------------- #
    p.add_argument("--probe_json", default="",
                   help="splits JSON the probe draws its fixed scenes from "
                        "(default: --splits_json)")
    p.add_argument("--probe_splits",
                   default="failure_train,success_train,failure_heldout,success_heldout",
                   help="splits the probe follows, one labelled group each")
    p.add_argument("--probe_per_split", type=int, default=4,
                   help="probe scenes per split (0 disables the probe). All of "
                        "them are rolled out in ONE batch, so the cost is one "
                        "extra eval pass per probe point.")
    p.add_argument("--probe_every", type=int, default=0,
                   help="dump a probe rollout every N iterations (0 = --eval_every)")
    p.add_argument("--probe_seed", type=int, default=0,
                   help="seed for WHICH scenes the probe follows (the rollout "
                        "itself always uses --eval_seed)")
    p.add_argument("--split_cap", type=int, default=48,
                   help="max scenes scored per split. Cost is linear in "
                        "cap x --eval_repeats x len(--splits) and is charged at "
                        "every --eval_every; a split larger than --eval_chunk is "
                        "rolled in chunks, so this is a time budget, not a "
                        "memory limit.")
    p.add_argument("--probe_offline", action="store_true",
                   help="do not train: re-select probe scenes POST-HOC from a "
                        "finished run so every transition bucket is populated "
                        "(the in-run probe fixes its scenes before training, so "
                        "its `regressed` bucket is empty whenever none of the "
                        "pre-picked successes happen to regress). Classifies "
                        "each split scene by its outcome at the last saved "
                        "checkpoint, picks --probe_offline_k per (base-outcome x "
                        "now-outcome x train/heldout) bucket, then re-rolls that "
                        "pinned set over the FROZEN base and every iter_* "
                        "checkpoint into --probe_offline_dir/probe for the "
                        "slider. GPU; reads --out_dir's checkpoints.")
    p.add_argument("--probe_offline_k", type=int, default=3,
                   help="scenes per transition bucket for --probe_offline")
    p.add_argument("--probe_offline_dir", default="",
                   help="where --probe_offline writes its probe/ dumps "
                        "(default <out_dir>/probe_offline)")
    p.add_argument("--verify_only", action="store_true",
                   help="do not train: re-score the FROZEN base per scene on "
                        "--splits and write verify_rows.jsonl, so the failure "
                        "splits can be rebuilt from scenes this policy actually "
                        "fails under the eval protocol (not under the scan's "
                        "deterministic one).")
    p.add_argument("--verify_repeats", type=int, default=4,
                   help="rollouts per scene in --verify_only; a scene counts as "
                        "a real failure only if the base fails ALL of them")
    p.add_argument("--verify_seed", type=int, default=777,
                   help="sampler seed for --verify_only. MUST differ from "
                        "--eval_seed: selecting scenes on the draws they are "
                        "later scored on bakes this sample's noise into the split.")
    p.add_argument("--verify_cap", default="",
                   help="per-split cap for --verify_only (one value or a comma "
                        "list aligned to --splits; -1 = all). success_train is "
                        "~23k scenes and does not need verifying -- only the "
                        "held-out sets are measurements.")
    p.add_argument("--verify_out", default="",
                   help="where --verify_only appends rows "
                        "(default <out_dir>/verify_rows.jsonl)")
    p.add_argument("--keep_every", type=int, default=0,
                   help="also write an un-overwritten snapshot to "
                        "<out_dir>/iter_%%06d every N iterations (0 = off). "
                        "`latest` alone cannot recover the best point of a long "
                        "run. Snapshots are weights only (~121 MB); the "
                        "optimizer state goes to `latest`, which is what a "
                        "resume reads.")
    p.add_argument("--eval_chunk", type=int, default=64,
                   help="scenes per eval rollout batch. A split longer than this "
                        "is scored in chunks and aggregated scene-weighted; the "
                        "result is what one batch of --split_cap would have given "
                        "and would not fit on the device.")
    return p


def main():
    args = build_argparser().parse_args()
    if not args.shard_glob and not args.tfrecord:
        raise SystemExit("give --shard_glob (full corpus) or --tfrecord (fixed batch)")
    if args.verify_only:
        if not args.splits_json:
            raise SystemExit("--verify_only needs --splits_json")
        if int(args.verify_seed) == int(args.eval_seed):
            raise SystemExit("--verify_seed must differ from --eval_seed")
        # `--splits` here names the CANDIDATE splits to re-measure, and the
        # trainer must not preload them as eval states (that is 4 x cap scenes
        # resident for nothing) -- verify streams its own chunks.
        names = [n.strip() for n in args.splits.split(",") if n.strip()]
        args.splits = ""
        tuner = SafetyFineTuner(args)
        if args.load_ckpt:
            # Score a FINE-TUNED checkpoint instead of the frozen base. The per-
            # scene rows are what a deployment gate needs: "run the fine-tune only
            # where a hardness score says so" can only be evaluated if the base
            # and the fine-tune are both known scene by scene, on the SAME draws.
            # Snapshot dirs (`iter_*`) carry weights only; `load` warns about the
            # missing optimizer state, which an eval does not use.
            print(f"[verify] scoring fine-tuned weights from {args.load_ckpt}",
                  flush=True)
            tuner.load(args.load_ckpt)
        else:
            print("[verify] scoring the FROZEN base checkpoint", flush=True)
        tuner.verify_splits(
            args.verify_out or (Path(args.out_dir) / "verify_rows.jsonl"),
            names=names, repeats=int(args.verify_repeats), ckpt=args.load_ckpt,
            seed=int(args.verify_seed), chunk=int(args.eval_chunk),
            caps=[c.strip() for c in str(args.verify_cap).split(",") if c.strip()])
        return
    if args.probe_offline:
        args.eval_only = True          # do not train; skip the in-run probe/marg
        tuner = SafetyFineTuner(args)
        if tuner.sampler is None:
            raise SystemExit("--probe_offline needs --shard_glob and --splits_json")
        run_dir = Path(args.out_dir)
        pdir = Path(args.probe_offline_dir) if args.probe_offline_dir else run_dir / "probe_offline"
        tuner.probe_offline(run_dir, pdir, int(args.probe_offline_k))
        return
    if args.eval_only:
        tuner = SafetyFineTuner(args)
        ckpt = args.load_ckpt or str(Path(args.out_dir) / "latest")
        dump = args.dump_dir or str(Path(args.out_dir) / "eval")
        print(f"[eval] scoring {ckpt}", flush=True)
        # Named splits (failure/success held-out) by explicit scene id, if given.
        if args.splits_json:
            if tuner.sampler is None:
                raise SystemExit("--splits_json needs --shard_glob (corpus mode)")
            spec = json.loads(Path(args.splits_json).read_text())
            if spec.get("shard_glob") and spec["shard_glob"] != args.shard_glob:
                raise SystemExit(
                    f"splits.json was built against a different glob:\n"
                    f"  splits: {spec['shard_glob']}\n  eval  : {args.shard_glob}\n"
                    f"scene ids are glob-relative; scoring them here would mislabel.")
            for name in [s for s in args.splits.split(",") if s.strip()]:
                ids = spec["splits"][name]["scene_ids"][: args.split_cap]
                if not ids:
                    print(f"[eval] split {name}: no scene ids, skipping", flush=True)
                    continue
                state = tuner.sampler.load_explicit(ids)
                tuner.dump_eval(dump, state, ckpt, tag=name)
        # Held-out batch (the run's own eval ids), then optionally an
        # in-distribution train-pool batch at --scenes_per_iter scenes.
        tuner.dump_eval(dump, tuner.eval_state, ckpt, tag="")
        if args.eval_train:
            if tuner.sampler is None:
                raise SystemExit("--eval_train needs --shard_glob (corpus mode)")
            train_state = tuner.sampler.next_train()   # size = scenes_per_iter
            tuner.dump_eval(dump, train_state, ckpt, tag="train")
        return
    SafetyFineTuner(args).train()


if __name__ == "__main__":
    main()
