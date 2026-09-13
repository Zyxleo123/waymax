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
and the state the denoiser sees is anchored to ground truth rather than to the
policy's own high-noise prefix. ``K=1`` degenerates to the one-step posterior
mean, which is the Jensen-gap control worth running as an ablation arm.

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
        --shard_glob '/zfsauton/scratch/eshau/womd/tf_example/training/*' \\
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
from rl.diffusion_ft.env import DiffusionWaymaxEnv
from rl.diffusion_ft.reward import RewardConfig
from rl.diffusion_ft.rl_actor import DiffusionRLActor
from rl.diffusion_ft.safety_loss import SafetyLossConfig, expert_clean_mask
from rl.diffusion_ft.safety_objective import SafetyObjectiveConfig, candidate_safety
from rl.diffusion_ft.scene_sampler import SceneSampler
from rl.diffusion_ft.unrolled_prediction import unroll_timestep_grid
from rl.diffusion_ft.train_dppo import load_batch


# --------------------------------------------------------------------------- #
class ReplanContext:
    """Everything needed to re-differentiate one replan after the env moved on.

    ``candidate_safety`` reads only ``log_trajectory``, ``object_metadata`` and
    ``roadgraph_points`` off the state -- all constant for the whole episode -- so
    the episode's state is held once by the caller and only ``t0`` varies here.
    """

    __slots__ = ("t0", "cond1", "cond2", "mask", "aux", "alive", "n_steps",
                 "n_exec", "expert_target", "x_prefix", "key_late")

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


class SafetyFineTuner:
    def __init__(self, args):
        self.args = args
        self.ckpt = load_diffusion_checkpoint(args.ckpt)
        self.sl_cfg = SafetyLossConfig(
            edge_margin_m=args.edge_margin, agent_margin_m=args.agent_margin,
            edge_tau_m=args.edge_tau, agent_tau_m=args.agent_tau,
            n_discs=args.n_discs, agent_geometry=args.agent_geometry)
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
            replan_interval_steps=args.replan_interval_steps, seed=args.seed)
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
                                     n_steps=ctx.n_steps, alive_b=ctx.alive)
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
                           "t_start": t_start_mean}
        return loss_fn

    # ------------------------------------------------------------------ #
    def _free_core(self, params, state, dyn, keys, *, n_steps, h,
                   has_expert, has_bc, has_base, has_prefix=False):
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
        if has_prefix:
            traj_ncT = self.actor.sample_late_for(
                params, dyn["cond1"], dyn["cond2"], dyn["mask"],
                dyn["x_prefix"], dyn["key_late"])
        else:
            traj_ncT = self.actor.sample_world_for(
                params, dyn["cond1"], dyn["cond2"], dyn["mask"], keys["propose"])
        world = self.env.norm_to_world_with(jnp.transpose(traj_ncT, (0, 2, 1)),
                                             dyn["aux"])
        terms = candidate_safety(self.objcfg, world, state=state, t0=t0,
                                 n_steps=n_steps, alive_b=dyn["alive"])
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
               "t_start": jnp.asarray(float(self.actor.diffusion.timesteps - 1))}
        return total, aux

    def _free_grad_fn(self):
        """Cached ``jit(value_and_grad(_free_core))``; grad w.r.t. ``params``."""
        if self._jit_free_grad is None:
            self._jit_free_grad = jax.jit(
                jax.value_and_grad(self._free_core, has_aux=True),
                static_argnames=("n_steps", "h", "has_expert", "has_bc",
                                 "has_base", "has_prefix"))
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
        if self.unroll_steps:
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
               "expert_target": expert_target,
               "bc_target": bc_target, "bc_cond1": bc_cond1, "bc_cond2": bc_cond2,
               "bc_mask": bc_mask, "bc_alive": bc_alive,
               "t0": jnp.asarray(ctx.t0, jnp.int32)}
        has_prefix = ctx.x_prefix is not None
        if has_prefix:
            dyn["x_prefix"], dyn["key_late"] = ctx.x_prefix, ctx.key_late

        # Canonicalize to device arrays of their own dtype. The first replan's
        # context differs from later ones in some leaf's abstract signature
        # (weak type / host array), which recompiled the gradient once more on
        # replan 1 (bench job 51578: 4 cache entries where 2 were expected).
        dyn = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype), dyn)
        keys = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype), keys)

        (loss, aux), grads = self._free_grad_fn()(
            self.actor._tparams, state, dyn, keys,
            n_steps=ctx.n_steps, h=h, has_expert=has_expert, has_bc=has_bc,
            has_base=has_base, has_prefix=has_prefix)
        return grads, loss, aux

    # ------------------------------------------------------------------ #
    def rollout(self, state, *, train: bool, params=None, rng=None,
                log_states=None):
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
            elif train and self.reuse_prefix:
                traj_ncT, (ctx.x_prefix, ctx.key_late) = \
                    self.actor.sample_world_with_prefix(
                        params, ctx.cond1, ctx.cond2, ctx.mask, rk)
                world = self.env.norm_to_world_with(
                    jnp.transpose(traj_ncT, (0, 2, 1)), ctx.aux)
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
               "data_seed": int(self.args.data_seed)}
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
            contexts, roll_m = self.rollout(state, train=True)
            t_collect = time.time() - t_start
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
            hist, step_hist = [], []
            buf, nbuf = None, 0
            for ep in range(a.epochs):
                for j in order:
                    ctx = contexts[int(j)]
                    if float(jnp.sum(ctx.alive)) <= 0.0:
                        continue
                    self.rng, rk = jax.random.split(self.rng)
                    # Paired by replan index so each step sees the human's
                    # states from the SAME window the safety term is scoring.
                    # Modular: the policy pass can end early (scenes absorb),
                    # so it may hold fewer contexts than the log pass.
                    bc_ctx = (bc_contexts[int(j) % len(bc_contexts)]
                              if bc_contexts else None)
                    grads, loss, aux = self._grads_for(state, ctx, rk, bc_ctx)
                    hist.append({"loss": float(np.asarray(loss)),
                                 **{k: float(np.asarray(v)) for k, v in aux.items()}})
                    buf = (grads if buf is None else
                           jax.tree_util.tree_map(jnp.add, buf, grads))
                    nbuf += 1
                    if nbuf >= accum:
                        step_hist.append(self.actor.apply_safety_grads(
                            jax.tree_util.tree_map(lambda x: x / nbuf, buf)))
                        buf, nbuf = None, 0
            if buf is not None:                  # partial group at the end
                step_hist.append(self.actor.apply_safety_grads(
                    jax.tree_util.tree_map(lambda x: x / nbuf, buf)))
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
            row = {"iter": it, "collect_s": round(t_collect, 2),
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
                   "anchor_last": hist[-1].get("anchor", 0.0),
                   "anchor_share": (a_term / (a_term + s_term)
                                    if a_term + s_term > 0 else 0.0),
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
                  f"ctx {row['n_ctx']} ({row['iter_s']}s)"
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
             "eval_repeats": self.args.eval_repeats}, indent=2))

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


# --------------------------------------------------------------------------- #
def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tfrecord", default="")
    p.add_argument("--indices", default="0,1,2,3,4,5,6,7")
    p.add_argument("--shard_glob", default="")
    p.add_argument("--index_dir", default="/zfsauton/scratch/yixiz/waymax_rs/womd_offsets")
    p.add_argument("--scenes_per_iter", type=int, default=16)
    p.add_argument("--eval_scenes", type=int, default=32)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--prefetch", type=int, default=1)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_replans", type=int, default=8)
    p.add_argument("--replan_interval_steps", type=int, default=10)
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
