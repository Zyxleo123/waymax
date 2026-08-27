"""DPPO fine-tuning of the diffusion policy against the simulator reward.

Primary direct-RL method (Phase 3E). Structure:

* **Collection** -- roll a batched episode in :class:`DiffusionWaymaxEnv`. At each
  replan the frozen scene tokenizer produces the condition; the actor
  (:class:`DiffusionRLActor`) samples a trajectory with the frozen base denoiser
  on the early denoising steps and the trainable copy on the last ``k_trainable``
  steps (RL sampling mode), recording those transitions and their old-policy
  log-probs; the sampled trajectory is executed by the env and its scalar reward
  stored. Only the executed prefix (``prefix_len`` model points) receives
  policy-gradient credit.
* **Advantages** -- GAE over env-step rewards with a state-only critic
  (:class:`ValueCritic`) on stopped-gradient scene features.
* **Update** -- clipped-PPO epochs on the trainable denoiser only (frozen old-
  policy snapshot per iteration; early stop on approx-KL), optionally plus one
  expert diffusion minibatch per update (DPPO + expert anchor), and a value MSE
  step.

Run via ``rl.diffusion_ft.train`` (unified CLI) or directly:
    python -m rl.diffusion_ft.train_dppo --indices 0,1,2,3,4,5,6,7 --iters 50
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from waymax import config as waymax_config

from data.scenario_loader import load_scenario_state_batch_fast
from rl.diffusion_ft.checkpoint import DEFAULT_CHECKPOINT_PATH, load_diffusion_checkpoint
from rl.diffusion_ft.env import DiffusionWaymaxEnv
from rl.diffusion_ft.reward import RewardConfig
from rl.diffusion_ft.rl_actor import DiffusionRLActor
from rl.diffusion_ft.scene_sampler import DEFAULT_INDEX_DIR, SceneSampler
from rl.diffusion_ft.value_net import ValueCritic


# --------------------------------------------------------------------------- #
def _rss_gb() -> float:
    """Current resident host RAM in GiB, from /proc/self/status (VmRSS).

    Printed per iteration so a slow memory creep (job 31985 host-OOM'd at 48G
    ~58 iters in) is visible as it grows, rather than as a surprise SIGKILL.
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)  # kB -> GiB
    except OSError:
        pass
    return 0.0


# --------------------------------------------------------------------------- #
def load_batch(tfr: str, indices: list[int], *, include_sdc_paths: bool = False,
               num_paths: int = 10, num_points_per_path: int = 300):
    """Load ``indices`` from ``tfr`` as a batched Waymax state.

    ``include_sdc_paths`` populates ``state.sdc_paths`` (required by the route
    reward's ``sdc_progression`` / ``sdc_off_route`` metrics). It stays off by
    default so goal-distance runs never pay for parsing routes. When on, the path
    tensor dims MUST match the tfrecord: ScenarioMax is ``10 x 300`` (the default,
    and what ``failures.tfrecord`` is), raw WOMD tf_example is ``45 x 800``.
    """
    n_unique = len(sorted(set(int(i) for i in indices)))
    ds_cfg = dataclasses.replace(
        waymax_config.WOD_1_3_1_TRAINING, path=str(tfr), max_num_objects=None,
        max_num_rg_points=30000, batch_dims=(n_unique,), shuffle_seed=0,
        include_sdc_paths=bool(include_sdc_paths),
        num_paths=int(num_paths), num_points_per_path=int(num_points_per_path),
    )
    state, idx_map = load_scenario_state_batch_fast(ds_cfg, indices)
    order = jnp.asarray([idx_map[int(i)] for i in indices], dtype=jnp.int32)
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x)[order], state)


def _check_routes(state, args, tag: str, labels: list[int] | None = None) -> None:
    """Report how many scenes carry a usable SDC route, when the route reward is on.

    The route reward is silently dead on a scene with no valid on-route path --
    its progression/off_route contribution is identically 0 forever. Printing the
    count up front makes a degenerate draw visible before a whole run, rather than
    read back afterwards as "the reward isn't learning". With full-corpus sampling
    the count is per-draw, so this reports the *first* batch of each set; a
    persistent shortfall means the path dims are wrong for the tfrecord (raw WOMD
    is 45x800, ScenarioMax 10x300).
    """
    if not args.route_reward:
        return
    sp = state.sdc_paths
    if sp is None:
        raise SystemExit(
            "--route_reward needs sdc_paths but the loader returned None "
            "(check num_paths/num_points_per_path match the tfrecord).")
    n = int(np.asarray(state.log_trajectory.x).shape[0])
    v = np.asarray(sp.valid).reshape(n, -1).any(axis=1)
    print(f"[dppo] route[{tag}]: {int(v.sum())}/{n} scenes have a valid sdc_path "
          f"(num_paths={args.num_paths}x{args.num_points_per_path})")
    if not v.all():
        bad = [labels[i] if labels else i for i in np.where(~v)[0]]
        print(f"[dppo] WARNING: no valid route for scenes {bad}; their "
              f"progression/off_route reward is identically 0.")


def compute_gae(rewards, values, done, last_value, gamma, lam):
    """GAE over ``[n_steps, B]`` env-step rewards. Returns (adv, returns).

    ``done`` is ``terminated | truncated``: both a real terminal (goal/crash) and
    the WOMD horizon cut-off end the finite-horizon return, so neither bootstraps
    the critic. This is what makes ``lam=1`` a genuine Monte-Carlo advantage --
    treating the horizon truncation as non-terminal (bootstrapping ``V(s_H)``)
    would leave the critic in the return path for exactly the scenes that end by
    reaching the horizon, i.e. almost all of them, defeating the intent.

    At ``lam=1`` this degenerates to the Monte-Carlo advantage: ``adv`` is the
    observed discounted return minus ``V(s_t)``, and ``returns`` is the observed
    discounted return alone. That matters here because ``returns = adv + values``
    is otherwise self-referential -- with a critic that under-predicts, the GAE
    deltas are large and positive, the targets inflate, and the critic chases a
    target it is itself pushing away (observed: return-to-go 8 -> 38 -> 86 while
    the true episode return stayed ~20). At ``lam=1`` the target is bounded by
    the rewards actually collected, so a bad critic costs variance, not drift.
    """
    n, B = rewards.shape
    adv = np.zeros((n, B), dtype=np.float64)
    lastgae = np.zeros(B, dtype=np.float64)
    for t in reversed(range(n)):
        nextnonterm = 1.0 - done[t]
        nextval = last_value if t == n - 1 else values[t + 1]
        delta = rewards[t] + gamma * nextval * nextnonterm - values[t]
        lastgae = delta + gamma * lam * nextnonterm * lastgae
        adv[t] = lastgae
    returns = adv + values
    return adv, returns


# --------------------------------------------------------------------------- #
class DPPOTrainer:
    def __init__(self, args):
        self.args = args
        self.ckpt = load_diffusion_checkpoint(args.ckpt)
        self.cond_dim = int(self.ckpt.metadata["cond_dim"])
        self.critic = ValueCritic(self.cond_dim, lr=args.value_lr, seed=args.seed)
        self._rng = jax.random.PRNGKey(args.seed + 1)
        self.env = None
        self.actor = None

    # --------------------------------------------------------------- #
    def _build_for(self, state):
        """Build the env + actor sized to the loaded scenarios' object count.

        The env asserts ``max_num_objects == num_objects`` and WOMD pads objects,
        so the env must adopt the batch's actual count (see Gate 1 notes).
        """
        n_obj = int(jnp.asarray(state.log_trajectory.x).shape[1])
        if self.env is not None and self.env._max_num_objects == n_obj:
            return
        a = self.args
        # Defaults reproduce the original bare RewardConfig() (progress=1.0/metre,
        # offroad=-5 once & non-terminal, collision=-10, goal=+10, route off), so a
        # launch that sets none of the reward flags is byte-identical. The V-Max
        # parity recipe (route arclength progress, terminate-on-offroad, bounded
        # magnitudes, goal shaping) is turned on explicitly via the new flags.
        reward_cfg = RewardConfig(
            progress=a.reward_progress,
            collision=a.reward_collision,
            offroad=a.reward_offroad,
            goal_bonus=a.reward_goal_bonus,
            terminate_on_collision=a.terminate_on_collision,
            terminate_on_offroad=a.terminate_on_offroad,
            offroad_per_step=a.offroad_per_step,
            collision_per_step=a.collision_per_step,
            route_reward=a.route_reward,
            # V-Max `progression` is a boolean indicator (+w on any arclength gain),
            # and `off_route` a bounded indicator (fires when off-route dist > 0).
            progression_indicator=a.route_reward,
            off_route_threshold_m=(0.0 if a.route_reward else None),
            off_route_penalty=a.reward_off_route,
        )
        self.env = DiffusionWaymaxEnv(
            self.ckpt, reward_config=reward_cfg, max_num_objects=n_obj,
            use_ema=self.args.use_ema,
            replan_interval_steps=self.args.replan_interval_steps, seed=self.args.seed,
            goal_shaping_coef=a.goal_shaping_coef,
            goal_shaping_discount=a.gamma,     # must match GAE's discount to telescope
            goal_shaping_normalize=a.goal_shaping_normalize,
        )
        self.actor = DiffusionRLActor(
            self.env.policy.diffusion,
            k_trainable=self.args.k_trainable,
            prefix_len=self.args.prefix_len,
            sigma_sample_floor=self.args.sigma_sample_floor,
            sigma_logprob_floor=self.args.sigma_logprob_floor,
            actor_lr=self.args.actor_lr,
            grad_clip_norm=self.args.grad_clip_norm,
            ppo_clip=self.args.ppo_clip,
            pg_scale=(self.args.pg_scale if self.args.pg_scale > 0 else None),
            num_sample_steps=(self.args.collect_steps if self.args.collect_steps > 0 else None),
            expert_lr=(self.args.expert_lr if self.args.expert_lr > 0 else None),
            seed=self.args.seed,
        )

    # --------------------------------------------------------------- #
    def collect(self, state):
        """Roll one batched episode; return the per-step buffer + GAE targets."""
        env, actor, critic = self.env, self.actor, self.critic
        features = env.reset(state)
        B = env.batch_size
        horizon = int(jnp.asarray(env.sim_state.log_trajectory.x).shape[-1])
        n_steps = self.args.max_replans or ((horizon - env.timestep) // self.args.replan_interval_steps)

        buf = {k: [] for k in ("cond1", "cond2", "mask", "trace", "expert_target",
                               "reward", "value", "done", "alive")}
        prev_done = np.zeros(B, dtype=bool)

        # Optional per-phase timing. jax dispatch is async, so a fair split needs a
        # block_until_ready barrier after each phase -- which itself costs time, so
        # this is gated behind --profile and off in real runs.
        prof = bool(getattr(self.args, "profile", False))
        t_sample = t_env = 0.0

        for _ in range(int(n_steps)):
            if prof:
                te = time.time()
            cond1, cond2 = env.compute_condition(features)
            mask = features["inst_valid"]
            self._rng, krng = jax.random.split(self._rng)
            if prof:
                jax.block_until_ready((cond1, cond2)); t_env += time.time() - te; ts = time.time()
            traj_ncT, trace = actor.collect(cond1, cond2, mask, rng=krng)
            if prof:
                jax.block_until_ready(traj_ncT); t_sample += time.time() - ts; te = time.time()
            world = env.norm_to_world(jnp.transpose(traj_ncT, (0, 2, 1)))
            value = np.asarray(critic.value(cond1))

            buf["cond1"].append(cond1); buf["cond2"].append(cond2); buf["mask"].append(mask)
            buf["trace"].append(trace); buf["expert_target"].append(features["ego_trajectory"])
            buf["value"].append(value); buf["alive"].append(~prev_done)

            out = env.step(trajectory_world_bt5=world)
            buf["reward"].append(np.asarray(out.reward_b))
            if prof:
                jax.block_until_ready(out.reward_b); t_env += time.time() - te
            # `done` = terminated | truncated: the horizon cut-off ends the
            # finite-horizon return just as a real terminal does, so GAE must not
            # bootstrap past it (see `compute_gae`).
            done = np.asarray(out.terminated_b) | np.asarray(out.truncated_b)
            buf["done"].append(done)
            prev_done = prev_done | done
            features = out.features
            if prev_done.all():
                break

        cond1_last, _ = env.compute_condition(features)
        last_value = np.asarray(critic.value(cond1_last))

        rewards = np.stack(buf["reward"], 0)
        values = np.stack(buf["value"], 0)
        done = np.stack(buf["done"], 0).astype(np.float64)
        adv, returns = compute_gae(rewards, values, done, last_value,
                                   self.args.gamma, self.args.gae_lambda)
        alive = np.stack(buf["alive"], 0)  # [n, B]
        self._prof_collect = {"t_sample": round(t_sample, 2), "t_env": round(t_env, 2)}
        return buf, adv, returns, alive, rewards

    # --------------------------------------------------------------- #
    def _merge(self, buf, adv, returns, alive):
        """Flatten the [n_steps, B] buffer into a single [n_steps*B] batch."""
        n = len(buf["cond1"])
        cat = lambda key: jnp.concatenate([buf[key][t] for t in range(n)], axis=0)
        cond1 = cat("cond1"); cond2 = cat("cond2"); mask = cat("mask")
        expert_target = cat("expert_target")
        # trace: list over steps of {field: [K, B, ...]} -> {field: [K, n*B, ...]}
        keys = buf["trace"][0].keys()
        trace = {k: jnp.concatenate([buf["trace"][t][k] for t in range(n)], axis=1) for k in keys}
        alive_f = jnp.asarray(alive.reshape(-1).astype(np.float32))
        adv_f = jnp.asarray(adv.reshape(-1).astype(np.float32)) * alive_f
        ret_f = jnp.asarray(returns.reshape(-1).astype(np.float32))
        # Normalize advantages over the alive set.
        m = alive.reshape(-1)
        if m.sum() > 1:
            a = np.asarray(adv_f)[m]
            adv_f = jnp.asarray(((np.asarray(adv_f) - a.mean()) / (a.std() + 1e-8)) * m.astype(np.float32))
        old_lp = trace["log_prob"]  # [K, n*B]
        return cond1, cond2, mask, trace, old_lp, adv_f, ret_f, expert_target, alive_f

    # --------------------------------------------------------------- #
    def update(self, merged, *, pg_enabled=True):
        cond1, cond2, mask, trace, old_lp, adv, ret, expert_target, alive = merged
        args = self.args
        last = {}
        # During warmup the actor is frozen: skip PG entirely and only fit the
        # critic on this iteration's GAE targets. `_behavior_tparams` is still
        # snapshotted in `collect`, just never stepped, so unfreezing later is a
        # no-op on the policy side.
        epoch_iter = range(args.ppo_epochs) if pg_enabled else range(0)
        if not pg_enabled:
            last["warmup"] = 1.0
        for epoch in epoch_iter:
            expert = None
            if args.expert_weight > 0.0:
                self._rng, erng = jax.random.split(self._rng)
                expert = (expert_target, cond1, cond2, mask, erng)
            m = self.actor.ppo_update(
                cond1, cond2, mask, trace, old_lp, adv, alive,
                expert=expert, expert_weight=args.expert_weight,
                target_kl=args.target_kl,
            )
            m = {k: float(np.asarray(v)) for k, v in m.items()}
            if m["applied"] < 0.5:
                # The step that would have crossed target_kl was declined inside
                # ppo_update; keep the last *applied* epoch's metrics and stop.
                # But retain the *declined* candidate's diagnostics (its KL and
                # grad norm) so a rejected step is distinguishable from a dead
                # gradient -- otherwise both print pg=gnorm=0 and a trust-region
                # that never opens looks like a plumbing bug (see the s*_k* sweep
                # where every epoch-0 candidate overshot a sigma-inflated KL).
                for k in ("grad_global_norm", "pg_raw", "pg_loss", "kl_gauss_candidate",
                          "approx_kl", "approx_kl_transition", "mean_ratio", "clip_frac"):
                    if k in m:
                        last.setdefault(k, m[k])
                last["stopped_epoch"] = epoch
                last["applied"] = m["applied"]
                break
            last = m
        # ---- expert (behavior-cloning) steps ----------------------------- #
        # Run AFTER the PPO epochs, never interleaved. A PPO epoch's importance
        # ratio is only valid against the behavior policy that collected this
        # iteration's trace; an expert step in between moves the parameters with no
        # ratio correction, so every subsequent epoch would be optimizing a
        # surrogate for a policy that no longer exists. Ordering them last keeps
        # both terms exact and makes `expert_steps : ppo_epochs` an honest ratio.
        # Targets are this iteration's on-policy states with their log-trajectory
        # continuations -- a DAgger-style recovery anchor, not offline BC.
        n_expert = int(args.expert_steps)
        if n_expert > 0:
            self.actor.set_expert_kl_probe(cond1, cond2, mask, trace)
            exp_losses = []
            for e in range(n_expert):
                self._rng, erng = jax.random.split(self._rng)
                em = self.actor.expert_update(
                    (expert_target, cond1, cond2, mask, erng), alive,
                    measure_kl_vs_behavior=(e == n_expert - 1),
                )
                exp_losses.append(em["expert_loss"])
                last["expert_grad_norm"] = em["expert_grad_norm"]
                if "kl_gauss_expert" in em:
                    last["kl_gauss_expert"] = em["kl_gauss_expert"]
            last["expert_steps"] = float(n_expert)
            last["expert_loss_first"] = exp_losses[0]
            last["expert_loss_last"] = exp_losses[-1]
            # The ratio must be read off the DPPO steps that were actually APPLIED,
            # not the requested --ppo_epochs: an iteration whose first candidate was
            # rejected by the KL gate did zero DPPO steps, and the configured ratio
            # would misreport that iteration as balanced when it was pure BC.
            # `stopped_epoch` is the index the early stop broke at, which equals the
            # number of epochs committed before it.
            ppo_applied = float(last.get("stopped_epoch",
                                         args.ppo_epochs if pg_enabled else 0))
            last["ppo_steps_applied"] = ppo_applied
            # -1 = "no DPPO step landed this iteration", which is a distinct fact
            # from a large ratio; a division would report it as +inf or a wild
            # finite number depending on the epsilon.
            last["expert_to_dppo_ratio"] = (
                n_expert / ppo_applied if ppo_applied > 0 else -1.0)
        # Returns from already-terminated envs are meaningless -- mask them out
        # (adv is masked in `_merge`; ret was not).
        vloss = self.critic.update(cond1, ret, weights=alive, epochs=args.value_epochs)
        last["value_loss"] = float(np.asarray(vloss["value_loss"]))
        last["value_loss_final"] = float(np.asarray(vloss["value_loss_final"]))
        return last

    # --------------------------------------------------------------- #
    def evaluate(self, state, *, deterministic: bool, seed_offset: int) -> tuple[float, float]:
        """Roll one batched episode with a *fixed* rng (no training).

        Returns a dict with ``return``/``solve`` plus the un-collapsed
        ``reached``/``collided``/``offroad`` rates. The shaped return is only a proxy
        (a clean solve is strongly positive, ~0 is break-even), so we also report
        the fraction of scenes actually *solved* -- reached the goal without ever
        colliding or going offroad -- the metric we care about. The per-replan
        ``reached/collision/offroad`` flags are OR-accumulated over the *live*
        steps only, matching how the return masks terminated worlds.

        Stochastic eval (``deterministic=False``) samples from the same policy PPO
        optimizes; deterministic eval runs the standard sampler over the trainable
        late steps -- the deployment trajectory, free of exploration-floor jitter.
        A fixed seed makes both comparable across iterations, so an improvement is
        the policy moving, not the noise landing differently.
        """
        env, actor = self.env, self.actor
        features = env.reset(state)
        horizon = int(jnp.asarray(env.sim_state.log_trajectory.x).shape[-1])
        n_steps = self.args.max_replans or ((horizon - env.timestep) // self.args.replan_interval_steps)
        rng = jax.random.PRNGKey(int(self.args.seed) + int(seed_offset))
        B = env.batch_size
        prev_done = np.zeros(B, dtype=bool)
        ret = np.zeros(B, dtype=np.float64)
        reached = np.zeros(B, dtype=bool)
        collided = np.zeros(B, dtype=bool)
        offroad = np.zeros(B, dtype=bool)
        for _ in range(int(n_steps)):
            cond1, cond2 = env.compute_condition(features)
            mask = features["inst_valid"]
            rng, krng = jax.random.split(rng)
            traj_ncT = actor.act(cond1, cond2, mask, rng=krng, deterministic=deterministic)
            world = env.norm_to_world(jnp.transpose(traj_ncT, (0, 2, 1)))
            out = env.step(trajectory_world_bt5=world)
            live = ~prev_done
            comps = out.components
            # Report the TRUE task return, not the shaped one: potential-based
            # goal shaping is a training-time credit accelerator (V-Max applies it
            # to the train env only), so subtract it back out here to keep the eval
            # number comparable to un-shaped runs. It is 0 when shaping is off.
            task_reward_b = np.asarray(out.reward_b) - np.asarray(
                comps.get("shaping", np.zeros(B, dtype=np.float32)))
            ret += task_reward_b * live
            zb = np.zeros(B, dtype=bool)
            reached |= (np.asarray(comps.get("reached_flag", zb)).astype(bool) & live)
            collided |= (np.asarray(comps.get("collision_flag", zb)).astype(bool) & live)
            offroad |= (np.asarray(comps.get("offroad_flag", zb)).astype(bool) & live)
            prev_done = prev_done | np.asarray(out.terminated_b) | np.asarray(out.truncated_b)
            features = out.features
            if prev_done.all():
                break
        solved = reached & ~collided & ~offroad
        # Return the un-collapsed rates too: `solved` is an AND gate over the whole
        # episode, so a flat 0 hides *why* (never reached vs reached-but-dirty).
        return {
            "return": float(ret.mean()),
            "solve": float(solved.mean()),
            "reached": float(reached.mean()),
            "collided": float(collided.mean()),
            "offroad": float(offroad.mean()),
        }

    # --------------------------------------------------------------- #
    def _init_wandb(self, indices, out_dir):
        """Start a W&B run mirroring the JSONL rows, or return None if disabled."""
        args = self.args
        if not args.wandb:
            return None
        import wandb

        name = args.wandb_name or (
            f"s{args.sigma_sample_floor}_k{args.target_kl}_w{args.warmup_iters}")
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=name, group=args.wandb_group,
            config={**vars(args), "indices": indices, "n_scenes": len(indices),
                    # With full-corpus sampling `indices` is vestigial -- record
                    # what the run actually trains on so a sweep's arms are
                    # comparable in the W&B config table.
                    "data_regime": "corpus" if self.sampler is not None else "fixed",
                    "expert_to_dppo": f"{args.expert_steps}:{args.ppo_epochs}",
                    "out_dir": str(out_dir)},
        )
        # One shared step axis = the DPPO iteration. Sparse keys (eval every N,
        # or pg metrics suppressed during warmup) plot with gaps, which is the
        # honest picture -- no forward-fill.
        wandb.define_metric("iter")
        wandb.define_metric("*", step_metric="iter")
        return run

    # --------------------------------------------------------------- #
    def train(self):
        args = self.args
        out_dir = Path(args.out_dir) / time.strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "train_log.jsonl"
        indices = [int(x) for x in args.indices.split(",") if x.strip()]

        # Two data regimes:
        #  * `--shard_glob` (full corpus): a FRESH batch of scenes every
        #    `--resample_every` iterations, drawn uniformly over all 1000 raw-WOMD
        #    training shards, with a fixed held-out eval set. This is the regime
        #    where the expert term is a coherent anchor -- it is the corpus the base
        #    checkpoint was pretrained on -- and where the eval number means
        #    generalization rather than memorization.
        #  * `--indices` on one tfrecord (the historical overfit target): unchanged
        #    behavior, one fixed batch loaded once, eval on the training scenes.
        self.sampler = None
        if args.shard_glob:
            self.sampler = SceneSampler(
                args.shard_glob, batch_size=args.scenes_per_iter,
                eval_size=args.eval_scenes, include_sdc_paths=args.route_reward,
                num_paths=args.num_paths, num_points_per_path=args.num_points_per_path,
                seed=args.data_seed, prefetch=args.prefetch,
                index_dir=(args.index_dir or None),
            )
            print(f"[dppo] data: {self.sampler.describe()} "
                  f"(resample every {args.resample_every} iters)")
            state = self.sampler.next_train()
            eval_state = self.sampler.eval_batch() if args.eval_scenes > 0 else state
            _check_routes(state, args, "train")
            if args.eval_scenes > 0:
                _check_routes(eval_state, args, "eval")
        else:
            state = load_batch(
                args.tfrecord, indices, include_sdc_paths=args.route_reward,
                num_paths=args.num_paths, num_points_per_path=args.num_points_per_path,
            )
            eval_state = state
            _check_routes(state, args, "train", labels=indices)
        self._build_for(state)
        if args.resume:
            self.load(args.resume)
        wandb_run = self._init_wandb(
            indices if self.sampler is None else list(map(int, self.sampler.last_ids)),
            out_dir)

        scene_desc = (f"full-corpus x{args.scenes_per_iter}" if self.sampler is not None
                      else f"fixed {indices}")
        print(f"[dppo] out={out_dir} scenes={scene_desc} k_trainable={args.k_trainable} "
              f"expert_steps={args.expert_steps} ppo_epochs={args.ppo_epochs} "
              f"(ratio {args.expert_steps}:{args.ppo_epochs}) "
              f"expert_weight={args.expert_weight}")
        # The reward config decides what the run is measuring; log it so a .out
        # file identifies its own recipe without diffing the env file.
        print(f"[dppo] reward: route={int(args.route_reward)} "
              f"term_offroad={int(args.terminate_on_offroad)} "
              f"offroad_per_step={int(args.offroad_per_step)} "
              f"term_collision={int(args.terminate_on_collision)} "
              f"collision_per_step={int(args.collision_per_step)} "
              f"progress={args.reward_progress} offroad={args.reward_offroad} "
              f"collision={args.reward_collision} goal={args.reward_goal_bonus} "
              f"off_route={args.reward_off_route} shaping={args.goal_shaping_coef}")
        for it in range(args.iters):
            # Global step for logging/plots: lets a --resume run continue the
            # curve (--iter_offset) instead of restarting at 0. Loop control
            # (eval cadence, warmup, save) stays on the local `it`.
            gstep = it + int(args.iter_offset)
            t0 = time.time()
            # Fresh scenes. The prefetch thread already loaded them while the last
            # iteration was on the GPU, so `next_train()` is normally a queue pop.
            if self.sampler is not None and it > 0 and it % args.resample_every == 0:
                state = self.sampler.next_train()
            buf, adv, returns, alive, rewards = self.collect(state)
            t_collect = time.time() - t0
            # Advantage diagnostics on the *raw* (pre-normalization) GAE over the
            # alive transitions. A zero gnorm with a healthy log-prob Jacobian is
            # not a plumbing bug -- it is a degenerate advantage batch: `_merge`
            # normalizes `(a - mean) / (std + 1e-8)`, so a zero-variance alive set
            # (one alive element, or all-equal returns) collapses every advantage
            # to exactly 0 and the PG gradient vanishes. Surface the cause here so
            # the real run reports it instead of looking like a dead gradient.
            m_alive = alive.reshape(-1).astype(bool)
            a_alive = adv.reshape(-1)[m_alive]
            r_alive = rewards.reshape(-1)[m_alive] if m_alive.any() else rewards.reshape(-1)
            adv_diag = {
                "n_alive": int(m_alive.sum()),
                "adv_std_raw": float(a_alive.std()) if a_alive.size else 0.0,
                "adv_absmean_raw": float(np.abs(a_alive).mean()) if a_alive.size else 0.0,
                "reward_std": float(r_alive.std()),
            }
            merged = self._merge(buf, adv, returns, alive)
            tu = time.time()
            metrics = self.update(merged, pg_enabled=(it >= args.warmup_iters))
            t_update = time.time() - tu
            ep_return = float(np.sum(rewards, axis=0).mean())
            eval_row: dict[str, float] = {}
            if args.eval_every > 0 and (it % args.eval_every == 0 or it + 1 == args.iters):
                # Average each eval over `eval_seeds` fixed seeds: single-seed eval
                # on a handful of scenes swings +-3-4 between points (the s*/lr*
                # sweep peaks were noise spikes that regressed), so report the mean
                # and the across-seed std -- the real noise band the trend must beat.
                nseed = max(1, int(args.eval_seeds))
                # Always the FIXED (held-out, when sampling) eval set -- never the
                # iteration's training draw, which would measure the draw, not the
                # policy.
                det = [self.evaluate(eval_state, deterministic=True, seed_offset=778 + 2 * s)
                       for s in range(nseed)]
                sto = [self.evaluate(eval_state, deterministic=False, seed_offset=1777 + 2 * s)
                       for s in range(nseed)]
                det_ret = np.array([d["return"] for d in det])
                sto_ret = np.array([d["return"] for d in sto])
                det_solve = np.array([d["solve"] for d in det])
                sto_solve = np.array([d["solve"] for d in sto])
                eval_row["eval_return_deterministic"] = float(det_ret.mean())
                eval_row["eval_return_stochastic"] = float(sto_ret.mean())
                # Solve rate = reached goal & no collision & no offroad (the real
                # objective; the shaped return is only a proxy).
                eval_row["eval_solve_deterministic"] = float(det_solve.mean())
                eval_row["eval_solve_stochastic"] = float(sto_solve.mean())
                # Un-collapsed rates: when solve is pinned at 0, these say whether
                # the wall is "never reaches goal" or "reaches but goes dirty".
                for k in ("reached", "collided", "offroad"):
                    eval_row[f"eval_{k}_deterministic"] = float(np.mean([d[k] for d in det]))
                    eval_row[f"eval_{k}_stochastic"] = float(np.mean([d[k] for d in sto]))
                if nseed > 1:
                    eval_row["eval_return_deterministic_std"] = float(det_ret.std())
                    eval_row["eval_return_stochastic_std"] = float(sto_ret.std())
                    eval_row["eval_solve_deterministic_std"] = float(det_solve.std())
            prof_row = {"t_collect": round(t_collect, 2), "t_update": round(t_update, 2),
                        **getattr(self, "_prof_collect", {})} if args.profile else {}
            data_row = ({"scenes_seen": self.sampler.n_loaded,
                         "data_load_s": round(self.sampler.load_seconds, 1)}
                        if self.sampler is not None else {})
            row = {
                "iter": gstep, "mean_episode_return": ep_return,
                "mean_return_to_go": float(returns.mean()),
                "n_steps": len(buf["cond1"]), "time_s": round(time.time() - t0, 2),
                "rss_gb": round(_rss_gb(), 2),
                **prof_row, **data_row, **adv_diag, **metrics, **eval_row,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            if wandb_run is not None:
                import wandb
                wandb.log(row, step=gstep)
            # Fixed-point formats rounded pg/kl/(ratio-1) to zero and hid the fact
            # that the gradient was nonzero all along -- keep these in scientific
            # notation, and show the pg-vs-expert split that decides what the
            # gradient is actually optimizing. Report both the averaged KL (what
            # the clip acts on) and the per-transition KL (the real trust region).
            if not eval_row:
                eval_str = ""
            else:
                pm = (f"(±{eval_row['eval_return_deterministic_std']:.2f})"
                      if "eval_return_deterministic_std" in eval_row else "")
                eval_str = (f"eval[s/d]={eval_row['eval_return_stochastic']:.3f}/"
                            f"{eval_row['eval_return_deterministic']:.3f}{pm} "
                            f"solve[d]={eval_row['eval_solve_deterministic']:.2f} ")
            warm_str = "WARMUP " if it < args.warmup_iters else ""
            print(f"[dppo] it={gstep:04d} {warm_str}return={ep_return:.3f} {eval_str}"
                  f"n_alive={adv_diag['n_alive']} advstd={adv_diag['adv_std_raw']:.3e} "
                  f"rstd={adv_diag['reward_std']:.3e} "
                  f"pg={metrics.get('pg_loss',0):+.3e} exp={metrics.get('expert_term',0):.3e} "
                  f"xL={metrics.get('expert_loss_last',metrics.get('expert_loss',0)):.3e} "
                  f"kl_x={metrics.get('kl_gauss_expert',0):.3e} "
                  f"nx/np={int(metrics.get('expert_steps',0))}/"
                  f"{int(metrics.get('ppo_steps_applied',0))} "
                  f"gnorm={metrics.get('grad_global_norm',0):.3e} "
                  f"kl_g={metrics.get('kl_gauss',0):.3e} "
                  f"klc={metrics.get('kl_gauss_candidate',0):.3e} "
                  f"kl={metrics.get('approx_kl',0):.3e} "
                  f"kl_tx={metrics.get('approx_kl_transition',0):.3e} "
                  f"ratio-1={metrics.get('mean_ratio',1)-1:+.3e} "
                  f"clipf={metrics.get('clip_frac',0):.3f} "
                  f"vloss={metrics.get('value_loss',0):.1f}->{metrics.get('value_loss_final',0):.1f} "
                  f"t={row['time_s']}s rss={row['rss_gb']:.1f}G nstep={row['n_steps']}"
                  + (f" [collect={row['t_collect']}s(sample={row.get('t_sample','?')}"
                     f"/env={row.get('t_env','?')}) update={row['t_update']}s]"
                     if args.profile else ""))
            if (it + 1) % args.save_every == 0 or it + 1 == args.iters:
                self.save(out_dir / f"ckpt_{it+1:04d}")
            # Host RAM grew ~0.6 GiB/iter at a *constant* batch shape (leak probe
            # job 32647): not recompilation. gc.collect() alone did NOT help (probe
            # 32673, identical slope), so it is not Python reference cycles but
            # retention inside JAX/flax caches -- the per-iter, un-jitted
            # value_and_grad retraces (fresh loss closures + nnx.merge rebuilding
            # modules every critic/actor epoch). clear_caches flushes those; it also
            # drops the good _jit_log_probs/_jit_means_std executables, so it is
            # opt-in (a probe / stopgap) until the update steps are jitted properly.
            gc.collect()
            if args.clear_caches:
                jax.clear_caches()
        if wandb_run is not None:
            import wandb
            wandb.finish()
        print(f"[dppo] done -> {out_dir}")

    def save(self, path: Path):
        import pickle
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "actor_value.pkl", "wb") as f:
            pickle.dump({
                "actor": jax.tree_util.tree_map(np.asarray, self.actor.state_dict()),
                "critic": jax.tree_util.tree_map(np.asarray, self.critic.state_dict()),
            }, f)

    def load(self, path: str) -> None:
        """Restore the fine-tuned actor+critic from a `save()` checkpoint (dir or
        the actor_value.pkl itself), so a run can continue past its iter budget.

        The base diffusion ckpt is still loaded normally in __init__ (it provides
        the frozen denoiser + graphdefs); this overrides only the *trainable* copy
        and the critic. Arrays are re-cast to jnp so the jitted steps trace cleanly.
        """
        import pickle
        p = Path(path)
        pkl = p / "actor_value.pkl" if p.is_dir() else p
        with open(pkl, "rb") as f:
            sd = pickle.load(f)
        to_j = lambda d: jax.tree_util.tree_map(jnp.asarray, d)
        self.actor.load_state_dict(to_j(sd["actor"]))
        self.critic.load_state_dict(to_j(sd["critic"]))
        print(f"[dppo] resumed actor+critic from {pkl}")


# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DPPO fine-tuning of the diffusion policy.")
    p.add_argument("--ckpt", type=str, default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--tfrecord", type=str,
                   default="/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-00000-of-01000")
    p.add_argument("--indices", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument("--out_dir", type=str, default="/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/dppo")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--resume", type=str, default="",
                   help="restore actor+critic from a save() checkpoint dir (or its "
                        "actor_value.pkl) and continue training -- e.g. to push a run "
                        "that was still improving past its iter budget")
    p.add_argument("--iter_offset", type=int, default=0,
                   help="add to the logged/plotted iter (not loop control) so a "
                        "--resume run continues the curve instead of restarting at 0")
    p.add_argument("--seed", type=int, default=0)
    # DPPO / diffusion
    p.add_argument("--collect_steps", type=int, default=0,
                   help="0 = full diffusion reverse chain (timesteps). A value <timesteps "
                        "DDIM-subsamples the frozen prefix to speed up collection; the "
                        "k_trainable late steps are unchanged. A/B win-rate before adopting.")
    p.add_argument("--ab_steps", type=str, default="",
                   help="Comma list of num_sample_steps for the DDIM A/B bench "
                        "(0 = full chain), e.g. '0,40,30,25,20'. Unused by training.")
    p.add_argument("--k_trainable", type=int, default=10)
    p.add_argument("--prefix_len", type=int, default=5)
    p.add_argument("--replan_interval_steps", type=int, default=10)
    p.add_argument("--max_replans", type=int, default=0, help="0 = run to horizon")
    # Sampling and log-prob MUST use the same floor -- the actor enforces it. A
    # smaller sample floor made every recorded action off-policy w.r.t. the
    # density PPO optimizes; keeping them equal is a correctness requirement, not
    # a tuning knob.
    p.add_argument("--sigma_sample_floor", type=float, default=0.1)
    p.add_argument("--sigma_logprob_floor", type=float, default=0.1)
    p.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True,
                   help="init the frozen base + trainable copy from EMA weights -- "
                        "the actual evaluated baseline (use --no-use_ema for raw)")
    p.add_argument("--eval_every", type=int, default=5,
                   help="fixed-seed stochastic+deterministic eval cadence (0 = off)")
    p.add_argument("--profile", action=argparse.BooleanOptionalAction, default=False,
                   help="per-phase timing (collect sample vs env, and update) via "
                        "block_until_ready barriers. Adds sync overhead, so off in real "
                        "runs; use to see where an iteration's wall-clock goes")
    p.add_argument("--clear_caches", action=argparse.BooleanOptionalAction, default=False,
                   help="call jax.clear_caches() each iter -- flushes JAX/flax caches to "
                        "test/stopgap the ~0.6 GiB/iter host-RAM leak (probe 32647/32673). "
                        "Also drops the good jitted executables, so it re-compiles each iter")
    p.add_argument("--eval_seeds", type=int, default=1,
                   help="average each eval over this many fixed seeds. Single-seed eval "
                        "on a few scenes swings +-3-4 between points and hides the trend; "
                        ">1 reports the mean and the across-seed std (the real noise band)")
    # Reward shaping (V-Max parity). Defaults reproduce the original bare
    # RewardConfig(), so a launch that sets none of these is byte-identical to the
    # runs before this change. The parity recipe (route arclength progress +
    # terminate-on-offroad + bounded magnitudes + goal shaping) is turned on
    # explicitly; see submit_train_dppo.sh. The diagnosis showed the old default
    # (unbounded goal-distance progress + offroad charged once & non-terminal) is
    # farmable: cutting the corner off-road strictly pays.
    p.add_argument("--route_reward", action=argparse.BooleanOptionalAction, default=False,
                   help="dense progress = on-route arclength gain (V-Max `progression`) "
                        "instead of straight-line goal-distance closed. A beeline off "
                        "the road earns no arclength, so the corner-cut stops paying. "
                        "Needs sdc_paths (auto-loaded when on; set --num_paths).")
    p.add_argument("--terminate_on_offroad", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="end the episode on offroad (V-Max termination_keys). The real "
                        "cost of leaving the road becomes the forfeited remaining "
                        "progress (~16x the -1 penalty), so magnitudes stay small.")
    p.add_argument("--offroad_per_step", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="charge --reward_offroad on EVERY off-road world step instead "
                        "of once per episode. The alternative to --terminate_on_offroad "
                        "for making offroad unprofitable: a noisy rollout keeps its full "
                        "8 replans of signal (terminating collapsed collection to a "
                        "one-step bandit at every usable sigma), while a long excursion "
                        "still costs more than a clipped corner. Cost scales with dwell, "
                        "so use ~-0.2, not -1.0. Pair with --no-terminate_on_offroad.")
    p.add_argument("--terminate_on_collision", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="end the episode on collision (the historical default). "
                        "Turn OFF together with --collision_per_step: while offroad "
                        "is dwell-charged and collision is a cheap one-time terminal "
                        "cost, ending the episode is an ESCAPE from the remaining "
                        "off-road bill (measured job 36466: corr(return, n_alive) "
                        "= -0.80, and the policy learned to crash early).")
    p.add_argument("--collision_per_step", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="charge --reward_collision on EVERY colliding world step "
                        "instead of once per episode -- the collision twin of "
                        "--offroad_per_step. With both per-step flags on and both "
                        "terminations off, no absorbing state is reachable except the "
                        "goal, so the policy cannot shorten its way out of a penalty "
                        "and every episode yields its full replan budget of signal. "
                        "Cost scales with dwell, so use ~-0.2, not -1.0.")
    p.add_argument("--reward_progress", type=float, default=1.0,
                   help="progress weight. 1.0 = per-metre straight-line (old default); "
                        "with --route_reward use 0.2 (V-Max per-step indicator).")
    p.add_argument("--reward_offroad", type=float, default=-5.0,
                   help="offroad penalty; terminal with --terminate_on_offroad. "
                        "Old default -5.0; V-Max parity -1.0.")
    p.add_argument("--reward_collision", type=float, default=-10.0,
                   help="collision penalty; terminal unless --no-terminate_on_collision. "
                        "Old default -10.0; parity -1.0; -0.2 with "
                        "--collision_per_step (the cost then scales with dwell).")
    p.add_argument("--reward_goal_bonus", type=float, default=10.0,
                   help="one-time goal bonus. Old default 10.0; parity 5.0 (goal "
                        "shaping carries the credit assignment).")
    p.add_argument("--reward_off_route", type=float, default=-0.2,
                   help="bounded off-route penalty per step (fires when off-route>0). "
                        "Only used with --route_reward. V-Max parity -0.2.")
    p.add_argument("--goal_shaping_coef", type=float, default=0.0,
                   help="potential-based goal shaping F=coef*(gamma*Phi(s')-Phi(s)), "
                        "Phi=-dist/dist0 (V-Max BraxWrapper). Policy-invariant credit "
                        "accelerator applied per replan step at --gamma. 0=off; parity 1.0.")
    p.add_argument("--goal_shaping_normalize", action=argparse.BooleanOptionalAction,
                   default=True, help="divide Phi by the initial goal distance (scale-free).")
    p.add_argument("--num_paths", type=int, default=10,
                   help="sdc_paths dim for --route_reward. ScenarioMax (failures."
                        "tfrecord) = 10; raw WOMD tf_example = 45.")
    p.add_argument("--num_points_per_path", type=int, default=300,
                   help="sdc_paths dim for --route_reward. ScenarioMax = 300; WOMD = 800.")
    # PPO
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=1.0,
                   help="1.0 = Monte-Carlo advantage (critic is a baseline only, not a "
                        "bootstrap); episodes here are ~8 replan steps, so MC is cheap")
    p.add_argument("--ppo_clip", type=float, default=0.01,
                   help="clip acts on the *averaged* log-density ratio; a per-"
                        "transition ratio compounds this over channels*prefix, so "
                        "start tight (~0.01) and watch kl_tx")
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--warmup_iters", type=int, default=0,
                   help="critic-only iterations before the actor is unfrozen: collect "
                        "rollouts and fit the value head with PG off, so advantages are "
                        "trustworthy before the first policy step (an unfit critic + "
                        "advstd~50 walked the pretrained policy downhill in the k=1e9 run)")
    p.add_argument("--target_kl", type=float, default=0.05,
                   help="early-stop on the exact Gaussian *trajectory* KL(old||candidate) "
                        "in nats (kl_g), measured on the candidate before it commits. This "
                        "is the true per-iteration policy move, not the averaged approx_kl; "
                        "0.05 is tight -- sweep alongside sigma once rollouts look sane")
    p.add_argument("--actor_lr", type=float, default=3e-5)
    p.add_argument("--value_lr", type=float, default=1e-3)
    p.add_argument("--value_epochs", type=int, default=10,
                   help="critic gradient steps per iteration (1 cannot track the GAE target)")
    p.add_argument("--pg_scale", type=float, default=0.0,
                   help="0 = auto (channels*prefix_len), undoing the log-density averaging")
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--expert_weight", type=float, default=0.0,
                   help="in-loss expert anchor: adds weight*(expert diffusion loss) to "
                        "the SAME gradient as the PG term. Balancing two incommensurable "
                        "loss scales by a coefficient is what went wrong at 0.1 (the "
                        "anchor was ~1000x the PG term); prefer --expert_steps. "
                        "0 = off; if used, ~0.001-0.01.")
    p.add_argument("--expert_steps", type=int, default=0,
                   help="expert-only gradient steps per iteration, run AFTER the PPO "
                        "epochs. With --ppo_epochs this is the expert:DPPO STEP RATIO -- "
                        "a count rather than a coefficient, so it needs no rescaling when "
                        "the PG magnitude moves, and it has its own Adam state so the two "
                        "objectives do not share moment estimates. 0 = pure DPPO.")
    p.add_argument("--expert_lr", type=float, default=0.0,
                   help="0 = reuse --actor_lr for the expert steps. Set only to decouple "
                        "the expert step SIZE from its step COUNT; leaving it equal keeps "
                        "--expert_steps the single variable of the ratio sweep.")
    # ---- full-corpus scene sampling ---------------------------------------- #
    p.add_argument("--shard_glob", type=str, default="",
                   help="train on the WHOLE corpus: glob over tfrecord shards, from "
                        "which a fresh batch is drawn every --resample_every iterations "
                        "(O(1) random access via the byte-offset indexes). Empty = the "
                        "historical fixed --indices batch on --tfrecord.")
    p.add_argument("--index_dir", type=str, default=DEFAULT_INDEX_DIR,
                   help="where the shards' .offsets.npy indexes live (empty = the "
                        "sidecar-next-to-the-shard convention). Build with "
                        "rl/diffusion_ft/slurm/submit_build_womd_offsets.sh")
    p.add_argument("--scenes_per_iter", type=int, default=32,
                   help="training batch size when --shard_glob is set (B=64 OOM'd in the "
                        "actor grad; 32 is 4x the data of 8 for +11% time)")
    p.add_argument("--resample_every", type=int, default=1,
                   help="draw a new training batch every N iterations. 1 = every "
                        "iteration (maximum coverage); >1 amortizes the load and lets "
                        "consecutive PPO iterations reuse a batch")
    p.add_argument("--eval_scenes", type=int, default=32,
                   help="size of the FIXED held-out eval set (drawn once from the same "
                        "corpus and excluded from every training draw), so the eval curve "
                        "is comparable across iterations and measures generalization. "
                        "Only used with --shard_glob")
    p.add_argument("--data_seed", type=int, default=0,
                   help="seed for the scene draw. Change it to re-randomize the data "
                        "stream while holding the training seed fixed")
    p.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=True,
                   help="load the next scene batch on a background thread while the "
                        "current one trains (parsing is TF/CPU, training is GPU, so this "
                        "hides essentially all of the load cost)")
    # Weights & Biases (off by default; the JSONL log is always written regardless)
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False,
                   help="mirror every per-iter row to Weights & Biases (same keys as the JSONL)")
    p.add_argument("--wandb_project", type=str, default="diffusion_ft_dppo")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None,
                   help="run name; defaults to a sigma/kl/warmup tag")
    p.add_argument("--wandb_group", type=str, default=None,
                   help="group name so a sweep's arms plot together")
    return p


def main():
    args = build_argparser().parse_args()
    if args.max_replans <= 0:
        args.max_replans = None
    DPPOTrainer(args).train()


if __name__ == "__main__":
    main()
