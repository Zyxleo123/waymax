"""Training-free proposal selection: the base stays the default, a second
candidate has to EARN each intervention.

WHY. Every way of deploying the safety fine-tune measured so far loses or breaks
even at the corpus level: ungated -0.011, hardness-gated +0.0005 of a +0.0039
ceiling. The gate failed for a structural reason -- scene difficulty predicts
where the fine-tune helps AND where it hurts, because both concentrate in the
same high-risk scenes. A hard scene can still contain an excellent base
proposal. So the decision should be made per REPLAN on the two specific
proposals on the table, not per scene on how hard the scene looks.

WHAT ONE REPLAN DOES (selection arms). The base draws its candidate as usual;
the frozen prefix output ``x_{k-1}`` and its late-chain key are kept
(:meth:`DiffusionRLActor.sample_world_with_prefix`). The alternative is then:

* ``ft``       -- the fine-tuned late chain from the SAME prefix and the SAME
  late noise (:meth:`sample_late_for`). The two candidates differ only by the
  weights of the last ``k_trainable`` steps, so a switch is attributable to the
  fine-tune and not to sampler luck.
* ``base2``    -- the BASE late chain from the same prefix with a fresh late key.
  Identical compute to ``ft`` (k extra denoiser evals). This is the matched
  control: "a second base sample, then choose" vs "a repaired sample, then
  choose".
* ``basefull`` -- an independent full base sample (2x compute): the extra-sample
  control with its own coarse manoeuvre.

Both candidates are scored on geometry only -- min road-edge clearance and min
agent gap over the next ``window`` world steps (the same exact clearances whose
signs ARE waymax's offroad/overlap), plus progress toward the goal -- and a
selector decides per scene. ``cons`` switches only when the base candidate is
predicted to violate, the alternative is predicted clean, and it does not give
up more than ``tol`` metres of progress; everywhere else the base proposal is
executed unchanged.

WHAT IT MEASURES. Per (scene, arm, repeat) the closed-loop outcome, all arms on
the SAME sampler seeds, so every scene's arms are paired draw for draw. The
``base`` and ``ft`` arms are also run whole (the base-only and fine-tune-only
policies), so the analysis gets, from one set of rows:

* paired labels vs the base: +1 recovered / -1 newly broken / 0 unchanged;
* the selector's corpus-level delta on all three strata (failure, success and
  the intermittent scenes the older marginal excluded);
* the "any candidate succeeds" upper bound -- and so how much of the gap is
  selection and how much is proposal quality.

Two further selectors: ``look`` executes each candidate plus ``--look_replans - 1``
base replans from a snapshot and switches only if that removes a violation
(:mod:`rl.diffusion_ft.branching`); ``learned`` applies a selector fit to
per-decision counterfactual labels (``tools/fit_selector.py``).

Resumable: a task appends rows per finished chunk and skips scenes whose every
(arm, repeat) row exists. Submitted via ``slurm/submit_select_eval.sh``.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

ALTS = ("ft", "base2", "basefull")
SELECTORS = ("cons", "margin", "always", "learned", "look")
# Per-replan feature columns stored for every arm (alt columns NaN when the arm
# has no alternative). Kept as data so a learned selector can be fit on CPU
# later without re-rolling anything.
FEAT_COLS = ("clear_p", "edge_p", "agent_p", "prog_p",
             "clear_a", "edge_a", "agent_a", "prog_a")
# A finite stand-in for "no valid neighbour in the window" (c_agent is +inf
# there): large enough to never trigger a threshold, small enough for JSON/npz.
NO_AGENT_M = 1.0e3


@dataclass(frozen=True)
class Arm:
    name: str
    primary: str            # "base" | "ft": whose candidate is the default
    alt: str | None = None  # ALTS, or None for a single-policy arm
    selector: str | None = None
    window: int = 20        # world steps scored by the selector
    # `learned` only: switch threshold overriding the fitted model's own. The
    # fitted theta maximizes a PER-DECISION objective that cannot price
    # over-switching (see tools/fit_selector.py), so theta is swept HERE, where
    # the objective is the episode outcome the deployment cares about.
    theta: float | None = None

    @property
    def needs_ft(self) -> bool:
        return self.primary == "ft" or self.alt == "ft"


def parse_arms(spec: str, default_window: int = 20) -> list[Arm]:
    """``base,ft,bf_cons=ft:cons:20,bb_cons=base2:cons:20`` -> arms.

    ``base`` and ``ft`` are the single-policy arms. Anything else is
    ``name=alt:selector[:window[:theta]]`` with the base as primary; ``theta``
    overrides a ``learned`` selector's fitted switch threshold.
    """
    arms: list[Arm] = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if tok in ("base", "ft"):
            arms.append(Arm(tok, tok))
            continue
        if "=" not in tok:
            raise SystemExit(f"--arms: '{tok}' is not base/ft or name=alt:selector[:window]")
        name, rhs = tok.split("=", 1)
        parts = rhs.split(":")
        if len(parts) not in (2, 3, 4):
            raise SystemExit(f"--arms: '{tok}' needs alt:selector[:window[:theta]]")
        alt, sel = parts[0], parts[1]
        win = int(parts[2]) if len(parts) >= 3 else int(default_window)
        theta = float(parts[3]) if len(parts) == 4 else None
        if theta is not None and sel != "learned":
            raise SystemExit(f"--arms: '{tok}' sets a theta on selector '{sel}'; "
                             f"only 'learned' has one")
        if alt not in ALTS:
            raise SystemExit(f"--arms: alt '{alt}' not in {ALTS}")
        if sel not in SELECTORS:
            raise SystemExit(f"--arms: selector '{sel}' not in {SELECTORS}")
        if win < 1:
            raise SystemExit(f"--arms: window must be >= 1 world step, got {win}")
        arms.append(Arm(name, "base", alt, sel, win, theta))
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise SystemExit(f"--arms: duplicate arm names {names}")
    return arms


def select_switch(sel: str, *, clear_p, clear_a, prog_p, prog_a,
                  tol_m: float, margin_m: float, delta_m: float):
    """[B] bool: execute the alternative instead of the primary.

    Pure numpy (no jax) so the rule is unit-testable on CPU. ``clear_*`` is the
    min over the window of min(road-edge clearance, agent gap), metres,
    negative == predicted violation. ``prog_*`` is metres of goal-distance
    reduction over the window.

    * ``cons``   -- primary predicted to violate, alternative predicted clean,
      alternative keeps progress within ``tol_m``. The conservative default:
      a clean base candidate is never touched, however tight.
    * ``margin`` -- primary inside ``margin_m`` of a hazard and the alternative
      at least ``delta_m`` clearer (and itself clean), progress within tol.
    * ``always`` -- the alternative everywhere. With alt=ft this is exactly the
      fine-tune-only policy; kept as a plumbing check.

    ``learned`` (``tools/fit_selector.py``) and ``look`` (execute each candidate
    plus a few base replans, then judge) need model / env state and are decided
    in :meth:`ProposalSelectEval.rollout`, not here.
    """
    clear_p, clear_a = np.asarray(clear_p), np.asarray(clear_a)
    prog_ok = np.asarray(prog_a) >= np.asarray(prog_p) - float(tol_m)
    if sel == "cons":
        return (clear_p < 0.0) & (clear_a >= 0.0) & prog_ok
    if sel == "margin":
        return ((clear_p < float(margin_m)) & (clear_a >= 0.0)
                & (clear_a >= clear_p + float(delta_m)) & prog_ok)
    if sel == "always":
        return np.ones(clear_p.shape, dtype=bool)
    raise ValueError(f"unknown selector {sel!r}")


def make_scorer(objcfg):
    """Jitted ``score(world, state, t0, goal_xy, n) -> (clear, edge, agent, prog)``, [B] each.

    ``clear`` = min over the next ``n`` world steps of min(road-edge clearance,
    agent box gap), metres, negative = predicted violation (the same exact
    clearances whose signs are waymax's offroad / overlap). ``prog`` = metres of
    goal-distance reduction over the window. Shared by the selector arms and the
    counterfactual label job so a learned selector sees identical features.
    ``n`` is static; it changes only near the scenario end."""
    import jax
    import jax.numpy as jnp

    from rl.diffusion_ft.safety_objective import candidate_safety

    def score(world, state, t0, goal_xy, n):
        terms = candidate_safety(objcfg, world, state=state, t0=t0, n_steps=n)
        edge = jnp.min(terms["c_edge"], axis=1)
        agent = jnp.minimum(jnp.min(terms["c_agent"], axis=1), NO_AGENT_M)
        p0, pn = world[:, 0, :2], world[:, n, :2]
        prog = (jnp.linalg.norm(p0 - goal_xy, axis=-1)
                - jnp.linalg.norm(pn - goal_xy, axis=-1))
        return jnp.minimum(edge, agent), edge, agent, prog

    return jax.jit(score, static_argnums=(4,))


# --------------------------------------------------------------------------- #
class ProposalSelectEval:
    """Runs every arm closed-loop on the same scenes and seeds."""

    def __init__(self, tuner, *, ft_ckpt: str, arms: list[Arm], tol_m: float,
                 margin_m: float, delta_m: float, selector_json: str = "",
                 look_replans: int = 2):
        import jax
        import jax.numpy as jnp

        self.jax, self.jnp = jax, jnp
        self.t = tuner
        self.arms = arms
        self.tol_m, self.margin_m, self.delta_m = float(tol_m), float(margin_m), float(delta_m)
        actor = tuner.actor
        actor.reset_trainable_to_frozen()
        self.base_params = actor._tparams
        self.ft_params = None
        self.ft_ckpt = str(ft_ckpt or "")
        if any(a.needs_ft for a in arms):
            if not ft_ckpt:
                raise SystemExit("an arm uses the fine-tune but --ft_ckpt is empty")
            tuner.load(ft_ckpt)
            self.ft_params = actor._tparams
            actor._tparams = self.base_params
        self._score = make_scorer(tuner.objcfg)
        self.selector_model = None
        if any(a.selector == "learned" for a in arms):
            if not selector_json:
                raise SystemExit("a 'learned' arm needs --selector_json (tools/fit_selector.py)")
            self.selector_model = json.loads(Path(selector_json).read_text())
        self.look_replans = int(look_replans)
        self.br = None
        if any(a.selector == "look" for a in arms):
            from rl.diffusion_ft.branching import Brancher
            self.br = Brancher(tuner, self.base_params)

    def _lookahead(self, features, alive, clean, r, rk, world_p, world_a, trig):
        """Execute each candidate at replan ``r``, then ``look_replans - 1`` base
        replans, from a snapshot of NOW; restore; switch rows where the base
        candidate's branch violates, the alternative's does not, and the
        alternative ends no farther than ``tol`` from the goal. Both branches share
        their look-ahead sampler keys (common random numbers), so the comparison is
        the candidates' and not the noise's."""
        from rl.diffusion_ft.branching import Snapshot  # noqa: F401  (type only)
        jax = self.jax
        B = len(trig)
        carry = {"alive": np.asarray(alive, np.float32), "offroad": np.zeros(B, bool),
                 "overlap": np.zeros(B, bool), "reached": np.zeros(B, bool),
                 "clean": np.asarray(clean) > 0.5, "fail_replan": np.full(B, -1, np.int32)}
        snap = self.br.capture(features, carry, r)
        keys = [jax.random.fold_in(rk, 1000 + i) for i in range(int(self.t.args.max_replans))]
        eps = {}
        for tag, w in (("p", world_p), ("a", world_a)):
            eps[tag] = self.br.run(keys=keys, snapshot=snap,
                                   override=lambda rr, ctx, w=w: w if rr == r else None,
                                   stop_after=r + self.look_replans)
        self.br.restore(snap)
        vp, va = eps["p"].fail_replan >= 0, eps["a"].fail_replan >= 0
        closer = eps["a"].final_goal_dist <= eps["p"].final_goal_dist + self.tol_m
        return trig & vp & ~va & closer

    # ------------------------------------------------------------------ #
    def _world(self, x_ncT, aux):
        return self.t.env.norm_to_world_with(self.jnp.transpose(x_ncT, (0, 2, 1)), aux)

    def rollout(self, state, arm: Arm, rng):
        """One closed-loop episode for ``arm``. Mirrors ``SafetyFineTuner.rollout
        (train=False)`` step for step -- same env reset seed, same expert-clean
        gate, same key split per replan, same solved definition -- so the ``base``
        and ``ft`` arms are the ``--verify_only`` policies. They match those rows in
        DISTRIBUTION, not scene by scene: the sampler draws one noise tensor per
        batch, so a scene's draws depend on which chunk it lands in, and sharding
        changes the chunks. Pairing holds within this runner, across its arms."""
        jax, jnp, t = self.jax, self.jnp, self.t
        env, actor, args = t.env, t.actor, t.args
        actor.num_sample_steps = t._steps_eval
        primary = self.base_params if arm.primary == "base" else self.ft_params

        features = env.reset(state, seed=args.seed)
        clean_b = t._gate_expert_clean(env.sim_state)
        alive = clean_b
        B = env.batch_size
        offroad = np.zeros(B); overlap = np.zeros(B); reached = np.zeros(B)
        feats, switches, alives = [], [], []
        key = rng
        for r in range(args.max_replans):
            if float(jnp.sum(alive)) <= 0.0:
                break
            cond1, cond2 = env.compute_condition(features)
            mask = features["inst_valid"]
            t0 = env.timestep
            horizon = int(jnp.asarray(env.sim_state.log_trajectory.x).shape[-1])
            n_exec = min(args.replan_interval_steps, horizon - 1 - t0)
            if n_exec <= 0:
                break
            aux = env.last_aux
            key, rk = jax.random.split(key)
            x_p, (x_prefix, key_late) = actor.sample_world_with_prefix(
                primary, cond1, cond2, mask, rk)
            world_p = self._world(x_p, aux)
            n = int(min(arm.window, horizon - 1 - t0, world_p.shape[1] - 1))
            goal = env._goal_xy_b2
            cp, ep, ap, pp = (np.asarray(v) for v in
                              self._score(world_p, env.sim_state, jnp.asarray(t0, jnp.int32), goal, n))
            row = np.full((B, len(FEAT_COLS)), np.nan, np.float32)
            row[:, 0], row[:, 1], row[:, 2], row[:, 3] = cp, ep, ap, pp
            sw = np.zeros(B, dtype=bool)
            world = world_p
            if arm.alt is not None:
                if arm.alt == "ft":
                    x_a = actor.sample_late_for(self.ft_params, cond1, cond2, mask,
                                                x_prefix, key_late)
                elif arm.alt == "base2":
                    x_a = actor.sample_late_for(self.base_params, cond1, cond2, mask,
                                                x_prefix, jax.random.fold_in(key_late, 1))
                else:  # basefull
                    x_a, _ = actor.sample_world_with_prefix(
                        self.base_params, cond1, cond2, mask, jax.random.fold_in(rk, 1))
                world_a = self._world(x_a, aux)
                ca, ea, aa, pa = (np.asarray(v) for v in
                                  self._score(world_a, env.sim_state, jnp.asarray(t0, jnp.int32), goal, n))
                row[:, 4], row[:, 5], row[:, 6], row[:, 7] = ca, ea, aa, pa
                if arm.selector == "learned":
                    from rl.diffusion_ft.tools.fit_selector import learned_switch
                    sw = learned_switch(self.selector_model, row, r, theta=arm.theta)
                elif arm.selector == "look":
                    # Only pay for a look-ahead where either candidate is near a hazard.
                    trig = (np.asarray(alive) > 0.5) & (np.minimum(cp, ca) < self.margin_m)
                    sw = (self._lookahead(features, alive, clean_b, r, rk, world_p, world_a, trig)
                          if trig.any() else np.zeros(B, dtype=bool))
                else:
                    sw = select_switch(arm.selector, clear_p=cp, clear_a=ca, prog_p=pp,
                                       prog_a=pa, tol_m=self.tol_m, margin_m=self.margin_m,
                                       delta_m=self.delta_m)
                world = jnp.where(jnp.asarray(sw)[:, None, None], world_a, world_p)
            alive_np = np.asarray(alive) > 0.5
            feats.append(row); switches.append(sw & alive_np); alives.append(alive_np)

            out = env.step(trajectory_world_bt5=world)
            comp = out.components
            offroad = np.maximum(offroad, np.asarray(comp["offroad_flag"], np.float32) > 0)
            overlap = np.maximum(overlap, np.asarray(comp["collision_flag"], np.float32) > 0)
            reached = np.maximum(reached, np.asarray(comp["reached_flag"], np.float32) > 0)
            done = jnp.asarray(out.terminated_b) | jnp.asarray(out.truncated_b)
            alive = alive * (1.0 - done.astype(jnp.float32))
            features = out.features

        R = len(feats)
        F = (np.stack(feats, 1) if R else np.zeros((B, 0, len(FEAT_COLS)), np.float32))
        S = np.stack(switches, 1) if R else np.zeros((B, 0), bool)
        A = np.stack(alives, 1) if R else np.zeros((B, 0), bool)
        solved = (reached > 0.5) & (offroad < 0.5) & (overlap < 0.5)
        return {"solved": solved, "offroad": offroad > 0.5, "overlap": overlap > 0.5,
                "reached": reached > 0.5, "clean": np.asarray(clean_b) > 0.5,
                "feat": F, "switch": S, "alive": A}

    # ------------------------------------------------------------------ #
    def run(self, out_dir: Path, *, names, repeats: int, seed: int, chunk: int,
            shard_index: int, num_shards: int, cap: int = -1):
        jax = self.jax
        t = self.t
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "feat").mkdir(exist_ok=True)
        rows_path = out_dir / "rows.jsonl"
        want = {(a.name, r) for a in self.arms for r in range(repeats)}
        have: dict[int, set] = {}
        if rows_path.exists():
            for line in rows_path.read_text().splitlines():
                try:
                    r = json.loads(line)
                except Exception:
                    continue            # torn last line from a killed task
                have.setdefault(int(r["scene_id"]), set()).add((r["arm"], int(r["repeat"])))
        spec = t._read_splits_spec(t.args.splits_json)
        key = jax.random.PRNGKey(int(seed))
        meta = {"arms": [a.__dict__ for a in self.arms], "repeats": repeats, "seed": seed,
                "chunk": int(chunk),
                "ft_ckpt": self.ft_ckpt, "tol_m": self.tol_m, "margin_m": self.margin_m,
                "delta_m": self.delta_m, "feat_cols": list(FEAT_COLS),
                "look_replans": self.look_replans,
                "selector_theta": (self.selector_model or {}).get("theta"),
                "shard": [shard_index, num_shards], "cap": int(cap), "splits": list(names),
                "splits_json": str(t.args.splits_json)}
        (out_dir / f"meta_shard{shard_index}.json").write_text(json.dumps(meta, indent=1))
        t_start = time.time()
        for name in names:
            ids = t._split_ids(spec, name)
            ids = (ids if cap < 0 else ids[:cap])[shard_index::num_shards]
            todo = [i for i in ids if not want <= have.get(i, set())]
            print(f"[select] {name}: {len(ids)} scenes in shard {shard_index}/{num_shards}, "
                  f"{len(todo)} to do", flush=True)
            # Chunks are FIXED slices of the shard's id list, never of `todo`. The
            # sampler draws one noise tensor for the whole batch, so a scene's
            # draws depend on its batch and position: re-chunking the remainder
            # after a resume would silently change them and break the pairing
            # with every other task scored on this shard (e.g. the base arm run
            # once and paired against three fine-tuned checkpoints).
            for start in range(0, len(ids), chunk):
                sub = ids[start:start + chunk]
                if all(want <= have.get(i, set()) for i in sub):
                    continue
                state = t.sampler.load_explicit(sub)
                recs, npz = [], {"scene_ids": np.asarray(sub, np.int64)}
                tally = {}
                for arm in self.arms:
                    for r in range(repeats):
                        # Same fold for every arm: the arms are paired draw for draw.
                        o = self.rollout(state, arm, jax.random.fold_in(key, r))
                        npz[f"{arm.name}/r{r}/feat"] = o["feat"]
                        npz[f"{arm.name}/r{r}/switch"] = o["switch"]
                        npz[f"{arm.name}/r{r}/alive"] = o["alive"]
                        tally.setdefault(arm.name, []).append(float(o["solved"].mean()))
                        for j, sid in enumerate(sub):
                            recs.append({
                                "scene_id": int(sid), "split": name, "arm": arm.name,
                                "repeat": r, "seed": int(seed),
                                # "" for arms that never touch the fine-tune, so
                                # the analysis can pool them across ckpt tasks.
                                "ckpt": self.ft_ckpt if arm.needs_ft else "",
                                "solved": bool(o["solved"][j]), "offroad": bool(o["offroad"][j]),
                                "overlap": bool(o["overlap"][j]), "reached": bool(o["reached"][j]),
                                "clean": bool(o["clean"][j]),
                                "n_replans": int(o["alive"][j].sum()),
                                "n_switch": int(o["switch"][j].sum())})
                np.savez_compressed(out_dir / "feat" / f"{name}_s{shard_index}_{start:05d}.npz", **npz)
                with rows_path.open("a") as fh:
                    fh.write("".join(json.dumps(x) + "\n" for x in recs))
                done = min(start + chunk, len(ids))
                print(f"[select] {name} {done}/{len(ids)}  " + "  ".join(
                    f"{k}={np.mean(v):.3f}" for k, v in tally.items())
                    + f"  {time.time() - t_start:.0f}s", flush=True)
        print(f"[select] wrote -> {rows_path}", flush=True)


def main():
    from rl.diffusion_ft.train_safety import SafetyFineTuner, build_argparser

    p = build_argparser()
    p.description = __doc__
    p.add_argument("--ft_ckpt", default="", help="fine-tuned weights (a run's iter_* dir)")
    p.add_argument("--arms", default="base,ft,bf_cons=ft:cons:20,bb_cons=base2:cons:20")
    p.add_argument("--select_out", default="", help="default <out_dir>")
    p.add_argument("--select_seed", type=int, default=4242)
    p.add_argument("--select_repeats", type=int, default=4)
    p.add_argument("--select_window", type=int, default=20,
                   help="default selector window (world steps) for arms without one")
    p.add_argument("--select_tol_m", type=float, default=1.0,
                   help="progress the alternative may give up, metres over the window")
    p.add_argument("--select_margin_m", type=float, default=0.3)
    p.add_argument("--select_delta_m", type=float, default=0.2)
    p.add_argument("--selector_json", default="", help="tools/fit_selector.py output, for 'learned' arms")
    p.add_argument("--look_replans", type=int, default=2,
                   help="'look' arms: replans simulated per candidate (1 = just the executed window)")
    p.add_argument("--select_cap", type=int, default=-1,
                   help="first N scenes of each split (before sharding); -1 = all. Smoke only")
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--selection_seeds", default="777",
                   help="seeds that SELECTED the splits; scoring on them is biased")
    args = p.parse_args()
    if not args.splits_json or not args.shard_glob:
        raise SystemExit("--splits_json and --shard_glob are required")
    banned = {int(s) for s in args.selection_seeds.split(",") if s.strip()} | {int(args.eval_seed)}
    if int(args.select_seed) in banned:
        raise SystemExit(f"--select_seed {args.select_seed} reuses a selection/eval seed {sorted(banned)}")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("need 0 <= shard_index < num_shards")
    arms = parse_arms(args.arms, args.select_window)
    names = [n.strip() for n in args.splits.split(",") if n.strip()]
    args.splits = ""              # stream chunks; do not preload eval states
    args.probe_per_split = 0
    tuner = SafetyFineTuner(args)
    runner = ProposalSelectEval(tuner, ft_ckpt=args.ft_ckpt, arms=arms,
                                tol_m=args.select_tol_m, margin_m=args.select_margin_m,
                                delta_m=args.select_delta_m, selector_json=args.selector_json,
                                look_replans=args.look_replans)
    print("[select] arms: " + "  ".join(
        a.name if a.alt is None else f"{a.name}=({a.primary}|{a.alt}:{a.selector}:w{a.window})"
        for a in arms), flush=True)
    runner.run(Path(args.select_out or args.out_dir), names=names,
               repeats=int(args.select_repeats), seed=int(args.select_seed),
               chunk=max(1, int(args.eval_chunk)), shard_index=int(args.shard_index),
               num_shards=int(args.num_shards), cap=int(args.select_cap))


if __name__ == "__main__":
    main()
