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
* ``bank``     -- the repair search's own candidate families applied to the BASE's
  proposal: a time warp along its path (arrive later / earlier) or a smooth
  lateral shift of its line (:mod:`rl.diffusion_ft.branching`). ZERO denoiser
  evaluations -- these are geometric edits of a trajectory already drawn -- so a
  bank of K costs K scorer calls and no network. WHY: ``counterfactual --mode
  repair`` verified that 82 of 85 kept repairs are exactly these two families
  (timing 42, lateral 40, re-noise 3), and that half the fixes are "arrive
  later", which no clearance gradient can express. Training on those repairs did
  not transfer (best 3-seed mean +0.0004, and growing the set 100 -> 160 targets
  moved it to -0.0001), so this offers the same interventions at DECISION time,
  where the geometric test evaluates the condition instead of a gradient having
  to learn it. Among the candidates the rule admits, the SMALLEST perturbation
  wins -- the repair search's own ordering (``Candidate.magnitude``).

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

import dataclasses
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

ALTS = ("ft", "base2", "basefull", "bank", "es")
SELECTORS = ("cons", "margin", "always", "learned", "look", "proj", "argmin", "argmin_c")
# One objective, no decision rules: run the bank's (lon) optimizer from each start
# (`init0`, `initprev`) on every alive row, execute the per-row argmin of the final
# loss whenever it moved. `argmin_c` adds the one rule under test: never replace a
# base predicted clean with a plan predicted to violate.
ARGMIN_SELECTORS = ("argmin", "argmin_c")
# Per-replan feature columns stored for every arm (alt columns NaN when the arm
# has no alternative). Kept as data so a learned selector can be fit on CPU
# later without re-rolling anything.
FEAT_COLS = ("clear_p", "edge_p", "agent_p", "prog_p",
             "clear_a", "edge_a", "agent_a", "prog_a")
# A finite stand-in for "no valid neighbour in the window" (c_agent is +inf
# there): large enough to never trigger a threshold, small enough for JSON/npz.
NO_AGENT_M = 1.0e3
# Magnitude offset per loss-optimized candidate kind, keyed by
# lateral_opt.init_kind -> (edits the previous plan, along-path profile on). The
# smallest-edit rule then runs a later kind only where every earlier kind is
# predicted to violate: fresh lateral (o_0's edit) < fresh lon < prev lateral <
# prev lon. Gaps of 100 m dwarf any edit's own size, so tiers never interleave.
OPT_TIER_M = {(False, False): 0.0, (False, True): 100.0, (True, False): 200.0,
              (True, True): 300.0}
# The emergency variants (lateral_opt.init_variant) rank below every ordinary kind.
EMERG_TIER_M = {"emerg": 400.0, "brake": 500.0}
# `proj` (always-on projection): on a row the base proposal is predicted CLEAN, an
# optimized edit replaces it only if it clears it by at least this much -- so an
# edit that did not move (zero loss: every corner and car beyond the loss margin)
# can never be executed, and a clean plan with nothing inside the margin is
# untouched exactly as under `cons`.
PROJ_MIN_GAIN_M = 0.01


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
    # `bank` alts only: which named bank supplies the candidates. "" = the
    # default bank (--select_bank). Named banks (--select_banks) let a lateral-only
    # and a timing-only arm run in the SAME job, so the ablation is paired draw for
    # draw rather than compared across submissions.
    bank: str = ""
    # `es` alts only: which named --select_es config runs the population search
    # (rl.diffusion_ft.es_population).
    es: str = ""
    # `adapter` primaries only (rl.diffusion_ft.adapter, --adapter_ckpt): the gate
    # mode (learned | open | closed) and the external strength in [0, 1].
    gate: str = ""
    strength: float = 1.0

    @property
    def needs_ft(self) -> bool:
        return self.primary == "ft" or self.alt == "ft"


def parse_arms(spec: str, default_window: int = 20) -> list[Arm]:
    """``base,ft,bf_cons=ft:cons:20,bb_cons=base2:cons:20`` -> arms.

    ``base`` and ``ft`` are the single-policy arms. Anything else is
    ``name=alt:selector[:window[:theta]]`` with the base as primary; ``theta``
    overrides a ``learned`` selector's fitted switch threshold, and for ``proj`` is
    the primary clearance (m) below which the optimizer runs on a clean row.

    The alt ``bank`` uses the default bank; ``bank@<name>`` uses a named one from
    ``--select_banks`` (``@`` rather than ``:``, which is the field separator, and
    the bank specs themselves are comma-separated so they cannot live here).

    ``name=adapter:<gate>[:<strength>]`` is a single-policy arm: the base sampler
    with the gated tail adapter of ``--adapter_ckpt`` in its last K steps, gate
    ``learned`` | ``open`` | ``closed``, strength in [0, 1] (default 1).
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
        if parts[0] == "adapter":
            from rl.diffusion_ft.adapter import GATE_MODES
            if len(parts) not in (2, 3) or parts[1] not in GATE_MODES:
                raise SystemExit(f"--arms: '{tok}' needs adapter:<{'|'.join(GATE_MODES)}>[:strength]")
            strength = float(parts[2]) if len(parts) == 3 else 1.0
            if not 0.0 <= strength <= 1.0:
                raise SystemExit(f"--arms: '{tok}' strength must be in [0, 1]")
            arms.append(Arm(name, "adapter", gate=parts[1], strength=strength))
            continue
        if len(parts) not in (2, 3, 4):
            raise SystemExit(f"--arms: '{tok}' needs alt:selector[:window[:theta]]")
        alt, sel = parts[0], parts[1]
        bank = es = ""
        if alt.startswith("bank@"):
            alt, bank = "bank", alt[len("bank@"):]
            if not bank:
                raise SystemExit(f"--arms: '{tok}' has an empty bank name after 'bank@'")
        elif alt.startswith("es@"):
            alt, es = "es", alt[len("es@"):]
            if not es:
                raise SystemExit(f"--arms: '{tok}' has an empty config name after 'es@'")
        elif alt == "es":
            raise SystemExit(f"--arms: '{tok}' needs a named config, 'es@<name>' (--select_es)")
        win = int(parts[2]) if len(parts) >= 3 else int(default_window)
        theta = float(parts[3]) if len(parts) == 4 else None
        if theta is not None and sel not in ("learned", "proj"):
            raise SystemExit(f"--arms: '{tok}' sets a theta on selector '{sel}'; "
                             f"only 'learned' and 'proj' have one")
        if sel in ("proj",) + ARGMIN_SELECTORS and alt != "bank":
            raise SystemExit(f"--arms: selector '{sel}' needs a loss-optimized bank, got '{tok}'")
        if alt not in ALTS:
            raise SystemExit(f"--arms: alt '{alt}' not in {ALTS}")
        if sel not in SELECTORS:
            raise SystemExit(f"--arms: selector '{sel}' not in {SELECTORS}")
        if win < 1:
            raise SystemExit(f"--arms: window must be >= 1 world step, got {win}")
        # `learned` scores ONE alternative's feature row and `look` branches the
        # env per candidate; neither has a defined meaning over a bank of K, and
        # silently scoring only the chosen one would make the arm a different
        # experiment than its name says.
        if alt == "bank" and sel in ("learned", "look"):
            raise SystemExit(f"--arms: alt 'bank' does not support selector '{sel}' "
                             f"(it ranks K candidates; use cons/margin/always)")
        # The population search runs only where the base proposal is predicted to
        # violate -- cons's trigger -- so no other selector has a defined meaning.
        if alt == "es" and sel != "cons":
            raise SystemExit(f"--arms: alt 'es' supports selector 'cons' only, got '{sel}'")
        arms.append(Arm(name, "base", alt, sel, win, theta, bank, es))
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
    * ``proj``   -- always-on projection (loss-optimized banks only): ``cons``
      where the primary is predicted to violate, and ALSO, where it is predicted
      clean, an alternative that is clean and at least ``PROJ_MIN_GAIN_M``
      clearer. The optimizer runs on clean rows too (``Arm.theta`` caps which:
      primary clearance below theta, default every row), and returns exactly the
      base where its loss is zero, which this rule can never pick.

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
    if sel == "proj":
        return (clear_a >= 0.0) & prog_ok & ((clear_p < 0.0) | (clear_a >= clear_p + PROJ_MIN_GAIN_M))
    raise ValueError(f"unknown selector {sel!r}")


def bank_choice(sel: str, *, clear_p, prog_p, clear_k, prog_k, magnitude,
                tol_m: float, margin_m: float, delta_m: float, prefer_k=None, tol_k=None):
    """[B] int32: index into the bank of the candidate to execute, ``-1`` = keep
    the base.

    Every candidate faces the SAME per-candidate rule as a single alternative
    (:func:`select_switch`), so ``cons``'s do-no-harm property is preserved
    verbatim: if the base proposal is predicted clean, no candidate is eligible
    and nothing is touched, however large the bank. Among the eligible ones the
    winner is the SMALLEST perturbation by ``Candidate.magnitude`` -- the repair
    search's own ordering (identity < time warp / lateral shift by size <
    re-noise), which is what makes a bank a minimal intervention rather than a
    licence to pick the most aggressive edit that happens to score well.

    Ties in magnitude go to the earlier bank entry, so the arm is a deterministic
    function of the candidate spec's order. Pure numpy: unit-testable on CPU.

    ``clear_k`` / ``prog_k`` are ``[K, B]``; ``magnitude`` is ``[K]``, or ``[K, B]``
    when a candidate's size differs per row (a loss-optimized edit,
    :mod:`rl.diffusion_ft.lateral_opt`, whose size is its peak offset).

    ``prefer_k`` ``[K, B]`` bool (optional) ranks eligible candidates in two tiers:
    preferred ones first (smallest among them), and only if none is preferred, the
    smallest eligible one. Eligibility itself is untouched, so coverage and the
    do-no-harm property are exactly those of the plain rule. Used to prefer edits
    that clear a MARGIN (``--select_bank_prefer``), not just zero.

    ``tol_k`` ``[K]`` (optional) overrides ``tol_m`` per candidate: the ``brake``
    emergency edit is allowed to give up the progress it brakes away.
    """
    clear_k = np.asarray(clear_k, np.float64)
    prog_k = np.asarray(prog_k, np.float64)
    if clear_k.ndim != 2 or clear_k.shape != prog_k.shape:
        raise ValueError(f"clear_k/prog_k must both be [K, B], got {clear_k.shape} {prog_k.shape}")
    K, B = clear_k.shape
    mag = np.asarray(magnitude, np.float64)
    if mag.ndim == 1:
        mag = mag[:, None]
    if mag.shape[0] != K or mag.shape[1] not in (1, B):
        raise ValueError(f"magnitude has shape {np.shape(magnitude)} for {K} candidates x {B} rows")
    tol_k = np.full(K, float(tol_m)) if tol_k is None else np.asarray(tol_k, np.float64)
    if tol_k.shape != (K,):
        raise ValueError(f"tol_k must be [K] = ({K},), got {tol_k.shape}")
    elig = np.stack([
        select_switch(sel, clear_p=clear_p, clear_a=clear_k[k], prog_p=prog_p,
                      prog_a=prog_k[k], tol_m=float(tol_k[k]), margin_m=margin_m,
                      delta_m=delta_m)
        for k in range(K)])                                          # [K, B]
    if prefer_k is not None:
        prefer_k = np.asarray(prefer_k, bool)
        if prefer_k.shape != (K, B):
            raise ValueError(f"prefer_k must be [K, B] = {(K, B)}, got {prefer_k.shape}")
        # Any tier offset larger than every finite magnitude works; magnitudes are
        # metres (< ~10), so 1e3 cannot reorder candidates within a tier.
        mag = mag + np.where(prefer_k, 0.0, 1.0e3)
    cost = np.where(elig, mag, np.inf)
    best = np.argmin(cost, axis=0)
    return np.where(np.isfinite(cost[best, np.arange(B)]), best, -1).astype(np.int32)


def episode_pace(sim_xy, log_xy, log_valid, *, t_start: int, t_end: int, goal_t,
                 goal_xy, dt, obs_v, obs_t, obs_alive, moving_mps: float = 0.5):
    """Per-episode pace covariates, ``{name: [B]}``. Pure numpy.

    WHY IN THE ROWS. The no-goal residual is trip length x pace (a constant
    fractional shortfall is free on a short trip and fatal on a long one), so a
    pace intervention has to be read per trip length, and ``reached`` alone is a
    3 m threshold on a continuous miss. Carrying these per row keeps that
    analysis on CPU over ``rows.jsonl`` with no scene reload and no dump.

    * ``trip_m``      -- the human's path length from the start pose to its goal
      step (valid segments only). The covariate.
    * ``goal_d0``     -- straight-line start-to-goal distance.
    * ``goal_d_min``  -- the ego's closest approach to the goal over the executed
      window: ``reached`` is ``goal_d_min <= 3 m``, this is how far off it was.
    * ``pace_moving`` -- ego displacement / human displacement over the executed
      steps where the HUMAN is moving (> ``moving_mps``), from POSITIONS. Same
      definition as ``failure_anatomy``'s ``speed_ratio_moving`` (0.90-0.91 for
      base/ft, 2026-09-19). NaN if the human never moves.
    * ``obs_v_sum`` / ``real_v_sum`` -- summed over replans r >= 1 while alive:
      the ego speed the observation SHOWED the policy (``|ego_state vel|``) and
      the speed its positions actually drove over the step into that anchor.
      Their ratio is the defect: ~1.13 under ``ego_vel_source=commanded``, and
      1.000 by construction under ``realised`` -- so every run proves which
      observation it was conditioned on instead of the flag being trusted.

    ``sim_xy``/``log_xy`` ``[B, T, 2]``, ``log_valid`` ``[B, T]``, ``dt`` /
    ``goal_t`` ``[B]``, ``goal_xy`` ``[B, 2]``; ``obs_v``/``obs_alive`` ``[B, R]``
    with anchor steps ``obs_t`` ``[R]``.
    """
    sim_xy, log_xy = np.asarray(sim_xy, np.float64), np.asarray(log_xy, np.float64)
    log_valid = np.asarray(log_valid, bool)
    dt = np.asarray(dt, np.float64)
    goal_xy = np.asarray(goal_xy, np.float64)
    goal_t = np.asarray(goal_t)
    T = log_xy.shape[1]
    tt = np.arange(1, T)                                   # segment (t-1 -> t) ends at t
    seg_log = np.linalg.norm(np.diff(log_xy, axis=1), axis=-1)            # [B, T-1]
    seg_ego = np.linalg.norm(np.diff(sim_xy, axis=1), axis=-1)
    seg_ok = log_valid[:, 1:] & log_valid[:, :-1]
    trip = ((tt[None] > t_start) & (tt[None] <= goal_t[:, None]) & seg_ok)
    win = (tt > t_start) & (tt <= t_end)
    moving = win[None] & seg_ok & (seg_log / dt[:, None] > float(moving_mps))
    den = (seg_log * moving).sum(1)
    pace = np.where(den > 0, (seg_ego * moving).sum(1) / np.maximum(den, 1e-9), np.nan)
    d_goal = np.linalg.norm(sim_xy[:, t_start:t_end + 1] - goal_xy[:, None], axis=-1)
    obs_v = np.asarray(obs_v, np.float64).reshape(len(dt), -1)
    obs_alive = np.asarray(obs_alive, bool).reshape(len(dt), -1)
    obs_t = np.asarray(obs_t, int)
    real = (np.linalg.norm(sim_xy[:, obs_t] - sim_xy[:, obs_t - 1], axis=-1) / dt[:, None]
            if len(obs_t) else np.zeros_like(obs_v))
    return {
        "trip_m": (seg_log * trip).sum(1),
        "goal_d0": np.linalg.norm(log_xy[:, t_start] - goal_xy, axis=-1),
        "goal_d_min": d_goal.min(1),
        "pace_moving": pace,
        "obs_v_sum": (obs_v * obs_alive).sum(1),
        "real_v_sum": (real * obs_alive).sum(1),
    }


PACE_COLS = ("trip_m", "goal_d0", "goal_d_min", "pace_moving", "obs_v_sum", "real_v_sum")
# The world step the run USED, and the scene's first-pair / median timestamp
# steps -- so a scene can be classed short-dt from any run, whichever method it
# ran under (data.preprocess.compute_world_dt_seconds).
DT_COLS = ("world_dt", "dt_first", "dt_median")


def _finite_or_none(v):
    v = float(v)
    return v if np.isfinite(v) else None


def make_bank(spec: str, allow_empty: bool = False):
    """Parse a bank spec into candidates the selector can apply to a drawn proposal.

    Only ``tw`` and ``lat`` are admitted. ``id`` IS the base proposal (keeping it
    would be a switch that changes nothing), ``h<n>`` holds a perturbation across
    replans -- meaningless here because the selector re-decides at every replan --
    and ``rn`` would cost a denoiser evaluation per candidate, which is the whole
    property this alternative has over ``ft``/``base2``. ``rn`` was also 3 of 85
    kept repairs, so it is not what the search says is worth buying.
    """
    from rl.diffusion_ft.branching import parse_candidates

    cands = parse_candidates(spec)
    for c in cands:
        if c.family not in ("tw", "lat"):
            raise SystemExit(f"--select_bank: candidate '{c.name}' is family "
                             f"'{c.family}'; only tw/lat are supported "
                             f"(id is the base proposal, rn costs a denoiser eval)")
        if c.hold != 1:
            raise SystemExit(f"--select_bank: '{c.name}' sets hold={c.hold}; the "
                             f"selector re-decides every replan, so hold has no meaning")
    if not cands and not allow_empty:
        raise SystemExit("--select_bank is empty but an arm uses alt 'bank'")
    return cands


def _kv_spec(spec: str, cast):
    """Parse ``'name=value;name=value'`` into a dict. Empty -> {}."""
    out = {}
    for tok in str(spec).split(";"):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise SystemExit(f"expected 'name=value', got '{tok}'")
        k, v = tok.split("=", 1)
        out[k.strip()] = cast(v.strip())
    return out


def make_banks(default_spec: str, named_spec: str = "",
               optimized: frozenset = frozenset()) -> dict[str, list]:
    """``{"": [...], "lat": [...], "tw": [...]}`` from the two bank flags.

    ``named_spec`` is ``name=spec;name=spec``, the spec itself comma-separated
    (``lat=lat0.25,lat-0.25;tw=tw0.85,tw1.15``). Each is validated by
    :func:`make_bank`, so the tw/lat-only restrictions apply to every one.
    A bank named in ``optimized`` (it has loss-optimized candidates,
    ``--select_bank_opts``) may have an empty fixed spec (``o=``).
    """
    banks = {"": make_bank(default_spec)} if default_spec else {}
    for tok in [t.strip() for t in named_spec.split(";") if t.strip()]:
        if "=" not in tok:
            raise SystemExit(f"--select_banks: '{tok}' is not name=spec")
        name, spec = tok.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"--select_banks: '{tok}' has an empty name")
        if name in banks:
            raise SystemExit(f"--select_banks: duplicate bank name '{name}'")
        banks[name] = make_bank(spec, allow_empty=name in optimized)
    return banks


def make_scorer(objcfg, neighbor_pred: str = "log", world_dt_method: str = "median"):
    """Jitted ``score(world, state, t0, goal_xy, n) -> (clear, edge, agent, prog)``, [B] each.

    ``clear`` = min over the next ``n`` world steps of min(road-edge clearance,
    agent box gap), metres, negative = predicted violation (the same exact
    clearances whose signs are waymax's offroad / overlap). ``prog`` = metres of
    goal-distance reduction over the window. Shared by the selector arms and the
    counterfactual label job so a learned selector sees identical features.
    ``n`` is static; it changes only near the scenario end.

    Neighbours come from the log, deliberately and with no ``neighbor_state``
    override, because this runs at PLAN time: the selector has to commit before
    the window is driven, so the NPCs' future is not available to it in any
    simulator. What changes with ``--sim_agents`` is the STATUS of the number,
    not its formula. Under log replay ``clear`` is exact -- the neighbours cannot
    deviate from what it read, which is what makes `cons`'s ``broken 0.0000``
    structural. Under ``idm`` it becomes a genuine prediction, and an arm's
    do-no-harm claim has to be re-earned rather than inherited.

    ``neighbor_pred`` is where the other agents' next ``n`` steps come from:
    ``"log"`` (the default above) is the recorded future -- under log replay an
    ORACLE of where every car will be, which no deployed planner has. ``"cv"``
    predicts them by constant velocity from their state at ``t0`` and ``"none"``
    ignores them (road edges only); neither reads past ``t0``. See
    ``safety_objective.predicted_neighbor_window``."""
    import jax
    import jax.numpy as jnp

    from rl.diffusion_ft.safety_objective import candidate_safety

    from data.preprocess import compute_world_dt_seconds

    def score(world, state, t0, goal_xy, n):
        dt_b = (None if neighbor_pred == "log"
                else compute_world_dt_seconds(state, 0.1, world_dt_method))
        terms = candidate_safety(objcfg, world, state=state, t0=t0, n_steps=n,
                                 neighbor_pred=neighbor_pred, neighbor_dt_b=dt_b)
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
                 look_replans: int = 2, bank_spec: str = "", banks_spec: str = "",
                 ramp_steps: int = 10, bank_ramps: dict | None = None,
                 bank_blends: dict | None = None, dump_traj: bool = False,
                 bank_neighbors: dict | None = None, dump_bc: bool = False,
                 bank_opts: dict | None = None, opt_cfg: str = "",
                 es_spec: str = "", bank_opt_cfgs: dict | None = None,
                 bank_prefer: dict | None = None, adapter_ckpt: str = "",
                 dump_pairs: bool = False):
        import jax
        import jax.numpy as jnp

        self.jax, self.jnp = jax, jnp
        self.t = tuner
        self.arms = arms
        # Paired-repair dump (rl/diffusion_ft/train_adapter.py): per alive replan,
        # everything needed to REPLAY the base draw's late chain -- the frozen
        # prefix output, the per-row late noise -- next to the base plan and the
        # plan the arm executed, both in world coordinates, plus the replan's
        # world transform and goal. *_train splits only.
        self.dump_pairs = bool(dump_pairs)
        self.adapter_ckpt = str(adapter_ckpt or "")
        self._adapter_tail = {}
        k_tr = int(tuner.actor.k_trainable)
        from rl.diffusion_ft.adapter import late_noise
        self._late_noise = jax.jit(lambda key, x: late_noise(key, x.shape, k_tr))
        if any(a.primary == "adapter" for a in arms):
            if not self.adapter_ckpt:
                raise SystemExit("an arm uses the tail adapter but --adapter_ckpt is empty")
            from rl.diffusion_ft.adapter import load_adapter, tail_from_prefix
            acfg, amod = load_adapter(self.adapter_ckpt)
            if acfg.k_trainable != k_tr:
                raise SystemExit(f"adapter was trained inside a {acfg.k_trainable}-step late "
                                 f"chain, this run samples {k_tr}")
            self.adapter_cfg = acfg
            diff, floor = tuner.actor.diffusion, float(tuner.actor.sigma_sample_floor)
            for mode in {a.gate for a in arms if a.primary == "adapter"}:
                self._adapter_tail[mode] = jax.jit(
                    lambda c1, c2, m, xp, noise, s, _m=mode: tail_from_prefix(
                        diff, amod, acfg, c1, c2, m, xp, noise, strength=s, gate_mode=_m,
                        sigma_floor=floor))
            print(f"[select] adapter {self.adapter_ckpt}: K={acfg.K} of k={k_tr} late steps, "
                  f"modes {sorted(self._adapter_tail)}", flush=True)
        # Behaviour-cloning dump: per replan, the condition the policy saw and the
        # plan the ARM executed, in the denoiser's normalized layout -- the
        # `train_safety --repair_data` format, so a selector arm (the bank) can be
        # distilled into weights. Targets go through `world_to_norm`, whose
        # uncovered tail (past the scenario end) cannot be inverted, hence `valid`.
        self.dump_bc = bool(dump_bc)
        if self.dump_bc:
            from rl.diffusion_ft.branching import model_step_valid, world_to_norm
            pcfg = tuner.ckpt.preprocess_cfg
            horizon = int(tuner.actor.diffusion.denoise_fn.horizon)
            self._w2n = lambda w, aux: world_to_norm(w, aux, pcfg)
            self._bc_valid = lambda aux: model_step_valid(aux, horizon, float(pcfg.model_dt))
            self._ego_range = float(pcfg.ego_range)
            self._bc_checked = False
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
        # Per-bank neighbour source (see make_scorer): "log" is the recorded
        # future, i.e. an oracle of the other cars under log replay; "cv"/"none"
        # never read past t0. Per BANK so the oracle and its honest counterparts
        # run as paired arms in one job. Everything not named stays "log".
        self.bank_neighbors = {k: str(v) for k, v in (bank_neighbors or {}).items()}
        from rl.diffusion_ft.safety_objective import NEIGHBOR_PREDICTIONS
        for k, v in self.bank_neighbors.items():
            if v not in NEIGHBOR_PREDICTIONS:
                raise SystemExit(f"--select_bank_neighbors {k}={v}: not in {NEIGHBOR_PREDICTIONS}")
        self._scorers = {"log": self._score}
        for v in set(self.bank_neighbors.values()) - {"log"}:
            self._scorers[v] = make_scorer(tuner.objcfg, v, tuner.env.world_dt_method)
        self.selector_model = None
        if any(a.selector == "learned" for a in arms):
            if not selector_json:
                raise SystemExit("a 'learned' arm needs --selector_json (tools/fit_selector.py)")
            self.selector_model = json.loads(Path(selector_json).read_text())
        self.look_replans = int(look_replans)
        # Repair-family bank: jit one closure per candidate so the perturbation is
        # a single fused kernel per replan rather than a pile of eager ops. The
        # candidate's parameter and the ramp are Python floats, hence static.
        self.banks, self._perturb = {}, {}
        self.ramp_steps = int(ramp_steps)
        self.dump_traj = bool(dump_traj)
        # Loss-optimized candidates (rl/diffusion_ft/lateral_opt.py), per bank:
        # {bank: [init tokens]}. They are appended AFTER the bank's fixed
        # candidates, so a bank's fixed indices keep their meaning in `choice`.
        from rl.diffusion_ft.lateral_opt import LateralOptConfig, parse_inits
        self.bank_opts = {k: parse_inits(v) for k, v in (bank_opts or {}).items()}
        self.opt_cfg = LateralOptConfig.parse(opt_cfg)
        # Per-bank overrides compose onto the global config (later keys win), so
        # two optimizer variants run as paired arms in one job.
        self.bank_opt_cfg = {k: LateralOptConfig.parse(f"{opt_cfg},{(bank_opt_cfgs or {}).get(k, '')}")
                             for k in self.bank_opts}
        extra = set(bank_opt_cfgs or {}) - set(self.bank_opts)
        if extra:
            raise SystemExit(f"--select_bank_opt_cfgs names {sorted(extra)} have no "
                             f"--select_bank_opts inits")
        self._optimizers, self._bank_optimizer = {}, {}
        # Per-bank MARGIN preference for the certificate's tie-break, {bank:
        # (edge_m, agent_m)}: among clean candidates, those clearing both margins
        # go first (bank_choice prefer_k). Eligibility is untouched.
        self.bank_prefer = {}
        for k, v in (bank_prefer or {}).items():
            parts = [float(x) for x in str(v).split(",") if x.strip()]
            if len(parts) != 2:
                raise SystemExit(f"--select_bank_prefer {k}={v}: want 'edge_m,agent_m'")
            self.bank_prefer[k] = tuple(parts)
        # A bank an ES config projects with is wanted too, so its ramp / blend /
        # neighbour overrides validate even when no bank arm uses it.
        from rl.diffusion_ft.es_population import parse_es_specs
        want = ({a.bank for a in arms if a.alt == "bank"}
                | {c.bank for k, c in parse_es_specs(es_spec).items()
                   if c.bank and any(a.es == k for a in arms if a.alt == "es")})
        if want:
            from rl.diffusion_ft.branching import lateral_shift, time_warp
            self.banks = make_banks(bank_spec, banks_spec, frozenset(self.bank_opts))
            missing = want - set(self.banks)
            if missing:
                raise SystemExit(f"--arms reference bank(s) {sorted(missing)} that "
                                 f"--select_bank/--select_banks do not define "
                                 f"(have {sorted(self.banks)})")
            self.banks = {k: v for k, v in self.banks.items() if k in want}
            dt = float(tuner.ckpt.preprocess_cfg.world_dt_fallback)
            # Ramp and easing are PER BANK so a comfort A/B runs as paired arms in
            # one job rather than across jobs. They are ride-quality knobs, not
            # cosmetic: the env executes through StateDynamics, which teleports the
            # ego to whatever states it is handed, so nothing downstream rejects an
            # abrupt edit. Peak lateral acceleration goes as |A|/T^2 and jerk as
            # |A|/T^3 with T = ramp_steps * dt, so the RAMP dominates; the quintic
            # easing only pays once T is long enough to resolve it (measured: at
            # ramp 10 it is slightly worse than cubic, at ramp 20+ it halves jerk).
            self.bank_ramps = {k: int(v) for k, v in (bank_ramps or {}).items()}
            self.bank_blends = dict(bank_blends or {})
            for k in list(self.bank_ramps) + list(self.bank_blends) + list(self.bank_neighbors):
                if k not in self.banks:
                    raise SystemExit(f"--select_bank_ramps/_blends/_neighbors name "
                                     f"'{k}' is not a bank (have {sorted(self.banks)})")
            for name, cands in self.banks.items():
                rs = int(self.bank_ramps.get(name, ramp_steps))
                bl = str(self.bank_blends.get(name, "cubic"))
                if bl not in ("cubic", "quintic"):
                    raise SystemExit(f"bank '{name}': blend '{bl}' is not cubic|quintic")
                self._perturb[name] = [
                    jax.jit(lambda w, _f=(time_warp if c.family == "tw" else lateral_shift),
                            _a=float(c.param), _rs=rs,
                            _kw=({} if c.family == "tw" else {"blend": bl}):
                            _f(w, _a, ramp_steps=_rs, dt=dt, **_kw))
                    for c in cands]
            for a in arms:
                inits = self.bank_opts.get(a.bank, [])
                is_am = a.selector in ARGMIN_SELECTORS
                if is_am and (self.banks.get(a.bank) or not inits
                              or set(inits) - {"init0", "initprev"} or "init0" not in inits
                              or not self.bank_opt_cfg[a.bank].lon):
                    raise SystemExit(f"arm {a.name}: '{a.selector}' needs a bank of starts "
                                     f"init0[,initprev] only, with lon=1 in its opt cfg")
                if not is_am and a.alt == "bank" and {"init0", "initprev"} & set(inits):
                    raise SystemExit(f"arm {a.name}: init0/initprev are argmin starts; "
                                     f"selector '{a.selector}' cannot use them")
                if a.selector == "proj" and (self.banks.get(a.bank) or not self.bank_opts.get(a.bank)):
                    raise SystemExit(f"arm {a.name}: 'proj' needs a bank of loss-optimized "
                                     f"candidates only (bank '{a.bank}' has fixed ones or none)")
            for name, inits in self.bank_opts.items():
                if name not in self.banks:
                    raise SystemExit(f"--select_bank_opts name '{name}' is not a bank "
                                     f"an arm uses (have {sorted(self.banks)})")
                if "best" in inits and not any(c.family == "lat" for c in self.banks[name]):
                    raise SystemExit(f"--select_bank_opts {name}: init 'best' needs a lat "
                                     f"candidate in the bank to start from")
            from rl.diffusion_ft.lateral_opt import make_lateral_optimizer
            # `_optimizers` stays keyed by neighbour source at the GLOBAL config
            # (es_population reuses it); per-bank variants go in `_bank_optimizer`,
            # sharing a compiled program wherever (source, config) coincide.
            from rl.diffusion_ft.lateral_opt import init_kind
            cache = {}

            def build(nb, cfg):
                if (nb, cfg) not in cache:
                    cache[(nb, cfg)] = make_lateral_optimizer(
                        tuner.objcfg, cfg, dt=dt, neighbor_pred=nb,
                        world_dt_method=tuner.env.world_dt_method)
                return cache[(nb, cfg)]
            from rl.diffusion_ft.lateral_opt import init_variant, variant_cfg
            # One optimizer per (bank, lon, variant) a token needs; compiled programs
            # are shared wherever (neighbour source, config) coincide.
            self._tok_optimizer = {}
            for k in self.bank_opts:
                nb, cfg = self.bank_neighbors.get(k, "log"), self.bank_opt_cfg[k]
                self._bank_optimizer[k] = build(nb, cfg)
                for t in self.bank_opts[k]:
                    key = (bool(cfg.lon or init_kind(t)[1]), init_variant(t))
                    if (k, key) not in self._tok_optimizer:
                        self._tok_optimizer[(k, key)] = build(nb, variant_cfg(cfg, key[1], key[0]))
                if cfg == self.opt_cfg:
                    self._optimizers[nb] = cache[(nb, cfg)]
            for k in self.bank_prefer:
                if k not in self.banks:
                    raise SystemExit(f"--select_bank_prefer name '{k}' is not a bank "
                                     f"an arm uses (have {sorted(self.banks)})")
            self._opt_dt = dt
        self.br = None
        if any(a.selector == "look" for a in arms):
            from rl.diffusion_ft.branching import Brancher
            self.br = Brancher(tuner, self.base_params)
        # Population search alternatives (rl.diffusion_ft.es_population): sample a
        # base population, project it with a named bank, smooth by re-noise /
        # denoise, offer the fittest feasible member to `cons`.
        from rl.diffusion_ft.es_population import build_es
        self._es_cfgs, self._es = build_es(
            self, es_spec, arms, bank_spec=bank_spec, banks_spec=banks_spec,
            ramp_steps=ramp_steps, bank_ramps=bank_ramps, bank_blends=bank_blends,
            bank_neighbors=self.bank_neighbors)

    def _prev_tail(self, prev_exec, W):
        """The plan executed at the previous replan from the current state on,
        ``[B, W, 5]`` (None on the first replan). Its last pose is held out to W;
        the tail is 41 steps of a 51-step plan, so the scored window (<= 20) never
        reaches the padding."""
        if prev_exec is None:
            return None
        jnp = self.jnp
        w_last, k_last = prev_exec
        tail = w_last[:, k_last:]
        pad = jnp.repeat(tail[:, -1:], W - tail.shape[1], axis=1)
        return jnp.concatenate([tail, pad], axis=1)

    def _optimized(self, arm, inits, world_p, base_sc, fixed_sc, cands, alive, t0, goal, n,
                   score, world_prev=None):
        """Loss-optimized lateral edits of ``world_p``, one per init token
        (:mod:`rl.diffusion_ft.lateral_opt`).

        Returns ``(worlds, scores, magnitudes, diag, tol_extra)``: lists of length
        K_opt in the bank branch's own formats, ``tol_extra`` [K_opt] metres added to
        the progress tolerance (``brake`` only), plus ``diag`` ``[B, K_opt, 4]`` = (loss at the
        init, loss reached, peak |d| m, peak |s| m -- 0 unless ``lon``), NaN on rows
        that were not optimized.

        ``prev`` / ``lonprev`` edit ``world_prev`` instead (the tail of the plan
        executed at the previous replan, anchored at the current state; ``None`` on
        the first replan); ``lon0`` / ``lonprev`` switch the along-path profile on
        for that candidate. Each kind is ranked by a magnitude offset from
        ``OPT_TIER_M``.

        Only TRIGGERED rows (alive, base predicted to violate) are optimized: ``cons``
        cannot switch anywhere else, so elsewhere the candidate is the base proposal
        itself, with the base's own scores -- and a replan with no triggered row
        skips the optimizer entirely, which is what keeps success_heldout cheap.
        Magnitude is the peak offset over the window + 0.1, the same units as a fixed
        ``latA`` candidate's ``|A| + 0.1``, so the smallest-edit rule compares them.
        """
        jnp = self.jnp
        from rl.diffusion_ft.branching import lateral_ease
        from rl.diffusion_ft.lateral_opt import (BRAKE_BACK_M, edit_world, init_kind,
                                                 init_variant)

        B, W = world_p.shape[0], world_p.shape[1]
        K = len(inits)
        alive_b = np.asarray(alive) > 0.5
        trig0 = alive_b & (np.asarray(base_sc[0]) < 0.0)
        # `proj`: the FRESH-sample candidates run on clean rows too (below theta);
        # `prev` ones stay violation-triggered, so a plan with zero loss is never
        # traded for the old one.
        proj = arm.selector == "proj"
        trig_fresh = trig0
        if proj:
            th = np.inf if arm.theta is None else float(arm.theta)
            trig_fresh = alive_b & (np.asarray(base_sc[0]) < th)
        diag = np.full((B, K, 4), np.nan, np.float32)
        tolx = [BRAKE_BACK_M if init_variant(t) == "brake" else 0.0 for t in inits]
        if not (trig0 | trig_fresh).any():
            return [world_p] * K, [base_sc] * K, [np.full(B, 0.1)] * K, diag, tolx
        rs = int(self.bank_ramps.get(arm.bank, self.ramp_steps))
        ease = np.asarray(lateral_ease(W, rs, str(self.bank_blends.get(arm.bank, "cubic"))))[1:n + 1]
        lat = [i for i, c in enumerate(cands) if c.family == "lat"]
        bank_lon = bool(self.bank_opt_cfg[arm.bank].lon)
        st, t0j = self.t.env.sim_state, jnp.asarray(t0, jnp.int32)
        worlds, scores, mags = [], [], []
        for k, tok in enumerate(inits):
            uses_prev, tok_lon = init_kind(tok)
            variant = init_variant(tok)
            lon = int(bank_lon or tok_lon)
            opt = self._tok_optimizer[(arm.bank, (bool(lon), variant))]
            src = world_p
            tier = EMERG_TIER_M[variant] if variant else OPT_TIER_M[(uses_prev, bool(lon))]
            trig = trig0 if (uses_prev or variant) else trig_fresh
            if variant == "brake":
                # A car's violation on a clean road edge only: braking cannot then
                # "clear" a static edge by sliding it past the scored window.
                trig = trig & (np.asarray(base_sc[1]) >= 0.0)
            if not trig.any():
                worlds.append(world_p); scores.append(base_sc); mags.append(np.full(B, 0.1))
                continue
            if uses_prev:
                if world_prev is None:
                    # First replan: nothing was executed yet. The base proposal with
                    # the base's own scores, which `cons` can never pick.
                    worlds.append(world_p); scores.append(base_sc); mags.append(np.full(B, 0.1))
                    continue
                src = world_prev
                amp = np.zeros(B)
            elif tok in ("0", "lon0", "emerg", "brake"):
                amp = np.zeros(B)
            elif tok == "best":
                clear_lat = np.stack([fixed_sc[i][0] for i in lat])          # [L, B]
                amp = np.asarray([cands[i].param for i in lat])[np.argmax(clear_lat, axis=0)]
            else:
                amp = np.full(B, float(tok[3:]))
            x0 = amp[:, None] * ease[None, :]
            if lon:
                x0 = np.concatenate([x0, np.zeros_like(x0)], axis=1)
            x0 = jnp.asarray(x0, jnp.float32)
            x, l_best, l0 = opt(src, x0, st, t0j, jnp.asarray(trig, jnp.float32), n)
            xa = np.asarray(x)
            peak = np.max(np.abs(xa[:, :n]), axis=1)
            peak_s = np.max(np.abs(xa[:, n:]), axis=1) if lon else np.zeros(B)
            use = trig
            if proj and not uses_prev:
                # An edit that did not move IS the base proposal (its world would
                # differ only by finite-difference velocities), with the base's own
                # scores -- which `proj` can never pick.
                use = trig & ((peak + peak_s) > 0.0)
            keep = jnp.asarray(use)[:, None, None]
            w = jnp.where(keep, edit_world(src, x, n=n, dt=self._opt_dt, lon=lon), world_p)
            worlds.append(w)
            sc_k = tuple(np.asarray(v) for v in score(w, st, t0j, goal, n))
            if proj and not uses_prev:
                sc_k = tuple(np.where(use, a, b) for a, b in zip(sc_k, base_sc))
            scores.append(sc_k)
            mags.append(np.where(trig, peak + peak_s + tier, 0.0) + 0.1)
            diag[trig, k] = np.stack([np.asarray(l0), np.asarray(l_best), peak, peak_s],
                                     -1)[trig]
        return worlds, scores, mags, diag, tolx

    def _argmin(self, arm, world_p, base_sc, alive, t0, goal, n, score, world_prev):
        """One objective, every alive row: the bank's optimizer from each start,
        and the plan with the lowest final loss. All starts optimize offsets of the
        FRESH plan (``initprev`` starts from the previous plan's tail projected onto
        it, :func:`lateral_opt.project_onto_path`), so their losses are the same
        function and the argmin is a fair comparison.

        Returns ``(world, switch, choice, clear_k [B, K], diag [B, K, 4], scores)``:
        the plan to execute, whether it differs from the base (it moved and, under
        ``argmin_c``, does not swap a predicted-clean base for a predicted
        violation), the winning start (-1 = base), each start's certified clearance,
        (loss0, loss, peak |d|, peak |s|) per start, and the winner's scores.
        """
        jnp = self.jnp
        from rl.diffusion_ft.lateral_opt import edit_world, project_onto_path

        inits = self.bank_opts[arm.bank]
        B, W = world_p.shape[0], world_p.shape[1]
        K = len(inits)
        alive_b = np.asarray(alive) > 0.5
        opt = self._bank_optimizer[arm.bank]
        st, t0j = self.t.env.sim_state, jnp.asarray(t0, jnp.int32)
        act = jnp.asarray(alive_b, jnp.float32)
        xs, losses = [], np.full((K, B), np.inf)
        diag = np.full((B, K, 4), np.nan, np.float32)
        clear_k = np.full((B, K), np.nan, np.float32)
        for k, tok in enumerate(inits):
            if tok == "initprev":
                if world_prev is None:
                    xs.append(np.zeros((B, 2 * n), np.float32))
                    continue
                x0 = project_onto_path(world_p, world_prev, n)
            else:
                x0 = np.zeros((B, 2 * n), np.float32)
            x, l_best, l0 = opt(world_p, jnp.asarray(x0), st, t0j, act, n)
            xa = np.asarray(x)
            xs.append(xa)
            losses[k] = np.where(alive_b, np.asarray(l_best), np.inf)
            w = edit_world(world_p, x, n=n, dt=self._opt_dt, lon=1)
            clear_k[:, k] = np.asarray(score(w, st, t0j, goal, n)[0])
            diag[alive_b, k] = np.stack([np.asarray(l0), np.asarray(l_best),
                                         np.abs(xa[:, :n]).max(1), np.abs(xa[:, n:]).max(1)],
                                        -1)[alive_b]
        # Ties go to the earlier start, so a zero-loss row keeps init0's x = 0.
        kstar = np.argmin(losses, axis=0)
        xstar = np.stack(xs)[kstar, np.arange(B)]
        moved = alive_b & np.any(xstar != 0.0, axis=1)
        w_star = edit_world(world_p, jnp.asarray(xstar), n=n, dt=self._opt_dt, lon=1)
        sc_star = tuple(np.asarray(v) for v in score(w_star, st, t0j, goal, n))
        sw = moved
        if arm.selector == "argmin_c":
            sw = sw & ~((np.asarray(base_sc[0]) >= 0.0) & (sc_star[0] < 0.0))
        world = jnp.where(jnp.asarray(sw)[:, None, None], w_star, world_p)
        choice = np.where(sw, kstar, -1).astype(np.int32)
        return world, sw, choice, clear_k, diag, sc_star

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

    def _bc_rows(self, *, x_p, world_p, world, sw, aux, cond1, cond2, mask, alive_np,
                 r, t0, choice):
        """One replan's behaviour-cloning rows (alive scenes only).

        ``target`` is the plan the arm executed, normalized ``[T, 5]``: the base's
        own sample ``x_p`` verbatim where the arm kept it, and ``world_to_norm`` of
        the executed world plan where it switched. ``valid`` marks the model steps
        the world time base covers (``branching.model_step_valid``); past them a
        switched target falls back to ``x_p``, as `counterfactual --mode repair`
        does. ``rt_err_m`` is the per-row round trip ``world_to_norm(world_p)``
        vs ``x_p`` on covered steps, metres -- the error every switched target
        carries.

        ``rejected`` is the base draw the arm turned down, for a preference pair
        (``train_safety --pref_weight``): on switched rows ``world_to_norm(world_p)``
        on exactly the steps where ``target`` is ``world_to_norm(world)``, and
        ``x_p`` everywhere else. Both sides of a pair then went through the same
        normalization path, so ``target - rejected`` is the edit and nothing else
        -- the round trip's 1-2 cm is not a difference the pair could teach. On a
        kept row it equals ``target``."""
        jnp = self.jnp
        base_norm = jnp.transpose(x_p, (0, 2, 1))                       # [B, T, 5]
        vmask = jnp.asarray(self._bc_valid(aux))                         # [B, T]
        err = jnp.abs(self._w2n(world_p, aux) - base_norm)[..., :2]
        rt = np.asarray(jnp.max(jnp.where(vmask[..., None], err, 0.0), axis=(1, 2))) * self._ego_range
        if not self._bc_checked:
            # Same bar as `counterfactual._roundtrip_check`: ~2 cm is the expected
            # interpolation residual, a metre means the targets would be a
            # DIFFERENT trajectory than the one that was executed.
            worst = float(rt[alive_np].max()) if alive_np.any() else 0.0
            print(f"[select] bc round trip on covered steps: max {worst:.3f} m over "
                  f"{int(alive_np.sum())} rows", flush=True)
            if worst > 1.0:
                raise SystemExit(f"[select] world<->norm round trip is lossy ({worst:.2f} m): "
                                 f"the bc targets would not be the executed plans. Refusing.")
            self._bc_checked = True
        tgt = rej = base_norm
        if sw.any():
            edit = jnp.asarray(sw)[:, None, None] & vmask[..., None]
            tgt = jnp.where(edit, self._w2n(world, aux), base_norm)
            rej = jnp.where(edit, self._w2n(world_p, aux), base_norm)
        i = np.nonzero(alive_np)[0]
        return {"b": i, "replan": np.full(len(i), r, np.int16),
                "t0": np.full(len(i), t0, np.int16),
                "switched": np.asarray(sw)[i],
                "choice": (np.asarray(choice)[i].astype(np.int8) if choice is not None
                           else np.full(len(i), -1, np.int8)),
                "cond1": np.asarray(cond1)[i], "cond2": np.asarray(cond2)[i],
                "mask": np.asarray(mask)[i], "target": np.asarray(tgt, np.float32)[i],
                "rejected": np.asarray(rej, np.float32)[i],
                "valid": np.asarray(vmask)[i], "rt_err_m": rt[i].astype(np.float32)}

    def _pair_rows(self, *, x_base, x_prefix, noise_late, world_b, world, sw, aux, cond1, cond2,
                   mask, alive_np, r, t0, n, goal, feat, choice):
        """One replan's paired-repair rows (alive scenes only); see ``dump_pairs``.

        ``noise_late`` is ``[k, B, C, T]``: row b's late chain is replayed exactly by
        ``adapter.base_steps`` from ``x_prefix[b]`` with ``noise_late[:, b]``, and
        must land on ``x_base[b]``, the base draw (``train_adapter`` re-checks it on
        every row it loads). ``world_b`` is that draw in world coordinates and
        ``world`` the plan the arm executed; they differ only where it switched."""
        i = np.nonzero(alive_np)[0]
        out = {"b": i, "replan": np.full(len(i), r, np.int16), "t0": np.full(len(i), t0, np.int16),
               "n": np.full(len(i), n, np.int16), "switched": np.asarray(sw)[i],
               "choice": (np.asarray(choice)[i].astype(np.int8) if choice is not None
                          else np.full(len(i), -1, np.int8)),
               "feat": np.asarray(feat, np.float32)[i],
               "cond1": np.asarray(cond1)[i], "cond2": np.asarray(cond2)[i],
               "mask": np.asarray(mask)[i],
               "x_prefix": np.asarray(x_prefix, np.float32)[i],
               "noise_late": np.ascontiguousarray(np.swapaxes(np.asarray(noise_late, np.float32), 0, 1)[i]),
               "x_base": np.asarray(x_base, np.float32)[i],
               "world_base": np.asarray(world_b, np.float32)[i],
               "world_exec": np.asarray(world, np.float32)[i],
               "goal": np.asarray(goal, np.float32)[i]}
        B = int(np.asarray(x_base).shape[0])
        for k, v in aux.items():
            v = np.asarray(v)
            if v.ndim == 0 or v.shape[0] != B:
                raise SystemExit(f"[select] aux leaf '{k}' {v.shape} has no batch axis; "
                                 f"a pair row could not be mapped back to world coordinates")
            out[f"aux/{k}"] = v[i]
        return out

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
        primary = self.ft_params if arm.primary == "ft" else self.base_params

        features = env.reset(state, seed=args.seed)
        clean_b = t._gate_expert_clean(env.sim_state)
        alive = clean_b
        B = env.batch_size
        t_start = env.timestep
        vmax = float(t.ckpt.preprocess_cfg.max_velocity)
        from data.preprocess import compute_world_dt_seconds
        fb = float(t.ckpt.preprocess_cfg.world_dt_fallback)
        dt_cols = {"world_dt": np.asarray(env.last_aux["world_dt_seconds"], np.float64),
                   "dt_first": np.asarray(compute_world_dt_seconds(env.sim_state, fb, "first")),
                   "dt_median": np.asarray(compute_world_dt_seconds(env.sim_state, fb, "median"))}
        obs_v, obs_t, obs_alive = [], [], []   # the ego speed each replan was SHOWN
        offroad = np.zeros(B); overlap = np.zeros(B); reached = np.zeros(B)
        # The same three flags counted only while the row is still ALIVE. The loop
        # keeps driving a terminated row (with the base proposal) until every row
        # in the batch is done, and the flags above OR in whatever it hits then --
        # so a car that reached its goal and later drifted off the road reads
        # unsolved, depending on which other scenes shared its batch. `*_term`
        # stops counting at the step the env terminated the row (collision or goal;
        # off-road does not terminate under RewardConfig's default).
        rcfg = env._reward_cfg
        off_t = np.zeros(B, bool); ovl_t = np.zeros(B, bool); rea_t = np.zeros(B, bool)
        feats, switches, alives, choices, clears_k = [], [], [], [], []
        trajs, trajs_b = [], []          # executed states, and the base counterfactual
        bc_rows = []                     # --dump_bc: (condition, executed plan) per replan
        pair_rows = []                   # --dump_pairs: replayable draw + executed plan
        ad_gate, ad_logit, ad_shift = [], [], []   # adapter arms: per replan [B]
        cands =self.banks.get(arm.bank, []) if arm.alt == "bank" else []
        perturb = self._perturb.get(arm.bank, []) if arm.alt == "bank" else []
        # The arm's own neighbour source scores BOTH the base proposal (the
        # trigger) and the candidates (the certificate): an honest arm must not
        # use the oracle for either half of its decision.
        score = self._scorers[self.bank_neighbors.get(arm.bank, "log") if arm.alt == "bank"
                              else self._es_cfgs[arm.es].nb if arm.alt == "es" else "log"]
        opt_inits = self.bank_opts.get(arm.bank, []) if arm.alt == "bank" else []
        nb = len(cands) + len(opt_inits)
        opt_ks = []                      # per replan [B, K_opt, 4]: loss0, loss, peak |d|, |s|
        # The plan executed at the previous replan and how many of its steps ran:
        # its tail, anchored at the current state, is what a `prev` candidate edits.
        prev_exec = None
        es_diags = []                    # es arms: per replan [B, len(ES_DIAG)]
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
            if r >= 1:
                # From r=1 the anchor was driven by the policy, so this is the
                # velocity it is conditioned on for its OWN motion (ego_state is
                # [x, y, vx, vy, yaw] normalized; the norm is rotation-free).
                obs_v.append(np.linalg.norm(np.asarray(features["ego_state"])[:, 2:4],
                                            axis=-1) * vmax)
                obs_t.append(t0)
                obs_alive.append(np.asarray(alive) > 0.5)
            aux = env.last_aux
            key, rk = jax.random.split(key)
            x_p, (x_prefix, key_late) = actor.sample_world_with_prefix(
                primary, cond1, cond2, mask, rk)
            x_draw = x_p
            noise_late = (self._late_noise(key_late, x_p)
                          if (self.dump_pairs or arm.primary == "adapter") else None)
            if arm.primary == "adapter":
                # The base arm's own draw (same prefix, same late noise); the tail
                # adapter rewrites only its last K steps.
                x_p, a_g, a_lg = self._adapter_tail[arm.gate](
                    cond1, cond2, mask, x_prefix, noise_late, float(arm.strength))
            world_p = self._world(x_p, aux)
            n = int(min(arm.window, horizon - 1 - t0, world_p.shape[1] - 1))
            world_draw = world_p
            if arm.primary == "adapter":
                world_draw = self._world(x_draw, aux)
                ad_gate.append(np.asarray(a_g, np.float32))
                ad_logit.append(np.asarray(a_lg, np.float32))
                ad_shift.append(np.asarray(jnp.max(jnp.linalg.norm(
                    world_p[:, 1:n + 1, :2] - world_draw[:, 1:n + 1, :2], axis=-1), axis=1)))
            goal = env._goal_xy_b2
            cp, ep, ap, pp = (np.asarray(v) for v in
                              score(world_p, env.sim_state, jnp.asarray(t0, jnp.int32), goal, n))
            row = np.full((B, len(FEAT_COLS)), np.nan, np.float32)
            row[:, 0], row[:, 1], row[:, 2], row[:, 3] = cp, ep, ap, pp
            sw = np.zeros(B, dtype=bool)
            world = world_p
            if arm.alt == "bank" and arm.selector in ARGMIN_SELECTORS:
                world_prev = self._prev_tail(prev_exec, world_p.shape[1])
                world, sw, ch, ck_row, ok, s_star = self._argmin(
                    arm, world_p, (cp, ep, ap, pp), alive, t0, goal, n, score, world_prev)
                for j, col in enumerate((4, 5, 6, 7)):
                    row[:, col] = s_star[j]
                choices.append(ch); clears_k.append(ck_row); opt_ks.append(ok)
            elif arm.alt == "bank":
                # K geometric edits of the base's OWN proposal -- no denoiser call.
                ws = [f(world_p) for f in perturb]
                sc = [tuple(np.asarray(v) for v in
                            score(w, env.sim_state, jnp.asarray(t0, jnp.int32), goal, n))
                      for w in ws]
                mags = [np.full(B, c.magnitude) for c in cands]
                tol_k = None                     # every candidate at self.tol_m
                if opt_inits:
                    # ...plus loss-optimized edits of the same proposal, appended
                    # after the fixed ones (lateral_opt.py).
                    from rl.diffusion_ft.lateral_opt import init_kind
                    world_prev = (self._prev_tail(prev_exec, world_p.shape[1])
                                  if any(init_kind(t)[0] for t in opt_inits) else None)
                    wo, so, mo, ok, tx = self._optimized(arm, opt_inits, world_p,
                                                         (cp, ep, ap, pp), sc, cands, alive, t0,
                                                         goal, n, score, world_prev=world_prev)
                    ws, sc, mags = ws + wo, sc + so, mags + mo
                    tol_k = [self.tol_m] * len(cands) + [self.tol_m + x for x in tx]
                    opt_ks.append(ok)
                ck = np.stack([x[0] for x in sc]); pk = np.stack([x[3] for x in sc])
                pref = None
                if arm.bank in self.bank_prefer:
                    em, am = self.bank_prefer[arm.bank]
                    pref = (np.stack([x[1] for x in sc]) >= em) & (np.stack([x[2] for x in sc]) >= am)
                ch = bank_choice(arm.selector, clear_p=cp, prog_p=pp, clear_k=ck, prog_k=pk,
                                 magnitude=np.stack(mags), prefer_k=pref, tol_k=tol_k,
                                 tol_m=self.tol_m, margin_m=self.margin_m, delta_m=self.delta_m)
                sw = ch >= 0
                bi = np.arange(B)
                # The `_a` columns report the CHOSEN candidate where one fired and
                # the clearest one otherwise, so they always describe the best
                # alternative that was on the table.
                rep = np.where(sw, ch, np.argmax(ck, axis=0))
                for j, col in enumerate((4, 5, 6, 7)):
                    row[:, col] = np.stack([x[j] for x in sc])[rep, bi]
                world = jnp.where(jnp.asarray(sw)[:, None, None],
                                  jnp.stack(ws, 0)[np.maximum(ch, 0), bi], world_p)
                choices.append(ch)
                clears_k.append(np.ascontiguousarray(ck.T, np.float32))
            elif arm.alt == "es":
                # A base POPULATION, each member projected with the bank and
                # smoothed by re-noise/denoise; runs only where the base proposal
                # is predicted to violate, and its winner still has to pass cons.
                w_es, s_es, dg = self._es[arm.es](
                    x_p=x_p, world_p=world_p, cp=cp, pp=pp, cond1=cond1, cond2=cond2,
                    mask=mask, aux=aux, state=env.sim_state, t0=t0, goal=goal, n=n,
                    rk=rk, alive=alive)
                if s_es is not None:
                    trig = dg[:, 0] > 0.5
                    for j, k in enumerate(("clear", "edge", "agent", "prog")):
                        row[trig, 4 + j] = s_es[k][trig]
                    sw = trig & select_switch(
                        arm.selector, clear_p=cp, clear_a=s_es["clear"], prog_p=pp,
                        prog_a=s_es["prog"], tol_m=self.tol_m, margin_m=self.margin_m,
                        delta_m=self.delta_m)
                    world = jnp.where(jnp.asarray(sw)[:, None, None], w_es, world_p)
                es_diags.append(dg)
            elif arm.alt is not None:
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
                                  score(world_a, env.sim_state, jnp.asarray(t0, jnp.int32), goal, n))
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
            if self.dump_bc and alive_np.any():
                bc_rows.append(self._bc_rows(
                    x_p=x_p, world_p=world_p, world=world, sw=sw & alive_np, aux=aux,
                    cond1=cond1, cond2=cond2, mask=mask, alive_np=alive_np, r=r, t0=t0,
                    choice=ch if arm.alt == "bank" else None))
            if self.dump_pairs and alive_np.any():
                pair_rows.append(self._pair_rows(
                    x_base=x_draw, x_prefix=x_prefix, noise_late=noise_late, world_b=world_draw,
                    world=world, sw=sw & alive_np, aux=aux, cond1=cond1, cond2=cond2, mask=mask,
                    alive_np=alive_np, r=r, t0=t0, n=n, goal=goal, feat=row,
                    choice=ch if arm.alt == "bank" else None))

            if self.dump_traj:
                # The EXECUTED slice only (the env advances n_exec of the candidate's
                # horizon), plus what the base would have driven over the same steps,
                # so a figure can overlay them. Padded to a rectangular array with
                # NaN; ~51 kB per arm-repeat chunk, so it is cheap but still gated.
                pad = int(args.replan_interval_steps)
                tr = np.asarray(world[:, :n_exec], np.float32)
                trb = np.asarray(world_draw[:, :n_exec], np.float32)   # the base draw
                if n_exec < pad:
                    z = np.full((B, pad - n_exec, tr.shape[-1]), np.nan, np.float32)
                    tr, trb = np.concatenate([tr, z], 1), np.concatenate([trb, z], 1)
                trajs.append(tr); trajs_b.append(trb)
            out = env.step(trajectory_world_bt5=world)
            prev_exec = (world, n_exec)
            comp = out.components
            offroad = np.maximum(offroad, np.asarray(comp["offroad_flag"], np.float32) > 0)
            overlap = np.maximum(overlap, np.asarray(comp["collision_flag"], np.float32) > 0)
            reached = np.maximum(reached, np.asarray(comp["reached_flag"], np.float32) > 0)
            psc = out.per_step_components
            act = alive_np.copy()
            for k in range(np.asarray(psc["offroad_flag"]).shape[-1]):
                o_k = np.asarray(psc["offroad_flag"])[:, k] > 0
                c_k = np.asarray(psc["collision_flag"])[:, k] > 0
                g_k = np.asarray(psc["reached_flag"])[:, k] > 0
                off_t |= act & o_k; ovl_t |= act & c_k; rea_t |= act & g_k
                act &= ~((c_k & bool(rcfg.terminate_on_collision))
                         | (o_k & bool(rcfg.terminate_on_offroad)) | g_k)
            done = jnp.asarray(out.terminated_b) | jnp.asarray(out.truncated_b)
            alive = alive * (1.0 - done.astype(jnp.float32))
            features = out.features

        R = len(feats)
        F = (np.stack(feats, 1) if R else np.zeros((B, 0, len(FEAT_COLS)), np.float32))
        S = np.stack(switches, 1) if R else np.zeros((B, 0), bool)
        A = np.stack(alives, 1) if R else np.zeros((B, 0), bool)
        # WHICH candidate fired, and every candidate's clearance -- kept as data so
        # the bank can be re-ranked or a selector fit offline without re-rolling.
        C = (np.stack(choices, 1) if choices else np.zeros((B, R), np.int32) - 1)
        CK = (np.stack(clears_k, 1) if clears_k
              else np.zeros((B, R, nb), np.float32))
        solved = (reached > 0.5) & (offroad < 0.5) & (overlap < 0.5)
        st = env.sim_state
        is_sdc = np.asarray(st.object_metadata.is_sdc)
        ego_valid = np.asarray(st.log_trajectory.valid)[np.arange(B), np.argmax(is_sdc, 1)]
        pace = episode_pace(
            np.asarray(env.ego_sim_world_trajectory())[..., :2],
            np.asarray(env.ego_log_world_trajectory())[..., :2], ego_valid,
            t_start=t_start, t_end=env.timestep, goal_t=np.asarray(env._goal_t_b),
            goal_xy=np.asarray(env._goal_xy_b2),
            dt=np.asarray(env.last_aux["world_dt_seconds"]),
            obs_v=(np.stack(obs_v, 1) if obs_v else np.zeros((B, 0))), obs_t=obs_t,
            obs_alive=(np.stack(obs_alive, 1) if obs_alive else np.zeros((B, 0), bool)))
        out_d = {"solved": solved, "offroad": offroad > 0.5, "overlap": overlap > 0.5,
                 "reached": reached > 0.5, "clean": np.asarray(clean_b) > 0.5,
                 "solved_term": rea_t & ~off_t & ~ovl_t, "offroad_term": off_t,
                 "overlap_term": ovl_t, "reached_term": rea_t,
                 "feat": F, "switch": S, "alive": A, "choice": C, "clear_k": CK,
                 "opt_k": np.stack(opt_ks, 1) if opt_ks else None,
                 **pace, **dt_cols}
        if self.dump_traj:
            out_d["traj"] = (np.stack(trajs, 1) if trajs
                             else np.zeros((B, 0, 0, 5), np.float32))
            out_d["traj_base"] = (np.stack(trajs_b, 1) if trajs_b
                                  else np.zeros((B, 0, 0, 5), np.float32))
        if self.dump_bc:
            out_d["bc"] = ({k: np.concatenate([d[k] for d in bc_rows], 0) for k in bc_rows[0]}
                           if bc_rows else None)
        if self.dump_pairs:
            out_d["pairs"] = ({k: np.concatenate([d[k] for d in pair_rows], 0) for k in pair_rows[0]}
                              if pair_rows else None)
        if arm.primary == "adapter":
            # Per replan: the gate, its logit and how far (m, max over the scored
            # window) the adapter moved the base draw. Post-termination replans of
            # a row are included; mask with `alive`.
            for k, v in (("adapter_gate", ad_gate), ("adapter_logit", ad_logit),
                         ("adapter_shift_m", ad_shift)):
                out_d[k] = np.stack(v, 1) if v else np.zeros((B, 0), np.float32)
        if arm.alt == "es":
            from rl.diffusion_ft.es_population import ES_DIAG
            out_d["es_diag"] = (np.stack(es_diags, 1) if es_diags
                                else np.zeros((B, 0, len(ES_DIAG)), np.float32))
        return out_d

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
                # Column order of clear_k/choice: fixed candidates, then "opt:<init>".
                "banks": {k: [c.name for c in v] + [f"opt:{i}" for i in self.bank_opts.get(k, [])]
                          for k, v in self.banks.items()},
                "bank_opts": self.bank_opts,
                "opt_cfg": dataclasses.asdict(self.opt_cfg),
                "bank_opt_cfg": {k: dataclasses.asdict(v) for k, v in self.bank_opt_cfg.items()},
                "bank_prefer": self.bank_prefer,
                "ramp_steps": self.ramp_steps,
                "dump_traj": self.dump_traj, "dump_bc": self.dump_bc,
                "dump_pairs": self.dump_pairs, "adapter_ckpt": self.adapter_ckpt,
                "bank_ramps": getattr(self, "bank_ramps", {}),
                "bank_blends": getattr(self, "bank_blends", {}),
                "bank_neighbors": getattr(self, "bank_neighbors", {}),
                "selector_theta": (self.selector_model or {}).get("theta"),
                "shard": [shard_index, num_shards], "cap": int(cap), "splits": list(names),
                "splits_json": str(t.args.splits_json),
                # Both change what an episode MEANS, so an analysis pairing runs
                # across jobs must be able to read them rather than assume them.
                "sim_agents": t.env.sim_agents, "ego_vel_source": t.env.ego_vel_source,
                "world_dt_method": t.env.world_dt_method}
        if self._es_cfgs:
            from rl.diffusion_ft.es_population import ES_DIAG
            meta["es"] = {k: dict(c.__dict__) for k, c in self._es_cfgs.items()}
            meta["es_diag_cols"] = list(ES_DIAG)
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
                bc, pairs = {}, {}
                for arm in self.arms:
                    for r in range(repeats):
                        # Same fold for every arm: the arms are paired draw for draw.
                        o = self.rollout(state, arm, jax.random.fold_in(key, r))
                        if o.get("bc") is not None:
                            d, b = dict(o["bc"]), o["bc"]["b"]
                            d.pop("b")
                            # Episode outcome per row, so the trainer can keep
                            # only the rollouts that solved (filtered BC).
                            d.update({"scene_id": np.asarray(sub, np.int64)[b],
                                      "repeat": np.full(len(b), r, np.int8),
                                      "split": np.full(len(b), name, dtype="U32"),
                                      "ep_solved": o["solved"][b], "ep_reached": o["reached"][b],
                                      "ep_offroad": o["offroad"][b], "ep_overlap": o["overlap"][b]})
                            bc.setdefault(arm.name, []).append(d)
                        if o.get("pairs") is not None:
                            d, b = dict(o["pairs"]), o["pairs"]["b"]
                            d.pop("b")
                            d.update({"scene_id": np.asarray(sub, np.int64)[b],
                                      "repeat": np.full(len(b), r, np.int8),
                                      "split": np.full(len(b), name, dtype="U32"),
                                      "arm": np.full(len(b), arm.name, dtype="U32"),
                                      **{f"ep_{m}": o[m][b] for m in (
                                          "solved", "reached", "offroad", "overlap",
                                          "solved_term", "offroad_term", "overlap_term",
                                          "reached_term")}})
                            pairs.setdefault(arm.name, []).append(d)
                        for k in ("adapter_gate", "adapter_logit", "adapter_shift_m"):
                            if k in o:
                                npz[f"{arm.name}/r{r}/{k[len('adapter_'):]}"] = o[k]
                        npz[f"{arm.name}/r{r}/feat"] = o["feat"]
                        npz[f"{arm.name}/r{r}/switch"] = o["switch"]
                        npz[f"{arm.name}/r{r}/alive"] = o["alive"]
                        if arm.alt == "bank":
                            npz[f"{arm.name}/r{r}/choice"] = o["choice"].astype(np.int8)
                            npz[f"{arm.name}/r{r}/clear_k"] = o["clear_k"]
                        if o.get("opt_k") is not None:
                            npz[f"{arm.name}/r{r}/opt_k"] = o["opt_k"]
                        if "es_diag" in o:
                            npz[f"{arm.name}/r{r}/es_diag"] = o["es_diag"]
                        if "traj" in o:
                            npz[f"{arm.name}/r{r}/traj"] = o["traj"]
                            npz[f"{arm.name}/r{r}/traj_base"] = o["traj_base"]
                        tally.setdefault(arm.name, []).append(float(o["solved"].mean()))
                        for j, sid in enumerate(sub):
                            recs.append({
                                "scene_id": int(sid), "split": name, "arm": arm.name,
                                "repeat": r, "seed": int(seed),
                                # "" for arms that never touch the fine-tune, so
                                # the analysis can pool them across ckpt tasks.
                                "ckpt": (self.ft_ckpt if arm.needs_ft else
                                         self.adapter_ckpt if arm.primary == "adapter" else ""),
                                "solved": bool(o["solved"][j]), "offroad": bool(o["offroad"][j]),
                                "overlap": bool(o["overlap"][j]), "reached": bool(o["reached"][j]),
                                "clean": bool(o["clean"][j]),
                                **{f"{m}_term": bool(o[f"{m}_term"][j])
                                   for m in ("solved", "offroad", "overlap", "reached")},
                                "n_replans": int(o["alive"][j].sum()),
                                "n_switch": int(o["switch"][j].sum()),
                                **{c: _finite_or_none(o[c][j]) for c in PACE_COLS + DT_COLS}})
                np.savez_compressed(out_dir / "feat" / f"{name}_s{shard_index}_{start:05d}.npz", **npz)
                # Before the rows, so a chunk whose rows exist always has its data;
                # a kill in between re-runs the chunk and overwrites the same file.
                for arm_name, parts in bc.items():
                    bdir = out_dir / "bc_data" / arm_name
                    bdir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        bdir / f"{name}_s{shard_index}_{start:05d}.npz",
                        **{k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]})
                for arm_name, parts in pairs.items():
                    pdir = out_dir / "pair_data" / arm_name
                    pdir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        pdir / f"{name}_s{shard_index}_{start:05d}.npz",
                        **{k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]})
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
    p.add_argument("--select_bank", default="tw0.85,tw1.15,lat0.25,lat-0.25,tw0.7,tw1.3,lat0.5,lat-0.5",
                   help="candidate bank for alt 'bank' (branching.Candidate specs, tw/lat only). "
                        "The default is the repair search's tw/lat families at the sizes it kept")
    p.add_argument("--select_banks", default="",
                   help="extra NAMED banks, 'name=spec;name=spec', referenced by an arm "
                        "as alt 'bank@name'. Lets a lateral-only and a timing-only arm "
                        "run in one job, paired draw for draw")
    p.add_argument("--select_ramp_steps", type=int, default=10,
                   help="perturbation ease-in, world steps (counterfactual --ramp_steps)")
    p.add_argument("--select_bank_ramps", default="",
                   help="per-bank ramp override, 'name=steps;name=steps'. Peak lateral "
                        "acceleration goes as |A|/T^2 and jerk as |A|/T^3 with "
                        "T = steps * world_dt, so this is the main ride-comfort knob. "
                        "NOTE a ramp longer than --replan_interval_steps is truncated "
                        "by the next replan, so the executed offset never reaches A.")
    p.add_argument("--dump_traj", action="store_true",
                   help="also save the EXECUTED states per replan (and the base "
                        "counterfactual) into feat/*.npz, for the comfort/jerk "
                        "figures. ~51 kB per arm-repeat chunk; use with a cap.")
    p.add_argument("--dump_bc", action="store_true",
                   help="also save, per alive replan, the condition and the plan each "
                        "arm EXECUTED (normalized) to bc_data/<arm>/*.npz -- the "
                        "train_safety --repair_data format, to distill a selector arm "
                        "into weights. ~33 kB per row (cond1 dominates); run it on "
                        "*_train splits, never the held-out ones")
    p.add_argument("--dump_pairs", action="store_true",
                   help="also save, per alive replan, a REPLAYABLE base draw (frozen "
                        "prefix output + per-row late noise), the base plan and the plan "
                        "each arm executed (world coords), the replan's world transform "
                        "and goal, to pair_data/<arm>/*.npz -- rl/diffusion_ft/"
                        "train_adapter.py's input. *_train splits only")
    p.add_argument("--adapter_ckpt", default="",
                   help="gated tail adapter (train_adapter.py checkpoint dir) for "
                        "'name=adapter:<gate>[:strength]' arms")
    p.add_argument("--select_bank_neighbors", default="",
                   help="per-bank source of the OTHER cars' next steps when scoring, "
                        "'name=log|cv|none'. log (default) = the recorded future, an "
                        "oracle under log replay; cv = constant-velocity prediction "
                        "from their current state; none = road edges only")
    p.add_argument("--select_bank_blends", default="",
                   help="per-bank easing, 'name=cubic|quintic'. cubic (default) is C1: "
                        "acceleration STEPS at the join. quintic is C2, but only helps "
                        "once the ramp resolves it (>=20 steps); at ramp 10 it is "
                        "slightly worse. lat only -- tw ignores it.")
    p.add_argument("--select_bank_opts", default="",
                   help="per-bank LOSS-OPTIMIZED lateral candidates (lateral_opt.py), "
                        "'name=inits;name=inits', inits comma-separated from 0 | best | "
                        "lat<m>. One optimized edit per init is appended after the bank's "
                        "fixed candidates; a bank with opts may have an empty fixed spec")
    p.add_argument("--select_opt_cfg", default="",
                   help="lateral_opt.LateralOptConfig overrides, 'key=value,key=value' "
                        "(iters, step_m, decay, damping, w_acc, w_jerk, w_prox, a_ref, "
                        "j_ref, d_ref, barrier_shape, edge_geometry, edge_margin_m, "
                        "agent_margin_m)")
    p.add_argument("--select_bank_opt_cfgs", default="",
                   help="per-bank overrides ON TOP of --select_opt_cfg, "
                        "'name=key=value,key=value;name=...', so optimizer variants run "
                        "as paired arms in one job")
    p.add_argument("--select_bank_prefer", default="",
                   help="per-bank margin PREFERENCE, 'name=edge_m,agent_m': among clean "
                        "candidates, those clearing both margins win first; if none does, "
                        "the smallest clean one as before. Eligibility is unchanged")
    p.add_argument("--look_replans", type=int, default=2,
                   help="'look' arms: replans simulated per candidate (1 = just the executed window)")
    p.add_argument("--select_cap", type=int, default=-1,
                   help="first N scenes of each split (before sharding); -1 = all. Smoke only")
    p.add_argument("--select_es", default="",
                   help="NAMED population-search configs for alt 'es@name', "
                        "'name=pop:16,iters:2,elite:4,bank:<bank>,t:1,scale:0,nb:cv,rank:dev;...'. "
                        "Projects each member with the named bank, then re-noises/denoises "
                        "it t steps at scale x the forward noise (rl/diffusion_ft/es_population.py)")
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
    if args.dump_bc and any(n.endswith("_heldout") for n in names):
        raise SystemExit(f"--dump_bc writes TRAINING data; refusing held-out splits {names}")
    if args.dump_pairs and any(n.endswith("_heldout") for n in names):
        raise SystemExit(f"--dump_pairs writes TRAINING data; refusing held-out splits {names}")
    args.splits = ""              # stream chunks; do not preload eval states
    args.probe_per_split = 0
    tuner = SafetyFineTuner(args)
    runner = ProposalSelectEval(tuner, ft_ckpt=args.ft_ckpt, arms=arms,
                                tol_m=args.select_tol_m, margin_m=args.select_margin_m,
                                delta_m=args.select_delta_m, selector_json=args.selector_json,
                                look_replans=args.look_replans, bank_spec=args.select_bank,
                                banks_spec=args.select_banks,
                                ramp_steps=args.select_ramp_steps,
                                bank_ramps=_kv_spec(args.select_bank_ramps, int),
                                bank_blends=_kv_spec(args.select_bank_blends, str),
                                bank_neighbors=_kv_spec(args.select_bank_neighbors, str),
                                dump_traj=bool(args.dump_traj),
                                dump_bc=bool(args.dump_bc),
                                bank_opts=_kv_spec(args.select_bank_opts, str),
                                opt_cfg=args.select_opt_cfg,
                                es_spec=args.select_es,
                                bank_opt_cfgs=_kv_spec(args.select_bank_opt_cfgs, str),
                                bank_prefer=_kv_spec(args.select_bank_prefer, str),
                                adapter_ckpt=args.adapter_ckpt,
                                dump_pairs=bool(args.dump_pairs))
    for name, c in runner._es_cfgs.items():
        print(f"[select] es {name}: pop={c.pop} iters={c.iters} elite={c.elite} "
              f"proj={'lateral_opt' if c.opt else (c.bank or '(none)')} order={c.order} "
              f"renoise t={c.t} scale={c.scale} nb={c.nb} "
              f"rank={c.rank}  (per triggered replan: {c.pop - 1} extra full samples, "
              f"then {c.t} denoiser evals per smoothed member per generation)", flush=True)
    for name, cands in runner.banks.items():
        print(f"[select] bank {name or '(default)'} ({len(cands)}, 0 denoiser evals): "
              + "  ".join(f"{c.name}(m={c.magnitude:.2f})" for c in cands), flush=True)
        if runner.bank_opts.get(name):
            print(f"[select]   + {len(runner.bank_opts[name])} loss-optimized, inits "
                  f"{runner.bank_opts[name]}: {runner.bank_opt_cfg[name]}", flush=True)
        if name in runner.bank_prefer:
            print(f"[select]   prefers edge >= {runner.bank_prefer[name][0]} m, "
                  f"agent >= {runner.bank_prefer[name][1]} m among clean candidates", flush=True)
    print("[select] arms: " + "  ".join(
        f"{a.name}=(adapter:{a.gate}:s{a.strength:g})" if a.primary == "adapter" else
        a.name if a.alt is None else
        f"{a.name}=({a.primary}|{a.alt}{'@' + a.bank if a.bank else ''}:{a.selector}:w{a.window})"
        for a in arms), flush=True)
    runner.run(Path(args.select_out or args.out_dir), names=names,
               repeats=int(args.select_repeats), seed=int(args.select_seed),
               chunk=max(1, int(args.eval_chunk)), shard_index=int(args.shard_index),
               num_shards=int(args.num_shards), cap=int(args.select_cap))
    # Host RAM is not observable after the fact on this cluster (sacct is down),
    # so the job reports its own peak; that is what --mem should be sized from.
    import resource
    print(f"[select] peak host RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GB",
          flush=True)


if __name__ == "__main__":
    main()
