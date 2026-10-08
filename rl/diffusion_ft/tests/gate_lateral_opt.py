"""Gates for the loss-optimized lateral repair (``lateral_opt.py``) on REAL scenes.

WHY. The pure tests pin the algebra; this pins the plumbing against the real
roadgraph, the real neighbour window and the real certificate, before a GPU is
spent on the select eval. No policy: the "base proposal" is the expert's own log
window, DISTORTED into violation three ways --

* ``ramp``  -- a rigid lateral shift (ramp 10): the case the l24 bank was built for;
* ``bump``  -- out and back (``A sin^2(pi t / n)``): a hazard on PART of the window,
  which no rigid translation undoes;
* ``rot``   -- the whole window rotated about the anchor (a heading error), so the
  deviation grows along the window;
* ``slow`` / ``fast`` -- the same path at 0.5x / 1.6x speed (``time_warp``): the car
  falls back into the traffic behind it or runs up on the traffic ahead, the
  case only the longitudinal half (``lon``) is built for --

and every triggered row (the certificate says the distorted window violates) is
offered to the same four arms ``submit_lateral_opt.sh`` runs: the ship bank
(``l24`` @ ramp 20, cubic), the optimizer from 0, the optimizer multi-start, and
the union (l24 + optimizer from the bank's clearest candidate and from 0). All
use the ``cons`` rule and the exact scorer of ``proposal_select.make_scorer``.

WHAT HAS TO HOLD (correctness, not the hypothesis):

* **O1 identity** -- where the loss is exactly 0 at the base line (every corner
  and car beyond its margin), the optimizer returns exactly d = 0.
* **O2 anchor** -- the optimized candidate keeps the anchor pose.
* **O3 descent** -- the returned loss never exceeds the init's, on any row.
* **O4 it repairs** -- the optimizer from 0 certifies >= 20% of the triggered rows
  and >= half as many as the bank. A sign error or a dead gradient repairs ~0%.
* **O5 finite** -- no NaN/inf in any optimized profile or loss.
* **O6 union contains the bank** -- every row the bank fires on, the union fires
  on too (the fixed candidates are appended FIRST and scored identically; an
  index bug in the appending breaks this).
* **L1-L4 the ``lon`` optimizer** (``opt0lon``, 2026-09-30): exact identity where
  the base line costs nothing, the anchor kept, the loss never above the init's,
  finite, and the along-path half never below ``-lon_back_m``.
* **L6 the lon optimizer MOVES** -- on triggered rows with a positive loss, the
  loss drops on >= 80% of them, on the distorted window AND on the same window
  with its last 10 poses held (the rollout's plans stop, and a ``prev`` tail is
  padded by holding its last pose; a NaN gradient there froze lon in the 09-30
  select run while O1-L5 all passed).
* **E1-E3 the emergency optimizers** (``emerg``, ``brake``: comfort x0.1, longer
  reach, ``brake`` up to BRAKE_BACK_M behind): finite, anchor kept, loss never
  above the init's, and ``brake`` never further back than BRAKE_BACK_M. Reported:
  the tiered union o_0 < lon < emerg < brake (``full``).
* **A1-A2 the argmin arms' optimizer** (lon on, ``w_lon_prox`` 0.05, no braking
  cap): exact identity where the base costs nothing; finite, anchor kept, loss
  never above the init's. Reported as ``argmin``: its plan where it moved and is
  predicted clean (one start here; the rollout adds the previous plan's tail).
* **L5 lon as an extra candidate** (``opt0+lon``, the o_l arm's bank: o_0's edit,
  then the lon edit ranked behind it by ``OPT_TIER_M``) fires on every row opt0
  fires on and executes opt0's own edit there -- so it can only add coverage.

WHAT IT REPORTS (the hypothesis, not gated -- the select eval decides): fix rate
per arm per distortion family, the chosen edit's size and ride quality (peak
lateral accel / jerk OF THE EDIT), how often the optimizer's edit is a bump that
comes back rather than a ramp, and seconds per call.

CPU, ~15-30 min. Submitted by ``submit_lateral_opt.sh`` (its first link)::

    sbatch rl/diffusion_ft/slurm/gate_lateral_opt.sbatch <env file>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from waymax import config as waymax_config
from waymax import datatypes
from waymax import dynamics as waymax_dynamics
from waymax import env as waymax_env

import dataclasses

from rl.diffusion_ft.branching import lateral_ease, lateral_offset, lateral_shift, time_warp
from rl.diffusion_ft.lateral_opt import (BRAKE_BACK_M, LateralOptConfig, comfort_operators,
                                         edit_world, expand_offset, make_lateral_optimizer,
                                         variant_cfg)
from rl.diffusion_ft.proposal_select import EMERG_TIER_M, OPT_TIER_M, bank_choice, make_scorer
from rl.diffusion_ft.safety_objective import SafetyObjectiveConfig
from rl.diffusion_ft.scene_sampler import DEFAULT_INDEX_DIR, SceneSampler
from rl.diffusion_ft.tests.test_neighbor_pred import ego_candidate

FAILS: list[str] = []
L24 = (0.1, 0.175, 0.25, 0.325, 0.4, 0.5, 0.6, 0.75, 0.9, 1.1, 1.35, 1.6)
MS_INITS = (0.0, 0.75, -0.75, 1.6, -1.6)
DT = 0.1
TOL = dict(tol_m=1.0, margin_m=0.3, delta_m=0.2)


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def rotate(world, theta):
    """Rotate the whole window about its anchor by ``theta`` (a heading error)."""
    c, s = np.cos(theta), np.sin(theta)
    R = jnp.asarray([[c, -s], [s, c]], jnp.float32)
    p0 = world[:, :1, :2]
    p = (world[..., :2] - p0) @ R.T + p0
    v = world[..., 3:5] @ R.T
    return world.at[..., :2].set(p).at[..., 2].add(theta).at[..., 3:5].set(v)


def distortions(expert, n):
    W = expert.shape[1]
    t = np.arange(W)
    bump = np.where(t <= n, np.sin(np.pi * np.minimum(t, n) / n) ** 2, 0.0)
    out = {}
    for a in (1.5, -1.5):
        out[f"ramp{a:+.1f}"] = lateral_shift(expert, a, ramp_steps=10, dt=DT)
        out[f"bump{a:+.1f}"] = lateral_offset(expert, jnp.asarray(a * bump[None], jnp.float32),
                                              dt=DT)
    for deg in (6.0, -6.0):
        out[f"rot{deg:+.0f}"] = rotate(expert, np.deg2rad(deg))
    out["slow0.5"] = time_warp(expert, 0.5, ramp_steps=10, dt=DT)
    out["fast1.6"] = time_warp(expert, 1.6, ramp_steps=10, dt=DT)
    return out


def edit_kin(d_bn):
    """Peak |lateral accel| and |jerk| OF THE EDIT, from its window profile [B, n]."""
    Ka, Kj = comfort_operators(d_bn.shape[1], DT)
    return np.abs(d_bn @ Ka.T).max(1), np.abs(d_bn @ Kj.T).max(1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits_json", required=True)
    p.add_argument("--shard_glob",
                   default="/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*")
    p.add_argument("--index_dir", default=DEFAULT_INDEX_DIR)
    p.add_argument("--n_scenes", type=int, default=16)
    p.add_argument("--t0s", default="25,50")
    p.add_argument("--n_steps", type=int, default=20)
    p.add_argument("--width", type=int, default=41, help="candidate length W (anchor + W-1)")
    p.add_argument("--neighbors", default="cv", help="log | cv | none (the arms' source)")
    p.add_argument("--opt_cfg", default="", help="LateralOptConfig overrides, k=v,k=v")
    p.add_argument("--verdict", default="")
    p.add_argument("--json_out", default="")
    args = p.parse_args(argv)

    spec = json.loads(Path(args.splits_json).read_text())
    ids = [int(i) for i in spec["splits"]["failure_heldout"]["scene_ids"][:args.n_scenes]]
    sampler = SceneSampler(args.shard_glob, batch_size=1, eval_size=0,
                           include_sdc_paths=False, prefetch=False, index_dir=args.index_dir)
    scen = sampler.load_explicit(ids)
    n_obj = int(np.asarray(scen.log_trajectory.x).shape[1])
    env = waymax_env.PlanningAgentEnvironment(
        dynamics_model=waymax_dynamics.StateDynamics(),
        config=waymax_config.EnvironmentConfig(
            max_num_objects=n_obj, controlled_object=waymax_config.ObjectType.SDC))
    st0 = env.reset(jax.tree_util.tree_map(jnp.asarray, scen))
    B, n, W = len(ids), int(args.n_steps), int(args.width)
    cfg = LateralOptConfig.parse(args.opt_cfg)
    objcfg = SafetyObjectiveConfig()
    score = make_scorer(objcfg, args.neighbors, "median")
    opt = make_lateral_optimizer(objcfg, cfg, dt=DT, neighbor_pred=args.neighbors,
                                 world_dt_method="median")
    cfg_lon = dataclasses.replace(cfg, lon=1)
    opt_lon = make_lateral_optimizer(objcfg, cfg_lon, dt=DT, neighbor_pred=args.neighbors,
                                     world_dt_method="median")
    cfg_am = dataclasses.replace(cfg, lon=1, w_lon_prox=0.05, lon_back_m=1000.0)
    opt_am = make_lateral_optimizer(objcfg, cfg_am, dt=DT, neighbor_pred=args.neighbors,
                                    world_dt_method="median")
    opt_em = {v: make_lateral_optimizer(objcfg, variant_cfg(cfg, v, True), dt=DT,
                                        neighbor_pred=args.neighbors, world_dt_method="median")
              for v in ("emerg", "brake")}
    ease20 = np.asarray(lateral_ease(W, 20, "cubic"))[1:n + 1]
    bank_amps = np.asarray([s * a for a in L24 for s in (1.0, -1.0)])
    print(f"[gate] {B} failure_heldout scenes, t0s {args.t0s}, window {n}, W {W}, "
          f"neighbors {args.neighbors}\n[gate] {cfg}", flush=True)

    is_sdc = np.asarray(scen.object_metadata.is_sdc)
    e = np.argmax(is_sdc, 1)
    lv = np.asarray(scen.log_trajectory.valid)[np.arange(B), e]            # [B, T]
    lxy = np.stack([np.asarray(scen.log_trajectory.x)[np.arange(B), e],
                    np.asarray(scen.log_trajectory.y)[np.arange(B), e]], -1)
    last = lv.shape[1] - 1 - np.argmax(lv[:, ::-1], axis=1)
    goal = jnp.asarray(lxy[np.arange(B), last], jnp.float32)

    arms = ("bank", "opt0", "optms", "union", "opt0lon", "opt0+lon", "full", "argmin")
    stats = {}          # family -> arm -> lists
    id_rows = id_exact = 0
    anchor_err = 0.0
    descent_bad = 0
    nonfinite = 0
    union_missing = 0
    t_opt, t_bank = [], []
    lid_rows = lid_exact = 0
    l_anchor = 0.0
    l_descent_bad = l_nonfinite = 0
    l_back_min = 0.0
    l_union_bad = 0
    l_moved = {"plain": [0, 0], "held": [0, 0]}
    a_id_rows = a_id_exact = a_bad = a_nonfinite = 0
    a_anchor = 0.0
    e_nonfinite = e_bad = 0
    e_anchor = 0.0
    e_back_min = 0.0
    for t0 in [int(s) for s in args.t0s.split(",") if s.strip()]:
        state = datatypes.update_state_by_log(st0, t0 - int(st0.timestep))
        assert int(state.timestep) == t0
        t0j = jnp.asarray(t0, jnp.int32)
        expert = ego_candidate(state, t0, W - 1)
        ok_rows = lv[:, t0:t0 + n + 1].all(1)

        # ---- O1: identity where the base line costs nothing ------------------- #
        x, lb, l0 = opt(expert, jnp.zeros((B, n)), state, t0j, jnp.ones(B), n)
        zero = ok_rows & (np.asarray(l0) == 0.0)
        id_rows += int(zero.sum())
        id_exact += int((np.abs(np.asarray(x))[zero].max(1) == 0.0).sum()) if zero.any() else 0
        xa_, _, l0a = opt_am(expert, jnp.zeros((B, 2 * n)), state, t0j, jnp.ones(B), n)
        za = ok_rows & (np.asarray(l0a) == 0.0)
        a_id_rows += int(za.sum())
        a_id_exact += int((np.abs(np.asarray(xa_))[za].max(1) == 0.0).sum()) if za.any() else 0
        x, lb, l0 = opt_lon(expert, jnp.zeros((B, 2 * n)), state, t0j, jnp.ones(B), n)
        zero = ok_rows & (np.asarray(l0) == 0.0)
        lid_rows += int(zero.sum())
        lid_exact += int((np.abs(np.asarray(x))[zero].max(1) == 0.0).sum()) if zero.any() else 0

        for fam, world in distortions(expert, n).items():
            cp, _, _, pp = (np.asarray(v) for v in score(world, state, t0j, goal, n))
            trig = ok_rows & (cp < 0.0)
            if not trig.any():
                continue
            act = jnp.asarray(trig, jnp.float32)
            s = stats.setdefault(fam, {a: {"fire": [], "mag": [], "acc": [], "jerk": [],
                                           "comes_back": []} for a in arms})
            s.setdefault("n_trig", 0)
            s["n_trig"] += int(trig.sum())

            # ---- the ship bank ---------------------------------------------- #
            tb = time.time()
            bank_sc = [tuple(np.asarray(v) for v in score(
                lateral_shift(world, float(a), ramp_steps=20, dt=DT), state, t0j, goal, n))
                for a in bank_amps]
            t_bank.append(time.time() - tb)
            bank_d = bank_amps[:, None, None] * ease20[None, None, :] * np.ones((1, B, 1))

            # ---- optimized candidates, one per init --------------------------- #
            def run(x0):
                tt = time.time()
                x, lb, l0 = opt(world, jnp.asarray(x0, jnp.float32), state, t0j, act, n)
                x = np.asarray(x)
                t_opt.append(time.time() - tt)
                w = lateral_offset(world, expand_offset(jnp.asarray(x), W), dt=DT)
                return x, np.asarray(lb), np.asarray(l0), w, tuple(
                    np.asarray(v) for v in score(w, state, t0j, goal, n))

            opt_runs = {a: run(a * ease20[None, :] * np.ones((B, 1))) for a in MS_INITS}
            clear_lat = np.stack([c[0] for c in bank_sc])
            best_amp = bank_amps[np.argmax(clear_lat, axis=0)]
            opt_runs["best"] = run(best_amp[:, None] * ease20[None, :])

            # ---- the lon optimizer from 0 (lateral + along-path) --------------- #
            xl, lbl, l0l = opt_lon(world, jnp.zeros((B, 2 * n), jnp.float32), state, t0j, act, n)
            xl = np.asarray(xl)
            wl = edit_world(world, jnp.asarray(xl), n=n, dt=DT, lon=1)
            lon_sc = tuple(np.asarray(v) for v in score(wl, state, t0j, goal, n))
            l_nonfinite += int((~np.isfinite(xl)).any() or (~np.isfinite(np.asarray(lbl)[trig])).any())
            l_descent_bad += int((np.asarray(lbl)[trig] > np.asarray(l0l)[trig] + 1e-5).sum())
            l_anchor = max(l_anchor, float(jnp.max(jnp.abs(wl[:, 0, :2] - world[:, 0, :2]))))
            l_back_min = min(l_back_min, float(xl[trig][:, n:].min()))
            held = world.at[:, -10:, :2].set(world[:, -11:-10, :2])
            _, lbh, l0h = opt_lon(held, jnp.zeros((B, 2 * n), jnp.float32), state, t0j, act, n)
            for key, (a_, b_) in (("plain", (lbl, l0l)), ("held", (lbh, l0h))):
                a_, b_ = np.asarray(a_), np.asarray(b_)
                pos = trig & (b_ > 0)
                l_moved[key][0] += int((a_[pos] < b_[pos]).sum()); l_moved[key][1] += int(pos.sum())

            nonlocal_bad = 0
            for key, (x, lb, l0, w, _) in opt_runs.items():
                nonlocal_bad += int((~np.isfinite(x)).any() or (~np.isfinite(lb[trig])).any())
                descent_bad += int((lb[trig] > l0[trig] + 1e-5).sum())
                anchor_err = max(anchor_err, float(jnp.max(jnp.abs(w[:, 0, :2] - world[:, 0, :2]))))
            nonfinite += nonlocal_bad

            def choose(sc_list, d_list):
                ck = np.stack([c[0] for c in sc_list]); pk = np.stack([c[3] for c in sc_list])
                mag = np.stack([np.abs(d).max(1) + 0.1 for d in d_list])
                ch = bank_choice("cons", clear_p=cp, prog_p=pp, clear_k=ck, prog_k=pk,
                                 magnitude=mag, **TOL)
                d_all = np.stack(d_list)                               # [K, B, n]
                return ch, d_all[np.maximum(ch, 0), np.arange(B)]

            o = lambda k: (opt_runs[k][4], opt_runs[k][0])
            ck = lon_sc[0][None]; pk = lon_sc[3][None]
            mag_l = (np.abs(xl[:, :n]).max(1) + np.abs(xl[:, n:]).max(1) + 0.1)[None]
            ch_l = bank_choice("cons", clear_p=cp, prog_p=pp, clear_k=ck, prog_k=pk,
                               magnitude=mag_l, **TOL)
            x0r = opt_runs[0.0][0]
            ck2 = np.stack([opt_runs[0.0][4][0], lon_sc[0]])
            pk2 = np.stack([opt_runs[0.0][4][3], lon_sc[3]])
            mag2 = np.stack([np.abs(x0r).max(1) + 0.1, mag_l[0] + OPT_TIER_M[(False, True)]])
            ch_u = bank_choice("cons", clear_p=cp, prog_p=pp, clear_k=ck2, prog_k=pk2,
                               magnitude=mag2, **TOL)
            ch_0 = bank_choice("cons", clear_p=cp, prog_p=pp, clear_k=ck2[:1], prog_k=pk2[:1],
                               magnitude=mag2[:1], **TOL)
            l_union_bad += int(((ch_0 >= 0) & (ch_u != 0))[trig].sum())
            d_u = np.where((ch_u == 1)[:, None], xl[:, :n], x0r)
            # ---- emergency tiers, from 0, on the same triggered rows ------------- #
            em_sc, em_d, em_mag = [], [], []
            for v, o_e in opt_em.items():
                act_v = act if v == "emerg" else jnp.asarray(trig & (np.asarray(
                    score(world, state, t0j, goal, n)[1]) >= 0.0), jnp.float32)
                xe, lbe, l0e = o_e(world, jnp.zeros((B, 2 * n), jnp.float32), state, t0j, act_v, n)
                xe = np.asarray(xe)
                we = edit_world(world, jnp.asarray(xe), n=n, dt=DT, lon=1)
                se = tuple(np.asarray(q) for q in score(we, state, t0j, goal, n))
                tv = np.asarray(act_v) > 0.5
                e_nonfinite += int((~np.isfinite(xe)).any() or (~np.isfinite(np.asarray(lbe)[tv])).any())
                e_bad += int((np.asarray(lbe)[tv] > np.asarray(l0e)[tv] + 1e-5).sum())
                e_anchor = max(e_anchor, float(jnp.max(jnp.abs(we[:, 0, :2] - world[:, 0, :2]))))
                if v == "brake" and tv.any():
                    e_back_min = min(e_back_min, float(xe[tv][:, n:].min()))
                # rows not triggered for this tier: it is the base (cannot fire)
                se = tuple(np.where(tv, a_, b_) for a_, b_ in zip(se, (cp, cp, cp, pp)))
                em_sc.append(se); em_d.append(xe[:, :n])
                em_mag.append(np.abs(xe[:, :n]).max(1) + np.abs(xe[:, n:]).max(1) + 0.1 + EMERG_TIER_M[v])
            ckf = np.concatenate([ck2, np.stack([q[0] for q in em_sc])])
            pkf = np.concatenate([pk2, np.stack([q[3] for q in em_sc])])
            ch_f = bank_choice("cons", clear_p=cp, prog_p=pp, clear_k=ckf, prog_k=pkf,
                               magnitude=np.concatenate([mag2, np.stack(em_mag)]),
                               tol_k=[TOL["tol_m"]] * 3 + [TOL["tol_m"] + BRAKE_BACK_M], **TOL)
            d_f = np.stack([x0r, xl[:, :n]] + em_d)[np.maximum(ch_f, 0), np.arange(B)]
            xam, lbam, l0am = opt_am(world, jnp.zeros((B, 2 * n), jnp.float32), state, t0j, act, n)
            xam = np.asarray(xam)
            wam = edit_world(world, jnp.asarray(xam), n=n, dt=DT, lon=1)
            cam = np.asarray(score(wam, state, t0j, goal, n)[0])
            a_nonfinite += int((~np.isfinite(xam)).any())
            a_bad += int((np.asarray(lbam)[trig] > np.asarray(l0am)[trig] + 1e-5).sum())
            a_anchor = max(a_anchor, float(jnp.max(jnp.abs(wam[:, 0, :2] - world[:, 0, :2]))))
            ch_am = np.where(np.any(xam != 0, 1) & (cam >= 0.0), 0, -1)
            picks = {
                "argmin": (ch_am, xam[:, :n]),
                "full": (ch_f, d_f),
                "opt0lon": (ch_l, xl[:, :n]),
                "opt0+lon": (ch_u, d_u),
                "bank": choose(bank_sc, list(bank_d)),
                "opt0": choose([o(0.0)[0]], [o(0.0)[1]]),
                "optms": choose([o(a)[0] for a in MS_INITS], [o(a)[1] for a in MS_INITS]),
                "union": choose(bank_sc + [o("best")[0], o(0.0)[0]],
                                list(bank_d) + [o("best")[1], o(0.0)[1]]),
            }
            union_missing += int(((picks["bank"][0] >= 0) & (picks["union"][0] < 0))[trig].sum())
            for a, (ch, d) in picks.items():
                fire = (ch >= 0) & trig
                acc, jerk = edit_kin(d)
                peak = np.abs(d).max(1)
                s[a]["fire"] += fire[trig].tolist()
                s[a]["mag"] += peak[fire].tolist()
                s[a]["acc"] += acc[fire].tolist()
                s[a]["jerk"] += jerk[fire].tolist()
                s[a]["comes_back"] += (np.abs(d[:, -1]) < 0.5 * peak)[fire].tolist()
        print(f"[gate] t0={t0} done", flush=True)

    # ---- report ---------------------------------------------------------------- #
    pct = lambda v, q: float(np.percentile(v, q)) if len(v) else float("nan")
    summary = {"cfg": cfg.__dict__, "neighbors": args.neighbors, "families": {}}
    pooled = {a: [] for a in arms}
    print(f"\n{'family':>9} {'trig':>5}  " + "  ".join(f"{a:>7}" for a in arms))
    for fam, s in stats.items():
        row = {a: float(np.mean(s[a]["fire"])) for a in arms}
        for a in arms:
            pooled[a] += s[a]["fire"]
        print(f"{fam:>9} {s['n_trig']:>5}  " + "  ".join(f"{row[a]:>7.3f}" for a in arms))
        summary["families"][fam] = {
            "n_trig": s["n_trig"],
            **{a: {"fix_rate": row[a], "peak_d_p50": pct(s[a]["mag"], 50),
                   "acc_p50": pct(s[a]["acc"], 50), "acc_p90": pct(s[a]["acc"], 90),
                   "jerk_p50": pct(s[a]["jerk"], 50), "jerk_p90": pct(s[a]["jerk"], 90),
                   "comes_back": float(np.mean(s[a]["comes_back"])) if s[a]["comes_back"] else None}
               for a in arms}}
    rate = {a: float(np.mean(pooled[a])) if pooled[a] else float("nan") for a in arms}
    summary["pooled_fix_rate"] = rate
    print(f"{'pooled':>9} {len(pooled['bank']):>5}  " + "  ".join(f"{rate[a]:>7.3f}" for a in arms))
    print("\nride quality of the chosen edit (p50 / p90), per arm, pooled over families:")
    for a in arms:
        acc = sum((s[a]["acc"] for s in stats.values()), [])
        jerk = sum((s[a]["jerk"] for s in stats.values()), [])
        back = sum((s[a]["comes_back"] for s in stats.values()), [])
        print(f"  {a:>6}: |a| {pct(acc, 50):.2f}/{pct(acc, 90):.2f} m/s^2   "
              f"|j| {pct(jerk, 50):.1f}/{pct(jerk, 90):.1f} m/s^3   "
              f"comes back {np.mean(back) if back else float('nan'):.2f}")
    summary["sec_per_opt_call"] = float(np.median(t_opt[1:])) if len(t_opt) > 1 else None
    summary["sec_per_bank"] = float(np.median(t_bank[1:])) if len(t_bank) > 1 else None
    print(f"\n[gate] median s/call: optimizer {summary['sec_per_opt_call']}, "
          f"24-candidate bank {summary['sec_per_bank']} (CPU; ratio is what transfers)")

    print()
    check("O1 identity where the base line costs nothing", id_rows > 0 and id_exact == id_rows,
          f"{id_exact}/{id_rows} rows returned exactly d=0")
    check("O2 the optimized candidate keeps the anchor", anchor_err < 1e-4,
          f"max anchor move {anchor_err:.1e} m")
    check("O3 the returned loss never exceeds the init's", descent_bad == 0,
          f"{descent_bad} row-runs worse")
    check("O4 the optimizer from 0 repairs",
          rate["opt0"] >= 0.2 and rate["opt0"] >= 0.5 * rate["bank"],
          f"opt0 {rate['opt0']:.3f} vs bank {rate['bank']:.3f}")
    check("O5 finite", nonfinite == 0, f"{nonfinite} runs with non-finite values")
    check("O6 the union fires wherever the bank fires", union_missing == 0,
          f"{union_missing} rows missing")
    check("L1 lon: identity where the base line costs nothing",
          lid_rows > 0 and lid_exact == lid_rows, f"{lid_exact}/{lid_rows} rows returned exactly 0")
    check("L2 lon: anchor kept, loss never above the init's",
          l_anchor < 1e-4 and l_descent_bad == 0,
          f"max anchor move {l_anchor:.1e} m, {l_descent_bad} row-runs worse")
    check("L3 lon: finite", l_nonfinite == 0, f"{l_nonfinite} runs with non-finite values")
    check("L4 lon: never further back than lon_back_m", l_back_min >= -cfg_lon.lon_back_m - 1e-5,
          f"min s {l_back_min:.3f} m vs -{cfg_lon.lon_back_m}")
    check("L5 opt0+lon runs opt0's own edit wherever opt0 fires", l_union_bad == 0,
          f"{l_union_bad} rows where it did not")
    check("A1 argmin optimizer: identity where the base costs nothing",
          a_id_rows > 0 and a_id_exact == a_id_rows, f"{a_id_exact}/{a_id_rows} rows exactly 0")
    check("A2 argmin optimizer: finite, anchor kept, loss never above the init's",
          a_nonfinite == 0 and a_anchor < 1e-4 and a_bad == 0,
          f"{a_nonfinite} non-finite, anchor {a_anchor:.1e} m, {a_bad} row-runs worse")
    check("E1 emergency optimizers finite", e_nonfinite == 0, f"{e_nonfinite} non-finite runs")
    check("E2 emergency: anchor kept, loss never above the init's", e_anchor < 1e-4 and e_bad == 0,
          f"max anchor move {e_anchor:.1e} m, {e_bad} row-runs worse")
    check("E3 brake never further back than BRAKE_BACK_M", e_back_min >= -BRAKE_BACK_M - 1e-5,
          f"min s {e_back_min:.3f} m vs -{BRAKE_BACK_M}")
    mv = {k: v[0] / max(v[1], 1) for k, v in l_moved.items()}
    check("L6 lon optimizer moves (plain and held-tail windows)",
          all(v[1] > 0 for v in l_moved.values()) and min(mv.values()) >= 0.8,
          " ".join(f"{k} {v[0]}/{v[1]}" for k, v in l_moved.items()))

    head = "PASS" if not FAILS else "FAIL: " + "; ".join(FAILS)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(summary, indent=1))
        print(f"[gate] numbers -> {args.json_out}")
    if args.verdict:
        Path(args.verdict).parent.mkdir(parents=True, exist_ok=True)
        Path(args.verdict).write_text(head + "\n" + json.dumps(rate) + "\n")
        print(f"[gate] verdict -> {args.verdict}")
    if FAILS:
        print(f"[gate] {len(FAILS)} FAILED")
        raise SystemExit(1)
    print("[gate] all lateral-opt gates passed")


if __name__ == "__main__":
    main()
