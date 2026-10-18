"""Does the safety loss fire on scenes where nothing is wrong?

``calibrate_expert.py`` asked what the loss charges the human demonstration and
answered it with *clearance percentiles* -- how close the human gets. This asks
the complementary question, which percentiles cannot see: how much of the loss
comes from geometry that is **already clear of the margin**, i.e. from terms
that the barrier was supposed to have switched off.

The mechanism, and why it is not a rounding error. The barrier is::

    tau * softplus((margin - c) / tau)

whose support is infinite: at ``c = margin + k*tau`` it is still
``~tau*exp(-k)``, and its gradient is still ``~exp(-k)``. Per term that is
negligible -- but the barrier is **summed**, deliberately, over 4 ego corners,
over every valid other agent, and over every step of the episode. A 91-step
scene with 40 agents sums 4*91 edge terms and 40*91 agent terms, so a per-term
epsilon is multiplied by a factor that grows with *how crowded and how long the
scene is*, not with anything the policy did. An empty straight road and a dense
but perfectly safe queue of traffic are then not scored the same, and the
difference is pure density.

So this probe splits every scene's loss two ways, exactly:

``bite``   the part contributed by corners/pairs actually inside the margin
           (``c < margin``) -- what the term is for.
``tail``   the part contributed by corners/pairs at or beyond the margin
           (``c >= margin``) -- the misfire, zero by construction under a
           compact-support barrier.

and reports the same split for the *gradient* mass, because that is what
actually moves the policy: a tail that is 30% of the loss but 3% of the
gradient is cosmetic, and a tail that is 30% of the gradient is a second,
unintended objective ("stay away from everything, always") riding along.

Scenes are then bucketed by whether anything in them is even close:

``trivial``    the expert log never comes within the margin of a road edge or
               of another agent, at any step. The loss on these should be 0.
               Whatever it is, is misfire, with no legitimate part to net off.
``bites``      something is inside the margin at some step.
``dirty``      the log is itself off-road -- excluded, as everywhere else.

Finally the same clearances are re-scored under every barrier shape
(:func:`rl.diffusion_ft.safety_loss.barrier`) so the fix can be read off the
same numbers: ``hinge2`` is the quadratic-smoothed hinge, C^1 like the softplus
but exactly zero with exactly zero gradient past the margin; ``hinge`` is the
unsmoothed reference.

Pure geometry over ``log_trajectory`` -- no env, no checkpoint, no GPU, same as
calibrate_expert.

Usage:
    python -m rl.diffusion_ft.tests.probe_barrier_tail \
        --tfrecord .../failures.tfrecord --n_scenes 64 --out_dir OUT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
import numpy as np

from rl.diffusion_ft.safety_loss import (
    SafetyLossConfig, agent_disc_gaps, agent_exact_gaps, barrier, ego_box_5dof,
    log_geometry, road_edge_clearance,
)
from rl.diffusion_ft.train_dppo import load_batch

SHAPES = ("softplus", "hinge2", "hinge")


def split_cost(c: np.ndarray, margin: float, tau: float, shape: str) -> dict:
    """Barrier value and |gradient| on a flat array of clearances, split at the margin.

    The gradient reported is ``|d barrier / d c|``, which is what scales the
    push on the trajectory; its sum over terms is the quantity a per-term
    epsilon inflates.
    """
    cj = jnp.asarray(c, jnp.float32)
    val = np.asarray(barrier(cj, margin, tau, shape), np.float64)
    # d/dc of tau*f((m-c)/tau) = -f'((m-c)/tau); all three shapes have a
    # closed-form derivative, so no autodiff pass is needed over ~1e7 terms.
    z = (margin - c) / tau
    if shape == "softplus":                       # sigmoid(z), stable form
        g = np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))),
                     np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))
    elif shape == "hinge2":
        g = np.clip(z, 0.0, 1.0)
    else:
        g = (z > 0).astype(np.float64)
    inside = c < margin
    return {
        "loss": float(val.sum()),
        "loss_bite": float(val[inside].sum()),
        "loss_tail": float(val[~inside].sum()),
        "grad": float(g.sum()),
        "grad_bite": float(g[inside].sum()),
        "grad_tail": float(g[~inside].sum()),
        "n_terms": int(c.size),
        "n_bite": int(inside.sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tfrecord", type=str,
                   default="/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord")
    p.add_argument("--n_scenes", type=int, default=64)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--init_steps", type=int, default=11)
    p.add_argument("--edge_margin_m", type=float, default=0.3)
    p.add_argument("--edge_tau_m", type=float, default=0.15)
    p.add_argument("--agent_margin_m", type=float, default=0.75)
    p.add_argument("--agent_tau_m", type=float, default=0.25)
    p.add_argument("--agent_geometry", type=str, default="exact",
                   choices=("exact", "disc"))
    p.add_argument("--n_discs", type=int, default=5)
    p.add_argument("--time_chunk", type=int, default=4)
    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SafetyLossConfig(
        edge_margin_m=args.edge_margin_m, edge_tau_m=args.edge_tau_m,
        agent_margin_m=args.agent_margin_m, agent_tau_m=args.agent_tau_m,
        n_discs=args.n_discs, time_chunk=args.time_chunk,
        agent_geometry=args.agent_geometry)
    print(f"[tail] {args.n_scenes} scenes from {Path(args.tfrecord).name}")
    print(f"[tail] edge margin {cfg.edge_margin_m} tau {cfg.edge_tau_m} | "
          f"agent margin {cfg.agent_margin_m} tau {cfg.agent_tau_m} "
          f"({cfg.agent_geometry})\n")

    rows = []
    idx_all = list(range(args.start, args.start + args.n_scenes))
    for c0 in range(0, len(idx_all), args.chunk):
        idx = idx_all[c0:c0 + args.chunk]
        state = load_batch(args.tfrecord, idx)
        geo = log_geometry(state)
        box = ego_box_5dof(geo["ego_pose"], geo["ego_len"][:, None],
                           geo["ego_wid"][:, None])
        valid = np.array(geo["ego_valid"])
        valid[:, :args.init_steps] = False

        c_edge_corners, _ = road_edge_clearance(
            box, geo["ego_z"], state.roadgraph_points,
            z_stretch=cfg.z_stretch, time_chunk=cfg.time_chunk,
            edge_geometry=cfg.edge_geometry)                      # [B,T,4]
        if cfg.agent_geometry == "exact":
            gaps, pair_valid = agent_exact_gaps(
                box, geo["other_box"], geo["other_valid"])
        else:
            gaps, pair_valid = agent_disc_gaps(
                box, geo["other_box"], geo["other_valid"],
                n_discs=cfg.n_discs, adaptive=cfg.adaptive_discs)
        c_edge_corners = np.asarray(c_edge_corners, np.float64)
        gaps = np.asarray(gaps, np.float64)
        pair_valid = np.asarray(pair_valid)
        n_agents = np.asarray(geo["other_valid"])                 # [B,T,N]

        for i, sc in enumerate(idx):
            m = valid[i]
            if not m.any():
                continue
            ce = c_edge_corners[i][m]                             # [t,4]
            gp = gaps[i][m][pair_valid[i][m]]                     # [P]
            dirty = bool((ce < 0).any())
            row = {
                "scene": int(sc), "n_steps": int(m.sum()),
                "mean_agents": float(n_agents[i][m].sum(-1).mean()),
                "n_edge_terms": int(ce.size), "n_agent_terms": int(gp.size),
                "min_c_edge": float(ce.min()),
                "min_c_agent": (float(gp.min()) if gp.size else None),
                "expert_dirty": dirty,
            }
            for sh in SHAPES:
                e = split_cost(ce.ravel(), cfg.edge_margin_m, cfg.edge_tau_m, sh)
                a = split_cost(gp, cfg.agent_margin_m, cfg.agent_tau_m, sh)
                row[sh] = {"edge": e, "agent": a,
                           "loss": e["loss"] + a["loss"],
                           "loss_tail": e["loss_tail"] + a["loss_tail"],
                           "grad": e["grad"] + a["grad"],
                           "grad_tail": e["grad_tail"] + a["grad_tail"]}
            # "Trivial" is about the *geometry*, not about the shape: nothing in
            # this scene is inside either margin at any step.
            row["trivial"] = bool(
                (not dirty)
                and (ce >= cfg.edge_margin_m).all()
                and (gp.size == 0 or (gp >= cfg.agent_margin_m).all()))
            rows.append(row)
        print(f"[tail] scenes {idx[0]}..{idx[-1]}  "
              f"trivial={sum(r['trivial'] for r in rows[-len(idx):])}/{len(idx)}")

    dirty = [r for r in rows if r["expert_dirty"]]
    live = [r for r in rows if not r["expert_dirty"]]
    trivial = [r for r in live if r["trivial"]]
    bites = [r for r in live if not r["trivial"]]
    print(f"\n[tail] {len(rows)} scenes: {len(trivial)} trivial, {len(bites)} "
          f"with something inside a margin, {len(dirty)} expert-dirty (excluded)")

    def agg(rs, sh, key):
        return sum(r[sh][key] for r in rs)

    print(f"\n[tail] === where the loss comes from (expert-clean scenes, "
          f"softplus) ===")
    print(f"[tail] {'bucket':>10} {'scenes':>7} {'loss':>12} {'tail':>12} "
          f"{'tail%':>7} {'grad':>12} {'grad tail%':>11}")
    for name, rs in (("trivial", trivial), ("bites", bites), ("all", live)):
        if not rs:
            continue
        L, Lt = agg(rs, "softplus", "loss"), agg(rs, "softplus", "loss_tail")
        G, Gt = agg(rs, "softplus", "grad"), agg(rs, "softplus", "grad_tail")
        print(f"[tail] {name:>10} {len(rs):>7} {L:>12.2f} {Lt:>12.2f} "
              f"{100*Lt/max(L,1e-9):>6.1f}% {G:>12.2f} "
              f"{100*Gt/max(G,1e-9):>10.1f}%")

    if trivial:
        pl = np.array([r["softplus"]["loss"] for r in trivial])
        pe = np.array([r["softplus"]["edge"]["loss"] for r in trivial])
        pa = np.array([r["softplus"]["agent"]["loss"] for r in trivial])
        na = np.array([r["mean_agents"] for r in trivial])
        print(f"\n[tail] === per-scene loss on scenes where NOTHING is inside a "
              f"margin ===")
        print(f"[tail] softplus: mean {pl.mean():.3f}  median "
              f"{np.median(pl):.3f}  p90 {np.percentile(pl,90):.3f}  "
              f"max {pl.max():.3f}   (edge {pe.mean():.3f} + agent "
              f"{pa.mean():.3f})")
        for sh in ("hinge2", "hinge"):
            q = np.array([r[sh]["loss"] for r in trivial])
            print(f"[tail] {sh:>8}: mean {q.mean():.3g}  max {q.max():.3g}")
        if na.size > 2 and na.std() > 0 and pl.std() > 0:
            r = float(np.corrcoef(na, pl)[0, 1])
            print(f"[tail] corr(mean # agents in scene, softplus loss) = {r:+.3f}"
                  f"  <- a safe scene is charged for its traffic density")

    if bites:
        print(f"\n[tail] === does dropping the tail change what the term "
              f"*wants*? (scenes with a real bite) ===")
        sp = np.array([r["softplus"]["loss"] for r in bites])
        h2 = np.array([r["hinge2"]["loss"] for r in bites])
        hi = np.array([r["hinge"]["loss"] for r in bites])
        print(f"[tail] total loss  softplus {sp.sum():.2f}  hinge2 "
              f"{h2.sum():.2f}  hinge {hi.sum():.2f}")
        if sp.std() > 0 and h2.std() > 0:
            rank = float(np.corrcoef(np.argsort(np.argsort(sp)),
                                     np.argsort(np.argsort(h2)))[0, 1])
            print(f"[tail] scene-ranking agreement softplus vs hinge2: "
                  f"Spearman {rank:+.3f}  (1.0 = the two orders the same scenes "
                  f"identically, so the tail only adds an offset)")

    summary = {"config": cfg.__dict__, "tfrecord": args.tfrecord,
               "n_scenes": len(rows), "n_trivial": len(trivial),
               "n_bites": len(bites), "n_dirty": len(dirty),
               "totals": {sh: {"loss": agg(live, sh, "loss"),
                               "loss_tail": agg(live, sh, "loss_tail"),
                               "grad": agg(live, sh, "grad"),
                               "grad_tail": agg(live, sh, "grad_tail")}
                          for sh in SHAPES},
               "trivial_loss": {sh: [r[sh]["loss"] for r in trivial]
                                for sh in SHAPES}}
    (out_dir / "barrier_tail.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "barrier_tail.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n[tail] wrote {out_dir}/barrier_tail.json and .jsonl")


if __name__ == "__main__":
    main()
