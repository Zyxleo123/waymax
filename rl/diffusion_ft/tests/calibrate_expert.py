"""What does the safety loss charge the *human demonstration*?

This is the question that decides whether the loss is usable at all, and it is
prior to every other question about it. The safety term will be trained against
an objective that pulls the policy *toward* the expert (a BC/denoising anchor,
an expert minibatch, a KL to the base policy). If the expert itself pays a large
safety bill, those two objectives have different minima and the policy settles
somewhere neither wants -- and the safety weight silently becomes a knob that
trades away imitation quality. A safety term the expert pays nothing for is free
regularization; one the expert pays heavily for is a "drive unlike humans" term
wearing a safety label.

Two failure modes are separated here because they need opposite fixes:

  * **Tight-but-legal driving.** The human holds a small clearance on purpose --
    lane discipline, not precision limits. On scenario 52 the measured margin is
    0.30 m to a road-edge median with ~3 m of open road on the other side. A
    0.5 m barrier margin charges that at 79% of full gradient strength. The fix
    is calibration: the margin has to sit below the clearance humans actually
    keep, which is what the percentile table below reports.

  * **Dirty ground truth.** On some scenes the logged human is itself off-road
    (12 in the curated failure set, the worst at +101.75 m -- a corrupt record
    rather than bad driving). Against an *unbounded* barrier one such scene is
    ~37,000 of loss while a clean one is ~1e-6, so it is 100% of a batch's
    gradient. The fix is not to bound the barrier -- unbounded is right for the
    policy, which must keep being pushed out of deep violation -- but to gate
    these scenes out. They are reported separately here and never pooled into
    the calibration statistics.

Deliberately does NOT construct DiffusionWaymaxEnv or load the diffusion
checkpoint: the human trajectory is already in ``log_trajectory``, and every
other agent is log-replayed too, so this is pure geometry. That makes it a
minutes-long CPU job instead of something queued behind the GPU probe.

Usage:
    python -m rl.diffusion_ft.tests.calibrate_expert \
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

import jax
import jax.numpy as jnp
import numpy as np

from rl.diffusion_ft.safety_loss import (
    SafetyLossConfig, barrier, best_disc_count, ego_box_5dof, log_geometry,
    safety_terms,
)
from rl.diffusion_ft.train_dppo import load_batch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tfrecord", type=str,
                   default="/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord")
    p.add_argument("--n_scenes", type=int, default=64)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--chunk", type=int, default=8,
                   help="scenes loaded/scored at once (memory knob only)")
    p.add_argument("--init_steps", type=int, default=11,
                   help="Waymax replays the log below this, so no policy is ever "
                        "responsible for it; those steps are excluded.")
    p.add_argument("--edge_margin_m", type=float, default=0.3)
    p.add_argument("--edge_tau_m", type=float, default=0.15)
    p.add_argument("--agent_margin_m", type=float, default=0.75)
    p.add_argument("--agent_tau_m", type=float, default=0.25)
    p.add_argument("--n_discs", type=int, default=5,
                   help="cap; the cover is sized per object from its L/W "
                        "(disc geometry only)")
    p.add_argument("--agent_geometry", type=str, default="exact",
                   choices=("exact", "disc"),
                   help="how the ego/other gap is measured. The margins below "
                        "were calibrated under 'disc', whose cover overhang "
                        "costs ~1 m per pair of cars; re-run this under 'exact' "
                        "to recalibrate agent_margin_m against real geometry.")
    p.add_argument("--edge_geometry", type=str, default="point",
                   choices=("point", "segment"),
                   help="road-edge distance: to the nearest SAMPLE (waymax, and "
                        "every margin on record) or to the polyline SEGMENT. "
                        "The sign, and so the dirty/clean split, is identical.")
    p.add_argument("--time_chunk", type=int, default=4)
    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SafetyLossConfig(
        edge_margin_m=args.edge_margin_m, edge_tau_m=args.edge_tau_m,
        agent_margin_m=args.agent_margin_m, agent_tau_m=args.agent_tau_m,
        n_discs=args.n_discs, time_chunk=args.time_chunk,
        agent_geometry=args.agent_geometry, edge_geometry=args.edge_geometry)
    print(f"[cal] {args.n_scenes} scenes from {Path(args.tfrecord).name}, "
          f"chunk={args.chunk}, devices={jax.devices()}")
    print(f"[cal] {cfg}\n")

    all_ce, all_ca, scene_rows = [], [], []
    # Does the disc cap bind? A negative expert agent clearance has two very
    # different explanations -- the cover being too coarse for long vehicles
    # (fixable by raising the cap), or genuinely close traffic in the log
    # (not a cover problem at all, and a reason to lower the agent margin rather
    # than refine the geometry). The aspect ratios say which.
    aspect: list[np.ndarray] = []
    idx_all = list(range(args.start, args.start + args.n_scenes))

    for c0 in range(0, len(idx_all), args.chunk):
        idx = idx_all[c0:c0 + args.chunk]
        state = load_batch(args.tfrecord, idx)
        geo = log_geometry(state)
        box = ego_box_5dof(geo["ego_pose"], geo["ego_len"][:, None], geo["ego_wid"][:, None])
        # Only steps the policy would ever be responsible for, and only where the
        # ego log is actually valid.
        valid = np.array(geo["ego_valid"])   # copy: np.asarray on a JAX array is read-only
        valid[:, :args.init_steps] = False
        t = safety_terms(
            cfg, ego_box_bt5=box, ego_z_bt=geo["ego_z"],
            roadgraph_points=state.roadgraph_points,
            other_box_btn5=geo["other_box"], other_valid_btn=geo["other_valid"],
            step_valid_bt=jnp.asarray(valid, jnp.float32))
        ce, ca = np.asarray(t["c_edge"]), np.asarray(t["c_agent"])
        ob = np.asarray(geo["other_box"]); ov = np.asarray(geo["other_valid"])
        sel = ov & (ob[..., 2] > 0.5)                 # real, sized objects
        aspect.append((ob[..., 2] / np.maximum(ob[..., 3], 1e-3))[sel])
        for i, sc in enumerate(idx):
            m = valid[i]
            if not m.any():
                continue
            cei, cai = ce[i][m], ca[i][m]
            fin = cai[np.isfinite(cai)]
            dirty = bool((cei < 0).any())
            scene_rows.append({
                "scene": int(sc), "n_steps": int(m.sum()),
                "min_c_edge": float(cei.min()), "p05_c_edge": float(np.percentile(cei, 5)),
                "median_c_edge": float(np.median(cei)),
                "min_c_agent": (float(fin.min()) if fin.size else None),
                "expert_dirty": dirty,
                "n_offroad_steps": int((cei < 0).sum()),
                "loss": float(np.asarray(t["loss"])[i][m].sum()),
            })
            if not dirty:                      # dirty scenes never pool in
                all_ce.append(cei); all_ca.append(fin)
        print(f"[cal] scenes {idx[0]}..{idx[-1]}  "
              f"dirty={sum(r['expert_dirty'] for r in scene_rows[-len(idx):])}/{len(idx)}")

    dirty = [r for r in scene_rows if r["expert_dirty"]]
    clean = [r for r in scene_rows if not r["expert_dirty"]]
    ce = np.concatenate(all_ce) if all_ce else np.zeros(0)
    ca = np.concatenate(all_ca) if all_ca else np.zeros(0)

    print(f"\n[cal] === expert-DIRTY scenes (log itself off-road) ===")
    print(f"[cal] {len(dirty)}/{len(scene_rows)} scenes = "
          f"{100*len(dirty)/max(len(scene_rows),1):.1f}%. Excluded from calibration.")
    for r in sorted(dirty, key=lambda r: r["min_c_edge"])[:8]:
        print(f"[cal]   scene {r['scene']:>4}  min c_edge {r['min_c_edge']:>10.2f} m  "
              f"{r['n_offroad_steps']:>3} offroad steps  loss {r['loss']:>12,.0f}")

    if ce.size == 0:
        print("\n[cal] no expert-clean steps; nothing to calibrate."); return

    print(f"\n[cal] === expert-CLEAN clearance distribution ({ce.size} steps, "
          f"{len(clean)} scenes) ===")
    qs = [0.1, 1, 5, 10, 25, 50]
    print("[cal]   road edge  " + "  ".join(
        f"p{q:g}={np.percentile(ce, q):+.2f}" for q in qs))
    if ca.size:
        print("[cal]   agents     " + "  ".join(
            f"p{q:g}={np.percentile(ca, q):+.2f}" for q in qs))

    # Only meaningful under the disc cover: with exact geometry there is no
    # cover to overhang, so a negative expert clearance is real close traffic.
    if aspect and cfg.agent_geometry == "disc":
        ar = np.concatenate(aspect)
        want = np.asarray(best_disc_count(
            jnp.asarray(ar, jnp.float32), jnp.ones_like(jnp.asarray(ar, jnp.float32)),
            32))
        binds = float((want >= cfg.n_discs).mean())
        print(f"\n[cal] === does the disc cap bind? ===")
        print(f"[cal] object L/W over {ar.size:,} object-steps: "
              f"p50={np.percentile(ar,50):.1f}  p95={np.percentile(ar,95):.1f}  "
              f"p99={np.percentile(ar,99):.1f}  max={ar.max():.1f}")
        print(f"[cal] {binds*100:.2f}% of objects want K >= the cap of {cfg.n_discs}"
              f" (optimum K uncapped: p99={np.percentile(want,99):.0f}, "
              f"max={want.max()})")
        print(f"[cal] -> {'RAISE the cap' if binds > 0.02 else 'cap is NOT the limit'}"
              f"; a negative expert agent clearance is "
              f"{'partly cover artifact' if binds > 0.02 else 'real close traffic in the log'}.")

    print(f"\n[cal] === margin decision table (expert-clean only) ===")
    print(f"[cal] {'margin':>8} {'mean loss/step':>15} {'% steps charged':>16} "
          f"{'mean grad strength':>19}")
    table = []
    for m in (0.75, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05):
        b = np.asarray(barrier(jnp.asarray(ce), m, cfg.edge_tau_m))
        # Stable logistic: (m - c)/tau reaches ~700 on a deeply-clear step and
        # the naive form overflows in exp().
        z = (m - ce) / cfg.edge_tau_m
        g = np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))),
                     np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))
        row = {"margin": m, "mean_loss_per_step": float(b.mean()),
               "frac_charged": float((b > 0.01).mean()), "mean_grad": float(g.mean())}
        table.append(row)
        print(f"[cal] {m:>8.2f} {row['mean_loss_per_step']:>15.4f} "
              f"{row['frac_charged']*100:>15.1f}% {row['mean_grad']*100:>18.1f}%")

    p01, p05 = float(np.percentile(ce, 1)), float(np.percentile(ce, 5))
    cur = min(table, key=lambda r: abs(r["margin"] - cfg.edge_margin_m))
    print(f"\n[cal] Expert-clean road-edge clearance: p01 {p01:+.2f} m, p05 {p05:+.2f} m.")
    print(f"[cal] The current {cfg.edge_margin_m:.2f} m margin charges "
          f"{cur['frac_charged']*100:.1f}% of expert steps at a mean gradient "
          f"strength of {cur['mean_grad']*100:.1f}%.")
    print(f"[cal] A margin at or below p01 ({p01:+.2f} m) leaves 99% of expert steps "
          f"on the plateau -- that is the number to set it from, not intuition.")

    summary = {"config": cfg.__dict__, "tfrecord": args.tfrecord,
               "n_scenes": len(scene_rows), "n_dirty": len(dirty), "n_clean": len(clean),
               "clean_steps": int(ce.size),
               "edge_percentiles": {f"p{q:g}": float(np.percentile(ce, q)) for q in qs},
               "agent_percentiles": ({f"p{q:g}": float(np.percentile(ca, q)) for q in qs}
                                     if ca.size else None),
               "margin_table": table,
               "dirty_scenes": [r["scene"] for r in dirty]}
    (out_dir / "expert_calibration.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "expert_calibration.jsonl", "w") as f:
        for r in scene_rows:
            f.write(json.dumps(r) + "\n")
    np.savez_compressed(out_dir / "expert_clearances.npz",
                        c_edge_clean=ce.astype(np.float32),
                        c_agent_clean=ca.astype(np.float32))
    print(f"\n[cal] wrote {out_dir}/expert_calibration.json, .jsonl and "
          f"expert_clearances.npz")


if __name__ == "__main__":
    main()
