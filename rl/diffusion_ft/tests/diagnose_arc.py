"""Is the policy driving too fast, or is its path chattering?

The gate reports the proposed window's **arc length** (`27.79 m` over 10 world
steps, i.e. 27.8 m/s) against the expert's `7.47 m`. Read alone that says the
policy drives 3.7x too fast. But the same window's mean deviation from the log is
only `2.87 m` -- and a car genuinely travelling 3.7x too fast is ~20 m past the
log after 1 s, which would show up as ~10 m of drift, not 2.9 m.

Two hypotheses, and they are distinguishable from the executed poses alone:

  A. **Speed.** Net displacement per step is ~3.7x the expert's. Then arc and
     displacement agree, path efficiency (net / arc) stays ~1, and the extra
     length is real progress.
  B. **Chatter.** Net displacement is normal but the path oscillates, so arc
     length counts distance the car never nets. Then efficiency collapses well
     below 1 and consecutive segments point in alternating directions (negative
     mean cosine between them).

This reads `ego_pose_base` / `ego_pose_expert` out of a `safety_probe.npz` -- the
**executed** rollout, not a proposal -- so whatever it finds is what the simulator
actually drove. No model, no GPU.

Usage: python -m rl.diffusion_ft.tests.diagnose_arc --probe_dir DIR
"""

from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np


def segments(pose_bt3, live_bt):
    """Per-step displacement vectors over live steps, [B, T-1, 2] and a mask."""
    xy = np.asarray(pose_bt3)[..., :2]
    d = np.diff(xy, axis=1)
    live = np.asarray(live_bt).astype(bool)
    return d, live[:, 1:] & live[:, :-1]


def stats(name, pose_bt3, live_bt, dt, window):
    d, m = segments(pose_bt3, live_bt)
    step = np.linalg.norm(d, axis=-1)
    speed = step[m] / dt

    # Path efficiency over sliding windows: net displacement / arc length. A
    # straight run is 1.0; a path that doubles back is well under.
    eff, arcs = [], []
    xy = np.asarray(pose_bt3)[..., :2]
    for b in range(xy.shape[0]):
        idx = np.where(np.asarray(live_bt)[b].astype(bool))[0]
        if idx.size <= window:
            continue
        for s in range(idx[0], idx[-1] - window):
            seg = xy[b, s:s + window + 1]
            arc = float(np.linalg.norm(np.diff(seg, axis=0), axis=-1).sum())
            net = float(np.linalg.norm(seg[-1] - seg[0]))
            if arc > 1e-3:
                eff.append(net / arc); arcs.append(arc)
    eff = np.asarray(eff) if eff else np.zeros(1)

    # Turn alternation: cosine between consecutive displacement vectors. Smooth
    # driving is ~+1; a zig-zag alternates and drags the mean toward -1.
    a, b_ = d[:, :-1, :], d[:, 1:, :]
    ok = m[:, :-1] & m[:, 1:]
    na = np.linalg.norm(a, axis=-1) * np.linalg.norm(b_, axis=-1)
    cos = np.where(na > 1e-9, np.sum(a * b_, axis=-1) / np.maximum(na, 1e-9), 1.0)[ok]

    # Heading chatter, straight off the yaw column.
    yaw = np.asarray(pose_bt3)[..., 2]
    dyaw = np.abs(np.diff(np.unwrap(yaw, axis=1), axis=1))[m]

    print(f"\n[arc] === {name} ({speed.size:,} live steps) ===")
    print(f"[arc]   speed m/s      p05={np.percentile(speed,5):.2f} "
          f"p50={np.percentile(speed,50):.2f} p95={np.percentile(speed,95):.2f} "
          f"max={speed.max():.2f}  mean={speed.mean():.2f}")
    print(f"[arc]   path efficiency over {window} steps  "
          f"p05={np.percentile(eff,5):.3f} p50={np.percentile(eff,50):.3f} "
          f"mean={eff.mean():.3f}   (1.0 = straight, <1 = doubles back)")
    print(f"[arc]   cos(consecutive steps)  p05={np.percentile(cos,5):+.3f} "
          f"p50={np.percentile(cos,50):+.3f} mean={cos.mean():+.3f}  "
          f"({float((cos < 0).mean())*100:.2f}% reversals)")
    print(f"[arc]   |dyaw| rad/step p50={np.percentile(dyaw,50):.4f} "
          f"p95={np.percentile(dyaw,95):.4f} max={dyaw.max():.4f}")
    return {"speed_p50": float(np.percentile(speed, 50)),
            "speed_mean": float(speed.mean()),
            "eff_p50": float(np.percentile(eff, 50)),
            "cos_mean": float(cos.mean()),
            "reversal_rate": float((cos < 0).mean()),
            "dyaw_p95": float(np.percentile(dyaw, 95))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe_dir", type=str, required=True)
    p.add_argument("--world_dt", type=float, default=0.1)
    p.add_argument("--window", type=int, default=10)
    args = p.parse_args()

    d = Path(args.probe_dir)
    z = np.load(d / "safety_probe.npz")
    print(f"[arc] {d}\n[arc] keys: {[k for k in z.files if 'pose' in k or 'live' in k]}")

    out = {"policy": stats("policy (executed rollout)", z["ego_pose_base"],
                           z["live_base"], args.world_dt, args.window),
           "expert": stats("expert (log replay)", z["ego_pose_expert"],
                           z["live_expert"], args.world_dt, args.window)}

    pol, exp = out["policy"], out["expert"]
    ratio_speed = pol["speed_p50"] / max(exp["speed_p50"], 1e-6)
    print(f"\n[arc] median speed ratio policy/expert = {ratio_speed:.2f}x")
    if ratio_speed > 2.0 and pol["eff_p50"] > 0.9:
        verdict = ("SPEED: the policy really is driving several times too fast, and "
                   "its path is straight while doing it.")
    elif pol["eff_p50"] < 0.8 and pol["eff_p50"] < exp["eff_p50"] - 0.1:
        verdict = ("CHATTER: net displacement is not the problem -- the path doubles "
                   "back on itself, so arc length counts distance never travelled.")
    elif ratio_speed < 1.5 and pol["eff_p50"] > 0.9:
        verdict = ("NEITHER: the executed rollout is smooth and at expert speed, so "
                   "the long arc is a property of the PROPOSAL, not of what was "
                   "driven -- look at the candidate before the replan truncates it.")
    else:
        verdict = "MIXED: see the numbers above; neither hypothesis is clean."
    print(f"[arc] {verdict}")
    (d / "arc_diagnosis.json").write_text(json.dumps(
        {**out, "speed_ratio": ratio_speed, "verdict": verdict}, indent=2))
    print(f"[arc] wrote {d}/arc_diagnosis.json")


if __name__ == "__main__":
    main()
