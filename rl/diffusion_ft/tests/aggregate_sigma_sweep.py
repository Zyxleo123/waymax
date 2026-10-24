"""Aggregate a `diagnose_drive` sigma-floor sweep into one table and figure.

The question this answers, from
`memory/project_dppo_parity_run_degenerate.md`: DPPO *collects* with the
stochastic policy, and at ``sigma_floor=0.1`` that policy is off-road on the very
first world step 41% of the time (96% of episodes offroad, versus 16% for the
same weights run deterministically). Under the parity reward offroad terminates,
so 98% of worlds absorb in the first replan and the episode degenerates to a
one-step bandit. Lowering sigma is not free either -- the candidate-KL trust
region stops opening (see the s x kl sweep) -- so the useful number is the
*largest* sigma whose stochastic driving is still close to the deterministic
policy's. That is a threshold, and this reads it off.

Input: a sweep root holding one `sigma_*/` subdirectory per point, each written
by `diagnose_drive` (rollouts.jsonl + term_attribution.json + sweep_point.json).
CPU only -- no model, no GPU -- so the table and figure re-render cheaply from
the artifacts whenever the threshold or the plot needs changing.

Two invariants are checked rather than assumed:
  * the deterministic arm must be *identical* across sigma (it never samples), so
    any spread there means sigma is leaking into the deterministic path;
  * a sweep point whose rollouts are missing is reported as missing, never
    silently averaged away.

Usage:
    python -m rl.diffusion_ft.tests.aggregate_sigma_sweep SWEEP_DIR \
        [--tolerance 0.10] [--out_dir DIR]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless compute node
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito, same slots viz_drive uses, so a metric keeps its colour across the
# whole diagnosis package.
C_OFFROAD = "#D55E00"
C_ABSORB = "#0072B2"
C_NSTEP = "#009E73"
C_REF = "#666666"


def _sigma_of(d: Path) -> float | None:
    """Sigma for one sweep point: the file the job wrote, else the dir name."""
    sp = d / "sweep_point.json"
    if sp.exists():
        try:
            return float(json.loads(sp.read_text())["sigma_floor"])
        except (ValueError, KeyError, json.JSONDecodeError):
            pass
    m = re.fullmatch(r"sigma_(\d+)p(\d+)", d.name)
    return float(f"{m.group(1)}.{m.group(2)}") if m else None


def _stats(rows: list[dict], arm: str) -> dict | None:
    """Mean outcome + absorption stats over every rollout of one arm."""
    rr = [r for r in rows if r.get("arm") == arm]
    if not rr:
        return None
    g = lambda k: float(np.mean([r[k] for r in rr if k in r])) if any(k in r for r in rr) else float("nan")
    steps = [r["absorb_step_since_start"] for r in rr
             if r.get("absorb_step_since_start", -1) > 0]
    return {
        "n": len(rr),
        "offroad_ep": g("offroad"), "collided": g("collided"),
        "reached": g("reached"), "solved": g("solved"), "return": g("return"),
        "absorb_at_replan1": g("absorbed_in_replan1"),
        "median_absorb_step": float(np.median(steps)) if steps else float("nan"),
        "mean_n_replans": g("n_replans"),
        # The trainer's own `nstep`: replans collected before ALL worlds absorbed.
        "mean_batch_nstep": g("n_steps_batch"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_dir", type=str)
    p.add_argument("--arm", type=str, default="base")
    p.add_argument("--tolerance", type=float, default=0.10,
                   help="secondary/diagnostic budget: how much worse than "
                        "deterministic the stochastic offroad-episode rate may be")
    p.add_argument("--nstep_frac", type=float, default=0.90,
                   help="PRIMARY criterion: a sigma is usable if the batch still "
                        "collects at least this fraction of the deterministic "
                        "episode length. That, not the offroad rate, is what "
                        "decides whether the trainer gets a learning signal.")
    p.add_argument("--sigma_ref", type=float, default=0.1,
                   help="sigma the trust region was last calibrated at (job 34866)")
    p.add_argument("--target_kl_ref", type=float, default=0.05,
                   help="target_kl that went with --sigma_ref")
    p.add_argument("--out_dir", type=str, default="")
    args = p.parse_args()

    sweep = Path(args.sweep_dir)
    out_dir = Path(args.out_dir) if args.out_dir else sweep
    out_dir.mkdir(parents=True, exist_ok=True)

    points, missing = [], []
    for d in sorted(sweep.glob("sigma_*")):
        if not d.is_dir():
            continue
        sig = _sigma_of(d)
        jl = d / "rollouts.jsonl"
        if sig is None or not jl.exists():
            missing.append(d.name)
            continue
        rows = [json.loads(l) for l in jl.open()]
        det = _stats(rows, f"{args.arm}/det")
        sto = _stats(rows, f"{args.arm}/sto")
        if det is None or sto is None:
            missing.append(d.name)
            continue
        points.append({"sigma": sig, "dir": d.name, "det": det, "sto": sto})
    points.sort(key=lambda x: x["sigma"])

    if missing:
        print(f"[sweep] WARNING: {len(missing)} point(s) without usable rollouts: "
              f"{missing}")
    if not points:
        # Gate, not a failure: say why and exit 0 so nothing downstream is
        # stranded on an unexplained DependencyNeverSatisfied.
        print(f"[sweep] GATE: no usable sweep points under {sweep}. Nothing to "
              f"aggregate -- did the GPU sweep job write any sigma_* dirs?")
        return

    # --- invariant: the deterministic arm never samples ------------------- #
    det_off = np.array([pt["det"]["offroad_ep"] for pt in points])
    spread = float(det_off.max() - det_off.min())
    if spread > 1e-9:
        print(f"[sweep] WARNING: deterministic offroad-ep varies across sigma by "
              f"{spread:.4f} (values {det_off.round(4).tolist()}). The "
              f"deterministic path must not see sigma -- treat the stochastic "
              f"columns below as suspect until that is explained.")
    else:
        print(f"[sweep] deterministic arm invariant across sigma "
              f"(offroad-ep {det_off[0]:.3f}) -- as it must be.")

    det_ref = float(det_off[0])
    budget = det_ref + args.tolerance
    det_nstep = float(points[0]["det"]["mean_batch_nstep"])
    nstep_floor = args.nstep_frac * det_nstep

    # --- table ------------------------------------------------------------ #
    print(f"\n[sweep] === {args.arm}: stochastic driving vs sigma_floor ===")
    print(f"{'sigma':>8} {'offroad-ep':>11} {'collided':>9} {'absorb@1':>9} "
          f"{'med step':>9} {'batch nstep':>12} {'reached':>8} {'solved':>7} {'ret':>8}")
    for pt in points:
        s = pt["sto"]
        print(f"{pt['sigma']:>8.4f} {s['offroad_ep']:>11.3f} {s['collided']:>9.3f} "
              f"{s['absorb_at_replan1']:>9.3f} {s['median_absorb_step']:>9.1f} "
              f"{s['mean_batch_nstep']:>12.2f} {s['reached']:>8.3f} "
              f"{s['solved']:>7.3f} {s['return']:>8.2f}")
    d = points[0]["det"]
    print(f"{'det':>8} {d['offroad_ep']:>11.3f} {d['collided']:>9.3f} "
          f"{d['absorb_at_replan1']:>9.3f} {d['median_absorb_step']:>9.1f} "
          f"{d['mean_batch_nstep']:>12.2f} {d['reached']:>8.3f} "
          f"{d['solved']:>7.3f} {d['return']:>8.2f}   <- sigma-free reference")

    # --- the threshold ---------------------------------------------------- #
    # Primary criterion is the collected episode length, NOT the offroad rate.
    # "Ever offroad" is a per-episode OR: it saturates while the trajectory is
    # still intact, so budgeting on it rejects sigmas that are perfectly usable.
    # What actually decides whether DPPO has anything to learn from is how many
    # replans the batch collects before every world absorbs -- at nstep 8 the
    # trainer sees the whole episode; at nstep 1 it is a one-step bandit.
    usable = [pt for pt in points if pt["sto"]["mean_batch_nstep"] >= nstep_floor]
    clean = [pt for pt in points if pt["sto"]["offroad_ep"] <= budget]
    print(f"\n[sweep] PRIMARY  usable = batch nstep >= {args.nstep_frac:.2f} x "
          f"deterministic {det_nstep:.2f} = {nstep_floor:.2f}")
    print(f"[sweep] SECONDARY clean  = stochastic offroad-ep <= deterministic "
          f"{det_ref:.3f} + tol {args.tolerance:.2f} = {budget:.3f}")
    if usable:
        best = max(usable, key=lambda x: x["sigma"])
        kl = args.target_kl_ref * (args.sigma_ref / best["sigma"]) ** 2
        print(f"[sweep] LARGEST USABLE sigma_floor = {best['sigma']:.4f} "
              f"(batch nstep {best['sto']['mean_batch_nstep']:.2f}, "
              f"offroad-ep {best['sto']['offroad_ep']:.3f}, "
              f"absorb@1 {best['sto']['absorb_at_replan1']:.3f})")
        # The trust region is the other half of the bind, and it is not
        # independent: sigma_logprob_floor is the denominator of the candidate-KL
        # gate, so KL ~ 1/sigma^2. Lowering sigma without raising target_kl is
        # exactly what froze the 2026-08-11 s x kl sweep -- it declined every
        # candidate on epoch 0. Carry the calibration across instead of guessing.
        print(f"[sweep] => at that sigma the gate needs target_kl ~ "
              f"{kl:.3g}  (= {args.target_kl_ref} x ({args.sigma_ref}/"
              f"{best['sigma']:.4g})^2, since candidate KL ~ 1/sigma^2). "
              f"Lowering sigma WITHOUT this is what froze the s x kl sweep.")
        if not any(pt["sigma"] == best["sigma"] for pt in clean):
            print(f"[sweep] note: it does not meet the secondary offroad budget "
                  f"({best['sto']['offroad_ep']:.3f} > {budget:.3f}). The "
                  f"episode survives, but the noisy policy still leaves the road "
                  f"more often than the deterministic one -- expect a noisier "
                  f"gradient, not a truncated one.")
    else:
        best = None
        print(f"[sweep] NO sigma in this sweep keeps the episode intact. Even "
              f"the smallest tried ({points[0]['sigma']:.4f}) collects only "
              f"{points[0]['sto']['mean_batch_nstep']:.2f} of "
              f"{det_nstep:.2f} replans. Either extend the sweep downward, or "
              f"the exploration noise is mis-shaped rather than merely too "
              f"large -- a uniform floor on every diffusion step perturbs the "
              f"whole trajectory, not just its tail.")

    # --- artifacts -------------------------------------------------------- #
    with (out_dir / "sigma_sweep.jsonl").open("w") as f:
        for pt in points:
            f.write(json.dumps(pt) + "\n")

    with (out_dir / "sigma_sweep.md").open("w") as f:
        f.write("# DPPO exploration-noise sweep\n\n")
        f.write(f"Arm `{args.arm}`, parity reward, {points[0]['sto']['n']} "
                f"rollouts per point. The deterministic reference "
                f"(offroad-ep {det_ref:.3f}) does not depend on sigma.\n\n")
        f.write("| sigma | offroad-ep | collided | absorb@1 | med step | "
                "batch nstep | reached | solved | return |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for pt in points:
            s = pt["sto"]
            f.write(f"| {pt['sigma']:.4f} | {s['offroad_ep']:.3f} | "
                    f"{s['collided']:.3f} | {s['absorb_at_replan1']:.3f} | "
                    f"{s['median_absorb_step']:.1f} | {s['mean_batch_nstep']:.2f} | "
                    f"{s['reached']:.3f} | {s['solved']:.3f} | {s['return']:.2f} |\n")
        f.write(f"| **det** | {d['offroad_ep']:.3f} | {d['collided']:.3f} | "
                f"{d['absorb_at_replan1']:.3f} | {d['median_absorb_step']:.1f} | "
                f"{d['mean_batch_nstep']:.2f} | {d['reached']:.3f} | "
                f"{d['solved']:.3f} | {d['return']:.2f} |\n\n")
        f.write(f"Primary criterion: batch nstep >= {args.nstep_frac:.2f} x "
                f"deterministic {det_nstep:.2f} = {nstep_floor:.2f}. "
                f"(Offroad-ep is a per-episode OR and saturates while the "
                f"trajectory is still intact, so it is diagnostic only.)\n\n")
        if best:
            kl = args.target_kl_ref * (args.sigma_ref / best["sigma"]) ** 2
            f.write(f"**Largest usable sigma_floor: {best['sigma']:.4f}** "
                    f"(batch nstep {best['sto']['mean_batch_nstep']:.2f}).\n\n"
                    f"Candidate KL scales as 1/sigma^2, so the trust region must "
                    f"move with it: at this sigma the gate needs "
                    f"**target_kl ~ {kl:.3g}** (= {args.target_kl_ref} x "
                    f"({args.sigma_ref}/{best['sigma']:.4g})^2). Lowering sigma "
                    f"without raising target_kl is what froze the 2026-08-11 "
                    f"s x kl sweep.\n")
        else:
            f.write(f"**No sigma in this sweep keeps the episode intact.**\n")

    sig = np.array([pt["sigma"] for pt in points])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    a = ax[0]
    a.plot(sig, [pt["sto"]["offroad_ep"] for pt in points], "o-",
           color=C_OFFROAD, label="offroad-episode rate")
    a.plot(sig, [pt["sto"]["absorb_at_replan1"] for pt in points], "s-",
           color=C_ABSORB, label="absorbed in replan 1")
    a.axhline(det_ref, color=C_REF, ls="--", lw=1,
              label=f"deterministic offroad-ep ({det_ref:.2f})")
    if best:
        a.axvline(best["sigma"], color=C_NSTEP, ls=":", lw=1.5,
                  label=f"largest usable ({best['sigma']:.3f})")
    a.set_xscale("log"); a.set_xlabel("sigma_floor"); a.set_ylim(-0.02, 1.02)
    a.set_ylabel("fraction of rollouts")
    a.set_title("Exploration noise destroys the behaviour policy")
    a.legend(fontsize=8, loc="center left"); a.grid(alpha=0.25)

    b = ax[1]
    b.plot(sig, [pt["sto"]["mean_batch_nstep"] for pt in points], "o-",
           color=C_NSTEP, label="replans collected (batch)")
    b.axhline(det_nstep, color=C_REF, ls="--", lw=1,
              label=f"deterministic ({det_nstep:.1f})")
    b.axhspan(nstep_floor, det_nstep * 1.05, color=C_NSTEP, alpha=0.10)
    if best:
        b.axvline(best["sigma"], color=C_NSTEP, ls=":", lw=1.5,
                  label=f"largest usable ({best['sigma']:.3f})")
    b.axhline(1.0, color=C_OFFROAD, ls=":", lw=1.5,
              label="one-step bandit (job 34866)")
    b.set_xscale("log"); b.set_xlabel("sigma_floor")
    b.set_ylabel("mean replan steps per episode")
    b.set_title("How much trajectory the trainer actually sees")
    b.legend(fontsize=8); b.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "sigma_sweep.png", dpi=150)
    print(f"\n[sweep] wrote {out_dir}/sigma_sweep.{{jsonl,md,png}}")


if __name__ == "__main__":
    main()
