"""Ride quality of the executed repairs: does the car jerk when the filter fires?

`solved` is reached & !offroad & !overlap -- it says nothing about HOW a scene was
recovered, and the env executes through ``StateDynamics``, which teleports the ego
to whatever states it is handed. Nothing downstream rejects a kinematically
absurd edit, so comfort has to be measured explicitly. This reads the executed
states `proposal_select.py --dump_traj` writes and reports, per arm:

* peak lateral acceleration, lateral jerk and yaw rate per episode, for the ARM
  and for the base counterfactual over the same steps (paired, so the question
  is what the FILTER added, not what the base was already doing);
* the same restricted to episodes where the filter actually fired;
* per-scene figures for the most extreme episodes.

Everything is computed in the path frame: lateral is the component along the left
normal ``(-sin yaw, cos yaw)`` of the vehicle heading, so a lateral acceleration
here is what a passenger feels sideways.

The summary JSON is the artefact; the figures redraw from the npz on disk, so
neither needs a GPU or a re-roll.

Usage (CPU):
  python -m rl.diffusion_ft.tools.viz_repair --select_dir <dir> [--split failure_heldout]
  python -m rl.diffusion_ft.tools.viz_repair --select_dir <dir> --top 8 --arm b_l24
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

# Driving-comfort convention, m/s^2 and m/s^3. Rough but standard enough to band by.
ACC_BANDS = ((2.0, "<2 calm"), (4.0, "2-4 brisk"), (6.0, "4-6 aggr"),
             (9.0, "6-9 evasive"), (float("inf"), ">9 limit"))
# NOTE on absolute comfort thresholds: the commonly cited ~2 m/s^3 ride-comfort
# ceiling is for sustained motion in a real car and is NOT a usable yardstick
# here -- these are 0.1 s-sampled sim trajectories whose BASE already sits at
# 15-40 m/s^3 after smoothing, so every episode "exceeds" it and the statistic
# carries no information. What is meaningful is the PAIRED comparison: at the
# replans where the filter fires, is the repair gentler than the base trajectory
# it replaced? Kept only as the reference line drawn on the figures.
JERK_REF = 2.0


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def kinematics(traj_rt5: np.ndarray, dt: float, smooth_w: int = 9) -> dict:
    """[R, T, 5] executed states (NaN-padded) -> path-frame comfort series.

    Replans are concatenated in time: the car drives replan 0's executed slice,
    then replan 1's, so the joins are exactly where a new edit can start and are
    the places a discontinuity would show up. (Measured: joins are continuous --
    the gap across one is the same as an ordinary per-step travel -- so they are
    not an artefact source.)

    DIFFERENTIATE WITH A LOCAL POLYNOMIAL FIT, NOT RAW FINITE DIFFERENCES. The
    diffusion sampler leaves ~4 mm of per-step position noise, and at dt=0.1 a
    plain second difference turns that into ~1.5 m/s^2 and a third difference
    into ~30 m/s^3 -- which is the whole of what raw differencing reported, on
    the BASE arm, with no edit applied at all. Savitzky-Golay (cubic over
    ``smooth_w`` samples) estimates the derivatives of the underlying path
    instead, which is what a passenger feels; the vehicle cannot follow 4 mm
    jitter anyway. ``smooth_w=0`` restores raw differencing for comparison.

    Yaw rate is a FIRST derivative and so is the one channel raw differencing
    already resolved; it is reported either way and the two should agree.
    """
    flat = traj_rt5.reshape(-1, traj_rt5.shape[-1])
    ok = np.isfinite(flat[:, 0])
    flat = flat[ok]
    if len(flat) < 4:
        return {}
    xy, yaw = flat[:, :2], np.unwrap(flat[:, 2])
    w = int(smooth_w)
    if w >= 5 and len(flat) >= w:
        from scipy.signal import savgol_filter
        w = min(w if w % 2 else w + 1, len(flat) - (1 - len(flat) % 2))
        sg = lambda y, d: savgol_filter(y, w, 3, deriv=d, delta=dt, axis=0)
        v, a, j = sg(xy, 1), sg(xy, 2), sg(xy, 3)
        yaw_rate = savgol_filter(yaw, w, 3, deriv=1, delta=dt)
    else:
        v = np.gradient(xy, dt, axis=0)                   # [T, 2]
        a = np.gradient(v, dt, axis=0)
        j = np.gradient(a, dt, axis=0)
        yaw_rate = np.gradient(yaw, dt)
    # Left normal of the heading; lateral = component along it.
    n = np.stack([-np.sin(yaw), np.cos(yaw)], axis=-1)
    tg = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)
    return {"speed": np.linalg.norm(v, axis=-1),
            "a_lat": (a * n).sum(-1), "a_lon": (a * tg).sum(-1),
            "j_lat": (j * n).sum(-1), "j_lon": (j * tg).sum(-1),
            "yaw_rate": yaw_rate,
            "xy": xy, "yaw": yaw, "n_steps": len(flat)}


FIELDS = ("peak_a_lat", "peak_j_lat", "peak_yaw_rate", "peak_a_lon", "peak_j_lon")


def _peaks(k: dict) -> dict:
    if not k:
        return {}
    return {"peak_a_lat": float(np.abs(k["a_lat"]).max()),
            "peak_j_lat": float(np.abs(k["j_lat"]).max()),
            # Longitudinal too (2026-09-30): an along-path repair (lateral_opt lon)
            # speeds up or brakes, which the lateral channels cannot see.
            "peak_a_lon": float(np.abs(k["a_lon"]).max()),
            "peak_j_lon": float(np.abs(k["j_lon"]).max()),
            "peak_yaw_rate": float(np.abs(k["yaw_rate"]).max()),
            "mean_speed": float(np.nanmean(k["speed"]))}


def collect(select_dir: Path, split: str, arms: set[str], dt: float,
            smooth_w: int = 9):
    """Yield one record per episode that has a dumped trajectory."""
    rows = {}
    for p in sorted(select_dir.glob("tasks/*/rows.jsonl")):
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            rows[(int(r["scene_id"]), r["arm"], int(r["repeat"]))] = r
    files = sorted(glob.glob(str(select_dir / "tasks" / "*" / "feat" / f"{split}_s*.npz")))
    if not files:
        raise SystemExit(f"no feat npz for '{split}' under {select_dir}")
    found = False
    for f in files:
        z = np.load(f)
        sids = z["scene_ids"]
        for key in [k for k in z.keys() if k.endswith("/traj")]:
            found = True
            arm, rep, _ = key.split("/")
            if arms and arm not in arms:
                continue
            tr, trb = z[key], z[f"{arm}/{rep}/traj_base"]
            sw = z[f"{arm}/{rep}/switch"]
            r = int(rep[1:])
            for b, sid in enumerate(sids):
                row = rows.get((int(sid), arm, r))
                if row is None:
                    continue
                yield {"arm": arm, "scene_id": int(sid), "repeat": r,
                       "solved": bool(row["solved"]), "n_switch": int(sw[b].sum()),
                       "k": kinematics(tr[b], dt, smooth_w), "k_base": kinematics(trb[b], dt, smooth_w),
                       "file": f}
    if not found:
        raise SystemExit(
            f"no '*/traj' arrays under {select_dir} -- the run was scored without "
            f"--dump_traj, so there are no executed states to measure. Re-score a "
            f"capped subset with DFT_DUMP_TRAJ=1.")


def summarize(recs: list[dict]) -> dict:
    out: dict = {"arms": {}}
    by_arm: dict[str, list] = {}
    for r in recs:
        by_arm.setdefault(r["arm"], []).append(r)
    for arm, rs in sorted(by_arm.items()):
        def agg(sel, field, which="k"):
            v = [_peaks(r[which]).get(field) for r in sel if r[which]]
            v = [x for x in v if x is not None]
            if not v:
                return None
            v = np.asarray(v)
            return {"n": len(v), "p50": float(np.percentile(v, 50)),
                    "p90": float(np.percentile(v, 90)),
                    "p99": float(np.percentile(v, 99)), "max": float(v.max())}
        fired = [r for r in rs if r["n_switch"] > 0]
        rec = {"n_episodes": len(rs), "n_fired": len(fired)}
        for label, sel in (("all", rs), ("fired", fired)):
            rec[label] = {f: agg(sel, f) for f in
                          FIELDS}
            rec[label + "_base"] = {f: agg(sel, f, "k_base") for f in
                                    FIELDS}
        if fired:
            pk = np.array([_peaks(r["k"])["peak_a_lat"] for r in fired if r["k"]])
            bands, lo = {}, 0.0
            for hi, lab in ACC_BANDS:
                bands[lab] = {"n": int(((pk >= lo) & (pk < hi)).sum()),
                              "frac": float(((pk >= lo) & (pk < hi)).mean())}
                lo = hi
            rec["fired_acc_bands"] = bands
            # Paired, per episode: repair vs the base counterfactual over the same
            # steps. <1 means the filter made that episode gentler than the
            # trajectory it replaced. This is the statistic that means something.
            for field in FIELDS:
                pair = [(_peaks(r["k"]).get(field), _peaks(r["k_base"]).get(field))
                        for r in fired if r["k"] and r["k_base"]]
                pair = [(a, b) for a, b in pair if a is not None and b
                        is not None and b > 0]
                if not pair:
                    continue
                ratio = np.array([a / b for a, b in pair])
                rec.setdefault("fired_vs_base", {})[field] = {
                    "n": len(ratio), "ratio_p50": float(np.percentile(ratio, 50)),
                    "ratio_p90": float(np.percentile(ratio, 90)),
                    "frac_gentler_than_base": float((ratio < 1.0).mean())}
        out["arms"][arm] = rec
    return out


def draw(recs: list[dict], out_dir: Path, top: int, dt: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fired = [r for r in recs if r["n_switch"] > 0 and r["k"]]
    if not fired:
        print("[viz] no episode with an intervention; nothing to draw")
        return
    fired.sort(key=lambda r: -_peaks(r["k"])["peak_a_lat"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in fired[:top]:
        k, kb = r["k"], r["k_base"]
        t = np.arange(k["n_steps"]) * dt
        fig, ax = plt.subplots(1, 4, figsize=(19, 4.1))
        ax[0].plot(kb["xy"][:, 0], kb["xy"][:, 1], "--", color="0.6", label="base")
        ax[0].plot(k["xy"][:, 0], k["xy"][:, 1], "-", color="C3", label="executed")
        ax[0].set_aspect("equal"); ax[0].legend(fontsize=7); ax[0].set_title("path")
        for a, key, lab, ref in ((ax[1], "a_lat", "lateral accel m/s$^2$", 2.0),
                                 (ax[2], "j_lat", "lateral jerk m/s$^3$", JERK_REF),
                                 (ax[3], "yaw_rate", "yaw rate rad/s", None)):
            a.plot(np.arange(kb["n_steps"]) * dt, kb[key], "--", color="0.6")
            a.plot(t, k[key], "-", color="C3")
            if ref is not None:
                for sgn in (1, -1):
                    a.axhline(sgn * ref, color="C0", lw=0.7, ls=":")
            a.set_xlabel("s"); a.set_title(lab)
        fig.suptitle(f"{r['arm']}  scene {r['scene_id']} rep {r['repeat']}  "
                     f"switches={r['n_switch']}  solved={r['solved']}  "
                     f"peak|a_lat|={_peaks(k)['peak_a_lat']:.1f}", fontsize=9)
        fig.tight_layout()
        f = out_dir / f"{r['arm']}_s{r['scene_id']}_r{r['repeat']}.png"
        fig.savefig(f, dpi=120); plt.close(fig)
    print(f"[viz] {min(top, len(fired))} figures -> {out_dir}")


def render(d: dict):
    print("\n=== ride quality of executed repairs ===")
    for arm, r in d["arms"].items():
        print(f"\n[{arm}] {r['n_episodes']} episodes, {r['n_fired']} with an intervention")
        for label in ("all", "fired"):
            for f in FIELDS:
                a, b = r[label].get(f), r[label + "_base"].get(f)
                if not a:
                    continue
                print(f"  {label:<6}{f:<15} arm  p50={a['p50']:6.2f} p90={a['p90']:6.2f} "
                      f"p99={a['p99']:6.2f} max={a['max']:6.2f}" +
                      (f"   | base p50={b['p50']:6.2f} p90={b['p90']:6.2f} "
                       f"max={b['max']:6.2f}" if b else ""))
        if "fired_acc_bands" in r:
            print("  peak lateral accel of intervened episodes: " + "  ".join(
                f"{k} {v['frac']*100:.0f}%" for k, v in r["fired_acc_bands"].items()))
        for field, v in (r.get("fired_vs_base") or {}).items():
            print(f"  paired {field:<14} repair/base  p50={v['ratio_p50']:.2f} "
                  f"p90={v['ratio_p90']:.2f}   gentler than base in "
                  f"{v['frac_gentler_than_base']*100:.0f}% of intervened episodes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select_dir", type=Path, required=True)
    ap.add_argument("--split", default="failure_heldout")
    ap.add_argument("--arm", default="", help="comma list; default = every arm with traj")
    ap.add_argument("--world_dt", type=float, default=0.1)
    ap.add_argument("--top", type=int, default=6, help="per-scene figures to draw")
    ap.add_argument("--smooth_w", type=int, default=9,
                    help="Savitzky-Golay window (cubic) for the derivatives. 0 = raw "
                         "finite differences, which at dt=0.1 are dominated by ~4mm of "
                         "diffusion sampling noise and cannot resolve the edit.")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    arms = {x for x in a.arm.split(",") if x.strip()}
    recs = list(collect(a.select_dir, a.split, arms, a.world_dt, a.smooth_w))
    d = summarize(recs)
    d |= {"select_dir": str(a.select_dir), "split": a.split, "world_dt": a.world_dt,
          "smooth_w": a.smooth_w}
    out = a.json or (a.select_dir / f"ride_{a.split}.json")
    out.write_text(json.dumps(d, indent=1))
    print(f"[viz] json -> {out}")
    render(d)
    draw(recs, a.select_dir / f"ride_{a.split}", a.top, a.world_dt)


if __name__ == "__main__":
    main()
