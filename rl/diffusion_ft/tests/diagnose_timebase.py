"""Why does the BC target not reconstruct to the human log?

`features["ego_trajectory"]` -- the normalized future used as the behaviour-cloning
target -- round-trips through `norm_to_world` to **60.9 m** from the log it is
supposed to BE, implying 78.4 m/s. Clamping the resampler's interpolation weight
(2026-08-30) fixed the magnitude (mean|x| 8486 -> 0.315) but not the content: a
clamp turns a wild extrapolation into an interpolation between two *wrong*
samples, which is bounded garbage rather than no garbage.

The suspected root cause is the time base. `_preprocess_single_batch` builds

    src_t = timestamp_micros * 1e-6 - timestamp_micros[anchor] * 1e-6

and hands it to `resample_trajectory_timebase_jax`, which calls `jnp.searchsorted`
-- and searchsorted requires a **sorted** array. Invalid log steps carry
timestamp 0, so a scene with any invalid step has a time base that jumps
backwards, searchsorted returns a meaningless bracket, and the resampled
"expert future" is an interpolation between two unrelated points in time.

This script does not assume that. It measures the time base directly, then
**proves or refutes the proposed fix before shared code is touched** by
resampling the same log two ways and scoring both against ground truth:

  A. the current timestamp-derived time base, and
  B. a step-index base, ``(arange(T) - anchor) * world_dt``, monotonic by
     construction and exact for a uniformly sampled log.

Ground truth is the log itself at the matching world steps, so the correct answer
is ~0 m for whichever base is right. If B lands at ~0 and A does not, the fix is
established rather than guessed. If *both* miss, the time base is innocent and
the bug is elsewhere -- which is equally worth knowing before another patch.

CPU only, no checkpoint, no policy: seconds of work on pure log arrays.
"""

from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import jax.numpy as jnp

from data.preprocess import (
    ego_source_time_seconds,
    resample_trajectory_timebase_jax,
)
from rl.diffusion_ft.train_dppo import load_batch


def ego_rows(state):
    log = state.log_trajectory
    is_sdc = np.asarray(state.object_metadata.is_sdc)
    ego = is_sdc.argmax(axis=1)

    def row(f):
        a = np.asarray(f)
        return np.stack([a[b, ego[b]] for b in range(a.shape[0])])

    return {k: row(getattr(log, k)) for k in
            ("x", "y", "vel_x", "vel_y", "yaw", "valid", "timestamp_micros")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tfrecord", type=str,
                    default="/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord")
    ap.add_argument("--n_scenes", type=int, default=8)
    ap.add_argument("--anchor", type=int, default=10, help="init_steps - 1")
    ap.add_argument("--predict_horizon", type=int, default=25)
    ap.add_argument("--model_dt", type=float, default=0.2)
    ap.add_argument("--world_dt", type=float, default=0.1)
    ap.add_argument("--out_dir", type=str, required=True)
    a = ap.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    idx = list(range(a.n_scenes))
    st = load_batch(a.tfrecord, idx)
    r = ego_rows(st)
    ts = r["timestamp_micros"].astype(np.float64) * 1e-6
    vl = r["valid"]; T = ts.shape[1]
    anchor = a.anchor

    print(f"[tb] {a.n_scenes} scenes, T={T}, anchor={anchor}\n")
    print(f"{'sc':>3} {'nvalid':>7} {'monotonic':>10} {'1st bad':>8} "
          f"{'dt values':>26} {'log speed':>10}")
    rows = []
    for b in range(len(idx)):
        s = ts[b] - ts[b, anchor]
        d = np.diff(s)
        bad = np.where(d <= 0)[0]
        uniq = np.unique(np.round(d, 4))
        xy = np.stack([r["x"][b], r["y"][b]], -1)
        sp = float(np.linalg.norm(np.diff(xy[anchor:anchor + 11], axis=0), axis=-1).mean()
                   / a.world_dt)
        rows.append({"scene": idx[b], "n_valid": int(vl[b].sum()),
                     "monotonic": bool((d > 0).all()),
                     "first_non_increasing": (int(bad[0]) if bad.size else None),
                     "dt_values": [float(v) for v in uniq[:5]], "log_speed_mps": sp})
        print(f"{idx[b]:>3} {int(vl[b].sum()):>7} {str(bool((d>0).all())):>10} "
              f"{(str(bad[0]) if bad.size else '-'):>8} {str(uniq[:3])[:26]:>26} {sp:>10.2f}")

    # ---- A vs B: resample the same log two ways, score against the log ---- #
    model_t = np.arange(a.predict_horizon + 1, dtype=np.float32) * a.model_dt
    traj = np.stack([r["x"], r["y"], r["vel_x"], r["vel_y"], r["yaw"]], axis=-1)
    src_ts = (ts - ts[:, anchor:anchor + 1]).astype(np.float32)
    step_idx = np.arange(T, dtype=np.float32)[None, :]
    src_ix = ((step_idx - anchor) * a.world_dt).astype(np.float32)

    def resample(src):
        return np.asarray(resample_trajectory_timebase_jax(
            jnp.asarray(traj), jnp.asarray(src), jnp.asarray(model_t), yaw_index=4))

    # C: what the production path now actually uses. A and B are hypotheses; this
    # is `data.preprocess.ego_source_time_seconds`, the guarded base that shipped
    # once B was confirmed. It must track B here and A on well-timestamped data --
    # if it ever tracks neither, the guard is choosing wrong and the BC target is
    # back to being interpolated between unrelated points in time.
    world_dt_b = np.full(len(idx), a.world_dt, np.float32)
    src_prod, _, used_ts = ego_source_time_seconds(
        jnp.asarray(ts, jnp.float32), jnp.full(len(idx), anchor, jnp.int32),
        jnp.asarray(world_dt_b), a.world_dt)
    src_prod = np.asarray(src_prod)

    got_A, got_B, got_C = resample(src_ts), resample(src_ix), resample(src_prod)
    # Ground truth: the log at the world steps each model_t lands on.
    want_steps = np.clip(anchor + np.round(model_t / a.world_dt).astype(int), 0, T - 1)
    want = np.stack([np.stack([r["x"][b][want_steps], r["y"][b][want_steps]], -1)
                     for b in range(len(idx))])
    errA = np.linalg.norm(got_A[..., :2] - want, axis=-1).mean(axis=1)
    errB = np.linalg.norm(got_B[..., :2] - want, axis=-1).mean(axis=1)
    errC = np.linalg.norm(got_C[..., :2] - want, axis=-1).mean(axis=1)

    print(f"\n[tb] === resampled future vs the log it should reproduce (metres) ===")
    print(f"{'sc':>3} {'A timestamps':>13} {'B step-index':>13} {'C production':>13} "
          f"{'C used ts':>10}")
    for b in range(len(idx)):
        print(f"{idx[b]:>3} {errA[b]:>13.3f} {errB[b]:>13.3f} {errC[b]:>13.3f} "
              f"{str(bool(used_ts[b])):>10}")
        rows[b].update({"err_timestamp_base": float(errA[b]),
                        "err_stepindex_base": float(errB[b]),
                        "err_production_base": float(errC[b]),
                        "production_used_timestamps": bool(used_ts[b])})
    print(f"{'mean':>3} {errA.mean():>13.3f} {errB.mean():>13.3f} {errC.mean():>13.3f}")

    ok_b = errB.mean() < 0.5
    ok_a = errA.mean() < 0.5
    ok_c = errC.mean() < 0.5
    print()
    print(f"[tb] {'PASS' if ok_c else 'FAIL'}  production base reproduces the log: "
          f"{errC.mean():.3f} m (timestamps used on "
          f"{int(np.asarray(used_ts).sum())}/{len(idx)} scenes)")
    if ok_b and not ok_a:
        verdict = ("CONFIRMED: the timestamp time base is the bug and the step-index "
                   "base fixes it.")
    elif ok_a and ok_b:
        verdict = ("REFUTED: both bases reproduce the log here, so the time base is "
                   "innocent -- look upstream at how ego_trajectory is normalized/sliced.")
    elif not ok_b and not ok_a:
        verdict = ("REFUTED: NEITHER base reproduces the log, so resampling is not the "
                   "(only) fault. Do not patch the time base on this evidence.")
    else:
        verdict = ("SURPRISE: the timestamp base is fine and the step-index base is not; "
                   "the log is not uniformly sampled.")
    print(f"[tb] {verdict}")
    (out / "timebase.json").write_text(json.dumps(
        {"scenes": rows, "mean_err_timestamp_base": float(errA.mean()),
         "mean_err_stepindex_base": float(errB.mean()), "verdict": verdict}, indent=2))
    print(f"[tb] wrote {out}/timebase.json")


if __name__ == "__main__":
    main()
