"""Gates for ``world_dt_method="median"``: the plan is executed on the log's
own clock.

THE BUG. ``compute_world_dt_seconds`` took each scene's world step from its
FIRST timestamp pair. The preprocessor lays the policy's plan onto a
``k * world_dt`` grid and the env advances one log step per grid point, so a
short first pair (p50 0.075 s where the median step is 0.100 s) executes the
plan at ~0.75x. Measured 2026-09-23: 44/173 ``failure_heldout`` scenes, 43 of
which never reach the goal; none in ``success_heldout`` / ``intermittent``.

WHAT HAS TO HOLD:

* **W0 ``"first"`` is the old function**, bit for bit (verbatim copy below), so
  every earlier result stays reproducible.
* **W1 ``"median"`` is the log's clock**: it matches the mean step over the
  window the policy executes (steps 10..90) to 1% at p95, in every split.
  W1b: the short-first-pair population EXISTS in ``failure_heldout`` (else this
  file tests nothing) and is absent from ``success_heldout``.
* **W2 the flag reaches the preprocessor**: ``aux["world_dt_seconds"]`` equals
  the method's own value, on the jitted and the eager replan paths.
* **W3 the human's own trajectory, executed as the plan, stays on the log.**
  Each replan hands ``env.step`` the preprocessor's own BC target
  (``features["ego_trajectory"]`` -- the logged future on the model's time
  base) through ``norm_to_world``, so every stage of the time base is in the
  loop: stamped resample -> model grid -> world grid -> one log step per point.
  Under ``"median"`` the ego must track the log to ``--track_tol`` m (p95, every
  executed step, both scene classes). Under ``"first"`` the short-dt scenes must
  visibly lag (the gate has teeth), and the normal ones must not move.
  (Pace is not the teeth here: this plan aims at the human's ABSOLUTE future, so
  each replan re-closes the lag. The policy plans from its own pose and does not,
  which is why the bug costs it the goal.)
* **W4 jit == eager** under ``"median"``.

No denoiser: every plan is supplied. CPU, minutes. Scenes whose executed window
contains a dropped log frame (a gap > 0.15 s) are left out of W3 -- a uniform grid
cannot follow a 0.2 s jump in the log, whichever method sets it. Submitted::

    bash rl/diffusion_ft/slurm/submit_world_dt.sh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
import numpy as np

from data.preprocess import _first_true_index, _gather_ego, compute_world_dt_seconds, extract_ego_index
from rl.diffusion_ft.checkpoint import DEFAULT_CHECKPOINT_PATH, load_diffusion_checkpoint
from rl.diffusion_ft.env import DiffusionWaymaxEnv
from rl.diffusion_ft.reward import RewardConfig
from rl.diffusion_ft.scene_sampler import DEFAULT_INDEX_DIR, SceneSampler

FAILS: list[str] = []
NUMS: dict[str, float] = {}
T0 = 10          # waymax reset leaves timestep at init_steps - 1


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""),
          flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def _dt_reference(state, fallback_dt: float = 0.1):
    """``compute_world_dt_seconds`` as it was before ``method`` existed
    (2026-09-23), verbatim. W0 holds ``"first"`` to it."""
    ego_idx = extract_ego_index(state)
    ego_ts_us = _gather_ego(state.log_trajectory.timestamp_micros, ego_idx).astype(jnp.float32)
    ego_valid = _gather_ego(state.log_trajectory.valid, ego_idx)
    dt = (ego_ts_us[:, 1:] - ego_ts_us[:, :-1]) * 1e-6
    valid_pair = ego_valid[:, 1:] & ego_valid[:, :-1] & (dt > 0.0)
    any_valid = jnp.any(valid_pair, axis=-1)
    first_idx = _first_true_index(valid_pair)
    first_dt = jnp.take_along_axis(dt, first_idx[:, None], axis=1)[:, 0]
    return jnp.where(any_valid, first_dt, jnp.full_like(first_dt, fallback_dt))


def scene_clock(state, window_end: int):
    """Per scene: both methods, the reference, the mean step over the executed
    window, the largest gap inside the tracking window, and how far the human
    drives in it."""
    b = np.arange(np.asarray(state.log_trajectory.x).shape[0])
    e = np.argmax(np.asarray(state.object_metadata.is_sdc), 1)
    ts = np.asarray(state.log_trajectory.timestamp_micros)[b, e].astype(np.int64)
    x = np.asarray(state.log_trajectory.x, np.float64)[b, e]
    y = np.asarray(state.log_trajectory.y, np.float64)[b, e]
    gaps = np.diff(ts, axis=1) * 1e-6
    return {
        "first": np.asarray(compute_world_dt_seconds(state, 0.1, "first"), np.float64),
        "ref": np.asarray(_dt_reference(state, 0.1), np.float64),
        "median": np.asarray(compute_world_dt_seconds(state, 0.1, "median"), np.float64),
        "window_step": (ts[:, 90] - ts[:, T0]) * 1e-6 / (90 - T0),
        "max_gap_track": gaps[:, T0:window_end].max(1),
        "drive_m": np.hypot(np.diff(x[:, T0:window_end + 1], axis=1),
                            np.diff(y[:, T0:window_end + 1], axis=1)).sum(1),
    }


def track(env, state, replans: int):
    """Execute the logged future as the plan. Returns per executed step the
    ego-to-log distance ``[B, n]``, and the world dt the preprocessor used."""
    feats = env.reset(state)
    t0 = env.timestep
    used = np.asarray(env.last_aux["world_dt_seconds"], np.float64)
    for _ in range(replans):
        plan = env.norm_to_world(feats["ego_trajectory"])
        feats = env.step(trajectory_world_bt5=plan).features
    t1 = env.timestep
    ego = np.asarray(env.ego_sim_world_trajectory(), np.float64)[:, t0 + 1:t1 + 1, :2]
    log = np.asarray(env.ego_log_world_trajectory(), np.float64)[:, t0 + 1:t1 + 1, :2]
    return np.linalg.norm(ego - log, axis=-1), used


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits_json", required=True)
    p.add_argument("--shard_glob",
                   default="/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*")
    p.add_argument("--index_dir", default=DEFAULT_INDEX_DIR)
    p.add_argument("--ckpt", default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--n_failure", type=int, default=-1, help="failure_heldout scenes for W0/W1 (-1 = all)")
    p.add_argument("--n_success", type=int, default=200)
    p.add_argument("--n_track", type=int, default=8, help="tracking scenes per class")
    p.add_argument("--replans", type=int, default=4)
    p.add_argument("--replan_interval_steps", type=int, default=10)
    p.add_argument("--short_dt", type=float, default=0.09)
    p.add_argument("--track_tol", type=float, default=0.25)
    p.add_argument("--load_batch", type=int, default=24)
    p.add_argument("--verdict", default="")
    args = p.parse_args(argv)

    spec = json.loads(Path(args.splits_json).read_text())
    sampler = SceneSampler(args.shard_glob, batch_size=1, eval_size=0,
                           include_sdc_paths=False, prefetch=False, index_dir=args.index_dir)
    window_end = T0 + args.replans * args.replan_interval_steps

    # ---- W0 / W1 on the clocks ---------------------------------------------- #
    clocks = {}
    for split, n in (("failure_heldout", args.n_failure), ("success_heldout", args.n_success)):
        ids = [int(i) for i in spec["splits"][split]["scene_ids"]]
        ids = ids if n < 0 else ids[:n]
        parts = []
        for lo in range(0, len(ids), args.load_batch):
            parts.append(scene_clock(sampler.load_explicit(ids[lo:lo + args.load_batch]), window_end))
        clocks[split] = {k: np.concatenate([c[k] for c in parts]) for k in parts[0]}
        clocks[split]["ids"] = np.asarray(ids)
        c = clocks[split]
        rel = np.abs(c["median"] / c["window_step"] - 1.0)
        short = c["first"] < args.short_dt
        NUMS[f"{split}_n"] = len(ids)
        NUMS[f"{split}_short_first"] = int(short.sum())
        NUMS[f"{split}_median_vs_window_p95"] = float(np.percentile(rel, 95))
        print(f"[gate] {split}: {len(ids)} scenes; first-pair dt < {args.short_dt} in "
              f"{int(short.sum())}; first p5/p50 {np.percentile(c['first'], [5, 50]).round(4)}; "
              f"median p5/p50/p95 {np.percentile(c['median'], [5, 50, 95]).round(4)}; "
              f"|median / window step - 1| p95 {np.percentile(rel, 95):.4f}")
        d0 = float(np.max(np.abs(c["first"] - c["ref"])))
        check(f"W0[{split}] 'first' is the pre-change function, bit for bit", d0 == 0.0,
              f"max |diff| {d0:.2e}")
        check(f"W1[{split}] 'median' matches the executed window's clock to 1% (p95)",
              float(np.percentile(rel, 95)) < 0.01, f"p95 {np.percentile(rel, 95):.4f}")
    check("W1b the short-first-pair population exists in failure_heldout",
          NUMS["failure_heldout_short_first"] > 0, f"{NUMS['failure_heldout_short_first']} scenes")
    check("W1b ...and is absent from success_heldout",
          NUMS["success_heldout_short_first"] == 0, f"{NUMS['success_heldout_short_first']} scenes")

    # ---- W2-W4 in the env ---------------------------------------------------- #
    c = clocks["failure_heldout"]
    ok = (c["max_gap_track"] < 0.15) & (c["drive_m"] > 5.0)
    pick_s = c["ids"][ok & (c["first"] < args.short_dt)][:args.n_track]
    pick_n = c["ids"][ok & (c["first"] >= args.short_dt)][:args.n_track]
    ids = [int(i) for i in np.concatenate([pick_s, pick_n])]
    is_short = np.array([True] * len(pick_s) + [False] * len(pick_n))
    print(f"[gate] tracking on {len(pick_s)} short-dt + {len(pick_n)} normal scenes "
          f"(no dropped frame in steps {T0}..{window_end}, human drives > 5 m)")
    state = sampler.load_explicit(ids)
    ck = scene_clock(state, window_end)
    ckpt = load_diffusion_checkpoint(args.ckpt)
    n_obj = int(np.asarray(state.log_trajectory.x).shape[1])

    def make(method, jit=True):
        return DiffusionWaymaxEnv(ckpt, reward_config=RewardConfig(), max_num_objects=n_obj,
                                  replan_interval_steps=args.replan_interval_steps, seed=0,
                                  jit_replan=jit, world_dt_method=method)

    res = {}
    for tag, method, jit in (("first", "first", True), ("median", "median", True),
                             ("median_eager", "median", False)):
        err, used = track(make(method, jit), state, args.replans)
        res[tag] = err
        want = ck["first"] if method == "first" else ck["median"]
        gap = float(np.max(np.abs(used - want)))
        check(f"W2[{tag}] the preprocessor used the '{method}' dt", gap < 1e-6,
              f"max |aux world_dt - {method}| {gap:.2e} s")
        for cls, m in (("short", is_short), ("normal", ~is_short)):
            if not m.any():
                continue
            v = err[m].ravel()
            NUMS[f"track_{tag}_{cls}_p50"] = float(np.percentile(v, 50))
            NUMS[f"track_{tag}_{cls}_p95"] = float(np.percentile(v, 95))
            NUMS[f"track_{tag}_{cls}_max"] = float(v.max())
            print(f"[gate] {tag:<13} {cls:<6} ego-to-log p50/p95/max "
                  f"{np.percentile(v, 50):.3f} / {np.percentile(v, 95):.3f} / {v.max():.3f} m")

    for cls, m in (("short", is_short), ("normal", ~is_short)):
        if m.any():
            p95 = float(np.percentile(res["median"][m], 95))
            check(f"W3[{cls}] under 'median' the human's own plan stays on the log",
                  p95 < args.track_tol, f"p95 {p95:.3f} m, tol {args.track_tol}")
    if is_short.any():
        p95 = float(np.percentile(res["first"][is_short], 95))
        check("W3 under 'first' the short-dt scenes visibly lag (the gate has teeth)",
              p95 > 2 * args.track_tol, f"p95 {p95:.3f} m")
    if (~is_short).any():
        d = abs(float(np.percentile(res["first"][~is_short], 95))
                - float(np.percentile(res["median"][~is_short], 95)))
        check("W3 normal scenes track the same under both methods", d < 0.05,
              f"|p95 first - p95 median| {d:.3f} m")
    d_je = float(np.max(np.abs(res["median"] - res["median_eager"])))
    check("W4 'median': jitted and eager replan paths agree", d_je < 1e-3,
          f"max |diff| {d_je:.2e} m")

    print()
    head = "PASS" if not FAILS else "FAIL: " + "; ".join(FAILS)
    if args.verdict:
        Path(args.verdict).parent.mkdir(parents=True, exist_ok=True)
        Path(args.verdict).write_text(head + "\n" + json.dumps(NUMS) + "\n")
        print(f"[gate] verdict -> {args.verdict}")
    if FAILS:
        print(f"[gate] {len(FAILS)} FAILED")
        raise SystemExit(1)
    print("[gate] all world-dt gates passed")


if __name__ == "__main__":
    main()
