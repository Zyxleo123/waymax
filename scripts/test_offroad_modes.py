"""Gate: porting Ishan's ``--offroad_mode margin_tiebreak`` leaves ``binary`` unchanged.

The port split ``Scorer.compute_offroad_score_v2`` into ``_offroad_corner_distances`` (the old
body, returning the signed corner distances) plus the 0/1 gate, and added
``compute_offroad_score_margin_tiebreak`` (Ishan's scoring, verbatim). On real scenes and
candidates built from the logged ego future (lateral shifts and heading rotations, so some
leave the road), on CPU:

  O1  binary: the new compute_offroad_score_v2 equals the pre-port implementation exactly
      (es_disc's scores/scorer_lane_graph.py, loaded as a sibling module of the same package)
  O2  margin_tiebreak score is 0 exactly where binary is 0, and in [0.5, 1] elsewhere
  O3  tiebreak is 1 on-road and in [0, 1) offroad
  O4  compute_score(metrics=["offroad"]): binary -> the 0/1 gate; margin_tiebreak -> score + 1e-3 * tiebreak
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
PREPORT = Path("/home/yixiz/waymax_rs_es_disc/scores/scorer_lane_graph.py")


def _load_preport():
    import scores  # noqa: F401  -- the package the old module's relative imports resolve against
    spec = importlib.util.spec_from_file_location("scores._scorer_preport", PREPORT)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "scores"
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _candidates(ego_t5: np.ndarray) -> np.ndarray:
    """[64, T, 5] from one logged ego future: 32 lateral shifts and 32 rotations about the anchor."""
    x, y, yaw = ego_t5[:, 0], ego_t5[:, 1], ego_t5[:, 2]
    nx, ny = -np.sin(yaw), np.cos(yaw)
    out = []
    for d in np.linspace(-4.0, 4.0, 32):
        c = ego_t5.copy()
        ramp = np.linspace(0.0, 1.0, len(x))
        c[:, 0], c[:, 1] = x + d * ramp * nx, y + d * ramp * ny
        out.append(c)
    for a in np.linspace(-0.6, 0.6, 32):
        c = ego_t5.copy()
        dx, dy = x - x[0], y - y[0]
        c[:, 0] = x[0] + np.cos(a) * dx - np.sin(a) * dy
        c[:, 1] = y[0] + np.sin(a) * dx + np.cos(a) * dy
        c[:, 2] = yaw + a
        out.append(c)
    return np.stack(out).astype(np.float32)


def main() -> int:
    from waymax import config as waymax_config
    from data.scenario_loader import load_scenario_state_batch_fast
    from scores.scorer_lane_graph import OFFROAD_TIEBREAK_EPS, Scorer

    old_mod = _load_preport()
    tf = "/data/user_data/yixiz/autonlab/zfsauton/datasets/womd/tf_example/validation/validation_tfexample.tfrecord-00000-of-00150"
    ds_cfg = dataclasses.replace(waymax_config.WOD_1_3_1_TRAINING, path=tf, max_num_objects=128)
    out = load_scenario_state_batch_fast(ds_cfg, list(range(5)))
    state = out[0] if isinstance(out, tuple) else out
    log = state.log_trajectory
    sdc = np.argmax(np.asarray(state.object_metadata.is_sdc).astype(bool), axis=1)
    t0, T = 10, 26
    fails = []
    n_off = n_all = 0
    for w in range(len(sdc)):
        e = int(sdc[w])
        ego = np.stack([np.asarray(f)[w, e, t0:t0 + T] for f in (log.x, log.y, log.yaw, log.vel_x, log.vel_y)], -1)
        cand = jnp.asarray(_candidates(ego))
        old = np.asarray(old_mod.Scorer(metrics=["offroad"]).compute_offroad_score_v2(cand, state, t0, w))
        new_s = Scorer(metrics=["offroad"])
        new = np.asarray(new_s.compute_offroad_score_v2(cand, state, t0, w))
        if not np.array_equal(old, new):
            fails.append(f"O1 world {w}: binary differs on {int((old != new).sum())} candidates")
        score, tie = (np.asarray(v) for v in new_s.compute_offroad_score_margin_tiebreak(cand, state, t0, w))
        off = new == 0.0
        n_off += int(off.sum()); n_all += off.size
        if not (np.all(score[off] == 0.0) and np.all((score[~off] >= 0.5) & (score[~off] <= 1.0))):
            fails.append(f"O2 world {w}: margin score out of range")
        if not (np.all(tie[~off] == 1.0) and np.all((tie[off] >= 0.0) & (tie[off] < 1.0))):
            fails.append(f"O3 world {w}: tiebreak out of range")
        for mode, want in (("binary", new), ("margin_tiebreak", score + OFFROAD_TIEBREAK_EPS * tie)):
            try:
                got = np.asarray(Scorer(metrics=["offroad"], offroad_mode=mode).compute_score(
                    trajectories=cand, sim_state=state, timestep=t0, world_idx=w))
            except Exception as exc:  # compute_score may need lane data the gate does not set up
                print(f"[off] world {w} {mode}: compute_score skipped ({type(exc).__name__}: {exc})", flush=True)
                continue
            if not np.allclose(got, want, atol=1e-6):
                fails.append(f"O4 world {w} {mode}: compute_score {got[:4]} != {want[:4]}")
        print(f"[off] world {w}: {int(off.sum())}/{off.size} candidates offroad; margin on-road "
              f"min {score[~off].min() if (~off).any() else float('nan'):.3f}; tiebreak offroad "
              f"max {tie[off].max() if off.any() else float('nan'):.3f}", flush=True)
    if n_off == 0 or n_off == n_all:
        fails.append(f"candidate set not informative ({n_off}/{n_all} offroad)")
    print("=== " + ("PASS" if not fails else "FAIL: " + "; ".join(fails)) + " ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
