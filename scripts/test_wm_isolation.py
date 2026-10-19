"""Gate: the multi-future WM keeps its OWN data modules next to new_model's repo preprocess.

new_model's repo ``data/preprocess.py`` samples the map differently (nearest segments, random
points, an extra ``rng`` argument) from the copy the WM was trained with. The WM's inference
module does ``from data.preprocess import _build_map_features`` at import time, so if the repo's
module is already in ``sys.modules`` it would be reused (Ishan's 1b2f7e6). Checks, on CPU:

  W1  after loading, the WM's ``_build_map_features`` comes from the WM package's file
  W2  the repo's ``data.preprocess`` is restored in ``sys.modules`` (same module object)
  W3  a second load (the discriminator's own WM copy) keeps both bindings
  W4  a WM forward on real scenes runs and gives finite means
  W5  Ishan's package (multiFuture_GaussianWorldModel_ishan/, --wm_backend multifuture_ishan) loads in
      the same process with ITS OWN map builder, Sagar's binding is untouched, and it forecasts too
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import data.preprocess as repo_pp  # noqa: E402  -- imported FIRST, as the planner does

MF = "/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20260916_200620/"
ISHAN = ("/data/user_data/yixiz/autonlab/zfsauton/scratch/ishanj/world_model/ckpts_multifuture_gaussian/"
         "multiFuture_GaussianWorldModel_20261005_092950/epoch_0450.pt")


def main() -> int:
    from es_discriminator.real_es.wm_cache import jax_state, load_forecaster, predict_step
    from waymax import config as waymax_config
    from data.scenario_loader import load_scenario_state_batch_fast

    fails = []
    fc = load_forecaster(MF + "latest.pt", "cpu")
    inf_mod = sys.modules[type(fc).__module__]
    src = Path(inf_mod._build_map_features.__code__.co_filename).resolve()
    pkg = (REPO_ROOT / "multiFuture_GaussianWorldModel" / "data" / "preprocess.py").resolve()
    print(f"[wm] WM inference module {inf_mod.__name__}; its _build_map_features from {src}", flush=True)
    if src != pkg:
        fails.append(f"W1 WM map builder is {src}, not the package's {pkg}")
    if sys.modules.get("data.preprocess") is not repo_pp:
        fails.append("W2 repo data.preprocess was not restored")
    fc2 = load_forecaster(MF + "epoch_0450.pt", "cpu")
    src2 = Path(sys.modules[type(fc2).__module__]._build_map_features.__code__.co_filename).resolve()
    if src2 != pkg or sys.modules.get("data.preprocess") is not repo_pp:
        fails.append("W3 second load changed a binding")
    print(f"[wm] repo data.preprocess restored: {sys.modules.get('data.preprocess') is repo_pp}; "
          f"second load map builder {src2.name} in {src2.parent.parent.name}", flush=True)

    tf = "/data/user_data/yixiz/autonlab/zfsauton/datasets/womd/tf_example/validation/validation_tfexample.tfrecord-00000-of-00150"
    ds_cfg = dataclasses.replace(waymax_config.WOD_1_3_1_TRAINING, path=tf, max_num_objects=128)
    out = load_scenario_state_batch_fast(ds_cfg, list(range(3)))
    state = out[0] if isinstance(out, tuple) else out
    p = predict_step(fc2, jax_state(state), 10)
    m = p["mean_xy"][p["valid"]]
    print(f"[wm] forward on 3 scenes: {int(p['valid'].sum())} valid agents, mean |xy| {np.abs(m).mean():.1f}, "
          f"finite {bool(np.isfinite(m).all())}", flush=True)
    if not (m.size and np.isfinite(m).all()):
        fails.append("W4 WM forward produced no finite means")

    from simulation.run_simulation import _load_isolated_wm_forecaster
    ishan_pkg = REPO_ROOT / "multiFuture_GaussianWorldModel_ishan"
    fi = _load_isolated_wm_forecaster(ishan_pkg, checkpoint_path=ISHAN, device="cpu",
                                      planning_mode="expected_independent")
    src_i = Path(sys.modules[type(fi).__module__]._build_map_features.__code__.co_filename).resolve()
    src_s = Path(inf_mod._build_map_features.__code__.co_filename).resolve()
    print(f"[wm] Ishan module {type(fi).__module__}; map builder from {src_i.parent.parent.name}; "
          f"Sagar's still from {src_s.parent.parent.name}", flush=True)
    if src_i != (ishan_pkg / "data" / "preprocess.py").resolve() or src_s != pkg:
        fails.append(f"W5 bindings wrong: ishan {src_i}, sagar {src_s}")
    if sys.modules.get("data.preprocess") is not repo_pp:
        fails.append("W5 repo data.preprocess was not restored after Ishan's load")
    # Through the forecaster's own API -- what the ES scorer and the repair call.
    for name, f in (("sagar", fc), ("ishan", fi)):
        fm = f.forecast_modes(jax_state(state), timestep=10, traj_length=26, world_idx=0)
        traj, probs, mask = (np.asarray(fm[k]) for k in ("traj_k", "mode_probs", "masks"))
        ego = np.asarray([[[float(state.log_trajectory.x[0, j, t]), float(state.log_trajectory.y[0, j, t]),
                            float(state.log_trajectory.yaw[0, j, t]), 0.0, 0.0] for t in range(10, 36)]
                          for j in [int(np.argmax(np.asarray(state.object_metadata.is_sdc)[0]))]], np.float32)
        sc, _ = f.score_independent_mode_collision(ego, world_idx=0)
        ok = np.isfinite(traj[mask.any(-1)]).all() and np.isfinite(np.asarray(sc)).all()
        print(f"[wm] {name}: forecast_modes traj_k {traj.shape}, mode probs sum "
              f"{probs.sum(-1).mean():.3f}; logged-ego collision score {np.asarray(sc).ravel()[:1]}; finite {ok}", flush=True)
        if not ok:
            fails.append(f"W5 {name} forecaster API produced non-finite output")
    print("=== " + ("PASS" if not fails else "FAIL: " + "; ".join(fails)) + " ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
