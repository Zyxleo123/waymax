"""Offline checks for the MF collision-method ablation (no closed-loop rollout).

On real fail1500 scenes, with 64 synthetic ego candidates around the logged ego
path at several replan steps, checks:
  1  mean_learned == expected_independent (existing code path)
  2  mean_uniform uses exactly w = 1/K
  3  sample tensor shapes
  4  samples are centred on the mode means
  5  per-axis std at model steps matches the model's predicted sigma
  6  sigma -> 0: gaussian_uniform -> mean_uniform, gaussian_learned -> mean_learned
  7  one draw per prediction: repeated forecast() / split candidate batches reuse it
  8  fixed seed reproduces samples and scores; another seed changes them
and times one scoring call per method (mean, M=4/8/16).

Run from the waymax_rs repo root::

  PYTHONPATH=$PWD python multiFuture_GaussianWorldModel/evaluation/validate_mf_collision_methods.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.scenario_loader import load_scenario_state_batch_fast  # noqa: E402
from simulation.run_simulation import _load_isolated_wm_forecaster  # noqa: E402
from waymax import config as waymax_config  # noqa: E402

PKG = ROOT / "multiFuture_GaussianWorldModel"
CKPT = "/zfsauton/scratch/sbellad/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20260916_200620/epoch_0450.pt"
SUBSET = Path("/zfsauton/scratch/sbellad/immediate_exp/es_failure_subset_n1500_seed0")
H, DT = 50, 0.1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--timesteps", default="10,40,70")
    p.add_argument("--num_candidates", type=int, default=64)
    p.add_argument("--out_json", default=None)
    return p.parse_args()


def load_scenes():
    targets = json.loads((SUBSET / "target_scenarios.json").read_text())
    batches = []
    for shard in sorted(targets)[:2]:
        idx = targets[shard][:5]
        cfg = dataclasses.replace(
            waymax_config.WOD_1_3_1_TRAINING, path=str(SUBSET / "tfrecords" / "training" / shard),
            max_num_objects=128, batch_dims=(len(idx),), shuffle_seed=0,
        )
        sim_state, _, scene_ids = load_scenario_state_batch_fast(cfg, idx, return_scenario_ids=True)
        batches.append((shard, idx, sim_state, scene_ids))
    return batches


def ego_candidates(sim_state, b, t, e):
    """e boxes [E, H, 5]: logged ego path time-warped (speed x0.4..1.4) + lateral offset ramp."""
    traj = sim_state.log_trajectory
    ego = int(np.argmax(np.asarray(sim_state.object_metadata.is_sdc[b])))
    xy = np.stack([np.asarray(traj.x[b, ego]), np.asarray(traj.y[b, ego])], -1)
    valid = np.asarray(traj.valid[b, ego]).astype(bool)
    last = np.nonzero(valid)[0].max()
    xy = xy[: last + 1]
    yaw0 = float(np.asarray(traj.yaw[b, ego, t]))
    normal = np.array([-np.sin(yaw0), np.cos(yaw0)])
    n_side = int(np.sqrt(e))
    speeds = np.linspace(0.4, 1.4, n_side)
    offsets = np.linspace(-3.0, 3.0, e // n_side)
    out = []
    for s in speeds:
        src = np.clip(t + np.arange(H) * s, 0, len(xy) - 1)
        base = np.stack([np.interp(src, np.arange(len(xy)), xy[:, j]) for j in range(2)], -1)
        for o in offsets:
            p = base + (np.linspace(0, 1, H)[:, None] * o) * normal[None]
            d = np.diff(p, axis=0, prepend=p[:1] - (p[1:2] - p[:1]))
            yaw = np.arctan2(d[:, 1], d[:, 0])
            out.append(np.stack([p[:, 0], p[:, 1], yaw,
                                 np.full(H, float(traj.length[b, ego, t])),
                                 np.full(H, float(traj.width[b, ego, t]))], -1))
    return np.asarray(out[:e], dtype=np.float32)


def make(args, method=None, m=8, seed=0):
    extra = None if method is None else {
        "mf_collision_method": method, "mf_gaussian_num_samples": m, "mf_gaussian_seed": seed,
    }
    return _load_isolated_wm_forecaster(
        PKG, checkpoint_path=args.ckpt, device=args.device,
        planning_mode="expected_independent", extra_kwargs=extra,
    )


def score(fc, sim_state, t, b, ego):
    fc.forecast(sim_state, timestep=t, traj_length=H, world_idx=b, dt=DT)
    s, d = fc.score_independent_mode_collision(ego, world_idx=b)
    return np.asarray(s), d


def main():
    args = parse_args()
    ts = [int(x) for x in args.timesteps.split(",")]
    batches = load_scenes()

    base = make(args)  # existing expected_independent
    planning = sys.modules[[k for k in sys.modules if k.endswith("model.world_model.planning") and "_wmpkg_" in k][0]]
    fcs = {
        "mean_learned": make(args, "mean_learned"),
        "mean_uniform": make(args, "mean_uniform"),
        **{f"gaussian_uniform_M{m:02d}": make(args, "gaussian_uniform", m) for m in (4, 8, 16)},
        **{f"gaussian_learned_M{m:02d}": make(args, "gaussian_learned", m) for m in (4, 8, 16)},
    }
    report = {"checks": {}, "timing_s_per_call": {}, "nan": False}
    chk = report["checks"]
    diffs = {"c1_mean_learned_vs_expected": [], "c1_new_combine_vs_existing": [], "c2_mean_uniform": [],
             "c6_gu_tiny_vs_mu": [], "c6_gl_tiny_vs_ml": []}
    shapes_ok, centred, std_ratio, c7, c8 = True, [], [], [], []
    timings = {k: [] for k in ["expected_independent", *fcs]}
    k_modes = None

    for shard, idx, sim_state, scene_ids in batches:
        for fc in fcs.values():
            fc.set_scene_ids(scene_ids)
        for b in range(len(idx)):
            for t in ts:
                ego = ego_candidates(sim_state, b, t, args.num_candidates)
                t0 = time.perf_counter(); s_base, _ = score(base, sim_state, t, b, ego)
                timings["expected_independent"].append(time.perf_counter() - t0)
                res = {}
                for name, fc in fcs.items():
                    fc.forecast(sim_state, timestep=t, traj_length=H, world_idx=b, dt=DT)
                    t0 = time.perf_counter()
                    s, d = fc.score_independent_mode_collision(ego, world_idx=b)
                    timings[name].append(time.perf_counter() - t0)
                    res[name] = (np.asarray(s), d)
                    report["nan"] |= bool(np.isnan(s).any())

                # 1
                diffs["c1_mean_learned_vs_expected"].append(float(np.abs(res["mean_learned"][0] - s_base).max()))
                fc = fcs["mean_uniform"]
                cache = next(iter(fc._cache.values()))
                traj_k, probs, masks = cache["traj_k"][b], cache["mode_probs"][b], cache["masks"][b]
                k_modes = traj_k.shape[1]
                valid = masks.any(-1)
                c = planning.ego_agent_mode_collisions(ego, traj_k, masks)
                s_new, _ = planning.combine_mf_collision(c, planning.mf_mode_weights("mean_learned", probs), valid)
                diffs["c1_new_combine_vs_existing"].append(float(np.abs(s_new - s_base).max()))
                # 2
                w = planning.mf_mode_weights("mean_uniform", probs)
                assert np.all(w == np.float32(1.0 / k_modes))
                manual = np.prod(1.0 - c.mean(-1) * valid[None], axis=-1)
                diffs["c2_mean_uniform"].append(float(np.abs(res["mean_uniform"][0] - manual).max()))

                # 3, 6, 7, 8 on M=8
                for kind in ("uniform", "learned"):
                    fc = fcs[f"gaussian_{kind}_M08"]
                    cache = next(iter(fc._cache.values()))
                    noise = cache["sample_noise_k"]
                    shapes_ok &= noise.shape == (len(idx), traj_k.shape[0], k_modes, 8, H, 2)
                    shapes_ok &= res[f"gaussian_{kind}_M08"][1]["c_hat"].shape == (len(ego), traj_k.shape[0], k_modes)
                    # 7: a later ES round re-forecasts the same replan -> same object, same scores
                    fc.forecast(sim_state, timestep=t, traj_length=H, world_idx=b, dt=DT)
                    same_obj = next(iter(fc._cache.values()))["sample_noise_k"] is noise
                    s_full = res[f"gaussian_{kind}_M08"][0]
                    s_a, _ = fc.score_independent_mode_collision(ego[:20], world_idx=b)
                    s_b, _ = fc.score_independent_mode_collision(ego[20:], world_idx=b)
                    c7.append(same_obj and np.array_equal(np.concatenate([s_a, s_b]), s_full))
                    # 6: shrink the cached draw
                    cache["sample_noise_k"] = noise * 1e-6
                    s_tiny, _ = fc.score_independent_mode_collision(ego, world_idx=b)
                    cache["sample_noise_k"] = noise
                    ref = res["mean_uniform" if kind == "uniform" else "mean_learned"][0]
                    diffs[f"c6_g{kind[0]}_tiny_vs_m{kind[0]}"].append(float(np.abs(s_tiny - ref).max()))

                # 8: fresh instances, same / different seed
                if b == 0:
                    f1, f2 = make(args, "gaussian_learned", 8, 0), make(args, "gaussian_learned", 8, 1)
                    f1.set_scene_ids(scene_ids)
                    f2.set_scene_ids(scene_ids)
                    s1, _ = score(f1, sim_state, t, b, ego)
                    s2, _ = score(f2, sim_state, t, b, ego)
                    n0 = next(iter(fcs["gaussian_learned_M08"]._cache.values()))["sample_noise_k"]
                    n1 = next(iter(f1._cache.values()))["sample_noise_k"]
                    n2 = next(iter(f2._cache.values()))["sample_noise_k"]
                    c8.append({"same_seed_noise_equal": bool(np.array_equal(n0, n1)),
                               "same_seed_scores_equal": bool(np.array_equal(s1, res["gaussian_learned_M08"][0])),
                               "other_seed_noise_differs": bool(not np.array_equal(n0, n2)),
                               "other_seed_score_maxdiff": float(np.abs(s2 - s1).max())})

                # 4, 5: large-M draw from the forecaster's own sampler and model sigma
                if b == 0 and t == ts[0]:
                    fc = fcs["gaussian_learned_M08"]
                    feats, meta = fc._prepare_features(sim_state, t)
                    import torch
                    with torch.inference_mode():
                        pred = fc.model(feats)
                    sigma = pred["pred_sigma_xy_m_k"].float().cpu().numpy()
                    old_m = fc.mf_gaussian_num_samples
                    fc.mf_gaussian_num_samples = 512
                    big = fc._sample_gaussian_noise(sigma, meta, traj_length=H, dt=DT)[b]  # [N,K,M,H,2]
                    fc.mf_gaussian_num_samples = old_m
                    moved = np.abs(big).sum(axis=(1, 2, 3, 4)) > 0
                    nz = big[moved]
                    sig = sigma[b, moved]  # [Na, K, T, 2]
                    se = sig[:, :, None, :, :].mean() / np.sqrt(512)
                    centred.append(float(np.abs(nz.mean(axis=2)).mean() / se))
                    yaw = float(meta["ego_yaw"][b])
                    c_, s_ = np.cos(-yaw), np.sin(-yaw)
                    model_steps = nz[:, :, :, 2::2][:, :, :, : sig.shape[2]]  # dst t=0.2j == model step j
                    loc = np.stack([model_steps[..., 0] * c_ - model_steps[..., 1] * s_,
                                    model_steps[..., 0] * s_ + model_steps[..., 1] * c_], -1)
                    emp = loc.std(axis=2)  # [Na, K, T', 2]
                    std_ratio.append(float(np.median(emp / sig[:, :, : emp.shape[2]])))
                    corr = np.corrcoef(loc[..., 0].ravel(), loc[..., 1].ravel())[0, 1]
                    chk["c5_xy_corr_ego_frame"] = float(corr)
                    chk["c3_sample_shape_example"] = list(big.shape)

    chk.update({k: {"max_abs_diff": max(v)} for k, v in diffs.items()})
    chk["c2_uniform_weight_value"] = 1.0 / k_modes
    chk["c3_shapes_ok"] = bool(shapes_ok)
    chk["c4_mean_abs_noise_mean_in_std_errors"] = centred
    chk["c5_median_empirical_std_over_model_sigma"] = std_ratio
    chk["c7_same_cached_samples"] = bool(all(c7))
    chk["c8_seed"] = c8
    report["scenes"] = sum(len(b[1]) for b in batches)
    report["timesteps"] = ts
    report["timing_s_per_call"] = {k: float(np.mean(v[1:])) for k, v in timings.items()}
    print(json.dumps(report, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
