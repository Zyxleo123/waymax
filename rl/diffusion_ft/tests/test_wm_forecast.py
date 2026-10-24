"""Gates for the learned world-model neighbour sources (``world_model_forecast.py``).

WHY. The repair's honest neighbour source was constant velocity; ``wm_det`` /
``wm_mf`` / ``wm_mf_top1`` replace it with the es_WM world models, vendored under
``wm_vendor/``. Before a GPU is spent on the select eval, pin the plumbing on REAL
scenes advanced by expert log replay (so ``sim_trajectory`` holds real history):

* **W0 isolation** -- importing the vendored WMs leaves this repo's ``data``
  untouched (ours has ``compute_world_dt_seconds(method=)``), while the vendored
  code bound THEIR ``_build_map_features`` (it takes ``max_segments=``).
* **W1 decode parity** -- our vectorized decode at ``k * 0.1 s`` equals theirs
  (``WorldModelForecaster._predict_all_worlds``, dt 0.1) on the same input, per
  mode for ``wm_mf``, within 1 mm / 1 mrad; masks identical.
* **W2 weights** -- ``wm_mf`` weights sum to 1 per agent; the ego and agents
  invalid at t0 are masked.
* **W3 plumbing** -- the external-box path of ``candidate_safety`` reproduces cv
  EXACTLY when handed cv's own boxes (weight 1), and again when they are split into
  K copies of weight 1/K (the loss is linear in the weights; ``c_agent`` unchanged).
* **W4 accuracy** -- under log replay the recorded future is the truth: over the
  next 2 s, ``wm_det``'s ADE and the multi-future model's BEST mode's ADE must not
  be grossly worse than cv (<= 1.5x cv). A frame / rotation / yaw bug costs metres
  on every mode, not centimetres. NOT gated: the MF top-1 and probability-weighted
  ADE -- those measure the model's mode probabilities, which are a property of the
  checkpoint (the first smoke, job 80121: top-1 0.60 m vs cv 0.34 m, best mode
  0.19 m), not of this plumbing. Reported per source.
* **W5 the optimizer** -- ``o_am``'s optimizer on a distorted expert window, per
  wm source: finite, anchor kept, loss never above the init's; and the WM ran
  ONCE for the replan although two starts and the scorer all asked.

CPU (torch + jax), no policy. Submitted by ``submit_wm_posthoc.sh`` (first link).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from waymax import config as waymax_config
from waymax import datatypes
from waymax import dynamics as waymax_dynamics
from waymax import env as waymax_env

from rl.diffusion_ft.branching import lateral_shift
from rl.diffusion_ft.lateral_opt import LateralOptConfig, edit_world, make_lateral_optimizer
from rl.diffusion_ft.proposal_select import make_scorer
from rl.diffusion_ft.safety_objective import (WM_PREDICTIONS, SafetyObjectiveConfig,
                                              candidate_safety, predicted_neighbor_window)
from rl.diffusion_ft.scene_sampler import DEFAULT_INDEX_DIR, SceneSampler
from rl.diffusion_ft import world_model_forecast as wmf

FAILS: list[str] = []
DT = 0.1
AM_CFG = "lon=1,w_lon_prox=0.05,lon_back_m=1000"


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def ego_window(state, t0: int, W: int):
    """The ego's own log window ``[B, W, 5]`` (x, y, yaw, vx, vy), anchor first."""
    log = state.log_trajectory
    e = jnp.argmax(jnp.asarray(state.object_metadata.is_sdc).astype(jnp.int32), 1)

    def row(f):
        return jnp.take_along_axis(jnp.asarray(f), e[:, None, None], axis=1)[:, 0, t0:t0 + W]
    return jnp.stack([row(log.x), row(log.y), row(log.yaw), row(log.vel_x), row(log.vel_y)], -1)


def theirs(fc, state, t0, n):
    """Their decode at dt 0.1 on the same driven-history input, as [B, n, N*K, 5] in our order."""
    hist = state.replace(log_trajectory=state.sim_trajectory)
    if fc.source == "wm_det":
        traj, mask = fc.base._predict_all_worlds(hist, timestep=t0, traj_length=n + 1, dt=DT)[:2]
        traj = traj[:, :, None]                                              # [B, N, 1, H, 5]
    else:
        out = fc.base._predict_all_worlds(hist, timestep=t0, traj_length=n + 1, dt=DT,
                                          decode_all_modes=True)
        traj, mask = ((out["traj_k"], out["masks"]) if fc.source == "wm_mf"
                      else (out["top1_traj"][:, :, None], out["masks"]))
    B, N, K = traj.shape[:3]
    traj = traj[:, :, :, 1:n + 1][..., [0, 1, 3, 4, 2]]                   # -> x, y, len, wid, yaw
    box = np.moveaxis(traj, 3, 1).reshape(B, n, N * K, 5)
    valid = np.broadcast_to(mask[:, :, None, 1:n + 1], (B, N, K, n))
    return box, np.moveaxis(valid, 3, 1).reshape(B, n, N * K)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits_json", required=True)
    p.add_argument("--split", default="failure_heldout")
    p.add_argument("--shard_glob",
                   default="/data/user_data/yixiz/autonlab/zfsauton/datasets/womd/tf_example/training/training_tfexample.tfrecord-*")
    p.add_argument("--index_dir", default=DEFAULT_INDEX_DIR)
    p.add_argument("--n_scenes", type=int, default=8)
    p.add_argument("--t0s", default="20,40,60")
    p.add_argument("--n_steps", type=int, default=20)
    p.add_argument("--width", type=int, default=41)
    p.add_argument("--sources", default=",".join(WM_PREDICTIONS))
    p.add_argument("--wm_det_ckpt", default="")
    p.add_argument("--wm_mf_ckpt", default="")
    p.add_argument("--verdict", default="")
    p.add_argument("--json_out", default="")
    args = p.parse_args(argv)
    sources = [s for s in args.sources.split(",") if s]
    wmf.configure(det_ckpt=args.wm_det_ckpt, mf_ckpt=args.wm_mf_ckpt, device="cpu")
    report = {"sources": sources, "ckpts": wmf.describe()["ckpts"]}

    # ---- W0 isolation ------------------------------------------------------ #
    import inspect
    from data import preprocess as ours
    fcs = {s: wmf.get_forecaster(s) for s in sources}
    import torch
    for pkg, path in wmf.DEFAULT_CKPTS.items():
        ck = wmf._CONFIG[pkg]
        meta = torch.load(ck, map_location="cpu", weights_only=False)
        report[f"ckpt_{pkg}"] = {"path": ck, "epoch": meta.get("epoch"),
                                 "keys": sorted(k for k in meta if k != "model_state_dict")}
        print(f"[wm] {pkg}: {ck} epoch={meta.get('epoch')} keys={report[f'ckpt_{pkg}']['keys']}")
        del meta
    from data import preprocess as ours_after
    vendored = [sys.modules[m]._build_map_features for m in ("wm_det.inference", "wm_mf.inference")
                if m in sys.modules]
    check("W0 isolation",
          ours_after is ours and bool(vendored)
          and "method" in inspect.signature(ours.compute_world_dt_seconds).parameters
          and "max_segments" not in inspect.signature(ours._build_map_features).parameters
          and all("max_segments" in inspect.signature(f).parameters for f in vendored),
          f"ours {ours.__file__}, theirs {[inspect.getsourcefile(f) for f in vendored]}")

    # ---- scenes ------------------------------------------------------------ #
    spec = json.loads(Path(args.splits_json).read_text())
    ids = [int(i) for i in spec["splits"][args.split]["scene_ids"][:args.n_scenes]]
    sampler = SceneSampler(args.shard_glob, batch_size=1, eval_size=0,
                           include_sdc_paths=False, prefetch=False, index_dir=args.index_dir)
    scen = sampler.load_explicit(ids)
    n_obj = int(np.asarray(scen.log_trajectory.x).shape[1])
    env = waymax_env.PlanningAgentEnvironment(
        dynamics_model=waymax_dynamics.StateDynamics(),
        config=waymax_config.EnvironmentConfig(
            max_num_objects=n_obj, controlled_object=waymax_config.ObjectType.SDC))
    st0 = env.reset(jax.tree_util.tree_map(jnp.asarray, scen))
    B, n, W = len(ids), int(args.n_steps), int(args.width)
    objcfg = SafetyObjectiveConfig()
    print(f"[gate] {B} {args.split} scenes, {n_obj} objects, t0s {args.t0s}, n {n}", flush=True)

    log = scen.log_trajectory
    lx, ly = np.asarray(log.x), np.asarray(log.y)
    lv = np.asarray(log.valid)
    is_sdc = np.asarray(scen.object_metadata.is_sdc)

    acc = {s: [] for s in ["cv"] + sources + (["wm_mf_min"] if "wm_mf" in sources else [])}
    par_err = {s: 0.0 for s in sources}
    mask_bad = {s: 0 for s in sources}
    w2_bad = 0
    w3 = []
    opt_bad = {s: [] for s in sources}
    t_wm, t_opt = {s: [] for s in sources}, {s: [] for s in sources}
    opt_am = {s: make_lateral_optimizer(objcfg, LateralOptConfig.parse(AM_CFG), dt=DT,
                                        neighbor_pred=s, world_dt_method="median")
              for s in sources}
    score = {s: make_scorer(objcfg, s, "median") for s in sources}
    dt_b = np.full(B, DT)

    for t0 in [int(t) for t in args.t0s.split(",")]:
        st = datatypes.update_state_by_log(st0, int(t0 - int(st0.timestep)))
        assert int(st.timestep) == t0
        t0j = jnp.asarray(t0, jnp.int32)
        # Truth: agents valid at t0 and over the whole window, not the ego.
        fut = slice(t0 + 1, t0 + n + 1)
        ok = lv[:, :, t0] & lv[:, :, fut].all(-1) & ~is_sdc                  # [B, N]
        gt = np.stack([lx[:, :, fut], ly[:, :, fut]], -1)                     # [B, N, n, 2]

        cv_box, _ = predicted_neighbor_window(st, t0j, n, mode="cv", dt_b=jnp.asarray(dt_b))
        cv_xy = np.moveaxis(np.asarray(cv_box)[..., :2], 1, 2)                # [B, N, n, 2]
        err = np.linalg.norm(cv_xy - gt, axis=-1)
        acc["cv"].append((err.mean(-1)[ok], err[..., -1][ok]))

        for s in sources:
            fc = fcs[s]
            t = time.time()
            box, valid, weight = (np.asarray(a) for a in fc(st, t0j, n, dt_b))
            t_wm[s].append(time.time() - t)
            K = box.shape[2] // n_obj
            # W1 parity
            tb, tv = theirs(fc, st, t0, n)
            e_xy = np.abs(box[..., :2] - tb[..., :2])
            e_yaw = np.abs(np.angle(np.exp(1j * (box[..., 4] - tb[..., 4]))))
            e_sz = np.abs(box[..., 2:4] - tb[..., 2:4])
            vm = tv
            par_err[s] = max(par_err[s], float(max(e_xy[vm].max(initial=0), e_yaw[vm].max(initial=0),
                                                   e_sz[vm].max(initial=0))))
            mask_bad[s] += int((valid != tv).sum())
            # W2 weights / masks
            wk = weight[:, 0].reshape(B, n_obj, K)
            vk = valid[:, 0].reshape(B, n_obj, K)[..., 0]
            cur_ok = lv[:, :, t0] & ~is_sdc
            w2_bad += int(np.sum(np.abs(wk.sum(-1) - 1.0)[cur_ok] > 1e-4)
                          + np.sum(vk != cur_ok))
            # W4 accuracy
            xy = np.moveaxis(box[..., :2].reshape(B, n, n_obj, K, 2), 1, 3)  # [B, N, K, n, 2]
            ek = np.linalg.norm(xy - gt[:, :, None], axis=-1)                 # [B, N, K, n]
            if s == "wm_mf":
                acc[s].append(((ek.mean(-1) * wk).sum(-1)[ok], (ek[..., -1] * wk).sum(-1)[ok]))
                kmin = ek.mean(-1).argmin(-1)
                pick = np.take_along_axis(ek, kmin[..., None, None], 2)[:, :, 0]
                acc["wm_mf_min"].append((pick.mean(-1)[ok], pick[..., -1][ok]))
            else:
                acc[s].append((ek[:, :, 0].mean(-1)[ok], ek[:, :, 0, -1][ok]))

            # W5 the optimizer on a distorted expert window, two starts + the scorer
            world = lateral_shift(ego_window(st, t0, W), 1.5, ramp_steps=10, dt=DT)
            q0 = fc.n_queries
            t = time.time()
            for _ in range(2):
                x, lb, l0 = opt_am[s](world, jnp.zeros((B, 2 * n)), st, t0j,
                                      jnp.ones((B,), jnp.float32), n)
                jax.block_until_ready(x)
            t_opt[s].append((time.time() - t) / 2)
            w_rep = edit_world(world, x, n=n, dt=DT, lon=1)
            score[s](w_rep, st, t0j, jnp.asarray(np.asarray(world)[:, -1, :2]), n)
            x, lb, l0, w_rep = (np.asarray(a) for a in (x, lb, l0, w_rep))
            opt_bad[s].append(dict(
                nonfinite=int((~np.isfinite(x)).sum() + (~np.isfinite(lb)).sum()),
                anchor=float(np.abs(w_rep[:, 0, :2] - np.asarray(world)[:, 0, :2]).max()),
                ascent=int((lb > l0 + 1e-6).sum()), extra_queries=int(fc.n_queries - q0),
                moved=int((np.abs(x).max(1) > 0).sum()), loss0=float(l0.mean()),
                loss=float(lb.mean())))

        # W3 plumbing: cv's boxes through the external path, whole and split in K
        cand = ego_window(st, t0, n + 1)
        ref = candidate_safety(objcfg, cand, state=st, t0=t0j, n_steps=n, neighbor_pred="cv",
                               neighbor_dt_b=jnp.asarray(dt_b))
        cvb, cvv = predicted_neighbor_window(st, t0j, n, mode="cv", dt_b=jnp.asarray(dt_b))
        one = candidate_safety(objcfg, cand, state=st, t0=t0j, n_steps=n, neighbor_pred="wm_det",
                               neighbor_boxes=(cvb, cvv, jnp.ones(cvv.shape, jnp.float32)))
        K = 3
        split = candidate_safety(objcfg, cand, state=st, t0=t0j, n_steps=n,
                                 neighbor_pred="wm_mf",
                                 neighbor_boxes=(jnp.repeat(cvb, K, axis=2), jnp.repeat(cvv, K, axis=2),
                                                 jnp.full(cvv.shape[:2] + (cvv.shape[2] * K,), 1.0 / K)))
        w3.append(dict(
            exact=float(np.abs(np.asarray(one["loss_b"]) - np.asarray(ref["loss_b"])).max()
                        + np.abs(np.asarray(one["c_agent"]) - np.asarray(ref["c_agent"])).max()),
            split=float(np.abs(np.asarray(split["loss_b"]) - np.asarray(ref["loss_b"])).max()
                        / max(1e-6, float(np.abs(np.asarray(ref["loss_b"])).max()))),
            split_c=float(np.abs(np.asarray(split["c_agent"]) - np.asarray(ref["c_agent"])).max()),
            nn=float(abs(float(split["n_neighbors"]) - float(ref["n_neighbors"])))))

    # ---- verdicts ---------------------------------------------------------- #
    for s in sources:
        check(f"W1 decode parity {s}", par_err[s] < 1e-3 and mask_bad[s] == 0,
              f"max |ours - theirs| {par_err[s]:.2e}, mask mismatches {mask_bad[s]}")
    check("W2 weights sum to 1, ego/invalid masked", w2_bad == 0, f"{w2_bad} bad")
    check("W3 external boxes == cv (exact; split K=3 relative)",
          all(r["exact"] == 0.0 and r["split"] < 1e-5 and r["split_c"] == 0.0 and r["nn"] < 1e-4
              for r in w3), json.dumps(w3))
    ade = {k: (float(np.mean(np.concatenate([a for a, _ in v]))),
               float(np.mean(np.concatenate([f for _, f in v])))) for k, v in acc.items()}
    n_ag = int(sum(len(a) for a, _ in acc["cv"]))
    for k, (a, f) in ade.items():
        print(f"[acc] {k:12s} ADE {a:.3f} m  FDE@{n * DT:.1f}s {f:.3f} m  ({n_ag} agent-windows)")
    for s in [s for s in ("wm_det", "wm_mf_min") if s in ade]:
        check(f"W4 {s} not grossly worse than cv", ade[s][0] <= 1.5 * ade["cv"][0],
              f"ADE {ade[s][0]:.3f} vs cv {ade['cv'][0]:.3f}")
    for s in sources:
        r = opt_bad[s]
        check(f"W5 optimizer {s}",
              all(x["nonfinite"] == 0 and x["anchor"] < 1e-5 and x["ascent"] == 0
                  and x["extra_queries"] <= 1 for x in r), json.dumps(r))
        print(f"[time] {s}: WM forward+decode {np.mean(t_wm[s]):.2f}s (first {t_wm[s][0]:.2f}s), "
              f"optimizer call {np.mean(t_opt[s][1:] or t_opt[s]):.2f}s (CPU)")
    report.update(ade=ade, n_agent_windows=n_ag, parity=par_err, w3=w3, opt=opt_bad,
                  t_wm={s: float(np.mean(v)) for s, v in t_wm.items()},
                  t_opt={s: float(np.mean(v)) for s, v in t_opt.items()}, fails=FAILS)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=1, default=str))
    verdict = "PASS" if not FAILS else "FAIL: " + "; ".join(FAILS)
    print(f"=== {verdict} ===")
    if args.verdict:
        Path(args.verdict).write_text(verdict + "\n")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
