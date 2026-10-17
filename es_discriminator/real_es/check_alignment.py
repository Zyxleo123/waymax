"""Time/frame consistency of recorded ES candidates, labels and discriminator context.

Checks, per replan group of N batch files (exit code 1 if any check fails):

  origin     group_origin / group_heading equal the executed ego pose at t
  executed   the selected candidate's first `replan_interval` steps equal the executed
             ego trajectory at t+1 .. t+interval (candidate index i = world step t+1+i)
  ade        ADE between candidates and `expert_future` (same index convention, over
             the label horizon and the expert's valid steps) equals the stored gt_ade_m
  subsample  the training subsample cand_traj[:, 1:50:2] equals es_adapter.pool_world_to_ego
             on the reconstructed world-frame population (anchor at index 0)
  context    context rebuilt via es_adapter.context_from_sim_state on the reconstructed
             replaced state (original log, ego overwritten by the executed trajectory)
             equals the stored context, i.e. deployment features match training features

  python -m es_discriminator.real_es.check_alignment --data_dir .../es_candidates \
      --tfrecord_dir .../tfrecords/training --max_files 5 --out .../alignment.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from es_discriminator.analyze_es_decisions import batch_files, load_batch_state
from es_discriminator.data import CONTEXT_KEYS, DataConfig
from es_discriminator.es_adapter import context_from_sim_state, pool_world_to_ego
from es_discriminator.label_bank import _to_world

START_T = 10
TOL = {"origin_m": 1e-3, "heading_rad": 1e-3, "executed_m": 2e-2, "ade_m": 1e-3,
       "subsample_m": 1e-3, "context": 1e-4}


def recorder_config(data_dir: str) -> DataConfig:
    path = os.path.join(data_dir, "run_config.json")
    if not os.path.exists(path):
        return DataConfig()
    cc = json.load(open(path))["context_config"]
    return DataConfig(**{k: v for k, v in cc.items() if k in DataConfig.__dataclass_fields__})


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _canonical(key: str, ctx: dict) -> np.ndarray:
    """Map / light tokens are an unordered set for the model (attention without positions);
    nearest-point ties can permute them, so compare them sorted."""
    x = np.asarray(ctx[key], np.float64)
    if key in ("map", "lights"):
        m = np.asarray(ctx[key + "_mask"]).astype(bool)
        rows = x[m]
        rows = rows[np.lexsort(np.round(rows, 4).T[::-1])] if len(rows) else rows
        return np.concatenate([rows, np.zeros((len(x) - len(rows),) + x.shape[1:])])
    if key in ("map_mask", "lights_mask"):
        return np.sort(x)[::-1]
    return x


def replaced_state(st, d: dict):
    """Original log with each ego's whole trajectory replaced by the executed one (valid from START_T)."""
    log = st.log_trajectory
    is_sdc = np.asarray(st.object_metadata.is_sdc).astype(bool)
    f = {k: np.array(getattr(log, k), copy=True) for k in ("x", "y", "yaw", "vel_x", "vel_y", "valid")}
    for w in range(is_sdc.shape[0]):
        e = int(np.argmax(is_sdc[w]))
        ex = d["scen_executed_world"][w]
        for i, k in enumerate(("x", "y", "yaw", "vel_x", "vel_y")):
            f[k][w, e, START_T:] = ex[START_T:, i]
        f["valid"][w, e, START_T:] = True
    return st.replace(log_trajectory=log.replace(**f))


def check_file(path: str, tfrecord_dir: str, cfg: DataConfig, interval: int, with_context: bool) -> dict:
    d = dict(np.load(path))
    lab = dict(np.load(path[:-4] + ".labels.npz"))
    err = {k: 0.0 for k in ("origin_m", "heading_rad", "executed_m", "ade_m", "subsample_m", "context")}
    n_groups = 0
    rs = None
    if with_context:
        import jax
        rs = jax.device_get(replaced_state(load_batch_state(d, tfrecord_dir), d))
    stride, horizon = cfg.future_stride, cfg.horizon
    for g in np.flatnonzero(d["group_valid"].astype(bool)):
        n_groups += 1
        rows = np.flatnonzero(d["cand_group"] == g)
        w, t = int(d["group_scen"][g]), int(d["group_t"][g])
        ex = d["scen_executed_world"][w]
        o, hd = d["group_origin"][g], float(d["group_heading"][g])
        err["origin_m"] = max(err["origin_m"], float(np.linalg.norm(o - ex[t, :2])))
        err["heading_rad"] = max(err["heading_rad"], abs(float(_wrap(hd - ex[t, 2]))))

        sel = rows[d["cand_selected"][rows].astype(bool)][0]
        n = min(interval, 90 - t)
        sel_w = _to_world(d["cand_traj"][sel][:n].astype(np.float64), o, hd)
        err["executed_m"] = max(err["executed_m"], float(np.linalg.norm(sel_w[:, :2] - ex[t + 1:t + 1 + n, :2], axis=-1).max()))

        h = int(lab["label_horizon"][g])
        ev = d["expert_future_valid"][g][:h]
        if ev.any():
            xy = np.nan_to_num(d["cand_traj"][rows][:, :h, :2].astype(np.float64))
            ade = np.linalg.norm(xy[:, ev] - d["expert_future"][g][:h][ev][None, :, :2], axis=-1).mean(1)
            err["ade_m"] = max(err["ade_m"], float(np.nanmax(np.abs(ade - lab["gt_ade_m"][rows]))))

        cand = np.nan_to_num(d["cand_traj"][rows].astype(np.float64))
        world = _to_world(cand, o, hd)
        pool = np.concatenate([np.broadcast_to(np.array([o[0], o[1], hd]), (len(rows), 1, 3)), world], axis=1)
        if pool.shape[1] > stride * horizon:
            ego = pool_world_to_ego(pool, o, hd, cfg)
            sub = cand[:, stride - 1:stride * horizon:stride]
            dxy = np.linalg.norm(ego[..., :2] - sub[..., :2], axis=-1).max()
            dyaw = np.abs(_wrap(ego[..., 2] - sub[..., 2])).max()
            err["subsample_m"] = max(err["subsample_m"], float(max(dxy, dyaw)))

        if rs is not None:
            ctx = context_from_sim_state(rs, w, t, d["scen_goal_world"][w], cfg, use_sim_trajectory=False)
            if ctx is None:
                err["context"] = float("inf")
            else:
                rec = {k: d[k][g] for k in CONTEXT_KEYS}
                for k in CONTEXT_KEYS:
                    a, b = _canonical(k, ctx), _canonical(k, rec)
                    err["context"] = max(err["context"], float(np.abs(a - b).max()) if a.size else 0.0)
    return {"file": os.path.basename(path), "groups": n_groups, "max_err": err}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--tfrecord_dir", required=True)
    p.add_argument("--max_files", type=int, default=5)
    p.add_argument("--replan_interval", type=int, default=10)
    p.add_argument("--skip_context", action="store_true", help="skip the context check (no tfrecord load)")
    p.add_argument("--out", default="")
    a = p.parse_args()
    cfg = recorder_config(a.data_dir)
    files = batch_files(a.data_dir)
    if a.max_files:
        idx = np.linspace(0, len(files) - 1, min(a.max_files, len(files))).round().astype(int)
        files = [files[i] for i in sorted(set(idx))]
    per_file = [check_file(f, a.tfrecord_dir, cfg, a.replan_interval, not a.skip_context) for f in files]
    worst = {k: max(r["max_err"][k] for r in per_file) for k in TOL}
    if a.skip_context:
        worst.pop("context")
    failed = {k: v for k, v in worst.items() if not v <= TOL[k]}
    out = {"files": len(per_file), "groups": sum(r["groups"] for r in per_file), "tolerance": TOL,
           "max_err": worst, "failed": failed, "per_file": per_file}
    print(json.dumps({k: out[k] for k in ("files", "groups", "max_err", "failed")}, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
    if failed:
        raise SystemExit(f"alignment check FAILED: {failed}")


if __name__ == "__main__":
    main()
