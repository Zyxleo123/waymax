"""Are the K mode probabilities actually used, or do 1-2 modes take all the mass?

Uses the same validation scenes as evaluate_per_anchor.py (same split, loader
order and --max_scenes). Agents counted: valid now with at least one valid
future step (the set the mode classifier is trained on).

Per anchor and agent group (all / moving > --moving_mps / near ego):
  mean p per mode id     a mode id with ~0 mass everywhere is dead
  top-1 / best freq      how often each mode id is argmax p / the min-ADE mode
  sorted p               mean of p_(1) >= p_(2) >= ... per agent
  entropy, eff. modes    exp(H); 1 = one-hot, K = uniform
  #modes p>0.05 / p>0.01 histogram over agents
  top-1 acc, p(best)     argmax p == min-ADE mode; mass on the min-ADE mode
  calibration            top-1 accuracy per max-p bin
  spread                 mean pairwise / max-from-top-1 endpoint distance (5 s, m)

Run from multiFuture_GaussianWorldModel/ with it and its parent repo on PYTHONPATH::

  PYTHONPATH=$PWD:$PWD/.. python evaluation/evaluate_mode_probs.py \
      --cache_dir <cache> --world_model_ckpt <ckpt.pt>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.utils import resolve_cache_paths, split_cache_paths
from data.world_model_cache_loader import build_world_model_cache_loader
from model.world_model import LatentMultiAgentWorldModel, WorldModelConfig

_CAL_BINS = (0.0, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0001)


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multifuture WM mode-probability usage.")
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--world_model_ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_scenes", type=int, default=1024, help="Per anchor; same scenes for every anchor.")
    p.add_argument("--anchor_steps", type=_parse_int_tuple, default=(10, 20, 30, 40, 50, 60, 70, 80))
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--moving_mps", type=float, default=1.0)
    p.add_argument("--out_json", default=None)
    return p.parse_args()


@torch.inference_mode()
def _collect_anchor(model, loader, cfg, max_scenes, device) -> tuple[dict[str, np.ndarray], int]:
    scale = float(cfg.max_range)
    cols: dict[str, list[np.ndarray]] = {k: [] for k in ("probs", "best", "speed", "dist", "spread_pair", "spread_top1")}
    scenes = 0
    for batch in loader:
        if max_scenes > 0 and scenes >= max_scenes:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        scenes += int(batch["target_valid"].shape[0])
        out = model(batch)

        target = batch["target_delta_xy"].float()
        valid = batch["target_valid"].bool()
        pred_k = out["pred_delta_xy_norm_k"].float()
        probs = out["mode_probs"].float()
        keep = batch["agent_valid"].bool() & valid.any(dim=-1)

        err = torch.linalg.vector_norm(pred_k - target[:, :, None], dim=-1) * scale
        vf = valid.float()[:, :, None, :]
        ade_k = (err * vf).sum(dim=-1) / vf.sum(dim=-1).clamp_min(1.0)
        best = ade_k.argmin(dim=-1)

        end = pred_k[..., -1, :] * scale
        pair = torch.cdist(end.flatten(0, 1), end.flatten(0, 1)).view(*end.shape[:2], end.shape[2], end.shape[2])
        k = pair.shape[-1]
        spread_pair = pair.sum(dim=(-1, -2)) / (k * (k - 1))
        top1 = probs.argmax(dim=-1)
        spread_top1 = pair.gather(-2, top1[..., None, None].expand(-1, -1, 1, k)).squeeze(-2).amax(dim=-1)

        speed = torch.linalg.vector_norm(batch["agent_history"][:, :, -1, 2:4].float(), dim=-1) * float(cfg.max_velocity)
        dist = torch.linalg.vector_norm(batch["current_local_xy"].float(), dim=-1)

        for name, x in (
            ("probs", probs), ("best", best), ("speed", speed), ("dist", dist),
            ("spread_pair", spread_pair), ("spread_top1", spread_top1),
        ):
            cols[name].append(x[keep].cpu().numpy())
    return {k: np.concatenate(v) for k, v in cols.items()}, scenes


def _summarize(d: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    p = d["probs"][mask]
    best = d["best"][mask]
    n, k = p.shape
    if n == 0:
        return {"n_agents": 0}
    top1 = p.argmax(axis=-1)
    maxp = p.max(axis=-1)
    ent = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=-1)
    above05 = (p > 0.05).sum(axis=-1)
    above01 = (p > 0.01).sum(axis=-1)
    correct = top1 == best
    cal = []
    for lo, hi in zip(_CAL_BINS[:-1], _CAL_BINS[1:]):
        m = (maxp >= lo) & (maxp < hi)
        cal.append({"bin": f"[{lo:g},{min(hi, 1.0):g})", "frac": float(m.mean()),
                    "acc": float(correct[m].mean()) if m.any() else None,
                    "mean_maxp": float(maxp[m].mean()) if m.any() else None})
    return {
        "n_agents": int(n),
        "mean_p_by_mode": p.mean(axis=0).tolist(),
        "top1_freq_by_mode": np.bincount(top1, minlength=k).astype(float).__truediv__(n).tolist(),
        "best_freq_by_mode": np.bincount(best, minlength=k).astype(float).__truediv__(n).tolist(),
        "sorted_p_mean": np.sort(p, axis=-1)[:, ::-1].mean(axis=0).tolist(),
        "entropy": float(ent.mean()),
        "entropy_norm": float(ent.mean() / np.log(k)),
        "eff_modes": float(np.exp(ent).mean()),
        "n_above_0.05_hist": np.bincount(above05, minlength=k + 1)[1:].astype(float).__truediv__(n).tolist(),
        "n_above_0.05_mean": float(above05.mean()),
        "n_above_0.01_mean": float(above01.mean()),
        "frac_maxp_gt_0.9": float((maxp > 0.9).mean()),
        "frac_maxp_gt_0.99": float((maxp > 0.99).mean()),
        "top1_acc": float(correct.mean()),
        "p_best_mean": float(p[np.arange(n), best].mean()),
        "calibration": cal,
        "spread_pair_m": float(d["spread_pair"][mask].mean()),
        "spread_top1_max_m": float(d["spread_top1"][mask].mean()),
    }


def _groups(d: dict[str, np.ndarray], moving_mps: float, near_m: float) -> dict[str, np.ndarray]:
    return {
        "all": np.ones_like(d["best"], dtype=bool),
        "moving": d["speed"] > moving_mps,
        "near": d["dist"] <= near_m,
    }


def _fmt(v: list[float]) -> str:
    return " ".join(f"{x:.2f}" for x in v)


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.world_model_ckpt)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "latest.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = WorldModelConfig.from_dict(dict(ckpt["config"]))
    model = LatentMultiAgentWorldModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device(args.device)
    model.to(device).eval()

    paths = resolve_cache_paths(args.cache_dir, None)
    _, val_paths = split_cache_paths(paths, val_fraction=args.val_fraction)
    near_m = float(cfg.collision_focus_radius_m)

    results, scene_counts, pooled = {}, {}, []
    for anchor in args.anchor_steps:
        loader = build_world_model_cache_loader(
            args.cache_dir, cfg, cache_paths=val_paths, batch_size=args.batch_size,
            anchor_steps=(int(anchor),), shuffle=False,
        )
        d, scenes = _collect_anchor(model, loader, cfg, args.max_scenes, device)
        pooled.append(d)
        scene_counts[str(anchor)] = scenes
        results[str(anchor)] = {g: _summarize(d, m) for g, m in _groups(d, args.moving_mps, near_m).items()}
        print(f"anchor {anchor} done ({scenes} scenes)", flush=True)

    d_all = {k: np.concatenate([d[k] for d in pooled]) for k in pooled[0]}
    results["pooled"] = {g: _summarize(d_all, m) for g, m in _groups(d_all, args.moving_mps, near_m).items()}

    k = int(cfg.num_modes)
    print(f"\nckpt={ckpt_path}  K={k}  scenes/anchor={scene_counts}")
    print(f"groups: moving = speed > {args.moving_mps:g} m/s, near = within {near_m:g} m of ego\n")
    header = (
        f"{'anchor':>6} {'group':>6} {'agents':>7} {'mean p by mode id':>{5 * k}} {'top-1 freq by mode id':>{5 * k}} "
        f"{'sorted p':>{5 * k}} {'effK':>5} {'Hn':>5} {'#p>.05':>6} {'max>.9':>6} {'acc':>5} {'p(best)':>7} {'spread':>6}"
    )
    print(header)
    print("-" * len(header))
    for anchor, by_g in results.items():
        for g, r in by_g.items():
            if r["n_agents"] == 0:
                continue
            print(
                f"{anchor:>6} {g:>6} {r['n_agents']:7d} {_fmt(r['mean_p_by_mode']):>{5 * k}} "
                f"{_fmt(r['top1_freq_by_mode']):>{5 * k}} {_fmt(r['sorted_p_mean']):>{5 * k}} "
                f"{r['eff_modes']:5.2f} {r['entropy_norm']:5.2f} {r['n_above_0.05_mean']:6.2f} "
                f"{r['frac_maxp_gt_0.9']:6.2f} {r['top1_acc']:5.2f} {r['p_best_mean']:7.2f} {r['spread_pair_m']:6.2f}"
            )
        print()

    r = results["pooled"]["all"]
    print("pooled/all: best-mode freq by mode id:", _fmt(r["best_freq_by_mode"]))
    print("pooled/all: fraction of agents with exactly j modes p>0.05 (j=1..K):", _fmt(r["n_above_0.05_hist"]))
    print("pooled/all calibration (bin: frac of agents, top-1 acc):")
    for c in r["calibration"]:
        acc = f"{c['acc']:.2f}" if c["acc"] is not None else "—"
        print(f"  {c['bin']:>12}: {c['frac']:.3f}  acc {acc}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps({"ckpt": str(ckpt_path), "scenes_per_anchor": scene_counts, "results": results}, indent=2)
        )
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
