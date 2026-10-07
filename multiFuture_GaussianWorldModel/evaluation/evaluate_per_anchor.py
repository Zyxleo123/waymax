"""Per-anchor accuracy of the multifuture WM vs constant velocity.

The WM is trained on a few anchors (e.g. 20/30/40) but ES queries it at every
replan (10, 20, ..., 80). This scores the same validation scenes at each anchor.
ADE/FDE are computed over a fixed window (first 1 s, 2 s, ...) so anchors with a
shorter remaining log stay comparable; a window past the end of the log is skipped.

Per anchor and window H (agents valid at every step up to H):
  ADE/FDE   top-1 mode mean (what ES top1 uses), best-of-K, constant velocity
  ratio     top-1 ADE / CV ADE  (< 1 means WM beats CV)
  cov1/2    fraction of targets at H inside the top-1 mode's 1σ / 2σ ellipse
            (well calibrated 2D Gaussian: 0.393 / 0.865)
  near      top-1 and CV ADE for agents within collision_focus_radius_m of the ego

Run from multiFuture_GaussianWorldModel/ with it and its parent repo on PYTHONPATH::

  PYTHONPATH=$PWD:$PWD/.. python evaluation/evaluate_per_anchor.py \
      --cache_dir <cache> --world_model_ckpt <ckpt.pt>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.utils import resolve_cache_paths, split_cache_paths
from data.world_model_cache_loader import build_world_model_cache_loader
from model.world_model import LatentMultiAgentWorldModel, WorldModelConfig, gather_mode

_COV1 = 1.0
_COV2 = 4.0
_SUMS = (
    "n", "ade1", "fde1", "adek", "fdek", "adecv", "fdecv", "cov1", "cov2",
    "n_near", "ade1_near", "adecv_near",
)


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in value.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-anchor multifuture WM vs CV accuracy.")
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--world_model_ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_scenes", type=int, default=1024, help="Per anchor; same scenes for every anchor.")
    p.add_argument("--anchor_steps", type=_parse_int_tuple, default=(10, 20, 30, 40, 50, 60, 70, 80))
    p.add_argument("--horizons_s", type=_parse_float_tuple, default=(1.0, 2.0, 3.0, 5.0))
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--out_json", default=None)
    return p.parse_args()


@torch.inference_mode()
def _eval_anchor(model, loader, cfg, horizons_s, max_scenes, device) -> tuple[dict, int]:
    acc = {h: dict.fromkeys(_SUMS, 0.0) for h in horizons_s}
    times = torch.arange(1, cfg.future_steps + 1, device=device, dtype=torch.float32) * float(cfg.future_dt)
    scale = float(cfg.max_range)
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
        top1 = out["top1_mode"]
        pred1 = gather_mode(pred_k, top1)
        sigma1 = gather_mode(out["pred_sigma_xy_norm_k"].float(), top1)

        vel = batch["agent_history"][:, :, -1, 2:4].float()
        cv = vel[:, :, None, :] * float(cfg.max_velocity) * times[None, None, :, None] / scale

        err1 = torch.linalg.vector_norm(pred1 - target, dim=-1) * scale
        errk = torch.linalg.vector_norm(pred_k - target[:, :, None], dim=-1) * scale
        errcv = torch.linalg.vector_norm(cv - target, dim=-1) * scale
        d2 = (((pred1 - target) / sigma1.clamp_min(1e-6)) ** 2).sum(dim=-1)

        near = torch.linalg.vector_norm(batch["current_local_xy"].float(), dim=-1) <= float(cfg.collision_focus_radius_m)

        for h in horizons_s:
            ti = int(round(h / float(cfg.future_dt))) - 1
            m = valid[:, :, : ti + 1].all(dim=-1)
            mn = m & near
            a = acc[h]
            a["n"] += float(m.sum())
            a["ade1"] += float(err1[:, :, : ti + 1].mean(dim=-1)[m].sum())
            a["fde1"] += float(err1[:, :, ti][m].sum())
            a["adek"] += float(errk[..., : ti + 1].mean(dim=-1).amin(dim=-1)[m].sum())
            a["fdek"] += float(errk[..., ti].amin(dim=-1)[m].sum())
            a["adecv"] += float(errcv[:, :, : ti + 1].mean(dim=-1)[m].sum())
            a["fdecv"] += float(errcv[:, :, ti][m].sum())
            a["cov1"] += float((d2[:, :, ti][m] <= _COV1).sum())
            a["cov2"] += float((d2[:, :, ti][m] <= _COV2).sum())
            a["n_near"] += float(mn.sum())
            a["ade1_near"] += float(err1[:, :, : ti + 1].mean(dim=-1)[mn].sum())
            a["adecv_near"] += float(errcv[:, :, : ti + 1].mean(dim=-1)[mn].sum())

    result = {}
    for h, a in acc.items():
        n, nn = a["n"], a["n_near"]
        if n == 0:
            result[f"{h:g}s"] = {"n": 0}
            continue
        r = {
            "n_agents": int(n),
            "top1_ade": a["ade1"] / n, "top1_fde": a["fde1"] / n,
            "minK_ade": a["adek"] / n, "minK_fde": a["fdek"] / n,
            "cv_ade": a["adecv"] / n, "cv_fde": a["fdecv"] / n,
            "cov1": a["cov1"] / n, "cov2": a["cov2"] / n,
            "n_near": int(nn),
        }
        r["ratio_ade"] = r["top1_ade"] / max(r["cv_ade"], 1e-9)
        if nn > 0:
            r["near_top1_ade"] = a["ade1_near"] / nn
            r["near_cv_ade"] = a["adecv_near"] / nn
            r["near_ratio_ade"] = r["near_top1_ade"] / max(r["near_cv_ade"], 1e-9)
        result[f"{h:g}s"] = r
    return result, scenes


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

    results, scene_counts = {}, {}
    for anchor in args.anchor_steps:
        loader = build_world_model_cache_loader(
            args.cache_dir, cfg, cache_paths=val_paths, batch_size=args.batch_size,
            anchor_steps=(int(anchor),), shuffle=False,
        )
        results[str(anchor)], scene_counts[str(anchor)] = _eval_anchor(
            model, loader, cfg, args.horizons_s, args.max_scenes, device
        )
        print(f"anchor {anchor} done ({scene_counts[str(anchor)]} scenes)", flush=True)

    print(f"\nckpt={ckpt_path}  K={cfg.num_modes}  scenes/anchor={scene_counts}")
    print("ADE/FDE in metres over the first H seconds. ratio = top-1 ADE / CV ADE (< 1: WM beats CV).")
    print("cov1/cov2 at H, ideal 0.393 / 0.865.\n")
    header = (
        f"{'anchor':>6} {'H':>3} {'agents':>7} {'top1 ADE':>8} {'top1 FDE':>8} {'minK ADE':>8} {'minK FDE':>8} "
        f"{'CV ADE':>7} {'CV FDE':>7} {'ratio':>6} {'cov1':>5} {'cov2':>5} {'near ratio':>10}"
    )
    print(header)
    print("-" * len(header))
    for anchor, by_h in results.items():
        for h, r in by_h.items():
            if r.get("n", 1) == 0:
                print(f"{anchor:>6} {h:>3} {'—':>7}  (window extends past end of log)")
                continue
            near = f"{r['near_ratio_ade']:10.2f}" if "near_ratio_ade" in r else f"{'—':>10}"
            print(
                f"{anchor:>6} {h:>3} {r['n_agents']:7d} {r['top1_ade']:8.2f} {r['top1_fde']:8.2f} "
                f"{r['minK_ade']:8.2f} {r['minK_fde']:8.2f} {r['cv_ade']:7.2f} {r['cv_fde']:7.2f} "
                f"{r['ratio_ade']:6.2f} {r['cov1']:5.2f} {r['cov2']:5.2f} {near}"
            )
        print()

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps({"ckpt": str(ckpt_path), "scenes_per_anchor": scene_counts, "results": results}, indent=2)
        )
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
