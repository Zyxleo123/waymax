from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.utils import resolve_cache_paths, split_cache_paths
from data.world_model_cache_loader import build_world_model_cache_loader
from model.world_model.config import WorldModelConfig
from model.world_model.network import LatentMultiAgentWorldModel
from train_world_model import _horizon_index

HORIZONS_S = (1.0, 3.0, 5.0)
DEAD_MODE_PROB = 0.01


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline multimodal WM-vs-CV evaluation.")
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--world_model_ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_batches", type=int, default=0, help="0 = whole validation split")
    p.add_argument("--anchor_steps", type=_parse_int_tuple, default=(20, 30, 40))
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--out_json", type=str, default=None)
    return p.parse_args()


class _Mean:
    """Running per-item mean: sum(values * mask) / sum(mask)."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0.0

    def add(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        m = mask.to(torch.float64)
        self.total += float((values.to(torch.float64) * m).sum())
        self.count += float(m.sum())

    @property
    def value(self) -> float:
        return self.total / self.count if self.count > 0 else float("nan")


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
    loader = build_world_model_cache_loader(
        args.cache_dir,
        cfg,
        cache_paths=val_paths,
        batch_size=args.batch_size,
        anchor_steps=args.anchor_steps,
        shuffle=False,
    )

    k = int(cfg.num_modes)
    max_range = float(cfg.max_range)
    horizon_idx = {f"{int(s)}s": _horizon_index(s, cfg) for s in HORIZONS_S}
    times = torch.arange(1, cfg.future_steps + 1, device=device, dtype=torch.float32) * float(cfg.future_dt)

    # Accuracy: averaged per agent (ADE) or per agent valid at the horizon (FDE).
    acc = {name: _Mean() for name in ("min_ade_m", "top1_ade_m", "cv_ade_m")}
    for tag in horizon_idx:
        for name in ("min_fde_m", "top1_fde_m", "cv_fde_m"):
            acc[f"{name}@{tag}"] = _Mean()
    # Mode health.
    winner_counts = torch.zeros(k, dtype=torch.float64)
    prob_sums = torch.zeros(k, dtype=torch.float64)
    dead_counts = torch.zeros(k, dtype=torch.float64)
    num_agents = 0.0
    entropy = _Mean()
    top1_conf = _Mean()
    sigma = {name: _Mean() for name in ("sigma_0_m", "sigma_1_m")}
    batches = 0

    with torch.inference_mode():
        for bi, batch in enumerate(loader):
            if args.max_batches > 0 and bi >= args.max_batches:
                break
            batch = {key: v.to(device) for key, v in batch.items()}
            out = model(batch)

            pred_k = out["pred_delta_xy_norm_k"].float()  # [B, N, K, T, 2] ego frame
            target = batch["target_delta_xy"].float()  # [B, N, T, 2]
            valid = batch["target_valid"].bool()  # [B, N, T]
            probs = out["mode_probs"].float()  # [B, N, K]
            top1 = out["top1_mode"]  # [B, N]

            has_future = valid.any(dim=-1)  # [B, N]
            valid_f = valid.float()
            steps = valid_f.sum(dim=-1).clamp_min(1.0)

            err_k = torch.linalg.vector_norm(pred_k - target[:, :, None], dim=-1) * max_range  # [B, N, K, T]
            ade_k = (err_k * valid_f[:, :, None]).sum(dim=-1) / steps[:, :, None]  # [B, N, K]
            best = ade_k.argmin(dim=-1)  # same winner-take-all rule as training

            acc["min_ade_m"].add(ade_k.min(dim=-1).values, has_future)
            acc["top1_ade_m"].add(ade_k.gather(-1, top1[..., None]).squeeze(-1), has_future)

            # Constant-velocity baseline from the last history step (ego frame).
            vel = batch["agent_history"][:, :, -1, 2:4].float() * float(cfg.max_velocity)
            cv_pred = vel[:, :, None, :] * times[None, None, :, None] / max_range
            cv_err = torch.linalg.vector_norm(cv_pred - target, dim=-1) * max_range  # [B, N, T]
            acc["cv_ade_m"].add((cv_err * valid_f).sum(dim=-1) / steps, has_future)

            for tag, ti in horizon_idx.items():
                v_t = valid[:, :, ti]
                err_t = err_k[..., ti]  # [B, N, K]
                acc[f"min_fde_m@{tag}"].add(err_t.min(dim=-1).values, v_t)
                acc[f"top1_fde_m@{tag}"].add(err_t.gather(-1, top1[..., None]).squeeze(-1), v_t)
                acc[f"cv_fde_m@{tag}"].add(cv_err[..., ti], v_t)

            agents = has_future
            num_agents += float(agents.sum())
            winner_counts += torch.bincount(best[agents], minlength=k).to(torch.float64).cpu()
            prob_sums += probs[agents].to(torch.float64).sum(dim=0).cpu()
            dead_counts += (probs[agents] < DEAD_MODE_PROB).to(torch.float64).sum(dim=0).cpu()
            entropy.add(-(probs * probs.clamp_min(1e-8).log()).sum(dim=-1), agents)
            top1_conf.add(probs.gather(-1, top1[..., None]).squeeze(-1), agents)

            sigma_best = out["pred_sigma_xy_m_k"].float().gather(
                2, best[:, :, None, None, None].expand(-1, -1, 1, *out["pred_sigma_xy_m_k"].shape[3:])
            ).squeeze(2)  # [B, N, T, 2]
            sigma["sigma_0_m"].add(sigma_best[..., 0], valid)
            sigma["sigma_1_m"].add(sigma_best[..., 1], valid)
            batches += 1

    if batches == 0 or num_agents == 0:
        raise RuntimeError("No validation agents were produced.")

    agent_frame = bool(getattr(cfg, "agent_frame_decoding", False))
    sigma_names = ("sigma_lon_m", "sigma_lat_m") if agent_frame else ("sigma_x_m", "sigma_y_m")
    results = {
        "checkpoint": str(ckpt_path),
        "decoder_type": getattr(cfg, "decoder_type", "per_step"),
        "agent_frame_decoding": agent_frame,
        "anchor_steps": list(args.anchor_steps),
        "batches": batches,
        "num_agents": int(num_agents),
        "num_modes": k,
        **{name: m.value for name, m in acc.items()},
        "winner_hist": (winner_counts / num_agents).tolist(),
        "mean_mode_prob": (prob_sums / num_agents).tolist(),
        "dead_mode_frac": (dead_counts / num_agents).tolist(),
        "mode_entropy": entropy.value,
        "top1_conf": top1_conf.value,
        sigma_names[0]: sigma["sigma_0_m"].value,
        sigma_names[1]: sigma["sigma_1_m"].value,
    }

    print(f"checkpoint={ckpt_path}")
    print(
        f"decoder={results['decoder_type']}  agent_frame={agent_frame}  "
        f"batches={batches}  agents={int(num_agents)}  K={k}"
    )
    print(f"{'':10s}{'ADE':>8s}" + "".join(f"{'FDE@' + tag:>9s}" for tag in horizon_idx))
    for label, prefix in ((f"min@{k}", "min"), ("top-1", "top1"), ("CV", "cv")):
        row = f"{label:10s}{results[f'{prefix}_ade_m']:8.3f}"
        row += "".join(f"{results[f'{prefix}_fde_m@{tag}']:9.3f}" for tag in horizon_idx)
        print(row)
    print("mode       " + "".join(f"{i:>7d}" for i in range(k)))
    print("winner     " + "".join(f"{v:7.3f}" for v in results["winner_hist"]))
    print("mean prob  " + "".join(f"{v:7.3f}" for v in results["mean_mode_prob"]))
    print(f"p<{DEAD_MODE_PROB:<7}  " + "".join(f"{v:7.3f}" for v in results["dead_mode_frac"]))
    print(f"mode entropy={results['mode_entropy']:.3f}  top-1 conf={results['top1_conf']:.3f}")
    print(
        f"mean {sigma_names[0]}={results[sigma_names[0]]:.3f} m  "
        f"{sigma_names[1]}={results[sigma_names[1]]:.3f} m"
    )

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
