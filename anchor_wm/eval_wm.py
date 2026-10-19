"""Forecasting comparison on WOMD validation shards: AnchorWM checkpoints vs an MF-Gaussian checkpoint.

Same batches for every model (`data.build_arrays` reproduces MF's inputs and targets). Agents =
valid non-SDC dynamic agents with a future. Metrics per model:
  min_ade_6 / min_fde_6 / miss_rate_6   over the 6 planner modes (AnchorWM: endpoint-NMS picks)
  top1_ade / top1_fde                   most probable mode
  safety_recall                         interacting agents (logged box within 3 m of the logged ego
                                        box at some step) whose position at the closest-approach
                                        step is within 2 m of a planner mode with p >= 0.05
                                        (p renormalized over the 6 planner modes)
  endpoint_spread_m                     mean pairwise endpoint distance of the 6 modes
  jerk_top1                             mean |third difference| of the top-1 path (m/s^3)
AnchorWM also reports min_ade_all / min_fde_all over every anchor.

  python -m anchor_wm.eval_wm --cache_dir ... --mf_ckpt .../epoch_0450.pt --ckpts a1/final.pt a2/final.pt
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

from anchor_wm.data import make_loader, resolve_cache_paths, split_cache_paths
from anchor_wm.eval_bank import load_model
from anchor_wm.geometry import box_corners, sat_gap
from anchor_wm.losses import ade_fde
from anchor_wm.model import select_modes

REPO = Path(__file__).resolve().parents[1]


def load_mf(ckpt_path: str, device: torch.device):
    """MF-Gaussian network from its own package (read-only import under a private alias)."""
    alias = "_anchor_wm_mf"
    if alias not in sys.modules:
        pkg = types.ModuleType(alias)
        pkg.__path__ = [str(REPO / "multiFuture_GaussianWorldModel" / "model" / "world_model")]
        sys.modules[alias] = pkg
    cfg_mod = importlib.import_module(f"{alias}.config")
    net_mod = importlib.import_module(f"{alias}.network")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = net_mod.LatentMultiAgentWorldModel(cfg_mod.WorldModelConfig.from_dict(dict(ck["config"])))
    model.load_state_dict(ck["model_state_dict"])
    return model.to(device).eval()


class _Acc:
    def __init__(self):
        self.s, self.n = {}, {}

    def add(self, key: str, values: torch.Tensor, mask: torch.Tensor) -> None:
        self.s[key] = self.s.get(key, 0.0) + float((values * mask).sum())
        self.n[key] = self.n.get(key, 0.0) + float(mask.sum())

    def result(self) -> dict:
        return {k: self.s[k] / max(self.n[k], 1.0) for k in self.s}


def interacting(f: dict, threshold_m: float = 3.0) -> tuple[torch.Tensor, torch.Tensor]:
    """[B,N] interacting-agent mask and the closest-approach step [B,N] (logged ego vs logged agent)."""
    b, n, t, _ = f["fut_xy"].shape
    ego_c = box_corners(f["ego_future_xy"].float(), f["ego_future_yaw"].float(),
                        f["ego_lw"].float()[:, None].expand(b, t, 2))[:, None]            # [B,1,T,4,2]
    oc = box_corners(f["fut_xy"].float(), f["fut_yaw"].float(), f["fut_lw"].float())
    gap = sat_gap(ego_c, f["ego_future_yaw"].float()[:, None], oc, f["fut_yaw"].float())  # [B,N,T]
    ok = f["target_valid"].bool() & f["ego_future_valid"].bool()[:, None]
    gap = gap.masked_fill(~ok, float("inf"))
    gmin, tstar = gap.min(-1)
    return gmin < threshold_m, tstar


def metrics_for(pred_k: torch.Tensor, probs: torch.Tensor, sel: torch.Tensor, f: dict, acc: _Acc, rng: float,
                inter: torch.Tensor, tstar: torch.Tensor, dt: float) -> None:
    valid = f["target_valid"].bool()
    target = f["target_delta_xy"].float()
    ade, fde, has = ade_fde(pred_k, target, valid, rng)
    m = has.float()
    a6, f6 = ade.gather(-1, sel), fde.gather(-1, sel)
    top1 = probs.argmax(-1, keepdim=True)
    acc.add("min_ade_6", a6.amin(-1), m)
    acc.add("min_fde_6", f6.amin(-1), m)
    acc.add("miss_rate_6", (f6.amin(-1) > 2.0).float(), m)
    acc.add("top1_ade", ade.gather(-1, top1).squeeze(-1), m)
    acc.add("top1_fde", fde.gather(-1, top1).squeeze(-1), m)
    if pred_k.shape[2] > sel.shape[-1]:
        acc.add("min_ade_all", ade.amin(-1), m)
        acc.add("min_fde_all", fde.amin(-1), m)
    # Safety recall at the closest-approach step.
    p6 = pred_k.gather(2, sel[..., None, None].expand(-1, -1, -1, pred_k.shape[3], 2))           # [B,N,6,T,2]
    ts = tstar[:, :, None, None, None].expand(-1, -1, p6.shape[2], 1, 2)
    err = (p6.gather(3, ts).squeeze(3) - target.gather(2, tstar[..., None, None].expand(-1, -1, 1, 2))) .norm(dim=-1) * rng
    # Probabilities renormalized over the planner modes, as the planner sees them (identity for MF's 6 modes).
    prob6 = probs.gather(-1, sel)
    prob6 = prob6 / prob6.sum(-1, keepdim=True).clamp_min(1e-9)
    covered = ((err < 2.0) & (prob6 >= 0.05)).any(-1)
    im = (inter & has).float()
    acc.add("safety_recall", covered.float(), im)
    acc.add("interacting_agents_per_scene", im.sum(1), torch.ones_like(im[:, 0]))
    end = p6[..., -1, :] * rng
    pd = (end[:, :, :, None] - end[:, :, None]).norm(dim=-1)
    k = pd.shape[-1]
    acc.add("endpoint_spread_m", pd.sum((-1, -2)) / (k * (k - 1)), m)
    path = pred_k.gather(2, top1[..., None, None].expand(-1, -1, 1, pred_k.shape[3], 2)).squeeze(2) * rng
    jerk = torch.diff(path, n=3, dim=-2).norm(dim=-1).mean(-1) / dt ** 3
    acc.add("jerk_top1", jerk, m)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--mf_ckpt", default="")
    p.add_argument("--ckpts", nargs="*", default=[])
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--max_batches", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--anchor_steps", type=int, nargs="+", default=[20, 30, 40])
    p.add_argument("--out", default="")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    models = {}
    if args.mf_ckpt:
        models["mf_gaussian"] = ("mf", load_mf(args.mf_ckpt, device))
    for c in args.ckpts:
        models[c] = ("anchor", load_model(c, device))
    if not models:
        raise SystemExit("nothing to evaluate")
    cfg = next(m for kind, m in models.values() if kind == "anchor").config if any(
        k == "anchor" for k, _ in models.values()) else None
    if cfg is None:
        from anchor_wm.config import AnchorWMConfig
        cfg = AnchorWMConfig()
    _, val_paths = split_cache_paths(resolve_cache_paths(args.cache_dir), args.val_fraction)
    loader = make_loader(val_paths, cfg, batch_size=args.batch_size, anchor_steps=tuple(args.anchor_steps),
                         shuffle=False, seed=0, num_workers=4)
    accs = {name: _Acc() for name in models}
    rng = float(cfg.max_range)
    for i, batch in enumerate(loader):
        if i >= args.max_batches:
            break
        f = {k: v.to(device) for k, v in batch.items()}
        inter, tstar = interacting(f)
        for name, (kind, model) in models.items():
            if kind == "mf":
                out = model(f)
                pred, probs = out["pred_delta_xy_norm_k"].float(), out["mode_probs"].float()
                sel = torch.arange(pred.shape[2], device=device).expand(*probs.shape[:2], -1)
            else:
                _, dec = model(f)
                pred, probs = dec["pred_xy_k"].float(), dec["probs"].float()
                sel = select_modes(dec, cfg.num_out_modes, cfg.nms_dist_m, rng)
            metrics_for(pred, probs, sel, f, accs[name], rng, inter, tstar, cfg.future_dt)
        if (i + 1) % 50 == 0:
            print(f"{i + 1} batches", flush=True)
    report = {name: {k: round(v, 4) for k, v in a.result().items()} for name, a in accs.items()}
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
