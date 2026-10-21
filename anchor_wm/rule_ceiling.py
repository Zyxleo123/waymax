"""How far the scorer's rule features (model.rule_features) rank safety on recorded bank groups.

Scores every candidate of the bank's val groups with hand-set combinations of the per-candidate rule
summaries (computed from the checkpoint's own forecasts) and with the checkpoint's learned total, and
reports bank_eval_metrics for each, so the learned scorer can be compared with what the features alone
give before fine-tuning on them.

  python -m anchor_wm.rule_ceiling --ckpt .../best.pt --bank_dir .../es_candidates --cache_dir ... \
      --split_json .../scenario_split_seed0.json --invalid_json .../invalid_at_start.json --groups 400
"""

from __future__ import annotations

import argparse
import json
import math

import torch

from anchor_wm import finetune
from anchor_wm.bank_data import build_index, make_bank_loader
from anchor_wm.geometry import path_yaw
from anchor_wm.losses import bank_eval_metrics, bank_targets
from anchor_wm.model import rule_features
from anchor_wm.train import move

# Per-candidate rule summaries (model.rule_features "end"): 0 p_soft max, 1 p_hard max, 2 p_near max,
# 3 offroad max, 4 min gap / 5, 5 max signed distance / 2, 6 time to first violation, 7 mean p_soft.
COMBOS = {
    "hard": lambda e: -(e[..., 1] + e[..., 3]),
    "soft": lambda e: -(e[..., 0] + e[..., 3]),
    "soft+ttv": lambda e: -(e[..., 0] + e[..., 3]) + 0.5 * e[..., 6],
    "geometry": lambda e: e[..., 4] - e[..., 5],
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("ckpt", "bank_dir", "cache_dir", "split_json"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--invalid_json", default="")
    p.add_argument("--split", default="val")
    p.add_argument("--groups", type=int, default=400)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--path_heading", action="store_true",
                   help="rule features from the travel direction from xy instead of the recorded headings")
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fa = type("A", (), dict(init_ckpt=args.ckpt, no_goal_features=False, no_route_features=False, no_aux_heads=False,
                            rule_features=True, scorer_agents=32, scorer_modes=6))()
    model, _ = finetune.build_model(fa, device)
    model.eval()
    index, counts = build_index(args.bank_dir, args.cache_dir, args.split_json, args.invalid_json, (args.split,))
    loader = make_bank_loader(index, args.cache_dir, batch_groups=4, cands_per_group=0, shuffle=False, seed=0,
                              num_workers=args.num_workers)
    sums, n = {}, 0
    with torch.no_grad():
        for batch in loader:
            f = move(batch, device)
            enc, dec = model(f)
            dec = {k: v.float() for k, v in dec.items()}
            ctx = model.scorer.context(enc, dec, f, 0.0)
            learned, _ = model.scorer(ctx, f["cand_xy"], f["cand_yaw"], chunk=64, heading=f["cand_yaw"])
            cand_yaw = path_yaw(f["cand_xy"].float(), f["cand_yaw"].float()) if args.path_heading else f["cand_yaw"].float()
            ends = [rule_features(f["cand_xy"][:, s:s + 64], cand_yaw[:, s:s + 64], ctx["ego_lw"], ctx["wm"],
                                  ctx["label_edges"])[1] for s in range(0, f["cand_xy"].shape[1], 64)]
            end = torch.cat(ends, 1)
            tg = bank_targets(f)
            # Same summaries with the steps past the label horizon (scenario end) dropped.
            t = f["cand_xy"].shape[2]
            in_h = (2 * torch.arange(1, t + 1, device=device))[None] <= f["horizon"][:, None]          # [B,T]
            steps = torch.cat([rule_features(f["cand_xy"][:, s:s + 64], cand_yaw[:, s:s + 64], ctx["ego_lw"],
                                             ctx["wm"], ctx["label_edges"])[0] for s in range(0, f["cand_xy"].shape[1], 64)], 1)
            m = in_h[:, None, :, None]
            p_soft = torch.where(m[..., 0], steps[..., 2], torch.zeros_like(steps[..., 2])).amax(-1)
            off = torch.where(m[..., 0], steps[..., 6], torch.zeros_like(steps[..., 6])).amax(-1)
            p_hard = torch.where(m[..., 0], steps[..., 3], torch.zeros_like(steps[..., 3])).amax(-1)
            # The cache's log-replay geometry (0.1 s, already masked to the horizon): the label ceiling.
            cache = -(torch.sigmoid(-f["b_gap"].float() / 0.5).amax(-1) + torch.sigmoid(f["b_offroad"].float() / 0.5).amax(-1))
            cache_off = -torch.sigmoid(f["b_offroad"].float() / 0.5).amax(-1)
            scores = {"learned": learned.float(), **{k: fn(end) for k, fn in COMBOS.items()},
                      "hard_in_horizon": -(p_hard + off), "soft_in_horizon": -(p_soft + off),
                      "offroad_in_horizon": -off, "cache_geometry": cache, "cache_offroad": cache_off}
            full = f["horizon"] >= 2 * t
            for name, sc in scores.items():
                for sub, sel in (("", slice(None)), ("full_horizon/", full)):
                    if sub and not bool(full.any()):
                        continue
                    nb = int(sc[sel].shape[0])
                    for k, v in bank_eval_metrics(sc[sel], {key: val[sel] for key, val in tg.items()}).items():
                        if math.isfinite(v):
                            sums[f"{sub}{name}/{k}"] = sums.get(f"{sub}{name}/{k}", 0.0) + v * nb
                            sums[f"{sub}{name}/{k}#n"] = sums.get(f"{sub}{name}/{k}#n", 0) + nb
            b = learned.shape[0]
            n += b
            if n >= args.groups:
                break
    out = {"ckpt": args.ckpt, "index": counts, "groups": n,
           **{k: round(v / sums[k + "#n"], 4) for k, v in sorted(sums.items()) if not k.endswith("#n")}}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
