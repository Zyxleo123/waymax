"""Fine-tune the AnchorWM scorer on a recorded Diffusion-ES bank (cache_bank output + Waymax labels).

Starts from a trained AnchorWM checkpoint (e.g. A2's final.pt), adds the scorer features of the
fine-tuning plan (goal geometry, route context, time-to-violation / goal-distance heads and, with
--rule_features, the rule reward's exact collision / offroad geometry; all start at zero output, so
step 0 scores exactly like the checkpoint) and trains with losses.bank_score_loss:
B0's safe>unsafe and goal-gain pairs, unsafe-vs-unsafe pairs by time to first violation and severity,
goal-precision pairs, a time-weighted listwise reward and soft / auxiliary targets. A fraction of steps
use the self-generated WOMD candidates (train.scorer_step) so decoded-anchor candidates stay covered.

The world model (encoder + anchor decoder) is frozen by default; --unfreeze_decoder trains the decoder
with a small learning rate and lets the scoring gradient reach it (the joint ablation).

  python -m anchor_wm.finetune --init_ckpt .../joint_<ts>/final.pt --bank_dir .../es_candidates \
      --cache_dir .../bank_cache/<set> --split_json .../scenario_split_seed0.json \
      --invalid_json .../invalid_at_start.json --save_dir .../anchor_wm/finetune --run_name ft_v1
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from anchor_wm.bank_data import build_index, make_bank_loader
from anchor_wm.config import AnchorWMConfig
from anchor_wm.data import make_loader, resolve_cache_paths, split_cache_paths
from anchor_wm.losses import bank_eval_metrics, bank_score_loss, bank_targets
from anchor_wm.model import ENCODER_PREFIXES, AnchorWorldModel
from anchor_wm.train import move, scorer_step

NEW_SCORER_PREFIXES = ("scorer.goal_step.", "scorer.goal_end.", "scorer.route_step.", "scorer.ttv_head.",
                       "scorer.goaldist_head.", "scorer.rule_step.", "scorer.rule_end.")
DECODER_PREFIXES = ("anchor_embed.", "goal_embed.", "query_fuse.", "traj_head.", "traj_embed.", "mode_cls.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init_ckpt", required=True)
    p.add_argument("--resume", default="", help="a latest.pt / final.pt of this script: continue its weights, optimizer, "
                                               "step and learning-rate schedule (same --max_steps and model flags)")
    p.add_argument("--bank_dir", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--split_json", required=True)
    p.add_argument("--invalid_json", default="")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--run_name", default="ft")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=30000)
    p.add_argument("--max_hours", type=float, default=0.0)
    p.add_argument("--batch_groups", type=int, default=16)
    p.add_argument("--cands_per_group", type=int, default=64)
    p.add_argument("--max_unsafe_frac", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--decoder_lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--no_amp", action="store_true")
    # model additions
    p.add_argument("--no_goal_features", action="store_true")
    p.add_argument("--no_route_features", action="store_true")
    p.add_argument("--no_aux_heads", action="store_true")
    p.add_argument("--rule_features", action="store_true",
                   help="exact box gaps to the forecast modes + Waymax-rule edge distance as scorer inputs")
    p.add_argument("--input_heading", action="store_true",
                   help="score bank candidates with their recorded headings (not the travel direction from xy)")
    p.add_argument("--scorer_agents", type=int, default=32)
    p.add_argument("--scorer_modes", type=int, default=6)
    p.add_argument("--unfreeze_decoder", action="store_true")
    # loss
    p.add_argument("--w_safety", type=float, default=1.0)
    p.add_argument("--w_progress", type=float, default=1.0)
    p.add_argument("--w_goal_pairs", type=float, default=0.5)
    p.add_argument("--w_unsafe_time", type=float, default=0.5)
    p.add_argument("--w_unsafe_severity", type=float, default=0.25)
    p.add_argument("--w_sub", type=float, default=1.0)
    p.add_argument("--w_list", type=float, default=0.5)
    p.add_argument("--w_aux", type=float, default=0.5)
    p.add_argument("--w_distill", type=float, default=0.0)
    p.add_argument("--reward_weights", type=float, nargs=5, default=[5.0, 5.0, 2.0, 1.0, 0.5],
                   help="listwise reward: overlap, offroad, goal gain, goal distance, comfort")
    # self-generated candidates (WOMD cache shards)
    p.add_argument("--self_cache_dir", default="", help="WOMD cache shards for self-generated batches")
    p.add_argument("--self_frac", type=float, default=0.25)
    # bookkeeping
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--val_groups", type=int, default=800)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--wandb_project", default="anchor_wm")
    p.add_argument("--wandb_entity", default="")
    p.add_argument("--wandb_mode", default="online")
    return p.parse_args()


def build_model(args, device: torch.device) -> tuple[AnchorWorldModel, dict]:
    ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
    cfg = AnchorWMConfig.from_dict(dict(ck["config"]))
    cfg.scorer_goal_features = not args.no_goal_features
    cfg.scorer_route_features = not args.no_route_features
    cfg.scorer_aux_heads = not args.no_aux_heads
    cfg.scorer_rule_features = bool(args.rule_features)
    cfg.scorer_input_heading = bool(getattr(args, "input_heading", False))
    cfg.scorer_agents, cfg.scorer_modes = int(args.scorer_agents), int(args.scorer_modes)
    model = AnchorWorldModel(cfg, ck["model"]["anchors"] * float(cfg.max_range))
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    bad = [k for k in missing if not k.startswith(NEW_SCORER_PREFIXES)]
    if unexpected or bad:
        raise ValueError(f"init checkpoint mismatch: unexpected {unexpected[:5]}, missing {bad[:5]}")
    return model.to(device), {"init_ckpt": args.init_ckpt, "init_step": ck.get("step"), "new_tensors": len(missing)}


def set_trainable(model: AnchorWorldModel, unfreeze_decoder: bool) -> None:
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("scorer.") or (unfreeze_decoder and name.startswith(DECODER_PREFIXES))


def train_mode(model: AnchorWorldModel) -> None:
    """Scorer in train mode (dropout); the world model stays in eval mode so its outputs are deterministic."""
    model.eval()
    model.scorer.train()


def score_bank(model, f: dict, scale: float, train_wm: bool, chunk: int = 0):
    if train_wm:
        enc, dec = model(f)
    else:
        with torch.no_grad():
            enc, dec = model(f)
    dec = {k: v.float() for k, v in dec.items()}
    ctx = model.scorer.context(enc, dec, f, scale)
    total, sub = model.scorer(ctx, f["cand_xy"], f["cand_yaw"], chunk=chunk, heading=f["cand_yaw"])
    return total, sub, dict(model.scorer.last_aux), enc, dec


@torch.no_grad()
def validate(model, val_index, args, device, amp) -> dict:
    model.eval()
    loader = make_bank_loader(val_index, args.cache_dir, batch_groups=4, cands_per_group=0, shuffle=False,
                              seed=0, num_workers=2)
    # Each metric is averaged over the batches where it is defined (safety AUROC needs a group with both
    # safe and unsafe candidates, which most batches of 4 lack), not over all batches.
    sums, counts, n = {}, {}, 0
    for batch in loader:
        f = move(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            total, _, _, _, _ = score_bank(model, f, 0.0, False, chunk=64)
        m = bank_eval_metrics(total.float(), bank_targets(f))
        b = total.shape[0]
        for k, v in m.items():
            if math.isfinite(v):
                sums[k] = sums.get(k, 0.0) + v * b
                counts[k] = counts.get(k, 0) + b
        n += b
        if n >= args.val_groups:
            break
    train_mode(model)
    return {k: v / counts[k] for k, v in sums.items()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    amp = device.type == "cuda" and not args.no_amp and torch.cuda.is_bf16_supported()
    model, init_info = build_model(args, device)
    cfg = model.config
    set_trainable(model, args.unfreeze_decoder)
    scorer_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("scorer.")]
    dec_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("scorer.")]
    groups = [{"params": scorer_params, "lr": args.lr}]
    if dec_params:
        groups.append({"params": dec_params, "lr": args.decoder_lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in groups]
    scale = 1.0 if args.unfreeze_decoder else 0.0

    train_index, train_counts = build_index(args.bank_dir, args.cache_dir, args.split_json, args.invalid_json, ("train",))
    val_index, val_counts = build_index(args.bank_dir, args.cache_dir, args.split_json, args.invalid_json, ("val",))
    if not train_index or not val_index:
        raise RuntimeError(f"empty bank index: train {train_counts}, val {val_counts}")
    bank_loader = lambda epoch: make_bank_loader(  # noqa: E731
        train_index, args.cache_dir, batch_groups=args.batch_groups, cands_per_group=args.cands_per_group,
        max_unsafe_frac=args.max_unsafe_frac, shuffle=True, seed=args.seed + epoch, num_workers=args.num_workers)

    def cycle(fn):
        epoch = 0
        while True:
            for b in fn(epoch):
                yield b
            epoch += 1

    bank_batches = cycle(bank_loader)
    self_batches = None
    targs = None
    if args.self_cache_dir and args.self_frac > 0:
        paths, _ = split_cache_paths(resolve_cache_paths(args.self_cache_dir), 0.1)
        self_batches = cycle(lambda epoch: make_loader(paths, cfg, batch_size=args.batch_groups, anchor_steps=(20, 30, 40),
                                                       shuffle=True, seed=args.seed + 100 + epoch,
                                                       num_workers=max(1, args.num_workers // 2)))
        targs = SimpleNamespace(train_cands=48, score_chunk=0, progress_key="goal_gain", progress_margin=2.0,
                                w_sub=1.0, w_pair=1.0, w_list=0.5)

    run_dir = Path(args.save_dir) / f"{args.run_name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"config": cfg.to_dict(), "args": vars(args), "init": init_info, "train_index": train_counts,
            "val_index": val_counts, "trainable_params": int(sum(p.numel() for p in scorer_params + dec_params))}
    (run_dir / "config.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps(meta, default=str), flush=True)
    wb = None
    if args.wandb_mode != "disabled":
        import wandb
        wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, name=run_dir.name,
                        mode=args.wandb_mode, dir=str(run_dir), config=meta)

    def save(tag: str, step: int, val: dict | None = None) -> None:
        torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "anchors": "in state_dict",
                    "args": vars(args), "step": step, "val": val, "init": init_info, "best": best,
                    "opt": opt.state_dict() if tag in ("latest", "final") else None}, run_dir / f"{tag}.pt")

    w_pairs = {"safety": args.w_safety, "progress": args.w_progress, "goal": args.w_goal_pairs,
               "unsafe_time": args.w_unsafe_time, "unsafe_severity": args.w_unsafe_severity}
    gen = torch.Generator().manual_seed(args.seed)
    rng = random.Random(args.seed)
    best, step, t0 = None, 0, time.time()
    if args.resume:
        rk = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(rk["model"])
        opt.load_state_dict(rk["opt"])
        step, best = int(rk["step"]), rk.get("best")
        print(json.dumps({"resumed": args.resume, "step": step, "best": best}), flush=True)
    running, n_run = {}, 0
    val0 = validate(model, val_index, args, device, amp)
    print(json.dumps({"step": step, "val": {k: round(v, 4) for k, v in val0.items()}}), flush=True)
    if wb is not None:
        wb.log({f"val/{k}": v for k, v in val0.items()}, step=step)
    train_mode(model)
    while step < args.max_steps:
        lr_mult = min(1.0, (step + 1) / args.warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.max_steps)))
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * lr_mult
        use_self = self_batches is not None and rng.random() < args.self_frac
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            if use_self:
                f = move(next(self_batches), device)
                with torch.no_grad():
                    enc, dec = model(f)
                dec = {k: v.float() for k, v in dec.items()}
                loss, m = scorer_step(model, enc, dec, f, targs, cfg, 0.0, gen)
                m = {f"self/{k}": v for k, v in m.items()}
            else:
                f = move(next(bank_batches), device)
                total, sub, aux, _, _ = score_bank(model, f, scale, args.unfreeze_decoder)
                tg = bank_targets(f)
                loss, m = bank_score_loss(total, sub, aux, tg, f, w_pairs=w_pairs, w_sub=args.w_sub, w_list=args.w_list,
                                          w_aux=args.w_aux, w_distill=args.w_distill,
                                          reward_weights=tuple(args.reward_weights))
                m = {f"bank/{k}": v for k, v in m.items()}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], args.grad_clip)
        opt.step()
        step += 1
        m["grad_norm"] = float(gnorm)
        for k, v in m.items():
            if math.isfinite(v):
                running[k] = running.get(k, 0.0) + v
                running[k + "#n"] = running.get(k + "#n", 0) + 1
        n_run += 1
        if step % args.log_every == 0:
            avg = {k: running[k] / running[k + "#n"] for k in running if not k.endswith("#n")}
            avg["hours"] = (time.time() - t0) / 3600
            print(json.dumps({"step": step, **{k: round(v, 4) for k, v in avg.items()}}), flush=True)
            if wb is not None:
                wb.log({f"train/{k}": v for k, v in avg.items()}, step=step)
            running, n_run = {}, 0
        stop = bool(args.max_hours) and (time.time() - t0) / 3600 >= args.max_hours
        if step % args.val_every == 0 or step == args.max_steps or stop:
            vm = validate(model, val_index, args, device, amp)
            print(json.dumps({"step": step, "val": {k: round(v, 4) for k, v in vm.items()}}), flush=True)
            if wb is not None:
                wb.log({f"val/{k}": v for k, v in vm.items()}, step=step)
            sel = vm.get("pairwise_accuracy", float("nan")) - vm.get("picked_unsafe", 0.0)
            if math.isfinite(sel) and (best is None or sel > best):
                best = sel
                save("best", step, vm)
        if step % args.save_every == 0:
            save("latest", step)
        if stop:
            break
    save("latest", step)
    save("final", step)
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
