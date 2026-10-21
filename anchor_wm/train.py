"""Train AnchorWM (forecasting + ego-plan scoring) on WOMD cache shards.

Arms of the 1-day test (same data, steps and warm start):
  A1 separate  --joint_grad_scale 0     scorer trains on detached WM outputs
  A2 joint     --joint_grad_scale 1     scorer gradients reach the decoder and encoder
                                        (ramped in after --joint_warmup_steps)
  WM only      --no_scorer              (profiling / ablation)

  python -m anchor_wm.train --cache_dir .../sim_state_cache_npz --anchors .../anchors_k64.npz \
      --init_ckpt .../multiFuture_GaussianWorldModel_20261005_092950/epoch_0450.pt \
      --save_dir .../anchor_wm/runs --run_name a2_joint --joint_grad_scale 1 --max_steps 75000

`--profile_steps N` runs N steps, prints seconds/step (data wait vs compute) and peak GPU memory,
and exits without saving.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from anchor_wm.candidates import KIND_GT, build_candidates, candidate_headings, label_candidates
from anchor_wm.config import AnchorWMConfig
from anchor_wm.data import make_loader, resolve_cache_paths, split_cache_paths
from anchor_wm.losses import score_loss, wm_loss
from anchor_wm.model import ENCODER_PREFIXES, AnchorWorldModel, warm_start_encoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--anchors", required=True, help="npz from anchor_wm.anchors")
    p.add_argument("--init_ckpt", default="", help="MF-Gaussian checkpoint for the encoder warm start")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--run_name", default="anchor_wm")
    p.add_argument("--resume", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=75000)
    p.add_argument("--max_hours", type=float, default=0.0, help="stop (and save) after this wall time; 0 = off")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--anchor_steps", type=int, nargs="+", default=[20, 30, 40])
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--no_amp", action="store_true")
    # scorer / joint training
    p.add_argument("--no_scorer", action="store_true")
    p.add_argument("--joint_grad_scale", type=float, default=0.0, help="0 = A1 (separate), >0 = A2 (joint)")
    p.add_argument("--joint_warmup_steps", type=int, default=3000)
    p.add_argument("--joint_ramp_steps", type=int, default=5000)
    p.add_argument("--score_weight", type=float, default=1.0)
    p.add_argument("--train_cands", type=int, default=48, help="candidates scored per scene (vocabulary dropout)")
    p.add_argument("--score_chunk", type=int, default=0)
    p.add_argument("--w_sub", type=float, default=1.0)
    p.add_argument("--w_pair", type=float, default=1.0)
    p.add_argument("--w_list", type=float, default=0.5)
    p.add_argument("--progress_key", default="goal_gain", choices=("goal_gain", "progress"))
    p.add_argument("--progress_margin", type=float, default=2.0)
    # model
    p.add_argument("--num_anchors", type=int, default=64)
    p.add_argument("--cls_soft_temp_m", type=float, default=1.0)
    p.add_argument("--scorer_hidden", type=int, default=128)
    # bookkeeping
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=2500)
    p.add_argument("--val_batches", type=int, default=150)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--profile_steps", type=int, default=0)
    p.add_argument("--wandb_project", default="anchor_wm")
    p.add_argument("--wandb_entity", default="")
    p.add_argument("--wandb_mode", default="online")
    return p.parse_args()


def joint_scale(args, step: int) -> float:
    if args.joint_grad_scale <= 0:
        return 0.0
    ramp = (step - args.joint_warmup_steps) / max(1, args.joint_ramp_steps)
    return float(args.joint_grad_scale) * float(min(1.0, max(0.0, ramp)))


def subsample_candidates(cands: dict, n: int, generator: torch.Generator) -> torch.Tensor:
    """Indices [B,n]: the logged future first, then a random subset of the remaining valid candidates."""
    b, c = cands["valid"].shape
    gt_col = int((cands["kinds"] == KIND_GT).nonzero()[0])
    noise = torch.rand((b, c), generator=generator, device="cpu").to(cands["valid"].device)
    noise = noise.masked_fill(~cands["valid"], 2.0)
    noise[:, gt_col] = -1.0
    return noise.argsort(1)[:, :min(n, c)]


def take(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return x.gather(1, idx.view(*idx.shape, *([1] * (x.dim() - 2))).expand(*idx.shape, *x.shape[2:]))


def scorer_step(model: AnchorWorldModel, enc: dict, dec: dict, f: dict, args, cfg: AnchorWMConfig,
                scale: float, generator: torch.Generator) -> tuple[torch.Tensor, dict]:
    cands = build_candidates(f, dec, cfg.max_range, cfg.future_dt)
    idx = subsample_candidates(cands, args.train_cands, generator)
    xy, yaw_fb, valid = take(cands["xy"], idx), take(cands["yaw_fallback"], idx), take(cands["valid"], idx)
    labels = label_candidates(f, xy, candidate_headings(xy, yaw_fb), cfg.future_dt)
    # Drop scenes whose logged future is incomplete or itself labelled unsafe (label noise),
    # as es_discriminator's expert filter does.
    gt_ok = f["ego_future_valid"].bool().all(-1) & ~(labels["overlap"][:, 0] | labels["offroad"][:, 0])
    valid = valid & gt_ok[:, None]
    ctx = model.scorer.context(enc, dec, f, scale)
    total, sub = model.scorer(ctx, xy, yaw_fb, chunk=args.score_chunk)
    loss, metrics = score_loss(total, sub, labels, valid, progress_key=args.progress_key,
                               progress_margin=args.progress_margin,
                               w_sub=args.w_sub, w_pair=args.w_pair, w_list=args.w_list)
    metrics["scene_kept"] = float(gt_ok.float().mean())
    kinds = take(cands["kinds"][None].expand(idx.shape[0], -1), idx)
    unsafe = (labels["overlap"] | labels["offroad"]).float()
    for name, kd in (("anchor", 0), ("perturb", 2)):
        m = (kinds == kd) & valid
        metrics[f"unsafe_rate_{name}"] = float((unsafe * m).sum() / m.sum().clamp_min(1))
    return loss, metrics


def move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def infinite(loader_fn):
    epoch = 0
    while True:
        for batch in loader_fn(epoch):
            yield batch
        epoch += 1


@torch.no_grad()
def validate(model, val_batches, args, cfg, device, amp, scale) -> dict:
    model.eval()
    sums, count = {}, 0
    g = torch.Generator().manual_seed(1234)
    for batch in val_batches:
        f = move(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            enc, dec = model(f)
            dec = {k: v.float() for k, v in dec.items()}
            _, m = wm_loss(dec, f, cfg)
            if not args.no_scorer:
                _, ms = scorer_step(model, enc, dec, f, args, cfg, scale, g)
                m.update(ms)
        for k, v in m.items():
            if math.isfinite(v):
                sums[k] = sums.get(k, 0.0) + v
        count += 1
    model.train()
    return {k: v / max(count, 1) for k, v in sums.items()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    amp = device.type == "cuda" and not args.no_amp and torch.cuda.is_bf16_supported()

    anchors = torch.from_numpy(np.load(args.anchors)["anchors"])
    cfg = AnchorWMConfig(num_anchors=int(anchors.shape[1]), cls_soft_temp_m=args.cls_soft_temp_m,
                         scorer_hidden=args.scorer_hidden)
    model = AnchorWorldModel(cfg, anchors).to(device)
    warm = warm_start_encoder(model, args.init_ckpt) if args.init_ckpt else None
    enc_params = [p for n, p in model.named_parameters() if n.startswith(ENCODER_PREFIXES)]
    other = [p for n, p in model.named_parameters() if not n.startswith(ENCODER_PREFIXES)]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.encoder_lr}, {"params": other, "lr": args.lr}],
                            weight_decay=args.weight_decay)
    base_lrs = [args.encoder_lr, args.lr]
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step = int(ck["step"])

    train_paths, val_paths = split_cache_paths(resolve_cache_paths(args.cache_dir), args.val_fraction)
    loader = lambda epoch: make_loader(train_paths, cfg, batch_size=args.batch_size,  # noqa: E731
                                       anchor_steps=tuple(args.anchor_steps), shuffle=True,
                                       seed=args.seed + epoch + step, num_workers=args.num_workers)
    batches = infinite(loader)

    run_dir = Path(args.save_dir) / f"{args.run_name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wb = None
    if not args.profile_steps:
        run_dir.mkdir(parents=True, exist_ok=True)
        meta = {"config": cfg.to_dict(), "args": vars(args), "warm_start": warm,
                "params": int(sum(p.numel() for p in model.parameters())),
                "params_scorer": int(sum(p.numel() for p in model.scorer.parameters())),
                "train_shards": len(train_paths), "val_shards": len(val_paths)}
        (run_dir / "config.json").write_text(json.dumps(meta, indent=1, default=str))
        print(json.dumps(meta, default=str), flush=True)
        val_batches = []
        for i, b in enumerate(make_loader(val_paths, cfg, batch_size=args.batch_size,
                                          anchor_steps=tuple(args.anchor_steps), shuffle=False, seed=0,
                                          num_workers=min(args.num_workers, 2))):
            if i >= args.val_batches:
                break
            val_batches.append(b)
        if args.wandb_mode != "disabled":
            import wandb
            wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, name=run_dir.name,
                            mode=args.wandb_mode, dir=str(run_dir), config=meta)

    def save(tag: str) -> None:
        ck = {"model": model.state_dict(), "optimizer": opt.state_dict(), "config": cfg.to_dict(),
              "anchors": args.anchors, "args": vars(args), "step": step}
        torch.save(ck, run_dir / f"{tag}.pt")

    gen = torch.Generator().manual_seed(args.seed)
    t_start = time.time()
    running, n_run = {}, 0
    t_data = t_comp = 0.0
    prof = []
    model.train()
    while step < args.max_steps:
        t0 = time.time()
        f = move(next(batches), device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        lr_mult = min(1.0, (step + 1) / args.warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.max_steps)))
        for grp, base in zip(opt.param_groups, base_lrs):
            grp["lr"] = base * lr_mult
        scale = joint_scale(args, step)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            enc, dec = model(f)
            dec = {k: v.float() for k, v in dec.items()}
            loss, metrics = wm_loss(dec, f, cfg)
            if not args.no_scorer:
                l_s, m_s = scorer_step(model, enc, dec, f, args, cfg, scale, gen)
                loss = loss + args.score_weight * l_s
                metrics.update(m_s)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.time()
        step += 1
        t_data += t1 - t0
        t_comp += t2 - t1
        metrics.update(grad_norm=float(gnorm), joint_scale=scale, lr=opt.param_groups[1]["lr"])
        for k, v in metrics.items():
            if math.isfinite(v):
                running[k] = running.get(k, 0.0) + v
        n_run += 1

        if args.profile_steps:
            if step > 5:
                prof.append((t1 - t0, t2 - t1))
            if step >= args.profile_steps + 5:
                d, c = np.mean([p[0] for p in prof]), np.mean([p[1] for p in prof])
                out = {"scorer": not args.no_scorer, "joint_grad_scale": args.joint_grad_scale,
                       "batch_size": args.batch_size, "train_cands": args.train_cands,
                       "data_s_per_step": round(float(d), 4), "compute_s_per_step": round(float(c), 4),
                       "total_s_per_step": round(float(d + c), 4),
                       "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if device.type == "cuda" else None,
                       "last_metrics": {k: round(v, 4) for k, v in metrics.items()}}
                print("PROFILE " + json.dumps(out), flush=True)
                return
            continue

        if step % args.log_every == 0:
            avg = {k: v / n_run for k, v in running.items()}
            avg.update(s_data=t_data / n_run, s_compute=t_comp / n_run, hours=(time.time() - t_start) / 3600)
            print(json.dumps({"step": step, **{k: round(v, 4) for k, v in avg.items()}}), flush=True)
            if wb is not None:
                wb.log({f"train/{k}": v for k, v in avg.items()}, step=step)
            running, n_run, t_data, t_comp = {}, 0, 0.0, 0.0
        if step % args.val_every == 0 or step == args.max_steps:
            vm = validate(model, val_batches, args, cfg, device, amp, scale)
            print(json.dumps({"step": step, "val": {k: round(v, 4) for k, v in vm.items()}}), flush=True)
            if wb is not None:
                wb.log({f"val/{k}": v for k, v in vm.items()}, step=step)
        if step % args.save_every == 0:
            save("latest")
        if args.max_hours and (time.time() - t_start) / 3600 >= args.max_hours:
            print(f"max_hours reached at step {step}", flush=True)
            vm = validate(model, val_batches, args, cfg, device, amp, scale)
            print(json.dumps({"step": step, "val": {k: round(v, 4) for k, v in vm.items()}}), flush=True)
            if wb is not None:
                wb.log({f"val/{k}": v for k, v in vm.items()}, step=step)
            break
    save("latest")
    save("final")
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
