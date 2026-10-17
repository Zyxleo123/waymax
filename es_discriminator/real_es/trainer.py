"""Training loop of the safety_progress discriminator (`WMTrajectoryDiscriminator`, full Gaussian WM).

The method (`safety_progress/train.py`) supplies `loss_fn(scores [B,K] tensor, cands {key:
[B,K] numpy}, groups {key: [B] numpy}, rng) -> (loss, stats)`, the candidate label keys it
needs, the validation metric used for checkpoint selection and the candidate sampler for
`--train-cands`. This module handles data (with the cached world-model / road-edge inputs),
split, filters, model, W&B, checkpoints, predictions and offline evaluation. Checkpoints are
selected on validation metrics only; test predictions and reports are produced once, after
training, from the selected checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch

from es_discriminator.model import make_candidate_features_torch

from . import evaluate as ev
from .dataset import (
    DEFAULT_EDGE_DIR, DEFAULT_INVALID, DEFAULT_RUN, DEFAULT_SPLIT, DEFAULT_WM_DIR, Bank, all_scenario_ids,
    cand_batch, context_batch, edge_batch, group_split, invalid_scene_keys, iterate_batches, load_bank,
    load_or_create_split, training_mask, wm_batch,
)
from .check_alignment import recorder_config
from .wm_model import VARIANT, WMTrajectoryDiscriminator

LossFn = Callable[[torch.Tensor, dict, dict, np.random.Generator], tuple[torch.Tensor, dict]]
Sampler = Callable[[dict, np.random.Generator, int], np.ndarray]


@dataclass
class Method:
    name: str
    loss_fn: LossFn
    cand_keys: tuple[str, ...]
    group_keys: tuple[str, ...]
    select_key: str          # key of evaluate.quick_metrics
    select_mode: str         # "min" or "max"
    sampler: Sampler


def common_parser(description: str, *, default_out: str, default_name: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DEFAULT_RUN)
    p.add_argument("--split", default=DEFAULT_SPLIT, help="scenario split JSON (created on first use)")
    p.add_argument("--invalid-at-start", default=DEFAULT_INVALID)
    p.add_argument("--output-dir", default=default_out)
    p.add_argument("--no-expert-filter", action="store_true",
                   help="keep replans whose logged expert future fails Waymax overlap/offroad")
    p.add_argument("--max-files", type=int, default=0, help="load only N batch files (smoke tests)")
    p.add_argument("--max-train-groups", type=int, default=0)
    p.add_argument("--max-eval-groups", type=int, default=0)
    p.add_argument("--max-steps-per-epoch", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-groups", type=int, default=16, help="replan groups (x320 candidates) per step")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pos-scale", type=float, default=20.0,
                   help="candidate x/y feature divisor in metres; candidates of one replan differ by "
                        "tenths of a metre, which 100 (the jitter discriminator) compresses too much")
    p.add_argument("--vel-scale", type=float, default=5.0, help="candidate velocity feature divisor (m/s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true",
                   help="bf16 autocast on CUDA; off by default because score differences between candidates "
                        "of one replan are below bf16 resolution and listwise training collapsed under it")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--progress-key", default="goal_gain", choices=["goal_gain", "progress"],
                   help="progress label used by the safety+progress pairs and their ranking metric")
    p.add_argument("--progress-margin", type=float, default=2.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--wandb-project", default="discriminator")
    p.add_argument("--wandb-entity", default="scbellad2025-carnegie-mellon-university")
    p.add_argument("--wandb-name", default=default_name)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wm-dir", default=DEFAULT_WM_DIR)
    p.add_argument("--edge-dir", default=DEFAULT_EDGE_DIR)
    p.add_argument("--train-cands", type=int, default=0,
                   help="candidates per group per training step (0 = all); evaluation always scores all")
    p.add_argument("--eval-chunk", type=int, default=64, help="candidates scored per chunk at evaluation")
    p.add_argument("--select-key", default="", help="override the method's validation selection metric")
    p.add_argument("--select-mode", default="", choices=["", "min", "max"])
    return p


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_torch(ctx: dict, device: torch.device) -> dict:
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device, non_blocking=True) for k, v in ctx.items()}


def _traj(bank: Bank, gidx: np.ndarray, cand_idx: np.ndarray | None) -> np.ndarray:
    traj = cand_batch(bank, gidx, ("traj",))["traj"]
    return traj if cand_idx is None else np.take_along_axis(traj, cand_idx[..., None, None], axis=1)


def score_batch(model, bank: Bank, gidx: np.ndarray, device: torch.device, feat: dict,
                cand_idx: np.ndarray | None = None, chunk: int = 0) -> torch.Tensor:
    """Scores [B,K] of groups `gidx` (all candidates, or `cand_idx` [B,K'] per group)."""
    traj = torch.from_numpy(_traj(bank, gidx, cand_idx)).to(device)
    feats = make_candidate_features_torch(traj, **feat)
    ctx = _to_torch(context_batch(bank, gidx), device)
    return model.score_groups(ctx, feats, traj, wm=_to_torch(wm_batch(bank, gidx), device),
                              edges=_to_torch(edge_batch(bank, gidx), device),
                              ego_size=torch.from_numpy(bank.groups["ego_size"][gidx]).to(device), chunk=chunk)


@torch.no_grad()
def predict(model, bank: Bank, gidx: np.ndarray, device: torch.device, *, batch_groups: int, feat: dict,
            amp: bool, chunk: int = 0) -> np.ndarray:
    model.eval()
    out = []
    for b in iterate_batches(gidx, batch_groups):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            s = score_batch(model, bank, b, device, feat, chunk=chunk)
        out.append(s.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, bank.k), np.float32)


def build_model(model_kwargs: dict) -> torch.nn.Module:
    kw = dict(model_kwargs)
    variant = kw.pop("variant", VARIANT)
    if variant != VARIANT:
        raise ValueError(f"checkpoint variant {variant!r} is not supported; only {VARIANT!r}")
    return WMTrajectoryDiscriminator(kw["hidden"], kw["heads"], kw["dropout"],
                                     pos_scale=kw["pos_scale"], vel_scale=kw["vel_scale"])


def _subsample(idx: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(idx, n, replace=False)) if n and len(idx) > n else idx


def run(args: argparse.Namespace, method: Method) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    device = _device(args.device)
    amp = device.type == "cuda" and args.amp and torch.cuda.is_bf16_supported()
    cfg = recorder_config(args.data_dir)
    feat = {"dt": 0.1 * cfg.future_stride,  # recorded world steps are 0.1 s apart
            "pos_scale": args.pos_scale, "vel_scale": args.vel_scale}

    select_key = args.select_key or method.select_key
    select_mode = args.select_mode or method.select_mode
    chunk = args.eval_chunk

    t0 = time.time()
    bank = load_bank(args.data_dir, max_files=args.max_files, cfg=cfg, wm_dir=args.wm_dir, edge_dir=args.edge_dir)
    split = load_or_create_split(args.split, all_scenario_ids(args.data_dir), seed=args.split_seed)
    invalid = invalid_scene_keys(args.invalid_at_start)
    gsplit = group_split(bank, split)
    clean, filters = training_mask(bank, invalid, filter_expert=not args.no_expert_filter)
    gs = bank.groups
    usable = gs["group_valid"] & np.isfinite(gs["min_ade"])
    train_idx = _subsample(np.flatnonzero(clean & (gsplit == "train")), args.max_train_groups, rng)
    val_idx = _subsample(np.flatnonzero(clean & (gsplit == "val")), args.max_eval_groups, rng)
    pred_idx = {s: _subsample(np.flatnonzero(usable & (gsplit == s)), args.max_eval_groups, rng)
                for s in ("val", "test")}
    data_info = {
        "batch_files": len(bank.files), "groups": bank.num_groups, "candidates_per_group": bank.k,
        "filters": filters, "train_groups": int(len(train_idx)), "val_groups": int(len(val_idx)),
        "prediction_groups": {s: int(len(v)) for s, v in pred_idx.items()},
        "scenes_per_split": {s: int(len(np.unique(gs["scen"][gsplit == s]))) for s in ("train", "val", "test")},
        "load_seconds": round(time.time() - t0, 1), "device": str(device), "bf16_autocast": amp,
        "wm_variant": VARIANT, "train_cands": args.train_cands or bank.k,
        "selection": {"key": select_key, "mode": select_mode, "split": "val"},
    }
    print(json.dumps(data_info, indent=1), flush=True)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("empty train or validation set; check --max-files / split")

    model_kwargs = {"hidden": args.hidden, "heads": args.heads, "dropout": args.dropout, "variant": VARIANT,
                    "pos_scale": args.pos_scale, "vel_scale": args.vel_scale}
    model = build_model(model_kwargs).to(device)
    data_info["parameters"] = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_idx) / args.batch_groups)
    if args.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    total = max(1, steps_per_epoch * args.epochs)
    warm = min(args.warmup_steps, total // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    config = {**vars(args), "method": method.name, "data": data_info, "data_config": asdict(cfg)}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=1, default=str)
    run_wb = None
    if args.wandb_mode != "disabled" and args.wandb_project:
        import wandb
        run_wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, name=args.wandb_name,
                            mode=args.wandb_mode, dir=out, config=config)
        with open(os.path.join(out, "wandb_run.txt"), "w") as f:
            f.write(run_wb.path + "\n")

    eval_kw = {"progress_key": args.progress_key, "margin": args.progress_margin}
    best_val, best_epoch, history, step = None, 0, [], 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums: dict[str, float] = {}
        n = 0
        for i, b in enumerate(iterate_batches(train_idx, args.batch_groups, rng)):
            if i >= steps_per_epoch:
                break
            cands = cand_batch(bank, b, method.cand_keys)
            groups = {k: gs[k][b] for k in method.group_keys}
            cand_idx = None
            if args.train_cands and args.train_cands < bank.k:
                cand_idx = method.sampler(cands, rng, args.train_cands)
                cands = {k: np.take_along_axis(v, cand_idx, axis=1) for k, v in cands.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                scores = score_batch(model, bank, b, device, feat, cand_idx)
            loss, stats = method.loss_fn(scores.float(), cands, groups, rng)
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            opt.step()
            sched.step()
            step += 1
            stats = {"loss": float(loss.detach()), "grad_norm": gnorm, **stats}
            for k, v in stats.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
            if run_wb is not None and step % args.log_every == 0:
                run_wb.log({**{f"train_step/{k}": v for k, v in stats.items()},
                            "lr": sched.get_last_lr()[0], "epoch": epoch}, step=step)
        train_stats = {k: v / max(n, 1) for k, v in sums.items()}
        val_scores = predict(model, bank, val_idx, device, batch_groups=args.batch_groups * 2, feat=feat, amp=amp,
                             chunk=chunk)
        val_stats = ev.quick_metrics(bank, val_idx, val_scores, **eval_kw)
        rec = {"epoch": epoch, "step": step, "train": train_stats, "val": val_stats,
               "elapsed_min": round((time.time() - t0) / 60, 1)}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if run_wb is not None:
            run_wb.log({"epoch": epoch, **{f"train/{k}": v for k, v in train_stats.items()},
                        **{f"val/{k}": v for k, v in val_stats.items()}}, step=step)
        sel = val_stats[select_key]
        better = best_val is None or (sel < best_val if select_mode == "min" else sel > best_val)
        ckpt = {"model": model.state_dict(), "model_kwargs": model_kwargs,
                "method": method.name, "data_config": asdict(cfg), "candidate_features": feat,
                "epoch": epoch, "val": val_stats, "args": vars(args)}
        torch.save(ckpt, os.path.join(out, "last.pt"))
        if better and np.isfinite(sel):
            best_val, best_epoch = sel, epoch
            torch.save(ckpt, os.path.join(out, "best.pt"))
            if run_wb is not None:
                run_wb.summary.update({"best/epoch": epoch, f"best/val_{select_key}": sel})
        with open(os.path.join(out, "history.json"), "w") as f:
            json.dump(history, f, indent=1)

    best = torch.load(os.path.join(out, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    result = {"best_epoch": best_epoch, f"best_val_{select_key}": best_val, "selection_split": "val"}
    for s, idx in pred_idx.items():
        scores = predict(model, bank, idx, device, batch_groups=args.batch_groups * 2, feat=feat, amp=amp,
                         chunk=chunk)
        ev.save_predictions(os.path.join(out, f"{s}_predictions.npz"), bank, idx, scores)
        rep = ev.full_report(bank, idx, scores, invalid=invalid, n_boot=args.n_boot, **eval_kw)
        with open(os.path.join(out, f"eval_{s}.json"), "w") as f:
            json.dump(rep, f, indent=1)
        md = ev.to_markdown(rep, f"{method.name}: offline evaluation on held-out {s} scenarios "
                                 f"(best epoch {best_epoch})")
        with open(os.path.join(out, f"eval_{s}.md"), "w") as f:
            f.write(md)
        print(md, flush=True)
        if run_wb is not None:
            flat = {}
            for sub in ("filtered", "es_unsafe_scenes_all_replans", "es_unsafe_scenes_first_bad_replan"):
                r = rep[sub]
                flat[f"{s}/{sub}/n_groups"] = r.get("n_groups", 0)
                for p in (*ev.PICKERS, "discriminator_minus_es", "ranking"):
                    for m, b in r.get(p, {}).items():
                        if b.get("mean") is not None:
                            flat[f"{s}/{sub}/{p}/{m}"] = b["mean"]
            run_wb.summary.update(flat)
            import wandb
            run_wb.log({f"{s}_eval_report": wandb.Html(f"<pre>{md}</pre>")})
    if run_wb is not None:
        run_wb.finish()
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    return result
