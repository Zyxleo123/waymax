"""How large is the scoring gradient on the shared WM weights, relative to the forecasting gradient?

Loads a training checkpoint and, on a few training batches, computes the gradients of
L_wm and score_weight * L_score (scorer inputs at joint scale 1, as in the joint arm) separately
on the encoder and on the anchor decoder. Reports per-batch norms, the scorer's share
|g_score| / (|g_wm| + |g_score|) and the cosine between the two. Read-only (nothing is saved).

  python -m anchor_wm.grad_split --ckpt .../latest.pt --cache_dir ... --batches 20
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
import torch

from anchor_wm.data import make_loader, resolve_cache_paths, split_cache_paths
from anchor_wm.eval_bank import load_model
from anchor_wm.losses import wm_loss
from anchor_wm.model import ENCODER_PREFIXES
from anchor_wm.train import move, scorer_step

DECODER_PREFIXES = ("anchor_embed.", "goal_embed.", "query_fuse.", "traj_head.", "traj_embed.", "mode_cls.")


def _norm(gs) -> float:
    return float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)))


def _cos(a, b) -> float:
    dot = sum((x.float() * y.float()).sum() for x, y in zip(a, b) if x is not None and y is not None)
    return float(dot / max(_norm(a) * _norm(b), 1e-12))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--joint_scale", type=float, default=1.0)
    p.add_argument("--out", default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    targs = SimpleNamespace(**ck["args"])
    model = load_model(args.ckpt, device).train()
    cfg = model.config
    groups = {"encoder": [p_ for n, p_ in model.named_parameters() if n.startswith(ENCODER_PREFIXES)],
              "decoder": [p_ for n, p_ in model.named_parameters() if n.startswith(DECODER_PREFIXES)]}
    train_paths, _ = split_cache_paths(resolve_cache_paths(args.cache_dir), targs.val_fraction)
    loader = make_loader(train_paths, cfg, batch_size=targs.batch_size, anchor_steps=tuple(targs.anchor_steps),
                         shuffle=True, seed=123, num_workers=4)
    gen = torch.Generator().manual_seed(0)
    rows = []
    for i, batch in enumerate(loader):
        if i >= args.batches:
            break
        f = move(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            enc, dec = model(f)
            dec = {k: v.float() for k, v in dec.items()}
            l_wm, _ = wm_loss(dec, f, cfg)
            l_s, _ = scorer_step(model, enc, dec, f, targs, cfg, args.joint_scale, gen)
            l_s = targs.score_weight * l_s
        row = {"step_ckpt": int(ck["step"]), "loss_wm": float(l_wm.detach()), "loss_score": float(l_s.detach())}
        params = groups["encoder"] + groups["decoder"]
        g_wm = torch.autograd.grad(l_wm, params, retain_graph=True, allow_unused=True)
        g_s = torch.autograd.grad(l_s, params, allow_unused=True)
        ne = len(groups["encoder"])
        for name, sl in (("encoder", slice(0, ne)), ("decoder", slice(ne, None))):
            a, b = g_wm[sl], g_s[sl]
            nw, ns = _norm(a), _norm(b)
            row.update({f"{name}_wm": nw, f"{name}_score": ns, f"{name}_score_share": ns / max(nw + ns, 1e-12),
                        f"{name}_cos": _cos(a, b)})
        rows.append(row)
        print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
    keys = [k for k in rows[0] if k not in ("step_ckpt",)]
    summary = {k: {"median": float(np.median([r[k] for r in rows])), "min": float(np.min([r[k] for r in rows])),
                   "max": float(np.max([r[k] for r in rows]))} for k in keys}
    summary["batches"] = len(rows)
    summary["ckpt_step"] = int(ck["step"])
    print("SUMMARY " + json.dumps(summary, indent=1))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"rows": rows, "summary": summary}, fh, indent=1)


if __name__ == "__main__":
    main()
