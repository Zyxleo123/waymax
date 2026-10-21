"""Replay a training step that went non-finite, from a nonfinite_step<N>.pt dump written by anchor_wm.train
(pre-update weights, the batch and the RNG state of that step), and report where the NaN / inf comes from:

  1. the loss terms (forecasting / scorer) that are non-finite,
  2. the first module, in execution order, whose forward output is non-finite,
  3. the parameters whose gradient is non-finite,
  4. with a finite forward pass: the backward op that first produces a NaN (autograd anomaly mode).

  python -m anchor_wm.replay_nonfinite <run_dir>/nonfinite_step42787.pt [--device cuda] [--no_amp]
"""

from __future__ import annotations

import argparse
import math
import traceback
from argparse import Namespace

import torch

from anchor_wm.config import AnchorWMConfig
from anchor_wm.losses import wm_loss
from anchor_wm.model import AnchorWorldModel
from anchor_wm.train import scorer_step


def _finite(x) -> bool:
    if isinstance(x, torch.Tensor):
        return bool(torch.isfinite(x).all()) if x.is_floating_point() else True
    if isinstance(x, dict):
        return all(_finite(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(_finite(v) for v in x)
    return True


def _summary(x) -> str:
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        bad = ~torch.isfinite(x)
        fin = x[~bad].float()
        rng = f"finite range [{fin.min():.3g}, {fin.max():.3g}]" if fin.numel() else "no finite values"
        return f"{tuple(x.shape)} {x.dtype}: {int(bad.sum())} non-finite ({int(torch.isnan(x).sum())} nan), {rng}"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}: {_summary(v)}" for k, v in x.items() if not _finite(v)) + "}"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_summary(v) for v in x if not _finite(v)) + "]"
    return type(x).__name__


def run_step(model, f, args, cfg, scale, amp, rng, device, hooks: bool):
    torch.set_rng_state(rng["cpu"])
    if rng["cuda"] is not None and device.type == "cuda":
        torch.cuda.set_rng_state(rng["cuda"])
    gen = torch.Generator()
    gen.set_state(rng["gen"])
    first_bad: list = []
    handles = []
    if hooks:
        def hook(mod, inp, out, name=None):
            if not first_bad and not _finite(out):
                first_bad.append((name, type(mod).__name__, _finite(inp), _summary(out), _summary(inp)))
        for name, mod in model.named_modules():
            handles.append(mod.register_forward_hook(lambda m, i, o, name=name: hook(m, i, o, name)))
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            enc, dec = model(f)
            dec = {k: v.float() for k, v in dec.items()}
            l_wm, m_wm = wm_loss(dec, f, cfg)
            l_s, m_s = (scorer_step(model, enc, dec, f, args, cfg, scale, gen) if not args.no_scorer
                        else (torch.zeros((), device=device), {}))
            loss = l_wm + args.score_weight * l_s
    finally:
        for h in handles:
            h.remove()
    return loss, l_wm, l_s, {**m_wm, **m_s}, first_bad, enc, dec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dump")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_amp", action="store_true", help="replay in fp32 (is the NaN a bf16 effect?)")
    a = p.parse_args()

    ck = torch.load(a.dump, map_location="cpu", weights_only=False)
    args = Namespace(**ck["args"])
    cfg = AnchorWMConfig.from_dict(ck["config"])
    device = torch.device(a.device)
    model = AnchorWorldModel(cfg, ck["model"]["anchors"] * float(cfg.max_range)).to(device)
    model.load_state_dict(ck["model"])
    model.train()
    amp = device.type == "cuda" and not a.no_amp and not args.no_amp and torch.cuda.is_bf16_supported()
    f = {k: v.to(device) for k, v in ck["batch"].items()}
    print(f"step {ck['step']}: logged loss {ck['loss']}, grad_norm {ck['grad_norm']}, scale {ck['scale']}, amp {amp}")
    print("logged non-finite metrics:", sorted(k for k, v in ck["metrics"].items() if not math.isfinite(v)))
    bad_in = {k: _summary(v) for k, v in f.items() if not _finite(v)}
    print("non-finite batch inputs:", bad_in or "none")

    loss, l_wm, l_s, metrics, first_bad, enc, dec = run_step(model, f, args, cfg, ck["scale"], amp, ck["rng"],
                                                            device, hooks=True)
    print(f"replayed: loss {float(loss)}, wm {float(l_wm)}, scorer {float(l_s)}")
    print("replayed non-finite metrics:", sorted(k for k, v in metrics.items() if not math.isfinite(v)))
    print("non-finite encoder outputs:", {k: _summary(v) for k, v in enc.items() if not _finite(v)} or "none")
    print("non-finite decoder outputs:", {k: _summary(v) for k, v in dec.items() if not _finite(v)} or "none")
    if first_bad:
        name, kind, inp_ok, out, inp = first_bad[0]
        print(f"first non-finite module output: {name} ({kind}); inputs finite: {inp_ok}\n  out: {out}\n  in: {inp}")
    else:
        print("forward: every module output finite")

    model.zero_grad(set_to_none=True)
    loss.backward()
    bad_grads = [(n, _summary(q.grad)) for n, q in model.named_parameters() if q.grad is not None and not _finite(q.grad)]
    print(f"parameters with non-finite grads: {len(bad_grads)}")
    for n, s in bad_grads[:15]:
        print("  ", n, s)

    if math.isfinite(float(loss)) and bad_grads:
        print("forward finite, backward not: replaying under autograd anomaly detection")
        model.zero_grad(set_to_none=True)
        with torch.autograd.detect_anomaly(check_nan=True):
            loss2, *_ = run_step(model, f, args, cfg, ck["scale"], amp, ck["rng"], device, hooks=False)
            try:
                loss2.backward()
                print("anomaly replay: no error raised (non-deterministic?)")
            except RuntimeError:
                traceback.print_exc()


if __name__ == "__main__":
    main()
