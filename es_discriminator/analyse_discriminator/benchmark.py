"""Trainable parameters and inference latency of A0-A4 for one ES scoring call (64 candidates).

Inputs are a real recorded replan (context, WM mixture, road edges, first 64 candidates), repeated
over `--worlds` replans (the planner scores 5 worlds x 64 candidates per ES iteration). Timed:
`encode_context` (once per replan in deployment), `score_encoded` (every ES iteration) and both.
Latency does not depend on the weights, so models without a checkpoint use random weights.

  PYTHONPATH=. python -m es_discriminator.analyse_discriminator.benchmark --device cuda \
      --run A1=/zfsauton/scratch/sbellad/es_discriminator/safety_progress/<A1 run>/best.pt \
      --out /zfsauton/scratch/sbellad/es_discriminator/analyse_discriminator/benchmark_cuda.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import numpy as np
import torch

from es_discriminator.analyse_discriminator.ablations import ABLATIONS, DESCRIPTIONS, ablation_of, flags
from es_discriminator.analyse_discriminator.verify_a0 import A0_CKPT
from es_discriminator.data import DataConfig
from es_discriminator.model import make_candidate_features_torch
from es_discriminator.real_es.dataset import cand_batch, context_batch, edge_batch, load_bank, wm_batch
from es_discriminator.real_es.trainer import build_model


def _inputs(a0: dict, worlds: int, k: int, device: torch.device):
    args = a0["args"]
    bank = load_bank(args["data_dir"], max_files=1, cfg=DataConfig(**a0["data_config"]), wm_dir=args["wm_dir"],
                     edge_dir=args["edge_dir"], verbose=False)
    g = np.flatnonzero(bank.groups["group_valid"])[:1].repeat(worlds)
    to = lambda d: {key: torch.from_numpy(np.ascontiguousarray(v)).to(device) for key, v in d.items()}  # noqa: E731
    traj = torch.from_numpy(cand_batch(bank, g, ("traj",))["traj"][:, :k]).to(device)
    return dict(ctx=to(context_batch(bank, g)), wm=to(wm_batch(bank, g)), edges=to(edge_batch(bank, g)),
                size=torch.from_numpy(bank.groups["ego_size"][g]).to(device), traj=traj,
                feats=make_candidate_features_torch(traj, **a0["candidate_features"]))


def _time(fn, device: torch.device, warmup: int, iters: int) -> dict:
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(1e3 * (time.perf_counter() - t0))
    return {"median_ms": statistics.median(ts), "p90_ms": float(np.percentile(ts, 90)), "iters": iters}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", default=[], metavar="NAME=CKPT",
                   help="trained checkpoint per ablation (others: random weights, same architecture)")
    p.add_argument("--a0", default=A0_CKPT, help="A0 checkpoint: hyper-parameters, data paths, features")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--worlds", default="1,5", help="replans per scoring call")
    p.add_argument("--candidates", type=int, default=64)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    p.add_argument("--out", default="")
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    a0 = torch.load(a.a0, map_location="cpu", weights_only=False)
    ckpts = dict(r.split("=", 1) for r in a.run)
    worlds = [int(w) for w in a.worlds.split(",")]
    inputs = {w: _inputs(a0, w, a.candidates, dev) for w in worlds}

    out = {"device": str(dev), "candidates": a.candidates, "threads": torch.get_num_threads(),
           "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None, "models": {}}
    for name in ABLATIONS:
        path = ckpts.get(name, a.a0 if name == "A0" else "")
        if path:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            if ablation_of(ck["model_kwargs"]) != name:
                raise ValueError(f"{path} is {ablation_of(ck['model_kwargs'])}, not {name}")
            model = build_model(ck["model_kwargs"])
            model.load_state_dict(ck["model"], strict=True)
        else:
            model = build_model({**a0["model_kwargs"], "ablation": name, **flags(name)})
        model = model.to(dev).eval()
        rec = {"description": DESCRIPTIONS[name], "checkpoint": path or None,
               "trainable_parameters": int(sum(q.numel() for q in model.parameters() if q.requires_grad)),
               "latency": {}}
        for w, x in inputs.items():
            with torch.no_grad():
                enc = lambda: model.encode_context(x["ctx"], x["wm"], x["edges"], x["size"])  # noqa: E731
                e = enc()
                score = lambda: model.score_encoded(e, x["feats"], x["traj"])  # noqa: E731
                both = lambda: model.score_encoded(enc(), x["feats"], x["traj"])  # noqa: E731
                rec["latency"][f"worlds_{w}"] = {"encode_context": _time(enc, dev, a.warmup, a.iters),
                                                 "score_encoded": _time(score, dev, a.warmup, a.iters),
                                                 "total": _time(both, dev, a.warmup, a.iters)}
        out["models"][name] = rec
        print(name, json.dumps({"params": rec["trainable_parameters"],
                                **{w: round(v["total"]["median_ms"], 2) for w, v in rec["latency"].items()}}),
              flush=True)
    a0n = out["models"]["A0"]
    for name, rec in out["models"].items():
        rec["parameters_vs_A0"] = rec["trainable_parameters"] / a0n["trainable_parameters"]
        for w in rec["latency"]:
            rec["latency"][w]["total_vs_A0"] = (rec["latency"][w]["total"]["median_ms"]
                                                / a0n["latency"][w]["total"]["median_ms"])
    text = json.dumps(out, indent=1)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
