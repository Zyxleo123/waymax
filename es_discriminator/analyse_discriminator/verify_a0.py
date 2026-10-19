"""Verify the trained A0 discriminator: configuration, WM provenance, and that the current model code
(with every ablation flag off) reproduces A0's saved test scores and the pre-ablation model code.

  PYTHONPATH=. python -m es_discriminator.analyse_discriminator.verify_a0 --max-files 12 \
      --out /zfsauton/scratch/sbellad/es_discriminator/analyse_discriminator/A0_verify.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import types

import numpy as np
import torch

from es_discriminator.analyse_discriminator.ablations import ablation_of
from es_discriminator.data import DataConfig
from es_discriminator.real_es.dataset import load_bank
from es_discriminator.real_es.trainer import build_model, score_batch

A0_CKPT = ("/zfsauton/scratch/sbellad/es_discriminator/safety_progress/"
           "safety_progress_v3_20261016_155456_86103/best.pt")
WM_CKPT = ("/zfsauton/scratch/sbellad/world_model/ckpts_multifuture_gaussian/"
           "multiFuture_GaussianWorldModel_20260916_200620/epoch_0450.pt")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def reference_model_class(rev: str):
    """`WMTrajectoryDiscriminator` as of git `rev` (before the ablation flags)."""
    def src(path: str) -> str:
        return subprocess.run(["git", "-C", REPO, "show", f"{rev}:{path}"], capture_output=True, text=True,
                              check=True).stdout
    base = types.ModuleType("ref_model")
    exec(compile(src("es_discriminator/model.py"), "ref_model", "exec"), base.__dict__)
    wm = types.ModuleType("ref_wm_model")
    wm.__dict__.update(TrajectoryDiscriminator=base.TrajectoryDiscriminator, _MLP=base._MLP)
    code = src("es_discriminator/real_es/wm_model.py").replace(
        "from es_discriminator.model import TrajectoryDiscriminator, _MLP", "")
    exec(compile(code, "ref_wm_model", "exec"), wm.__dict__)
    return wm.WMTrajectoryDiscriminator


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=A0_CKPT)
    p.add_argument("--wm-checkpoint", default=WM_CKPT)
    p.add_argument("--max-files", type=int, default=12, help="batch files to rescore (first N of the bank)")
    p.add_argument("--reference-rev", default="HEAD", help="git revision with the pre-ablation model code")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-groups", type=int, default=1, help="replans scored per forward (memory)")
    p.add_argument("--rel-tol", type=float, default=0.01,
                   help="max |rescored - saved| as a fraction of the median within-replan score std")
    p.add_argument("--out", default="")
    a = p.parse_args()
    dev = torch.device(a.device)
    run_dir = os.path.dirname(a.checkpoint)

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg_json = json.load(open(os.path.join(run_dir, "config.json")))
    args = ck["args"]
    model = build_model(ck["model_kwargs"]).to(dev).eval()
    model.load_state_dict(ck["model"], strict=True)
    ref = reference_model_class(a.reference_rev)(
        ck["model_kwargs"]["hidden"], ck["model_kwargs"]["heads"], ck["model_kwargs"]["dropout"],
        pos_scale=ck["model_kwargs"]["pos_scale"], vel_scale=ck["model_kwargs"]["vel_scale"]).to(dev).eval()
    ref.load_state_dict(ck["model"], strict=True)

    manifest = json.load(open(os.path.join(args["wm_dir"], "manifest.json")))
    report = {
        "checkpoint": a.checkpoint,
        "method": ck.get("method"), "epoch": ck.get("epoch"), "ablation": ablation_of(ck["model_kwargs"]),
        "model_kwargs": ck["model_kwargs"], "architecture": type(model).__name__, "wm_variant": model.variant,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "parameters_in_config": cfg_json["data"]["parameters"],
        "candidate_features": ck["candidate_features"],
        "training": {k: args[k] for k in ("epochs", "batch_groups", "lr", "weight_decay", "warmup_steps",
                                          "train_cands", "eval_chunk", "max_pairs", "safety_frac",
                                          "progress_key", "progress_margin", "seed", "split_seed", "amp")},
        "data": {k: args[k] for k in ("data_dir", "split", "invalid_at_start", "wm_dir", "edge_dir")},
        "groups": {k: cfg_json["data"][k] for k in ("train_groups", "val_groups", "prediction_groups",
                                                     "scenes_per_split")},
        "val_at_best_epoch": ck.get("val"),
        "wm_cache_checkpoint": manifest["checkpoint"],
        "wm_cache_matches_given_wm": os.path.realpath(manifest["checkpoint"]) == os.path.realpath(a.wm_checkpoint),
    }

    pred = np.load(os.path.join(run_dir, "test_predictions.npz"))
    saved = {(str(s), int(t)): sc for s, t, sc in zip(pred["scen_id"], pred["group_t"], pred["scores"])}
    bank = load_bank(args["data_dir"], max_files=a.max_files, cfg=DataConfig(**ck["data_config"]),
                     wm_dir=args["wm_dir"], edge_dir=args["edge_dir"], verbose=False)
    keys = [(str(s), int(t)) for s, t in zip(bank.groups["scen_id"], bank.groups["group_t"])]
    gidx = np.array([g for g, key in enumerate(keys) if key in saved], dtype=np.int64)
    if len(gidx) == 0:
        raise RuntimeError(f"no saved test groups in the first {a.max_files} batch files; raise --max-files")
    feat = dict(ck["candidate_features"])
    bg = a.batch_groups
    with torch.no_grad():
        new = np.concatenate([score_batch(model, bank, gidx[i:i + bg], dev, feat, chunk=64).cpu().numpy()
                              for i in range(0, len(gidx), bg)])
        old = np.concatenate([score_batch(ref, bank, gidx[i:i + bg], dev, feat, chunk=64).cpu().numpy()
                              for i in range(0, len(gidx), bg)])
    want = np.stack([saved[keys[g]] for g in gidx])
    same_top1 = float((new.argmax(1) == want.argmax(1)).mean())
    within_group_std = float(np.median(want.std(1)))
    report["rescore"] = {
        "groups": int(len(gidx)), "candidates_per_group": int(bank.k), "device": str(dev),
        "max_abs_diff_vs_saved_test_scores": float(np.abs(new - want).max()),
        "median_within_group_score_std": within_group_std,
        "top1_agreement_vs_saved": same_top1,
        "max_abs_diff_vs_pre_ablation_code": float(np.abs(new - old).max()),
    }
    # Saved scores come from the GPU run; CPU fp32 differs at the 1e-3 level, so compare to the score spread.
    report["ok"] = bool(report["rescore"]["max_abs_diff_vs_pre_ablation_code"] == 0.0
                        and report["rescore"]["max_abs_diff_vs_saved_test_scores"] < a.rel_tol * within_group_std
                        and same_top1 == 1.0
                        and report["ablation"] == "A0" and report["wm_cache_matches_given_wm"])
    text = json.dumps(report, indent=1, default=str)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(text + "\n")
    if not report["ok"]:
        raise SystemExit("A0 verification FAILED")


if __name__ == "__main__":
    main()
