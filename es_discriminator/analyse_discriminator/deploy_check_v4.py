"""disc_v4 (= ablation A4) deployment checks, side by side with A0 (disc_v3), before closed-loop runs.

1. Configuration: each checkpoint is loaded through the deployment path (`real_es.deploy.
   DiscriminatorScorer`, strict state-dict load); the ablation recorded in its model_kwargs, the
   flags on the built model, and the parameter count must match the expected ablation.
2. Online vs offline consistency (`real_es.deploy` check), extended with what matters for ES:
   whether the online scores (live WM forward from the reconstructed sim state) select the same
   top-1 and the same top-8 elite as the offline scores (cached WM inputs) on recorded groups.

  JAX_PLATFORMS=cpu PYTHONPATH=. python -m es_discriminator.analyse_discriminator.deploy_check_v4 \
      --max-files 2 --out /zfsauton/scratch/sbellad/es_discriminator/analyse_discriminator/disc_v4_deploy_check.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from es_discriminator.analyse_discriminator.ablations import ablation_of, flags
from es_discriminator.analyse_discriminator.verify_a0 import A0_CKPT, WM_CKPT

SCRATCH = "/zfsauton/scratch/sbellad"
A4_CKPT = f"{SCRATCH}/es_discriminator/safety_progress/safety_progress_v3_A4_20261017_212234_87831/best.pt"
DATA = f"{SCRATCH}/es_discriminator/training_sets/fail1500_plus_womd_train_s02_11"
TFRECORDS = "/zfsauton/datasets/womd/tf_example/training"
ELITE = 8


def check_config(name: str, ckpt: str, scorer) -> dict:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = scorer.model
    built = {"goal_fusion": bool(m.goal_fusion), "map_tokens": bool(m.map_tokens),
             "agent_tokens": bool(m.agent_tokens)}
    cfg = json.load(open(os.path.join(os.path.dirname(ckpt), "config.json")))
    n = int(sum(p.numel() for p in m.parameters()))
    out = {"checkpoint": ckpt, "epoch": ck.get("epoch"), "ablation_in_ckpt": ablation_of(ck["model_kwargs"]),
           "flags_on_built_model": built, "expected_flags": flags(name), "parameters": n,
           "parameters_in_training_config": cfg["data"]["parameters"], "variant": scorer.variant,
           "method": scorer.method,
           "data_config_matches_training": json.loads(json.dumps(ck["data_config"])) == cfg["data_config"]}
    out["ok"] = bool(out["ablation_in_ckpt"] == name and built == flags(name)
                     and n == out["parameters_in_training_config"] and out["data_config_matches_training"])
    return out


def online_offline(scorer, bank, files, tfrecord_dir: str,
                   chunk: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same groups and inputs as `real_es.deploy.main`; returns online, offline scores and unsafe labels [G,K].
    Candidates are scored independently, `chunk` at a time (as the trainer's --eval-chunk)."""
    import jax

    from es_discriminator.analyze_es_decisions import load_batch_state
    from es_discriminator.data import CONTEXT_KEYS
    from es_discriminator.label_bank import _to_world
    from es_discriminator.real_es.check_alignment import replaced_state
    from es_discriminator.real_es.dataset import EDGE_KEYS, WM_KEYS

    keys = CONTEXT_KEYS + (*WM_KEYS, *EDGE_KEYS, "ego_size")
    unsafe_all = bank.cands["unsafe"].reshape(-1, bank.k)
    traj_all = bank.cands["traj"].reshape(-1, bank.k, *bank.cands["traj"].shape[1:])
    on_l, off_l, uns_l = [], [], []
    g0 = 0
    for f in files:
        d = dict(np.load(f))
        rs = jax.device_get(replaced_state(load_batch_state(d, tfrecord_dir), d))
        for t in np.unique(d["group_t"]):
            gl = np.flatnonzero((d["group_t"] == t) & d["group_valid"].astype(bool))
            if len(gl) == 0:
                continue
            pops = np.zeros((len(d["scen_id"]), bank.k, d["cand_traj"].shape[1] + 1, 5))
            for g in gl:
                w = int(d["group_scen"][g])
                rows = np.flatnonzero(d["cand_group"] == g)
                o, hd = d["group_origin"][g], float(d["group_heading"][g])
                pops[w, :, 0, :3] = [o[0], o[1], hd]
                pops[w, :, 1:, :3] = _to_world(np.nan_to_num(d["cand_traj"][rows].astype(np.float64)), o, hd)
            online = np.concatenate([scorer.score(rs, int(t), d["scen_goal_world"], pops[:, i:i + chunk])
                                     for i in range(0, bank.k, chunk)], 1)
            gidx = g0 + gl
            ctx = {k: bank.groups[k][gidx] for k in keys}
            on_l.append(online[d["group_scen"][gl].astype(int)])
            off_l.append(np.concatenate([scorer.score_ego(ctx, traj_all[gidx, i:i + chunk])
                                         for i in range(0, bank.k, chunk)], 1))
            uns_l.append(unsafe_all[gidx])
        g0 += len(d["group_t"])
    return np.concatenate(on_l), np.concatenate(off_l), np.concatenate(uns_l).astype(bool)


def selection_stats(on: np.ndarray, off: np.ndarray, unsafe: np.ndarray) -> dict:
    g = np.arange(len(on))
    top_on, top_off = on.argmax(1), off.argmax(1)
    flip = top_on != top_off
    el_on, el_off = np.argsort(-on, 1)[:, :ELITE], np.argsort(-off, 1)[:, :ELITE]
    overlap = np.array([len(set(a) & set(b)) for a, b in zip(el_on, el_off)]) / ELITE
    std = off.std(1)
    srt = np.sort(off, 1)
    rank = lambda x: np.argsort(np.argsort(x, 1), 1)  # noqa: E731
    ro, rf = rank(on).astype(float), rank(off).astype(float)
    rho = [np.corrcoef(a, b)[0, 1] for a, b in zip(ro, rf)]
    return {
        "groups": int(len(on)),
        "max_abs_score_diff": float(np.abs(on - off).max()),
        "median_abs_score_diff": float(np.median(np.abs(on - off))),
        "median_within_group_score_std": float(np.median(std)),
        "median_offline_top1_minus_top2_gap": float(np.median(srt[:, -1] - srt[:, -2])),
        "top1_agreement": float(1 - flip.mean()),
        "top1_flips": int(flip.sum()),
        "flip_offline_score_gap": [float(off[i, top_off[i]] - off[i, top_on[i]]) for i in np.flatnonzero(flip)],
        "flip_changes_unsafe_label": int((unsafe[g, top_on] != unsafe[g, top_off])[flip].sum()),
        "top1_unsafe_rate_online": float(unsafe[g, top_on].mean()),
        "top1_unsafe_rate_offline": float(unsafe[g, top_off].mean()),
        "top8_elite_overlap_mean": float(overlap.mean()),
        "top8_elite_overlap_min": float(overlap.min()),
        "spearman_online_vs_offline_mean": float(np.mean(rho)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a0", default=A0_CKPT)
    p.add_argument("--a4", default=A4_CKPT)
    p.add_argument("--wm-checkpoint", default=WM_CKPT)
    p.add_argument("--data-dir", default=f"{DATA}/es_candidates")
    p.add_argument("--wm-dir", default=f"{DATA}/wm_predictions")
    p.add_argument("--edge-dir", default=f"{DATA}/road_edges")
    p.add_argument("--tfrecord-dir", default=TFRECORDS)
    p.add_argument("--max-files", type=int, default=2)
    p.add_argument("--chunk", type=int, default=64, help="candidates per forward (memory only)")
    p.add_argument("--out", default="")
    a = p.parse_args()

    from es_discriminator.real_es.dataset import load_bank
    from es_discriminator.real_es.deploy import DiscriminatorScorer
    from es_discriminator.real_es.wm_cache import load_forecaster

    forecaster = load_forecaster(a.wm_checkpoint, "cpu")
    report = {"wm_checkpoint": a.wm_checkpoint, "max_files": a.max_files, "models": {}}
    bank = files = None
    for name, ckpt in (("A0", a.a0), ("A4", a.a4)):
        scorer = DiscriminatorScorer(ckpt, device="cpu", forecaster=forecaster)
        entry = {"config": check_config(name, ckpt, scorer)}
        if bank is None:
            bank = load_bank(a.data_dir, max_files=a.max_files, cfg=scorer.cfg, verbose=False,
                             wm_dir=a.wm_dir, edge_dir=a.edge_dir)
            files = bank.files
        on, off, unsafe = online_offline(scorer, bank, files, a.tfrecord_dir, chunk=a.chunk)
        entry["online_vs_offline"] = selection_stats(on, off, unsafe)
        report["models"][name] = entry
        print(name, json.dumps(entry, default=str), flush=True)
    text = json.dumps(report, indent=1, default=str)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
