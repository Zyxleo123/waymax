"""Offline ranking check (G1) of an AnchorWM scorer against a trained es_discriminator run, from the bank cache.

Scores exactly the reference run's saved held-out groups (`<ref_dir>/<split>_predictions.npz`, all
candidates per group, in the bank's candidate order) using cache_bank inputs (no tfrecord rebuild), then
runs es_discriminator.real_es.evaluate.full_report on both score sets (same filters, bootstrap and pickers).

  python -m anchor_wm.eval_bank_cached --ckpt .../ft_<ts>/best.pt \
      --ref_dir /zfsauton/scratch/sbellad/es_discriminator/safety_progress/safety_progress_v3_20261016_155456_86103 \
      --cache_dir .../bank_cache/fail1500_plus_womd_train_s02_11 --out_dir ...
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from anchor_wm.bank_data import collate, load_file_groups
from anchor_wm.eval_bank import comparison_table, file_offsets, load_model
from anchor_wm.train import move


@torch.no_grad()
def score_groups(model, batch_file: str, cache_dir: str, local_groups: np.ndarray, device, amp: bool,
                 batch_groups: int, chunk: int) -> np.ndarray:
    recs = load_file_groups(batch_file, cache_dir, local_groups, cands_per_group=0, max_unsafe_frac=0.5,
                            rng=np.random.default_rng(0))
    out = []
    for s0 in range(0, len(recs), batch_groups):
        f = move(collate(recs[s0:s0 + batch_groups]), device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            enc, dec = model(f)
            dec = {k: v.float() for k, v in dec.items()}
            ctx = model.scorer.context(enc, dec, f, 0.0)
            total, _ = model.scorer(ctx, f["cand_xy"], f["cand_yaw"], chunk=chunk, heading=f["cand_yaw"])
        out.append(total.float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ref_dir", required=True, help="es_discriminator run dir (config.json + <split>_predictions.npz)")
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--batch_groups", type=int, default=4)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from es_discriminator.real_es import evaluate as ev
    from es_discriminator.real_es.check_alignment import recorder_config
    from es_discriminator.real_es.dataset import invalid_scene_keys, load_bank

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    ref_cfg = json.load(open(os.path.join(args.ref_dir, "config.json")))
    ref = dict(np.load(os.path.join(args.ref_dir, f"{args.split}_predictions.npz")))
    data_dir = ref_cfg["data_dir"]
    bank = load_bank(data_dir, cfg=recorder_config(data_dir), wm_dir=ref_cfg.get("wm_dir") or None,
                     edge_dir=ref_cfg.get("edge_dir") or None)
    if [os.path.basename(f) for f in bank.files] != [str(x) for x in ref["files"]]:
        raise RuntimeError("bank files differ from the reference prediction files")
    gidx = ref["group_index"].astype(np.int64)
    if not (np.array_equal(bank.groups["scen_id"][gidx], ref["scen_id"])
            and np.array_equal(bank.groups["group_t"][gidx], ref["group_t"])):
        raise RuntimeError("reference group index does not match the bank")
    offsets = file_offsets(bank.files)
    fidx = np.searchsorted(offsets, gidx, side="right") - 1
    cached = np.array([os.path.exists(os.path.join(args.cache_dir, os.path.basename(f)[:-4] + ".anchor_cache.npz"))
                       for f in bank.files])
    keep = cached[fidx]
    if args.max_files:
        keep &= np.isin(fidx, np.unique(fidx[keep])[:args.max_files])
    if not keep.all():
        print(f"scoring {int(keep.sum())}/{len(keep)} groups (files without cache or --max_files)", flush=True)
    gidx, fidx, ref_scores = gidx[keep], fidx[keep], ref["scores"][keep]

    model = load_model(args.ckpt, device)
    scores = np.zeros((len(gidx), bank.k), np.float32)
    t0 = time.time()
    files = np.unique(fidx)
    for n, fi in enumerate(files):
        pos = np.flatnonzero(fidx == fi)
        scores[pos] = score_groups(model, bank.files[fi], args.cache_dir, gidx[pos] - offsets[fi], device, amp,
                                   args.batch_groups, args.chunk)
        if (n + 1) % 50 == 0 or n + 1 == len(files):
            print(f"scored {n + 1}/{len(files)} files ({time.time() - t0:.0f}s)", flush=True)

    invalid = invalid_scene_keys(ref_cfg["invalid_at_start"])
    kw = {"progress_key": ref_cfg.get("progress_key", "goal_gain"), "margin": ref_cfg.get("progress_margin", 2.0)}
    rep = ev.full_report(bank, gidx, scores, invalid=invalid, n_boot=args.n_boot, **kw)
    rep_ref = ev.full_report(bank, gidx, ref_scores, invalid=invalid, n_boot=args.n_boot, **kw)
    os.makedirs(args.out_dir, exist_ok=True)
    ev.save_predictions(os.path.join(args.out_dir, f"{args.split}_predictions.npz"), bank, gidx, scores)
    json.dump(rep, open(os.path.join(args.out_dir, f"eval_{args.split}.json"), "w"), indent=1)
    json.dump(rep_ref, open(os.path.join(args.out_dir, f"ref_eval_{args.split}.json"), "w"), indent=1)
    header = (f"# AnchorWM scorer vs reference on {args.split} ({len(gidx)} groups)\n\nAnchorWM: {args.ckpt}\n\n"
              f"Reference (column 'B0 discriminator'): {args.ref_dir}\n\n")
    table = comparison_table(rep, rep_ref)
    with open(os.path.join(args.out_dir, "comparison.md"), "w") as fh:
        fh.write(header + table + "\n")
    print(header + table, flush=True)


if __name__ == "__main__":
    main()
