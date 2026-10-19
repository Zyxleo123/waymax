"""Score recorded Diffusion-ES candidates with an AnchorWM checkpoint and compare with B0.

B0 is a trained es_discriminator run (`--b0_dir`, with `test_predictions.npz`). This scores the
same groups: for every batch file it rebuilds the replan state exactly as
es_discriminator.real_es.wm_cache does (original log with the executed ego trajectory),
builds AnchorWM inputs with the training preprocessing (`data.build_arrays`), runs the model
once per replan step and scores the group's recorded candidates. Goal and ego velocity come
from the bank context, as B0 sees them. Both score sets go through
es_discriminator.real_es.evaluate.full_report (same filters and bootstrap).

Writes <out_dir>/{eval_test.json, eval_test.md, b0_eval_test.json, comparison.md, test_predictions.npz}.

  python -m anchor_wm.eval_bank --ckpt .../final.pt --b0_dir .../safety_progress/<run> \
      --tfrecord_dir ... --out_dir ...
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from anchor_wm.config import AnchorWMConfig
from anchor_wm.data import build_arrays, payload_from_sim_state
from anchor_wm.model import AnchorWorldModel

KEY_ROWS = (("discriminator", "unsafe"), ("discriminator", "overlap"), ("discriminator", "offroad"),
            ("discriminator", "progress"), ("discriminator", "goal_gain"), ("discriminator", "ade"),
            ("ranking", "pairwise_accuracy"), ("ranking", "safety_auroc"))


def load_model(path: str, device: torch.device) -> AnchorWorldModel:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = AnchorWMConfig.from_dict(ck["config"])
    anchors_m = ck["model"]["anchors"] * float(cfg.max_range)
    model = AnchorWorldModel(cfg, anchors_m)
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def file_offsets(files: list[str]) -> np.ndarray:
    counts = []
    for f in files:
        with np.load(f) as z:
            counts.append(len(z["group_t"]))
    return np.concatenate([[0], np.cumsum(counts)])


@torch.no_grad()
def score_file(model, path: str, local_groups: np.ndarray, bank, gidx_global: np.ndarray, tfrecord_dir: str,
               device: torch.device, chunk: int, amp: bool) -> tuple[np.ndarray, dict]:
    import jax
    from es_discriminator.analyze_es_decisions import load_batch_state
    from es_discriminator.real_es.check_alignment import replaced_state

    cfg = model.config
    d = dict(np.load(path))
    rs = replaced_state(jax.device_get(load_batch_state(d, tfrecord_dir)), d)
    payload = payload_from_sim_state(rs)
    k = bank.k
    out = np.zeros((len(local_groups), k), np.float32)
    stats = {"origin_err_m": 0.0, "heading_err_rad": 0.0}
    by_t = defaultdict(list)
    for i, g in enumerate(local_groups):
        by_t[int(d["group_t"][g])].append(i)
    for t, items in by_t.items():
        arrays = build_arrays(payload, anchor_step=t, cfg=cfg, with_targets=False)
        worlds = np.array([int(d["group_scen"][local_groups[i]]) for i in items])
        rows = {w: j for j, w in enumerate(np.unique(worlds))}
        sel = np.array(sorted(rows, key=rows.get))
        f = {key: torch.from_numpy(np.ascontiguousarray(v[sel])).to(device) for key, v in arrays.items()
             if key not in ("origin_xy", "origin_yaw")}
        for i in items:
            g, gg = local_groups[i], gidx_global[i]
            j = rows[int(d["group_scen"][g])]
            stats["origin_err_m"] = max(stats["origin_err_m"], float(np.abs(arrays["origin_xy"][sel[j]] - d["group_origin"][g]).max()))
            dh = (arrays["origin_yaw"][sel[j]] - float(d["group_heading"][g]) + np.pi) % (2 * np.pi) - np.pi
            stats["heading_err_rad"] = max(stats["heading_err_rad"], float(abs(dh)))
            # Goal / ego velocity exactly as B0 sees them (bank context: goal / 100, ego_vel / 25).
            f["goal_xy"][j] = torch.from_numpy(bank.groups["goal"][gg] * 100.0).to(device)
            f["ego_vel"][j] = torch.from_numpy(bank.groups["ego_vel"][gg] * 25.0).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            enc, dec = model(f)
            dec = {key: v.float() for key, v in dec.items()}
            ctx = model.scorer.context(enc, dec, f, 0.0)
            for i in items:
                g, gg = local_groups[i], gidx_global[i]
                j = rows[int(d["group_scen"][g])]
                one = {key: ({kk: vv[j:j + 1] for kk, vv in v.items()} if isinstance(v, dict)
                             else v[j:j + 1] if torch.is_tensor(v) else v) for key, v in ctx.items()}
                traj = torch.from_numpy(bank.cands["traj"][bank.rows(int(gg))]).to(device)[None]   # [1,K,25,3]
                total, _ = model.scorer(one, traj[..., :2], traj[..., 2], chunk=chunk)
                out[i] = total[0].float().cpu().numpy()
    return out, stats


def comparison_table(ours: dict, b0: dict) -> str:
    lines = ["| subset | metric | AnchorWM | B0 discriminator | ES reward | GT oracle | random |",
             "|---|---|---|---|---|---|---|"]
    for sub in ("filtered", "es_unsafe_scenes_all_replans", "es_unsafe_scenes_first_bad_replan"):
        o, b = ours.get(sub, {}), b0.get(sub, {})
        for picker, metric in KEY_ROWS:
            mean = lambda r, p=picker: (r.get(p, {}).get(metric, {}) or {}).get("mean")  # noqa: E731
            fmt = lambda v: "-" if v is None else f"{v:.4f}"  # noqa: E731
            extra = ["-", "-", "-"] if picker == "ranking" else [
                fmt((o.get(p, {}).get(metric, {}) or {}).get("mean")) for p in ("es_reward", "gt_oracle", "random")]
            lines.append(f"| {sub} (n={o.get('n_groups', 0)}) | {picker}/{metric} | {fmt(mean(o))} | {fmt(mean(b))} | "
                         + " | ".join(extra) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--b0_dir", required=True, help="es_discriminator safety_progress run dir")
    p.add_argument("--bank_dir", default="", help="es_candidates dir (default: B0's data_dir)")
    p.add_argument("--tfrecord_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--max_files", type=int, default=0, help="only groups from the first N files (smoke)")
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from es_discriminator.real_es import evaluate as ev
    from es_discriminator.real_es.check_alignment import recorder_config
    from es_discriminator.real_es.dataset import invalid_scene_keys, load_bank

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    b0_cfg = json.load(open(os.path.join(args.b0_dir, "config.json")))
    b0 = dict(np.load(os.path.join(args.b0_dir, f"{args.split}_predictions.npz")))
    # Same file set as B0: its data_dir, filtered to files with its WM / road-edge caches.
    bank_dir = args.bank_dir or b0_cfg["data_dir"]
    bank = load_bank(bank_dir, cfg=recorder_config(bank_dir), wm_dir=b0_cfg.get("wm_dir") or None,
                     edge_dir=b0_cfg.get("edge_dir") or None)
    names = [os.path.basename(f) for f in bank.files]
    if names != [str(x) for x in b0["files"]]:
        raise RuntimeError("bank files differ from the B0 prediction files; pass B0's training set")
    gidx = b0["group_index"].astype(np.int64)
    if not (np.array_equal(bank.groups["scen_id"][gidx], b0["scen_id"])
            and np.array_equal(bank.groups["group_t"][gidx], b0["group_t"])):
        raise RuntimeError("B0 group index does not match the bank")
    offsets = file_offsets(bank.files)
    fidx = np.searchsorted(offsets, gidx, side="right") - 1
    keep = np.ones(len(gidx), bool) if not args.max_files else fidx < args.max_files
    gidx, fidx, b0_scores = gidx[keep], fidx[keep], b0["scores"][keep]

    model = load_model(args.ckpt, device)
    scores = np.zeros((len(gidx), bank.k), np.float32)
    worst = {"origin_err_m": 0.0, "heading_err_rad": 0.0}
    t0 = time.time()
    files = np.unique(fidx)
    for n, fi in enumerate(files):
        pos = np.flatnonzero(fidx == fi)
        local = gidx[pos] - offsets[fi]
        s, st = score_file(model, bank.files[fi], local, bank, gidx[pos], args.tfrecord_dir, device, args.chunk, amp)
        scores[pos] = s
        worst = {k: max(worst[k], st[k]) for k in worst}
        if (n + 1) % 10 == 0 or n + 1 == len(files):
            print(f"scored {n + 1}/{len(files)} files ({time.time() - t0:.0f}s) frame check {worst}", flush=True)
    if worst["origin_err_m"] > 1e-2 or worst["heading_err_rad"] > 1e-3:
        raise RuntimeError(f"model frame != recorded group frame: {worst}")

    invalid = invalid_scene_keys(b0_cfg["invalid_at_start"])
    kw = {"progress_key": b0_cfg.get("progress_key", "goal_gain"), "margin": b0_cfg.get("progress_margin", 2.0)}
    rep = ev.full_report(bank, gidx, scores, invalid=invalid, n_boot=args.n_boot, **kw)
    rep_b0 = ev.full_report(bank, gidx, b0_scores, invalid=invalid, n_boot=args.n_boot, **kw)
    os.makedirs(args.out_dir, exist_ok=True)
    ev.save_predictions(os.path.join(args.out_dir, f"{args.split}_predictions.npz"), bank, gidx, scores)
    json.dump(rep, open(os.path.join(args.out_dir, f"eval_{args.split}.json"), "w"), indent=1)
    json.dump(rep_b0, open(os.path.join(args.out_dir, f"b0_eval_{args.split}.json"), "w"), indent=1)
    with open(os.path.join(args.out_dir, f"eval_{args.split}.md"), "w") as fh:
        fh.write(ev.to_markdown(rep, f"AnchorWM scorer on held-out {args.split} groups ({args.ckpt})"))
    table = comparison_table(rep, rep_b0)
    header = (f"# AnchorWM vs B0 on {args.split} ({len(gidx)} groups)\n\nAnchorWM: {args.ckpt}\n\n"
              f"B0: {args.b0_dir}\n\nframe check: {worst}\n\n")
    with open(os.path.join(args.out_dir, "comparison.md"), "w") as fh:
        fh.write(header + table + "\n")
    print(header + table, flush=True)


if __name__ == "__main__":
    main()
