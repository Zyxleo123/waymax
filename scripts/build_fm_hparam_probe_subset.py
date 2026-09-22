#!/usr/bin/env python3
"""Build a GT probe subset for FM-ES hyperparam sweeps.

Sources
-------
``fail1500`` (default)
  Stratified sample of N scenes from the shared failure subset
  (``es_failure_subset_n1500_seed0``). Does **not** require prior FM-ES
  results — use this for the first corrected-conditioning sweep.

``mixed``
  Classic mix requiring both Diff-ES GT fail1500 and FM-ES GT fail1500:
    - Diff success / FM fail  (diagnose under-noising / local search)
    - Diff fail               (hard ceiling)

Writes under ``--out_dir`` (default: Flow_matching_sim)::

  target_scenarios.json
  subset_manifest.json
  tfrecords/training/   (symlinks)

Examples::

  python scripts/build_fm_hparam_probe_subset.py --source fail1500 --n 250 --seed 0
  python scripts/build_fm_hparam_probe_subset.py --source mixed --n 250 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_EXP = Path("/zfsauton/scratch/sbellad/immediate_exp")
DEFAULT_FM_ROOT = Path("/zfsauton/scratch/sbellad/Flow_matching_sim")
DEFAULT_OUT = DEFAULT_FM_ROOT / "fm_hparam_probe_n250_seed0"
DEFAULT_FAIL1500 = DEFAULT_EXP / "es_failure_subset_n1500_seed0"
DEFAULT_TF = "/zfsauton/scratch/eshau/womd/tf_example/training"
DEFAULT_LG = "/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
DEFAULT_GT_FILTER = (
    DEFAULT_EXP / "gt_filtered_fail1500_20260911_170730" / "gt_filter" / "gt_filter_summary.json"
)


def _latest_sim(run_folder: Path) -> Path:
    sims = sorted(run_folder.glob("simulation_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not sims:
        raise FileNotFoundError(f"No simulation_* under {run_folder}")
    return sims[0]


def load_results(run_dir: Path) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for p in run_dir.glob("*.json"):
        if p.name == "summary.json" or "instruction" in p.name:
            continue
        row = json.loads(p.read_text())
        if "scenario_idx" not in row:
            continue
        tf = Path(str(row.get("tfrecord", p.name))).name
        out[(tf, int(row["scenario_idx"]))] = row
    return out


def fail_type(row: dict) -> str:
    if row.get("overlap") and row.get("offroad"):
        return "overlap+offroad"
    if row.get("overlap"):
        return "overlap"
    if row.get("offroad"):
        return "offroad"
    if not row.get("goal_reached"):
        return "goal"
    return "other"


def stratified_sample(
    keys: list,
    *,
    type_of,
    n: int,
    rng: random.Random,
) -> list:
    if n <= 0 or not keys:
        return []
    by: dict[str, list] = defaultdict(list)
    for k in keys:
        by[type_of(k)].append(k)
    for lst in by.values():
        rng.shuffle(lst)

    quotas: dict[str, int] = {}
    total = len(keys)
    for t, lst in by.items():
        quotas[t] = max(1, round(n * len(lst) / total)) if total else 0
    while sum(quotas.values()) > n:
        t = max(quotas, key=lambda x: quotas[x])
        if quotas[t] > 1:
            quotas[t] -= 1
        else:
            break
    while sum(quotas.values()) < n and by:
        t = max(by, key=lambda x: len(by[x]) - quotas.get(x, 0))
        quotas[t] = quotas.get(t, 0) + 1

    chosen: list = []
    for t, q in quotas.items():
        chosen.extend(by[t][:q])
    if len(chosen) < n:
        leftover = [k for k in keys if k not in chosen]
        rng.shuffle(leftover)
        chosen.extend(leftover[: n - len(chosen)])
    return chosen[:n]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=str,
        default="fail1500",
        choices=["fail1500", "mixed"],
        help="fail1500 = stratified from shared hard subset; mixed needs Diff+FM fail1500.",
    )
    p.add_argument("--exp_root", type=str, default=str(DEFAULT_EXP))
    p.add_argument("--fail1500_dir", type=str, default=str(DEFAULT_FAIL1500))
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--tf_root", type=str, default=DEFAULT_TF)
    p.add_argument("--lane_graph_dir", type=str, default=DEFAULT_LG)
    p.add_argument("--gt_filter_summary", type=str, default=str(DEFAULT_GT_FILTER))
    p.add_argument("--n", type=int, default=250, help="Total scenes (must be divisible by num_worlds)")
    p.add_argument("--n_diff_ok_fm_fail", type=int, default=150, help="mixed source only")
    p.add_argument("--n_diff_fail", type=int, default=100, help="mixed source only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--num_worlds",
        type=int,
        default=1,
        help="Must divide --n. Default 1 keeps exactly N scenes across mixed shards.",
    )
    p.add_argument("--prefer_gt_ok", action="store_true", default=True)
    p.add_argument("--no_prefer_gt_ok", action="store_false", dest="prefer_gt_ok")
    return p.parse_args()


def _write_subset(
    *,
    out_dir: Path,
    target_map: dict[str, list[int]],
    kept_rows: list[dict],
    tf_root: Path,
    lg_dir: Path,
    args: argparse.Namespace,
    extra_manifest: dict,
) -> None:
    tf_link = out_dir / "tfrecords" / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    tf_link.mkdir(parents=True, exist_ok=True)

    missing = []
    for tf in target_map:
        src = tf_root / tf
        lg = lg_dir / f"{tf}.lanegraph.zip"
        dst = tf_link / tf
        if not src.exists():
            missing.append(f"tf:{src}")
        if not lg.exists():
            missing.append(f"lanegraph:{lg}")
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(src, dst)
    if missing:
        raise SystemExit("Missing inputs:\n  " + "\n  ".join(missing))

    manifest = {
        "source": args.source,
        "n_requested": args.n,
        "n_final": sum(len(v) for v in target_map.values()),
        "num_worlds": int(args.num_worlds),
        "seed": args.seed,
        "prefer_gt_ok": bool(args.prefer_gt_ok),
        "bucket_counts": dict(Counter(r["bucket"] for r in kept_rows)),
        "cases": kept_rows,
        "note": (
            "All GT other_agent_forecast. Use with slurm_fm_hparam_probe.sh. "
            f"Outputs live under {out_dir}."
        ),
        **extra_manifest,
    }
    (out_dir / "target_scenarios.json").write_text(json.dumps(target_map, indent=2) + "\n")
    (out_dir / "subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out_dir}")
    print(f"n_final={manifest['n_final']} buckets={manifest['bucket_counts']}")


def _pack_by_tf(
    chosen_rows: list[dict],
    *,
    nw: int,
) -> tuple[dict[str, list[int]], list[dict], list[tuple[str, int]]]:
    by_tf: dict[str, list[int]] = defaultdict(list)
    for row in chosen_rows:
        by_tf[row["tfrecord"]].append(int(row["scenario_idx"]))

    target_map: dict[str, list[int]] = {}
    dropped: list[tuple[str, int]] = []
    kept_rows: list[dict] = []
    for tf, idxs in sorted(by_tf.items()):
        idxs = sorted(set(idxs))
        n_keep = (len(idxs) // nw) * nw
        if n_keep < len(idxs):
            dropped.extend((tf, i) for i in idxs[n_keep:])
        keep = idxs[:n_keep]
        if keep:
            target_map[tf] = keep
        keep_set = set(keep)
        kept_rows.extend(
            r for r in chosen_rows if r["tfrecord"] == tf and r["scenario_idx"] in keep_set
        )
    return target_map, kept_rows, dropped


def build_from_fail1500(args: argparse.Namespace) -> None:
    fail_dir = Path(args.fail1500_dir)
    manifest_path = fail_dir / "subset_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"Missing {manifest_path}. Build fail1500 first:\n"
            "  python scripts/build_failure_subset_for_es.py --n 1500 --seed 0"
        )
    full = json.loads(manifest_path.read_text())
    scenarios = list(full["scenarios"])
    if len(scenarios) < args.n:
        raise SystemExit(f"fail1500 only has {len(scenarios)} scenes; need --n={args.n}")

    rng = random.Random(args.seed)
    keys = [(s["tfrecord"], int(s["scenario_idx"])) for s in scenarios]
    meta = {(s["tfrecord"], int(s["scenario_idx"])): s for s in scenarios}
    picked = stratified_sample(
        keys,
        type_of=lambda k: str(meta[k].get("fail_type", "other")),
        n=args.n,
        rng=rng,
    )
    rows = []
    for tf, idx in picked:
        s = meta[(tf, idx)]
        rows.append(
            {
                "tfrecord": tf,
                "scenario_idx": int(idx),
                "bucket": "fail1500",
                "fail_type": str(s.get("fail_type", "other")),
                "overlap": bool(s.get("overlap", False)),
                "offroad": bool(s.get("offroad", False)),
                "goal_reached": bool(s.get("goal_reached", True)),
            }
        )

    target_map, kept_rows, dropped = _pack_by_tf(rows, nw=int(args.num_worlds))
    if dropped:
        print(f"[warn] dropped {len(dropped)} scenes to satisfy num_worlds={args.num_worlds}")

    n_final = sum(len(v) for v in target_map.values())
    if n_final != args.n:
        raise SystemExit(
            f"Could not pack exactly n={args.n} with num_worlds={args.num_worlds}; got {n_final}. "
            "Use --num_worlds 1 (default) for an exact count."
        )

    _write_subset(
        out_dir=Path(args.out_dir),
        target_map=target_map,
        kept_rows=kept_rows,
        tf_root=Path(args.tf_root),
        lg_dir=Path(args.lane_graph_dir),
        args=args,
        extra_manifest={
            "fail1500_dir": str(fail_dir),
            "fail_type_counts": dict(Counter(r["fail_type"] for r in kept_rows)),
        },
    )
    print(f"fail_type mix={dict(Counter(r['fail_type'] for r in kept_rows))}")


def build_mixed(args: argparse.Namespace) -> None:
    if args.n_diff_ok_fm_fail + args.n_diff_fail != args.n:
        raise SystemExit("--n must equal --n_diff_ok_fm_fail + --n_diff_fail")

    exp = Path(args.exp_root)
    diff_dir = exp / "es_diffusion_gt_fail1500"
    fm_dir = exp / "es_fm_gt_fail1500"
    if not fm_dir.exists():
        # Also check Flow_matching_sim for a re-run.
        alt = DEFAULT_FM_ROOT / "es_fm_gt_fail1500"
        if alt.exists():
            fm_dir = alt
    diff = load_results(_latest_sim(diff_dir))
    fm = load_results(_latest_sim(fm_dir))
    keys = sorted(set(diff) & set(fm))

    gt_fail: set[tuple[str, int]] = set()
    gt_path = Path(args.gt_filter_summary)
    if args.prefer_gt_ok and gt_path.exists():
        gs = json.loads(gt_path.read_text())
        gt_fail = {(x["tfrecord"], int(x["scenario_idx"])) for x in gs.get("failing_keys", [])}
        print(f"Loaded GT-fail set size={len(gt_fail)} from {gt_path}")
    elif args.prefer_gt_ok:
        print(f"[warn] GT filter summary missing: {gt_path}; not filtering")

    dok_fmf = [k for k in keys if diff[k]["success"] and not fm[k]["success"]]
    dfail = [k for k in keys if not diff[k]["success"]]
    if args.prefer_gt_ok and gt_fail:
        dok_fmf_g = [k for k in dok_fmf if k not in gt_fail]
        dfail_g = [k for k in dfail if k not in gt_fail]
        if len(dok_fmf_g) >= args.n_diff_ok_fm_fail:
            dok_fmf = dok_fmf_g
        if len(dfail_g) >= args.n_diff_fail:
            dfail = dfail_g

    rng = random.Random(args.seed)
    pick_a = stratified_sample(
        dok_fmf,
        type_of=lambda k: fail_type(fm[k]),
        n=args.n_diff_ok_fm_fail,
        rng=rng,
    )
    pick_b = stratified_sample(
        [k for k in dfail if k not in pick_a],
        type_of=lambda k: fail_type(diff[k]),
        n=args.n_diff_fail,
        rng=rng,
    )
    chosen = pick_a + pick_b
    if len(chosen) != args.n:
        raise SystemExit(f"Only selected {len(chosen)} / {args.n} scenes")

    rows = []
    for tf, idx in chosen:
        bucket = "diff_ok_fm_fail" if (tf, idx) in pick_a else "diff_fail"
        rows.append(
            {
                "tfrecord": tf,
                "scenario_idx": int(idx),
                "bucket": bucket,
                "fm_fail_type": fail_type(fm[(tf, idx)]),
                "diff_fail_type": fail_type(diff[(tf, idx)]) if not diff[(tf, idx)]["success"] else "success",
                "diff_success": bool(diff[(tf, idx)]["success"]),
                "fm_success": bool(fm[(tf, idx)]["success"]),
                "gt_filter_ok": (tf, idx) not in gt_fail if gt_fail else None,
            }
        )

    target_map, kept_rows, dropped = _pack_by_tf(rows, nw=int(args.num_worlds))
    if dropped:
        print(f"[warn] dropped {len(dropped)} scenes to satisfy num_worlds={args.num_worlds}: {dropped}")
    n_final = sum(len(v) for v in target_map.values())
    if n_final != args.n:
        raise SystemExit(f"Packed n_final={n_final} != requested n={args.n}")

    _write_subset(
        out_dir=Path(args.out_dir),
        target_map=target_map,
        kept_rows=kept_rows,
        tf_root=Path(args.tf_root),
        lg_dir=Path(args.lane_graph_dir),
        args=args,
        extra_manifest={
            "fm_fail_type_counts_in_diff_ok_bucket": dict(
                Counter(r["fm_fail_type"] for r in kept_rows if r["bucket"] == "diff_ok_fm_fail")
            ),
            "diff_fail_type_counts_in_diff_fail_bucket": dict(
                Counter(r["diff_fail_type"] for r in kept_rows if r["bucket"] == "diff_fail")
            ),
        },
    )


def main() -> None:
    args = parse_args()
    nw = int(args.num_worlds)
    if args.n % nw != 0:
        raise SystemExit(f"--n={args.n} must be divisible by --num_worlds={nw}")

    if args.source == "fail1500":
        build_from_fail1500(args)
    else:
        build_mixed(args)


if __name__ == "__main__":
    main()
