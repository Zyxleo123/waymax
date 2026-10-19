"""Closed-loop report: one run against baseline runs, paired on common scenes.

Writes report.md (tables overall and per group, bootstrap CIs, fixed / broken counts), report.json and
per_scene.csv to --out_dir, and a DONE marker once the run has --expected scenes (or, with --done_glob,
once every matching marker file exists).

  # fail1500 (G2): default baselines, grouped by the bank's train / val / test split
  python -m anchor_wm.g2_report --run_dir .../closed_loop/<run> --out_dir .../results/<run> --expected 1500
  # WOMD val shards (G3): grouped by shard
  python -m anchor_wm.g2_report --run_dir .../closed_loop_val/<run> --out_dir ... --group shard \
      --baseline "MF + rule=/zfsauton/scratch/ishanj/world_model_sim/<mf val run>" --done_glob "shard_*/DONE" --expected_done 10
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import os
import re

import numpy as np

CL = "/zfsauton/scratch/ishanj/anchor_wm/closed_loop/"
BASELINES = {
    "MF + rule": CL + "fail1500_es_mfwm_multiFuture_GaussianWorldModel_20261005_092950_epoch_0450_reward_rule_offbinary_seed0",
    "MF + v3": "/zfsauton/scratch/ishanj/es_discriminator/closed_loop/fail1500_es_disc_sbellad_v3_discwm_20261005_092950_epoch_0450_wm_mf_20261005_092950_epoch_0450_offbinary_seed0",
    "A2 + rule": CL + "fail1500_es_anchorwm_joint_20261018_011644_final_reward_rule_offbinary_seed0",
    "A2 scorer": CL + "fail1500_es_anchorwm_joint_20261018_011644_final_reward_scorer_offbinary_seed0",
    # margin_tiebreak offroad only changes the rule reward; a learned reward ignores the offroad mode,
    # so these are equally valid baselines for a learned-scorer run (and the stronger rule baselines).
    "MF + rule (margin)": "/zfsauton/scratch/ishanj/world_model_sim/wm_temperature_sweep/es_diffusion_wm_mf_20261005_092950_epoch_0450_expected_independent_log_offmargin_tiebreak_it4_rp10_T1_fail1500_waymaxz",
    "A2 + rule (margin)": CL + "fail1500_es_anchorwm_joint_20261018_011644_final_reward_rule_offmargin_tiebreak_seed0",
}
SPLIT_BANK = ("/zfsauton/scratch/ishanj/es_discriminator/es_candidate_data/fail1500_es_diffusion_wm_mf_20261005_092950"
              "_epoch_0450_expected_independent_offbinary_seed0/train_set")
KEYS = ("success", "overlap", "offroad", "goal_reached")
RX = re.compile(r"^(.+)\.scenario_(\d+)\.json$")


def load(run_dir: str) -> dict:
    out = {}
    for f in glob.glob(os.path.join(run_dir, "*", "*.scenario_*.json")):   # chunk_* / shard_* subdirs
        m = RX.match(os.path.basename(f))
        if m:
            with open(f) as fh:
                out[(m.group(1), int(m.group(2)))] = json.load(fh)
    return out


def scene_splits() -> dict:
    with open(os.path.join(SPLIT_BANK, "scenario_split_seed0.json")) as fh:
        assign = json.load(fh)["assignment"]
    split = {}
    for f in glob.glob(os.path.join(SPLIT_BANK, "es_candidates", "*.npz")):
        if re.search(r"\.scen_[\d_]+\.npz$", f):
            with np.load(f) as z:
                for tf, i, s in zip(z["scen_tfrecord"], z["scen_index"], z["scen_id"]):
                    split[(str(tf), int(i))] = assign.get(str(s), "")
    return split


def summarize(d: dict, keys: list) -> dict:
    return {**{k: float(np.mean([bool(d[s][k]) for s in keys])) for k in KEYS},
            "goal_dist_median_m": float(np.median([d[s]["final_goal_distance_m"] for s in keys]))}


def paired(a: dict, b: dict, keys: list, n_boot: int = 2000) -> dict:
    diff = np.array([float(a[s]["success"]) - float(b[s]["success"]) for s in keys])
    rng = np.random.default_rng(0)
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    return {"delta_pts": 100 * float(diff.mean()), "ci95": [100 * float(np.percentile(boot, 2.5)), 100 * float(np.percentile(boot, 97.5))],
            "fixed": int(sum((not b[s]["success"]) and a[s]["success"] for s in keys)),
            "broken": int(sum(b[s]["success"] and not a[s]["success"] for s in keys))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--label", default="ft_rule_hdg scorer")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--expected", type=int, default=1500)
    p.add_argument("--baseline", action="append", default=[], metavar="NAME=DIR",
                   help="baseline run (repeatable); default: the fail1500 baselines")
    p.add_argument("--group", choices=("split", "shard"), default="split",
                   help="per-group tables: the fail1500 bank split, or the source tfrecord shard")
    p.add_argument("--done_glob", default="", help="complete once --expected_done files match this glob in --run_dir")
    p.add_argument("--expected_done", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    baselines = dict(b.split("=", 1) for b in args.baseline) if args.baseline else BASELINES
    runs = {args.label: load(args.run_dir), **{n: load(d) for n, d in baselines.items()}}
    if args.group == "split":
        split = scene_splits()
        groups = ("train", "val", "test")
    else:
        split = {k: re.sub(r".*tfrecord-(\d+)-of-.*", r"shard \1", k[0]) for k in runs[args.label]}
        groups = sorted(set(split.values()))
    main_run = runs[args.label]
    common = sorted(set.intersection(*[set(d) for d in runs.values()]))
    if args.done_glob:
        complete = len(glob.glob(os.path.join(args.run_dir, args.done_glob))) >= args.expected_done
    else:
        complete = len(main_run) >= args.expected
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    report = {"generated": stamp, "run_dir": args.run_dir, "scenes_done": len(main_run), "expected": args.expected,
              "complete": complete, "splits": {}}
    target = f"{args.expected_done} shards" if args.done_glob else f"{args.expected} scenes"
    report["target"] = target
    lines = [f"# Closed loop: {args.label} vs baselines", "",
             f"Generated {stamp}. Run: `{args.run_dir}`", "",
             f"**{len(main_run)} scenes done, target {target}** ({'final' if complete else 'partial'}); "
             f"paired comparisons use the {len(common)} scenes every run has.", ""]
    for sp in ("all",) + tuple(groups):
        keys = common if sp == "all" else [k for k in common if split.get(k) == sp]
        if not keys:
            continue
        rows = {n: summarize(d, keys) for n, d in runs.items()}
        deltas = {n: paired(main_run, runs[n], keys) for n in baselines}
        report["splits"][sp] = {"n": len(keys), "metrics": rows, f"{args.label} minus": deltas}
        lines += [f"## {sp} (n = {len(keys)})", "",
                  "| Reward | Success | Overlap | Offroad | Goal reached | Median goal dist |",
                  "|---|---|---|---|---|---|"]
        for n, r in rows.items():
            lines.append(f"| {n} | {100 * r['success']:.1f}% | {100 * r['overlap']:.1f}% | {100 * r['offroad']:.1f}% | "
                         f"{100 * r['goal_reached']:.1f}% | {r['goal_dist_median_m']:.2f} m |")
        lines += ["", f"| {args.label} minus | Δ success (pts) | 95% CI | Fixed / broken |", "|---|---|---|---|"]
        for n, dd in deltas.items():
            lines.append(f"| {n} | {dd['delta_pts']:+.1f} | {dd['ci95'][0]:+.1f} to {dd['ci95'][1]:+.1f} | {dd['fixed']} / {dd['broken']} |")
        lines.append("")

    with open(os.path.join(args.out_dir, "report.md"), "w") as fh:
        fh.write("\n".join(lines))
    with open(os.path.join(args.out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    with open(os.path.join(args.out_dir, "per_scene.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        names = list(runs)
        w.writerow(["tfrecord", "scenario", "split"] + [f"{n}:{k}" for n in names for k in KEYS + ("final_goal_distance_m",)])
        for key in sorted(main_run):
            row = [key[0], key[1], split.get(key, "")]
            for n in names:
                r = runs[n].get(key)
                row += [r.get(k) if r else "" for k in KEYS + ("final_goal_distance_m",)]
            w.writerow(row)
    if complete:
        open(os.path.join(args.out_dir, "DONE"), "w").write(stamp + "\n")
    print(f"{stamp}: {len(main_run)}/{args.expected} scenes, {len(common)} paired, complete={complete}", flush=True)


if __name__ == "__main__":
    main()
