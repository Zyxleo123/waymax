"""Closed-loop comparison: learned-discriminator ES reward vs the matching handcrafted ES baseline.

Pairs runs scene by scene on (tfrecord, scenario_idx) and reports success, collision (overlap), offroad,
goal reached, progress (d0 - final goal distance, d0 = ego distance to goal at the start step), goal
distances, paired bootstrap CIs, fixed/broken counts, the fail1500 breakdown by discriminator split
(train/val/test; train/val scenes are in-distribution for the discriminator) and the discriminator
overhead from discriminator_stats.json.

    python -m es_discriminator.real_es.compare_closed_loop \
        --pair "fail1500/gt=<disc_sim_dir>::<baseline_sim_dir>" [--pair ...] --out_dir DIR
A side may list several simulation dirs separated by "," (trainval = TRAIN dir,VAL dir).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

WOMD_TF_ROOT = "/zfsauton/datasets/womd/tf_example"
SPLIT_JSON = "/zfsauton/scratch/sbellad/es_discriminator/splits/fail1500_scenario_split_seed0.json"
START_T = 10
SCENE_RE = re.compile(r"^(?P<tf>.+)\.scenario_(?P<idx>\d+)\.json$")
METRICS = ("success", "collision", "offroad", "goal_reached", "progress_m", "final_goal_dist_m",
           "min_goal_dist_m")
BINARY = {"success", "collision", "offroad", "goal_reached"}


def load_scenes(dirs: list[str]) -> dict[tuple[str, int], dict]:
    out = {}
    for d in dirs:
        for p in sorted(Path(d).glob("*.scenario_*.json")):
            m = SCENE_RE.match(p.name)
            if m is None:
                continue
            r = json.loads(p.read_text())
            key = (m["tf"], int(m["idx"]))
            if key in out:
                raise ValueError(f"duplicate scene {key} in {dirs}")
            out[key] = {
                "success": float(r["success"]), "collision": float(r["overlap"]),
                "offroad": float(r["offroad"]), "goal_reached": float(r["goal_reached"]),
                "final_goal_dist_m": float(r["final_goal_distance_m"]),
                "min_goal_dist_m": float(r["min_goal_distance_m"]),
                "goal_xy": r["goal_xy"], "ego_idx": int(r["ego_idx"]),
            }
    return out


_TF_INFO: dict[tuple[str, int], tuple[str, np.ndarray]] = {}


def tfrecord_info(keys) -> None:
    """scenario_id and ego (x, y) at START_T for every (tfrecord, idx, ego_idx) not yet cached."""
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    feats = {"scenario/id": tf.io.FixedLenFeature([], tf.string),
             "state/current/x": tf.io.FixedLenFeature([128], tf.float32),
             "state/current/y": tf.io.FixedLenFeature([128], tf.float32)}
    by_file: dict[str, set[int]] = {}
    for tfn, idx in keys:
        if (tfn, idx) not in _TF_INFO:
            by_file.setdefault(tfn, set()).add(idx)
    for tfn, idxs in by_file.items():
        sub = "training" if tfn.startswith("training") else "validation"
        path = os.path.join(WOMD_TF_ROOT, sub, tfn)
        want = max(idxs)
        for i, rec in enumerate(tf.data.TFRecordDataset(path)):
            if i in idxs:
                ex = tf.io.parse_single_example(rec, feats)
                xy = np.stack([ex["state/current/x"].numpy(), ex["state/current/y"].numpy()], -1)
                _TF_INFO[(tfn, i)] = (ex["scenario/id"].numpy().decode(), xy)
            if i >= want:
                break


def add_progress(scenes: dict) -> None:
    for key, r in scenes.items():
        sid, xy = _TF_INFO[key]
        r["scenario_id"] = sid
        d0 = float(np.linalg.norm(xy[r["ego_idx"]] - np.asarray(r["goal_xy"])))
        r["d0_m"] = d0
        r["progress_m"] = d0 - r["final_goal_dist_m"]


def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> list[float]:
    if len(x) == 0:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def paired(disc: dict, base: dict, keys: list) -> dict:
    out = {"n": len(keys)}
    for m in METRICS:
        a = np.array([disc[k][m] for k in keys])
        b = np.array([base[k][m] for k in keys])
        res = {"learned": float(a.mean()) if len(a) else float("nan"),
               "handcrafted": float(b.mean()) if len(b) else float("nan"),
               "diff": float((a - b).mean()) if len(a) else float("nan"), "diff_ci95": boot_ci(a - b)}
        if m in BINARY:
            good = m in ("success", "goal_reached")
            res["fixed"] = int(((a > b) if good else (a < b)).sum())
            res["broken"] = int(((a < b) if good else (a > b)).sum())
        out[m] = res
    return out


def overhead(dirs: list[str]) -> dict | None:
    stats = [json.loads(Path(d, "discriminator_stats.json").read_text())
             for d in dirs if Path(d, "discriminator_stats.json").exists()]
    if not stats:
        return None
    out = {"replans": sum(s.get("replans_completed", 0) for s in stats)}
    for k in ("discriminator_ms_per_replan", "context_ms_per_replan", "wm_forward_ms_per_replan",
              "score_ms_per_call"):
        vals = [s[k] for s in stats if isinstance(s.get(k), dict) and s[k].get("n")]
        if vals:
            n = sum(v["n"] for v in vals)
            out[k] = {"mean": sum(v["mean"] * v["n"] for v in vals) / n, "n": n,
                      "median_per_dir": [v["median"] for v in vals], "p90_per_dir": [v["p90"] for v in vals]}
    out["wm_source"] = sorted({s.get("wm_source") for s in stats})
    out["calls_per_replan"] = sorted({int(c) for s in stats for c in s.get("scoring_calls_per_replan", {})})
    out["max_anchor_err_m"] = max(s.get("max_anchor_err_m", 0.0) for s in stats)
    return out


def fmt(r: dict, m: str) -> str:
    s = 100 if m in BINARY else 1
    lo, hi = r["diff_ci95"]
    extra = f" (+{r['fixed']}/-{r['broken']})" if "fixed" in r else ""
    return (f"{r['learned'] * s:.2f} vs {r['handcrafted'] * s:.2f}; {r['diff'] * s:+.2f} "
            f"[{lo * s:+.2f}, {hi * s:+.2f}]{extra}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True,
                    help="name=disc_dir[,disc_dir...]::baseline_dir[,baseline_dir...]")
    ap.add_argument("--split_json", default=SPLIT_JSON)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--allow_partial", action="store_true",
                    help="compare the scenes both runs finished (default: require identical scene sets)")
    args = ap.parse_args()
    split = json.loads(Path(args.split_json).read_text())["assignment"]

    report, lines = {}, ["# Closed-loop: learned discriminator reward vs handcrafted ES", "",
                         "Rates in %, distances in m; learned vs handcrafted; paired diff [95% bootstrap CI] "
                         "(+fixed/-broken scenes).", ""]
    for spec in args.pair:
        name, rest = spec.split("=", 1)
        dsides, bsides = rest.split("::")
        ddirs, bdirs = dsides.split(","), bsides.split(",")
        disc, base = load_scenes(ddirs), load_scenes(bdirs)
        n_disc, n_base = len(disc), len(base)
        common = sorted(set(disc) & set(base))
        missing = sorted(set(base) - set(disc))
        if missing and not args.allow_partial:
            raise SystemExit(f"{name}: {len(missing)} baseline scenes missing in learned run (use --allow_partial)")
        tfrecord_info(common)
        disc = {k: disc[k] for k in common}
        base = {k: base[k] for k in common}
        add_progress(disc)
        add_progress(base)
        groups = {"all": common}
        if "fail1500" in name:
            for s in ("train", "val", "test"):
                groups[f"disc_split_{s}"] = [k for k in common if split.get(disc[k]["scenario_id"]) == s]
            groups["disc_split_unassigned"] = [k for k in common if disc[k]["scenario_id"] not in split]
        if any(k[0].startswith("validation") for k in common):
            groups["womd_train_shards"] = [k for k in common if k[0].startswith("training")]
            groups["womd_val_shards"] = [k for k in common if k[0].startswith("validation")]
        rep = {"disc_dirs": ddirs, "baseline_dirs": bdirs, "n_learned": n_disc, "n_baseline": n_base,
               "n_common": len(common), "groups": {g: paired(disc, base, ks) for g, ks in groups.items() if ks},
               "overhead": overhead(ddirs)}
        report[name] = rep
        lines += [f"## {name}", "", f"learned: {', '.join(ddirs)}", f"baseline: {', '.join(bdirs)}",
                  f"scenes: learned {n_disc}, baseline {n_base}, paired {len(common)}", ""]
        lines += ["| group | n | " + " | ".join(METRICS) + " |", "|" + "---|" * (len(METRICS) + 2)]
        for g, r in rep["groups"].items():
            lines.append(f"| {g} | {r['n']} | " + " | ".join(fmt(r[m], m) for m in METRICS) + " |")
        if rep["overhead"]:
            o = rep["overhead"]
            dm = o.get("discriminator_ms_per_replan", {}).get("mean", float("nan"))
            wm = o.get("wm_forward_ms_per_replan", {}).get("mean", float("nan"))
            lines += ["", f"overhead: discriminator {dm:.1f} ms/replan (WM forward {wm:.1f} ms), "
                          f"{o['replans']} replans, WM source {o['wm_source']}, calls/replan {o['calls_per_replan']}"]
        lines.append("")
        per_scene = [{"tfrecord": k[0], "scenario_idx": k[1], "scenario_id": disc[k]["scenario_id"],
                      "disc_split": split.get(disc[k]["scenario_id"]),
                      **{f"learned_{m}": disc[k][m] for m in METRICS},
                      **{f"handcrafted_{m}": base[k][m] for m in METRICS}} for k in common]
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"per_scene_{name.replace('/', '_')}.json").write_text(json.dumps(per_scene, indent=1))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "closed_loop_comparison.json").write_text(json.dumps(report, indent=2))
    (out / "closed_loop_comparison.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
