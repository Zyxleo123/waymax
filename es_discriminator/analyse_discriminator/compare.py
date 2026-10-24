"""Ablation comparison against a baseline (A0 by default) on the identical held-out test replans.

Per run (a safety_progress output directory): ablation, trainable parameters, best epoch, and the
discriminator top-1 metrics from `eval_test.json` (unsafe, Waymax overlap = collision, offroad, goal
gain, route progress) plus pairwise ranking accuracy and safety AUROC, on the filtered test replans
and on the ES-unsafe scenes' first bad replan. Fails unless every run scored exactly the baseline's
test groups with its training settings. `--benchmark` adds latency; `--paired` loads the bank and
bootstraps each model minus the baseline per replan (same groups, scenario-level resampling).

  PYTHONPATH=. python -m es_discriminator.analyse_discriminator.compare \
      --run A1=<dir> --run A2=<dir> --run A3=<dir> --run A4=<dir> \
      --benchmark /zfsauton/scratch/sbellad/es_discriminator/analyse_discriminator/benchmark_cuda.json \
      --paired --out-dir /zfsauton/scratch/sbellad/es_discriminator/analyse_discriminator/compare

  # second round against A4 (A0 is then left out unless given with --run)
  ... compare --baseline A4 --run A4=<dir> --run A5=<dir> ... --run A8=<dir> --paired --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from es_discriminator.analyse_discriminator.ablations import ABLATIONS, DESCRIPTIONS, ablation_of
from es_discriminator.analyse_discriminator.verify_a0 import A0_CKPT
from es_discriminator.real_es.trainer import build_model

METRICS = (  # (key in eval json picker block, label, percent)
    ("unsafe", "top-1 unsafe %", True),
    ("overlap", "collision %", True),
    ("offroad", "offroad %", True),
    ("goal_gain", "goal gain m", False),
    ("progress", "progress m", False),
)
RANKING = (("pairwise_accuracy", "pairwise acc", False), ("safety_auroc", "safety AUROC", False))
SUBSETS = ("filtered", "es_unsafe_scenes_first_bad_replan")


def _fmt(b: dict | None, pct: bool) -> str:
    if not b or b.get("mean") is None:
        return "-"
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.3f}" if abs(x) < 2 else f"{x:.2f}")
    return f"{f(b['mean'])} [{f(b['lo'])}, {f(b['hi'])}]"


def _keys(run_dir: str, split: str) -> list[tuple[str, int]]:
    z = np.load(os.path.join(run_dir, f"{split}_predictions.npz"))
    return [(str(s), int(t)) for s, t in zip(z["scen_id"], z["group_t"])]


def load_run(name: str, run_dir: str, split: str) -> dict:
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=False)
    got = ablation_of(ck["model_kwargs"])
    if got != name:
        raise ValueError(f"{run_dir} holds {got}, not {name}")
    model = build_model(ck["model_kwargs"])
    return {
        "dir": run_dir,
        "best_epoch": json.load(open(os.path.join(run_dir, "result.json")))["best_epoch"],
        "trainable_parameters": int(sum(q.numel() for q in model.parameters() if q.requires_grad)),
        "eval": json.load(open(os.path.join(run_dir, f"eval_{split}.json"))),
        "args": ck["args"],
    }


def paired_deltas(runs: dict, split: str, n_boot: int, max_files: int = 0, base: str = "A0") -> dict:
    """Per replan (filtered test groups): model metric minus the baseline's, bootstrapped over scenarios.
    `max_files` (smoke tests) loads part of the bank and keeps only the groups found in it."""
    from es_discriminator.data import DataConfig
    from es_discriminator.real_es import evaluate as ev
    from es_discriminator.real_es.dataset import invalid_scene_keys, load_bank, training_mask

    a0 = runs[base]["args"]
    ck = torch.load(os.path.join(runs[base]["dir"], "best.pt"), map_location="cpu", weights_only=False)
    bank = load_bank(a0["data_dir"], max_files=max_files, cfg=DataConfig(**ck["data_config"]), verbose=False)
    gkey = {(s, int(t)): g for g, (s, t) in enumerate(zip(bank.groups["scen_id"], bank.groups["group_t"]))}
    clean, _ = training_mask(bank, invalid_scene_keys(a0["invalid_at_start"]))
    vals = {}
    for name, r in runs.items():
        z = np.load(os.path.join(r["dir"], f"{split}_predictions.npz"))
        found = np.array([(str(s), int(t)) in gkey for s, t in zip(z["scen_id"], z["group_t"])])
        if not max_files and not found.all():
            raise RuntimeError(f"{name}: {int((~found).sum())} {split} replans are missing from the bank")
        gidx = np.array([gkey[(str(s), int(t))] for s, t, f in zip(z["scen_id"], z["group_t"], found) if f],
                        dtype=np.int64)
        keep = clean[gidx]
        v = ev.per_group_values(bank, gidx[keep], z["scores"][found][keep], progress_key=a0["progress_key"],
                                margin=a0["progress_margin"])
        vals[name] = (gidx[keep], {**v["discriminator"], **v["ranking"]})
    g0, v0 = vals[base]
    scen = bank.groups["scen"][g0]
    out = {}
    for name, (g, v) in vals.items():
        if name == base:
            continue
        if not np.array_equal(g, g0):
            raise RuntimeError(f"{name} filtered test groups differ from {base}")
        out[name] = {m: ev.bootstrap(v[m] - v0[m], scen, n_boot=n_boot)
                     for m in (*(k for k, _, _ in METRICS), *(k for k, _, _ in RANKING))}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", default=[], metavar="NAME=DIR", help="ablation run directory")
    p.add_argument("--a0-dir", default=os.path.dirname(A0_CKPT))
    p.add_argument("--baseline", default="A0", help="ablation the others are compared with (needs its --run "
                                                    "unless A0)")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--benchmark", default="", help="benchmark.py JSON (latency, parameters)")
    p.add_argument("--latency-worlds", default="worlds_1")
    p.add_argument("--paired", action="store_true", help="load the bank and bootstrap model - baseline per replan")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--max-files", type=int, default=0, help="--paired on part of the bank (smoke tests only)")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()

    base = a.baseline
    dirs = {**({"A0": a.a0_dir} if base == "A0" else {}), **dict(r.split("=", 1) for r in a.run)}
    if base not in dirs:
        raise SystemExit(f"baseline {base} needs --run {base}=<dir>")
    runs = {n: load_run(n, d, a.split) for n, d in sorted(dirs.items()) if n in ABLATIONS}
    ref = _keys(runs[base]["dir"], a.split)
    for name, r in runs.items():
        if _keys(r["dir"], a.split) != ref:
            raise RuntimeError(f"{name} was evaluated on different {a.split} replans than {base}")
        for k in ("data_dir", "split", "invalid_at_start", "wm_dir", "edge_dir", "epochs", "batch_groups", "lr",
                  "weight_decay", "warmup_steps", "hidden", "heads", "dropout", "train_cands", "max_pairs",
                  "safety_frac", "progress_key", "progress_margin", "seed", "split_seed"):
            if r["args"].get(k) != runs[base]["args"].get(k):
                raise RuntimeError(f"{name} differs from {base} in training setting {k!r}: "
                                   f"{r['args'].get(k)!r} vs {runs[base]['args'].get(k)!r}")
    bench = json.load(open(a.benchmark)) if a.benchmark else None
    paired = paired_deltas(runs, a.split, a.n_boot, a.max_files, base=base) if a.paired else None

    L = [f"# Discriminator ablations: held-out {a.split} replans ({len(ref)} replans, identical for every model)", ""]
    lat_col = f" | latency ms ({a.latency_worlds}, {bench['device']})" if bench else ""
    for sub in SUBSETS:
        n = runs[base]["eval"][sub].get("n_groups", 0)
        L += [f"## {sub} ({n} replans)", "",
              "| model | change | params | best epoch" + lat_col + " | "
              + " | ".join(lbl for _, lbl, _ in (*METRICS, *RANKING)) + " |",
              "|---" * (4 + bool(bench) + len(METRICS) + len(RANKING)) + "|"]
        for name, r in runs.items():
            e = r["eval"][sub]
            row = [name, DESCRIPTIONS[name], f"{r['trainable_parameters']:,}", str(r["best_epoch"])]
            if bench:
                lat = bench["models"].get(name, {}).get("latency", {}).get(a.latency_worlds)
                row.append(f"{lat['total']['median_ms']:.1f}" if lat else "-")
            row += [_fmt(e["discriminator"][m], pct) for m, _, pct in METRICS]
            row += [_fmt(e["ranking"][m], pct) for m, _, pct in RANKING]
            L.append("| " + " | ".join(row) + " |")
        L.append("")
    if paired:
        L += [f"## Paired difference vs {base} (filtered replans, model minus {base}, 95% CI over scenarios)", "",
              "| model | " + " | ".join(lbl for _, lbl, _ in (*METRICS, *RANKING)) + " |",
              "|---" * (1 + len(METRICS) + len(RANKING)) + "|"]
        for name, d in paired.items():
            L.append(f"| {name} | " + " | ".join(_fmt(d[m], pct) for m, _, pct in (*METRICS, *RANKING)) + " |")
        L.append("")
    md = "\n".join(L)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"ablations_{a.split}.md"), "w") as f:
        f.write(md)
    summary = {name: {"dir": r["dir"], "best_epoch": r["best_epoch"], "trainable_parameters": r["trainable_parameters"],
                      **{sub: {**{m: r["eval"][sub]["discriminator"][m] for m, _, _ in METRICS},
                               **{m: r["eval"][sub]["ranking"][m] for m, _, _ in RANKING}} for sub in SUBSETS}}
               for name, r in runs.items()}
    with open(os.path.join(a.out_dir, f"ablations_{a.split}.json"), "w") as f:
        json.dump({"baseline": base, "runs": summary, f"paired_vs_{base}": paired, "benchmark": bench}, f, indent=1)
    print(md)


if __name__ == "__main__":
    main()
