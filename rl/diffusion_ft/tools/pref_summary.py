"""One table for the Diffusion-DPO run (submit_pref_dpo.sh). CPU, re-reads artifacts.

Inputs under ``--run``:
  eval_<arm>/compare_it<NNN>_vs_{base,bank,bcd,bcsw50}.json   (compare_select)
  train/dpo_<arm>_s<seed>/train_log.jsonl                     (train_safety)

Closed-loop table, per arm x scored iteration (seed means, select_eval draws,
paired per (scene, repeat)):
  failure_heldout   solved / reached / overlap (collision) / offroad
  success_heldout   solved, and the share of scenes made worse (= broken)
  failure scenes fixed (share made better), intermittent solved
  m3 (corpus) delta vs base, vs the bank teacher, vs repaired-trajectory BC
  (bcd = plain, bcsw50 = half-edit batches), each with its 95% CI

Training curves, per arm (seed mean of the 1-repeat training eval, seed 12345):
  failure/success heldout solved, marginal delta, and the DPO diagnostics
  (z, accuracy, the four denoising losses, both gradient norms).
  `curve_pick` applies the plan's rule -- most failure gain with the success
  split within --success_budget of base -- ON THE HELD-OUT CURVE, so it is an
  optimistic number; the scored iterations above were fixed in advance.

    python -m rl.diffusion_ft.tools.pref_summary --run DIR
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

CURVE_KEYS = ("failure_heldout/solved_rate", "success_heldout/solved_rate",
              "marginal/delta_vs_base", "pref/loss", "pref/logit_z", "pref/z_abs",
              "pref/accuracy", "pref/good_model_loss", "pref/good_ref_loss",
              "pref/bad_model_loss", "pref/bad_ref_loss", "pref/grad_norm",
              "pref/anchor_grad_norm", "pref/anchor_loss")


def _ci(r):
    c = r.get("ci95") or [float("nan")] * 2
    return f"{r['delta']:+.4f} [{c[0]:+.4f},{c[1]:+.4f}]"


def closed_loop(run: Path):
    rows = []
    for d in sorted(run.glob("eval_*")):
        arm = d.name[len("eval_"):]
        its = sorted({m.group(1) for f in d.glob("compare_it*_vs_*.json")
                      if (m := re.match(r"compare_it(\d+)_vs_", f.name))})
        for it in its:
            c = {}
            for ref in ("base", "bank", "bcd", "bcsw50"):
                f = d / f"compare_it{it}_vs_{ref}.json"
                if f.exists():
                    c[ref] = json.loads(f.read_text())
            if "base" not in c:
                continue
            b = c["base"]
            s = b["splits"]
            row = {"arm": arm, "iter": int(it), "n_ckpts": sum(
                1 for k in b["per_ckpt"] if k.startswith(b["labels"][1] + ":"))}
            la, lb = b["labels"]
            for sp in ("failure_heldout", "success_heldout", "intermittent_heldout"):
                if sp not in s:
                    continue
                x = s[sp]
                row[sp] = {m: x[f"{lb}_{m}"] for m in ("solved", "reached", "overlap", "offroad")}
                row[sp]["base_solved"] = x[f"{la}_solved"]
                n = max(x["n_scenes"], 1)
                row[sp]["scenes_better"] = x["scenes_b_better"] / n
                row[sp]["scenes_worse"] = x["scenes_a_better"] / n
            for ref, r in c.items():
                if "marginal3" in r:
                    row[f"m3_vs_{ref}"] = r["marginal3"]
            rows.append(row)
    return rows


def curves(run: Path, budget: float):
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in sorted(glob.glob(str(run / "train" / "dpo_*_s*" / "train_log.jsonl"))):
        arm = re.match(r"dpo_(.+)_s\d+$", Path(f).parent.name).group(1)
        for line in Path(f).read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k in CURVE_KEYS + ("base_success_heldout/solved_rate",
                                   "base_failure_heldout/solved_rate"):
                if k in r:
                    by[arm][int(r["iter"])][k].append(float(r[k]))
    out = {}
    for arm, its in by.items():
        tab = {it: {k: float(np.mean(v)) for k, v in d.items()} | {"n_seeds": len(
            d.get("pref/loss", []))} for it, d in sorted(its.items())}
        ev = [(it, d) for it, d in tab.items() if "failure_heldout/solved_rate" in d
              and "base_success_heldout/solved_rate" in d]
        ok = [(it, d) for it, d in ev if d["success_heldout/solved_rate"]
              >= d["base_success_heldout/solved_rate"] - budget]
        pick = max(ok, key=lambda x: x[1]["failure_heldout/solved_rate"]) if ok else None
        out[arm] = {"curve": tab, "curve_pick": (
            {"iter": pick[0], "budget": budget,
             **{k: pick[1].get(k) for k in ("failure_heldout/solved_rate",
                                            "success_heldout/solved_rate",
                                            "marginal/delta_vs_base")}} if pick else None)}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--success_budget", type=float, default=0.005,
                   help="largest success_heldout drop the curve pick may spend")
    a = p.parse_args()
    run = Path(a.run)
    cl, cv = closed_loop(run), curves(run, a.success_budget)
    res = {"closed_loop": cl, "curves": cv}
    (run / "summary.json").write_text(json.dumps(res, indent=1))

    L = ["# Diffusion-DPO summary", "", f"run `{run}`", "",
         "## Closed loop (select_eval, seed 4242, 4 repeats; seed means, paired)", "",
         "| arm | iter | fail solved (base) | fail reached | fail overlap | fail offroad "
         "| fail fixed | succ solved (base) | succ broken | interm | m3 vs base | vs bank "
         "| vs bcd | vs bcsw50 |", "|" + "---|" * 14]
    for r in cl:
        f, s = r.get("failure_heldout", {}), r.get("success_heldout", {})
        i = r.get("intermittent_heldout", {})
        L.append("| " + " | ".join([
            r["arm"], str(r["iter"]),
            f"{f.get('solved', float('nan')):.3f} ({f.get('base_solved', float('nan')):.3f})",
            f"{f.get('reached', float('nan')):.3f}", f"{f.get('overlap', float('nan')):.3f}",
            f"{f.get('offroad', float('nan')):.3f}", f"{f.get('scenes_better', float('nan')):.3f}",
            f"{s.get('solved', float('nan')):.4f} ({s.get('base_solved', float('nan')):.4f})",
            f"{s.get('scenes_worse', float('nan')):.4f}", f"{i.get('solved', float('nan')):.3f}",
            *[_ci(r[k]) if k in r else "-" for k in
              ("m3_vs_base", "m3_vs_bank", "m3_vs_bcd", "m3_vs_bcsw50")]]) + " |")
    L += ["", "## Training curves (seed mean, 1-repeat training eval)", ""]
    for arm, d in sorted(cv.items()):
        L += [f"### {arm}", "", "| iter | fail | succ | margD | loss | z | acc | good m-r "
              "| bad m-r | grad pref / anchor |", "|" + "---|" * 10]
        for it, x in d["curve"].items():
            if "failure_heldout/solved_rate" not in x:
                continue
            g = x.get("pref/good_model_loss", np.nan) - x.get("pref/good_ref_loss", np.nan)
            b = x.get("pref/bad_model_loss", np.nan) - x.get("pref/bad_ref_loss", np.nan)
            L.append(f"| {it} | {x['failure_heldout/solved_rate']:.3f} | "
                     f"{x.get('success_heldout/solved_rate', np.nan):.4f} | "
                     f"{x.get('marginal/delta_vs_base', np.nan):+.4f} | "
                     f"{x.get('pref/loss', np.nan):.4f} | {x.get('pref/logit_z', np.nan):+.2e} | "
                     f"{x.get('pref/accuracy', np.nan):.2f} | {g:+.2e} | {b:+.2e} | "
                     f"{x.get('pref/grad_norm', np.nan):.1e} / "
                     f"{x.get('pref/anchor_grad_norm', np.nan):.1e} |")
        if d["curve_pick"]:
            cp = d["curve_pick"]
            L += ["", f"curve pick (optimistic, held-out curve, success within "
                      f"{cp['budget']}): iter {cp['iter']}, fail "
                      f"{cp['failure_heldout/solved_rate']:.3f}, succ "
                      f"{cp['success_heldout/solved_rate']:.4f}, margD "
                      f"{cp['marginal/delta_vs_base']:+.4f}"]
        L.append("")
    (run / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"[pref_summary] wrote {run / 'summary.md'} and summary.json")


if __name__ == "__main__":
    main()
