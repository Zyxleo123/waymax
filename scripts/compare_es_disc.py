"""Paired comparison of the Diffusion-ES arms under each of es_WM's evaluators.

Layout (slurm/sim/submit_es_disc.sh): ``<root>/<run view>/<arm>/<group>/<tfrecord>.scenario_<idx>.json``
with run view ``log`` (NPCs replay the log), ``idm`` (es_WM's closed-loop IDM: the planner
sees the simulated NPCs, simulation/idm_closed_loop.py) or ``idm_brake`` (the same loop with
IDM capped at each NPC's logged speed, braking for the ego only, simulation/sim_agents.py).
Each run view is its own planning run, so views are compared arm by arm, never scene-paired.

Reported views (all over the scenes EVERY arm of that run view finished):
  log / idm                failures counted over [t=10, goal step] (Ishan's 4219b45 window)
  log_oldwin / idm_oldwin  es_WM's original window (metric rows [10:] -> t=20..goal+20)
  log_idm_evalonly         the log run's executed ego re-scored with evaluation-only IDM NPCs
                           (the previous protocol's "idm" view; not a closed loop)
  idm_brake / idm_brake_oldwin, log_idm_brake_evalonly   the same, with idm_brake NPCs
Per arm: success, collision, off-road, goal-not-reached rates; per pair (b:a) the paired
success difference with a scene-bootstrap 95% CI and the discordant counts.

``--split_json`` (es_discriminator's scenario-hash split, ``{"assignment": {scenario_id: label}}``)
additionally slices the log and idm views by the label each scene had when the discriminator
was trained (train / val / test; "unseen" = not in its training set), so an in-sample gain
can be told apart from a held-out one.

    python scripts/compare_es_disc.py --root <run root> --arms diff,cv,wm_mf,wm_mf_disc \
        --views log,idm --pairs wm_mf_disc:wm_mf --out <root>/compare     # -> .json + .md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

NAME_RE = re.compile(r"^(?P<tf>.+)\.scenario_(?P<idx>\d+)\.json$")
KEYS = ("success", "offroad", "overlap", "goal_reached")
# report view -> (run view directory, {metric: scene-JSON key})
VIEWS = {
    "log": ("log", {k: k for k in KEYS}),
    "log_oldwin": ("log", {"success": "success_oldwin", "offroad": "offroad_oldwin",
                           "overlap": "overlap_oldwin", "goal_reached": "goal_reached"}),
    "log_idm_evalonly": ("log", {"success": "idm_success", "offroad": "idm_offroad",
                                 "overlap": "idm_overlap", "goal_reached": "goal_reached"}),
    "idm": ("idm", {k: k for k in KEYS}),
    "idm_oldwin": ("idm", {"success": "success_oldwin", "offroad": "offroad_oldwin",
                           "overlap": "overlap_oldwin", "goal_reached": "goal_reached"}),
    "log_idm_brake_evalonly": ("log", {"success": "idm_brake_success", "offroad": "idm_brake_offroad",
                                       "overlap": "idm_brake_overlap", "goal_reached": "goal_reached"}),
    "idm_brake": ("idm_brake", {k: k for k in KEYS}),
    "idm_brake_oldwin": ("idm_brake", {"success": "success_oldwin", "offroad": "offroad_oldwin",
                                       "overlap": "overlap_oldwin", "goal_reached": "goal_reached"}),
}
TITLES = {"log": "log replay", "log_oldwin": "log replay, es_WM original window",
          "log_idm_evalonly": "log run re-scored with evaluation-only IDM (old protocol)",
          "idm": "closed-loop IDM (es_WM evaluator)", "idm_oldwin": "closed-loop IDM, es_WM original window",
          "log_idm_brake_evalonly": "log run re-scored with evaluation-only idm_brake NPCs",
          "idm_brake": "closed-loop idm_brake (NPCs capped at logged speed, brake for ego)",
          "idm_brake_oldwin": "closed-loop idm_brake, es_WM original window"}
SPLIT_VIEWS = ("log", "idm", "idm_brake")


def load_arm(d: Path, expected_agents: str) -> dict:
    """{(tfrecord, idx): scene JSON} for one arm of one run view; checks the run view matches."""
    rows = {}
    for f in d.rglob("*.json"):
        m = NAME_RE.match(f.name)
        if m is None:
            continue
        r = json.loads(f.read_text())
        got = r.get("other_agents", "log")
        if got != expected_agents:
            raise SystemExit(f"{f}: other_agents={got!r} but it sits under the {expected_agents!r} view")
        rows[(m["tf"], int(m["idx"]))] = r
    return rows


def summarize(mat: dict, arms: list, pairs: list, n_boot: int, tag: str):
    """Rates per arm and paired success deltas over the scenes of ``mat`` -> (json, markdown lines)."""
    n = len(next(iter(mat.values()))["success"])
    out = {"n": n, "arms": {}, "pairs": {}}
    md = ["| arm | success | collision | off-road | goal not reached |", "|---|---|---|---|---|"]
    for arm in arms:
        if arm not in mat:
            continue
        c = {"success": int(mat[arm]["success"].sum()), "collision": int(mat[arm]["overlap"].sum()),
             "offroad": int(mat[arm]["offroad"].sum()), "goal_not_reached": int((~mat[arm]["goal_reached"]).sum())}
        out["arms"][arm] = {"counts": c, "rates": {k: v / n for k, v in c.items()}}
        print(f"[compare] {tag} {arm:20s} " + " ".join(f"{k} {v:5d} ({100 * v / n:5.1f}%)" for k, v in c.items()))
        md.append(f"| {arm} | " + " | ".join(f"{100 * v / n:.2f}%" for v in c.values()) + " |")
    idx = np.random.default_rng(0).integers(0, n, size=(n_boot, n))
    rows = []
    for pair in pairs:
        b, ref = pair.split(":")
        if b not in mat or ref not in mat:
            continue
        sb, sa = mat[b]["success"].astype(float), mat[ref]["success"].astype(float)
        d = sb - sa
        boot = d[idx].mean(axis=1)
        o = {"delta_success": float(d.mean()),
             "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
             "b_better": int(((sb == 1) & (sa == 0)).sum()), "a_better": int(((sb == 0) & (sa == 1)).sum())}
        for k in ("offroad", "overlap", "goal_reached"):
            o[f"delta_{k}"] = float(mat[b][k].mean() - mat[ref][k].mean())
        out["pairs"][f"{b}_vs_{ref}"] = o
        print(f"[compare] {tag} {b} vs {ref}: success {o['delta_success']:+.4f} "
              f"[{o['ci95'][0]:+.4f}, {o['ci95'][1]:+.4f}]  b>a {o['b_better']} a>b {o['a_better']}")
        rows.append(f"| {b} vs {ref} | {100 * o['delta_success']:+.2f} [{100 * o['ci95'][0]:+.2f}, "
                    f"{100 * o['ci95'][1]:+.2f}] | {o['b_better']} / {o['a_better']} | {100 * o['delta_overlap']:+.2f} | "
                    f"{100 * o['delta_offroad']:+.2f} | {-100 * o['delta_goal_reached']:+.2f} |")
    if rows:
        md += ["", "| pair (b vs a) | Δ success, pts [95% CI] | b better / a better | Δ collision | Δ off-road "
               "| Δ goal not reached |", "|---|---|---|---|---|---|"] + rows
    return out, md


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--arms", required=True)
    p.add_argument("--views", default="log,idm", help="run views to read (directories under --root)")
    p.add_argument("--pairs", default="")
    p.add_argument("--expect", type=int, default=0, help="scenes each arm should have (0 = no check)")
    p.add_argument("--split_json", default="", help="es_discriminator scenario split (optional)")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--out", required=True, help="output stem: writes <out>.json and <out>.md")
    a = p.parse_args()

    arms = [x for x in a.arms.split(",") if x]
    pairs = [x for x in a.pairs.split(",") if x]
    run_views = [x for x in a.views.split(",") if x]
    split = json.loads(Path(a.split_json).read_text())["assignment"] if a.split_json else None
    raw = {rv: {arm: load_arm(Path(a.root) / rv / arm, rv) for arm in arms} for rv in run_views}
    rep = {"root": a.root, "arms": arms, "expect": a.expect, "split_json": a.split_json or None, "views": {}}
    md = [f"# ES arms under es_WM's evaluators\n\nroot `{a.root}`\n"]
    for view, (rv, keys) in VIEWS.items():
        if rv not in raw:
            continue
        # an arm counts for a view only if every one of its scenes carries that view's keys
        have = {arm: rows for arm, rows in raw[rv].items()
                if rows and all(all(k in r for k in keys.values()) for r in rows.values())}
        if not have:
            continue
        counts = {arm: len(rows) for arm, rows in have.items()}
        common = sorted(set.intersection(*(set(rows) for rows in have.values())))
        rv_out = {"run_view": rv, "title": TITLES[view], "counts": counts, "n_common": len(common),
                  "complete": bool(a.expect) and all(c == a.expect for c in counts.values()),
                  "missing_arms": [x for x in arms if x not in have]}
        rep["views"][view] = rv_out
        print(f"[compare] ===== {view}: {TITLES[view]}; scenes per arm {counts}; common {len(common)} =====")
        if not common:
            continue
        mat = {arm: {k: np.array([bool(have[arm][s][src]) for s in common]) for k, src in keys.items()}
               for arm in have}
        summary, lines = summarize(mat, arms, pairs, a.n_boot, view)
        rv_out.update(summary)
        md += [f"\n## {TITLES[view]} (`{view}`), {len(common)} scenes common to {len(have)} arms\n"] + lines
        if split is None or view not in SPLIT_VIEWS:
            continue
        any_arm = next(iter(have.values()))
        label = np.array([split.get(any_arm[s].get("scenario_id") or "", "unseen") for s in common])
        if all(any_arm[s].get("scenario_id") is None for s in common):
            print(f"[compare] {view}: scene JSONs carry no scenario_id; no split slices")
            continue
        rv_out["by_disc_split"] = {}
        for lab in sorted(set(label)):
            m = label == lab
            sub = {arm: {k: v[m] for k, v in d.items()} for arm, d in mat.items()}
            s_out, s_lines = summarize(sub, arms, pairs, a.n_boot, f"{view}@{lab}")
            rv_out["by_disc_split"][lab] = s_out
            md += [f"\n### `{view}`, scenes the discriminator had in its **{lab}** split ({int(m.sum())})\n"] + s_lines
    Path(a.out + ".json").write_text(json.dumps(rep, indent=1) + "\n")
    Path(a.out + ".md").write_text("\n".join(md) + "\n")
    print(f"[compare] -> {a.out}.json, {a.out}.md")


if __name__ == "__main__":
    main()
