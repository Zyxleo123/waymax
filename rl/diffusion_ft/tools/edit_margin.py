"""What margin did each repair arm actually leave, and how big were its edits?

WHY. The loss-optimized repair (``o_0``, rl/diffusion_ft/lateral_opt.py) beat the
ramp-20 bank (``b_l24``) by +0.110 on failure_heldout, but two things differ at
once: the SHAPE of the edit (solved per replan vs one fixed curve) and the
MARGIN it aims for (the loss pushes clearance to 0.3 m edge / 0.75 m agent; the
bank executes its SMALLEST candidate that is clean at 0). This reads the saved
per-replan features and reports, at every replan where an arm switched:

* the chosen candidate's predicted clearance (``clear_a`` / ``edge_a`` /
  ``agent_a`` over the scored window) -- p10/p50/p90, and the fraction that
  clears both loss margins;
* for bank arms with optimized candidates, the share of switches that executed an
  optimized edit, and that edit's peak offset (``opt_k``) -- including the share
  beyond the bank's +/-1.6 m range, which a rigid bank cannot reach at all;
* for fixed candidates, the executed amplitude |A|.

If the optimized arms switch into much larger clearances than the bank, margin is
a live explanation; if they also routinely exceed 1.6 m, range is. The paired
2x2 (submit_lateral_opt_attrib.sh) settles it; this is the free first look.

Pure re-read of ``<select_dir>/tasks/*/feat/*.npz`` + meta; CPU, no GPU::

    python -m rl.diffusion_ft.tools.edit_margin --select_dir <dir> [--split failure_heldout]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

# FEAT_COLS in proposal_select: clear_p edge_p agent_p prog_p clear_a edge_a agent_a prog_a
CLEAR_A, EDGE_A, AGENT_A = 4, 5, 6
NO_AGENT_M = 1.0e3


def _pct(v, q):
    return float(np.percentile(v, q)) if len(v) else None


def summarize(select_dir: Path, split: str, edge_m: float, agent_m: float) -> dict:
    metas = sorted(select_dir.glob("tasks/*/meta_shard*.json"))
    if not metas:
        raise SystemExit(f"no tasks/*/meta_shard*.json under {select_dir}")
    meta = json.loads(metas[0].read_text())
    if list(meta.get("feat_cols", []))[CLEAR_A:AGENT_A + 1] != ["clear_a", "edge_a", "agent_a"]:
        raise SystemExit(f"feat_cols changed: {meta.get('feat_cols')}")
    arm_bank = {a["name"]: a.get("bank", "") for a in meta["arms"] if a.get("alt") == "bank"}
    names = meta.get("banks", {})
    acc = {a: {"clear": [], "edge": [], "agent": [], "opt": [], "opt_peak": [], "amp": []}
           for a in arm_bank}
    files = sorted(glob.glob(str(select_dir / "tasks" / "*" / "feat" / f"{split}_s*.npz")))
    if not files:
        raise SystemExit(f"no feat/{split}_s*.npz under {select_dir}/tasks")
    for f in files:
        z = np.load(f)
        for arm, bank in arm_bank.items():
            cand = names.get(bank, [])
            n_fixed = sum(1 for c in cand if not c.startswith("opt:"))
            reps = sorted({k.split("/")[1] for k in z.files if k.startswith(f"{arm}/")})
            for r in reps:
                feat, sw = z[f"{arm}/{r}/feat"], z[f"{arm}/{r}/switch"].astype(bool)
                if not sw.any():
                    continue
                a = acc[arm]
                a["clear"] += feat[..., CLEAR_A][sw].tolist()
                a["edge"] += feat[..., EDGE_A][sw].tolist()
                a["agent"] += feat[..., AGENT_A][sw].tolist()
                ch = z[f"{arm}/{r}/choice"].astype(int)
                is_opt = sw & (ch >= n_fixed)
                a["opt"] += is_opt[sw].tolist()
                if is_opt.any() and f"{arm}/{r}/opt_k" in z.files:
                    ok = z[f"{arm}/{r}/opt_k"]                          # [B, R, K_opt, 3]
                    b, t = np.nonzero(is_opt)
                    a["opt_peak"] += ok[b, t, ch[b, t] - n_fixed, 2].tolist()
                fixed = sw & (ch < n_fixed)
                if fixed.any():
                    amp = np.array([abs(float(c[3:])) if c.startswith("lat") else np.nan
                                    for c in cand[:n_fixed]])
                    a["amp"] += amp[ch[fixed]].tolist()
    out = {"select_dir": str(select_dir), "split": split,
           "margins": {"edge_m": edge_m, "agent_m": agent_m}, "arms": {}}
    for arm, a in acc.items():
        edge, agent = np.asarray(a["edge"]), np.asarray(a["agent"])
        agent_seen = agent < NO_AGENT_M
        peak = np.asarray(a["opt_peak"])
        out["arms"][arm] = {
            "n_switch": len(a["clear"]),
            "clear_a": {q: _pct(a["clear"], q) for q in (10, 50, 90)},
            "edge_a": {q: _pct(a["edge"], q) for q in (10, 50, 90)},
            "agent_a_with_agent": {q: _pct(agent[agent_seen], q) for q in (10, 50, 90)},
            "frac_clears_both_margins": (float(np.mean((edge >= edge_m) & (agent >= agent_m)))
                                         if len(edge) else None),
            "frac_edge_below_0.1": float(np.mean(edge < 0.1)) if len(edge) else None,
            "frac_switch_to_opt": float(np.mean(a["opt"])) if a["opt"] else None,
            "opt_peak_d": {q: _pct(peak, q) for q in (10, 50, 90)},
            "opt_frac_beyond_1.6m": float(np.mean(peak > 1.6)) if len(peak) else None,
            "fixed_amp": {q: _pct(a["amp"], q) for q in (10, 50, 90)},
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select_dir", type=Path, required=True)
    ap.add_argument("--split", default="failure_heldout")
    ap.add_argument("--edge_m", type=float, default=0.3, help="the loss's edge margin")
    ap.add_argument("--agent_m", type=float, default=0.75, help="the loss's agent margin")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    s = summarize(args.select_dir, args.split, args.edge_m, args.agent_m)
    f2 = lambda v: "  -  " if v is None else f"{v:5.2f}"
    print(f"### {args.split}: the chosen candidate at every switch "
          f"(margins {args.edge_m} edge / {args.agent_m} agent)")
    print(f"{'arm':>8} {'n':>5} | {'clear p10/50/90':>17} | {'edge p10/50/90':>17} | "
          f"{'both margins':>12} {'edge<0.1':>8} | {'->opt':>5} {'opt |d| p50/p90':>15} "
          f"{'>1.6m':>5} | {'fixed |A| p50/p90':>17}")
    for arm, v in s["arms"].items():
        c, e, p, fa = v["clear_a"], v["edge_a"], v["opt_peak_d"], v["fixed_amp"]
        print(f"{arm:>8} {v['n_switch']:>5} | {f2(c[10])}/{f2(c[50])}/{f2(c[90])} | "
              f"{f2(e[10])}/{f2(e[50])}/{f2(e[90])} | {f2(v['frac_clears_both_margins']):>12} "
              f"{f2(v['frac_edge_below_0.1']):>8} | {f2(v['frac_switch_to_opt'])} "
              f"{f2(p[50])}/{f2(p[90]):>7} {f2(v['opt_frac_beyond_1.6m'])} | "
              f"{f2(fa[50])}/{f2(fa[90])}")
    if args.json:
        args.json.write_text(json.dumps(s, indent=1))
        print(f"-> {args.json}")


if __name__ == "__main__":
    main()
