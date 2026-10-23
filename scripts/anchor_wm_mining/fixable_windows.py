"""Could a perfect scorer have avoided the closed-loop failure, and how early? For every replan of a failed episode
(ego pose still clean at the replan), bucketed by time before the first overlap / offroad, does the ES pool hold a
candidate with no Waymax overlap / offroad over its label horizon (safe), and one that also keeps the selected
plan's progress (goal gain within 2 m)?

  python fixable_windows.py <es_candidates dir> [...]
"""
import sys

import numpy as np

from anchor_wm.bank_data import _batch_files, _scene_failure_t

WINDOWS = ((0, 30), (30, 50), (50, 80))      # 0.1 s steps before the failure: 0-3 s, 3-5 s, 5-8 s


def rows_of(bank):
    rows = []
    for f in _batch_files(bank):
        z = np.load(f)
        lw = np.load(f[:-4] + ".labels_waymax.npz")
        l1 = np.load(f[:-4] + ".labels.npz")
        bad = lw["anchor_overlap"].astype(bool) | lw["anchor_offroad"].astype(bool)
        uns = (lw["gt_overlap_step"] > 0) | (lw["gt_offroad_step"] > 0)
        gain = l1["gt_goal_gain_m"]
        cg, g_t, g_s = z["cand_group"], z["group_t"].astype(int), z["group_scen"].astype(int)
        lead = _scene_failure_t(z, bad)[g_s] - g_t
        sel = z["cand_selected"].astype(bool)
        kind = np.where(z["scen_overlap"].astype(bool), "overlap", np.where(z["scen_offroad"].astype(bool), "offroad", "ok"))
        for g in range(len(g_t)):
            if bad[g] or not z["group_valid"][g] or kind[g_s[g]] == "ok" or lead[g] <= 0:
                continue
            m = cg == g
            s = sel & m
            safe = m & ~uns
            sel_gain = float(gain[s][0]) if s.any() else np.nan
            rows.append(dict(lead=int(lead[g]), kind=str(kind[g_s[g]]), sel_unsafe=bool((s & uns).any()),
                             any_safe=bool(safe.any()), frac_safe=float(safe.sum() / max(m.sum(), 1)),
                             safe_progress=bool(safe.any() and np.nanmax(gain[safe]) >= sel_gain - 2.0)))
    return rows


for bank in sys.argv[1:]:
    rows = rows_of(bank)
    print(f"\n== {bank.split('/')[-2]}: {len(rows)} replans of failed episodes")
    print(f"{'window':>9} {'kind':>8} {'n':>5} {'sel unsafe':>11} {'pool has safe':>14} {'safe share':>11} {'safe+progress':>14}")
    for lo, hi in WINDOWS:
        for kind in ("all", "offroad", "overlap"):
            rs = [r for r in rows if lo < r["lead"] <= hi and (kind == "all" or r["kind"] == kind)]
            if not rs:
                continue
            a = lambda k: 100 * np.mean([r[k] for r in rs])  # noqa: E731
            print(f"{lo / 10:.0f}-{hi / 10:.0f} s {kind:>8} {len(rs):5d} {a('sel_unsafe'):10.0f}% {a('any_safe'):13.0f}% "
                  f"{a('frac_safe'):10.0f}% {a('safe_progress'):13.0f}%")
