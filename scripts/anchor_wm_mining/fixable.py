"""Could a perfect scorer have avoided the failure? For replans in the window before a closed-loop failure (and near
misses), does the ES pool hold a candidate with no Waymax overlap / offroad over its label horizon, and how good is
its progress compared with the selected (unsafe) plan."""
import sys

import numpy as np

from anchor_wm.bank_data import _batch_files, _scene_failure_t

rows = []
for f in _batch_files(sys.argv[1]):
    z = np.load(f)
    lw = np.load(f[:-4] + ".labels_waymax.npz")
    l1 = np.load(f[:-4] + ".labels.npz")
    bad = lw["anchor_overlap"].astype(bool) | lw["anchor_offroad"].astype(bool)
    ov, off = lw["gt_overlap_step"] > 0, lw["gt_offroad_step"] > 0
    uns = ov | off
    gain = l1["gt_goal_gain_m"]
    cg, g_t, g_s = z["cand_group"], z["group_t"].astype(int), z["group_scen"].astype(int)
    lead = _scene_failure_t(z, bad)[g_s] - g_t
    sel = z["cand_selected"].astype(bool)
    scen_kind = np.where(z["scen_overlap"].astype(bool), "overlap", np.where(z["scen_offroad"].astype(bool), "offroad", "ok"))
    for g in range(len(g_t)):
        if bad[g] or not z["group_valid"][g]:
            continue
        m = cg == g
        s = sel & m
        pre = 0 < lead[g] <= 30
        near = bool((s & uns).any())
        if not (pre or near):
            continue
        safe = m & ~uns
        sel_gain = float(gain[s][0]) if s.any() else np.nan
        best_safe_gain = float(np.nanmax(gain[safe])) if safe.any() else np.nan
        rows.append(dict(pre=pre, near=near, kind=scen_kind[g_s[g]], sel_unsafe=bool((s & uns).any()),
                         sel_off=bool((s & off).any()), any_safe=bool(safe.any()), frac_safe=float(safe.sum() / m.sum()),
                         safe_keeps_progress=bool(safe.any() and best_safe_gain >= sel_gain - 2.0)))

def report(name, rs):
    if not rs:
        print(name, "none"); return
    a = lambda k: np.mean([r[k] for r in rs])
    print(f"{name}: {len(rs)} groups | selected plan unsafe {a('sel_unsafe'):.0%} (offroad {a('sel_off'):.0%}) | "
          f"pool has a safe plan {a('any_safe'):.0%} | mean safe share of pool {a('frac_safe'):.0%} | "
          f"a safe plan within 2 m of the selected plan's progress {a('safe_keeps_progress'):.0%}")

report("pre-failure (3 s), all", [r for r in rows if r["pre"]])
report("pre-failure, offroad episodes", [r for r in rows if r["pre"] and r["kind"] == "offroad"])
report("pre-failure, overlap episodes", [r for r in rows if r["pre"] and r["kind"] == "overlap"])
report("near misses (selected unsafe)", [r for r in rows if r["near"]])
report("pre-failure with selected plan labelled SAFE", [r for r in rows if r["pre"] and not r["sel_unsafe"]])
