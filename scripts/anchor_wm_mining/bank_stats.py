"""Failure statistics of ES banks (name=es_candidates_dir args): unsafe candidates, mixed groups, focus groups."""
import os
import sys

import numpy as np

from anchor_wm.bank_data import _batch_files, _scene_failure_t


def stats(files):
    t = dict.fromkeys(("files", "scenes", "failed", "groups", "usable", "cands", "unsafe", "mixed", "pre_fail", "near_miss"), 0)
    for f in files:
        z = np.load(f)
        lw = np.load(f[:-4] + ".labels_waymax.npz")
        bad = lw["anchor_overlap"].astype(bool) | lw["anchor_offroad"].astype(bool)
        uns = (lw["gt_overlap_step"] > 0) | (lw["gt_offroad_step"] > 0)
        cg, g_t, g_s = z["cand_group"], z["group_t"].astype(int), z["group_scen"].astype(int)
        lead = _scene_failure_t(z, bad)[g_s] - g_t
        ok = z["group_valid"].astype(bool) & ~bad
        n = np.bincount(cg, minlength=len(g_t))
        nu = np.bincount(cg, weights=uns, minlength=len(g_t))
        sel = np.zeros(len(g_t), bool)
        sel[np.unique(cg[z["cand_selected"].astype(bool) & uns])] = True
        t["files"] += 1
        t["scenes"] += len(z["scen_index"])
        t["failed"] += int((~z["scen_success"].astype(bool)).sum())
        t["groups"] += len(g_t)
        t["usable"] += int(ok.sum())
        t["cands"] += int(n[ok].sum())
        t["unsafe"] += int(nu[ok].sum())
        t["mixed"] += int((ok & (nu > 0) & (nu < n)).sum())
        t["pre_fail"] += int((ok & (lead > 0) & (lead <= 30)).sum())
        t["near_miss"] += int((ok & sel).sum())
    g = t["usable"]
    return (f"{t['files']} files, {t['scenes']} scenes ({100 * t['failed'] / t['scenes']:.1f}% failed), {g} usable groups, "
            f"{t['cands'] / 1e6:.2f}M cands | unsafe cands {100 * t['unsafe'] / t['cands']:.1f}% | "
            f"mixed groups {100 * t['mixed'] / g:.1f}% | pre-failure 3 s {t['pre_fail']} ({100 * t['pre_fail'] / g:.2f}%) | "
            f"near-miss {t['near_miss']} ({100 * t['near_miss'] / g:.2f}%)")


for arg in sys.argv[1:]:
    name, d = arg.split("=", 1)
    files = [f for f in _batch_files(d) if os.path.exists(f[:-4] + ".labels_waymax.npz")]
    print(f"{name}: {stats(files)}", flush=True)
