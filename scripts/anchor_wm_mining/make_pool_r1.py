"""Round-1 mining pool: revisit bank-0 scenes with a high failure score (PLR-style) + fresh random moving scenes."""
import collections
import glob
import json
import os
import random

import numpy as np

from anchor_wm.bank_data import _batch_files

M = "/data/user_data/mineuih/ishan/"
BANK0 = M + "anchorwm_bank/dagger_r0_A2_pool/es_candidates"
OUT = M + "mining/pool_r1"
os.makedirs(OUT, exist_ok=True)

score = collections.Counter()
for f in _batch_files(BANK0):
    z = np.load(f)
    lw = np.load(f[:-4] + ".labels_waymax.npz")
    uns = (lw["gt_overlap_step"] > 0) | (lw["gt_offroad_step"] > 0)
    keys = [(t, int(i)) for t, i in zip(z["scen_tfrecord"].astype(str), z["scen_index"])]
    for s, k in enumerate(keys):
        if not bool(z["scen_success"][s]):
            score[k] += 3.0                                        # closed-loop failure
    sel = z["cand_selected"].astype(bool) & uns                    # near miss: chosen plan labelled unsafe
    for g in np.unique(z["cand_group"][sel]):
        score[keys[int(z["group_scen"][g])]] += 1.0
es = {(os.path.basename(r["tfrecord"]), int(r["scenario_idx"])): r for r in
      (json.load(open(f)) for f in glob.glob(M + "mining/es_pdms_flagged_r0/chunk_*/*.scenario_*.json"))}
for k, r in es.items():
    if r["overlap"] or r["offroad"]:
        score[k] += 2.0                                            # an independent ES also fails it
lr = {(os.path.basename(r["tfrecord"]), int(r["scenario_idx"])): r for r in
      (json.load(open(f)) for f in glob.glob(M + "mining/log_replay_flagged_r0/chunk_*/*.scenario_*.json"))}
invalid = {k for k, r in lr.items() if r["overlap"] or r["offroad"]}
revisit = [k for k, v in score.most_common() if v > 0 and k not in invalid]

pool0 = json.load(open(M + "mining/pool_r0/pool.json"))
used = {(t, i) for t, v in pool0.items() for i in v}
calm = []
for f in glob.glob(M + "mining/pdm_screen_train_r10/*/*.scenario_*.json"):
    r = json.load(open(f))
    k = (os.path.basename(r["tfrecord"]), int(r["scenario_idx"]))
    if r["success"] and r.get("reached_timestep") != 10 and k not in used:
        calm.append(k)
random.Random(1).shuffle(calm)
fresh = calm[:250]

def dump(name, keys):
    d = collections.defaultdict(list)
    for t, i in keys:
        d[t].append(i)
    json.dump({t: sorted(v) for t, v in sorted(d.items())}, open(f"{OUT}/{name}.json", "w"), indent=0)

dump("revisit", revisit)
dump("fresh_random", fresh)
dump("pool", revisit + fresh)
print("score histogram:", sorted(collections.Counter(min(int(v), 6) for v in score.values()).items()))
print(f"revisit {len(revisit)} (closed-loop A2 failures {sum(score[k] >= 3 for k in revisit)}), fresh random {len(fresh)}, "
      f"pool {len(revisit) + len(fresh)}, dropped invalid {len(invalid & set(score))}")
