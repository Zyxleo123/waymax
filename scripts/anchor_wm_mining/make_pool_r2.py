"""Bank-2 mining pool, in two stages.

  flagged   screen-2 shards (00141-00181): PDM-Closed overlap / offroad after t=20, ego moving -> pool_r2/flagged.json
  pool      after log replay of those: valid flagged + a replay slice (bank-1 scenes M1 failed or nearly failed)
            + fresh random moving scenes from the same shards -> pool_r2/pool.json
"""
import collections
import glob
import json
import os
import random
import sys

M = "/data/user_data/mineuih/ishan/"
SCREEN = M + "mining/pdm_screen_train_r10/"
OUT = M + "mining/pool_r2"
SHARDS = range(141, 182)
os.makedirs(OUT, exist_ok=True)


def key(r):
    return os.path.basename(r["tfrecord"]), int(r["scenario_idx"])


def dump(name, keys):
    d = collections.defaultdict(list)
    for t, i in keys:
        d[t].append(i)
    json.dump({t: sorted(v) for t, v in sorted(d.items())}, open(f"{OUT}/{name}.json", "w"), indent=0)
    print(f"{name}: {len(keys)} scenes in {len(d)} tfrecords")


screen = [json.load(open(f)) for s in SHARDS for f in glob.glob(SCREEN + "%05d-of-01000/*.scenario_*.json" % s)]
moving = [r for r in screen if r.get("reached_timestep") != 10]

if sys.argv[1] == "flagged":
    flagged = []
    for r in moving:
        steps = [t for t in (r.get("overlap_timestep", -1), r.get("offroad_timestep", -1)) if t is not None and t >= 0]
        if (r["overlap"] or r["offroad"]) and steps and min(steps) > 20:
            flagged.append(key(r))
    dump("flagged", flagged)
    sys.exit()

lr = {key(r): r for r in (json.load(open(f)) for f in glob.glob(M + "mining/log_replay_flagged_r2/chunk_*/*.scenario_*.json"))}
flagged = json.load(open(f"{OUT}/flagged.json"))
flagged = [(t, i) for t, v in flagged.items() for i in v]
valid = [k for k in flagged if k in lr and not (lr[k]["overlap"] or lr[k]["offroad"])]
print(f"flagged {len(flagged)}, log-replayed {sum(k in lr for k in flagged)}, valid {len(valid)}")

# Replay slice: bank-1 scenes where M1 failed, then scenes with the most near-miss replans under M1.
import numpy as np  # noqa: E402

from anchor_wm.bank_data import _batch_files  # noqa: E402

score = collections.Counter()
for f in _batch_files(M + "anchorwm_bank/dagger_r1_M1_pool/es_candidates"):
    z = np.load(f)
    lw = np.load(f[:-4] + ".labels_waymax.npz")
    uns = (lw["gt_overlap_step"] > 0) | (lw["gt_offroad_step"] > 0)
    keys = [(t, int(i)) for t, i in zip(z["scen_tfrecord"].astype(str), z["scen_index"])]
    for s, k in enumerate(keys):
        score[k] += 0.0 if bool(z["scen_success"][s]) else 3.0
    for g in np.unique(z["cand_group"][z["cand_selected"].astype(bool) & uns]):
        score[keys[int(z["group_scen"][g])]] += 1.0
replay = [k for k, v in score.most_common(50) if v > 0]

fresh_pool = [key(r) for r in moving if r["success"] and key(r) not in set(flagged)]
random.Random(2).shuffle(fresh_pool)
fresh = fresh_pool[:150]
dump("replay", replay)
dump("fresh_random", fresh)
dump("pool", valid + replay + fresh)
