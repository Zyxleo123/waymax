"""finetune_set: every valid hard WOMD training scene from the PDM-Closed screens (shards 00100-00181), defined
independently of the AnchorWM models: PDM-Closed overlap / offroad with the first failure after t=20 and a moving ego,
logged ego clean under log replay. Each scene is annotated with Diffusion-ES + PDMS outcomes only (seed 0 from the
earlier runs, seed 1 from the sanity run once it exists); tier 1 = ES+PDMS overlap / offroad.

Writes <OUT>/finetune_set.json ({tfrecord: [indices]}), finetune_set_tiers.json and finetune_set_tier1.json.
"""
import collections
import glob
import json
import os

M = "/data/user_data/mineuih/ishan/"
OUT = M + "mining/hard1500"


def results(pattern):
    return {(os.path.basename(r["tfrecord"]), int(r["scenario_idx"])): r
            for r in (json.load(open(f)) for f in glob.glob(pattern))}


def keys_of(path):
    return [(t, int(i)) for t, v in json.load(open(path)).items() for i in v]


def outcome(r):
    return "success" if r["success"] else "overlap" if r["overlap"] else "offroad" if r["offroad"] else "goal_miss"


def dump(name, keys):
    d = collections.defaultdict(list)
    for t, i in keys:
        d[t].append(i)
    json.dump({t: sorted(v) for t, v in sorted(d.items())}, open(f"{OUT}/{name}.json", "w"), indent=0)


bad = lambda r: bool(r["overlap"] or r["offroad"])  # noqa: E731
flagged = keys_of(M + "mining/pool_r0/flagged.json") + keys_of(M + "mining/pool_r2/flagged.json")
logrep = {**results(M + "mining/log_replay_flagged_r0/chunk_*/*.scenario_*.json"),
          **results(M + "mining/log_replay_flagged_r2/chunk_*/*.scenario_*.json")}
es0 = {**results(M + "mining/es_pdms_flagged_r0/chunk_*/*.scenario_*.json"),
       **results(M + "mining/es_pdms_hard1500_r2/chunk_*/*.scenario_*.json")}
es1 = results(M + "mining/es_pdms_finetune_set_seed1/chunk_*/*.scenario_*.json")
valid = sorted(k for k in flagged if k in logrep and not bad(logrep[k]))

tiers = {f"{t}:{i}": {"es_pdms_seed0": outcome(es0[(t, i)]) if (t, i) in es0 else None,
                      "es_pdms_seed1": outcome(es1[(t, i)]) if (t, i) in es1 else None} for t, i in valid}
tier1 = [k for k in valid if (k in es0 and bad(es0[k])) or (k in es1 and bad(es1[k]))]
dump("finetune_set", valid)
dump("finetune_set_tier1", tier1)
json.dump(tiers, open(f"{OUT}/finetune_set_tiers.json", "w"), indent=0)
both = [k for k in valid if k in es0 and k in es1]
print(f"finetune_set {len(valid)} scenes; ES+PDMS seed0 run {sum(k in es0 for k in valid)}, seed1 run "
      f"{sum(k in es1 for k in valid)}; tier1 {len(tier1)}")
if both:
    agree = sum(bad(es0[k]) == bad(es1[k]) for k in both)
    print(f"seed0 vs seed1 on {len(both)}: fail seed0 {sum(bad(es0[k]) for k in both)}, seed1 "
          f"{sum(bad(es1[k]) for k in both)}, both {sum(bad(es0[k]) and bad(es1[k]) for k in both)}, agreement {agree / len(both):.1%}")
