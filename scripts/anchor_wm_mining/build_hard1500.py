"""hard1500: valid hard WOMD training scenes from the PDM-Closed screens (shards 00100-00181).

Candidates: PDM-Closed overlap / offroad with the first failure after t=20 and a moving ego (screens), whose logged
ego is itself clean (log replay). Membership is fixed on the first run (members.json): every candidate a planner is
known to fail (ES+PDMS, A2 bank 0, M1 bank 2) is kept, the rest are dropped at random (seed 0) down to 1500.
Every run rewrites the annotations (tiers.json) and the per-tier / per-split target lists, so rerunning after more
planner results (e.g. ES+PDMS on the second half) only updates the tiers.

Writes <OUT>/{members,tiers}.json and {all,tier1_es_pdms,any_planner_fail,train,val,test}.json
({tfrecord: [scenario indices]}, the format of slurm_targets_baseline.sh / slurm_collect_anchorwm_bank.sh TARGETS).
"""
import collections
import glob
import json
import os
import random

from es_discriminator.real_es.dataset import make_split

M = "/data/user_data/mineuih/ishan/"
OUT = M + "mining/hard1500"
os.makedirs(OUT, exist_ok=True)


def results(pattern):
    return {(os.path.basename(r["tfrecord"]), int(r["scenario_idx"])): r
            for r in (json.load(open(f)) for f in glob.glob(pattern))}


def keys_of(path):
    return [(t, int(i)) for t, v in json.load(open(path)).items() for i in v]


bad = lambda r: bool(r["overlap"] or r["offroad"])  # noqa: E731
flagged = keys_of(M + "mining/pool_r0/flagged.json") + keys_of(M + "mining/pool_r2/flagged.json")
logrep = {**results(M + "mining/log_replay_flagged_r0/chunk_*/*.scenario_*.json"),
          **results(M + "mining/log_replay_flagged_r2/chunk_*/*.scenario_*.json")}
screen = results(M + "mining/pdm_screen_train_r10/*/*.scenario_*.json")
es = {**results(M + "mining/es_pdms_flagged_r0/chunk_*/*.scenario_*.json"),
      **results(M + "mining/es_pdms_hard1500_r2/chunk_*/*.scenario_*.json")}
a2 = results(M + "anchorwm_bank/dagger_r0_A2_pool/*.scenario_*.json")
m1 = results(M + "anchorwm_bank/dagger_r2_M1_pool/*.scenario_*.json")
valid = [k for k in flagged if k in logrep and not bad(logrep[k])]


def planner_fail(k):
    return any(k in d and not d[k]["success"] for d in (es, a2, m1))


members_path = f"{OUT}/members.json"
if os.path.exists(members_path):
    members = [tuple(k) for k in json.load(open(members_path))]
else:
    keep = [k for k in valid if planner_fail(k)]
    rest = sorted(k for k in valid if not planner_fail(k))
    random.Random(0).shuffle(rest)
    members = sorted(keep + rest[: max(0, 1500 - len(keep))])
    json.dump(members, open(members_path, "w"))

split = make_split([f"{t}:{i}" for t, i in members], seed=0)
tiers = {}
for k in members:
    s = screen[k]
    tiers[f"{k[0]}:{k[1]}"] = {
        "pdm_closed": "overlap" if s["overlap"] else "offroad",
        "pdm_fail_step": min(t for t in (s.get("overlap_timestep", -1), s.get("offroad_timestep", -1)) if t >= 0),
        "es_pdms": None if k not in es else ("success" if es[k]["success"] else "overlap" if es[k]["overlap"]
                                             else "offroad" if es[k]["offroad"] else "goal_miss"),
        "a2": None if k not in a2 else ("success" if a2[k]["success"] else "fail"),
        "m1": None if k not in m1 else ("success" if m1[k]["success"] else "fail"),
        "split": split[f"{k[0]}:{k[1]}"],
    }
json.dump(tiers, open(f"{OUT}/tiers.json", "w"), indent=0)


def dump(name, keys):
    d = collections.defaultdict(list)
    for t, i in keys:
        d[t].append(i)
    json.dump({t: sorted(v) for t, v in sorted(d.items())}, open(f"{OUT}/{name}.json", "w"), indent=0)
    return len(keys)


counts = {
    "all": dump("all", members),
    "tier1_es_pdms": dump("tier1_es_pdms", [k for k in members if k in es and bad(es[k])]),
    "any_planner_fail": dump("any_planner_fail", [k for k in members if planner_fail(k)]),
}
counts["es_pdms_todo"] = dump("es_pdms_todo", [k for k in members if k not in es])
for name in ("train", "val", "test"):
    counts[name] = dump(name, [k for k in members if split[f"{k[0]}:{k[1]}"] == name])
cov = {"es_pdms_run": sum(k in es for k in members), "a2_run": sum(k in a2 for k in members), "m1_run": sum(k in m1 for k in members)}
print(f"valid candidates {len(valid)} -> members {len(members)}; {counts}; coverage {cov}")
print("pdm_closed failure types:", dict(collections.Counter(v["pdm_closed"] for v in tiers.values())))
