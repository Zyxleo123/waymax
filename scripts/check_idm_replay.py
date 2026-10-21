"""Does replaying a closed-loop run's executed ego (diagnose_idm_gap.py) reproduce that run's own
flags? Joins the replay JSONL rows of one NPC model with the run's per-scene JSONs.

    python scripts/check_idm_replay.py <replay.jsonl> <run root>/<view>/<arm> --npc idm
"""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("jsonl")
ap.add_argument("arm_dir")
ap.add_argument("--npc", default="idm")
a = ap.parse_args()
rep = {}
for line in open(a.jsonl):
    r = json.loads(line)
    if r["npc"] == a.npc:
        rep[(r["shard"], r["scenario_idx"])] = r
run = {}
for p in glob.glob(os.path.join(a.arm_dir, "**", "*.json"), recursive=True):
    if ".scenario_" not in os.path.basename(p):
        continue
    r = json.load(open(p))
    run[(os.path.basename(r["tfrecord"]), r["scenario_idx"])] = r
common = sorted(set(rep) & set(run))
for key in ("overlap", "offroad", "goal_reached", "success"):
    agree = sum(rep[k][key] == run[k][key] for k in common)
    print(f"{key:12s} replay == run on {agree}/{len(common)}")
bad = [k for k in common if rep[k]["success"] != run[k]["success"]]
for k in bad[:10]:
    print("  differs:", k, {x: (run[k][x], rep[k][x]) for x in ("overlap", "offroad", "goal_reached")})
