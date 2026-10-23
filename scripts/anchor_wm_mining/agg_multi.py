"""Aggregate valfail1568 closed-loop runs (name=run_dir args) and pairwise scene overlap with the first run."""
import glob
import json
import os
import sys

KEYS = ("success", "overlap", "offroad", "goal_reached")
runs = {}
for arg in sys.argv[1:]:
    name, d = arg.split("=", 1)
    out = {}
    for f in glob.glob(f"{d}/chunk_*/*.scenario_*.json"):
        r = json.load(open(f))
        out[os.path.basename(f)] = {k: bool(r.get(k)) for k in KEYS}
    runs[name] = out
common = sorted(set.intersection(*(set(v) for v in runs.values())))
n = len(common)
print(f"scenes per run: " + ", ".join(f"{k} {len(v)}" for k, v in runs.items()) + f"; common {n}")
print(f"{'':28s} {'success':>14s} {'overlap':>14s} {'offroad':>14s} {'goal miss':>14s}")
for name, r in runs.items():
    s = sum(r[x]["success"] for x in common)
    o = sum(r[x]["overlap"] for x in common)
    f = sum(r[x]["offroad"] for x in common)
    g = sum(not r[x]["goal_reached"] for x in common)
    print(f"{name:28s} " + " ".join(f"{v:5d} ({100 * v / n:4.1f}%)" for v in (s, o, f, g)))
names = list(runs)
ref = names[0]
for other in names[1:]:
    a, b = runs[ref], runs[other]
    both = sum(a[x]["success"] and b[x]["success"] for x in common)
    oa = sum(a[x]["success"] and not b[x]["success"] for x in common)
    ob = sum(b[x]["success"] and not a[x]["success"] for x in common)
    print(f"paired {ref} vs {other}: both {both}, {ref} only {oa}, {other} only {ob}, neither {n - both - oa - ob}")
