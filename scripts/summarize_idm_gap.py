"""Tabulate scripts/diagnose_idm_gap.py output: rates per (ego, NPC model), and why each
overlap that log replay does not have happened (the colliding NPC vs its own recording).

Collider classes, first match wins:
  teleport      a single-step jump > 5 m (50 m/s) before contact: an object placed at the
                (-1, -1) sentinel by waymax and then projected back onto its route
  past_log_end  the NPC's log had already ended: kept alive and driven past it
  log_gap       its log is invalid at contact but valid before and after
  ahead         > 1 m AHEAD of its logged schedule along its own path (racing)
  behind        > 1 m BEHIND its logged schedule (braked / stalled where the human did not)
  parked_moved  a parked car (logged path < 2 m) displaced > 0.5 m
  off_path      > 1 m laterally off its logged path
  on_schedule   where its recording puts it (+-1 m)
"""

from __future__ import annotations

import argparse
import collections
import json


def collider_class(c: dict) -> str:
    if c.get("max_jump_m", 0) > 5.0:
        return "teleport"
    if not c["log_valid"]:
        return "past_log_end" if c.get("past_end_m") is not None else "log_gap"
    s = c.get("sched_m")
    if s is not None and s > 1.0:
        return "ahead"
    if s is not None and s < -1.0:
        return "behind"
    if c.get("path_len_m", 99) < 2.0 and (c.get("dev_m") or 0) > 0.5:
        return "parked_moved"
    if c.get("lat_m", 0) > 1.0:
        return "off_path"
    return "on_schedule"


def bearing_class(c: dict) -> str:
    a = abs(c["bearing_deg"])
    return "front" if a < 45 else ("rear" if a > 135 else "side")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--base", default="log", help="NPC model the others are compared against")
    a = ap.parse_args()
    rows = [json.loads(line) for p in a.jsonl for line in open(p)]
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["ego"], r["npc"])][(r["shard"], r["scenario_idx"])] = r
    egos = sorted({e for e, _ in by})
    for ego in egos:
        models = [m for (e, m) in by if e == ego]
        common = set.intersection(*(set(by[(ego, m)]) for m in models))
        print(f"\n######## ego = {ego}   ({len(common)} scenes in every NPC model)")
        for m in models:
            rr = [by[(ego, m)][k] for k in common]
            n = len(rr)
            agree = sum(r["metric_agrees"] for r in rr)
            print(f"  {m:10s} success {sum(r['success'] for r in rr) / n:.4f}  collision {sum(r['overlap'] for r in rr) / n:.4f}"
                  f"  offroad {sum(r['offroad'] for r in rr) / n:.4f}  no-goal {sum(not r['goal_reached'] for r in rr) / n:.4f}"
                  f"  | NPC dev p99 (scene median) {sorted(r['npc_dev_p99'] for r in rr)[n // 2]:.2f} m"
                  f"  | NPC steps alive past log {sum(r['npc_alive_past_log'] for r in rr) / n:.1f}/scene"
                  f"  | metric==per-object {agree}/{n}")
        if a.base not in models:
            continue
        base = by[(ego, a.base)]
        for m in models:
            if m == a.base:
                continue
            cur = by[(ego, m)]
            new = [k for k in common if cur[k]["overlap"] and not base[k]["overlap"]]
            gone = [k for k in common if base[k]["overlap"] and not cur[k]["overlap"]]
            ng_new = [k for k in common if not cur[k]["goal_reached"] and base[k]["goal_reached"]]
            print(f"\n  {m} vs {a.base}: +{len(new)} collisions, -{len(gone)} collisions, +{len(ng_new)} no-goal  (n={len(common)})")
            cls = collections.Counter()
            cls_b = collections.Counter()
            typ = collections.Counter()
            for k in new:
                c = cur[k]["collider"]
                cc = collider_class(c)
                cls[cc] += 1
                cls_b[(cc, bearing_class(c))] += 1
                typ[c["type"]] += 1
            for cc, cnt in cls.most_common():
                br = ", ".join(f"{b} {cls_b[(cc, b)]}" for b in ("front", "side", "rear") if cls_b[(cc, b)])
                print(f"    {cc:13s} {cnt:4d}  ({cnt / max(len(new), 1):.0%})   {br}")
            print(f"    collider types: {dict(typ)}")
            if ng_new:
                bc = collections.Counter()
                for k in ng_new:
                    b = cur[k].get("blocker")
                    bc["no NPC ahead" if b is None else collider_class(b)] += 1
                print(f"    no-goal blockers: {dict(bc.most_common())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
