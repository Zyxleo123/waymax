"""Assemble a self-contained scene bundle for the deploy-time repair: real
trajectories of failure/success scenes, base vs the bank arms, on the map.

The select run already dumped the EXECUTED world paths per arm (``<arm>/r<rep>/traj``
in ``feat/*.npz`` when scored with ``--dump_traj``). What it did NOT persist is the
static geometry -- roadgraph, other agents, goal -- because that is the same for
every arm and costs nothing to reload. This job reloads exactly that geometry for
a chosen handful of scenes (a plain ``sampler.load_explicit`` + ``env.reset``, NO
denoiser, NO rollout), pairs it with the dumped paths, and writes one small
``scenes.json`` that :mod:`rl.diffusion_ft.tools.build_scene_viz` turns into an
interactive page where each arm's path toggles on and off.

Why reload rather than re-roll: the paths are on disk and deterministic; re-rolling
would re-pay the denoiser and risk a different draw. Why not replay them through
``env.step`` for per-step offroad marks: ``env.step`` expects an anchor at index 0
and would misalign a bare executed segment, so the honest signal here is the
episode outcome (base OFFROAD/OVERLAP, repair SOLVED), read from the run's own
``rows.jsonl``. Geometry needs the diffusion env, which only initialises on
a6000/a5000, so this is a short GPU job even though it never calls the model.

Scenes default to the curated recovered/unrecovered/preserved set (override with
env/flags): recovered = base fails, the bank solves; unrecovered = base fails and
the bank still does not solve it (the residual, which is what the line is judged
against next); preserved = base solves and the bank leaves it solved (do-no-harm).

The bundle also carries a ``reasons`` block computed over the WHOLE failure split,
not just the drawn scenes, so the page can say what the residual is MADE of
without anyone re-running the eval.

Usage (GPU, geometry only), all the env/sampler flags of proposal_select plus:
    python -m rl.diffusion_ft.tools.dump_reach_from_select \
      --select_dir <select run dir> --out_json <dir>/scenes.json \
      --arms base,c_r10,c_r20 \
      --recovered 59208,60665,86688,126300,12956,17880 \
      --unrecovered 7040,18412,383195,291317,51493,5757 \
      --preserved 8844,5536,50
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

# WOMD roadgraph type codes (waymax): lane centers vs road boundaries.
LANE_TYPES = (1, 2, 3)
EDGE_TYPES = (14, 15, 16)


def _path_index(select_dir: Path):
    """Map scene_id -> (npz file, column, split) across all shards' feat/*.npz."""
    idx = {}
    for f in sorted(glob.glob(str(select_dir / "tasks/*/feat/*.npz"))):
        name = Path(f).name
        split = name.split("_s")[0]
        sids = np.load(f)["scene_ids"]
        for j, sid in enumerate(sids):
            idx.setdefault(int(sid), (f, j, split))
    return idx


def _episode_path(npz, arm, rep, col):
    """Concatenate the per-replan executed windows into one [T,5] path, nan-trimmed."""
    key = f"{arm}/r{rep}/traj"
    if key not in npz.files:
        return None
    tr = npz[key][col]                     # [replans, steps, 5]
    tr = tr.reshape(-1, tr.shape[-1])      # [replans*steps, 5]
    keep = ~np.isnan(tr[:, 0])
    return tr[keep].astype(np.float32)


def _rows_by_scene(select_dir: Path):
    R = {}
    for f in glob.glob(str(select_dir / "tasks/*/rows.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            R[(int(r["scene_id"]), r["arm"], int(r["repeat"]))] = r
    return R


def _cause(r):
    """``no-goal`` / ``offroad`` / ``overlap`` combination of one episode, or ``solved``."""
    if r["solved"]:
        return "solved"
    return "+".join(n for n, hit in (("no-goal", not r["reached"]), ("offroad", r["offroad"]),
                                     ("overlap", r["overlap"])) if hit) or "other"


def _auto_pick(rows, arm, n, reps=4, base="base"):
    """Pick up to ``n`` scenes per kind from the run's own rows, diversified.

    Per scene, over the paired repeats: ``bs`` = repeats the base solves, ``as_`` =
    repeats ``arm`` solves.

    * recovered   -- bs == 0, as_ >= 1; most-often-recovered first.
    * unrecovered -- bs == 0, as_ == 0 (the residual).
    * regressed   -- some repeat where the base solves and the arm does not; the
      scenes with the most such repeats first. Usually few: show them all.
    * preserved   -- bs == as_ == reps; n-1 the arm TOUCHED (fired, still clean)
      plus 1 it never touched, from success_heldout first.

    Diversity: within a kind, candidates are bucketed by (split, failure cause)
    and drawn round-robin, largest bucket first, so the page shows every mode of
    each kind instead of the most common one n times. Ties break on scene id, so
    the pick is a deterministic function of the rows.
    """
    by = {}
    for (sid, a, rp), r in rows.items():
        by.setdefault(sid, {}).setdefault(a, {})[rp] = r
    cand = {"recovered": [], "unrecovered": [], "regressed": [], "preserved": []}
    for sid, d in by.items():
        b, x = d.get(base, {}), d.get(arm, {})
        common = [rp for rp in range(reps) if rp in b and rp in x]
        if len(common) < reps:
            continue
        split = b[0]["split"]
        bs = sum(b[rp]["solved"] for rp in common)
        as_ = sum(x[rp]["solved"] for rp in common)
        reg = [rp for rp in common if b[rp]["solved"] and not x[rp]["solved"]]
        fired = sum(bool(x[rp].get("n_switch", 0)) for rp in common)
        if reg:
            cand["regressed"].append((-len(reg), sid, (split, _cause(x[reg[0]]))))
        if bs == 0 and as_ > 0:
            cand["recovered"].append((-as_, sid, (split, _cause(b[0]))))
        elif bs == 0:
            worst = max(common, key=lambda rp: x[rp].get("n_switch", 0))
            cand["unrecovered"].append((0, sid, (split, _cause(x[worst]), fired > 0)))
        elif bs == as_ == reps:
            cand["preserved"].append((0 if split == "success_heldout" else 1, sid,
                                      (fired > 0,)))

    def round_robin(items, k):
        groups = {}
        for pri, sid, key in sorted(items):
            groups.setdefault(key, []).append(sid)
        order = sorted(groups.values(), key=len, reverse=True)
        out = []
        while len(out) < k and any(order):
            for g in order:
                if g and len(out) < k:
                    out.append(g.pop(0))
        return out

    picked = {k: round_robin(v, n) for k, v in cand.items() if k != "preserved"}
    touched = [c for c in sorted(cand["preserved"]) if c[2][0]]
    untouched = [c for c in sorted(cand["preserved"]) if not c[2][0]]
    picked["preserved"] = [c[1] for c in touched[:max(n - 1, 0)]] + [c[1] for c in untouched[:1]]
    counts = {k: len(v) for k, v in cand.items()}
    return picked, counts


def _pick_repeat(rows, sid, arm, kind, reps=4, base="base"):
    """Which of the paired repeats to draw for this scene.

    regressed:   the first repeat where the base solves and the arm does not.
    recovered:   the first repeat the bank actually solves -- the point of the card.
    unrecovered: the repeat where the filter fired the MOST, so a card that still
                 fails shows the strongest repair attempt rather than an arbitrary
                 draw. Ties, and the scenes where it never fires at all, keep 0.
    preserved:   the repeat the arm fired in most (0 if it never did).
    """
    if kind == "regressed":
        for rp in range(reps):
            b, r = rows.get((sid, base, rp)), rows.get((sid, arm, rp))
            if b and r and b["solved"] and not r["solved"]:
                return rp
    if kind == "recovered":
        for rp in range(reps):
            r = rows.get((sid, arm, rp))
            if r and r["solved"]:
                return rp
    elif kind in ("unrecovered", "preserved"):
        # unrecovered: the strongest repair attempt. preserved: a repeat the arm
        # actually TOUCHED, so a "still clean" card shows the edit it survived.
        fires = [(rows.get((sid, arm, rp)) or {}).get("n_switch", 0) for rp in range(reps)]
        if max(fires, default=0) > 0:
            return int(max(range(reps), key=lambda i: fires[i]))
    return 0


def _reason_summary(rows, split, arms, reps=4):
    """Why episodes fail, per arm, over the whole split -- not just the drawn scenes.

    ``solved`` is reached & !offroad & !overlap, so one failed episode can carry up
    to three causes at once. ``tags`` keeps the exact combination ("no-goal+overlap")
    for the composition bar; ``marginal`` counts each cause independently, and it is
    the marginal row read ACROSS arms that carries the finding -- the filter can only
    delete violations, it cannot make the car arrive, so a no-goal count that barely
    moves while offroad/overlap collapse says the residual is a reachability problem
    and not a repair-strength one.

    ``scene_recovery`` collapses the repeats (a scene is ``never`` recovered when the
    arm solves 0 of them) and splits the never-recovered scenes by whether the filter
    ever fired -- i.e. whether the residual is a repair that was never ATTEMPTED (the
    violation predictor stayed quiet) or one that was attempted and was not enough.
    """
    sids = sorted({sid for (sid, arm, rp), r in rows.items() if r["split"] == split})
    out = {"split": split, "repeats": reps, "n_scenes": len(sids),
           "arms": {}, "scene_recovery": {}}
    for arm in arms:
        eps = [rows[(sid, arm, rp)] for sid in sids for rp in range(reps)
               if (sid, arm, rp) in rows]
        if not eps:
            continue
        tags, marg, n_solved = {}, {"no_goal": 0, "offroad": 0, "overlap": 0}, 0
        for r in eps:
            causes = [("no-goal", not r["reached"]), ("offroad", r["offroad"]),
                      ("overlap", r["overlap"])]
            for name, hit in causes:
                if hit:
                    marg[name.replace("-", "_")] += 1
            if r["solved"]:
                n_solved += 1
                continue
            t = "+".join(n for n, hit in causes if hit) or "other"
            tags[t] = tags.get(t, 0) + 1
        got = lambda sid, rp: rows.get((sid, arm, rp)) or {}
        never = [s for s in sids if not any(got(s, rp).get("solved") for rp in range(reps))]
        always = [s for s in sids if all(got(s, rp).get("solved") for rp in range(reps))]
        nofire = [s for s in never
                  if all(got(s, rp).get("n_switch", 0) == 0 for rp in range(reps))]
        out["arms"][arm] = {"n_episodes": len(eps), "n_solved": n_solved,
                            "tags": tags, "marginal": marg}
        out["scene_recovery"][arm] = {
            "never": len(never), "partial": len(sids) - len(never) - len(always),
            "always": len(always), "never_no_fire": len(nofire),
            "never_fired": len(never) - len(nofire),
            "never_ids": never}
    return out


def _crop_mask(x, y, valid, bbox, stride=1):
    lo_x, hi_x, lo_y, hi_y = bbox
    m = valid & (x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y)
    idx = np.nonzero(m)[0][::stride]
    return idx


def main():
    from rl.diffusion_ft.train_safety import SafetyFineTuner, build_argparser
    p = build_argparser()
    p.add_argument("--select_dir", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--viz_arms", default="base,c_r10,c_r20",
                   help="arms to pull executed paths for (must exist in the dump)")
    p.add_argument("--recovered", default="",
                   help="comma scene_ids: base fails, a bank arm solves")
    p.add_argument("--unrecovered", default="",
                   help="comma scene_ids: base fails AND the arm still fails (the residual)")
    p.add_argument("--preserved", default="",
                   help="comma scene_ids: base solves, arm leaves it solved")
    p.add_argument("--regressed", default="",
                   help="comma scene_ids: base solves, the arm fails (do-harm)")
    p.add_argument("--auto_pick", type=int, default=0,
                   help="N>0: ignore the id lists and pick up to N scenes per kind "
                        "(recovered/unrecovered/regressed/preserved) from the rows, "
                        "judged on --solved_arm vs base; see _auto_pick")
    p.add_argument("--reps", type=int, default=4, help="paired repeats per scene in the run")
    p.add_argument("--reason_split", default="failure_heldout",
                   help="split the failure-reason breakdown is computed over")
    p.add_argument("--solved_arm", default="c_r10",
                   help="arm whose solve picks the repeat for recovered scenes")
    p.add_argument("--pad_m", type=float, default=12.0, help="map crop margin around paths")
    p.add_argument("--lane_stride", type=int, default=3, help="downsample lane points")
    a = p.parse_args()

    # Mirror proposal_select's lazy setup: no eval preload, load scenes on demand.
    a.splits = ""
    a.eval_scenes = 0
    a.probe_per_split = 0
    a.prefetch = 0

    select_dir = Path(a.select_dir)
    viz_arms = [x for x in a.viz_arms.split(",") if x]
    rows = _rows_by_scene(select_dir)
    if a.auto_pick > 0:
        if not rows:
            print(f"[dump_reach] NOT RUN: no rows under {select_dir} (upstream gated out "
                  f"or never ran)", flush=True)
            return
        picked, counts = _auto_pick(rows, a.solved_arm, a.auto_pick, a.reps)
        print(f"[dump_reach] auto-pick vs {a.solved_arm}: population {counts}; drawing "
              f"{ {k: len(v) for k, v in picked.items()} }", flush=True)
        rec, unrec = picked["recovered"], picked["unrecovered"]
        reg, pres = picked["regressed"], picked["preserved"]
    else:
        rec = [int(x) for x in a.recovered.split(",") if x.strip()]
        unrec = [int(x) for x in a.unrecovered.split(",") if x.strip()]
        reg = [int(x) for x in a.regressed.split(",") if x.strip()]
        pres = [int(x) for x in a.preserved.split(",") if x.strip()]
        counts = None
    if not (rec or unrec or reg or pres):
        raise SystemExit("give --recovered/--unrecovered/--regressed/--preserved ids or --auto_pick")
    scene_kind = ({sid: "recovered" for sid in rec} | {sid: "unrecovered" for sid in unrec}
                  | {sid: "regressed" for sid in reg} | {sid: "preserved" for sid in pres})
    ids = list(dict.fromkeys(rec + unrec + reg + pres))

    pidx = _path_index(select_dir)
    missing = [s for s in ids if s not in pidx]
    if missing:
        print(f"[dump_reach] not in select dump, dropping: {missing}", flush=True)
        ids = [s for s in ids if s in pidx]

    print(f"[dump_reach] building env (geometry only, no denoiser) ...", flush=True)
    tuner = SafetyFineTuner(a)
    env = tuner.env

    import jax.numpy as jnp
    state = tuner.sampler.load_explicit(ids)
    env.reset(state)
    st = env.sim_state
    rg = st.roadgraph_points
    log = st.log_trajectory
    is_sdc = np.asarray(st.object_metadata.is_sdc)                 # [B, N]
    goal_xy = np.asarray(env._goal_xy_b2, np.float32)             # [B, 2]
    ego_log = np.asarray(env.ego_log_world_trajectory(), np.float32)  # [B, T, 5]
    start_step = int(env._start_timestep)

    rg_x = np.asarray(rg.x, np.float32); rg_y = np.asarray(rg.y, np.float32)
    rg_t = np.asarray(rg.types, np.int16); rg_v = np.asarray(rg.valid, bool)
    ag_x = np.asarray(log.x, np.float32); ag_y = np.asarray(log.y, np.float32)
    ag_yaw = np.asarray(log.yaw, np.float32)
    ag_len = np.asarray(log.length, np.float32); ag_wid = np.asarray(log.width, np.float32)
    ag_valid = np.asarray(log.valid, bool)                        # [B, N, T]

    scenes = []
    for b, sid in enumerate(ids):
        f, col, split = pidx[sid]
        npz = np.load(f)
        rp = _pick_repeat(rows, sid, a.solved_arm, scene_kind[sid], a.reps)
        paths, outcomes = {}, {}
        pxy_all = []
        for arm in viz_arms:
            pth = _episode_path(npz, arm, rp, col)
            if pth is None:
                continue
            paths[arm] = pth[:, :3].tolist()          # x, y, yaw
            pxy_all.append(pth[:, :2])
            r = rows.get((sid, arm, rp))
            reps_r = [rows.get((sid, arm, q)) for q in range(a.reps)]
            outcomes[arm] = ({"solved": r["solved"], "offroad": r["offroad"],
                              "overlap": r["overlap"], "reached": r["reached"],
                              "n_switch": r.get("n_switch", 0),
                              # over ALL repeats, so one drawn card cannot read as
                              # luckier (or unluckier) than the scene actually is
                              "n_solved_reps": sum(bool(q and q["solved"]) for q in reps_r),
                              "n_reps": sum(q is not None for q in reps_r),
                              "fired_reps": sum(bool(q and q.get("n_switch", 0)) for q in reps_r)}
                             if r else {})
        if not pxy_all:
            print(f"[dump_reach] scene {sid}: no arm paths, skipping", flush=True)
            continue
        pxy = np.concatenate(pxy_all + [goal_xy[b][None], ego_log[b, start_step:, :2]], 0)
        bbox = (float(pxy[:, 0].min()) - a.pad_m, float(pxy[:, 0].max()) + a.pad_m,
                float(pxy[:, 1].min()) - a.pad_m, float(pxy[:, 1].max()) + a.pad_m)

        # roadgraph, cropped to the action and downsampled
        eidx = _crop_mask(rg_x[b], rg_y[b], rg_v[b] & np.isin(rg_t[b], EDGE_TYPES), bbox)
        lidx = _crop_mask(rg_x[b], rg_y[b], rg_v[b] & np.isin(rg_t[b], LANE_TYPES), bbox, a.lane_stride)
        edges = np.stack([rg_x[b][eidx], rg_y[b][eidx]], -1).round(2).tolist()
        lanes = np.stack([rg_x[b][lidx], rg_y[b][lidx]], -1).round(2).tolist()

        # other agents: any non-sdc object with a valid step inside the window,
        # exported as its box (center path + constant size) over valid steps.
        agents = []
        for n in range(is_sdc.shape[1]):
            if is_sdc[b, n]:
                continue
            v = ag_valid[b, n]
            inwin = v & (ag_x[b, n] >= bbox[0]) & (ag_x[b, n] <= bbox[1]) \
                      & (ag_y[b, n] >= bbox[2]) & (ag_y[b, n] <= bbox[3])
            if not inwin.any():
                continue
            steps = np.nonzero(v)[0]
            agents.append({
                "l": float(np.nanmedian(ag_len[b, n][v])),
                "w": float(np.nanmedian(ag_wid[b, n][v])),
                "t": steps[::2].tolist(),
                "xy": np.stack([ag_x[b, n][steps[::2]], ag_y[b, n][steps[::2]]], -1).round(2).tolist(),
                "yaw": ag_yaw[b, n][steps[::2]].round(3).tolist(),
            })

        scenes.append({
            "scene_id": sid, "split": split, "kind": scene_kind[sid], "repeat": rp,
            "bbox": [round(x, 2) for x in bbox],
            "start_step": start_step,
            "goal": goal_xy[b].round(2).tolist(),
            "ego_log": ego_log[b, :, :2].round(2).tolist(),
            "edges": edges, "lanes": lanes, "agents": agents,
            "paths": {k: np.round(np.asarray(v), 3).tolist() for k, v in paths.items()},
            "outcomes": outcomes,
        })
        print(f"[dump_reach] scene {sid} ({scene_kind[sid]}, {split}, r{rp}): "
              f"{len(edges)} edge / {len(lanes)} lane pts, {len(agents)} agents, "
              f"arms {list(paths)}", flush=True)

    out = {"arms": viz_arms, "start_step": start_step, "kind_counts": counts,
           "select_dir": str(select_dir), "scenes": scenes,
           "reasons": _reason_summary(rows, a.reason_split, viz_arms, a.reps)}
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(out))
    print(f"[dump_reach] {len(scenes)} scenes -> {a.out_json}", flush=True)


if __name__ == "__main__":
    main()
