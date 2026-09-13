"""Dump the geometry a few scenes need for an *interactive* safety-loss page.

The training figures answer "did the loss move the trajectory?". This answers a
prior and more basic question: **what does the loss actually charge, pose by
pose?** For that you want to grab the ego box and move it -- toward the kerb,
into the next lane, alongside a parked car -- and watch the two clearances and
the two barriers respond. That is not something a rendered PNG can do, so this
writes the raw geometry and :mod:`rl.diffusion_ft.tools.viz_safety_drag` turns it
into a page that recomputes the loss in the browser as you drag.

Recomputing in the browser means the JS must reproduce
:mod:`rl.diffusion_ft.safety_loss` exactly, so two things are done here to make
that possible and checkable:

* **The nearest-edge sign needs its neighbour.** ``_anchors_one_frame`` takes the
  argmin point and compares its half-plane cross product with that of the point
  *before it in the array*, keeping the smaller when both lie on the same curve.
  Filtering points by radius destroys that adjacency, so ``prior_dir`` and
  ``same_curve`` are resolved here, in the full unfiltered array, and travel with
  each kept point. The JS then needs no notion of ordering at all.
* **A self-check ships with the data.** ``probes`` holds a few dozen random pose
  offsets scored by the Python loss itself. The page recomputes them and prints
  the largest disagreement; if the JS ever drifts from the source of truth, the
  page says so instead of quietly lying.

Only ``log_trajectory`` is read -- no policy, no checkpoint, no GPU -- so this is
a short CPU job, like ``tests/calibrate_expert.py``.

Usage::

    python -m rl.diffusion_ft.tools.dump_safety_scenes \\
        --tfrecord .../failures.tfrecord --indices 0,1,2,3 --out_dir OUT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from waymax import datatypes

from rl.diffusion_ft.safety_loss import (
    SafetyLossConfig, ego_box_5dof, log_geometry, safety_terms,
)
from rl.diffusion_ft.train_dppo import load_batch

_LANE_TYPES = (1, 2, 3)


def _r(a, n=2):
    """Round for JSON. Coordinates are metres; 1 cm is far below any margin."""
    return np.round(np.asarray(a, np.float64), n).tolist()


def pick_timestep(c_edge_t: np.ndarray, c_agent_t: np.ndarray,
                  valid_t: np.ndarray, margin_edge: float) -> int:
    """The step worth opening the scene on: where the human is closest to charge.

    A page that opens on an empty straight teaches nothing -- the barrier is flat
    there and dragging is the only way to see anything happen. Opening on the
    tightest *valid* step puts the ego where the geometry is already interesting,
    with the road edge (preferred, since it is the term with 4 corners in it) or
    the nearest agent within reach of a small drag.
    """
    idx = np.flatnonzero(valid_t)
    if idx.size == 0:
        return int(len(valid_t) // 2)
    ce = c_edge_t[idx]
    if np.nanmin(ce) < margin_edge + 1.5:
        return int(idx[int(np.nanargmin(ce))])
    ca = np.where(np.isfinite(c_agent_t[idx]), c_agent_t[idx], np.inf)
    if np.isfinite(ca).any():
        return int(idx[int(np.argmin(ca))])
    return int(idx[int(np.nanargmin(ce))])


def edge_points(state, b: int, cx: float, cy: float, ego_z: float,
                radius: float, z_stretch: float) -> dict:
    """Road-edge points within ``radius``, each carrying its own sign context.

    ``dz2`` is ``(z_pt - ego_z)^2 * z_stretch^2``, the part of waymax's stretched
    3-D distance that does not change while the ego is dragged in the plane. The
    browser adds it to the planar squared distance and reproduces the same argmin
    the Python takes, without carrying z or knowing about the stretch.
    """
    rg = state.roadgraph_points
    xyz = np.asarray(rg.xyz)[b]
    dirxy = np.asarray(rg.dir_xyz)[b][:, :2]
    ids = np.asarray(rg.ids)[b]
    keep = np.asarray(rg.valid)[b] & np.asarray(
        datatypes.is_road_edge(jnp.asarray(rg.types)))[b]

    # Resolve the prior-point comparison here, in FULL array order, before any
    # filtering can break the adjacency `_anchors_one_frame` relies on.
    prior = np.maximum(np.arange(xyz.shape[0]) - 1, 0)
    prior_dir = dirxy[prior]
    same_curve = ids[prior] == ids

    # The incident polyline segments, resolved in full array order for the same
    # reason `prior_dir` is: filtering by radius destroys index adjacency, and
    # the "segment" geometry measures to these two chords.
    nxt = np.minimum(np.arange(xyz.shape[0]) + 1, xyz.shape[0] - 1)
    prev_ok = same_curve & keep[prior] & (prior != np.arange(xyz.shape[0]))
    next_ok = (ids[nxt] == ids) & keep[nxt] & (nxt != np.arange(xyz.shape[0]))

    near = keep & (np.abs(xyz[:, 0] - cx) < radius) & (np.abs(xyz[:, 1] - cy) < radius)
    sel = np.flatnonzero(near)
    dz = (xyz[sel, 2] - ego_z) * z_stretch
    return {"xy": _r(xyz[sel, :2]), "dir": _r(dirxy[sel], 4),
            "pdir": _r(prior_dir[sel], 4),
            "same": [int(v) for v in same_curve[sel]],
            "dz2": _r(dz ** 2, 3),
            "pxy": _r(xyz[prior[sel], :2]), "nxy": _r(xyz[nxt[sel], :2]),
            "pok": [int(v) for v in prev_ok[sel]],
            "nok": [int(v) for v in next_ok[sel]]}


def lane_points(state, b: int, cx: float, cy: float, radius: float, stride: int):
    """Decimated lane centrelines -- context only, never scored."""
    rg = state.roadgraph_points
    xyz = np.asarray(rg.xyz)[b]
    types = np.asarray(rg.types)[b]
    keep = np.asarray(rg.valid)[b] & np.isin(types, _LANE_TYPES)
    near = keep & (np.abs(xyz[:, 0] - cx) < radius) & (np.abs(xyz[:, 1] - cy) < radius)
    return _r(xyz[np.flatnonzero(near)[::stride], :2])


def python_probes(cfg, ego_box_5, ego_z, roadgraph_points, other_box_n5,
                  other_valid_n, rng, n: int) -> list[dict]:
    """Score random pose offsets with the *Python* loss, for the page to check.

    Offsets are drawn wide enough (+-6 m, +-0.6 rad) to land some poses in
    violation, because agreement on the plateau is worth nothing -- both sides
    return ~0 there whatever they compute.

    ``roadgraph_points`` must already be sliced to THIS scene with a leading axis
    of 1: the probe poses ride on the batch axis (one world, ``n`` "timesteps"),
    so a roadgraph still carrying the whole batch makes ``safety_terms``' vmap
    disagree with itself -- 6 worlds of roadgraph against 1 world of query.
    """
    dx = rng.uniform(-6.0, 6.0, n)
    dy = rng.uniform(-6.0, 6.0, n)
    dyaw = rng.uniform(-0.6, 0.6, n)
    box = np.repeat(np.asarray(ego_box_5, np.float32)[None, :], n, axis=0)
    box[:, 0] += dx; box[:, 1] += dy; box[:, 4] += dyaw
    terms = safety_terms(
        cfg,
        ego_box_bt5=jnp.asarray(box)[None],                    # [1, n, 5]
        ego_z_bt=jnp.full((1, n), float(ego_z), jnp.float32),
        roadgraph_points=roadgraph_points,
        other_box_btn5=jnp.asarray(other_box_n5)[None, None].repeat(n, axis=1),
        other_valid_btn=jnp.asarray(other_valid_n)[None, None].repeat(n, axis=1),
    )
    return [{"d": [round(float(dx[i]), 4), round(float(dy[i]), 4),
                   round(float(dyaw[i]), 5)],
             "ce": round(float(np.asarray(terms["c_edge"])[0, i]), 5),
             "ca": (round(float(np.asarray(terms["c_agent"])[0, i]), 5)
                    if np.isfinite(np.asarray(terms["c_agent"])[0, i]) else None),
             "el": round(float(np.asarray(terms["edge_loss"])[0, i]), 5),
             "al": round(float(np.asarray(terms["agent_loss"])[0, i]), 5)}
            for i in range(n)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tfrecord",
                   default="/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord")
    p.add_argument("--indices", default="0,1,2,3,4,5")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--radius", type=float, default=70.0,
                   help="metres of roadgraph kept around the opening pose. Must "
                        "exceed the draggable area or the browser's argmin can "
                        "miss the true nearest edge.")
    p.add_argument("--lane_stride", type=int, default=3)
    p.add_argument("--init_steps", type=int, default=11,
                   help="log-replayed prefix; never the policy's responsibility.")
    p.add_argument("--n_probes", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--edge_margin", type=float, default=0.3)
    p.add_argument("--edge_tau", type=float, default=0.15)
    p.add_argument("--agent_margin", type=float, default=0.75)
    p.add_argument("--edge_geometry", default="point", choices=("point", "segment"),
                   help="which road-edge distance the page reproduces; the "
                        "self-check compares against whichever is dumped.")
    p.add_argument("--agent_tau", type=float, default=0.25)
    args = p.parse_args()

    idx = [int(s) for s in args.indices.split(",") if s.strip()]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SafetyLossConfig(
        edge_margin_m=args.edge_margin, edge_tau_m=args.edge_tau,
        agent_margin_m=args.agent_margin, agent_tau_m=args.agent_tau,
        agent_geometry="exact", edge_geometry=args.edge_geometry)
    print(f"[dump] {len(idx)} scenes from {Path(args.tfrecord).name}  {cfg}", flush=True)

    state = load_batch(args.tfrecord, idx)
    geo = log_geometry(state)
    ego_box = ego_box_5dof(geo["ego_pose"], geo["ego_len"][:, None],
                           geo["ego_wid"][:, None])                # [B, T, 5]
    valid = np.array(geo["ego_valid"])
    valid[:, :args.init_steps] = False
    terms = safety_terms(
        cfg, ego_box_bt5=ego_box, ego_z_bt=geo["ego_z"],
        roadgraph_points=state.roadgraph_points,
        other_box_btn5=geo["other_box"], other_valid_btn=geo["other_valid"],
        step_valid_bt=jnp.asarray(valid, jnp.float32))
    c_edge = np.asarray(terms["c_edge"]); c_agent = np.asarray(terms["c_agent"])
    ego_box_np = np.asarray(ego_box)
    other_box = np.asarray(geo["other_box"]); other_valid = np.asarray(geo["other_valid"])
    ego_z = np.asarray(geo["ego_z"])
    rng = np.random.default_rng(args.seed)

    def rg_b(b: int):
        """This scene's roadgraph, keeping the leading batch axis at 1."""
        return jax.tree_util.tree_map(
            lambda x: jnp.asarray(x)[b:b + 1], state.roadgraph_points)

    scenes = []
    for b, sc in enumerate(idx):
        t = pick_timestep(c_edge[b], c_agent[b], valid[b], cfg.edge_margin_m)
        box_t = ego_box_np[b, t]
        cx, cy = float(box_t[0]), float(box_t[1])
        ov = other_valid[b, t] & (other_box[b, t, :, 2] > 0.5)
        ob = other_box[b, t][ov]
        # Neighbours far outside the view cannot be reached by any drag and cost
        # bytes and a barrier evaluation each; the plateau makes them exactly 0.
        keep = (np.abs(ob[:, 0] - cx) < args.radius) & (np.abs(ob[:, 1] - cy) < args.radius)
        ob = ob[keep]
        path = np.asarray(geo["ego_pose"])[b][np.asarray(geo["ego_valid"])[b].astype(bool), :2]
        scenes.append({
            "id": int(sc), "t": int(t),
            "view": [round(cx, 2), round(cy, 2), 40.0],
            "ego": _r(box_t, 4), "ego_z": round(float(ego_z[b, t]), 3),
            "others": _r(ob, 3),
            "path": _r(path[::2]),
            "log": {"ce": round(float(c_edge[b, t]), 4),
                    "ca": (round(float(c_agent[b, t]), 4)
                           if np.isfinite(c_agent[b, t]) else None),
                    "el": round(float(np.asarray(terms["edge_loss"])[b, t]), 4),
                    "al": round(float(np.asarray(terms["agent_loss"])[b, t]), 4)},
            "edges": edge_points(state, b, cx, cy, float(ego_z[b, t]),
                                 args.radius, cfg.z_stretch),
            "lanes": lane_points(state, b, cx, cy, args.radius, args.lane_stride),
            # An empty neighbour set would reduce over a zero-size axis inside
            # `safety_terms`; one invalid placeholder keeps the shape non-empty
            # and contributes nothing (the `where` sends it to +inf).
            "probes": python_probes(
                cfg, box_t, ego_z[b, t], rg_b(b),
                ob if len(ob) else np.zeros((1, 5), np.float32),
                np.ones(len(ob), bool) if len(ob) else np.zeros(1, bool),
                rng, args.n_probes),
        })
        print(f"[dump] scene {sc}: t={t}  c_edge={c_edge[b, t]:+.2f} "
              f"c_agent={c_agent[b, t]:+.2f}  edge_pts={len(scenes[-1]['edges']['xy'])} "
              f"others={len(ob)}", flush=True)

    out = out_dir / "safety_scenes.json"
    payload = {"config": {k: v for k, v in cfg.__dict__.items()},
               "tfrecord": args.tfrecord, "scenes": scenes}
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"[dump] wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
