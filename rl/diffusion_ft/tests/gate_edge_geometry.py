"""Does ``edge_geometry="segment"`` remove the cliff without moving the boundary?

The road-edge term measures to the nearest roadgraph *sample*, which is waymax's
definition and is what makes ``c_edge``'s sign identical to ``offroad``. But the
sign flips across the line through that sample while the distance to it does not
go to zero, so a corner crossing the polyline midway between two samples flips
sign at a *finite* magnitude: measured on ``failures.tfrecord`` (1.01 m spacing),
``c_edge`` jumps +0.224 -> -0.224 m in one 3 mm step, and the lateral gradient
collapses to 0.054 just inside the crossing where it should be 1. A barrier
cannot shape what it cannot see, and that cliff is exactly at the boundary the
whole term exists to keep the policy on the right side of.

``edge_geometry="segment"`` measures to the polyline segment incident to the
anchor instead. Three things must all hold for that to be a fix rather than a
different loss, and this gate asserts each:

  1. **The boundary does not move.** The sign is untouched by construction, so
     ``sign(c_edge)`` must be identical under the two geometries at every pose --
     not close, identical. If it is not, ``offroad`` agreement is lost and the
     module's central claim goes with it.
  2. **The cliff is gone.** Sweeping a corner across an edge, the largest
     step-to-step jump must fall to the sweep resolution. The gate also asserts
     the *old* geometry still jumps, so a fixture that never crosses an edge
     cannot pass this vacuously.
  3. **The gradient returns.** ``|d c_edge / d lateral|`` at the crossing must be
     ~1 (it is a distance to a line the corner is moving perpendicular to).

Gate 0 re-checks "point" against waymax, because the refactor that added the
segment path also rewrote the anchor plumbing.

Runs on a synthetic corridor in seconds with no corpus. ``--tfrecord`` repeats
1 and 2 on real scenes, where the roadgraph is not uniformly sampled.

Run: ``python -m rl.diffusion_ft.tests.gate_edge_geometry [--tfrecord PATH]``
"""

from __future__ import annotations

import argparse
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

from waymax.metrics import roadgraph as wrg
from waymax.utils import geometry as wgeom

from rl.diffusion_ft.safety_loss import (
    SafetyLossConfig, ego_box_5dof, log_geometry, road_edge_anchors,
    road_edge_clearance, safety_terms, signed_distance_from_anchors,
)
from rl.diffusion_ft.tests.test_safety_loss import report, straight_road, unbatched

EGO_L, EGO_W = 4.8, 2.0


def _clearance(rg, xs, ys, yaw, geometry, z=0.0):
    """``[N, 4]`` per-corner clearance for a line of poses, one call.

    ``z`` is not cosmetic on real scenes. The nearest-point search ranks by the
    z-STRETCHED distance but reports the planar one, so querying at the wrong
    altitude does not merely add a constant -- it selects a different anchor, and
    the reported distance then jumps around as the query moves. Passing z=0 on a
    scene whose roadgraph sits at z=-10 made this gate report ~2 m jumps under
    *both* geometries (its own bug, read as the fix not working).
    """
    pose = jnp.stack([jnp.asarray(xs, jnp.float32), jnp.asarray(ys, jnp.float32),
                      jnp.full((len(xs),), yaw, jnp.float32)], axis=-1)[None]
    box = ego_box_5dof(pose, jnp.asarray([[EGO_L]]), jnp.asarray([[EGO_W]]))
    z_bt = jnp.full((1, len(xs)), float(z), jnp.float32)
    c, foot = road_edge_clearance(box, z_bt, rg, time_chunk=8,
                                  edge_geometry=geometry)
    return np.asarray(c)[0], np.asarray(foot)[0]


def gate0_waymax_parity() -> bool:
    """"point" still reproduces waymax after the anchor plumbing was rewritten."""
    rg = straight_road()
    ys = np.linspace(-6.0, 6.0, 40)
    pose = jnp.stack([jnp.full(40, 20.0), jnp.asarray(ys, jnp.float32),
                      jnp.zeros(40)], axis=-1)[None].astype(jnp.float32)
    box = ego_box_5dof(pose, jnp.asarray([[EGO_L]]), jnp.asarray([[EGO_W]]))
    corners = wgeom.corners_from_bboxes(box)
    corners_xyz = jnp.concatenate([corners, jnp.zeros(corners.shape[:-1] + (1,))], -1)
    near, sign = road_edge_anchors(corners_xyz, rg, time_chunk=8)
    ours = np.asarray(signed_distance_from_anchors(corners, near, sign))[0]
    theirs = np.asarray(wrg.compute_signed_distance_to_nearest_road_edge_point(
        corners_xyz[0].reshape(-1, 3), unbatched(rg))).reshape(ours.shape)
    err = float(np.max(np.abs(ours - theirs)))
    return report("gate 0  point == waymax", err < 1e-4,
                  f"max |ours - waymax| = {err:.2e} m over {ours.size} corners")


def gate1_same_boundary(rg, label="synthetic") -> bool:
    """The two geometries must agree on the SIGN everywhere -- exactly."""
    rng = np.random.default_rng(0)
    n = 600
    xs = rng.uniform(0.0, 100.0, n); ys = rng.uniform(-9.0, 9.0, n)
    bad = 0
    for yaw in (0.0, 0.7, -1.3):
        cp, _ = _clearance(rg, xs, ys, yaw, "point")
        cs, _ = _clearance(rg, xs, ys, yaw, "segment")
        bad += int(np.sum(np.sign(cp) != np.sign(cs)))
    total = 3 * n * 4
    return report(f"gate 1  boundary unmoved [{label}]", bad == 0,
                  f"{bad}/{total} corners disagree on sign "
                  f"(any disagreement breaks `offroad` identity)")


def gate2_continuity(rg, x0, y_from, y_to, yaw=0.0, label="synthetic") -> bool:
    """Sweep across the edge: the old geometry cliffs, the new one must not."""
    n = 4001
    ys = np.linspace(y_from, y_to, n)
    xs = np.full(n, x0)
    step = abs(ys[1] - ys[0])
    ok = True
    jumps = {}
    for geom in ("point", "segment"):
        c, _ = _clearance(rg, xs, ys, yaw, geom)
        cmin = c.min(axis=-1)
        if not (np.any(cmin > 0) and np.any(cmin < 0)):
            return report(f"gate 2  continuity [{label}]", False,
                          "sweep never crosses the edge -- the fixture is wrong, "
                          "and a no-crossing sweep would pass vacuously")
        jumps[geom] = float(np.max(np.abs(np.diff(cmin))))
    # The defect must be present, or "fixed" means nothing here.
    ok &= report(f"gate 2a old geometry cliffs [{label}]", jumps["point"] > 10 * step,
                 f"max jump {jumps['point']*100:.1f} cm over a {step*100:.2f} cm step")
    ok &= report(f"gate 2b new geometry continuous [{label}]",
                 jumps["segment"] < max(0.02, 4 * step),
                 f"max jump {jumps['segment']*100:.2f} cm "
                 f"({jumps['point']/max(jumps['segment'], 1e-9):.0f}x smaller)")
    return ok


def gate3_gradient_at_crossing(rg) -> bool:
    """At the boundary the clearance must move 1:1 with lateral position."""
    def c_of_y(y):
        pose = jnp.stack([jnp.float32(20.0), y, jnp.float32(0.0)])[None, None]
        box = ego_box_5dof(pose, jnp.asarray([[EGO_L]]), jnp.asarray([[EGO_W]]))
        c, _ = road_edge_clearance(box, jnp.zeros((1, 1)), rg, time_chunk=1,
                                   edge_geometry="segment")
        return jnp.min(c)                      # min over corners == the metric's

    # Find the crossing to ~mm, then differentiate just inside it.
    lo, hi = 0.0, 8.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if float(c_of_y(jnp.float32(mid))) > 0: lo = mid
        else: hi = mid
    g = float(jax.grad(c_of_y)(jnp.float32(lo - 1e-3)))
    return report("gate 3  gradient at the crossing", abs(g) > 0.8,
                  f"|d c_edge / d y| = {abs(g):.3f} at the boundary "
                  f"(y = {lo:.4f}; expect ~1)")


def real_scene_gates(tfr: str, indices: list[int]) -> bool:
    """Repeat 1 and 2 where the roadgraph is real: gaps, curves, junctions."""
    from rl.diffusion_ft.train_dppo import load_batch
    ok = True
    state = load_batch(tfr, indices)
    geo = log_geometry(state)
    ego_pose = np.asarray(geo["ego_pose"])
    for b, sc in enumerate(indices):
        rg = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[b:b + 1],
                                    state.roadgraph_points)
        t = int(np.asarray(geo["ego_valid"])[b].astype(bool).sum() // 2)
        x0, y0, yaw = ego_pose[b, t]
        # Sweep laterally through the pose: real scenes put an edge within a few
        # metres of the driven line, which is the whole reason they were curated.
        z = float(np.asarray(geo["ego_z"])[b, t])       # the log's own altitude
        nx, ny = -np.sin(yaw), np.cos(yaw)
        ts = np.linspace(-8.0, 8.0, 4001)
        xs, ys = x0 + nx * ts, y0 + ny * ts
        cp, _ = _clearance(rg, xs, ys, float(yaw), "point", z)
        cs, _ = _clearance(rg, xs, ys, float(yaw), "segment", z)
        bad = int(np.sum(np.sign(cp) != np.sign(cs)))
        ok &= report(f"gate 1  boundary unmoved [scene {sc}]", bad == 0,
                     f"{bad}/{cp.size} corners disagree on sign")
        pm, sm = cp.min(-1), cs.min(-1)
        if np.any(pm > 0) and np.any(pm < 0):
            jp = float(np.max(np.abs(np.diff(pm))))
            dj = np.abs(np.diff(sm))
            js = float(np.max(dj))
            ok &= report(f"gate 2  continuity [scene {sc}]", js < 0.02,
                         f"max jump: point {jp*100:5.1f} cm -> segment {js*100:.2f} cm")
            if js >= 0.02:
                # Two different defects can survive the fix, and they need
                # opposite responses: a jump *at* a sign flip means the segment
                # was rejected there (a curve junction, so no valid neighbour and
                # the point distance is the fallback); a jump with no sign flip
                # is the anchor switching curves, which is the z-stretched argmin
                # and not something this geometry claims to address.
                i = int(np.argmax(dj))
                flipped = np.sign(sm[i]) != np.sign(sm[i + 1])
                print(f"        worst jump at t={ts[i]:+.3f} m: "
                      f"{sm[i]:+.3f} -> {sm[i+1]:+.3f} m, sign "
                      f"{'FLIPS (junction fallback)' if flipped else 'unchanged (anchor switched curve)'}")
        else:
            print(f"  ....  scene {sc}: sweep never crosses an edge, continuity "
                  f"not measurable here")
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tfrecord", default="")
    p.add_argument("--indices", default="0,1,2,3,4,5")
    args = p.parse_args()

    print(f"[gate] edge_geometry: point (waymax) vs segment  "
          f"default={SafetyLossConfig().edge_geometry}\n")
    rg = straight_road()                       # edges at y = +-4, 1.0 m spacing
    ok = gate0_waymax_parity()
    ok &= gate1_same_boundary(rg)
    # x0 sits BETWEEN two samples (they fall on the 1.0 m grid from -50), which
    # is the configuration that produces the cliff; crossing at a sample would
    # hide it.
    ok &= gate2_continuity(rg, x0=20.5, y_from=0.0, y_to=9.0)
    ok &= gate3_gradient_at_crossing(rg)

    if args.tfrecord:
        print()
        ok &= real_scene_gates(args.tfrecord,
                               [int(i) for i in args.indices.split(",") if i.strip()])

    print(f"\n[gate] {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
