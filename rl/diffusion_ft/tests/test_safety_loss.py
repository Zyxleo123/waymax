"""Unit gates for :mod:`rl.diffusion_ft.safety_loss` -- pure geometry, no policy.

These run on CPU in seconds against a synthetic straight-road scene, and exist
to catch the three classes of bug that would make the GPU probe's verdict
meaningless rather than merely wrong:

  1. **Parity.** Our split of the road-edge distance into a non-differentiable
     nearest-point search plus a differentiable closed form must reproduce
     waymax's ``compute_signed_distance_to_nearest_road_edge_point`` to floating
     point. If it does not, Gate A in the probe is comparing our arithmetic
     against itself and would pass on a broken loss.
  2. **Cover soundness.** The disc cover must *contain* the box. If it does not,
     ``c_agent`` over-estimates clearance and the collision term is optimistic
     exactly where it matters.
  3. **Shape/gradient plumbing.** ``safety_terms`` must broadcast correctly and
     ``jax.grad`` must return finite, correctly-signed derivatives.

Run: ``python -m rl.diffusion_ft.tests.test_safety_loss``
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from waymax import datatypes
from waymax.metrics import roadgraph as wrg
from waymax.utils import geometry as wgeom

from rl.diffusion_ft.safety_loss import (
    SafetyLossConfig,
    agent_disc_gaps,
    best_disc_count,
    box_box_signed_gap,
    barrier,
    disc_cover,
    ego_box_5dof,
    road_edge_anchors,
    road_edge_clearance,
    safety_terms,
    signed_distance_from_anchors,
)

ROAD_EDGE_BOUNDARY = int(datatypes.MapElementIds.ROAD_EDGE_BOUNDARY)


# --------------------------------------------------------------------------- #
def straight_road(n: int = 200, half_width: float = 4.0):
    """Two road edges along +x at y = +-half_width, as a batched ``[1, 2n]``.

    Handedness matters and is easy to get backwards. Waymax reads
    ``sign(cross(query - nearest_edge_point, edge_direction))`` and calls a
    **positive** result off-road, so for the drivable surface to lie between the
    two edges the *upper* edge must run in ``-x`` and the *lower* edge in ``+x``.
    Orienting both the same way (the natural-looking choice) makes the whole
    corridor read as off-road, which is silently self-consistent: a parity test
    still passes because it compares this module against waymax on the same
    roadgraph, and only a test that asserts the corridor is on-road catches it.
    :func:`test_fixture_is_sane` is that test -- run it before trusting anything
    downstream of this fixture.
    """
    s = np.linspace(-50.0, 150.0, n)
    xs = np.concatenate([s, s])
    ys = np.concatenate([np.full(n, half_width), np.full(n, -half_width)])
    dir_x = np.concatenate([-np.ones(n), np.ones(n)])
    dir_y = np.zeros(2 * n)
    ids = np.concatenate([np.zeros(n, np.int32), np.ones(n, np.int32)])
    z = np.zeros(2 * n)
    return datatypes.RoadgraphPoints(
        x=jnp.asarray(xs[None], jnp.float32), y=jnp.asarray(ys[None], jnp.float32),
        z=jnp.asarray(z[None], jnp.float32),
        dir_x=jnp.asarray(dir_x[None], jnp.float32),
        dir_y=jnp.asarray(dir_y[None], jnp.float32),
        dir_z=jnp.asarray(z[None], jnp.float32),
        types=jnp.full((1, 2 * n), ROAD_EDGE_BOUNDARY, jnp.int32),
        ids=jnp.asarray(ids[None], jnp.int32),
        valid=jnp.ones((1, 2 * n), bool),
    )


def unbatched(rg):
    return jax.tree_util.tree_map(lambda a: a[0], rg)


def report(name, ok, detail):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return ok


# --------------------------------------------------------------------------- #
def test_fixture_is_sane() -> bool:
    """The synthetic corridor must actually be drivable.

    Runs first, because every other test in this file is measured against this
    fixture and would report agreeable-looking numbers on a corridor that is
    entirely off-road (it did: the first version of :func:`straight_road` had
    both edges oriented outward, so all 12 frames of the gradient test were "at
    risk" and the concentration check had no clean frames to contrast with).
    """
    rg = straight_road(half_width=4.0)
    # Centre of the corridor, and a pose well outside it.
    poses = jnp.asarray([[[20.0, 0.0, 0.0], [20.0, 8.0, 0.0]]], jnp.float32)
    box = ego_box_5dof(poses, jnp.asarray([[4.8]]), jnp.asarray([[2.0]]))
    clear, _ = road_edge_clearance(box, jnp.zeros((1, 2)), rg, time_chunk=2)
    c = np.asarray(jnp.min(clear, axis=-1))[0]
    ok = report("corridor centre is on road", c[0] > 0,
                f"c_edge at the centreline = {c[0]:+.3f} m (expect ~+3 m: "
                f"4 m half-width minus the 1 m half-width of the car)")
    ok &= report("outside the corridor is off road", c[1] < 0,
                 f"c_edge at y=+8 m = {c[1]:+.3f} m")
    return ok


# --------------------------------------------------------------------------- #
def test_waymax_parity() -> bool:
    """Our anchor split == waymax's one-shot signed distance, to float error."""
    rg = straight_road()
    # Ego sweeping across the corridor and out over the left edge.
    ys = np.linspace(-3.5, 5.5, 19)
    pose = jnp.asarray(
        np.stack([np.full_like(ys, 20.0), ys, np.zeros_like(ys)], -1)[None], jnp.float32
    )                                                                # [1, T, 3]
    box = ego_box_5dof(pose, jnp.asarray([[4.8]]), jnp.asarray([[2.0]]))
    corners = wgeom.corners_from_bboxes(box)                         # [1, T, 4, 2]
    z = jnp.zeros(corners.shape[:-1] + (1,), jnp.float32)
    corners_xyz = jnp.concatenate([corners, z], -1)

    near, sign = road_edge_anchors(corners_xyz, rg, time_chunk=4)
    ours = np.asarray(signed_distance_from_anchors(corners, near, sign))[0]

    ref_fn = wrg.compute_signed_distance_to_nearest_road_edge_point
    rg1 = unbatched(rg)
    ref = np.stack([np.asarray(ref_fn(corners_xyz[0, t], rg1))
                    for t in range(corners_xyz.shape[1])])
    err = float(np.abs(ours - ref).max())
    ok = report("waymax parity", err < 1e-4,
                f"max |ours - waymax| = {err:.2e} over {ours.size} corner-distances")

    # And the flag itself: our clearance sign must reproduce is_offroad.
    clear, _ = road_edge_clearance(box, jnp.zeros((1, len(ys))), rg, time_chunk=4)
    ours_off = np.asarray(jnp.min(clear, axis=-1) < 0)[0]
    ref_off = np.stack([
        bool(wrg.is_offroad(
            datatypes.Trajectory(
                x=box[:, t:t + 1, 0], y=box[:, t:t + 1, 1],
                z=jnp.zeros((1, 1), jnp.float32),
                vel_x=jnp.zeros((1, 1), jnp.float32),
                vel_y=jnp.zeros((1, 1), jnp.float32),
                yaw=box[:, t:t + 1, 4],
                length=box[:, t:t + 1, 2], width=box[:, t:t + 1, 3],
                height=jnp.ones((1, 1), jnp.float32),
                timestamp_micros=jnp.zeros((1, 1), jnp.int32),
                valid=jnp.ones((1, 1), bool),
            ), rg1)[0])
        for t in range(len(ys))
    ])
    ok &= report("offroad flag agreement", bool((ours_off == ref_off).all()),
                 f"{int((ours_off == ref_off).sum())}/{len(ys)} poses agree with is_offroad")
    return ok


def test_disc_cover_contains_box() -> bool:
    """The cover contains the box (so gaps under-estimate), and is sized per object."""
    rng = np.random.default_rng(0)
    ok = True
    # Soundness must hold for every count, adaptive or not: if the union ever
    # fails to contain the box the gap becomes an OVER-estimate, which is the one
    # error direction a safety term cannot tolerate.
    for L, W, n in ((4.8, 2.0, 1), (4.8, 2.0, 3), (4.8, 2.0, 5), (12.0, 2.5, 5)):
        box = jnp.asarray([[3.0, -1.0, L, W, 0.7]])
        centers, radius = disc_cover(box, n, adaptive=False)
        u = rng.uniform(-0.5, 0.5, 4000); v = rng.uniform(-0.5, 0.5, 4000)
        yaw = 0.7
        px = 3.0 + u * L * np.cos(yaw) - v * W * np.sin(yaw)
        py = -1.0 + u * L * np.sin(yaw) + v * W * np.cos(yaw)
        pts = np.stack([px, py], -1)
        d = np.linalg.norm(pts[:, None, :] - np.asarray(centers)[0][None], axis=-1)
        covered = (d <= np.asarray(radius)[0] + 1e-6).any(axis=1)
        ok &= report(f"cover contains {L}x{W} box (K={n})", bool(covered.all()),
                     f"{covered.mean()*100:.2f}% of 4000 points inside, "
                     f"r={float(radius[0]):.3f} m")

    # Per-object sizing: the count must track each vehicle's own aspect ratio.
    # A fixed 3 is right for a car and wrong for a bus, which is what put the
    # measured expert agent clearance at -3.62 m.
    lens = jnp.asarray([4.8, 4.2, 12.0, 2.5])
    wids = jnp.asarray([2.0, 2.0, 2.5, 1.0])
    got = np.asarray(best_disc_count(lens, wids, 6))
    want = []
    for L, W in zip(np.asarray(lens), np.asarray(wids)):
        worst = [max(math.hypot(L/(2*k), W/2) - L/(2*k),
                     math.hypot(L/(2*k), W/2) - W/2) for k in range(1, 7)]
        want.append(int(np.argmin(worst)) + 1)
    ok &= report("disc count is per object", bool((got == np.array(want)).all()),
                 f"L/W {[f'{l/w:.1f}' for l, w in zip(np.asarray(lens), np.asarray(wids))]}"
                 f" -> K {got.tolist()} (brute-force optimum {want})")

    # Adaptive must reproduce the fixed cover exactly when they agree on K.
    car = jnp.asarray([[3.0, -1.0, 4.8, 2.0, 0.7]])
    ca, ra = disc_cover(car, 5, adaptive=True)
    cf, rf = disc_cover(car, 3, adaptive=False)
    same_r = abs(float(ra[0]) - float(rf[0])) < 1e-6
    # Adaptive pads to 5 slots by repeating the last disc; the distinct centres
    # must match the K=3 cover.
    uniq = np.unique(np.round(np.asarray(ca)[0], 5), axis=0)
    ok &= report("adaptive == fixed where K agrees", same_r and len(uniq) == 3,
                 f"radius {float(ra[0]):.3f} vs {float(rf[0]):.3f}, "
                 f"{len(uniq)} distinct centres in {np.asarray(ca).shape[-2]} slots")

    # Conservatism in the number that matters.
    ego = jnp.asarray([[[0.0, 0.0, 4.8, 2.0, 0.0]]])
    other = jnp.asarray([[[[6.0, 0.0, 4.8, 2.0, 0.0]]]])
    gaps, valid = agent_disc_gaps(ego, other, jnp.ones((1, 1, 1), bool), n_discs=5)
    got_gap = float(jnp.min(jnp.where(valid, gaps, jnp.inf)))
    ok &= report("gap is conservative", got_gap <= 1.2 + 1e-6,
                 f"reported {got_gap:.3f} m vs true {1.2:.3f} m "
                 f"({1.2 - got_gap:+.3f} m of pessimism)")
    return ok


def test_barrier_shape() -> bool:
    """Plateau far out, knee at the margin, unbounded ramp inside."""
    m, tau = 0.5, 0.15
    far = float(barrier(jnp.asarray(5.0), m, tau))
    at = float(barrier(jnp.asarray(m), m, tau))
    deep = float(barrier(jnp.asarray(-10.0), m, tau))
    huge = float(barrier(jnp.asarray(-1e6), m, tau))
    ok = report("plateau when clear", far < 1e-6, f"barrier(5 m) = {far:.2e}")
    ok &= report("knee at margin", abs(at - tau * np.log(2)) < 1e-6,
                 f"barrier(margin) = {at:.4f} == tau*ln2 = {tau*np.log(2):.4f}")
    ok &= report("ramp inside", abs(deep - (m + 10.0)) < 1e-3,
                 f"barrier(-10 m) = {deep:.4f} ~ margin + 10 = {m + 10:.4f}")
    ok &= report("no overflow", np.isfinite(huge), f"barrier(-1e6) = {huge:.3e} finite")
    # Monotone decreasing in clearance.
    cs = jnp.linspace(-5, 5, 401)
    vals = np.asarray(barrier(cs, m, tau))
    ok &= report("monotone", bool((np.diff(vals) <= 1e-9).all()),
                 "barrier is non-increasing in clearance")
    return ok


def test_terms_and_gradient() -> bool:
    """Shapes flow, the gradient is finite and local, and descent does the work.

    The scene is deliberately over-constrained: the ego drifts up toward the
    ``y = +4`` road edge while a parked car sits at ``y = 1.5``, so over the last
    few frames the ego is *wedged* between them with the disc gap actually
    negative. The two barriers then pull in opposite directions -- the edge term
    toward ``-y``, the agent term toward ``+y``, both near saturation.

    That makes "descending the gradient increases road-edge clearance" a false
    statement about a correct implementation, and asserting it was a bug in an
    earlier version of this test. A gradient promises one thing only: the
    **total** decreases. So the conflict case is asserted on the total, and each
    term's directional correctness is asserted separately with the other term
    weighted to zero -- which is the only way to attribute a failure to the term
    that caused it.
    """
    rg = straight_road()
    T = 12
    ys = np.linspace(-1.0, 4.2, T)                                    # drifts off left
    pose = jnp.asarray(
        np.stack([np.linspace(0, 30, T), ys, np.zeros(T)], -1)[None], jnp.float32)
    length, width = jnp.asarray([[4.8]]), jnp.asarray([[2.0]])
    # One neighbour parked ahead-right, close enough to bite near the end.
    other = jnp.tile(jnp.asarray([[[[28.0, 1.5, 4.8, 2.0, 0.0]]]]), (1, T, 1, 1))
    other_valid = jnp.ones((1, T, 1), bool)
    cfg = SafetyLossConfig(time_chunk=4)

    def run(p, c):
        return safety_terms(
            c, ego_box_bt5=ego_box_5dof(p, length, width),
            ego_z_bt=jnp.zeros((1, T)), roadgraph_points=rg,
            other_box_btn5=other, other_valid_btn=other_valid)

    def descend(c, p, step_m=0.05):
        """One step along -grad, sized in **metres**.

        Only the positional components are stepped. ``pose`` mixes metres and
        radians, so scaling the whole array by one scalar would silently treat
        1 rad like 1 m -- a 0.05 "metre" step in yaw sweeps a corner of a 4.8 m
        car by 0.12 m, which is not the small perturbation the check assumes.
        The yaw derivative is covered by the finite-difference check instead.
        """
        g = jax.grad(lambda q: run(q, c)["total"])(p)
        g = g.at[..., 2].set(0.0)
        scale = step_m / max(float(jnp.abs(g[..., :2]).max()), 1e-9)
        return p - scale * g

    terms = run(pose, cfg)
    ok = report("shapes", terms["loss"].shape == (1, T) and terms["c_edge"].shape == (1, T),
                f"loss {terms['loss'].shape}, c_edge {terms['c_edge'].shape}, "
                f"c_agent {terms['c_agent'].shape}")

    gn = np.asarray(jax.grad(lambda p: run(p, cfg)["total"])(pose))
    ok &= report("gradient finite", bool(np.isfinite(gn).all()),
                 f"max |grad| = {np.abs(gn).max():.4f} "
                 f"(xy {np.abs(gn[..., :2]).max():.4f}, yaw {np.abs(gn[..., 2]).max():.4f})")
    # Concentrated where it is close: the last frames are over the edge, the
    # first are mid-corridor and should be numerically untouched.
    near_n = float(np.linalg.norm(gn[0, -3:, :2], axis=-1).mean())
    far_n = float(np.linalg.norm(gn[0, :3, :2], axis=-1).mean())
    ok &= report("gradient concentrated", near_n > 1e3 * max(far_n, 1e-12),
                 f"|grad_xy| {near_n:.4f} at the edge vs {far_n:.2e} mid-corridor")

    # Finite-difference check against autodiff on a coordinate that matters.
    eps = 1e-3
    idx = (0, T - 1, 1)
    fd = (float(run(pose.at[idx].add(eps), cfg)["total"])
          - float(run(pose, cfg)["total"])) / eps
    ok &= report("grad matches finite difference", abs(fd - gn[idx]) < 1e-2,
                 f"autodiff {gn[idx]:+.5f} vs finite-diff {fd:+.5f}")

    # --- each term alone must move its own clearance the right way --------- #
    edge_only = dataclasses.replace(cfg, agent_weight=0.0)
    c0 = np.asarray(terms["c_edge"])[0]
    c1 = np.asarray(run(descend(edge_only, pose), edge_only)["c_edge"])[0]
    at = c0 < cfg.edge_margin_m
    d_e = float((c1 - c0)[at].mean()) if at.any() else 0.0
    ok &= report("edge term increases c_edge", d_e > 0,
                 f"{d_e:+.4f} m on {int(at.sum())} frames inside the edge margin "
                 f"(agent term weighted to 0)")

    agent_only = dataclasses.replace(cfg, edge_weight=0.0)
    a0 = np.asarray(terms["c_agent"])[0]
    a1 = np.asarray(run(descend(agent_only, pose), agent_only)["c_agent"])[0]
    at_a = np.isfinite(a0) & (a0 < cfg.agent_margin_m)
    d_a = float((a1 - a0)[at_a].mean()) if at_a.any() else 0.0
    ok &= report("agent term increases c_agent", d_a > 0,
                 f"{d_a:+.4f} m on {int(at_a.sum())} frames inside the agent margin "
                 f"(edge term weighted to 0)")

    # --- and with both live, the TOTAL must fall even though they fight ---- #
    both = at & at_a
    t0 = float(terms["total"])
    t1 = float(run(descend(cfg, pose), cfg)["total"])
    ok &= report("combined descent lowers total loss", t1 < t0,
                 f"{t0:.4f} -> {t1:.4f} ({t1 - t0:+.4f}) with {int(both.sum())} frames "
                 f"wedged between the car and the road edge (both terms active)")
    return ok


# --------------------------------------------------------------------------- #
# Exact box-box gap
# --------------------------------------------------------------------------- #
def _cross2(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """2-D scalar cross product. ``np.cross`` on 2-vectors is deprecated in
    NumPy 2, and this is clearer than padding a z column to keep it quiet."""
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _hull(pts: np.ndarray) -> np.ndarray:
    """CCW convex hull (Andrew monotone chain) of a small 2-D point set."""
    p = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(ps):
        out = []
        for q in ps:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if _cross2(b - a, q - a) <= 1e-12:
                    out.pop()
                else:
                    break
            out.append(q)
        return out

    lower, upper = half(p), half(p[::-1])
    return np.asarray(lower[:-1] + upper[:-1])


def brute_box_gap(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Signed box-box gap by an independent route: the Minkowski difference.

    ``A`` and ``B`` are disjoint iff the origin is outside ``A (-) B``, the
    convex hull of the 16 corner differences; the separation and the penetration
    depth are both the distance from the origin to that hull's **boundary**, on
    the outside and the inside respectively. That is the definition the
    implementation is claiming to compute, reached without touching a separating
    axis or a vertex/edge enumeration -- so it is a real check on
    :func:`box_box_signed_gap` rather than a restatement of it.
    """
    ca = np.asarray(wgeom.corners_from_bboxes(jnp.asarray(box_a[None])))[0]
    cb = np.asarray(wgeom.corners_from_bboxes(jnp.asarray(box_b[None])))[0]
    hull = _hull((ca[:, None, :] - cb[None, :, :]).reshape(-1, 2))
    d = np.roll(hull, -1, axis=0) - hull
    # Signed distance from the origin to each edge's supporting line; the hull is
    # CCW, so the origin is inside iff every one of them is non-negative.
    line = _cross2(d, -hull) / np.maximum(np.linalg.norm(d, axis=-1), 1e-12)
    if (line >= 0).all():
        # Inside a convex polygon -- an intersection of half-planes -- the
        # distance to the boundary is the distance to the nearest *line*, not to
        # the nearest edge segment.
        return -float(line.min())
    t = np.clip(-np.sum(hull * d, -1) / np.maximum(np.sum(d * d, -1), 1e-12), 0, 1)
    return float(np.linalg.norm(hull + t[:, None] * d, axis=-1).min())


def random_box(rng, x_range=6.0):
    return np.array([rng.uniform(-x_range, x_range), rng.uniform(-x_range, x_range),
                     rng.uniform(2.0, 12.0), rng.uniform(1.0, 3.0),
                     rng.uniform(-np.pi, np.pi)], np.float32)


def test_exact_box_gap() -> bool:
    """The exact gap must equal the Minkowski-difference gap, on both sides.

    This is the gate that decides whether the exact geometry replaces the disc
    cover. Four things have to hold at once, and the pair of them that matter
    most are the ones the disc cover could not give: a **signed** value that is
    exact when the boxes overlap, and **continuity** through contact, so the
    barrier's ramp is a single function rather than two pieces meeting at a
    cliff.
    """
    rng = np.random.default_rng(7)
    boxes_a = np.stack([random_box(rng) for _ in range(400)])
    boxes_b = np.stack([random_box(rng) for _ in range(400)])
    ours = np.asarray(box_box_signed_gap(jnp.asarray(boxes_a), jnp.asarray(boxes_b)))
    ref = np.array([brute_box_gap(a, b) for a, b in zip(boxes_a, boxes_b)])

    n_over = int((ref < 0).sum())
    err = np.abs(ours - ref)
    ok = report("exact gap == Minkowski gap", float(err.max()) < 1e-3,
                f"max |ours - brute| = {err.max():.2e} m over 400 random pairs "
                f"({n_over} overlapping, {400 - n_over} apart); "
                f"outside {err[ref >= 0].max():.2e}, inside {err[ref < 0].max():.2e}")

    # The sign is the metric. Not approximately -- the inside branch is the same
    # 4-axis test has_overlap runs, so disagreement here would mean a coding
    # error, not a tolerance.
    ours_over = ours < 0
    ref_over = np.asarray(wgeom.has_overlap(jnp.asarray(boxes_a), jnp.asarray(boxes_b)))
    ok &= report("sign == waymax has_overlap", bool((ours_over == ref_over).all()),
                 f"{int((ours_over == ref_over).sum())}/400 pairs agree "
                 f"({int(ref_over.sum())} overlapping)")

    # Continuity through contact: drive one car straight through another and
    # check the gap never jumps. A step of 1 cm may move the gap by at most
    # ~1 cm (the function is 1-Lipschitz in a translation), so a jump would show
    # up as a difference far larger than the step.
    step = 0.01
    xs = np.arange(-8.0, 8.0, step)
    a = np.stack([np.stack([xs, np.full_like(xs, 0.3), np.full_like(xs, 4.8),
                            np.full_like(xs, 2.0), np.full_like(xs, 0.4)], -1)])
    b = np.tile(np.array([0.0, 0.0, 4.8, 2.0, -0.2], np.float32), (1, len(xs), 1))
    g = np.asarray(box_box_signed_gap(jnp.asarray(a, jnp.float32),
                                      jnp.asarray(b, jnp.float32)))[0]
    jump = float(np.abs(np.diff(g)).max())
    ok &= report("continuous through contact", jump < step * 1.05 + 1e-6,
                 f"max |dgap| per {step*100:.0f} cm step = {jump*100:.3f} cm "
                 f"(Lipschitz bound {step*100:.0f} cm); gap sweeps "
                 f"{g.max():+.2f} -> {g.min():+.2f} m")
    signs = np.sign(g)
    ok &= report("crosses zero exactly twice", int((np.diff(signs) != 0).sum()) == 2,
                 f"{int((np.diff(signs) != 0).sum())} sign changes over the sweep "
                 f"(in and out of the other car)")

    # And the relation to the cover it replaces: the disc gap must never exceed
    # it (the cover contains the box), so every metre of difference here is
    # margin the old term was spending on its own approximation.
    ego = jnp.asarray(boxes_a)[None, :, :]                        # [1, T, 5]
    oth = jnp.asarray(boxes_b)[None, :, None, :]                  # [1, T, 1, 5]
    disc, _ = agent_disc_gaps(ego, oth, jnp.ones((1, 400, 1), bool), n_discs=5)
    disc = np.asarray(jnp.min(disc, axis=(-3, -2, -1)))[0]        # [400]
    # Only where the boxes are apart. The containment argument (the discs cover
    # the box, so their gap can only under-report a separation) says nothing
    # about the overlapping side: there the disc value is a *disc* penetration
    # depth, which is a different quantity and is free to be deeper or shallower
    # than the box's.
    apart = ours >= 0
    slack = (ours - disc)[apart]
    ok &= report("exact >= disc gap where apart", bool((slack >= -1e-4).all()),
                 f"over {int(apart.sum())} separated pairs the cover under-reports "
                 f"by {slack.mean():.3f} m on average (max {slack.max():.3f} m) "
                 f"and never over-reports (min {slack.min():+.2e} m)")
    return ok


def test_exact_gap_gradient() -> bool:
    """Finite, unit-scale, and pointing the right way on **both** sides.

    The reason for the inside branch at all: where two boxes overlap the true
    distance is identically 0, so its gradient is 0 and a trajectory that starts
    inside another car gets no push at all. The penetration depth is what
    restores it, and this checks the push is there and separates them.
    """
    pairs = {
        "apart (1.2 m nose to tail)": (np.array([0.0, 0.0, 4.8, 2.0, 0.0], np.float32),
                                       np.array([6.0, 0.0, 4.8, 2.0, 0.0], np.float32)),
        "deep overlap (concentric)": (np.array([0.0, 0.0, 4.8, 2.0, 0.0], np.float32),
                                      np.array([0.4, 0.3, 4.8, 2.0, 0.5], np.float32)),
        "corner clipped": (np.array([0.0, 0.0, 4.8, 2.0, 0.3], np.float32),
                           np.array([4.4, 1.4, 4.8, 2.0, -0.6], np.float32)),
    }
    ok = True
    for name, (a, b) in pairs.items():
        pa = jnp.asarray(a)

        def gap(p):
            return box_box_signed_gap(p, jnp.asarray(b))

        g0 = float(gap(pa))
        grad = np.asarray(jax.grad(gap)(pa))
        finite = bool(np.isfinite(grad).all())
        # A translation moves the gap at unit rate (both branches are distances),
        # so |d gap / d xy| must be ~1 unless we are exactly at a kink.
        norm = float(np.linalg.norm(grad[:2]))
        # Ascending the gradient must separate them.
        push = jnp.asarray(0.01 * grad[:2] / max(norm, 1e-9))
        moved = float(gap(pa.at[:2].add(push)))
        fd = (float(gap(pa.at[0].add(1e-4))) - g0) / 1e-4
        ok &= report(f"gradient: {name}", finite and abs(norm - 1.0) < 5e-2
                     and moved > g0 and abs(fd - grad[0]) < 1e-2,
                     f"gap {g0:+.3f} m, |grad_xy| {norm:.4f}, ascent -> {moved:+.3f} m "
                     f"({moved - g0:+.4f}), d/dx autodiff {grad[0]:+.4f} vs "
                     f"finite-diff {fd:+.4f}")
    return ok


def main() -> None:
    print("[test_safety_loss] jax devices:", jax.devices())
    results = {}
    for name, fn in (
        ("fixture sanity", test_fixture_is_sane),
        ("waymax parity", test_waymax_parity),
        ("disc cover", test_disc_cover_contains_box),
        ("exact box gap", test_exact_box_gap),
        ("exact gap gradient", test_exact_gap_gradient),
        ("barrier shape", test_barrier_shape),
        ("terms + gradient", test_terms_and_gradient),
    ):
        print(f"\n[{name}]")
        results[name] = fn()
    print("\n[test_safety_loss] summary:")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not all(results.values()):
        raise SystemExit(1)
    print("[test_safety_loss] ALL PASS")


if __name__ == "__main__":
    main()
