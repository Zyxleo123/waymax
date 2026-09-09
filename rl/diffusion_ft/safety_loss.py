"""A differentiable safety loss over the ego's executed geometry.

The diffusion policy's two dominant failure modes are *geometric*, not strategic:
it clips the road edge while turning sharply (the rear inside corner swings out
over the boundary), and it drives close enough to another vehicle to overlap.
The simulator already scores both, but only as **indicator metrics** --
``offroad`` and ``overlap`` are 0/1 step functions of the same geometry. An
indicator has zero gradient everywhere it is defined, which is why every attempt
so far has had to buy the signal back through the reward (RL), paying for it in
variance and sample count.

This module computes the *underlying continuous quantity* those indicators
threshold, and shapes it into a loss with the profile asked for: flat when the
ego is comfortably clear, rising smoothly as it closes on a hazard, and steep
and unbounded once it is in violation.

Two clearances, both in metres, both positive == safe:

``c_edge``
    For each of the ego's 4 bounding-box corners, the *signed* distance to the
    nearest road edge, negated. This is exactly the quantity waymax's
    ``is_offroad`` thresholds (:func:`waymax.metrics.roadgraph.is_offroad` calls
    ``compute_signed_distance_to_nearest_road_edge_point`` on the same corners
    and asks whether **any** is positive), so ``min_corner(c_edge) < 0`` and
    ``offroad == 1`` agree by construction -- not approximately, identically.
    That identity is what makes the loss trustworthy: its zero level set *is*
    the metric's decision boundary.

``c_agent``
    Signed gap between the ego box and each other agent's box: the exact
    distance between the two rectangles when they are apart, and the exact
    penetration depth (the shortest separating translation) when they overlap,
    so the quantity is continuous through contact and unbounded on the violating
    side. Its sign *is* waymax's ``overlap``: the inside branch runs the same
    4-axis separating-axis test as ``waymax.utils.geometry.has_overlap``, so
    ``c_agent < 0`` and ``overlap == 1`` agree by construction -- the same
    identity the edge term has with ``offroad``. What waymax throws away is the
    depth, and that is exactly what is recovered here. See
    :func:`box_box_signed_gap`.

    The older **disc cover** (``agent_geometry="disc"``) remains selectable,
    and is what every calibrated agent number below was measured under: each
    vehicle box is covered by ``n_discs`` discs laid along its centreline, each
    with the radius of the half-diagonal of the box segment it covers. Because
    the union of those discs *contains* the box, the disc gap is never larger
    than the true box gap -- the approximation is **conservative**, and it can
    only ever report a hazard early, never late. Smoothness is bought there
    with pessimism, in the safe direction, and that pessimism is
    **anisotropic** (see :func:`disc_cover`) -- for a 4.8 x 2.0 m car a disc
    sticks out 0.48 m past the nose and 0.28 m past the flank, so a *pair* of
    cars reads ~0.96 m closer than they are nose-to-tail. The cover is sized
    per object (:func:`best_disc_count`) precisely because that budget depends
    on the vehicle: a fixed count tuned for a car overhangs a bus's flanks by
    1.11 m. ``agent_margin_m`` must be read through whatever budget remains,
    which is why it is set from the expert's measured clearance rather than
    from intuition.

Both feed the same one-sided barrier::

    barrier(c) = tau * softplus((margin - c) / tau)

which is ~0 for ``c >> margin`` (the plateau), has a knee of width ~``tau``
around ``c == margin``, and grows ~``(margin - c)`` once inside -- an unbounded
ramp whose gradient magnitude saturates at the weight rather than vanishing, so
a trajectory that starts deep in violation is still pushed out.

The barrier is applied **per corner** and **per disc pair** and summed, rather
than to an aggregated minimum. That is deliberate and does real work: it is
smooth (no ``max`` kink to differentiate through), and it makes two corners near
the edge cost more than one -- which is precisely the sharp-turn geometry we are
trying to price. Distant corners and distant agents contribute ~0 through the
softplus tail, so the sum is effectively over whatever is currently close.

Gradient structure
------------------
The expensive part of the road-edge distance -- searching 30k roadgraph points
for the nearest one -- is **not differentiated through**, and does not need to
be. The search returns an integer index; ``jnp.sign`` of the half-plane cross
product is piecewise constant. Both have zero derivative almost everywhere, so
holding them fixed is not an approximation, it is the exact a.e. gradient. What
remains differentiable is a closed form over a handful of numbers::

    signed_dist = ||nearest_xy - corner_xy|| * offroad_sign

so :func:`road_edge_anchors` (heavy, non-differentiable, chunked over time to
bound memory) is split from :func:`signed_distance_from_anchors` (trivial,
differentiable). The backward pass therefore carries a ``[B, T, 4, 2]`` residual
instead of a ``[B, T, 4, 30000, 3]`` one.

Everything here is a pure function of arrays. Nothing reads or steps the
simulator, so the same code scores a rollout after the fact (see
``tests/probe_safety_loss.py``) and, later, differentiates a candidate
trajectory during training.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from waymax import datatypes

__all__ = [
    "SafetyLossConfig",
    "barrier",
    "disc_cover",
    "best_disc_count",
    "ego_box_5dof",
    "road_edge_anchors",
    "signed_distance_from_anchors",
    "road_edge_clearance",
    "agent_disc_gaps",
    "box_box_signed_gap",
    "agent_exact_gaps",
    "safety_terms",
    "log_geometry",
    "expert_clean_mask",
]


@dataclass(frozen=True)
class SafetyLossConfig:
    """Geometry and shaping constants for :func:`safety_terms`.

    The margins are where the loss *starts* to bite, not where the violation is,
    and they are **calibrated against the human demonstration** rather than
    chosen for how safe they sound. ``tests/calibrate_expert.py`` scores the
    logged expert through this same loss; over 16,880 expert-clean steps of the
    218-scene failure set the human's own road-edge clearance runs::

        p0.1 +0.23   p1 +0.31   p5 +0.40   p10 +0.49   p25 +0.70   p50 +1.89

    The original 0.5 m edge margin therefore charged **35.9%** of expert steps --
    it sat exactly on the human p10. Since the safety term is trained alongside
    something that pulls the policy *toward* the expert (a BC/denoising anchor,
    an expert minibatch, a KL to the base policy), charging the demonstration
    means the two objectives have different minima and the safety weight
    silently trades away imitation quality. The defaults below sit at the
    measured p01 (edge) and p05 (agent) so ~99%/95% of expert steps stay on the
    plateau.

    Note this cannot be driven to zero by the margin alone: the barrier's tail
    reaches ~2.7*tau past the margin, so ``*_tau_m`` is as much of a calibration
    knob as the margin is. Reaching *exactly* zero on the expert needs a
    per-scene margin anchored to that scene's own expert clearance, which is a
    change to what the loss means and is not what these defaults do.

    ``*_tau_m`` sets the width of the knee. It also sets the loss value *at* the
    margin (``tau * log 2``) and the scale over which the gradient turns on, so
    it should be small relative to the margin but not so small that the barrier
    is numerically a hinge -- 0.1-0.3 m is the usable band.
    """

    # --- road edge --- #
    edge_margin_m: float = 0.3          # expert p01; was 0.5 (charged 35.9%)
    edge_tau_m: float = 0.15
    edge_weight: float = 1.0
    # How the corner-to-edge distance is measured once the nearest roadgraph
    # sample is found. "point" is waymax's own definition -- the distance to that
    # SAMPLE -- and it carries a defect the barrier cannot absorb: the offroad
    # sign flips across the line through the sample, but the distance to the
    # sample does not go to zero there, so crossing the polyline midway between
    # two samples flips the sign while the magnitude stays at ~half the spacing.
    # Measured on failures.tfrecord (points are 1.01 m apart), `c_edge` jumps
    # +0.224 -> -0.224 m in one 3 mm step of motion, and the loss with it:
    #
    #     scene 0   jump 0.72 m   edge loss 0.60
    #     scene 1   jump 0.96 m   edge loss 0.75
    #
    # Worse than the cliff, the gradient dies where the barrier should push
    # hardest: at the crossing the offset to the anchor is almost entirely
    # tangential, so |d c_edge / d lateral| measured 0.054 just inside scene 1's
    # crossing against 0.74 a mere 25 cm out (it should be 1).
    #
    # "segment" measures to the polyline SEGMENT incident to that sample instead.
    # The sign logic is untouched, so the zero level set -- and therefore
    # agreement with waymax's `offroad` -- is unchanged; only the magnitude
    # moves, and it now goes to zero continuously at the boundary. On the same
    # sweeps the jump falls to <=0.5 cm (the sweep resolution) and the gradient
    # at the crossing returns to 0.86-1.00.
    #
    # Left at "point" so no calibrated number changes silently: the margins were
    # measured under it, and clearances near an edge move by up to half a sample
    # spacing under "segment". Flip the default only after
    # tests/gate_edge_geometry.py passes and calibrate_expert.py has been re-run.
    edge_geometry: str = "point"
    # --- other agents --- #
    # Expert p05, re-measured under the exact box gap (4,880 expert-clean steps
    # of the failure set, job 39985)::
    #
    #     p0.1 -0.27   p1 +0.17   p5 +0.91   p10 +1.17   p25 +2.38   p50 +4.96
    #
    # The disc cover put those same steps at p0.1 = -3.62 m and p1 = -0.41 m --
    # it called the human demonstration *overlapping* on >1% of clean steps,
    # which no pair of real cars can explain and which was the cover's own
    # overhang. Under exact geometry the negative tail collapses to p0.1 = -0.27
    # m: a handful of genuinely touching log frames, not an artifact.
    #
    # 0.3 was p05 under the cover; p05 under real geometry is 0.91. Note this is
    # not a loosening: at the old setting the barrier bit at a *disc* gap of 0.3,
    # which for a pair of cars is a real box gap of ~1.3 m. Real clearance is a
    # slightly tighter envelope, measured rather than inherited from an
    # approximation, and it keeps ~95% of expert steps on the plateau.
    #
    # Set from **WOMD**, not from the failure set, because that is the corpus the
    # fine-tune trains on and this number is corpus-specific (job 40089, 5,120
    # expert-clean steps)::
    #
    #     p0.1 +0.36   p1 +0.51   p5 +0.76   p10 +0.99   p25 +1.25   p50 +1.90
    #
    # Ordinary WOMD traffic runs *closer* than the failure set does (p50 1.90 vs
    # 4.96 m): those scenes are geometrically hard, not crowded. The road-edge
    # margin needs no such adjustment -- p01 is +0.31 m on both corpora.
    agent_margin_m: float = 0.75
    agent_tau_m: float = 0.25
    agent_weight: float = 1.0
    # How the ego/other gap is measured. "exact" is the signed box-box gap
    # (:func:`box_box_signed_gap`): true separation outside, true penetration
    # depth inside, and its sign is waymax's ``overlap`` flag term for term.
    # "disc" is the older conservative disc cover, kept because every calibrated
    # number below (and in tests/calibrate_expert.py) was measured under it --
    # switching geometry changes what a margin *means*, so the two are selectable
    # rather than one silently replacing the other. Under "exact" the margin is
    # no longer absorbing ~1 m of cover overhang, so it is free to be larger;
    # re-run calibrate_expert.py before trusting `agent_margin_m` here.
    agent_geometry: str = "exact"
    # **Cap**, not a count, when `adaptive_discs` is on: each object is covered
    # with its own :func:`best_disc_count`, so a car gets 3 and a bus gets 5. The
    # cap has to clear the longest vehicles present (bus L/W ~= 4.8), and costs
    # cap^2 in the pair count.
    n_discs: int = 5
    # Size the cover per object rather than using one count for every vehicle.
    # A fixed 3 is correct for a car and badly wrong for a bus (1.11 m of flank
    # overhang instead of 0.28 m); that is what put the expert's p0.1 agent
    # clearance at -3.62 m, far past anything a pair of cars can explain.
    adaptive_discs: bool = True
    # Altitude stretch used when associating roadgraph points, so an overpass
    # directly above does not capture the query. Matches waymax's default.
    z_stretch: float = 2.0
    # Timesteps per chunk in the roadgraph nearest-point search. Peak memory is
    # ~chunk * 4 * num_rg_points * 3 floats per world; 8 keeps a 30k-point scene
    # under ~12 MB/world.
    time_chunk: int = 8


# --------------------------------------------------------------------------- #
# Shaping
# --------------------------------------------------------------------------- #
def barrier(clearance: jax.Array, margin: float, tau: float) -> jax.Array:
    """One-sided smooth barrier: ~0 when clear, ~``margin - clearance`` when not.

    ``tau * softplus((margin - c) / tau)``. Written through ``jax.nn.softplus``
    (which is the numerically stable ``log1p(exp(-|z|)) + max(z, 0)`` form) so
    that a deep violation -- ``c`` tens of metres negative, which happens on the
    broken records in the failure set -- does not overflow to ``inf`` and poison
    the whole batch's gradient with NaN.
    """
    return tau * jax.nn.softplus((margin - clearance) / tau)


# --------------------------------------------------------------------------- #
# Box geometry
# --------------------------------------------------------------------------- #
def ego_box_5dof(
    xy_yaw: jax.Array, length: jax.Array, width: jax.Array
) -> jax.Array:
    """Assemble ``[..., 5] = (x, y, length, width, yaw)`` from pose + shape.

    ``xy_yaw`` is ``[..., 3]``; ``length``/``width`` broadcast against its prefix.
    Only ``xy_yaw`` is expected to carry gradient -- vehicle extents are fixed.
    """
    x, y, yaw = xy_yaw[..., 0], xy_yaw[..., 1], xy_yaw[..., 2]
    length = jnp.broadcast_to(jnp.asarray(length, jnp.float32), x.shape)
    width = jnp.broadcast_to(jnp.asarray(width, jnp.float32), x.shape)
    return jnp.stack([x, y, length, width, yaw], axis=-1)


def best_disc_count(length: jax.Array, width: jax.Array, max_discs: int) -> jax.Array:
    """Per-object disc count that minimises the cover's worst-case overhang.

    A disc protrudes ``r - L/2K`` past the nose and tail and ``r - W/2`` past the
    flanks, and those move in *opposite* directions with ``K``: more discs hug the
    sides but bulge further fore and aft. They are equal at ``K = L/W``, so the
    worst case is minimised near there -- but the better integer is not always
    ``ceil`` or ``floor`` of it (for a 4.8x2.0 car, K*=2.4 and K=3 wins 0.481 to
    K=2's 0.562; for a 4.2x2.0 car, K*=2.1 and K=2 wins 0.450 to 0.521). So every
    candidate is evaluated and the argmin taken, which is exact and costs nothing
    for the small ``max_discs`` this is used with.

    This is why a fixed count is wrong. ``K = 3`` was chosen for a typical car and
    is right for one, but a 12 x 2.5 m bus has ``L/W = 4.8`` and at K=3 overhangs
    its flanks by **1.11 m** -- so a pair of buses reads about 4.7 m closer than
    they are. Measured on the expert logs, that is exactly where the agent
    clearance went negative (p0.1 = -3.62 m over 16,880 human steps, far past the
    ~0.96 m a pair of *cars* can explain).
    """
    ks = jnp.arange(1, int(max_discs) + 1, dtype=jnp.float32)        # [K]
    # Padded/invalid objects carry zero extents; floor them so the ratio and the
    # sqrt stay finite. A NaN here would survive the validity `where` in the
    # forward pass and still poison the gradient.
    L = jnp.maximum(jnp.asarray(length, jnp.float32), 1e-3)[..., None]
    W = jnp.maximum(jnp.asarray(width, jnp.float32), 1e-3)[..., None]
    r = jnp.sqrt((L / (2.0 * ks)) ** 2 + (W / 2.0) ** 2)             # [..., K]
    worst = jnp.maximum(r - L / (2.0 * ks), r - W / 2.0)
    return jnp.argmin(worst, axis=-1) + 1                            # [...] in 1..max


def disc_cover(
    box_5: jax.Array, n_discs: int, *, adaptive: bool = True
) -> tuple[jax.Array, jax.Array]:
    """Cover each box with discs along its centreline.

    The box is cut into ``K`` equal longitudinal segments; one disc sits at each
    segment's centre with the radius of that segment's half-diagonal,
    ``sqrt((L/2K)^2 + (W/2)^2)``. That radius is what makes the union of discs
    *contain* the rectangle: every point of a segment is within its half-diagonal
    of the segment centre. Using the half-width instead (the tempting choice, and
    the one that makes the discs look right when drawn) would leave the four
    corners uncovered and turn the gap into an over-estimate -- optimistic about
    clearance, which is the one error a safety term must not make.

    With ``adaptive`` (the default) ``n_discs`` is a **cap**, not a count: each
    object gets its own :func:`best_disc_count` for its own aspect ratio, so a car
    and a bus are both covered tightly. The returned array still has ``n_discs``
    slots -- JAX needs a static shape -- and an object using ``K_eff < n_discs``
    simply repeats its last disc. Duplicates are harmless: the gap is a minimum
    over pairs, and a repeated disc contributes a value already present.

    Args:
      box_5: ``[..., 5]`` as ``(x, y, length, width, yaw)``.
      n_discs: cap on discs per box (exact count when ``adaptive=False``), >= 1.
      adaptive: size the cover per object rather than using ``n_discs`` for all.

    Returns:
      ``(centers, radius)`` of shapes ``[..., n_discs, 2]`` and ``[...]``.
    """
    if n_discs < 1:
        raise ValueError(f"n_discs must be >= 1, got {n_discs}")
    x, y = box_5[..., 0], box_5[..., 1]
    length, width, yaw = box_5[..., 2], box_5[..., 3], box_5[..., 4]

    j = jnp.arange(n_discs, dtype=jnp.float32)                       # [K] slots
    if adaptive:
        k_eff = best_disc_count(length, width, n_discs).astype(jnp.float32)
        k_eff_e = k_eff[..., None]
        seg = jnp.minimum(j, k_eff_e - 1.0)      # spare slots repeat the last disc
    else:
        k_eff_e = jnp.full(length.shape + (1,), float(n_discs))
        seg = jnp.broadcast_to(j, length.shape + (n_discs,))

    Lp = jnp.maximum(length, 1e-3)[..., None]
    s = -Lp / 2.0 + (seg + 0.5) * (Lp / k_eff_e)
    cos_y, sin_y = jnp.cos(yaw)[..., None], jnp.sin(yaw)[..., None]
    centers = jnp.stack([x[..., None] + s * cos_y, y[..., None] + s * sin_y], axis=-1)
    radius = jnp.sqrt(
        (jnp.maximum(length, 1e-3) / (2.0 * k_eff_e[..., 0])) ** 2 + (width / 2.0) ** 2
    )
    return centers, radius


def _anchors_one_frame(
    query_xyz: jax.Array,
    rg_xyz: jax.Array,
    rg_dir_xy: jax.Array,
    rg_ids: jax.Array,
    rg_keep: jax.Array,
    z_stretch: float,
) -> tuple[jax.Array, jax.Array]:
    """Nearest road-edge point and off-road sign for ``[Q, 3]`` queries.

    Reproduces ``waymax.metrics.roadgraph
    .compute_signed_distance_to_nearest_road_edge_point`` term for term, but
    returns the *anchor* (nearest point, sign) instead of the finished distance
    so the caller can form the distance differentiably. ``rg_keep`` is the
    precomputed ``valid & is_road_edge`` mask.
    """
    # [Q, P, 3] -> stretch z so an overpass above does not win the argmin.
    diffs = rg_xyz - query_xyz[:, None, :]
    diffs = diffs * jnp.array([[[1.0, 1.0, z_stretch]]], dtype=jnp.float32)
    sq = jnp.sum(diffs**2, axis=-1)                                  # [Q, P]
    sq = jnp.where(rg_keep[None, :], sq, jnp.inf)
    nearest = jnp.argmin(sq, axis=-1)                                # [Q]
    prior = jnp.maximum(nearest - 1, 0)

    nearest_xy = rg_xyz[nearest, :2]                                 # [Q, 2]
    nearest_dir = rg_dir_xy[nearest]                                 # [Q, 2]
    prior_dir = rg_dir_xy[prior]

    to_edge = query_xyz[:, :2] - nearest_xy
    cross = jnp.cross(to_edge, nearest_dir)
    cross_prior = jnp.cross(to_edge, prior_dir)
    same_curve = jnp.equal(rg_ids[nearest], rg_ids[prior])
    sign = jnp.sign(
        jnp.where(same_curve & (cross_prior < cross), cross_prior, cross)
    )

    # The two polyline segments incident to the nearest sample, for the
    # "segment" geometry. A neighbour forms a segment only if it is a real point
    # on the SAME curve -- across a curve boundary the two samples are unrelated
    # and the chord between them is not road edge. `prior == nearest` at index 0
    # (the clamp above) would be a degenerate zero-length segment, so it is
    # excluded too. All of this is gathered by index and carries no gradient.
    nxt = jnp.minimum(nearest + 1, rg_xyz.shape[0] - 1)
    prev_ok = same_curve & rg_keep[prior] & (prior != nearest)
    next_ok = jnp.equal(rg_ids[nxt], rg_ids[nearest]) & rg_keep[nxt] & (nxt != nearest)
    return nearest_xy, sign, rg_xyz[prior, :2], rg_xyz[nxt, :2], prev_ok, next_ok


def road_edge_anchors(
    corners_xyz: jax.Array,
    roadgraph_points: datatypes.RoadgraphPoints,
    *,
    z_stretch: float = 2.0,
    time_chunk: int = 8,
    with_segments: bool = False,
) -> tuple[jax.Array, ...]:
    """Nearest road-edge point + off-road sign for every corner of every frame.

    This is the whole cost of the road-edge term and **carries no gradient**: the
    outputs are a gather index's result and a ``sign``, both piecewise constant
    in the query. It is wrapped in ``stop_gradient`` so XLA cannot be tempted to
    keep the ``[chunk, 4, P, 3]`` difference tensor alive for a backward pass
    that would be zero anyway.

    Args:
      corners_xyz: ``[B, T, 4, 3]`` bbox corners with the ego's z appended.
      roadgraph_points: batched ``[B, P]`` roadgraph.
      z_stretch: altitude tolerance, as in waymax.
      time_chunk: timesteps searched per iteration (memory knob only).

      with_segments: also return the polyline segment incident to each anchor,
        which the ``"segment"`` edge geometry measures to.

    Returns:
      ``(nearest_xy, offroad_sign)`` of shapes ``[B, T, 4, 2]`` and ``[B, T, 4]``,
      followed -- when ``with_segments`` -- by ``(prev_xy, next_xy, prev_ok,
      next_ok)``: the neighbouring samples and whether each forms a real
      same-curve segment with the anchor.
    """
    corners_xyz = jax.lax.stop_gradient(jnp.asarray(corners_xyz, jnp.float32))
    bsz, n_t, n_corner, _ = corners_xyz.shape
    chunk = max(1, int(time_chunk))
    pad = (-n_t) % chunk
    if pad:
        corners_xyz = jnp.concatenate(
            [corners_xyz, jnp.repeat(corners_xyz[:, -1:], pad, axis=1)], axis=1
        )
    n_pad = n_t + pad

    rg_xyz = jnp.asarray(roadgraph_points.xyz, jnp.float32)          # [B, P, 3]
    rg_dir = jnp.asarray(roadgraph_points.dir_xyz, jnp.float32)[..., :2]
    rg_ids = jnp.asarray(roadgraph_points.ids)
    rg_keep = jnp.asarray(roadgraph_points.valid) & datatypes.is_road_edge(
        jnp.asarray(roadgraph_points.types)
    )

    def per_world(corners_t43, xyz, dir_xy, ids, keep):
        blocks = corners_t43.reshape(n_pad // chunk, chunk * n_corner, 3)

        def one_block(q):
            return _anchors_one_frame(q, xyz, dir_xy, ids, keep, z_stretch)

        near, sign, prev_xy, next_xy, pok, nok = jax.lax.map(one_block, blocks)
        xy2 = lambda a: a.reshape(n_pad, n_corner, 2)
        s1 = lambda a: a.reshape(n_pad, n_corner)
        return (xy2(near), s1(sign), xy2(prev_xy), xy2(next_xy), s1(pok), s1(nok))

    out = jax.vmap(per_world)(corners_xyz, rg_xyz, rg_dir, rg_ids, rg_keep)
    out = tuple(jax.lax.stop_gradient(a[:, :n_t]) for a in out)
    return out if with_segments else out[:2]


def _edge_distance_and_foot(
    corners_xy: jax.Array, nearest_xy: jax.Array, segments=None
) -> tuple[jax.Array, jax.Array]:
    """Unsigned distance to the road edge, and the point on it that realises it.

    With ``segments is None`` this is the distance to the nearest *sample*, which
    is what waymax measures. With segments it is the distance to the nearer of
    the two polyline segments incident to that sample -- the closest point on a
    segment is never farther than its endpoint, so taking the minimum with the
    sample distance also supplies the fallback for an anchor with no valid
    neighbour (a curve of one point, or both neighbours on other curves).

    The ``1e6`` stand-in for an invalid segment is a finite number rather than
    ``inf`` on purpose: it loses the ``min`` exactly as ``inf`` would, but a NaN
    cannot be manufactured from it on the backward pass.
    """
    delta = jnp.asarray(corners_xy, jnp.float32) - nearest_xy
    # Distances are compared against margins ~0.5 m, so the 1e-12 floor is far
    # below anything meaningful; it exists only so the norm's gradient stays
    # finite if a corner lands exactly on a roadgraph point.
    d_pt = jnp.sqrt(jnp.sum(delta**2, axis=-1) + 1e-12)
    if segments is None:
        return d_pt, nearest_xy

    prev_xy, next_xy, prev_ok, next_ok = segments
    big = jnp.float32(1e6)
    d_prev = jnp.where(
        prev_ok, _point_segment_distance(corners_xy, prev_xy, nearest_xy), big)
    d_next = jnp.where(
        next_ok, _point_segment_distance(corners_xy, nearest_xy, next_xy), big)
    dist = jnp.minimum(jnp.minimum(d_prev, d_next), d_pt)
    # The foot is a diagnostic (the probe and the drag page draw it), so it is
    # selected rather than differentiated through.
    foot = jnp.where(
        (d_prev <= d_next)[..., None] & (d_prev <= d_pt)[..., None],
        _segment_foot(corners_xy, prev_xy, nearest_xy),
        jnp.where((d_next <= d_pt)[..., None],
                  _segment_foot(corners_xy, nearest_xy, next_xy), nearest_xy))
    return dist, jax.lax.stop_gradient(foot)


def signed_distance_from_anchors(
    corners_xy: jax.Array, nearest_xy: jax.Array, offroad_sign: jax.Array,
    segments=None,
) -> jax.Array:
    """``distance-to-edge * sign`` -- the differentiable half of the metric.

    Positive == off road, matching waymax's convention. This is the only place
    the ego pose enters the road-edge term, and its gradient is the unit vector
    from the edge back to the corner, scaled by the sign: descending it pushes an
    on-road corner *away* from the edge and an off-road corner back *across* it,
    which is correct in both regimes.

    ``segments`` selects which distance. ``None`` reproduces waymax exactly (to
    the nearest sample); passing the anchors' incident segments measures to the
    polyline itself, which changes no sign and therefore no zero level set, but
    makes the magnitude continuous through the boundary. See
    ``SafetyLossConfig.edge_geometry`` for what that is worth in metres.
    """
    dist, _ = _edge_distance_and_foot(corners_xy, nearest_xy, segments)
    return dist * offroad_sign


def road_edge_clearance(
    ego_box_bt5: jax.Array,
    ego_z_bt: jax.Array,
    roadgraph_points: datatypes.RoadgraphPoints,
    *,
    z_stretch: float = 2.0,
    time_chunk: int = 8,
    edge_geometry: str = "point",
) -> tuple[jax.Array, jax.Array]:
    """Per-corner clearance to the road edge, ``[B, T, 4]`` (positive == on road).

    Returns ``(clearance, anchor_xy)``; the anchor comes back too because the
    probe and the drag page draw it and use it to check the gradient's direction.
    Under ``edge_geometry="segment"`` the anchor is the foot on the polyline, not
    the sample -- drawing the sample there would show a line that is not the one
    being measured.
    """
    from waymax.utils import geometry as wgeom

    if edge_geometry not in ("point", "segment"):
        raise ValueError(
            f"edge_geometry must be 'point' or 'segment', got {edge_geometry!r}")
    corners_xy = wgeom.corners_from_bboxes(ego_box_bt5)              # [B, T, 4, 2]
    z = jnp.broadcast_to(
        jnp.asarray(ego_z_bt, jnp.float32)[..., None, None], corners_xy.shape[:-1] + (1,)
    )
    corners_xyz = jnp.concatenate([corners_xy, z], axis=-1)
    want_seg = edge_geometry == "segment"
    anchors = road_edge_anchors(
        corners_xyz, roadgraph_points, z_stretch=z_stretch, time_chunk=time_chunk,
        with_segments=want_seg,
    )
    nearest_xy, sign = anchors[0], anchors[1]
    segments = anchors[2:] if want_seg else None
    dist, foot = _edge_distance_and_foot(corners_xy, nearest_xy, segments)
    return -(dist * sign), foot


# --------------------------------------------------------------------------- #
# Other agents
# --------------------------------------------------------------------------- #
def agent_disc_gaps(
    ego_box_bt5: jax.Array,
    other_box_btn5: jax.Array,
    other_valid_btn: jax.Array,
    *,
    n_discs: int = 5,
    adaptive: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Disc-pair gaps between the ego and every other agent.

    Args:
      ego_box_bt5: ``[B, T, 5]`` ego box (the only differentiable input).
      other_box_btn5: ``[B, T, N, 5]`` other agents' boxes.
      other_valid_btn: ``[B, T, N]`` validity; the ego's own row must already be
        masked out by the caller.
      n_discs: discs per box.

    Returns:
      ``(gaps, valid)`` of shape ``[B, T, N, K, K]`` and ``[B, T, N, 1, 1]``.
      A gap is ``centre distance - r_ego - r_other``: positive is a real
      separation in metres, negative is disc penetration depth.
    """
    ego_c, ego_r = disc_cover(ego_box_bt5, n_discs, adaptive=adaptive)
    oth_c, oth_r = disc_cover(other_box_btn5, n_discs, adaptive=adaptive)

    # [B, T, 1, K, 1, 2] vs [B, T, N, 1, K, 2] -> [B, T, N, K, K]
    delta = ego_c[:, :, None, :, None, :] - oth_c[:, :, :, None, :, :]
    dist = jnp.sqrt(jnp.sum(delta**2, axis=-1) + 1e-12)
    gaps = dist - ego_r[:, :, None, None, None] - oth_r[:, :, :, None, None]
    return gaps, jnp.asarray(other_valid_btn, bool)[:, :, :, None, None]


# --------------------------------------------------------------------------- #
# Exact box-box gap
# --------------------------------------------------------------------------- #
# The disc cover buys smoothness with pessimism (~0.96 m for a pair of cars,
# more for a bus). The pessimism is in the safe direction, but it is *large
# relative to the margin* -- at `agent_margin_m = 0.3` the cover's own overhang
# is three times the margin, so what the barrier actually prices is mostly the
# approximation, not the traffic. The exact gap removes that budget entirely and
# lets the margin mean what it says.
#
# Two formulas, one branch, because a rectangle pair has two regimes:
#
#   disjoint   the gap is the distance between two convex polygons, which is
#              attained at a vertex of one and an edge (possibly a vertex) of
#              the other -- so a min over the 32 vertex/edge pairs is exact, not
#              a bound. No intersection test is needed: the clamped projection
#              is defined everywhere.
#   overlapping the distance is 0 -- flat, no gradient, no magnitude, exactly
#              where the policy is most wrong. So the inside branch reports
#              *penetration depth* instead: the length of the shortest
#              translation that separates the boxes. In 2D that translation is
#              always along a face normal of one of the two convex polygons, so
#              the min over the 4 SAT axis overlaps is again exact.
#
# Both branches go to 0 at contact, so the combined function is continuous,
# a.e. differentiable, signed, and unbounded on the violating side -- which is
# the whole property the barrier needs and the one the disc cover was standing
# in for.
#
# The branch itself is *waymax's own overlap predicate*: `has_overlap` tests the
# sign of these same 4 axis projections and returns `all(> 0)`, and `pen > 0`
# here has the same sign by construction (the magnitudes differ -- see
# :func:`_sat_penetration`), so it is that flag. `gap < 0` and `overlap == 1`
# therefore agree by construction, exactly as `c_edge < 0` and `offroad == 1`
# do -- the zero level set of this term *is* the metric's decision boundary, on
# both sides now.
def _point_segment_distance(p: jax.Array, a: jax.Array, b: jax.Array) -> jax.Array:
    """Distance from point ``p`` to segment ``ab``, all ``[..., 2]``.

    The clamp is what makes this a *segment* rather than a line, and it is also
    where the a.e. in "a.e. differentiable" lives: at ``t in {0, 1}`` the closest
    feature switches from the edge's interior to its endpoint and the gradient
    turns a corner. It stays continuous through the switch (the projected point
    is continuous in ``t``), so the value never jumps -- only the direction of
    the derivative does, the same way ``min`` behaves at a tie.
    """
    d = b - a
    denom = jnp.maximum(jnp.sum(d * d, axis=-1), 1e-12)
    t = jnp.clip(jnp.sum((p - a) * d, axis=-1) / denom, 0.0, 1.0)
    delta = p - (a + t[..., None] * d)
    return jnp.sqrt(jnp.sum(delta**2, axis=-1) + 1e-12)


def _segment_foot(p: jax.Array, a: jax.Array, b: jax.Array) -> jax.Array:
    """The point on segment ``ab`` closest to ``p`` -- the same clamp, kept.

    Only ever used for drawing, so it is not required to be differentiable; it is
    written as the projection rather than recovered from the distance so the two
    cannot disagree about which feature is closest.
    """
    d = b - a
    denom = jnp.maximum(jnp.sum(d * d, axis=-1), 1e-12)
    t = jnp.clip(jnp.sum((p - a) * d, axis=-1) / denom, 0.0, 1.0)
    return a + t[..., None] * d


def _polygon_distance(corners_a: jax.Array, corners_b: jax.Array) -> jax.Array:
    """Distance between two convex polygons, ``[..., 4, 2]`` corners each.

    Correct only where they are disjoint (it returns a positive number for
    overlapping boxes, being the distance between their boundaries' nearest
    *non-crossing* features); the caller selects it with the SAT test.
    """
    def vertex_edge(v, poly):
        edge_a = poly
        edge_b = jnp.roll(poly, -1, axis=-2)
        return _point_segment_distance(
            v[..., :, None, :], edge_a[..., None, :, :], edge_b[..., None, :, :]
        )                                                     # [..., 4v, 4e]

    d_ab = jnp.min(vertex_edge(corners_a, corners_b), axis=(-2, -1))
    d_ba = jnp.min(vertex_edge(corners_b, corners_a), axis=(-2, -1))
    return jnp.minimum(d_ab, d_ba)


def _sat_penetration(
    box_a_5: jax.Array, box_b_5: jax.Array,
    corners_a: jax.Array, corners_b: jax.Array,
) -> jax.Array:
    """Min over the 4 face-normal axes of the projection overlap, ``[...]``.

    Positive == overlapping, and the value is then the penetration depth (the
    shortest separating translation). Negative == disjoint, and the value is the
    largest separating-axis clearance, which is a *lower* bound on the true gap
    and is thrown away by the caller in favour of the exact polygon distance.

    The axes are the two boxes' local x and y, which for a rectangle span its
    face normals -- the same four axes ``waymax.utils.geometry.has_overlap``
    projects onto. The per-axis quantity is **not** the same expression, though,
    and the difference is not cosmetic. Waymax needs a predicate, so it takes the
    length of the projections' intersection, ``min(maxA, maxB) - max(minA,
    minB)``. That is the right thing to test the sign of and the wrong thing to
    measure: when one projection is *contained* in the other it returns the
    inner box's own extent, which does not depend on where the box is, so its
    gradient is zero exactly in the deepest-overlap case. What is needed is how
    far the boxes must move to stop overlapping along this axis, ``min(maxA -
    minB, maxB - minA)`` -- the distance from the origin to the near end of the
    projected Minkowski difference. The two agree whenever the projections
    overlap only partially (the usual case, and why this reads as a rounding
    detail until a car is swallowed by a bus), and always share a sign, so
    ``has_overlap`` is reproduced exactly while the magnitude stays a real
    translation. A unit gate covers this: with the intersection-length form, a
    concentric pair reported a 2.0 m depth with a **zero** gradient.
    """
    def axes_of(box):
        c, s = jnp.cos(box[..., 4]), jnp.sin(box[..., 4])
        return jnp.stack([jnp.stack([c, s], axis=-1),
                          jnp.stack([-s, c], axis=-1)], axis=-2)     # [..., 2, 2]

    axes = jnp.concatenate([axes_of(box_a_5), axes_of(box_b_5)], axis=-2)  # [...,4,2]

    def project(corners):
        # [..., 4 corners, 1, 2] * [..., 1, 4 axes, 2] -> [..., 4 corners, 4 axes]
        return jnp.sum(corners[..., :, None, :] * axes[..., None, :, :], axis=-1)

    pa, pb = project(corners_a), project(corners_b)
    max_a, min_a = jnp.max(pa, axis=-2), jnp.min(pa, axis=-2)
    max_b, min_b = jnp.max(pb, axis=-2), jnp.min(pb, axis=-2)
    overlap = jnp.minimum(max_a - min_b, max_b - min_a)              # [..., 4]
    return jnp.min(overlap, axis=-1)


def box_box_signed_gap(box_a_5: jax.Array, box_b_5: jax.Array) -> jax.Array:
    """Signed gap between two oriented boxes, in metres, ``[...]``.

    Positive is the exact separation between the rectangles; negative is the
    exact penetration depth. Zero is contact, from both sides. Inputs are
    ``[..., 5] = (x, y, length, width, yaw)`` and must broadcast against each
    other (the caller adds the pair axis).
    """
    box_a_5, box_b_5 = jnp.broadcast_arrays(
        jnp.asarray(box_a_5, jnp.float32), jnp.asarray(box_b_5, jnp.float32))
    from waymax.utils import geometry as wgeom

    corners_a = wgeom.corners_from_bboxes(box_a_5)                   # [..., 4, 2]
    corners_b = wgeom.corners_from_bboxes(box_b_5)
    pen = _sat_penetration(box_a_5, box_b_5, corners_a, corners_b)
    dist = _polygon_distance(corners_a, corners_b)
    return jnp.where(pen > 0.0, -pen, dist)


def agent_exact_gaps(
    ego_box_bt5: jax.Array,
    other_box_btn5: jax.Array,
    other_valid_btn: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Exact ego/other box gaps, shaped like :func:`agent_disc_gaps`.

    Returns ``(gaps, valid)`` of shape ``[B, T, N, 1, 1]``: the two trailing
    singleton axes stand in for the disc-pair axes so both geometries flow
    through the same reduction in :func:`safety_terms`. There is exactly one
    number per (world, step, agent) here -- the pair loop is gone, which is why
    this is *cheaper* than a 5x5 cover despite the extra arithmetic.
    """
    ego = jnp.asarray(ego_box_bt5, jnp.float32)[:, :, None, :]       # [B, T, 1, 5]
    gaps = box_box_signed_gap(ego, jnp.asarray(other_box_btn5, jnp.float32))
    return (gaps[..., None, None],
            jnp.asarray(other_valid_btn, bool)[:, :, :, None, None])

# --------------------------------------------------------------------------- #
# The loss
# --------------------------------------------------------------------------- #
def safety_terms(
    cfg: SafetyLossConfig,
    *,
    ego_box_bt5: jax.Array,
    ego_z_bt: jax.Array,
    roadgraph_points: datatypes.RoadgraphPoints,
    other_box_btn5: jax.Array,
    other_valid_btn: jax.Array,
    step_valid_bt: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """Both barriers plus the diagnostics needed to validate them.

    ``step_valid_bt`` masks steps that were never executed (a world that
    terminated early keeps being carried by the batched env, and its post-mortem
    poses are not the policy's doing -- including them would score a phantom).

    Returns a dict with, per world-step ``[B, T]``: ``loss`` (the total),
    ``edge_loss`` / ``agent_loss``, and the *hard* clearances ``c_edge`` /
    ``c_agent`` (min over corners / over all valid agent pairs) whose signs are
    what get compared against waymax's ``offroad`` / ``overlap`` flags. Plus
    ``total`` (scalar) and per-world ``[B]`` sums.
    """
    ego_box_bt5 = jnp.asarray(ego_box_bt5, jnp.float32)
    bsz, n_t = ego_box_bt5.shape[0], ego_box_bt5.shape[1]
    if step_valid_bt is None:
        step_mask = jnp.ones((bsz, n_t), jnp.float32)
    else:
        step_mask = jnp.asarray(step_valid_bt, jnp.float32)

    # --- road edge ---------------------------------------------------------- #
    c_edge_corners, nearest_xy = road_edge_clearance(
        ego_box_bt5, ego_z_bt, roadgraph_points,
        z_stretch=cfg.z_stretch, time_chunk=cfg.time_chunk,
        edge_geometry=cfg.edge_geometry,
    )                                                                # [B, T, 4]
    edge_bt = cfg.edge_weight * jnp.sum(
        barrier(c_edge_corners, cfg.edge_margin_m, cfg.edge_tau_m), axis=-1
    )
    edge_bt = edge_bt * step_mask

    # --- other agents ------------------------------------------------------- #
    if cfg.agent_geometry == "exact":
        gaps, pair_valid = agent_exact_gaps(
            ego_box_bt5, other_box_btn5, other_valid_btn)            # [B,T,N,1,1]
    elif cfg.agent_geometry == "disc":
        gaps, pair_valid = agent_disc_gaps(
            ego_box_bt5, other_box_btn5, other_valid_btn,
            n_discs=cfg.n_discs, adaptive=cfg.adaptive_discs,
        )                                                            # [B,T,N,K,K]
    else:
        raise ValueError(
            f"agent_geometry must be 'exact' or 'disc', got {cfg.agent_geometry!r}")
    pair_barrier = barrier(gaps, cfg.agent_margin_m, cfg.agent_tau_m)
    agent_bt = cfg.agent_weight * jnp.sum(
        jnp.where(pair_valid, pair_barrier, 0.0), axis=(-3, -2, -1)
    )
    agent_bt = agent_bt * step_mask

    # --- hard clearances, for agreement with the indicator metrics ---------- #
    # These are minima, not the smooth sums above: they are what the *metric*
    # thresholds, so they are the right thing to check the zero-crossing against.
    c_edge_bt = jnp.min(c_edge_corners, axis=-1)
    c_agent_bt = jnp.min(jnp.where(pair_valid, gaps, jnp.inf), axis=(-3, -2, -1))

    loss_bt = edge_bt + agent_bt
    return {
        "loss": loss_bt,
        "edge_loss": edge_bt,
        "agent_loss": agent_bt,
        "c_edge": c_edge_bt,
        "c_agent": c_agent_bt,
        "c_edge_corners": c_edge_corners,
        "nearest_edge_xy": nearest_xy,
        "loss_b": jnp.sum(loss_bt, axis=-1),
        "edge_loss_b": jnp.sum(edge_bt, axis=-1),
        "agent_loss_b": jnp.sum(agent_bt, axis=-1),
        "total": jnp.sum(loss_bt),
    }


# --------------------------------------------------------------------------- #
# Expert gating
# --------------------------------------------------------------------------- #
def log_geometry(state) -> dict[str, jax.Array]:
    """Ego pose/shape and every other agent's box, read from ``log_trajectory``.

    Everything comes from the log: the ego because it *is* the human
    demonstration, and the other agents because they are log-replayed anyway.
    Nothing here touches ``sim_trajectory``, which on a freshly loaded batch has
    never been written -- so this works on a raw ``load_batch`` result with no
    environment, no policy and no checkpoint.
    """
    log = state.log_trajectory
    ego_b = jnp.argmax(jnp.asarray(state.object_metadata.is_sdc).astype(jnp.int32), axis=1)
    n_obj = jnp.asarray(log.x).shape[1]

    def ego_row(f):
        return jnp.take_along_axis(jnp.asarray(f), ego_b[:, None, None], axis=1)[:, 0]

    def to_btn(f):
        return jnp.transpose(jnp.asarray(f), (0, 2, 1))          # [B,N,T] -> [B,T,N]

    not_ego = ~jax.nn.one_hot(ego_b, n_obj, dtype=bool)          # [B, N]
    return {
        "ego_pose": jnp.stack([ego_row(log.x), ego_row(log.y), ego_row(log.yaw)], axis=-1),
        "ego_z": ego_row(log.z),
        "ego_len": ego_row(log.length)[:, 0],
        "ego_wid": ego_row(log.width)[:, 0],
        "ego_valid": ego_row(log.valid),                          # [B, T]
        "other_box": jnp.stack(
            [to_btn(log.x), to_btn(log.y), to_btn(log.length),
             to_btn(log.width), to_btn(log.yaw)], axis=-1),
        "other_valid": to_btn(log.valid) & not_ego[:, None, :],
    }


def expert_clean_mask(
    cfg: SafetyLossConfig, state, *, init_steps: int = 11
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Which scenes have a human log that never leaves the road.

    A scene whose ground truth is itself off-road is not a scene the policy can
    be asked to solve, and under an **unbounded** barrier it is far worse than
    merely useless: measured over the 218-scene failure set, the 7 dirty scenes
    (3.2%) carry **56.5%** of all expert loss and the single worst carries 36.4%.
    Left in, they would dominate the gradient of any expert-anchored term.

    The fix is to gate them, not to bound the barrier -- unbounded is the right
    shape for the *policy* side, which must keep being pushed out of a deep
    violation however deep it starts.

    ``init_steps`` skips the prefix Waymax replays from the log before any policy
    takes over; nothing there is ever the policy's responsibility.

    Returns ``(clean_b, c_edge_bt, step_valid_bt)``.
    """
    geo = log_geometry(state)
    box = ego_box_5dof(geo["ego_pose"], geo["ego_len"][:, None], geo["ego_wid"][:, None])
    c_edge, _ = road_edge_clearance(
        box, geo["ego_z"], state.roadgraph_points,
        z_stretch=cfg.z_stretch, time_chunk=cfg.time_chunk,
    )
    c_edge_bt = jnp.min(c_edge, axis=-1)                          # [B, T]
    n_t = c_edge_bt.shape[1]
    step_valid = jnp.asarray(geo["ego_valid"], bool) & (
        jnp.arange(n_t)[None, :] >= int(init_steps)
    )
    ever_off = jnp.any((c_edge_bt < 0.0) & step_valid, axis=-1)
    return ~ever_off, c_edge_bt, step_valid
