"""PDM-Closed (Dauner et al., "Parting with Misconceptions about Learning-based Vehicle Motion Planning",
CoRL 2023) on Waymax, ported from tuplan_garage (github.com/autonomousvision/tuplan_garage, main) with the
parameters of its ``pdm_closed_planner.yaml``.

Ported as in tuplan_garage:
  - proposals: IDM at 5 fractions of the speed limit x {centerline, +-1 m offsets} (15), unrolled 4 s at 10 Hz,
    with the leading object searched in the driving corridor every 2nd step (PDMGenerator, BatchIDMPolicy);
  - observation: constant-velocity forecast of the nearest agents within 50 m, red lights on the route as
    obstacles, objects already overlapping ego ignored for the rest of the scenario (PDMObservation);
  - simulation of the proposals with the batch LQR tracker + kinematic bicycle model (PDMSimulator);
  - scoring: at-fault collision, drivable area, driving direction (multiplicative), progress (5), TTC (5),
    comfort (2) (PDMScorer); emergency brake on an at-fault collision within 2 s below 5 m/s (PDMEmergencyBrake).

WOMD has no route, lane polygons, speed limits or intersection polygons. Replacements (``_LaneMap``):
  - route: Dijkstra over lane successors (no lane changes, like PDM's search over the route lanes) from the
    lane ego is in to the lane of the episode goal (ego's last logged pose); if the goal lane is unreachable the
    route ends at the reachable lane closest to the goal. The centerline is extended past the goal lane along
    the straightest successors, since the episode succeeds by passing within 2 m of the goal, not by stopping
    at the end of the route. As in PDM, the centerline is built once per scenario.
  - speed limit: none in WOMD, so it is inferred as in V-Max's PDM baseline (github.com/valeoai/V-Max,
    ``infer_speed_limit_from_simulator_state``): the expert's top speed rounded up to the next San Francisco /
    Phoenix limit (25, 35, 45, 70 mph), at least 70 mph on freeway lanes and at most 45 mph on other lanes.
  - lane polygons: a point is in a lane when a node of that lane's centerline is within LANE_HALF_WIDTH.
    Drivable area uses the road-edge offroad test of the evaluation metric (``Scorer``), not lane polygons.
  - intersections (TTC rule): a point is in an intersection when lanes within LANE_HALF_WIDTH of it cross
    (headings differ by more than INTERSECTION_HEADING_SPREAD).
  - red lights: a 1 m stop-line box at the stop point of every on-route lane in a red (stop / arrow-stop) state.
  - vehicle geometry: ego's logged length and width, with the Pacifica's rear overhang and wheel base scaled
    to that length.

Output: nuPlan executes PDM's 8 s trajectory with the same LQR tracker and bicycle model; Waymax replays the
plan verbatim, so the planner returns the tracker's rollout of the chosen trajectory (8 s, 10 Hz, box centre).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.signal import savgol_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from planner.abstract_planner import AbstractPlanner, PlannerResult
from scores import helpers as scorer_helpers
from scores.scorer_lane_graph import Scorer

# --- pdm_closed_planner.yaml ---
DT = 0.1
PROPOSAL_STEPS = 40
TRAJECTORY_STEPS = 80
SPEED_LIMIT_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
MIN_GAP_TO_LEAD_AGENT = 1.0
HEADWAY_TIME = 1.5
ACCEL_MAX = 1.5
DECEL_MAX = 3.0
LATERAL_OFFSETS = (-1.0, 1.0)
MAP_RADIUS = 50.0

# --- tuplan_garage constants ---
IDM_ACCELERATION_EXPONENT = 10
LEADING_AGENT_UPDATE_RATE = 2
OBSERVATION_SAMPLE_RES = 2
MAX_DYNAMIC_OBJECTS = {1: 50, 2: 25, 3: 10}  # Waymax vehicle / pedestrian / cyclist
MAX_STATIC_OBJECTS = 50
WEIGHT_PROGRESS, WEIGHT_TTC, WEIGHT_COMFORT = 5.0, 5.0, 2.0
DRIVING_DIRECTION_COMPLIANCE_THRESHOLD = 2.0
DRIVING_DIRECTION_VIOLATION_THRESHOLD = 6.0
STOPPED_SPEED_THRESHOLD = 5e-3  # TTC
COLLISION_STOPPED_SPEED_THRESHOLD = 5e-2  # collision type, stopped track
PROGRESS_DISTANCE_THRESHOLD = 0.1
TTC_FUTURE_STEPS = (0, 3, 6, 9)
EMERGENCY_TIME_TO_INFRACTION = 2.0
EMERGENCY_MAX_EGO_SPEED = 5.0
EMERGENCY_MAX_LONG_ACCEL = 2.40
EMERGENCY_MIN_LONG_ACCEL = -4.05
# nuPlan comfort bounds (pdm_comfort_metrics.py)
MAX_ABS_MAG_JERK = 8.37
MAX_ABS_LAT_ACCEL = 4.89
MAX_LON_ACCEL, MIN_LON_ACCEL = 2.40, -4.05
MAX_ABS_YAW_ACCEL = 1.93
MAX_ABS_LON_JERK = 4.13
MAX_ABS_YAW_RATE = 0.95
# nuPlan Pacifica: length 5.176 m, rear axle 1.127 m from the rear bumper, wheel base 3.089 m
PACIFICA_LENGTH, PACIFICA_REAR_LENGTH, PACIFICA_WHEEL_BASE = 4.049 + 1.127, 1.127, 3.089

# --- WOMD stand-ins (see module docstring) ---
LANE_HALF_WIDTH = 1.75
INTERSECTION_HEADING_SPREAD = np.pi / 4
START_LANE_MAX_HEADING_ERROR = np.pi / 4
ROUTE_EXTENSION_M = 100.0
RED_LIGHT_STATES = (1, 4)  # Waymax ARROW_STOP, STOP
MPH = 1 / 2.237  # m/s
SPEED_LIMITS = np.array([25.0, 35.0, 45.0, 70.0]) * MPH
FREEWAY_SPEED_LIMIT, SURFACE_STREET_SPEED_LIMIT = 70.0 * MPH, 45.0 * MPH
LANE_FREEWAY, LANE_TYPES = 1, (1, 2, 3)  # Waymax roadgraph lane-centre types
CORRIDOR_SAMPLE_M = 0.25
AGENT_TYPES = (1, 2, 3)

# State array layout of tuplan_garage (rear-axle reference).
X, Y, HEADING, VX, VY, AX, AY, STEER, STEER_RATE, YAW_RATE, YAW_ACCEL = range(11)
N_STATE = 11

# Collision types (nuPlan CollisionType).
STOPPED_EGO, STOPPED_TRACK, ACTIVE_FRONT, ACTIVE_REAR, ACTIVE_LATERAL = range(5)


def _wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


@dataclass(frozen=True)
class _Vehicle:
    length: float
    width: float
    rear_length: float
    wheel_base: float

    @property
    def rear_axle_to_center(self) -> float:
        return self.length / 2 - self.rear_length

    @classmethod
    def from_box(cls, length: float, width: float) -> "_Vehicle":
        scale = float(length) / PACIFICA_LENGTH
        return cls(float(length), float(width), PACIFICA_REAR_LENGTH * scale, PACIFICA_WHEEL_BASE * scale)


# ----------------------------------------------------------------------------------------------------------
# Oriented boxes [..., 5] = (x, y, heading, length, width), centre referenced.


def _box_corners(boxes):
    """[..., 4, 2] corners in tuplan_garage's BBCoordsIndex order: front-left, rear-left, rear-right, front-right."""
    c, s = np.cos(boxes[..., 2]), np.sin(boxes[..., 2])
    lon = np.stack([c, s], -1) * (boxes[..., 3:4] / 2)
    lat = np.stack([-s, c], -1) * (boxes[..., 4:5] / 2)
    xy = boxes[..., :2]
    return np.stack([xy + lon + lat, xy - lon + lat, xy - lon - lat, xy + lon - lat], axis=-2)


def _boxes_intersect(a, b):
    """Separating-axis overlap test; broadcasts over leading dims."""
    d = b[..., :2] - a[..., :2]
    ua = np.stack([np.cos(a[..., 2]), np.sin(a[..., 2])], -1)
    ub = np.stack([np.cos(b[..., 2]), np.sin(b[..., 2])], -1)
    va, vb = np.stack([-ua[..., 1], ua[..., 0]], -1), np.stack([-ub[..., 1], ub[..., 0]], -1)
    separated = np.zeros(np.broadcast_shapes(a.shape[:-1], b.shape[:-1]), dtype=bool)
    for axis in (ua, va, ub, vb):
        dot = lambda p, q: np.abs(np.sum(p * q, axis=-1))
        radius_a = a[..., 3] / 2 * dot(ua, axis) + a[..., 4] / 2 * dot(va, axis)
        radius_b = b[..., 3] / 2 * dot(ub, axis) + b[..., 4] / 2 * dot(vb, axis)
        separated |= dot(d, axis) > radius_a + radius_b
    return ~separated


def _point_box_distance(points, boxes):
    """Distance from points [..., 2] to the area of boxes [..., 5] (0 inside); broadcasts."""
    d = points - boxes[..., :2]
    c, s = np.cos(boxes[..., 2]), np.sin(boxes[..., 2])
    ex = np.maximum(np.abs(d[..., 0] * c + d[..., 1] * s) - boxes[..., 3] / 2, 0.0)
    ey = np.maximum(np.abs(-d[..., 0] * s + d[..., 1] * c) - boxes[..., 4] / 2, 0.0)
    return np.hypot(ex, ey)


def _box_distance(a, b):
    """Polygon distance between boxes a [5] and b [K, 5] (vertex-to-polygon minimum for convex shapes)."""
    d_ab = _point_box_distance(_box_corners(a)[None, :, :], b[:, None, :]).min(-1)
    d_ba = _point_box_distance(_box_corners(b), a[None, None, :]).min(-1)
    return np.where(_boxes_intersect(a[None], b), 0.0, np.minimum(d_ab, d_ba))


def _relative_angle(ego_se2, agent_xy):
    """nuPlan get_agent_relative_angle: angle between ego heading and the direction to the agent."""
    v = agent_xy - ego_se2[:2]
    n = np.linalg.norm(v)
    if n == 0.0:
        return np.nan
    return float(np.arccos(np.clip((np.cos(ego_se2[2]) * v[0] + np.sin(ego_se2[2]) * v[1]) / n, -1.0, 1.0)))


def _is_agent_ahead(ego_se2, agent_xy):
    return _relative_angle(ego_se2, agent_xy) < np.deg2rad(30)


def _is_agent_behind(ego_se2, agent_xy):
    return _relative_angle(ego_se2, agent_xy) > np.deg2rad(150)


def _rear_axle_boxes(states, vehicle: _Vehicle):
    """Centre boxes [..., 5] of rear-axle states [..., N_STATE]."""
    h = states[..., HEADING]
    r = vehicle.rear_axle_to_center
    return np.stack(
        [
            states[..., X] + r * np.cos(h),
            states[..., Y] + r * np.sin(h),
            h,
            np.full_like(h, vehicle.length),
            np.full_like(h, vehicle.width),
        ],
        axis=-1,
    )


# ----------------------------------------------------------------------------------------------------------
# Paths


class _Path:
    """Polyline of (x, y, heading) parameterised by arc length (PDMPath)."""

    def __init__(self, xyh):
        xyh = np.asarray(xyh, dtype=np.float64)
        keep = np.r_[True, np.linalg.norm(np.diff(xyh[:, :2], axis=0), axis=-1) > 1e-6]
        self.xyh = xyh[keep].copy()
        self.xyh[:, 2] = np.unwrap(self.xyh[:, 2])
        seg = np.linalg.norm(np.diff(self.xyh[:, :2], axis=0), axis=-1)
        self.s = np.r_[0.0, np.cumsum(seg)]

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def interpolate(self, distances):
        d = np.clip(np.asarray(distances, dtype=np.float64), 1e-5, self.length)
        flat = d.reshape(-1)
        out = np.stack([np.interp(flat, self.s, self.xyh[:, k]) for k in range(3)], axis=-1)
        out[:, 2] = _wrap(out[:, 2])
        return out.reshape(d.shape + (3,))

    def project(self, points):
        """Arc length of the closest path point to each of points [..., 2] (shapely LineString.project)."""
        points = np.asarray(points, dtype=np.float64)
        q = points.reshape(-1, 2)
        a = self.xyh[:-1, :2]
        ab = np.diff(self.xyh[:, :2], axis=0)
        len2 = np.maximum(np.sum(ab**2, axis=-1), 1e-12)
        out = np.empty(len(q))
        for i in range(0, len(q), 256):
            qc = q[i : i + 256, None, :]
            t = np.clip(np.sum((qc - a) * ab, axis=-1) / len2, 0.0, 1.0)
            d2 = np.sum((qc - (a + t[..., None] * ab)) ** 2, axis=-1)
            j = np.argmin(d2, axis=-1)
            out[i : i + 256] = self.s[j] + t[np.arange(len(j)), j] * np.sqrt(len2[j])
        return out.reshape(points.shape[:-1])

    def parallel(self, offset: float) -> "_Path":
        h = self.xyh[:, 2]
        xy = self.xyh[:, :2] + offset * np.stack([-np.sin(h), np.cos(h)], axis=-1)
        return _Path(np.concatenate([xy, h[:, None]], axis=-1))


def _straight_path(x, y, heading, behind=50.0, ahead=200.0) -> _Path:
    s = np.arange(-behind, ahead + 0.5, 0.5)
    return _Path(np.stack([x + s * np.cos(heading), y + s * np.sin(heading), np.full_like(s, heading)], axis=-1))


# ----------------------------------------------------------------------------------------------------------
# Lane graph: route search, lane membership, intersections


class _LaneMap:
    def __init__(self, lane_graph):
        self.xy = np.asarray(lane_graph.nodes_xyz, dtype=np.float64)[:, :2]
        self.node_lane = np.asarray(lane_graph.node_lane_ids, dtype=np.int64)
        self.ranges = {
            int(lane): (int(a), int(b)) for lane, (a, b) in lane_graph.lane_id_to_node_range.items() if b > a
        }
        self.heading = np.zeros(len(self.xy))
        for a, b in self.ranges.values():
            if b - a > 1:
                d = np.diff(self.xy[a:b], axis=0)
                h = np.arctan2(d[:, 1], d[:, 0])
                self.heading[a : b - 1] = h
                self.heading[b - 1] = h[-1]
        self.tree = cKDTree(self.xy) if len(self.xy) else None

        self.lanes = np.array(sorted(self.ranges), dtype=np.int64)
        self.lane_index = {int(lane): i for i, lane in enumerate(self.lanes)}
        self.lane_length = np.array(
            [np.sum(np.linalg.norm(np.diff(self.xy[a:b], axis=0), axis=-1)) for a, b in (self.ranges[l] for l in self.lanes)]
        )
        first = {a: lane for lane, (a, _) in self.ranges.items()}
        last = {b - 1: lane for lane, (_, b) in self.ranges.items()}
        self.successors: dict[int, list[int]] = {}
        for src, dst in np.asarray(lane_graph.successor_edges, dtype=np.int64).reshape(-1, 2):
            u, v = last.get(int(src)), first.get(int(dst))
            if u is not None and v is not None and v not in self.successors.setdefault(u, []):
                self.successors[u].append(v)
        rows, cols = [], []
        for u, vs in self.successors.items():
            rows += [self.lane_index[u]] * len(vs)
            cols += [self.lane_index[v] for v in vs]
        weights = self.lane_length[rows] + 1e-3 if rows else np.zeros(0)
        n = len(self.lanes)
        self.graph = csr_matrix((weights, (rows, cols)), shape=(n, n))

    def _lanes_near(self, xy, radius):
        """{lane: nearest node} of lanes with a node within radius of xy."""
        out: dict[int, int] = {}
        if self.tree is None:
            return out
        for n in self.tree.query_ball_point(xy, radius):
            lane = int(self.node_lane[n])
            if lane in self.ranges and (lane not in out or np.sum((self.xy[n] - xy) ** 2) < np.sum((self.xy[out[lane]] - xy) ** 2)):
                out[lane] = n
        return out

    def _heading_aligned_lanes(self, xy, heading):
        """{lane: heading error} of lanes ego's point lies in (PDM: on-route lane polygons containing the rear axle)."""
        near = self._lanes_near(xy, LANE_HALF_WIDTH)
        if not near and self.tree is not None:
            _, n = self.tree.query(xy)
            near = {int(self.node_lane[n]): int(n)}
        return {lane: float(np.abs(_wrap(self.heading[n] - heading))) for lane, n in near.items() if lane in self.ranges}

    def route(self, start_xy, start_heading, goal_xy, goal_heading):
        """Lane sequence from ego's lane to the goal lane (or the reachable lane closest to the goal), plus
        whether the goal lane was reached."""
        starts = self._heading_aligned_lanes(start_xy, start_heading)
        if not starts:
            return None, False
        goals = [lane for lane, err in self._heading_aligned_lanes(goal_xy, goal_heading).items() if err < np.pi / 2]
        order = sorted(starts, key=lambda lane: starts[lane])
        src = [self.lane_index[lane] for lane in order]
        dist, pred = dijkstra(self.graph, directed=True, indices=src, return_predecessors=True)
        goal_idx = [self.lane_index[g] for g in goals]

        # PDM picks the on-route lane with the lowest heading error; on-route = the goal is reachable from it.
        # Only lanes running roughly along ego count, so a crossing lane that happens to reach the goal is not taken.
        for row, lane in enumerate(order):
            if starts[lane] < START_LANE_MAX_HEADING_ERROR and goal_idx and np.isfinite(dist[row, goal_idx]).any():
                target = goal_idx[int(np.argmin(dist[row, goal_idx]))]
                return self._backtrack(pred[row], self.lane_index[lane], target), True

        row = 0
        reachable = np.flatnonzero(np.isfinite(dist[row]))
        closest = np.array(
            [np.min(np.linalg.norm(self.xy[slice(*self.ranges[int(self.lanes[i])])] - goal_xy, axis=-1)) for i in reachable]
        )
        best = reachable[np.lexsort((dist[row, reachable], np.round(closest, 1)))[0]]
        return self._backtrack(pred[row], src[row], int(best)), False

    def _backtrack(self, pred_row, source, target):
        chain = [target]
        while chain[-1] != source:
            chain.append(int(pred_row[chain[-1]]))
        return [int(self.lanes[i]) for i in reversed(chain)]

    def extend(self, lanes, length):
        """Appends the straightest successors until ``length`` metres are added (or the graph ends)."""
        lanes, added = list(lanes), 0.0
        while added < length:
            a, b = self.ranges[lanes[-1]]
            options = [v for v in self.successors.get(lanes[-1], []) if v not in lanes]
            if not options:
                break
            nxt = min(options, key=lambda v: abs(_wrap(self.heading[self.ranges[v][0]] - self.heading[b - 1])))
            lanes.append(nxt)
            added += self.lane_length[self.lane_index[nxt]]
        return lanes

    def centerline(self, lanes) -> _Path:
        idx = np.concatenate([np.arange(*self.ranges[lane]) for lane in lanes])
        return _Path(np.concatenate([self.xy[idx], self.heading[idx, None]], axis=-1))

    def _neighbours(self, points, k=64):
        """Lane id [M, k] (-1 = none) and node heading [M, k] of lane nodes within LANE_HALF_WIDTH."""
        flat = points.reshape(-1, 2)
        if self.tree is None:
            return np.full((len(flat), 1), -1), np.zeros((len(flat), 1))
        k = min(k, len(self.xy))
        d, n = self.tree.query(flat, k=k, distance_upper_bound=LANE_HALF_WIDTH)
        d, n = d.reshape(len(flat), k), n.reshape(len(flat), k)
        hit = np.isfinite(d)
        n = np.where(hit, n, 0)
        return np.where(hit, self.node_lane[n], -1), self.heading[n]

    def lane_areas(self, corners, centers, route_heading):
        """Ego-area flags over [P, T]: in multiple lanes (corners spread over lanes, no lane holds all four),
        and oncoming (centre in no lane running within 90 deg of the route direction)."""
        p, t = centers.shape[:2]
        lanes, _ = self._neighbours(corners)  # [P*T*4, k]
        lanes = lanes.reshape(p * t, 4, -1)
        valid = lanes >= 0
        # A lane holds all four corners if a lane id of corner 0 appears for every other corner.
        holds_all = np.ones(lanes[:, 0].shape, dtype=bool) & valid[:, 0]
        for c in range(1, 4):
            holds_all &= np.any(lanes[:, 0, :, None] == np.where(valid[:, c], lanes[:, c], -2)[:, None, :], axis=-1)
        flat = np.sort(np.where(valid, lanes, -1).reshape(p * t, -1), axis=-1)
        distinct = np.sum((np.diff(flat, axis=-1) != 0) & (flat[:, 1:] >= 0), axis=-1) + (flat[:, 0] >= 0)
        multiple = (distinct > 1) & ~holds_all.any(-1)

        center_lanes, center_heading = self._neighbours(centers)
        aligned = (center_lanes >= 0) & (np.abs(_wrap(center_heading - route_heading.reshape(-1, 1))) < np.pi / 2)
        oncoming = ~aligned.any(-1)
        return multiple.reshape(p, t), oncoming.reshape(p, t)

    def in_intersection(self, xy) -> bool:
        lanes, heading = self._neighbours(np.asarray(xy, dtype=np.float64)[None])
        lanes, heading = lanes[0], heading[0]
        heading = heading[lanes >= 0]
        if len(np.unique(lanes[lanes >= 0])) < 2:
            return False
        return bool(np.max(np.abs(_wrap(heading[:, None] - heading[None, :]))) > INTERSECTION_HEADING_SPREAD)


# ----------------------------------------------------------------------------------------------------------
# Observation


@dataclass
class _Observation:
    boxes: np.ndarray  # [M, O, 5] boxes at t = 0, 2, 4, ... steps
    token: np.ndarray  # [O] Waymax object index; red lights are -1 - k
    is_agent: np.ndarray  # [O] vehicle / pedestrian / cyclist
    speed: np.ndarray  # [O] current speed (0 for static objects and red lights)
    heading: np.ndarray  # [O] current heading
    collided: set  # tokens ignored (ego overlapped them)

    @property
    def is_red_light(self):
        return self.token < 0

    def at(self, time_idx: int) -> np.ndarray:
        return self.boxes[min(time_idx // OBSERVATION_SAMPLE_RES, len(self.boxes) - 1)]


def _build_observation(sim_state, world_idx, t, ego_idx, ego_box, collided, red_boxes, velocity_ratios):
    traj = sim_state.log_trajectory
    get = lambda a: np.asarray(a[world_idx])[:, t].astype(np.float64)
    x, y, yaw, vx, vy, length, width = (get(a) for a in (traj.x, traj.y, traj.yaw, traj.vel_x, traj.vel_y, traj.length, traj.width))
    valid = np.asarray(traj.valid[world_idx])[:, t].astype(bool)
    types = np.asarray(sim_state.object_metadata.object_types[world_idx]).astype(int)
    dist = np.hypot(x - ego_box[0], y - ego_box[1])
    keep = valid & (np.arange(len(x)) != ego_idx) & (dist <= MAP_RADIUS)
    keep &= ~np.isin(np.arange(len(x)), list(collided))

    # nearest objects per type (PDMObjectManager)
    chosen = []
    for obj_type, cap in MAX_DYNAMIC_OBJECTS.items():
        idx = np.flatnonzero(keep & (types == obj_type))
        chosen.append(idx[np.argsort(dist[idx], kind="stable")][:cap])
    idx = np.flatnonzero(keep & ~np.isin(types, AGENT_TYPES))
    chosen.append(idx[np.argsort(dist[idx], kind="stable")][:MAX_STATIC_OBJECTS])
    objs = np.concatenate(chosen).astype(int)

    is_agent = np.isin(types[objs], AGENT_TYPES)
    speed = np.hypot(vx[objs], vy[objs])
    # velocity along the (possibly reversed) heading
    forward = np.abs(_wrap(yaw[objs] - np.arctan2(vy[objs], vx[objs]))) < np.pi / 2
    track_heading = np.where(forward, yaw[objs], _wrap(yaw[objs] + np.pi))
    dxy = np.where(is_agent[:, None], np.stack([np.cos(track_heading), np.sin(track_heading)], -1) * speed[:, None], 0.0)
    base = np.stack([x[objs], y[objs], yaw[objs], length[objs], width[objs]], axis=-1)

    n_samples = max(PROPOSAL_STEPS + int(1 / DT), TRAJECTORY_STEPS)
    times = np.arange(0, n_samples + OBSERVATION_SAMPLE_RES, OBSERVATION_SAMPLE_RES) * DT
    per_ratio = []
    for ratio in velocity_ratios:
        boxes = np.repeat(base[None], len(times), axis=0)
        boxes[..., :2] += times[:, None, None] * dxy[None] * ratio
        per_ratio.append(boxes)
    dyn_tokens = np.concatenate([objs] * len(velocity_ratios))
    dyn_agent = np.concatenate([is_agent] * len(velocity_ratios))
    dyn_speed = np.concatenate([np.where(is_agent, speed, 0.0)] * len(velocity_ratios))
    dyn_heading = np.concatenate([yaw[objs]] * len(velocity_ratios))
    boxes = np.concatenate(per_ratio, axis=1)

    if len(red_boxes):
        red = np.asarray(red_boxes, dtype=np.float64)
        boxes = np.concatenate([boxes, np.repeat(red[None], len(times), axis=0)], axis=1)
        dyn_tokens = np.concatenate([dyn_tokens, -1 - np.arange(len(red))])
        dyn_agent = np.concatenate([dyn_agent, np.zeros(len(red), bool)])
        dyn_speed = np.concatenate([dyn_speed, np.zeros(len(red))])
        dyn_heading = np.concatenate([dyn_heading, red[:, 2]])

    obs = _Observation(boxes, dyn_tokens.astype(int), dyn_agent, dyn_speed, dyn_heading, collided)
    # objects overlapping ego now are ignored from here on (red lights only when ego is fully inside: never,
    # for a 1 m stop-line box)
    overlapping = _boxes_intersect(ego_box[None], obs.boxes[0]) & ~obs.is_red_light
    collided.update(int(tok) for tok in obs.token[overlapping])
    return obs


def _speed_limit(sim_state, world_idx, ego_xyz, expert_max_speed):
    """V-Max's speed-limit inference: lane type of the nearest lane centre combined with the expert's top speed."""
    expert = SPEED_LIMITS[min(int(np.searchsorted(SPEED_LIMITS, expert_max_speed)), len(SPEED_LIMITS) - 1)]
    rg = sim_state.roadgraph_points
    xyz, types, valid = (np.asarray(a[world_idx]) for a in (rg.xyz, rg.types, rg.valid))
    lanes = valid.astype(bool) & np.isin(types, LANE_TYPES)
    if not lanes.any():
        return float(expert)
    nearest = np.flatnonzero(lanes)[np.argmin(np.sum((xyz[lanes] - ego_xyz) ** 2, axis=-1))]
    if types[nearest] == LANE_FREEWAY:
        return float(max(FREEWAY_SPEED_LIMIT, expert))
    return float(min(SURFACE_STREET_SPEED_LIMIT, expert))


def _red_light_boxes(sim_state, world_idx, t, route_lanes, lane_map: _LaneMap):
    tls = sim_state.log_traffic_light
    state, lane_ids, valid = (np.asarray(a[world_idx])[:, t] for a in (tls.state, tls.lane_ids, tls.valid))
    x, y = (np.asarray(a[world_idx])[:, t] for a in (tls.x, tls.y))
    boxes = []
    for k in np.flatnonzero(valid.astype(bool) & np.isin(state, RED_LIGHT_STATES)):
        lane = int(lane_ids[k])
        if lane not in route_lanes or lane not in lane_map.ranges:
            continue
        a, b = lane_map.ranges[lane]
        stop = np.array([x[k], y[k]], dtype=np.float64)
        n = a + int(np.argmin(np.linalg.norm(lane_map.xy[a:b] - stop, axis=-1)))
        h = lane_map.heading[n]
        boxes.append([stop[0] + 0.5 * np.cos(h), stop[1] + 0.5 * np.sin(h), h, 1.0, 2 * LANE_HALF_WIDTH])
    return boxes


# ----------------------------------------------------------------------------------------------------------
# Proposals: IDM along paths (PDMGenerator + BatchIDMPolicy)


class _Generator:
    def __init__(self, paths, obs: _Observation, ego_state, vehicle: _Vehicle, target_velocities):
        self.paths, self.obs, self.vehicle = paths, obs, vehicle
        self.target_velocities = np.asarray(target_velocities, dtype=np.float64)
        n_lon = len(self.target_velocities)
        self.lateral = np.repeat(np.arange(len(paths)), n_lon)
        self.longitudinal = np.tile(np.arange(n_lon), len(paths))
        n = len(self.lateral)
        self.se2 = np.zeros((n, TRAJECTORY_STEPS + 1, 3))
        self.idm = np.zeros((n, TRAJECTORY_STEPS + 1, 2))  # progress, velocity
        self.lead = np.zeros((n, TRAJECTORY_STEPS + 1, 3))  # progress, velocity, rear length
        self._corridor_hits: dict[int, np.ndarray] = {}
        self._collided = np.isin(obs.token, list(obs.collided))
        for lat, path in enumerate(paths):
            group = self.lateral == lat
            progress = float(path.project(ego_state[[X, Y]]))
            self.idm[group, 0] = (progress, ego_state[VX])
            self.se2[group, 0] = path.interpolate(progress)

    def unroll(self, proposals, start, stop):
        """Unrolls time steps start..stop (inclusive) for the given proposals, path by path."""
        proposals = np.asarray(proposals)
        for lat in np.unique(self.lateral[proposals]):
            group = proposals[self.lateral[proposals] == lat]
            for t in range(start, stop + 1):
                self._update_leading_agents(lat, group, t)
                self._update_idm(group, t)
                self.se2[group, t] = self.paths[lat].interpolate(self.idm[group, t, 0])

    def _update_idm(self, group, t):
        x, v = self.idm[group, t - 1, 0], self.idm[group, t - 1, 1]
        x_lead, v_lead, l_r_lead = self.lead[group, t].T
        v0 = self.target_velocities[self.longitudinal[group]]
        s_star = MIN_GAP_TO_LEAD_AGENT + v * HEADWAY_TIME + v * (v - v_lead) / (2 * np.sqrt(ACCEL_MAX * DECEL_MAX))
        s_alpha = np.maximum(x_lead - x - l_r_lead, MIN_GAP_TO_LEAD_AGENT)
        v_dot = ACCEL_MAX * (1 - (v / v0) ** IDM_ACCELERATION_EXPONENT - (s_star / s_alpha) ** 2)
        v_dot = np.clip(v_dot, -DECEL_MAX, ACCEL_MAX)
        self.idm[group, t] = np.stack([x + DT * v, v + DT * v_dot], axis=-1)

    def _corridor(self, lat):
        """[M, O] objects intersecting the path's driving corridor (path ahead of ego buffered by half the
        ego width) in each observation sample."""
        if lat not in self._corridor_hits:
            path = self.paths[lat]
            start = self.idm[self.lateral == lat][0, 0, 0]
            end = start + self.target_velocities.max() * TRAJECTORY_STEPS * DT
            half_width = self.vehicle.width / 2
            s = np.arange(start - half_width, end + half_width + CORRIDOR_SAMPLE_M, CORRIDOR_SAMPLE_M)
            pts = path.interpolate(np.clip(s, 0.0, path.length))[:, :2]
            hits = np.zeros(self.obs.boxes.shape[:2], dtype=bool)
            for m in range(len(hits)):
                if hits.shape[1]:
                    hits[m] = _point_box_distance(pts[None, :, :], self.obs.boxes[m][:, None, :]).min(-1) <= half_width
            self._corridor_hits[lat] = hits
        return self._corridor_hits[lat]

    def _update_leading_agents(self, lat, group, t):
        if t % LEADING_AGENT_UPDATE_RATE != 0:
            # As upstream: at t=1 this copies the zero-initialised entry of t=0.
            self.lead[group, t] = self.lead[group, t - 1]
            return
        boxes = self.obs.at(t)
        objs = np.flatnonzero(self._corridor(lat)[min(t // OBSERVATION_SAMPLE_RES, len(self.obs.boxes) - 1)] & ~self._collided)
        progress = self.paths[lat].project(boxes[objs, :2]) if len(objs) else np.zeros(0)
        # As upstream, one buffer is reused across the path's proposals, so fields a proposal does not write
        # keep the previous proposal's values.
        lead = np.zeros(3)
        for p in group:
            ego_progress = self.idm[p, t - 1, 0]
            ahead = progress > ego_progress
            if ahead.any():
                ego_box = _rear_axle_boxes(np.r_[self.se2[p, t - 1], np.zeros(N_STATE - 3)], self.vehicle)
                candidates = objs[ahead]
                distances = _box_distance(ego_box, boxes[candidates])
                k = int(np.argmin(distances))
                o = candidates[k]
                lead[0] = ego_progress + distances[k]
                if not self.obs.is_red_light[o]:
                    rel = _wrap(self.obs.heading[o] - self.se2[p, t - 1, 2])
                    lead[1] = self.obs.speed[o] * np.cos(rel) if self.obs.is_agent[o] else 0.0
            else:
                lead[0] = self.paths[lat].length
                lead[2] = self.vehicle.length / 2
            self.lead[p, t] = lead


# ----------------------------------------------------------------------------------------------------------
# Simulation: batch LQR tracker + kinematic bicycle (PDMSimulator)


def _fit_velocity_curvature(poses):
    """nuPlan's joint least-squares fit of velocity and curvature profiles to poses [B, N, 3]."""
    jerk_penalty, curvature_rate_penalty, initial_curvature_penalty = 1e-4, 1e-2, 1e-10
    diff = np.diff(poses, axis=1)
    xy_disp, heading_disp = diff[..., :2], _wrap(diff[..., 2])
    batch, m = heading_disp.shape
    bmm = lambda a, b: np.einsum("bij,bjk->bik", a, b)

    y = xy_disp.reshape(batch, -1)
    a_col = np.zeros(y.shape)
    a_col[:, 0::2] = np.cos(poses[:, :-1, 2])
    a_col[:, 1::2] = np.sin(poses[:, :-1, 2])
    a = np.repeat(a_col[..., None] * DT**2, m, axis=2)
    a[..., 0] = a_col * DT
    upper = np.repeat(np.triu(np.ones((m, m), dtype=bool), k=1), 2, axis=0)
    a[:, upper] = 0.0
    banded = np.zeros((m - 2, m - 1))
    banded[:, 1:] += np.eye(m - 2)
    banded[:, :-1] -= np.eye(m - 2)
    r = np.repeat(np.block([np.zeros((m - 2, 1)), banded])[None], batch, axis=0)
    at, rt = a.transpose(0, 2, 1), r.transpose(0, 2, 1)
    sol = np.einsum("bij,bj->bi", bmm(np.linalg.pinv(bmm(at, a) + jerk_penalty * bmm(rt, r)), at), y)
    velocity = sol[:, :1] + np.pad(np.cumsum(sol[:, 1:] * DT, axis=-1), [(0, 0), (1, 0)])

    a = np.repeat(np.tri(m)[None], batch, axis=0)
    a[:, :, 0] = velocity * DT
    a[:, 1:, 1:] *= (velocity * DT**2)[:, None, 1:].transpose(0, 2, 1)
    q = curvature_rate_penalty * np.eye(m)
    q[0, 0] = initial_curvature_penalty
    at = a.transpose(0, 2, 1)
    sol = np.einsum("bij,bj->bi", bmm(np.linalg.pinv(bmm(at, a) + q), at), heading_disp)
    curvature = sol[:, :1] + np.pad(np.cumsum(sol[:, 1:] * DT, axis=-1), [(0, 0), (1, 0)])
    return velocity, curvature


class _LQRTracker:
    """tuplan_garage BatchLQRTracker with nuPlan's default parameters."""

    q_longitudinal, r_longitudinal = 10.0, 1.0
    q_lateral, r_lateral = np.diag([1.0, 10.0, 0.0]), np.diag([1.0])
    horizon = 10
    stopping_gain, stopping_velocity = 0.5, 0.2

    def __init__(self, poses, wheel_base):
        self.poses, self.wheel_base = poses, wheel_base
        self.velocity, self.curvature = _fit_velocity_curvature(poses)

    def track(self, index, states):
        ref = self.poses[:, index]
        dx, dy = states[:, X] - ref[:, 0], states[:, Y] - ref[:, 1]
        lateral = np.stack(
            [-dx * np.sin(ref[:, 2]) + dy * np.cos(ref[:, 2]), _wrap(states[:, HEADING] - ref[:, 2]), states[:, STEER]],
            axis=-1,
        )
        v0 = states[:, VX]

        n = self.velocity.shape[1]
        ref_idx = min(index + self.horizon, n - 1)
        v_ref = self.velocity[:, ref_idx]
        curvature = np.zeros((len(states), self.horizon))
        length = ref_idx - index
        curvature[:, :length] = self.curvature[:, index:ref_idx]
        if length < self.horizon:
            curvature[:, length:] = self.curvature[:, ref_idx, None]

        accel = np.zeros(len(states))
        steering_rate = np.zeros(len(states))
        stop = (v_ref <= self.stopping_velocity) & (v0 <= self.stopping_velocity)
        accel[stop] = -self.stopping_gain * (v0[stop] - v_ref[stop])

        go = ~stop
        b = self.horizon * DT
        accel[go] = -1 / (b * self.q_longitudinal * b + self.r_longitudinal) * b * self.q_longitudinal * (v0[go] - v_ref[go])
        velocity_profile = (v0[go, None] + np.pad(np.cumsum(np.repeat(accel[go, None], self.horizon, -1) * DT, -1), [(0, 0), (1, 0)]))[:, : self.horizon]
        steering_rate[go] = self._lateral(lateral[go], velocity_profile, curvature[go])
        return accel, steering_rate

    def _lateral(self, x0, velocity, curvature):
        batch = len(x0)
        eye = np.eye(3)
        in_matrix = np.zeros((3, 1))
        in_matrix[2] = DT
        step = np.tile(eye[None, None], [self.horizon, batch, 1, 1])
        step[:, :, 0, 1] = velocity.T * DT
        step[:, :, 1, 2] = velocity.T * DT / self.wheel_base
        affine = np.zeros((self.horizon, batch, 3))
        affine[:, :, 1] = -velocity.T * curvature.T * DT
        a = np.tile(eye[None], [batch, 1, 1])
        b = np.zeros((batch, 3, 1))
        g = np.zeros((batch, 3))
        for m, c in zip(step, affine):
            a = np.einsum("bij,bjk->bik", m, a)
            b = np.einsum("bij,bjk->bik", m, b) + in_matrix
            g = np.einsum("bij,bj->bi", m, g) + c
        err = np.einsum("bij,bj->bi", a, x0) + g
        err[:, 1:] = _wrap(err[:, 1:])
        bt_q = np.einsum("bij,jk->bik", b.transpose(0, 2, 1), self.q_lateral)
        inv = -1 / (np.einsum("bij,bji->bi", bt_q, b) + self.r_lateral[0, 0])
        return (inv * np.einsum("bij,bj->bi", bt_q, err))[:, 0]


def _propagate_bicycle(states, accel_cmd, steering_rate_cmd, wheel_base):
    """nuPlan kinematic bicycle (rear axle) with first-order lag on acceleration and steering."""
    max_steer, accel_tc, steer_tc = np.pi / 3, 0.2, 0.05
    accel = DT / (DT + accel_tc) * (accel_cmd - states[:, AX]) + states[:, AX]
    ideal_steer = DT * steering_rate_cmd + states[:, STEER]
    steer = DT / (DT + steer_tc) * (ideal_steer - states[:, STEER]) + states[:, STEER]
    steer_rate = (steer - states[:, STEER]) / DT

    v, h = states[:, VX], states[:, HEADING]
    out = states.copy()
    out[:, X] = states[:, X] + DT * v * np.cos(h)
    out[:, Y] = states[:, Y] + DT * v * np.sin(h)
    out[:, HEADING] = _wrap(h + DT * v * np.tan(states[:, STEER]) / wheel_base)
    out[:, VX] = v + DT * accel
    out[:, VY] = 0.0
    out[:, STEER] = np.clip(states[:, STEER] + DT * steer_rate, -max_steer, max_steer)
    out[:, YAW_RATE] = out[:, VX] * np.tan(out[:, STEER]) / wheel_base
    out[:, AX], out[:, AY] = accel, 0.0
    out[:, YAW_ACCEL] = (out[:, YAW_RATE] - states[:, YAW_RATE]) / DT
    out[:, STEER_RATE] = steer_rate
    return out


def _simulate(poses, initial_state, vehicle: _Vehicle):
    """Tracks pose sequences [B, N + 1, 3] from the current ego state; returns states [B, N + 1, N_STATE]."""
    tracker = _LQRTracker(poses, vehicle.wheel_base)
    states = np.zeros(poses.shape[:2] + (N_STATE,))
    states[:, 0] = initial_state
    for t in range(1, poses.shape[1]):
        accel, steering_rate = tracker.track(t - 1, states[:, t - 1])
        states[:, t] = _propagate_bicycle(states[:, t - 1], accel, steering_rate, vehicle.wheel_base)
    return states


# ----------------------------------------------------------------------------------------------------------
# Scoring (PDMScorer)


def _savgol_derivative(y, n_time, window_length, poly_order=2, deriv_order=1):
    return savgol_filter(y, polyorder=poly_order, window_length=min(window_length, n_time), deriv=deriv_order, delta=DT, axis=-1)


def _comfortable(states):
    """nuPlan's six comfort checks over [P, T] simulated proposals."""
    n_time = states.shape[1]
    within = lambda m, lo, hi: np.all((m > lo) & (m < hi), axis=-1)

    def acceleration(coord, window_length=8):
        a = {"x": states[..., AX], "y": states[..., AY], "magnitude": np.hypot(states[..., AX], states[..., AY])}[coord]
        return np.round(savgol_filter(a, polyorder=2, window_length=min(window_length, n_time), axis=-1), 8)

    def jerk(coord):
        return np.round(_savgol_derivative(acceleration(coord), n_time, window_length=n_time), 8)

    headings = states[..., HEADING]
    unwrapped = headings - 2 * np.pi * np.pad(np.cumsum(np.round(np.diff(headings, axis=-1) / (2 * np.pi)), axis=-1), [(0, 0), (1, 0)])
    yaw_rate = np.round(_savgol_derivative(unwrapped, n_time, window_length=5), 8)
    yaw_accel = np.round(_savgol_derivative(unwrapped, n_time, window_length=5, poly_order=3, deriv_order=2), 8)
    checks = [
        within(acceleration("x", n_time), MIN_LON_ACCEL, MAX_LON_ACCEL),
        within(acceleration("y", n_time), -MAX_ABS_LAT_ACCEL, MAX_ABS_LAT_ACCEL),
        within(jerk("magnitude"), -MAX_ABS_MAG_JERK, MAX_ABS_MAG_JERK),
        within(jerk("x"), -MAX_ABS_LON_JERK, MAX_ABS_LON_JERK),
        within(yaw_accel, -MAX_ABS_YAW_ACCEL, MAX_ABS_YAW_ACCEL),
        within(yaw_rate, -MAX_ABS_YAW_RATE, MAX_ABS_YAW_RATE),
    ]
    return np.all(np.stack(checks, axis=-1), axis=-1)


def _collision_type(state, ego_box, obs: _Observation, o, box):
    if np.hypot(state[VX], state[VY]) <= COLLISION_STOPPED_SPEED_THRESHOLD:
        return STOPPED_EGO
    if not obs.is_agent[o] or obs.speed[o] <= COLLISION_STOPPED_SPEED_THRESHOLD:
        return STOPPED_TRACK
    if _is_agent_behind(state[:3], box[:2]):
        return ACTIVE_REAR
    # front bumper: the front edge of the ego box as a zero-length box
    h = ego_box[2]
    front = np.array([ego_box[0] + ego_box[3] / 2 * np.cos(h), ego_box[1] + ego_box[3] / 2 * np.sin(h), h, 0.0, ego_box[4]])
    if _boxes_intersect(front, box):
        return ACTIVE_FRONT
    return ACTIVE_LATERAL


@dataclass
class _Scores:
    total: np.ndarray
    collision_time: np.ndarray  # [P] seconds to the first at-fault collision (inf if none)


def _score_components(states, obs: _Observation, centerline: _Path, lane_map: _LaneMap | None, vehicle: _Vehicle,
                      offroad_fn) -> dict:
    """PDMScorer's per-proposal sub-scores over simulated states [P, T, N_STATE]: no_collision (0 / 0.5 / 1),
    drivable (0 / 1), direction (0 / 0.5 / 1), progress (m along the centerline), ttc (0 / 1), comfort (0 / 1) and
    collision_time (s to the first at-fault collision, inf if none)."""
    n, t_len = states.shape[:2]
    boxes = _rear_axle_boxes(states, vehicle)  # [P, T, 5]
    corners = _box_corners(boxes)
    centers = boxes[..., :2]

    nondrivable = offroad_fn(boxes)
    if lane_map is not None:
        route_heading = centerline.interpolate(centerline.project(centers))[..., 2]
        multiple, oncoming = lane_map.lane_areas(corners, centers, route_heading)
    else:
        multiple, oncoming = np.zeros((n, t_len), bool), np.zeros((n, t_len), bool)
    lateral_at_fault_area = multiple | nondrivable

    # at-fault collisions
    no_collision = np.ones(n)
    collision_idx = np.full(n, np.inf)
    collided = [set(obs.collided) for _ in range(n)]
    for t in range(t_len):
        obs_boxes = obs.at(t)
        for p, o in np.argwhere(_boxes_intersect(boxes[:, t, None], obs_boxes[None])):
            tok = int(obs.token[o])
            if obs.is_red_light[o] or tok in collided[p]:
                continue
            kind = _collision_type(states[p, t], boxes[p, t], obs, o, obs_boxes[o])
            if kind in (ACTIVE_FRONT, STOPPED_TRACK) or (lateral_at_fault_area[p, t] and kind == ACTIVE_LATERAL):
                no_collision[p] = min(no_collision[p], 0.0 if obs.is_agent[o] else 0.5)
                collision_idx[p] = min(t, collision_idx[p])
            else:
                collided[p].add(tok)

    # drivable area
    drivable = np.where(nondrivable.any(-1), 0.0, 1.0)

    # driving direction: longest stretch driven against traffic
    step = np.concatenate([np.zeros((n, 1)), np.linalg.norm(np.diff(centers, axis=1), axis=-1)], axis=1)
    step[~oncoming] = 0.0
    direction = np.ones(n)
    for p in range(n):
        splits = np.split(step[p], np.flatnonzero(np.diff(oncoming[p])) + 1)
        worst = max(s.sum() for s in splits)
        direction[p] = 1.0 if worst < DRIVING_DIRECTION_COMPLIANCE_THRESHOLD else (0.5 if worst < DRIVING_DIRECTION_VIOLATION_THRESHOLD else 0.0)

    # progress along the centerline
    progress = np.clip(centerline.project(centers[:, -1]) - centerline.project(centers[:, 0]), 0.0, None)

    # time to collision: ego projected 0, 0.3, 0.6, 0.9 s ahead at its current speed
    speeds = np.hypot(states[..., VX], states[..., VY])
    heading_dir = np.stack([np.cos(states[..., HEADING]), np.sin(states[..., HEADING])], -1)
    ttc = np.ones(n)
    temp_collided = [set(obs.collided) for _ in range(n)]
    for t in range(t_len):
        for f in TTC_FUTURE_STEPS:
            moved = boxes[:, t].copy()
            moved[:, :2] += heading_dir[:, t] * speeds[:, t, None] * f * DT
            obs_boxes = obs.at(t + f)
            for p, o in np.argwhere(_boxes_intersect(moved[:, None], obs_boxes[None])):
                tok = int(obs.token[o])
                if obs.is_red_light[o] or tok in temp_collided[p] or speeds[p, t] < STOPPED_SPEED_THRESHOLD:
                    continue
                rear = states[p, t, :3]
                track_xy = obs_boxes[o, :2]
                risky_area = lateral_at_fault_area[p, t] or (lane_map is not None and lane_map.in_intersection(rear[:2]))
                if _is_agent_ahead(rear, track_xy) or (risky_area and not _is_agent_behind(rear, track_xy)):
                    ttc[p] = 0.0
                else:
                    temp_collided[p].add(tok)

    comfort = _comfortable(states).astype(np.float64)
    return {"no_collision": no_collision, "drivable": drivable, "direction": direction, "progress": progress,
            "ttc": ttc, "comfort": comfort, "collision_time": collision_idx * DT}


def _score(states, obs: _Observation, centerline: _Path, lane_map: _LaneMap | None, vehicle: _Vehicle, offroad_fn):
    c = _score_components(states, obs, centerline, lane_map, vehicle, offroad_fn)
    no_collision, drivable, direction, progress, ttc, comfort = (
        c[k] for k in ("no_collision", "drivable", "direction", "progress", "ttc", "comfort"))
    multiplicative = no_collision * drivable * direction
    raw = progress * multiplicative
    if raw.max() > PROGRESS_DISTANCE_THRESHOLD:
        progress_score = raw / raw.max()
    else:
        progress_score = np.where(multiplicative == 0.0, 0.0, 1.0)
    weighted = (WEIGHT_PROGRESS * progress_score + WEIGHT_TTC * ttc + WEIGHT_COMFORT * comfort) / (
        WEIGHT_PROGRESS + WEIGHT_TTC + WEIGHT_COMFORT
    )
    return _Scores(multiplicative * weighted, c["collision_time"])


# ----------------------------------------------------------------------------------------------------------
# Planner


@dataclass
class _ScenarioState:
    key: tuple
    last_t: int
    lane_map: _LaneMap | None
    paths: list
    route_lanes: set
    goal_lane_reached: bool
    collided: set
    expert_max_speed: float  # from the log at the first replan, before plans overwrite ego's logged future


@dataclass
class PlanContext:
    """One world's replanning state, kept for scoring other trajectories the way PDM scores its proposals
    (planner.pdm_score): the vehicle, the current rear-axle ego state, the constant-velocity observation, the route
    centerline / lane map, the road-edge offroad test, and PDM-Closed's executed plan (rear-axle states, 10 Hz)."""
    vehicle: _Vehicle
    ego_state: np.ndarray
    obs: _Observation
    centerline: _Path
    lane_map: _LaneMap | None
    offroad_fn: Any
    executed: np.ndarray


class PDMClosedPlanner(AbstractPlanner):
    """PDM-Closed baseline; ``plan_trajectory`` matches the other planners' closed-loop interface.

    ``scorers`` are only used for their lane graph (set by the runner) and the evaluation's road-edge offroad test.
    ``cv_velocity_ratios=(0.8, 1.0)`` reproduces the two-speed forecast of the nuplan_rs fork of tuplan_garage;
    the default (1.0,) is upstream.
    """

    def __init__(self, num_worlds: int = 1, cv_velocity_ratios=(1.0,), **kwargs: Any) -> None:
        self.num_worlds = int(num_worlds)
        self.cv_velocity_ratios = tuple(float(r) for r in cv_velocity_ratios)
        self.scorers = [Scorer(metrics=["offroad"]) for _ in range(self.num_worlds)]
        self._scenarios: list[_ScenarioState | None] = [None] * self.num_worlds
        self.last_context: list[PlanContext | None] = [None] * self.num_worlds

    def plan_trajectory(self, sim_state, goal, *, rng=None, timestep: int = 0, **kwargs: Any) -> PlannerResult:
        goal = np.asarray(goal, dtype=np.float64).reshape(-1, 2)
        trajectories, best_scores, braking, goal_lane = [], [], [], []
        for w in range(self.num_worlds):
            traj, score, brake, reached = self._plan_world(sim_state, goal[w], int(timestep), w)
            trajectories.append(traj)
            best_scores.append(score)
            braking.append(brake)
            goal_lane.append(reached)
        traj_bt5 = jnp.asarray(np.stack(trajectories), dtype=jnp.float32)
        t_seconds = jnp.broadcast_to(jnp.arange(TRAJECTORY_STEPS + 1, dtype=jnp.float32) * DT, traj_bt5.shape[:2])
        return PlannerResult(
            start_t_b=jnp.full((self.num_worlds,), int(timestep), dtype=jnp.int32),
            trajectory_norm_btd=traj_bt5,
            trajectory_world_bt5=traj_bt5,
            world_t_seconds_bt=t_seconds,
            world_t_valid_bt=jnp.ones(traj_bt5.shape[:2], dtype=bool),
            aux={
                "world_dt_seconds": jnp.full((self.num_worlds,), DT, dtype=jnp.float32),
                "pdm_best_score_b": jnp.asarray(best_scores, dtype=jnp.float32),
                "pdm_emergency_brake_b": jnp.asarray(braking),
                "pdm_goal_lane_on_route_b": jnp.asarray(goal_lane),
            },
        )

    def _scenario(self, sim_state, goal_xy, t, w, ego_idx, rear_axle):
        lane_graph = self.scorers[w]._lane_graph
        key = (id(lane_graph), float(goal_xy[0]), float(goal_xy[1]))
        state = self._scenarios[w]
        if state is not None and state.key == key and t > state.last_t:
            state.last_t = t
            return state

        # New scenario: the centerline is built once, from the first replanning state (as PDM does at iteration 0).
        traj = sim_state.log_trajectory
        valid = np.asarray(traj.valid[w])[ego_idx].astype(bool)
        goal_t = int(np.flatnonzero(valid)[-1]) if valid.any() else t
        goal_heading = float(np.asarray(traj.yaw[w])[ego_idx, goal_t])
        speed = np.hypot(np.asarray(traj.vel_x[w])[ego_idx], np.asarray(traj.vel_y[w])[ego_idx])
        expert_max_speed = float(speed[valid].max()) if valid.any() else 0.0
        lane_map = _LaneMap(lane_graph) if lane_graph is not None and len(lane_graph.nodes_xyz) else None
        lanes, reached = (None, False)
        if lane_map is not None:
            lanes, reached = lane_map.route(rear_axle[:2], rear_axle[2], goal_xy, goal_heading)
        if lanes:
            lanes = lane_map.extend(lanes, ROUTE_EXTENSION_M)
            centerline = lane_map.centerline(lanes)
        if not lanes or centerline.length < 1.0:
            lanes, centerline = [], _straight_path(*rear_axle)
        paths = [centerline] + [centerline.parallel(offset) for offset in LATERAL_OFFSETS]
        state = _ScenarioState(key, t, lane_map, paths, set(lanes), reached, set(), expert_max_speed)
        self._scenarios[w] = state
        return state

    def _plan_world(self, sim_state, goal_xy, t, w):
        ego_idx = scorer_helpers.get_ego_idx(sim_state, w)
        traj = sim_state.log_trajectory
        get = lambda a, tt=t: float(np.asarray(a[w])[ego_idx, tt])
        length, width = get(traj.length), get(traj.width)
        vehicle = _Vehicle.from_box(length, width)
        cx, cy, yaw = get(traj.x), get(traj.y), get(traj.yaw)
        r = vehicle.rear_axle_to_center
        rear_axle = np.array([cx - r * np.cos(yaw), cy - r * np.sin(yaw), yaw])

        v_long = get(traj.vel_x) * np.cos(yaw) + get(traj.vel_y) * np.sin(yaw)
        accel, yaw_rate = 0.0, 0.0
        if t > 0 and bool(np.asarray(traj.valid[w])[ego_idx, t - 1]):
            yaw_prev = get(traj.yaw, t - 1)
            v_prev = get(traj.vel_x, t - 1) * np.cos(yaw_prev) + get(traj.vel_y, t - 1) * np.sin(yaw_prev)
            accel = (v_long - v_prev) / DT
            yaw_rate = float(_wrap(yaw - yaw_prev)) / DT
        steer = float(np.clip(np.arctan(yaw_rate * vehicle.wheel_base / v_long), -np.pi / 3, np.pi / 3)) if abs(v_long) > 0.5 else 0.0
        ego_state = np.zeros(N_STATE)
        ego_state[[X, Y, HEADING]] = rear_axle
        ego_state[VX], ego_state[AX], ego_state[STEER], ego_state[YAW_RATE] = v_long, accel, steer, yaw_rate

        scenario = self._scenario(sim_state, goal_xy, t, w, ego_idx, rear_axle)
        red = _red_light_boxes(sim_state, w, t, scenario.route_lanes, scenario.lane_map) if scenario.lane_map else []
        ego_box = np.array([cx, cy, yaw, length, width])
        obs = _build_observation(sim_state, w, t, ego_idx, ego_box, scenario.collided, red, self.cv_velocity_ratios)

        ego_xyz = np.array([cx, cy, get(traj.z)])
        speed_limit = _speed_limit(sim_state, w, ego_xyz, scenario.expert_max_speed)
        generator = _Generator(scenario.paths, obs, ego_state, vehicle, np.array(SPEED_LIMIT_FRACTIONS) * speed_limit)
        proposals = np.arange(len(generator.lateral))
        generator.unroll(proposals, 1, PROPOSAL_STEPS)
        simulated = _simulate(generator.se2[:, : PROPOSAL_STEPS + 1], ego_state, vehicle)

        def offroad_fn(boxes):
            distances = self.scorers[w]._offroad_corner_distances(jnp.asarray(boxes, dtype=jnp.float32), sim_state, t, world_idx=w)
            if distances is None:
                return np.zeros(boxes.shape[:2], dtype=bool)
            return np.asarray(distances > 0.0).any(-1)

        scores = _score(simulated, obs, scenario.paths[0], scenario.lane_map, vehicle, offroad_fn)
        best = int(np.argmax(scores.total))

        brake = scores.collision_time[best] <= EMERGENCY_TIME_TO_INFRACTION and abs(v_long) <= EMERGENCY_MAX_EGO_SPEED
        if brake:
            poses = self._brake_poses(cx, cy, yaw, v_long, accel, r)
        else:
            generator.unroll([best], PROPOSAL_STEPS + 1, TRAJECTORY_STEPS)
            poses = generator.se2[best : best + 1]
        executed = _simulate(poses, ego_state, vehicle)[0]
        self.last_context[w] = PlanContext(vehicle, ego_state, obs, scenario.paths[0], scenario.lane_map, offroad_fn,
                                           executed)
        return self._to_world(executed, r), float(scores.total[best]), bool(brake), scenario.goal_lane_reached

    @staticmethod
    def _brake_poses(cx, cy, yaw, v, a, rear_axle_to_center):
        """PDMEmergencyBrake trajectory (centre poses at a constant correcting speed), as rear-axle poses."""
        if v > 0.2:
            error = max(min(10.0 * -v, EMERGENCY_MAX_LONG_ACCEL), EMERGENCY_MIN_LONG_ACCEL)
            correcting_velocity = 11 / 10 * (v + error)
        else:
            correcting_velocity = max(min(4 * -v + 1 * -a, EMERGENCY_MAX_LONG_ACCEL), EMERGENCY_MIN_LONG_ACCEL)
        s = correcting_velocity * np.arange(TRAJECTORY_STEPS + 1) * DT - rear_axle_to_center
        return np.stack([cx + s * np.cos(yaw), cy + s * np.sin(yaw), np.full_like(s, yaw)], axis=-1)[None]

    @staticmethod
    def _to_world(states, rear_axle_to_center):
        """Rear-axle states [T, N_STATE] -> Waymax box-centre (x, y, yaw, vel_x, vel_y) [T, 5]."""
        h = states[:, HEADING]
        c, s = np.cos(h), np.sin(h)
        x = states[:, X] + rear_axle_to_center * c
        y = states[:, Y] + rear_axle_to_center * s
        omega = states[:, YAW_RATE] * rear_axle_to_center
        vx = states[:, VX] * c - omega * s
        vy = states[:, VX] * s + omega * c
        return np.stack([x, y, h, vx, vy], axis=-1).astype(np.float32)
