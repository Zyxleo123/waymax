"""Diffusion Planner (Zheng et al., ICLR 2025; github.com/ZhengYinan-AIR/Diffusion-Planner @ a3a621f) inputs from WOMD.

Ports ``diffusion_planner/data_process`` (nuPlan-specific) to Waymax states, producing the exact arrays their
``DiffusionPlannerData`` / ``DataProcessor.observation_adapter`` produce, so their model, loss, augmentation and
training loop run unchanged. The same ``build_sample`` is used for the training data (``convert_womd.py``) and
in closed loop (``planner/diffusion_planner_baseline.py``).

As in their code: up to 32 agents (vehicle / pedestrian / cyclist, at most 10 pedestrians + cyclists unless
there is room) present at the current step, nearest first, with 2 s of history (21 frames) where frames an
agent is missing take the next later observed state; 5 static objects; the 70 lanes nearest to ego within 100 m,
20 points each, with (point, vector to next point, offset to left / right boundary, traffic light one-hot);
up to 25 of those that are on the route; everything in ego's frame.

WOMD differences:
  - reference point: Waymax box centre (nuPlan: rear axle), for inputs and targets alike;
  - frames before the start of the log are filled like missing frames (with the earliest observed state);
  - lanes: lane-graph centrelines cut into pieces of at most LANE_SEGMENT_M (WOMD lanes run for hundreds of
    metres; nuPlan lanes and lane connectors are short), resampled to 20 points;
  - lane boundaries: per centreline point, the nearest road line / road edge point on each side within
    BOUNDARY_SEARCH_M (the tf.Example format has no lane-to-boundary links), else +-DEFAULT_HALF_WIDTH_M;
  - speed limits: not in the tf.Example format, so ``*_has_speed_limit`` is False everywhere;
  - route: the lanes the logged ego drives on (map-matched) plus their left / right neighbours, the analogue
    of nuPlan's route roadblocks (all lanes of the roadblocks on the expert's route);
  - static objects: Waymax type OTHER, encoded as nuPlan's GENERIC_OBJECT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

# --- Diffusion Planner defaults (train_predictor.py / data_processor.py) ---
AGENT_NUM = 32
STATIC_NUM = 5
MAX_PED_BIKE = 10
LANE_NUM = 70
LANE_LEN = 20
ROUTE_NUM = 25
ROUTE_LEN = 20
RADIUS = 100.0
TIME_LEN = 21  # 2 s of history at 10 Hz
PACIFICA_WHEEL_BASE = 3.089

# --- WOMD stand-ins ---
FUTURE_LEN = 50  # 5 s at 10 Hz, the horizon of the waymax_rs diffusion policy
LANE_SEGMENT_M = 30.0
BOUNDARY_SEARCH_M = 3.5
DEFAULT_HALF_WIDTH_M = 1.75
ROUTE_MATCH_M = 2.0
ROUTE_MATCH_HEADING = np.pi / 4

VEHICLE, PEDESTRIAN, CYCLIST, OTHER = 1, 2, 3, 4  # Waymax object types
BOUNDARY_TYPES = tuple(range(6, 14)) + (15, 16)  # road lines, road edges
# Waymax traffic-light states -> nuPlan one-hot [green, yellow, red, unknown]
TL_GREEN, TL_YELLOW, TL_RED = (3, 6), (2, 5, 8), (1, 4, 7)


def _wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def _to_local_xy(xy, origin):
    """World points [..., 2] into the frame of origin = (x, y, heading)."""
    c, s = np.cos(origin[2]), np.sin(origin[2])
    d = xy - origin[:2]
    return np.stack([d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c], axis=-1)


def _to_local_vec(v, heading):
    c, s = np.cos(heading), np.sin(heading)
    return np.stack([v[..., 0] * c + v[..., 1] * s, -v[..., 0] * s + v[..., 1] * c], axis=-1)


def _resample(line, num):
    """Their _interpolate_points: num points evenly spaced along the polyline's length."""
    seg = np.linalg.norm(np.diff(line, axis=0), axis=-1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] <= 0:
        return np.repeat(line[:1], num, axis=0)
    d = np.linspace(0.0, s[-1], num)
    return np.stack([np.interp(d, s, line[:, k]) for k in range(2)], axis=-1)


@dataclass
class WorldLog:
    """One world of a Waymax SimulatorState as numpy arrays [N, T] (log_trajectory) + map."""

    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    length: np.ndarray
    width: np.ndarray
    valid: np.ndarray
    types: np.ndarray  # [N]
    ego: int
    rg_xyz: np.ndarray  # roadgraph points
    rg_types: np.ndarray
    rg_valid: np.ndarray
    tl_state: np.ndarray  # [L, T]
    tl_lane: np.ndarray
    tl_valid: np.ndarray

    @classmethod
    def from_state(cls, state, w: int) -> "WorldLog":
        tr = state.log_trajectory
        g = lambda a: np.asarray(a[w])
        rg, tl = state.roadgraph_points, state.log_traffic_light
        is_sdc = g(state.object_metadata.is_sdc).astype(bool)
        return cls(
            *(g(a).astype(np.float64) for a in (tr.x, tr.y, tr.yaw, tr.vel_x, tr.vel_y, tr.length, tr.width)),
            valid=g(tr.valid).astype(bool),
            types=g(state.object_metadata.object_types).astype(int),
            ego=int(np.argmax(is_sdc)),
            rg_xyz=g(rg.xyz).astype(np.float64),
            rg_types=g(rg.types).astype(int),
            rg_valid=g(rg.valid).astype(bool),
            tl_state=g(tl.state).astype(int),
            tl_lane=g(tl.lane_ids).astype(int),
            tl_valid=g(tl.valid).astype(bool),
        )


class ScenarioMap:
    """Lane segments with boundaries, and the route, of one scenario (built once, from the original log)."""

    def __init__(self, lane_graph, log: WorldLog):
        xy = np.asarray(lane_graph.nodes_xyz, dtype=np.float64)[:, :2]
        node_lane = np.asarray(lane_graph.node_lane_ids, dtype=np.int64)
        ranges = {int(l): (int(a), int(b)) for l, (a, b) in lane_graph.lane_id_to_node_range.items() if b - a >= 2}
        heading = np.zeros(len(xy))
        for a, b in ranges.values():
            d = np.diff(xy[a:b], axis=0)
            h = np.arctan2(d[:, 1], d[:, 0])
            heading[a : b - 1], heading[b - 1] = h, h[-1]

        bmask = log.rg_valid & np.isin(log.rg_types, BOUNDARY_TYPES)
        bxy = log.rg_xyz[bmask, :2]
        btree = cKDTree(bxy) if len(bxy) else None

        # lane segments: centre, left, right [S, n_i, 2] (variable length), lane id
        self.centre, self.left, self.right, self.lane_id = [], [], [], []
        for lane, (a, b) in ranges.items():
            pts, hs = xy[a:b], heading[a:b]
            left, right = self._boundaries(pts, hs, bxy, btree)
            s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=-1))]
            n_seg = max(1, int(np.ceil(s[-1] / LANE_SEGMENT_M)))
            cuts = np.searchsorted(s, np.linspace(0.0, s[-1], n_seg + 1)[1:-1])
            for idx in np.split(np.arange(len(pts)), cuts):
                idx = np.r_[idx, idx[-1] + 1] if idx[-1] + 1 < len(pts) else idx  # share the joint point
                if len(idx) < 2:
                    continue
                self.centre.append(pts[idx])
                self.left.append(left[idx])
                self.right.append(right[idx])
                self.lane_id.append(lane)
        self.lane_id = np.array(self.lane_id, dtype=np.int64)
        self.route = self._route(lane_graph, xy, node_lane, heading, ranges, log)

    @staticmethod
    def _boundaries(pts, hs, bxy, btree):
        normal = np.stack([-np.sin(hs), np.cos(hs)], -1)  # left
        left = pts + DEFAULT_HALF_WIDTH_M * normal
        right = pts - DEFAULT_HALF_WIDTH_M * normal
        if btree is None:
            return left, right
        k = min(32, len(bxy))
        d, n = btree.query(pts, k=k, distance_upper_bound=BOUNDARY_SEARCH_M)
        d, n = d.reshape(len(pts), k), n.reshape(len(pts), k)
        hit = np.isfinite(d)
        cand = bxy[np.where(hit, n, 0)]  # [P, k, 2]
        lateral = np.sum((cand - pts[:, None]) * normal[:, None], -1)
        for sign, out in ((1.0, left), (-1.0, right)):
            ok = hit & (sign * lateral > 0.3)
            dist = np.where(ok, d, np.inf)
            j = np.argmin(dist, axis=1)
            found = np.isfinite(dist[np.arange(len(pts)), j])
            out[found] = cand[np.arange(len(pts)), j][found]
        return left, right

    @staticmethod
    def _route(lane_graph, xy, node_lane, heading, ranges, log: WorldLog):
        """Lanes the logged ego drives on, plus their left / right neighbours."""
        e = log.ego
        v = log.valid[e]
        pts = np.stack([log.x[e, v], log.y[e, v]], -1)
        yaws = log.yaw[e, v]
        route = set()
        if len(xy) and len(pts):
            tree = cKDTree(xy)
            k = min(16, len(xy))
            d, n = tree.query(pts, k=k, distance_upper_bound=ROUTE_MATCH_M)
            d, n = d.reshape(len(pts), k), n.reshape(len(pts), k)
            ok = np.isfinite(d)
            n = np.where(ok, n, 0)
            ok &= np.abs(_wrap(heading[n] - yaws[:, None])) < ROUTE_MATCH_HEADING
            dist = np.where(ok, d, np.inf)
            j = np.argmin(dist, axis=1)
            hit = np.isfinite(dist[np.arange(len(pts)), j])
            route = {int(node_lane[n[i, j[i]]]) for i in np.flatnonzero(hit)}
        neighbours = set()
        for edges in (lane_graph.left_neighbor_edges, lane_graph.right_neighbor_edges):
            for src, dst in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
                if int(node_lane[src]) in route:
                    neighbours.add(int(node_lane[dst]))
        return {lane for lane in route | neighbours if lane in ranges}


def _agent_history(log: WorldLog, t: int, objs: np.ndarray):
    """[A, TIME_LEN, 8] (x, y, cos, sin, vx, vy, width, length) in world frame, missing frames filled with the
    next later observed state (their _pad_agent_states(reverse=True)); frames before the log start likewise."""
    steps = np.arange(t - TIME_LEN + 1, t + 1)
    out = np.zeros((len(objs), TIME_LEN, 8))
    for i, o in enumerate(objs):
        last = None
        for k in range(TIME_LEN - 1, -1, -1):
            tt = steps[k]
            if tt >= 0 and log.valid[o, tt]:
                last = (log.x[o, tt], log.y[o, tt], log.yaw[o, tt], log.vx[o, tt], log.vy[o, tt], log.width[o, tt], log.length[o, tt])
            x, y, h, vx, vy, wd, ln = last
            out[i, k] = (x, y, np.cos(h), np.sin(h), vx, vy, wd, ln)
    return out


def _select_agents(dist, types):
    """Their agent selection: nearest first, at most MAX_PED_BIKE pedestrians/cyclists unless slots remain."""
    order = np.argsort(dist, kind="stable")
    ped_bike = [i for i in order if types[i] in (PEDESTRIAN, CYCLIST)]
    vehicles = [i for i in order if types[i] == VEHICLE]
    if len(ped_bike) + len(vehicles) <= AGENT_NUM:
        return list(order[:AGENT_NUM])
    selected = ped_bike[:MAX_PED_BIKE] + vehicles
    remaining = AGENT_NUM - len(selected)
    if remaining > 0:
        selected += ped_bike[MAX_PED_BIKE:][:remaining]
    return sorted(selected, key=lambda i: dist[i])[:AGENT_NUM]


def ego_pose(log: WorldLog, t: int) -> np.ndarray:
    e = log.ego
    return np.array([log.x[e, t], log.y[e, t], log.yaw[e, t]])


def ego_current_state(log: WorldLog, t: int) -> np.ndarray:
    """Their calculate_additional_ego_states: [0, 0, 1, 0, vx, vy, ax, ay, steering, yaw rate] in ego frame."""
    e = log.ego
    h = log.yaw[e, t]
    v = _to_local_vec(np.array([log.vx[e, t], log.vy[e, t]]), h)
    out = np.zeros(10, dtype=np.float32)
    out[2] = 1.0
    out[4:6] = v
    if t > 0 and log.valid[e, t - 1]:
        v_prev = _to_local_vec(np.array([log.vx[e, t - 1], log.vy[e, t - 1]]), log.yaw[e, t - 1])
        out[6:8] = (v - v_prev) / 0.1
        yaw_rate = _wrap(h - log.yaw[e, t - 1]) / 0.1
        if abs(v[0]) >= 0.2:
            out[8] = np.clip(np.arctan(yaw_rate * PACIFICA_WHEEL_BASE / abs(v[0])), -2 / 3 * np.pi, 2 / 3 * np.pi)
            out[9] = np.clip(yaw_rate, -0.95, 0.95)
    return out


def build_inputs(log: WorldLog, smap: ScenarioMap, t: int, ego_state: np.ndarray | None = None):
    """Model inputs at step t (unnormalised, unbatched), plus the selected agent indices."""
    origin = ego_pose(log, t)
    e = log.ego

    # --- agents ---
    present = log.valid[:, t] & np.isin(log.types, (VEHICLE, PEDESTRIAN, CYCLIST))
    present[e] = False
    cand = np.flatnonzero(present)
    dist = np.hypot(log.x[cand, t] - origin[0], log.y[cand, t] - origin[1])
    chosen = cand[_select_agents(dist, log.types[cand])] if len(cand) else np.zeros(0, int)
    hist = _agent_history(log, t, chosen)
    agents = np.zeros((AGENT_NUM, TIME_LEN, 11), dtype=np.float32)
    if len(chosen):
        local = hist.copy()
        local[..., :2] = _to_local_xy(hist[..., :2], origin)
        local[..., 2:4] = _to_local_vec(hist[..., 2:4], origin[2])
        local[..., 4:6] = _to_local_vec(hist[..., 4:6], origin[2])
        agents[: len(chosen), :, :8] = local
        onehot = np.stack([log.types[chosen] == VEHICLE, log.types[chosen] == PEDESTRIAN, log.types[chosen] == CYCLIST], -1)
        agents[: len(chosen), :, 8:] = onehot[:, None, :]

    # --- static objects (Waymax OTHER -> GENERIC_OBJECT) ---
    static = np.zeros((STATIC_NUM, 10), dtype=np.float32)
    so = np.flatnonzero(log.valid[:, t] & (log.types == OTHER))
    if len(so):
        sxy = _to_local_xy(np.stack([log.x[so, t], log.y[so, t]], -1), origin)
        order = np.argsort(np.linalg.norm(sxy, axis=-1), kind="stable")[:STATIC_NUM]
        sh = log.yaw[so[order], t] - origin[2]
        static[: len(order), :6] = np.stack([sxy[order, 0], sxy[order, 1], np.cos(sh), np.sin(sh), log.width[so[order], t], log.length[so[order], t]], -1)
        static[: len(order), 9] = 1.0

    # --- lanes ---
    tl_by_lane = {}
    for k in np.flatnonzero(log.tl_valid[:, t]):
        s = log.tl_state[k, t]
        tl_by_lane[int(log.tl_lane[k, t])] = 0 if s in TL_GREEN else 1 if s in TL_YELLOW else 2 if s in TL_RED else 3
    seg_dist = np.array([np.linalg.norm(c - origin[:2], axis=-1).min() for c in smap.centre]) if smap.centre else np.zeros(0)
    near = np.flatnonzero(seg_dist <= RADIUS)
    near = near[np.argsort(seg_dist[near], kind="stable")][:LANE_NUM]
    lanes = np.zeros((LANE_NUM, LANE_LEN, 12), dtype=np.float32)
    on_route = []
    for i, s in enumerate(near):
        centre = _to_local_xy(_resample(smap.centre[s], LANE_LEN), origin)
        left = _to_local_xy(_resample(smap.left[s], LANE_LEN), origin)
        right = _to_local_xy(_resample(smap.right[s], LANE_LEN), origin)
        if np.linalg.norm(left[-1] - centre[0]) < np.linalg.norm(left[0] - centre[0]):
            left = left[::-1]
        if np.linalg.norm(right[-1] - centre[0]) < np.linalg.norm(right[0] - centre[0]):
            right = right[::-1]
        vec = np.vstack([centre[1:] - centre[:-1], np.zeros((1, 2))])
        tl = np.zeros((LANE_LEN, 4))
        tl[:, tl_by_lane.get(int(smap.lane_id[s]), 3)] = 1.0
        lanes[i] = np.concatenate([centre, vec, left - centre, right - centre, tl], -1)
        on_route.append(int(smap.lane_id[s]) in smap.route)
    route_lanes = np.zeros((ROUTE_NUM, ROUTE_LEN, 12), dtype=np.float32)
    route_rows = [i for i, r in enumerate(on_route) if r][:ROUTE_NUM]
    route_lanes[: len(route_rows)] = lanes[route_rows]

    inputs = {
        "ego_current_state": ego_current_state(log, t) if ego_state is None else ego_state,
        "neighbor_agents_past": agents,
        "static_objects": static,
        "lanes": lanes,
        "lanes_speed_limit": np.zeros((LANE_NUM, 1), dtype=np.float32),
        "lanes_has_speed_limit": np.zeros((LANE_NUM, 1), dtype=bool),
        "route_lanes": route_lanes,
        "route_lanes_speed_limit": np.zeros((ROUTE_NUM, 1), dtype=np.float32),
        "route_lanes_has_speed_limit": np.zeros((ROUTE_NUM, 1), dtype=bool),
    }
    return inputs, chosen


def build_targets(log: WorldLog, t: int, chosen: np.ndarray):
    """Ego future [FUTURE_LEN, 3] and neighbour futures [AGENT_NUM, FUTURE_LEN, 3] (x, y, heading) in ego's frame
    at t; neighbours zero where not observed (their agent_future_process)."""
    origin = ego_pose(log, t)
    e = log.ego
    steps = np.arange(t + 1, t + 1 + FUTURE_LEN)
    ego = np.zeros((FUTURE_LEN, 3), dtype=np.float32)
    ego[:, :2] = _to_local_xy(np.stack([log.x[e, steps], log.y[e, steps]], -1), origin)
    ego[:, 2] = _wrap(log.yaw[e, steps] - origin[2])
    neigh = np.zeros((AGENT_NUM, FUTURE_LEN, 3), dtype=np.float32)
    for i, o in enumerate(chosen):
        v = log.valid[o, steps]
        xy = _to_local_xy(np.stack([log.x[o, steps], log.y[o, steps]], -1), origin)
        neigh[i, v, :2] = xy[v]
        neigh[i, v, 2] = _wrap(log.yaw[o, steps] - origin[2])[v]
    return ego, neigh


def valid_anchors(log: WorldLog) -> np.ndarray:
    """Steps with ego observed at t and over the whole FUTURE_LEN horizon (the waymax_rs policy's anchor rule)."""
    v = log.valid[log.ego]
    n = len(v)
    return np.array([t for t in range(n - FUTURE_LEN) if v[t : t + FUTURE_LEN + 1].all()], dtype=int)


def build_sample(log: WorldLog, smap: ScenarioMap, t: int) -> dict:
    """One training sample in the layout of their npz files (keys read by DiffusionPlannerData)."""
    inputs, chosen = build_inputs(log, smap, t)
    ego_future, neigh_future = build_targets(log, t, chosen)
    return {**inputs, "ego_agent_future": ego_future, "neighbor_agents_future": neigh_future}
