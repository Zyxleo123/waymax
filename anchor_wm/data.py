"""WOMD cache shards -> AnchorWM batches (model inputs, forecasting targets, label geometry).

Model inputs (`agent_history`, `map_features`, ...) are built exactly as in
`multiFuture_GaussianWorldModel/data/world_model_cache_loader.py`. The map and traffic-light
builders below are copied from its `preprocess_cached.py` (kept identical so the warm-started
encoder sees the same features); waymax constants are hard-coded so training needs no waymax.

Extra keys (all in the ego frame at the anchor step, metres / radians):
  target_valid_all [S,N,T]    forecasting targets incl. the SDC (MF's `target_valid` excludes it)
  current_yaw_local [S,N]     agent yaw relative to the ego
  ego_idx [S], ego_lw [S,2], ego_vel [S,2]
  goal_xy [S,2]               last valid logged ego position of the episode (as es_discriminator)
  fut_xy [S,N,T,2], fut_yaw [S,N,T], fut_lw [S,N,T,2], fut_valid [S,N,T]
                              logged boxes of every non-ego object (log replay, any type)
  ego_future_xy [S,T,2], ego_future_yaw [S,T], ego_future_valid [S,T]
  edge_xy/edge_dir/edge_prev_dir [S,E,2], edge_same_prev/edge_valid [S,E]
                              road-edge points for offroad labels; `edge_prev_dir` / `edge_same_prev`
                              are the preceding point of the full edge list, as Waymax's
                              `compute_signed_distance_to_nearest_road_edge_point` uses them
  tok_edge_xy/tok_edge_dir [S,Et,2], tok_edge_valid [S,Et]   scorer road-edge tokens
  lane_xy/lane_dir [S,L,2], lane_id/lane_valid [S,L]         lane centerline points (route context)
  origin_xy [S,2], origin_yaw [S]   world pose of the ego frame
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from anchor_wm.config import DYNAMIC_TYPES, ROAD_EDGE_TYPES, AnchorWMConfig

CACHE_FIELD_NAMES = (
    "object_metadata_is_sdc", "object_metadata_object_types",
    "log_trajectory_x", "log_trajectory_y", "log_trajectory_yaw", "log_trajectory_valid",
    "log_trajectory_vel_x", "log_trajectory_vel_y", "log_trajectory_length", "log_trajectory_width",
    "log_traffic_light_state", "log_traffic_light_x", "log_traffic_light_y", "log_traffic_light_valid",
    "roadgraph_points_x", "roadgraph_points_y", "roadgraph_points_ids", "roadgraph_points_types",
    "roadgraph_points_valid", "roadgraph_points_dir_x", "roadgraph_points_dir_y",
)


def load_npz(cache_path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(cache_path), allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    missing = [key for key in CACHE_FIELD_NAMES if key not in payload]
    if missing:
        raise KeyError(f"Cache file {cache_path} is missing required fields: {missing}")
    return payload


def payload_from_sim_state(state) -> dict[str, np.ndarray]:
    """Waymax SimulatorState (batched) -> the cache-shard payload layout used by `build_arrays`."""
    log, md, rg, tl = state.log_trajectory, state.object_metadata, state.roadgraph_points, state.log_traffic_light
    fields = {
        "log_trajectory_x": log.x, "log_trajectory_y": log.y, "log_trajectory_yaw": log.yaw,
        "log_trajectory_valid": log.valid, "log_trajectory_vel_x": log.vel_x, "log_trajectory_vel_y": log.vel_y,
        "log_trajectory_length": log.length, "log_trajectory_width": log.width,
        "object_metadata_is_sdc": md.is_sdc, "object_metadata_object_types": md.object_types,
        "log_traffic_light_state": tl.state, "log_traffic_light_x": tl.x, "log_traffic_light_y": tl.y,
        "log_traffic_light_valid": tl.valid,
        "roadgraph_points_x": rg.x, "roadgraph_points_y": rg.y, "roadgraph_points_ids": rg.ids,
        "roadgraph_points_types": rg.types, "roadgraph_points_valid": rg.valid,
        "roadgraph_points_dir_x": rg.dir_x, "roadgraph_points_dir_y": rg.dir_y,
    }
    return {k: np.asarray(v) for k, v in fields.items()}


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rotate_xy_np(xy: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """Rotate xy by -heading (world -> local). heading broadcasts over leading dimensions."""
    c = np.cos(-heading)
    s = np.sin(-heading)
    x = xy[..., 0]
    y = xy[..., 1]
    return np.stack([x * c - y * s, x * s + y * c], axis=-1)


# ---- copied from multiFuture_GaussianWorldModel/data/preprocess_cached.py --------------------

def _gather_time(arr_bnt: np.ndarray, t_b: np.ndarray) -> np.ndarray:
    idx = np.broadcast_to(t_b[:, None, None], (arr_bnt.shape[0], arr_bnt.shape[1], 1))
    return np.take_along_axis(arr_bnt, idx, axis=2)[:, :, 0]


def build_map_features(state: Mapping[str, np.ndarray], origin_xy_b2: np.ndarray, heading_b: np.ndarray,
                       cfg: AnchorWMConfig) -> tuple[np.ndarray, np.ndarray]:
    x_bm = np.asarray(state["roadgraph_points_x"])
    y_bm = np.asarray(state["roadgraph_points_y"])
    dir_x_bm = np.asarray(state["roadgraph_points_dir_x"])
    dir_y_bm = np.asarray(state["roadgraph_points_dir_y"])
    types_bm = np.asarray(state["roadgraph_points_types"]).astype(np.int32)
    valid_bm = np.asarray(state["roadgraph_points_valid"]).astype(bool)
    ids_bm = np.asarray(state["roadgraph_points_ids"]).astype(np.int32)

    max_segments = int(cfg.max_map_segments)
    max_points_per_segment = int(cfg.max_points_per_segment)
    feat_dim = 2 + 2 + int(cfg.num_map_type_classes)
    sentinel_id = np.iinfo(np.int32).max

    feats = np.zeros((x_bm.shape[0], max_segments, max_points_per_segment, feat_dim), dtype=np.float32)
    valids = np.zeros((x_bm.shape[0], max_segments, max_points_per_segment), dtype=bool)

    for batch_idx in range(x_bm.shape[0]):
        xy = np.stack([x_bm[batch_idx], y_bm[batch_idx]], axis=-1)
        dirs = np.stack([dir_x_bm[batch_idx], dir_y_bm[batch_idx]], axis=-1)
        types = types_bm[batch_idx]
        point_ids = ids_bm[batch_idx]
        valid = valid_bm[batch_idx]

        centered = xy - origin_xy_b2[batch_idx][None, :]
        dist = np.linalg.norm(centered, axis=-1)
        valid_and_range = valid & (dist < float(cfg.max_range))
        if not np.any(valid_and_range):
            continue

        # Group points by polyline id; stable sort keeps along-polyline order.
        sort_ids = np.where(valid_and_range, point_ids, sentinel_id)
        sort_dist = np.where(valid_and_range, dist, np.inf)
        order = np.argsort(sort_ids, kind="stable")
        sort_ids = sort_ids[order]
        sort_dist = sort_dist[order]
        sort_xy = xy[order]
        sort_dirs = dirs[order]
        sort_types = types[order]

        unique_ids, starts, counts = np.unique(sort_ids, return_index=True, return_counts=True)
        keep = unique_ids != sentinel_id
        unique_ids, starts, counts = unique_ids[keep], starts[keep], counts[keep]
        if unique_ids.size == 0:
            continue

        # Keep the polylines closest to the ego, nearest first.
        min_dist = np.minimum.reduceat(sort_dist, starts)
        pick = np.argsort(min_dist, kind="stable")[:max_segments]
        unique_ids, starts, counts = unique_ids[pick], starts[pick], counts[pick]

        point_rank = np.arange(max_points_per_segment, dtype=np.int32)
        stride_den = max(max_points_per_segment - 1, 1)
        for seg_idx, (count, start) in enumerate(zip(counts, starts)):
            count = int(count)
            start = int(start)
            # Subsample evenly along the polyline when it has too many points.
            if count > max_points_per_segment:
                point_idx = start + (point_rank * (count - 1)) // stride_den
            else:
                point_idx = start + point_rank[:count]
            span = point_idx.shape[0]

            seg_xy = np.zeros((max_points_per_segment, 2), dtype=np.float32)
            seg_dirs = np.zeros((max_points_per_segment, 2), dtype=np.float32)
            seg_types = np.zeros((max_points_per_segment,), dtype=np.int32)
            seg_xy[:span] = sort_xy[point_idx]
            seg_dirs[:span] = sort_dirs[point_idx]
            seg_types[:span] = sort_types[point_idx]

            xy_rel = rotate_xy_np(seg_xy - origin_xy_b2[batch_idx][None, :], heading_b[batch_idx]) / float(cfg.max_range)
            dir_rel = rotate_xy_np(seg_dirs, heading_b[batch_idx])
            dir_norm = np.linalg.norm(dir_rel, axis=-1, keepdims=True)
            dir_rel = dir_rel / np.maximum(dir_norm, 1e-6)

            ty_safe = np.where(
                (seg_types >= 0) & (seg_types < int(cfg.num_map_type_classes)),
                seg_types,
                np.full_like(seg_types, int(cfg.map_unknown_type_index)),
            )
            type_oh = np.eye(int(cfg.num_map_type_classes), dtype=np.float32)[ty_safe]
            seg_feat = np.concatenate([xy_rel.astype(np.float32), dir_rel.astype(np.float32), type_oh], axis=-1)
            seg_valid = point_rank < span
            seg_feat = np.where(seg_valid[:, None], seg_feat, 0.0)

            feats[batch_idx, seg_idx] = seg_feat
            valids[batch_idx, seg_idx] = seg_valid
    return feats, valids


def build_tl_features(state: Mapping[str, np.ndarray], anchor_step_b: np.ndarray, origin_xy_b2: np.ndarray,
                      heading_b: np.ndarray, cfg: AnchorWMConfig) -> tuple[np.ndarray, np.ndarray]:
    x_bl = _gather_time(np.asarray(state["log_traffic_light_x"]), anchor_step_b)
    y_bl = _gather_time(np.asarray(state["log_traffic_light_y"]), anchor_step_b)
    st_bl = _gather_time(np.asarray(state["log_traffic_light_state"]), anchor_step_b).astype(np.int32)
    va_bl = _gather_time(np.asarray(state["log_traffic_light_valid"]), anchor_step_b).astype(bool)

    l = min(int(cfg.max_tl_points), int(x_bl.shape[1]))
    x_bl, y_bl, st_bl, va_bl = x_bl[:, :l], y_bl[:, :l], st_bl[:, :l], va_bl[:, :l]

    xy = np.stack([x_bl, y_bl], axis=-1)
    xy_rel = rotate_xy_np(xy - origin_xy_b2[:, None, :], heading_b[:, None]) / float(cfg.max_range)
    st_oh = np.eye(9, dtype=np.float32)[np.clip(st_bl, 0, 8)]
    feat = np.concatenate([xy_rel.astype(np.float32), st_oh], axis=-1)
    feat = np.where(va_bl[:, :, None], feat, 0.0)

    pad = int(cfg.max_tl_points) - l
    if pad > 0:
        feat = np.pad(feat, ((0, 0), (0, pad), (0, 0)))
        va_bl = np.pad(va_bl, ((0, 0), (0, pad)), constant_values=False)
    return feat.astype(np.float32), va_bl.astype(bool)

# ---- end of copied code ------------------------------------------------------------------


def build_edges(payload: Mapping[str, np.ndarray], origin_xy: np.ndarray, heading: np.ndarray,
                cfg: AnchorWMConfig) -> dict[str, np.ndarray]:
    """Road-edge points in the ego frame: a dense set for offroad labels and a strided subset as
    scorer tokens. Each point keeps the direction of, and the same-polyline flag with, its
    predecessor in the full road-edge list (Waymax's "prior point" rule), so subsampling keeps the
    sign logic of the nearest point exact."""
    s = origin_xy.shape[0]
    e, et = int(cfg.num_label_edges), int(cfg.num_edge_tokens)
    out = {
        "edge_xy": np.zeros((s, e, 2), np.float32), "edge_dir": np.zeros((s, e, 2), np.float32),
        "edge_prev_dir": np.zeros((s, e, 2), np.float32), "edge_same_prev": np.zeros((s, e), bool),
        "edge_valid": np.zeros((s, e), bool),
        "tok_edge_xy": np.zeros((s, et, 2), np.float32), "tok_edge_dir": np.zeros((s, et, 2), np.float32),
        "tok_edge_valid": np.zeros((s, et), bool),
    }
    back, ahead, half = cfg.edge_box
    tback, tahead, thalf = cfg.edge_token_box
    for b in range(s):
        keep = (np.asarray(payload["roadgraph_points_valid"][b]).astype(bool)
                & np.isin(payload["roadgraph_points_types"][b], ROAD_EDGE_TYPES))
        idx = np.flatnonzero(keep)
        if len(idx) == 0:
            continue
        xy = np.stack([payload["roadgraph_points_x"][b][idx], payload["roadgraph_points_y"][b][idx]], -1)
        dirs = np.stack([payload["roadgraph_points_dir_x"][b][idx], payload["roadgraph_points_dir_y"][b][idx]], -1)
        ids = np.asarray(payload["roadgraph_points_ids"][b])[idx]
        same_prev = np.zeros(len(idx), bool)
        same_prev[1:] = ids[1:] == ids[:-1]
        prev_dirs = np.concatenate([dirs[:1], dirs[:-1]], 0)

        local = rotate_xy_np(xy.astype(np.float64) - origin_xy[b], heading[b])
        in_box = (local[:, 0] >= back) & (local[:, 0] <= ahead) & (np.abs(local[:, 1]) <= half)
        sel = np.flatnonzero(in_box)
        if len(sel) > e:
            sel = sel[::math.ceil(len(sel) / e)][:e]
        k = len(sel)
        out["edge_xy"][b, :k] = local[sel]
        out["edge_dir"][b, :k] = rotate_xy_np(dirs[sel], heading[b])
        out["edge_prev_dir"][b, :k] = rotate_xy_np(prev_dirs[sel], heading[b])
        out["edge_same_prev"][b, :k] = same_prev[sel]
        out["edge_valid"][b, :k] = True

        loc = local[sel]
        tsel = np.flatnonzero((loc[:, 0] >= tback) & (loc[:, 0] <= tahead) & (np.abs(loc[:, 1]) <= thalf))
        if len(tsel) > et:
            tsel = tsel[np.linspace(0, len(tsel) - 1, et).round().astype(np.int64)]
        kt = len(tsel)
        tdir = out["edge_dir"][b, tsel]
        out["tok_edge_xy"][b, :kt] = loc[tsel]
        out["tok_edge_dir"][b, :kt] = tdir / np.maximum(np.linalg.norm(tdir, axis=-1, keepdims=True), 1e-6)
        out["tok_edge_valid"][b, :kt] = True
    return out


LANE_TYPES = (1, 2, 3)                       # Waymax lane centers: freeway, surface street, bike lane
LANE_BOX = (-30.0, 150.0, 60.0)              # behind, ahead, half-width (m) in the ego frame
NUM_LANE_POINTS = 1024


def build_lanes(payload: Mapping[str, np.ndarray], origin_xy: np.ndarray, heading: np.ndarray) -> dict[str, np.ndarray]:
    """Lane centerline points (ego frame, strided to NUM_LANE_POINTS) for route context:
    lane_xy / lane_dir [S,L,2], lane_id [S,L], lane_valid [S,L]."""
    s = origin_xy.shape[0]
    out = {"lane_xy": np.zeros((s, NUM_LANE_POINTS, 2), np.float32),
           "lane_dir": np.zeros((s, NUM_LANE_POINTS, 2), np.float32),
           "lane_id": np.full((s, NUM_LANE_POINTS), -1, np.int32), "lane_valid": np.zeros((s, NUM_LANE_POINTS), bool)}
    back, ahead, half = LANE_BOX
    for b in range(s):
        keep = (np.asarray(payload["roadgraph_points_valid"][b]).astype(bool)
                & np.isin(payload["roadgraph_points_types"][b], LANE_TYPES))
        idx = np.flatnonzero(keep)
        if len(idx) == 0:
            continue
        xy = np.stack([payload["roadgraph_points_x"][b][idx], payload["roadgraph_points_y"][b][idx]], -1)
        local = rotate_xy_np(xy.astype(np.float64) - origin_xy[b], heading[b])
        sel = np.flatnonzero((local[:, 0] >= back) & (local[:, 0] <= ahead) & (np.abs(local[:, 1]) <= half))
        if len(sel) > NUM_LANE_POINTS:
            sel = sel[::math.ceil(len(sel) / NUM_LANE_POINTS)][:NUM_LANE_POINTS]
        k = len(sel)
        dirs = np.stack([payload["roadgraph_points_dir_x"][b][idx], payload["roadgraph_points_dir_y"][b][idx]], -1)
        d = rotate_xy_np(dirs[sel], heading[b])
        out["lane_xy"][b, :k] = local[sel]
        out["lane_dir"][b, :k] = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-6)
        out["lane_id"][b, :k] = np.asarray(payload["roadgraph_points_ids"][b])[idx][sel]
        out["lane_valid"][b, :k] = True
    return out


def future_boxes(payload: Mapping[str, np.ndarray], anchor: int, origin_xy: np.ndarray, heading: np.ndarray,
                 steps: int, stride: int, n: int) -> dict[str, np.ndarray]:
    """Logged boxes of the first n objects at anchor + stride * (1..steps), in the given frame."""
    num_steps = payload["log_trajectory_x"].shape[2]
    raw = anchor + np.arange(1, steps + 1) * int(stride)
    idx = np.clip(raw, 0, num_steps - 1)
    in_range = raw < num_steps
    g = lambda k: np.asarray(payload[k])[:, :n][:, :, idx]  # noqa: E731
    xy = np.stack([g("log_trajectory_x"), g("log_trajectory_y")], -1).astype(np.float64)
    local = rotate_xy_np(xy - origin_xy[:, None, None, :], heading[:, None, None])
    return {
        "xy": local.astype(np.float32),
        "yaw": wrap_to_pi(g("log_trajectory_yaw") - heading[:, None, None]).astype(np.float32),
        "lw": np.stack([g("log_trajectory_length"), g("log_trajectory_width")], -1).astype(np.float32),
        "valid": g("log_trajectory_valid").astype(bool) & in_range[None, None, :],
    }


def build_arrays(payload: Mapping[str, np.ndarray], *, anchor_step: int, cfg: AnchorWMConfig,
                 with_targets: bool = True) -> dict[str, np.ndarray]:
    """One anchor-time batch from one cached shard (or a `payload_from_sim_state` payload).

    Only history/current state feeds the model; logged futures are read only for targets and
    labels (skipped with `with_targets=False`, e.g. for recorded ES replans)."""
    x = np.asarray(payload["log_trajectory_x"], dtype=np.float32)
    y = np.asarray(payload["log_trajectory_y"], dtype=np.float32)
    vx = np.asarray(payload["log_trajectory_vel_x"], dtype=np.float32)
    vy = np.asarray(payload["log_trajectory_vel_y"], dtype=np.float32)
    yaw = np.asarray(payload["log_trajectory_yaw"], dtype=np.float32)
    length = np.asarray(payload["log_trajectory_length"], dtype=np.float32)
    width = np.asarray(payload["log_trajectory_width"], dtype=np.float32)
    valid = np.asarray(payload["log_trajectory_valid"]).astype(bool)
    is_sdc = np.asarray(payload["object_metadata_is_sdc"]).astype(bool)
    object_types = np.asarray(payload["object_metadata_object_types"], dtype=np.int32)
    if x.ndim != 3:
        raise ValueError(f"Expected log trajectories [S,N,T], got {x.shape}")

    s, n_total, num_steps = x.shape
    ego_idx = np.argmax(is_sdc.astype(np.int32), axis=1)
    n = min(int(cfg.max_agents), int(n_total))
    if np.any(ego_idx >= n):
        raise ValueError("max_agents truncates away the SDC; use max_agents >= max_num_objects (128).")
    x, y, vx, vy, yaw = x[:, :n], y[:, :n], vx[:, :n], vy[:, :n], yaw[:, :n]
    length, width, valid = length[:, :n], width[:, :n], valid[:, :n]
    is_sdc, object_types = is_sdc[:, :n], object_types[:, :n]
    ar = np.arange(s)

    anchor = int(np.clip(anchor_step, 0, num_steps - 1))
    origin_xy = np.stack([x[ar, ego_idx, anchor], y[ar, ego_idx, anchor]], axis=-1).astype(np.float64)
    ego_yaw = yaw[ar, ego_idx, anchor].astype(np.float64)

    # History input: H samples ending at the anchor (as MF-Gaussian).
    hist_raw_idx = anchor - np.arange(cfg.history_steps - 1, -1, -1, dtype=np.int32) * int(cfg.history_stride)
    hist_idx = np.clip(hist_raw_idx, 0, num_steps - 1)
    hvalid = valid[:, :, hist_idx] & (hist_raw_idx >= 0)[None, None, :]
    hist_xy_local = rotate_xy_np(np.stack([x[:, :, hist_idx], y[:, :, hist_idx]], -1) - origin_xy[:, None, None, :],
                                 ego_yaw[:, None, None])
    hist_vel_local = rotate_xy_np(np.stack([vx[:, :, hist_idx], vy[:, :, hist_idx]], -1), ego_yaw[:, None, None])
    hist_yaw_rel = wrap_to_pi(yaw[:, :, hist_idx] - ego_yaw[:, None, None])
    type_oh = np.eye(int(cfg.num_object_types), dtype=np.float32)[np.clip(object_types, 0, cfg.num_object_types - 1)]
    type_hist = np.broadcast_to(type_oh[:, :, None, :], (s, n, cfg.history_steps, cfg.num_object_types))
    agent_history = np.concatenate([
        hist_xy_local / float(cfg.max_range),
        hist_vel_local / float(cfg.max_velocity),
        np.sin(hist_yaw_rel)[..., None], np.cos(hist_yaw_rel)[..., None],
        length[:, :, hist_idx][..., None] / float(cfg.max_width),
        width[:, :, hist_idx][..., None] / float(cfg.max_width),
        type_hist,
    ], axis=-1).astype(np.float32)
    agent_history = np.where(hvalid[..., None], agent_history, 0.0)

    current_valid = valid[:, :, anchor]
    current_xy_local = rotate_xy_np(np.stack([x[:, :, anchor], y[:, :, anchor]], -1) - origin_xy[:, None, :],
                                    ego_yaw[:, None]).astype(np.float32)
    current_yaw_local = wrap_to_pi(yaw[:, :, anchor] - ego_yaw[:, None]).astype(np.float32)

    map_features, map_valid = build_map_features(payload, origin_xy, ego_yaw, cfg)
    tl_features, tl_valid = build_tl_features(payload, np.full((s,), anchor, np.int32), origin_xy, ego_yaw, cfg)

    # Goal: last valid logged ego position of the whole episode (es_discriminator's definition).
    ego_valid_all = valid[ar, ego_idx]                                                # [S,T_all]
    last = np.where(ego_valid_all.any(1), num_steps - 1 - np.argmax(ego_valid_all[:, ::-1], axis=1), anchor)
    goal_world = np.stack([x[ar, ego_idx, last], y[ar, ego_idx, last]], -1).astype(np.float64)

    out = {
        "agent_history": agent_history,
        "agent_history_valid": hvalid.astype(bool),
        "agent_valid": current_valid.astype(bool),
        "map_features": map_features.astype(np.float32),
        "map_valid": map_valid.astype(bool),
        "traffic_light_features": tl_features,
        "traffic_light_valid": tl_valid,
        "current_local_xy": current_xy_local,
        "current_yaw_local": current_yaw_local,
        "object_types": object_types.astype(np.int32),
        "is_sdc": is_sdc.astype(bool),
        "ego_idx": ego_idx.astype(np.int64),
        "ego_lw": np.stack([length[ar, ego_idx, anchor], width[ar, ego_idx, anchor]], -1).astype(np.float32),
        "ego_vel": rotate_xy_np(np.stack([vx[ar, ego_idx, anchor], vy[ar, ego_idx, anchor]], -1),
                                ego_yaw).astype(np.float32),
        "goal_xy": rotate_xy_np(goal_world - origin_xy, ego_yaw).astype(np.float32),
        "origin_xy": origin_xy.astype(np.float64),
        "origin_yaw": ego_yaw.astype(np.float64),
    }
    out.update(build_edges(payload, origin_xy, ego_yaw, cfg))
    out.update(build_lanes(payload, origin_xy, ego_yaw))
    if not with_targets:
        return out

    fut = future_boxes(payload, anchor, origin_xy, ego_yaw, cfg.future_steps, cfg.future_stride, n)
    dynamic = np.isin(object_types, DYNAMIC_TYPES) | is_sdc
    target_delta_xy = (fut["xy"] - current_xy_local[:, :, None, :]) / float(cfg.max_range)
    yaw_delta = wrap_to_pi(fut["yaw"] - current_yaw_local[:, :, None])
    target_valid_all = fut["valid"] & current_valid[:, :, None] & dynamic[:, :, None]
    others = ~is_sdc
    out.update(
        target_delta_xy=target_delta_xy.astype(np.float32),
        target_yaw_vec=np.stack([np.sin(yaw_delta), np.cos(yaw_delta)], -1).astype(np.float32),
        target_valid=(target_valid_all & others[:, :, None]).astype(bool),
        target_valid_all=target_valid_all.astype(bool),
        fut_xy=fut["xy"], fut_yaw=fut["yaw"], fut_lw=fut["lw"],
        fut_valid=(fut["valid"] & others[:, :, None]).astype(bool),
        ego_future_xy=fut["xy"][ar, ego_idx], ego_future_yaw=fut["yaw"][ar, ego_idx],
        ego_future_valid=fut["valid"][ar, ego_idx].astype(bool),
    )
    return out


class AnchorWMDataset(IterableDataset):
    """Shards -> anchor steps -> shuffled batches. Shards are split across DataLoader workers."""

    def __init__(self, cache_paths: tuple[str, ...], cfg: AnchorWMConfig, *, batch_size: int,
                 anchor_steps: tuple[int, ...], shuffle: bool, seed: int) -> None:
        super().__init__()
        self.cache_paths, self.cfg = tuple(cache_paths), cfg
        self.batch_size, self.anchor_steps = int(batch_size), tuple(anchor_steps)
        self.shuffle, self.seed = bool(shuffle), int(seed)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        rng = random.Random(self.seed * 1000 + wid)
        paths = list(self.cache_paths)
        if self.shuffle:
            rng.shuffle(paths)
        for path in paths[wid::nw]:
            payload = load_npz(path)
            steps = list(self.anchor_steps)
            if self.shuffle:
                rng.shuffle(steps)
            for step in steps:
                arrays = build_arrays(payload, anchor_step=int(step), cfg=self.cfg)
                count = arrays["agent_history"].shape[0]
                order = np.arange(count)
                if self.shuffle:
                    np.random.default_rng(rng.randrange(2 ** 31)).shuffle(order)
                for start in range(0, count, self.batch_size):
                    sel = order[start:start + self.batch_size]
                    yield {k: torch.from_numpy(np.ascontiguousarray(v[sel])) for k, v in arrays.items()}


def make_loader(cache_paths: tuple[str, ...], cfg: AnchorWMConfig, *, batch_size: int,
                anchor_steps: tuple[int, ...], shuffle: bool, seed: int, num_workers: int) -> DataLoader:
    ds = AnchorWMDataset(cache_paths, cfg, batch_size=batch_size, anchor_steps=anchor_steps,
                         shuffle=shuffle, seed=seed)
    return DataLoader(ds, batch_size=None, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      prefetch_factor=4 if num_workers > 0 else None, persistent_workers=False)


def resolve_cache_paths(cache_dir: str) -> tuple[str, ...]:
    paths = sorted(Path(cache_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No npz cache files in {cache_dir}")
    return tuple(str(p) for p in paths)


def split_cache_paths(paths: tuple[str, ...], val_fraction: float) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Same split as MF-Gaussian's `data.utils.split_cache_paths` (last fraction is validation)."""
    num_val = max(1, int(len(paths) * val_fraction))
    num_train = max(1, len(paths) - num_val)
    return paths[:num_train], paths[num_train:]
