from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch
from waymax.datatypes.object_state import ObjectTypeIds

from data.preprocess import _build_map_features, _build_tl_features
from data.types import PreprocessConfig

from .config import WorldModelConfig
from .network import LatentMultiAgentWorldModel


_DYNAMIC_TYPES = np.asarray(
    [
        int(ObjectTypeIds.VEHICLE.value),
        int(ObjectTypeIds.PEDESTRIAN.value),
        int(ObjectTypeIds.CYCLIST.value),
    ],
    dtype=np.int32,
)


def _wrap_to_pi_np(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _rotate_by_minus_heading(xy: np.ndarray, heading: np.ndarray) -> np.ndarray:
    c = np.cos(-heading)
    s = np.sin(-heading)
    x = xy[..., 0]
    y = xy[..., 1]
    return np.stack([x * c - y * s, x * s + y * c], axis=-1)


def _rotate_by_plus_heading(xy: np.ndarray, heading: np.ndarray) -> np.ndarray:
    c = np.cos(heading)
    s = np.sin(heading)
    x = xy[..., 0]
    y = xy[..., 1]
    return np.stack([x * c - y * s, x * s + y * c], axis=-1)


class WorldModelForecaster:
    """Runtime adapter from the learned latent world model to FM-ES scorer boxes.

    The neural model predicts only future motion. This adapter decodes the latent
    rollout to explicit [x, y, yaw, length, width] trajectories, preserving the
    existing static scoring functions.

    Predictions are cached per (sim_state, timestep, horizon), because an ES
    iteration scores many ego candidates against the same surrounding-agent
    future.
    """

    def __init__(
        self,
        model: LatentMultiAgentWorldModel,
        config: WorldModelConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self._cache: dict[tuple[int, int, int, float], tuple[np.ndarray, np.ndarray]] = {}
        self._scene_cache: dict[str, Any] = {}

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "WorldModelForecaster":
        path = Path(checkpoint_path).expanduser()
        if path.is_dir():
            path = path / "latest.pt"
        if not path.exists():
            raise FileNotFoundError(f"World-model checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "config" not in checkpoint:
            raise KeyError(f"Checkpoint {path} is missing 'config'.")
        config = WorldModelConfig.from_dict(dict(checkpoint["config"]))
        model = LatentMultiAgentWorldModel(config)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=True)
        return cls(model, config, device=device)

    def clear_cache(self) -> None:
        self._cache.clear()
        self._scene_cache.clear()

    def _prepare_features(self, sim_state: Any, timestep: int) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
        cfg = self.config

        x = np.asarray(sim_state.log_trajectory.x, dtype=np.float32)
        y = np.asarray(sim_state.log_trajectory.y, dtype=np.float32)
        vx = np.asarray(sim_state.log_trajectory.vel_x, dtype=np.float32)
        vy = np.asarray(sim_state.log_trajectory.vel_y, dtype=np.float32)
        yaw = np.asarray(sim_state.log_trajectory.yaw, dtype=np.float32)
        length = np.asarray(sim_state.log_trajectory.length, dtype=np.float32)
        width = np.asarray(sim_state.log_trajectory.width, dtype=np.float32)
        valid = np.asarray(sim_state.log_trajectory.valid).astype(bool)
        is_sdc = np.asarray(sim_state.object_metadata.is_sdc).astype(bool)
        object_types = np.asarray(sim_state.object_metadata.object_types, dtype=np.int32)

        if x.ndim != 3:
            raise ValueError(f"Expected batched Waymax trajectories [B,N,T], got {x.shape}")

        b, n_total, num_steps = x.shape
        n = min(int(cfg.max_agents), int(n_total))
        t = int(np.clip(int(timestep), 0, num_steps - 1))

        # Keep full arrays in metadata so decoded output preserves the exact
        # object count expected by the existing scorer.
        ego_idx_full = np.argmax(is_sdc.astype(np.int32), axis=1)
        if np.any(ego_idx_full >= n):
            raise ValueError(
                "WorldModelConfig.max_agents truncates away the SDC. "
                "Use max_agents >= simulation --max_num_objects (normally 128)."
            )
        origin_xy = np.stack(
            [
                x[np.arange(b), ego_idx_full, t],
                y[np.arange(b), ego_idx_full, t],
            ],
            axis=-1,
        ).astype(np.float32)
        ego_yaw = yaw[np.arange(b), ego_idx_full, t].astype(np.float32)

        # Modeled subset. Existing experiments use max_num_objects=128, so this
        # normally covers the full Waymax object list.
        x_m = x[:, :n]
        y_m = y[:, :n]
        vx_m = vx[:, :n]
        vy_m = vy[:, :n]
        yaw_m = yaw[:, :n]
        length_m = length[:, :n]
        width_m = width[:, :n]
        valid_m = valid[:, :n]
        type_m = object_types[:, :n]

        hist_offsets = np.arange(cfg.history_steps - 1, -1, -1, dtype=np.int32)
        hist_raw_idx = t - hist_offsets * int(cfg.history_stride)
        hist_idx = np.clip(hist_raw_idx, 0, num_steps - 1)
        hist_in_range = hist_raw_idx >= 0

        hx = x_m[:, :, hist_idx]
        hy = y_m[:, :, hist_idx]
        hvx = vx_m[:, :, hist_idx]
        hvy = vy_m[:, :, hist_idx]
        hyaw = yaw_m[:, :, hist_idx]
        hlen = length_m[:, :, hist_idx]
        hwid = width_m[:, :, hist_idx]
        hvalid = valid_m[:, :, hist_idx] & hist_in_range[None, None, :]

        hist_xy = np.stack([hx, hy], axis=-1)
        hist_vel = np.stack([hvx, hvy], axis=-1)
        hist_xy_local = _rotate_by_minus_heading(
            hist_xy - origin_xy[:, None, None, :],
            ego_yaw[:, None, None],
        )
        hist_vel_local = _rotate_by_minus_heading(
            hist_vel,
            ego_yaw[:, None, None],
        )
        hist_yaw_rel = _wrap_to_pi_np(hyaw - ego_yaw[:, None, None])

        type_safe = np.clip(type_m, 0, int(cfg.num_object_types) - 1)
        type_oh = np.eye(int(cfg.num_object_types), dtype=np.float32)[type_safe]
        type_hist = np.broadcast_to(
            type_oh[:, :, None, :],
            (b, n, cfg.history_steps, cfg.num_object_types),
        )

        agent_history = np.concatenate(
            [
                hist_xy_local / float(cfg.max_range),
                hist_vel_local / float(cfg.max_velocity),
                np.sin(hist_yaw_rel)[..., None],
                np.cos(hist_yaw_rel)[..., None],
                hlen[..., None] / float(cfg.max_width),
                hwid[..., None] / float(cfg.max_width),
                type_hist,
            ],
            axis=-1,
        ).astype(np.float32)
        agent_history = np.where(hvalid[..., None], agent_history, 0.0)
        agent_valid = valid_m[:, :, t]

        # Use the same map / traffic-light feature representation as the rest of
        # the project. These helpers read only static map + the anchor TL state.
        pp_cfg = PreprocessConfig(
            max_range=float(cfg.max_range),
            max_velocity=float(cfg.max_velocity),
            max_width=float(cfg.max_width),
            max_tl_points=int(cfg.max_tl_points),
            num_map_type_classes=int(cfg.num_map_type_classes),
            map_unknown_type_index=int(cfg.map_unknown_type_index),
            max_segments=int(cfg.max_map_segments),
            max_points_per_segment=int(cfg.max_points_per_segment),
            num_object_types=int(cfg.num_object_types),
        )
        origin_jax = jnp.asarray(origin_xy)
        ego_yaw_jax = jnp.asarray(ego_yaw)
        map_features, map_valid, _ = _build_map_features(
            sim_state,
            origin_jax,
            ego_yaw_jax,
            pp_cfg,
            max_segments=int(cfg.max_map_segments),
        )
        anchor_step_b = jnp.full((b,), t, dtype=jnp.int32)
        tl_features, tl_valid = _build_tl_features(
            sim_state,
            anchor_step_b,
            origin_jax,
            ego_yaw_jax,
            pp_cfg,
        )

        features_np = {
            "agent_history": agent_history,
            "agent_history_valid": hvalid.astype(bool),
            "agent_valid": agent_valid.astype(bool),
            "map_features": np.asarray(map_features, dtype=np.float32),
            "map_valid": np.asarray(map_valid).astype(bool),
            "traffic_light_features": np.asarray(tl_features, dtype=np.float32),
            "traffic_light_valid": np.asarray(tl_valid).astype(bool),
        }
        features = {
            key: torch.as_tensor(value, device=self.device)
            for key, value in features_np.items()
        }
        metadata = {
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "yaw": yaw,
            "length": length,
            "width": width,
            "valid": valid,
            "is_sdc": is_sdc,
            "object_types": object_types,
            "origin_xy": origin_xy,
            "ego_yaw": ego_yaw,
            "timestep": np.asarray(t, dtype=np.int32),
            "modeled_agents": np.asarray(n, dtype=np.int32),
        }
        return features, metadata

    def _predict_all_worlds(
        self,
        sim_state: Any,
        *,
        timestep: int,
        traj_length: int,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        features, meta = self._prepare_features(sim_state, timestep)
        with torch.inference_mode():
            pred = self.model(features)
        # Use the Gaussian mean trajectory for ES scoring.
        pred_delta = pred["pred_delta_xy_norm"].detach().float().cpu().numpy()
        pred_yaw_vec = pred["pred_yaw_vec"].detach().float().cpu().numpy()

        x = meta["x"]
        y = meta["y"]
        yaw = meta["yaw"]
        length = meta["length"]
        width = meta["width"]
        valid = meta["valid"]
        is_sdc = meta["is_sdc"]
        object_types = meta["object_types"]
        ego_yaw = meta["ego_yaw"]
        t = int(meta["timestep"])
        modeled_n = int(meta["modeled_agents"])

        b, n_total, _ = x.shape
        horizon = int(traj_length)
        dst_t = np.arange(horizon, dtype=np.float32) * float(dt)
        src_t = np.concatenate(
            [
                np.asarray([0.0], dtype=np.float32),
                np.arange(1, self.config.future_steps + 1, dtype=np.float32)
                * float(self.config.future_dt),
            ],
            axis=0,
        )

        current_xy = np.stack([x[:, :, t], y[:, :, t]], axis=-1)
        current_yaw = yaw[:, :, t]
        current_len = length[:, :, t]
        current_wid = width[:, :, t]
        current_valid = valid[:, :, t]
        dynamic = np.isin(object_types, _DYNAMIC_TYPES)

        # Default to a static hold. Dynamic modeled agents are replaced below.
        out_xy = np.broadcast_to(current_xy[:, :, None, :], (b, n_total, horizon, 2)).copy()
        out_yaw = np.broadcast_to(current_yaw[:, :, None], (b, n_total, horizon)).copy()

        # Model displacement is in the current ego coordinate frame.
        delta_local = pred_delta[:, :modeled_n] * float(self.config.max_range)
        delta_world = _rotate_by_plus_heading(
            delta_local,
            ego_yaw[:, None, None],
        )
        model_xy_future = current_xy[:, :modeled_n, None, :] + delta_world
        yaw_delta = np.arctan2(
            pred_yaw_vec[:, :modeled_n, :, 0],
            pred_yaw_vec[:, :modeled_n, :, 1],
        )
        model_yaw_future = _wrap_to_pi_np(
            current_yaw[:, :modeled_n, None] + yaw_delta
        )

        # Include t=0 current state, then interpolate the learned model-dt future
        # to the world-dt horizon expected by FM-ES collision scoring.
        for bi in range(b):
            for ni in range(modeled_n):
                if not (current_valid[bi, ni] and dynamic[bi, ni] and not is_sdc[bi, ni]):
                    continue
                xy_src = np.concatenate(
                    [current_xy[bi, ni][None, :], model_xy_future[bi, ni]],
                    axis=0,
                )
                yaw_src = np.concatenate(
                    [np.asarray([current_yaw[bi, ni]], dtype=np.float32), model_yaw_future[bi, ni]],
                    axis=0,
                )
                out_xy[bi, ni, :, 0] = np.interp(dst_t, src_t, xy_src[:, 0])
                out_xy[bi, ni, :, 1] = np.interp(dst_t, src_t, xy_src[:, 1])
                sin_interp = np.interp(dst_t, src_t, np.sin(yaw_src))
                cos_interp = np.interp(dst_t, src_t, np.cos(yaw_src))
                out_yaw[bi, ni] = np.arctan2(sin_interp, cos_interp)

        out_len = np.broadcast_to(current_len[:, :, None], (b, n_total, horizon))
        out_wid = np.broadcast_to(current_wid[:, :, None], (b, n_total, horizon))
        trajectories = np.stack(
            [out_xy[..., 0], out_xy[..., 1], out_yaw, out_len, out_wid],
            axis=-1,
        ).astype(np.float32)
        masks = np.broadcast_to(
            (current_valid & (~is_sdc))[:, :, None],
            (b, n_total, horizon),
        ).copy()
        return trajectories, masks

    def forecast(
        self,
        sim_state: Any,
        *,
        timestep: int,
        traj_length: int,
        world_idx: int = 0,
        dt: float = 0.1,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        key = (id(sim_state), int(timestep), int(traj_length), float(dt))
        cached = self._cache.get(key)
        if cached is None:
            cached = self._predict_all_worlds(
                sim_state,
                timestep=int(timestep),
                traj_length=int(traj_length),
                dt=float(dt),
            )
            # Keep only the most recent state/horizon. This prevents unbounded
            # growth over long simulation runs while retaining ES reuse.
            self._cache.clear()
            self._cache[key] = cached

        trajectories_bnt5, masks_bnt = cached
        idx = int(world_idx)
        object_trajectories = jnp.asarray(trajectories_bnt5[idx], dtype=jnp.float32)
        object_masks = jnp.asarray(masks_bnt[idx]).astype(jnp.bool_)
        # Collision ensemble currently equals the mean forecast. Sampling from
        # the predicted Gaussians is not wired into ES scoring yet.
        return (
            object_trajectories,
            object_masks,
            object_trajectories,
            object_masks,
        )

    def _cached_scene(self, sim_state: Any, timestep: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, np.ndarray]]:
        key = (id(sim_state), int(timestep))
        cached = self._scene_cache.get("entry")
        if cached is None or cached["key"] != key:
            features, meta = self._prepare_features(sim_state, timestep)
            with torch.inference_mode():
                scene = self.model.encode_scene(features)
            cached = {"key": key, "scene": scene, "features": features, "meta": meta}
            self._scene_cache["entry"] = cached
        return cached["scene"], cached["features"], cached["meta"]

    def _ego_candidates_to_features(
        self,
        ego_world_kt5: np.ndarray,
        *,
        origin_xy: np.ndarray,
        ego_yaw: float,
        dt: float,
    ) -> torch.Tensor:
        """ego_world_kt5: [K, T, 5] world boxes -> [1, K, T_model, 6] ego-frame features."""
        cfg = self.config
        ego = np.asarray(ego_world_kt5, dtype=np.float32)
        if ego.ndim != 3 or ego.shape[-1] < 3:
            raise ValueError(f"ego candidates must be [K,T,5], got {ego.shape}")
        xy = ego[..., :2] - np.asarray(origin_xy, dtype=np.float32).reshape(1, 1, 2)
        c = np.cos(-float(ego_yaw))
        s = np.sin(-float(ego_yaw))
        x = xy[..., 0] * c - xy[..., 1] * s
        y = xy[..., 0] * s + xy[..., 1] * c
        yaw = _wrap_to_pi_np(ego[..., 2] - float(ego_yaw))
        vel = np.zeros_like(xy)
        if ego.shape[1] > 1:
            vel[:, 1:] = (np.stack([x, y], axis=-1)[:, 1:] - np.stack([x, y], axis=-1)[:, :-1]) / max(float(dt), 1e-6)
            vel[:, 0] = vel[:, 1]
        feat = np.concatenate(
            [
                x[..., None] / float(cfg.max_range),
                y[..., None] / float(cfg.max_range),
                vel[..., 0:1] / float(cfg.max_velocity),
                vel[..., 1:2] / float(cfg.max_velocity),
                np.sin(yaw)[..., None],
                np.cos(yaw)[..., None],
            ],
            axis=-1,
        ).astype(np.float32)
        # Resample candidate horizon onto model future_steps if needed.
        t_src = np.arange(feat.shape[1], dtype=np.float32) * float(dt)
        t_dst = (np.arange(cfg.future_steps, dtype=np.float32) + 1.0) * float(cfg.future_dt)
        out = np.zeros((feat.shape[0], cfg.future_steps, feat.shape[-1]), dtype=np.float32)
        for k in range(feat.shape[0]):
            for d in range(feat.shape[-1]):
                out[k, :, d] = np.interp(t_dst, t_src, feat[k, :, d])
        tensor = torch.from_numpy(out).to(self.device)
        return tensor[None]

    def forecast_population(
        self,
        sim_state: Any,
        ego_candidates_world: np.ndarray,
        *,
        timestep: int,
        traj_length: int,
        world_idx: int = 0,
        dt: float = 0.1,
    ) -> dict[str, np.ndarray]:
        """Decode surrounding-agent futures for a batch of ego candidates.

        Scene encoding is reused. Only the ego GRU + fuse MLP + state decoder
        run per candidate. Returns pred_delta in ego frame plus probabilities
        are not used; boxes stay in the same local frame as the trajectory head.

        ego_candidates_world: [K, T, 5]
        pred_delta_xy_norm_k: [K, N, T_model, 2]
        """
        scene, features, meta = self._cached_scene(sim_state, timestep)
        origin = np.asarray(meta["origin_xy"][int(world_idx)])
        yaw0 = float(np.asarray(meta["ego_yaw"])[int(world_idx)])
        ego_feat = self._ego_candidates_to_features(
            ego_candidates_world,
            origin_xy=origin,
            ego_yaw=yaw0,
            dt=float(dt),
        )
        with torch.inference_mode():
            pred = self.model.decode_from_scene(scene, ego_future=ego_feat)
        return {
            "pred_delta_xy_norm_k": pred["pred_delta_xy_norm_k"][int(world_idx)].detach().float().cpu().numpy(),
            "pred_yaw_vec_k": pred["pred_yaw_vec_k"][int(world_idx)].detach().float().cpu().numpy(),
            "pred_sigma_xy_m_k": pred["pred_sigma_xy_m_k"][int(world_idx)].detach().float().cpu().numpy(),
            "pred_delta_xy_norm": pred["pred_delta_xy_norm"][int(world_idx)].detach().float().cpu().numpy(),
        }


__all__ = ["WorldModelForecaster"]
