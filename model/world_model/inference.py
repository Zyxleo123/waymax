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
        # V1 is deterministic. Collision ensemble == primary forecast. A later
        # multimodal extension can return multiple sampled worlds here.
        return (
            object_trajectories,
            object_masks,
            object_trajectories,
            object_masks,
        )


class ScheduledWorldModelForecaster:
    """WM forecaster with scheduled refresh + slice + shortfall re-query.

    Refresh schedule (sim steps):
      - refresh_steps=50 → preferred refreshes at 0, 50, 100, ...
      - refresh_steps=20 → 0, 20, 40, ...
      - refresh_steps=10 → every ego replan (with default replan_interval=10)

    Between refreshes, absolute-time slices of the last forecast are reused.
    If the cache cannot cover [timestep, timestep + traj_length), the model is
    re-queried at the current timestep (shortfall policy A).
    """

    def __init__(
        self,
        base: WorldModelForecaster,
        *,
        refresh_steps: int = 10,
    ) -> None:
        if int(refresh_steps) <= 0:
            raise ValueError(f"refresh_steps must be positive, got {refresh_steps}")
        self.base = base
        self.refresh_steps = int(refresh_steps)
        self._origin_t: int | None = None
        self._traj_bnt5: np.ndarray | None = None
        self._mask_bnt: np.ndarray | None = None
        self._dt: float | None = None
        self._last_timestep: int | None = None
        self.num_queries: int = 0
        self.num_slices: int = 0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        refresh_steps: int = 10,
    ) -> "ScheduledWorldModelForecaster":
        base = WorldModelForecaster.from_checkpoint(checkpoint_path, device=device)
        return cls(base, refresh_steps=refresh_steps)

    @property
    def config(self) -> WorldModelConfig:
        return self.base.config

    @property
    def device(self) -> torch.device:
        return self.base.device

    def clear_cache(self) -> None:
        self.base.clear_cache()
        self._origin_t = None
        self._traj_bnt5 = None
        self._mask_bnt = None
        self._dt = None
        self._last_timestep = None

    def native_horizon_steps(self, dt: float = 0.1) -> int:
        return max(1, int(round(float(self.config.future_steps) * float(self.config.future_dt) / float(dt))))

    def _covers(self, timestep: int, traj_length: int, dt: float) -> bool:
        if (
            self._origin_t is None
            or self._traj_bnt5 is None
            or self._mask_bnt is None
            or self._dt is None
            or abs(float(self._dt) - float(dt)) > 1e-8
        ):
            return False
        offset = int(timestep) - int(self._origin_t)
        if offset < 0:
            return False
        return offset + int(traj_length) <= int(self._traj_bnt5.shape[2])

    def _should_schedule_refresh(self, timestep: int) -> bool:
        return int(timestep) % self.refresh_steps == 0

    def _run_and_store(
        self,
        sim_state: Any,
        *,
        timestep: int,
        traj_length: int,
        dt: float,
    ) -> None:
        native = self.native_horizon_steps(dt)
        predict_len = max(int(traj_length), native)
        traj_bnt5, mask_bnt = self.base._predict_all_worlds(
            sim_state,
            timestep=int(timestep),
            traj_length=predict_len,
            dt=float(dt),
        )
        self._origin_t = int(timestep)
        self._traj_bnt5 = traj_bnt5
        self._mask_bnt = mask_bnt
        self._dt = float(dt)
        self._last_timestep = int(timestep)
        self.num_queries += 1
        # Keep base per-call cache aligned with the latest absolute rollout.
        self.base.clear_cache()
        self.base._cache[(id(sim_state), int(timestep), predict_len, float(dt))] = (
            traj_bnt5,
            mask_bnt,
        )

    def forecast(
        self,
        sim_state: Any,
        *,
        timestep: int,
        traj_length: int,
        world_idx: int = 0,
        dt: float = 0.1,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        t = int(timestep)
        # New scenario / episode restart: closed-loop timesteps jump backwards.
        # Do not key on id(sim_state): ego replacement allocates a new state each replan.
        if self._last_timestep is not None and t < int(self._last_timestep):
            self.clear_cache()

        length = int(traj_length)
        # Reuse an existing rollout from this exact timestep across ES candidates /
        # worlds. Otherwise refresh on schedule, or on shortfall.
        same_origin_covers = self._origin_t == t and self._covers(t, length, dt)
        need_refresh = (not same_origin_covers) and (
            self._should_schedule_refresh(t) or not self._covers(t, length, dt)
        )
        if need_refresh:
            self._run_and_store(
                sim_state,
                timestep=t,
                traj_length=length,
                dt=float(dt),
            )
        else:
            self.num_slices += 1
            self._last_timestep = t

        assert self._origin_t is not None and self._traj_bnt5 is not None and self._mask_bnt is not None
        offset = t - int(self._origin_t)
        if offset < 0 or offset + length > int(self._traj_bnt5.shape[2]):
            # Belts-and-suspenders: should be unreachable after _covers / refresh.
            self._run_and_store(
                sim_state,
                timestep=t,
                traj_length=length,
                dt=float(dt),
            )
            offset = 0

        idx = int(world_idx)
        object_trajectories = jnp.asarray(
            self._traj_bnt5[idx, :, offset : offset + length, :],
            dtype=jnp.float32,
        )
        object_masks = jnp.asarray(
            self._mask_bnt[idx, :, offset : offset + length],
        ).astype(jnp.bool_)
        return (
            object_trajectories,
            object_masks,
            object_trajectories,
            object_masks,
        )


__all__ = ["WorldModelForecaster", "ScheduledWorldModelForecaster"]
