from __future__ import annotations

import atexit
import json
import zlib
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch
from waymax.datatypes.object_state import ObjectTypeIds

from data.preprocess import _build_map_features, _build_tl_features
from data.types import PreprocessConfig

from .config import WorldModelConfig, canonicalize_planning_mode
from .network import LatentMultiAgentWorldModel
from .planning import (
    MFCollisionDiagnostics,
    canonicalize_mf_collision_method,
    combine_mf_collision,
    ego_agent_mode_collisions,
    mf_mode_weights,
    mf_uses_sampling,
    pack_collision_worlds,
    sampled_mode_collision_prob,
    score_independent_agent_collision,
)


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
        planning_mode: str | None = None,
        mf_collision_method: str | None = None,
        mf_gaussian_num_samples: int = 8,
        mf_gaussian_seed: int = 0,
        mf_diagnostics_dir: str | None = None,
        mf_debug_dump: bool = False,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.planning_mode = canonicalize_planning_mode(planning_mode or config.planning_mode)
        self._cache: dict[tuple[int, int, int, float], dict[str, np.ndarray]] = {}
        self._last_collision_debug: dict[str, Any] | None = None

        # Collision-method ablation; None keeps planning_mode scoring unchanged.
        self.mf_collision_method = (
            canonicalize_mf_collision_method(mf_collision_method) if mf_collision_method else None
        )
        self.mf_gaussian_num_samples = int(mf_gaussian_num_samples)
        self.mf_gaussian_seed = int(mf_gaussian_seed)
        self.mf_diagnostics_dir = mf_diagnostics_dir
        self.mf_debug_dump = bool(mf_debug_dump)
        self._mf_scene_ids: list[str] | None = None
        self._mf_diag: MFCollisionDiagnostics | None = None
        if self.mf_collision_method and mf_uses_sampling(self.mf_collision_method):
            if self.mf_gaussian_num_samples < 1:
                raise ValueError("mf_gaussian_num_samples must be >= 1")
            self._mf_diag = MFCollisionDiagnostics(
                self.mf_collision_method, self.mf_gaussian_num_samples, int(config.num_modes)
            )
            if self.mf_diagnostics_dir:
                atexit.register(self.write_mf_diagnostics)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        planning_mode: str | None = None,
        **mf_kwargs: Any,
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
        mode = canonicalize_planning_mode(planning_mode if planning_mode is not None else config.planning_mode)
        forecaster = cls(model, config, device=device, planning_mode=mode, **mf_kwargs)
        # Independent-agent risk is scored via score_independent_mode_collision;
        # do not advertise K joint worlds to the gaussian ensemble SAT path.
        forecaster.num_collision_worlds = 1
        forecaster.collision_reduce = "any"
        return forecaster

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

    def _decode_boxes(
        self,
        pred_delta: np.ndarray,
        pred_yaw_vec: np.ndarray,
        meta: dict[str, np.ndarray],
        *,
        traj_length: int,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode one [B, N_modeled, T, 2] mean trajectory to [B, N, H, 5] boxes."""
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

        out_xy = np.broadcast_to(current_xy[:, :, None, :], (b, n_total, horizon, 2)).copy()
        out_yaw = np.broadcast_to(current_yaw[:, :, None], (b, n_total, horizon)).copy()

        delta_local = pred_delta[:, :modeled_n] * float(self.config.max_range)
        delta_world = _rotate_by_plus_heading(delta_local, ego_yaw[:, None, None])
        model_xy_future = current_xy[:, :modeled_n, None, :] + delta_world
        yaw_delta = np.arctan2(
            pred_yaw_vec[:, :modeled_n, :, 0],
            pred_yaw_vec[:, :modeled_n, :, 1],
        )
        model_yaw_future = _wrap_to_pi_np(current_yaw[:, :modeled_n, None] + yaw_delta)

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

    def set_scene_ids(self, scene_ids: list[str] | None) -> None:
        """WOMD scenario/id per batch element (batch order), set by the runner for each loaded batch."""
        self._mf_scene_ids = None if scene_ids is None else [str(s) for s in scene_ids]
        if self.mf_collision_method is not None:
            self._cache.clear()

    def _mf_sample_seed(self, meta: dict[str, np.ndarray], world_idx: int) -> np.random.SeedSequence:
        """(mf_gaussian_seed, crc32(WOMD scenario/id), replan timestep) for one batch element."""
        ids = getattr(self, "_mf_scene_ids", None)
        b = int(meta["x"].shape[0])
        if ids is None or len(ids) != b:
            raise RuntimeError(
                f"gaussian_* collision methods need set_scene_ids() with {b} WOMD scenario ids "
                f"for this batch, got {None if ids is None else len(ids)}."
            )
        return np.random.SeedSequence(
            [self.mf_gaussian_seed, zlib.crc32(ids[world_idx].encode()), int(meta["timestep"])]
        )

    def _sample_gaussian_noise(
        self,
        sigma_m_k: np.ndarray,
        meta: dict[str, np.ndarray],
        *,
        traj_length: int,
        dt: float,
    ) -> np.ndarray:
        """World-frame xy offsets [B, N, K, M, H, 2] such that mean + offset ~ the model's Gaussian.

        The model's density is an independent diagonal Gaussian per (agent, mode,
        model step, axis) with std sigma (metres) in the anchor ego frame, so
        eps ~ N(0, I) is scaled by sigma, rotated to the world frame and linearly
        interpolated to dt exactly like the means in ``_decode_boxes`` (current
        pose fixed, zero offset). Agents that _decode_boxes keeps static get zero.
        Each scene draws from its own RNG, so its samples do not depend on the
        rest of the batch or its order.
        """
        b, n_mod, k, t_model, _ = sigma_m_k.shape
        m = int(self.mf_gaussian_num_samples)
        eps = np.stack(
            [
                np.random.default_rng(self._mf_sample_seed(meta, bi)).standard_normal(
                    (n_mod, k, m, t_model, 2), dtype=np.float32
                )
                for bi in range(b)
            ],
            axis=0,
        )
        local = eps * sigma_m_k[:, :, :, None].astype(np.float32)
        world = _rotate_by_plus_heading(local, meta["ego_yaw"][:, None, None, None, None]).astype(np.float32)

        horizon = int(traj_length)
        dst_t = np.arange(horizon, dtype=np.float32) * float(dt)
        src_t = np.concatenate(
            [[0.0], np.arange(1, self.config.future_steps + 1, dtype=np.float32) * float(self.config.future_dt)]
        ).astype(np.float32)
        # interp_w[h, j] reproduces np.interp(dst_t, src_t, .) as a linear map; column 0 is the noise-free current pose.
        interp_w = np.stack(
            [np.interp(dst_t, src_t, np.eye(src_t.size, dtype=np.float32)[j]) for j in range(src_t.size)], axis=-1
        ).astype(np.float32)
        noise = np.einsum("hj,bnkmjc->bnkmhc", interp_w[:, 1:], world).astype(np.float32)

        t = int(meta["timestep"])
        keep = (
            meta["valid"][:, :n_mod, t]
            & np.isin(meta["object_types"][:, :n_mod], _DYNAMIC_TYPES)
            & ~meta["is_sdc"][:, :n_mod]
        )
        noise *= keep[:, :, None, None, None, None]
        n_total = meta["x"].shape[1]
        if n_total > n_mod:
            pad = np.zeros((b, n_total - n_mod, k, m, horizon, 2), dtype=np.float32)
            noise = np.concatenate([noise, pad], axis=1)
        return noise

    def _predict_all_worlds(
        self,
        sim_state: Any,
        *,
        timestep: int,
        traj_length: int,
        dt: float,
        decode_all_modes: bool,
    ) -> dict[str, np.ndarray]:
        features, meta = self._prepare_features(sim_state, timestep)
        with torch.inference_mode():
            pred = self.model(features)

        # Top-1 mean path keeps the existing ES adapter contract.
        top1_traj, masks = self._decode_boxes(
            pred["pred_delta_xy_norm"].detach().float().cpu().numpy(),
            pred["pred_yaw_vec"].detach().float().cpu().numpy(),
            meta,
            traj_length=traj_length,
            dt=dt,
        )
        mode_probs = pred["mode_probs"].detach().float().cpu().numpy()
        top1_mode = pred["top1_mode"].detach().cpu().numpy().astype(np.int32)
        out = {
            "top1_traj": top1_traj,
            "masks": masks,
            "mode_probs": mode_probs,
            "top1_mode": top1_mode,
        }
        if decode_all_modes:
            delta_k = pred["pred_delta_xy_norm_k"].detach().float().cpu().numpy()
            yaw_k = pred["pred_yaw_vec_k"].detach().float().cpu().numpy()
            k = int(delta_k.shape[2])
            trajs = []
            for ki in range(k):
                traj_i, _ = self._decode_boxes(
                    delta_k[:, :, ki],
                    yaw_k[:, :, ki],
                    meta,
                    traj_length=traj_length,
                    dt=dt,
                )
                trajs.append(traj_i)
            out["traj_k"] = np.stack(trajs, axis=2).astype(np.float32)
            if self.mf_collision_method and mf_uses_sampling(self.mf_collision_method):
                # One draw per WM prediction, shared by every ES round / candidate / world of this replan.
                out["sample_noise_k"] = self._sample_gaussian_noise(
                    pred["pred_sigma_xy_m_k"].detach().float().cpu().numpy(),
                    meta,
                    traj_length=traj_length,
                    dt=dt,
                )
                if self._mf_diag is not None:
                    self._mf_diag.n_predictions += 1
                if self.mf_debug_dump and self.mf_diagnostics_dir:
                    dump_dir = Path(self.mf_diagnostics_dir) / "mf_samples_debug"
                    dump_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        dump_dir / f"pred{self._mf_diag.n_predictions:06d}_t{int(timestep):03d}.npz",
                        sample_noise_k=out["sample_noise_k"],
                        traj_k=out["traj_k"],
                        mode_probs=mode_probs,
                    )
        return out

    def forecast(
        self,
        sim_state: Any,
        *,
        timestep: int,
        traj_length: int,
        world_idx: int = 0,
        dt: float = 0.1,
        planning_mode: str | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        mode = canonicalize_planning_mode(planning_mode or self.planning_mode)
        # Decode all K modes so independent-agent scoring always has traj_k.
        key = (id(sim_state), int(timestep), int(traj_length), float(dt))
        cached = self._cache.get(key)
        if cached is None or "traj_k" not in cached:
            cached = self._predict_all_worlds(
                sim_state,
                timestep=int(timestep),
                traj_length=int(traj_length),
                dt=float(dt),
                decode_all_modes=True,
            )
            self._cache.clear()
            self._cache[key] = cached

        idx = int(world_idx)
        packed = pack_collision_worlds(
            np.asarray(cached["top1_traj"][idx]),
            np.asarray(cached["masks"][idx]),
            np.asarray(cached["traj_k"][idx]),
            None,
            planning_mode=mode,
        )
        return (
            jnp.asarray(packed[0], dtype=jnp.float32),
            jnp.asarray(packed[1]).astype(jnp.bool_),
            jnp.asarray(packed[2], dtype=jnp.float32),
            jnp.asarray(packed[3]).astype(jnp.bool_),
        )

    def score_independent_mode_collision(
        self,
        ego_trajectories_world: np.ndarray,
        *,
        world_idx: int = 0,
        planning_mode: str | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Agent-wise multimodal collision score for ES ego candidates.

        Requires a prior ``forecast()`` call that populated ``traj_k`` / ``mode_probs``.
        """
        if not self._cache:
            raise RuntimeError(
                "Call forecast() once to cache multimodal futures before independent-mode scoring."
            )
        if self.mf_collision_method is not None:
            return self._score_mf_collision_method(ego_trajectories_world, world_idx=int(world_idx))
        mode = canonicalize_planning_mode(planning_mode or self.planning_mode)
        cached = next(iter(self._cache.values()))
        idx = int(world_idx)
        if "traj_k" not in cached:
            # top1-only cache: synthesize K=1 from primary
            traj_k = np.asarray(cached["top1_traj"][idx])[:, None, ...]
            probs = np.ones((traj_k.shape[0], 1), dtype=np.float32)
        else:
            traj_k = np.asarray(cached["traj_k"][idx])
            probs = np.asarray(cached["mode_probs"][idx])
        masks = np.asarray(cached["masks"][idx])
        score, debug = score_independent_agent_collision(
            np.asarray(ego_trajectories_world, dtype=np.float32),
            traj_k,
            masks,
            probs,
            planning_mode=mode,
        )
        self._last_collision_debug = debug
        return score, debug

    def _score_mf_collision_method(
        self, ego_trajectories_world: np.ndarray, *, world_idx: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """mean_* / gaussian_* scoring against the cached prediction (see planning.py)."""
        method = self.mf_collision_method
        cached = next(iter(self._cache.values()))
        ego = np.asarray(ego_trajectories_world, dtype=np.float32)
        traj_k = np.asarray(cached["traj_k"][world_idx])
        probs = np.asarray(cached["mode_probs"][world_idx])
        masks = np.asarray(cached["masks"][world_idx])
        agent_valid = masks.any(axis=-1)
        weights = mf_mode_weights(method, probs)

        c_mean = ego_agent_mode_collisions(ego, traj_k, masks)  # [E, N, K] existing SAT on mode means
        debug: dict[str, Any] = {"mf_collision_method": method, "weights": weights, "c_mean": c_mean}
        if not mf_uses_sampling(method):
            score, p_collision_i = combine_mf_collision(c_mean, weights, agent_valid)
        else:
            c_hat, sampled, n_checks = sampled_mode_collision_prob(
                ego, traj_k, cached["sample_noise_k"][world_idx], masks, c_mean
            )
            score, p_collision_i = combine_mf_collision(c_hat, weights, agent_valid)
            score_mean, _ = combine_mf_collision(c_mean, weights, agent_valid)
            debug.update({"c_hat": c_hat, "sampled_agents": sampled, "n_sat_trajectory_checks": n_checks})
            if self._mf_diag is not None:
                self._mf_diag.update(
                    c_hat, c_mean, sampled, p_collision_i, score, score_mean, n_checks, ego.shape[1]
                )
                if self.mf_diagnostics_dir and self._mf_diag.n_score_calls % 500 == 0:
                    self.write_mf_diagnostics()
        debug.update({"p_collision_i": p_collision_i, "collision_score": score})
        self._last_collision_debug = debug
        return score, debug

    def write_mf_diagnostics(self) -> None:
        if self._mf_diag is None or not self.mf_diagnostics_dir:
            return
        path = Path(self.mf_diagnostics_dir) / "mf_collision_diagnostics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._mf_diag.to_dict(), indent=2))

    def forecast_modes(
        self,
        sim_state: Any,
        *,
        timestep: int,
        traj_length: int,
        world_idx: int = 0,
        dt: float = 0.1,
    ) -> dict[str, np.ndarray]:
        """Expose all K futures + probabilities without changing ES scoring."""
        key = (id(sim_state), int(timestep), int(traj_length), float(dt))
        cached = self._cache.get(key)
        if cached is None or "traj_k" not in cached:
            cached = self._predict_all_worlds(
                sim_state,
                timestep=int(timestep),
                traj_length=int(traj_length),
                dt=float(dt),
                decode_all_modes=True,
            )
            self._cache.clear()
            self._cache[key] = cached
        idx = int(world_idx)
        return {
            "top1_traj": cached["top1_traj"][idx],
            "masks": cached["masks"][idx],
            "traj_k": cached["traj_k"][idx],
            "mode_probs": cached["mode_probs"][idx],
            "top1_mode": cached["top1_mode"][idx],
        }


__all__ = ["WorldModelForecaster"]
