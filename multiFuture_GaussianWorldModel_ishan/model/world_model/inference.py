from __future__ import annotations

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
from .planning import pack_collision_worlds, score_independent_agent_collision


class ForecastStats:
    """Running mean / std of what the planner gets from the WM, once per forward pass.

    Counted over agents whose future is predicted (valid now, dynamic, not the ego).

    Mode probabilities, for ``raw`` (the classifier's own softmax, temperature 1) and
    ``tempered`` (what collision scoring uses): ``by_mode`` per mode index k (the k-th
    learned mode query); ``by_rank`` with each agent's probabilities sorted, so rank 0 is
    the top-1 probability; ``entropy`` (nats) and ``effective_modes`` = exp(entropy).

    Displacement errors (m) against the logged future, which is the agents' real future
    under log-replay sim agents (not under IDM). Per agent, ADE averages the valid future
    steps of the decoded horizon and FDE is taken at the last valid one: ``min_ade`` /
    ``min_fde`` over modes, ``top1_*`` for the most probable mode (temperature does not
    change it), ``expected_*`` = sum_k p_k * error_k, and ``p_best`` / ``nll_best`` = the
    probability (and -log) given to the min-ADE mode. The last two depend on temperature.
    ``error`` / ``<name>`` cover every agent; ``error_moving`` / ``<name>_moving`` only agents
    whose logged position moves more than MOVING_DISP_M over the horizon (parked cars are
    predicted near-perfectly and dominate the all-agent means).
    """

    MOVING_DISP_M = 1.0

    def __init__(self) -> None:
        self.num_forecasts = 0
        self.num_agents = 0
        self.num_agents_with_future = 0
        self.horizon_s = 0.0
        self._sums: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}

    def _add(self, key: str, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[0] == 0:
            return
        moments = np.stack([x.sum(axis=0), (x * x).sum(axis=0)])
        if key in self._sums:
            self._sums[key] += moments
            self._counts[key] += x.shape[0]
        else:
            self._sums[key] = moments
            self._counts[key] = x.shape[0]

    def update(
        self,
        probs: dict[str, np.ndarray],
        rows: np.ndarray,
        traj_k: np.ndarray | None = None,
        gt_xy: np.ndarray | None = None,
        gt_valid: np.ndarray | None = None,
        horizon_s: float = 0.0,
    ) -> None:
        """probs: name -> [B, N, K]; rows: [B, N] agents to count; traj_k [B, N, K, H, 2+]
        decoded futures with index h <-> gt_xy [B, N, H, 2] / gt_valid [B, N, H] (h = 0 is now)."""
        self.num_forecasts += 1
        n = int(rows.sum())
        if n == 0:
            return
        self.num_agents += n
        for name, p in probs.items():
            p = np.asarray(p, dtype=np.float64)[rows]                      # [n, K]
            entropy = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=-1)  # nats
            self._add(f"{name}/by_mode", p)
            self._add(f"{name}/by_rank", -np.sort(-p, axis=-1))
            self._add(f"{name}/entropy", entropy)
            self._add(f"{name}/effective_modes", np.exp(entropy))
        if traj_k is None or gt_xy is None:
            return
        pred = np.asarray(traj_k, dtype=np.float64)[rows][..., 1:, :2]    # [n, K, H-1, 2]
        gt = np.asarray(gt_xy, dtype=np.float64)[rows][:, None, 1:]       # [n, 1, H-1, 2]
        ok = np.asarray(gt_valid, dtype=bool)[rows][:, 1:]                # [n, H-1]
        has = ok.any(axis=-1)
        if not has.any():
            return
        pred, gt, ok = pred[has], gt[has], ok[has]
        self.num_agents_with_future += int(has.sum())
        self.horizon_s = max(self.horizon_s, float(horizon_s))
        err = np.linalg.norm(pred - gt, axis=-1)                           # [m, K, H-1]
        ade_k = (err * ok[:, None]).sum(-1) / ok.sum(-1)[:, None]          # [m, K]
        last = (ok.shape[1] - 1) - np.argmax(ok[:, ::-1], axis=-1)         # last valid step
        fde_k = np.take_along_axis(err, last[:, None, None], axis=-1)[..., 0]
        best = ade_k.argmin(-1)
        p_raw = np.asarray(probs["raw"], dtype=np.float64)[rows][has]
        top1 = p_raw.argmax(-1)
        start = np.asarray(gt_xy, dtype=np.float64)[rows][has][:, 0]                       # [m, 2] now
        disp = np.where(ok, np.linalg.norm(gt[:, 0] - start[:, None], axis=-1), 0.0).max(-1)
        moving = disp > self.MOVING_DISP_M
        pick = lambda a, idx: np.take_along_axis(a, idx[:, None], axis=-1)[:, 0]  # noqa: E731
        for suffix, sel in (("", slice(None)), ("_moving", moving)):
            self._add(f"error{suffix}/min_ade", ade_k.min(-1)[sel])
            self._add(f"error{suffix}/min_fde", fde_k.min(-1)[sel])
            self._add(f"error{suffix}/top1_ade", pick(ade_k, top1)[sel])
            self._add(f"error{suffix}/top1_fde", pick(fde_k, top1)[sel])
            self._add(f"error{suffix}/ade_by_mode", ade_k[sel])
            for name, p in probs.items():
                p = np.asarray(p, dtype=np.float64)[rows][has]
                w = p / np.clip(p.sum(-1, keepdims=True), 1e-12, None)
                self._add(f"{name}{suffix}/expected_ade", (w * ade_k).sum(-1)[sel])
                self._add(f"{name}{suffix}/expected_fde", (w * fde_k).sum(-1)[sel])
                self._add(f"{name}{suffix}/p_best", pick(w, best)[sel])
                self._add(f"{name}{suffix}/nll_best", -np.log(np.clip(pick(w, best), 1e-12, None))[sel])

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "num_forecasts": self.num_forecasts,
            "num_agents": self.num_agents,
            "num_agents_with_future": self.num_agents_with_future,
            "error_horizon_s": self.horizon_s,
        }
        for key, (s1, s2) in sorted(self._sums.items()):
            group, stat = key.split("/")
            n = max(self._counts[key], 1)
            mean = s1 / n
            std = np.sqrt(np.maximum(s2 / n - mean * mean, 0.0))
            if stat in ("by_mode", "by_rank", "ade_by_mode"):
                entry = {"mean": mean.tolist(), "std": std.tolist(), "n": self._counts[key]}
            else:
                entry = {"mean": float(mean[0]), "std": float(std[0]), "n": self._counts[key]}
            out.setdefault(group, {})[stat] = entry
        return out


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
        mode_temperature: float = 1.0,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.planning_mode = canonicalize_planning_mode(planning_mode or config.planning_mode)
        self._cache: dict[tuple[int, int, int, float], dict[str, np.ndarray]] = {}
        self._last_collision_debug: dict[str, Any] | None = None
        # Scoring uses softmax(mode_logits / mode_temperature); 1.0 = the classifier as trained.
        self.mode_temperature = float(mode_temperature)
        self.forecast_stats = ForecastStats()
        # If set, forecast_stats_summary() is also written here every stats_every forward passes.
        self.stats_path: str | Path | None = None
        self.stats_every = 50

    @property
    def mode_temperature(self) -> float:
        return self._mode_temperature

    @mode_temperature.setter
    def mode_temperature(self, value: float) -> None:
        if not float(value) > 0.0:
            raise ValueError(f"mode_temperature must be > 0, got {value}")
        self._mode_temperature = float(value)
        self._cache.clear()  # cached probabilities were computed at the old temperature

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        planning_mode: str | None = None,
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
        forecaster = cls(model, config, device=device, planning_mode=mode)
        # Independent-agent risk is scored via score_independent_mode_collision;
        # do not advertise K joint worlds to the gaussian ensemble SAT path.
        forecaster.num_collision_worlds = 1
        forecaster.collision_reduce = "any"
        return forecaster

    def clear_cache(self) -> None:
        self._cache.clear()

    def forecast_stats_summary(self) -> dict[str, Any]:
        """Mode-probability and ADE / FDE statistics accumulated so far (see ForecastStats)."""
        return {"mode_temperature": self.mode_temperature, **self.forecast_stats.summary()}

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
        raw_probs = pred["mode_probs"].detach().float()
        if self.mode_temperature == 1.0:
            mode_probs_t = raw_probs
        else:
            # Invalid agents have all-zero probabilities; keep them zero.
            valid = (raw_probs.sum(dim=-1, keepdim=True) > 0).to(raw_probs.dtype)
            mode_probs_t = torch.softmax(pred["mode_logits"].detach().float() / self.mode_temperature, dim=-1) * valid
        mode_probs = mode_probs_t.cpu().numpy()
        raw_probs = raw_probs.cpu().numpy()
        # Temperature does not change the argmax, so top-1 mode and top1_traj are unchanged.
        top1_mode = pred["top1_mode"].detach().cpu().numpy().astype(np.int32)
        modeled_n = min(int(meta["modeled_agents"]), mode_probs.shape[1])
        t_now = int(meta["timestep"])
        rows = (
            meta["valid"][:, :modeled_n, t_now]
            & np.isin(meta["object_types"][:, :modeled_n], _DYNAMIC_TYPES)
            & ~meta["is_sdc"][:, :modeled_n]
            & (raw_probs[:, :modeled_n].sum(axis=-1) > 0)
        )
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

        # Logged future at the decoded times (h * dt after now), for ADE / FDE.
        stats_kw: dict[str, Any] = {}
        if "traj_k" in out:
            num_steps = meta["x"].shape[-1]
            steps = t_now + np.rint(np.arange(int(traj_length)) * float(dt) / float(self.config.world_dt)).astype(int)
            in_log = steps < num_steps
            steps = np.minimum(steps, num_steps - 1)
            stats_kw = dict(
                traj_k=out["traj_k"][:, :modeled_n],
                gt_xy=np.stack([meta["x"][:, :modeled_n][..., steps], meta["y"][:, :modeled_n][..., steps]], axis=-1),
                gt_valid=meta["valid"][:, :modeled_n][..., steps] & in_log,
                horizon_s=(int(traj_length) - 1) * float(dt),
            )
        self.forecast_stats.update(
            {"raw": raw_probs[:, :modeled_n], "tempered": mode_probs[:, :modeled_n]}, rows, **stats_kw
        )
        if self.stats_path is not None and self.forecast_stats.num_forecasts % int(self.stats_every) == 0:
            self.write_forecast_stats(self.stats_path)
        return out

    def write_forecast_stats(self, path: str | Path) -> None:
        import json

        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.forecast_stats_summary(), indent=2))
        tmp.replace(path)

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


__all__ = ["ForecastStats", "WorldModelForecaster"]
