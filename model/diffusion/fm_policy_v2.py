"""FlowMatchingPolicy of branch new_model (model/diffusion/fm_policy.py at f82488d), for its checkpoints
(metadata without ``inst_dim``). fm_policy.py keeps the earlier instruction / subgoal-conditioned policy."""
from __future__ import annotations

from typing import Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, struct
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from model.diffusion.flow_matching_v2 import FlowMatching
from model.diffusion.unet import UNet1DConditioned
from model.diffusion.modules.scene_tokenizer import SceneTokenizer


# ======================================================================
# Policy feature container
# ======================================================================


@struct.dataclass
class PolicyFeatures:
    ego_state: jnp.ndarray
    goal_xy: jnp.ndarray
    remaining_timesteps: jnp.ndarray
    other_states: jnp.ndarray
    other_valid: jnp.ndarray
    map_features: jnp.ndarray
    map_valid: jnp.ndarray
    traffic_light_features: jnp.ndarray
    traffic_light_valid: jnp.ndarray
    ego_trajectory: Optional[jnp.ndarray] = None

    @staticmethod
    def from_mapping(
        features: Mapping[str, jnp.ndarray],
    ) -> "PolicyFeatures":
        required = (
            "ego_state",
            "goal_xy",
            "remaining_timesteps",
            "other_states",
            "other_valid",
            "map_features",
            "map_valid",
            "traffic_light_features",
            "traffic_light_valid",
        )
        missing = [key for key in required if key not in features]
        if missing:
            raise KeyError(f"Missing required feature keys: {missing}")
        return PolicyFeatures(
            ego_state=features["ego_state"],
            goal_xy=features["goal_xy"],
            remaining_timesteps=features["remaining_timesteps"],
            other_states=features["other_states"],
            other_valid=features["other_valid"],
            map_features=features["map_features"],
            map_valid=features["map_valid"],
            traffic_light_features=features["traffic_light_features"],
            traffic_light_valid=features["traffic_light_valid"],
            ego_trajectory=features.get("ego_trajectory"),
        )


# ======================================================================
# Flow Matching Policy
# ======================================================================


class FlowMatchingPolicy(nnx.Module):

    def __init__(
        self,
        rngs: nnx.Rngs,
        target_dim: int,
        hidden_dim: int,
        cond_dim: int,
        map_attr_dim: int,
        tl_attr_dim: int,
        ego_dim: int = 5,
        other_dim: int = 15,
        predict_horizon: int = 25,
        goal_conditioned: bool = True,
        num_denoising_steps: int = 50,
    ) -> None:
        self.target_dim = target_dim
        self.predict_horizon = predict_horizon
        self.cond_dim = cond_dim
        self.goal_conditioned = bool(goal_conditioned)

        # --------------------------------------------------------------
        # Scene conditioning
        # --------------------------------------------------------------

        self.scene_tokenizer = SceneTokenizer(
            ego_dim=ego_dim,
            other_dim=other_dim,
            map_attr_dim=map_attr_dim,
            tl_attr_dim=tl_attr_dim,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            token_num=32,
            include_goal=self.goal_conditioned,
            rngs=rngs,
        )

        # --------------------------------------------------------------
        # Velocity field
        # --------------------------------------------------------------

        velocity_fn = UNet1DConditioned(
            in_ch=target_dim,
            cond_dim=cond_dim,
            time_emb_dim=cond_dim,
            horizon=predict_horizon,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )

        self.flow = FlowMatching(
            velocity_fn=velocity_fn,
            num_denoising_steps=num_denoising_steps,
        )

        # --------------------------------------------------------------
        # Data parallel sharding
        # --------------------------------------------------------------

        self._mesh: Optional[Mesh] = None
        self._cond_sharding: Optional[
            NamedSharding
        ] = None
        self._mask_sharding: Optional[
            NamedSharding
        ] = None
        self._traj_sharding: Optional[
            NamedSharding
        ] = None

        self._init_data_parallel_sharding()

    # ==================================================================
    # Sharding
    # ==================================================================

    def _init_data_parallel_sharding(
        self,
    ) -> None:

        gpu_devices = [
            device
            for device in jax.devices()
            if device.platform == "gpu"
        ]

        if len(gpu_devices) <= 1:
            return

        self._mesh = Mesh(
            np.asarray(gpu_devices),
            ("data",),
        )

        self._cond_sharding = NamedSharding(
            self._mesh,
            P("data", None),
        )

        self._mask_sharding = NamedSharding(
            self._mesh,
            P("data"),
        )

        self._traj_sharding = NamedSharding(
            self._mesh,
            P("data", None, None),
        )

    @staticmethod
    def _as_features(
        features: (
            PolicyFeatures
            | Mapping[str, jnp.ndarray]
        ),
    ) -> PolicyFeatures:

        if isinstance(
            features,
            PolicyFeatures,
        ):
            return features

        return PolicyFeatures.from_mapping(
            features
        )

    def supports_data_parallel(
        self,
    ) -> bool:
        return self._mesh is not None

    def shard_condition(
        self,
        cond: jnp.ndarray,
    ) -> jnp.ndarray:

        if self._cond_sharding is None:
            return cond

        return jax.device_put(
            cond,
            self._cond_sharding,
        )

    def shard_mask(
        self,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:

        if self._mask_sharding is None:
            return mask

        return jax.device_put(
            mask,
            self._mask_sharding,
        )

    def shard_trajectory(
        self,
        traj: jnp.ndarray,
    ) -> jnp.ndarray:

        if self._traj_sharding is None:
            return traj

        return jax.device_put(
            traj,
            self._traj_sharding,
        )

    # ==================================================================
    # Conditioning
    # ==================================================================

    def _condition_impl(
        self,
        features: PolicyFeatures,
    ) -> jnp.ndarray:

        return self.scene_tokenizer._tokenize_scene_features(
            input_features={
                "ego_state": features.ego_state,
                "goal_xy": features.goal_xy,
                "remaining_timesteps": features.remaining_timesteps,
                "other_states": features.other_states,
                "other_valid": features.other_valid,
                "map_features": features.map_features,
                "map_valid": features.map_valid,
                "traffic_light_features": features.traffic_light_features,
                "traffic_light_valid": features.traffic_light_valid,
            },
            deterministic=True,
        )

    def compute_condition(
        self,
        input_features: (
            PolicyFeatures
            | Mapping[str, jnp.ndarray]
        ),
    ) -> jnp.ndarray:

        features = self._as_features(
            input_features
        )
        return self._condition_impl(features)

    def input_features_projection(
        self,
        input_features: (
            PolicyFeatures
            | Mapping[str, jnp.ndarray]
        ),
    ) -> jnp.ndarray:

        return self.compute_condition(
            input_features
        )

    # ==================================================================
    # Sampling from precomputed conditions
    # ==================================================================

    def _sample_from_condition_impl(
        self,
        cond: jnp.ndarray,
        rng: jax.Array,
        num_steps: Optional[int],
        solver: str,
    ) -> jnp.ndarray:

        batch_size = cond.shape[0]

        pred = self.flow.sample(
            shape=(
                batch_size,
                self.target_dim,
                self.predict_horizon,
            ),
            cond=cond,
            rng=rng,
            num_steps=num_steps,
            solver=solver,
        )

        # [B, D, T] -> [B, T, D]
        return jnp.transpose(
            pred,
            (0, 2, 1),
        )

    def sample_from_condition(
        self,
        cond: jnp.ndarray,
        *,
        rng: jax.Array,
        data_parallel: bool = False,
        num_steps: Optional[int] = None,
        solver: str = "euler",
    ) -> jnp.ndarray:

        if data_parallel:
            cond = self.shard_condition(cond)

        return self._sample_from_condition_impl(
            cond=cond,
            rng=rng,
            num_steps=num_steps,
            solver=solver,
        )

    # ==================================================================
    # Loss
    # ==================================================================

    def _loss_from_condition_impl(
        self,
        target_btd: jnp.ndarray,
        cond: jnp.ndarray,
        rng: jax.Array,
    ) -> jnp.ndarray:

        # Policy trajectory is [B, T, D]. The flow model uses [B, D, T].
        target_bdt = jnp.transpose(
            target_btd,
            (0, 2, 1),
        )

        return self.flow.loss(
            target_bdt,
            cond=cond,
            rng=rng,
        )

    def loss_from_condition(
        self,
        target_btd: jnp.ndarray,
        cond: jnp.ndarray,
        *,
        rng: jax.Array,
        data_parallel: bool = False,
        pre_sharded: bool = False,
    ) -> jnp.ndarray:

        if data_parallel:
            target_btd = (
                target_btd
                if pre_sharded
                else self.shard_trajectory(
                    target_btd
                )
            )
            cond = (
                cond
                if pre_sharded
                else self.shard_condition(cond)
            )

        return self._loss_from_condition_impl(
            target_btd=target_btd,
            cond=cond,
            rng=rng,
        )

    # ==================================================================
    # Proposal refinement
    # ==================================================================

    def _resample_from_condition_impl(
        self,
        cond: jnp.ndarray,
        proposals_btd: jnp.ndarray,
        rng: jax.Array,
        t_start: float,
        num_steps: Optional[int],
        solver: str,
    ) -> jnp.ndarray:

        proposals_bdt = jnp.transpose(
            proposals_btd,
            (0, 2, 1),
        )

        x = self.flow.resample(
            proposals=proposals_bdt,
            cond=cond,
            rng=rng,
            t_start=t_start,
            num_steps=num_steps,
            solver=solver,
        )

        return jnp.transpose(
            x,
            (0, 2, 1),
        )

    def resample_from_condition(
        self,
        cond: jnp.ndarray,
        proposals_btd: jnp.ndarray,
        *,
        rng: jax.Array,
        t_start: float = 0.2,
        num_steps: Optional[int] = None,
        solver: str = "euler",
        data_parallel: bool = False,
    ) -> jnp.ndarray:

        if data_parallel:
            cond = self.shard_condition(cond)
            proposals_btd = (
                self.shard_trajectory(
                    proposals_btd
                )
            )

        return self._resample_from_condition_impl(
            cond=cond,
            proposals_btd=proposals_btd,
            rng=rng,
            t_start=t_start,
            num_steps=num_steps,
            solver=solver,
        )

    # ==================================================================
    # Public policy API
    # ==================================================================

    def forward(
        self,
        input_features: (
            PolicyFeatures
            | Mapping[str, jnp.ndarray]
        ),
        rng: jax.Array,
        num_steps: Optional[int] = None,
        solver: str = "euler",
    ) -> jnp.ndarray:

        return self.sample(
            input_features,
            rng=rng,
            num_steps=num_steps,
            solver=solver,
        )

    def sample(
        self,
        input_features: (
            PolicyFeatures
            | Mapping[str, jnp.ndarray]
        ),
        *,
        rng: jax.Array,
        data_parallel: bool = False,
        num_steps: Optional[int] = None,
        solver: str = "euler",
    ) -> jnp.ndarray:

        features = self._as_features(
            input_features
        )
        cond = self.compute_condition(features)

        return self.sample_from_condition(
            cond=cond,
            rng=rng,
            data_parallel=data_parallel,
            num_steps=num_steps,
            solver=solver,
        )

    def loss(
        self,
        input_features: (
            PolicyFeatures
            | Mapping[str, jnp.ndarray]
        ),
        *,
        rng: jax.Array,
        data_parallel: bool = False,
    ) -> jnp.ndarray:

        features = self._as_features(
            input_features
        )

        if features.ego_trajectory is None:
            raise ValueError(
                "ego_trajectory is required "
                "for loss()"
            )

        target = features.ego_trajectory[
            :,
            : self.predict_horizon,
            :,
        ]
        cond = self.compute_condition(features)

        return self.loss_from_condition(
            target_btd=target,
            cond=cond,
            rng=rng,
            data_parallel=data_parallel,
        )

    def resample(
        self,
        input_features: (
            PolicyFeatures
            | Mapping[str, jnp.ndarray]
        ),
        proposals: jnp.ndarray,
        *,
        rng: jax.Array,
        t_start: float = 0.2,
        num_steps: Optional[int] = None,
        solver: str = "euler",
        data_parallel: bool = False,
    ) -> jnp.ndarray:

        features = self._as_features(
            input_features
        )
        cond = self.compute_condition(features)

        return self.resample_from_condition(
            cond=cond,
            proposals_btd=proposals,
            rng=rng,
            t_start=t_start,
            num_steps=num_steps,
            solver=solver,
            data_parallel=data_parallel,
        )
