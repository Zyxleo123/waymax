from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, struct
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from model.diffusion.flow_matching import UNet1DConditioned
from model.diffusion.modules.scene_tokenizer import SceneTokenizer
from model.diffusion.modules.mlp import MLP


# ======================================================================
# Flow Matching
# ======================================================================


class FlowMatching(nnx.Module):
    """Linear Flow Matching / Rectified Flow.

    Convention
    ----------
    t = 0:
        Data distribution.

    t = 1:
        Standard Gaussian distribution.

    Forward probability path:

        x_t = (1 - t) * x_0 + t * eps

        eps ~ N(0, I)

    Therefore,

        dx_t / dt = eps - x_0

    and the model learns

        v_theta(x_t, t, cond) ~= eps - x_0

    Sampling starts from Gaussian noise at t=1 and integrates the learned
    vector field backward toward t=0.
    """

    def __init__(
        self,
        velocity_fn: nnx.Module,
        default_num_steps: int = 100,
        clip_output: bool = False,
    ):
        self.velocity_fn = velocity_fn
        self.default_num_steps = int(default_num_steps)
        self.clip_output = bool(clip_output)

    # ------------------------------------------------------------------
    # Time utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_t(
        t: jnp.ndarray,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Expand [B] time to [B, 1, 1] for trajectory broadcasting."""
        return t.reshape((x.shape[0], 1, 1))

    def _model_time(
        self,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Map normalized FM time [0, 1] to timestep-embedding scale.

        The existing UNet was originally built for diffusion timesteps around
        [0, 100). Feeding raw [0, 1] values would make sinusoidal timestep
        embeddings vary relatively little.

        We therefore preserve the old embedding scale without changing the
        actual Flow Matching definition.
        """
        return t * float(self.default_num_steps - 1)

    # ------------------------------------------------------------------
    # Probability path
    # ------------------------------------------------------------------

    def interpolate(
        self,
        x0: jnp.ndarray,
        noise: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Linear interpolation between data and Gaussian noise.

        Args
        ----
        x0:
            [B, D, T]

        noise:
            [B, D, T]

        t:
            [B], continuous values in [0, 1]
        """
        t_expanded = self._expand_t(t, x0)

        return (
            (1.0 - t_expanded) * x0
            + t_expanded * noise
        )

    @staticmethod
    def target_velocity(
        x0: jnp.ndarray,
        noise: jnp.ndarray,
    ) -> jnp.ndarray:
        """Exact conditional velocity for the linear path."""
        return noise - x0

    # ------------------------------------------------------------------
    # Velocity prediction
    # ------------------------------------------------------------------

    def predict_velocity(
        self,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate learned vector field."""
        model_t = self._model_time(t)

        return self.velocity_fn(
            x_t,
            model_t,
            cond1,
            cond2,
            cond2_mask,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        x0: jnp.ndarray,
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        noise: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Flow Matching MSE objective."""

        x_t = self.interpolate(
            x0=x0,
            noise=noise,
            t=t,
        )

        pred_v = self.predict_velocity(
            x_t=x_t,
            t=t,
            cond1=cond1,
            cond2=cond2,
            cond2_mask=cond2_mask,
        )

        target_v = self.target_velocity(
            x0=x0,
            noise=noise,
        )

        return jnp.mean(
            (pred_v - target_v) ** 2
        )

    def loss(
        self,
        x0: jnp.ndarray,
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        *,
        rng: jax.Array,
    ) -> jnp.ndarray:
        """Sample continuous t and Gaussian endpoint and compute FM loss."""

        batch_size = x0.shape[0]

        key_t, key_noise = jax.random.split(rng)

        t = jax.random.uniform(
            key_t,
            shape=(batch_size,),
            minval=0.0,
            maxval=1.0,
            dtype=jnp.float32,
        )

        noise = jax.random.normal(
            key_noise,
            shape=x0.shape,
            dtype=x0.dtype,
        )

        return self.compute_loss(
            x0=x0,
            cond1=cond1,
            cond2=cond2,
            cond2_mask=cond2_mask,
            noise=noise,
            t=t,
        )

    # ------------------------------------------------------------------
    # ODE solvers
    # ------------------------------------------------------------------

    def _euler_step(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        dt: jnp.ndarray,
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """One backward Euler step.

        v_theta predicts the forward-time vector field dx/dt.

        Sampling goes from larger t to smaller t, therefore:

            x(t - dt) = x(t) - dt * v_theta(x(t), t)
        """

        v = self.predict_velocity(
            x_t=x,
            t=t,
            cond1=cond1,
            cond2=cond2,
            cond2_mask=cond2_mask,
        )

        return x - dt * v

    def _heun_step(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        dt: jnp.ndarray,
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Second-order Heun ODE step."""

        v1 = self.predict_velocity(
            x_t=x,
            t=t,
            cond1=cond1,
            cond2=cond2,
            cond2_mask=cond2_mask,
        )

        x_euler = x - dt * v1

        t_next = jnp.maximum(
            t - dt,
            0.0,
        )

        v2 = self.predict_velocity(
            x_t=x_euler,
            t=t_next,
            cond1=cond1,
            cond2=cond2,
            cond2_mask=cond2_mask,
        )

        return x - dt * 0.5 * (v1 + v2)

    def _integrate(
        self,
        x_init: jnp.ndarray,
        start_t: float,
        num_steps: int,
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        solver: str,
    ) -> jnp.ndarray:
        """Integrate vector field from start_t to 0."""

        if num_steps <= 0:
            raise ValueError(
                f"num_steps must be positive, got {num_steps}"
            )

        if solver not in ("euler", "heun"):
            raise ValueError(
                f"Unknown solver: {solver}"
            )

        batch_size = x_init.shape[0]

        dt = jnp.asarray(
            start_t / float(num_steps),
            dtype=x_init.dtype,
        )

        def body(
            i: int,
            x_curr: jnp.ndarray,
        ) -> jnp.ndarray:

            i_float = i.astype(x_init.dtype)

            t_scalar = (
                jnp.asarray(start_t, dtype=x_init.dtype)
                - i_float * dt
            )

            t = jnp.full(
                (batch_size,),
                t_scalar,
                dtype=x_init.dtype,
            )

            if solver == "euler":
                return self._euler_step(
                    x=x_curr,
                    t=t,
                    dt=dt,
                    cond1=cond1,
                    cond2=cond2,
                    cond2_mask=cond2_mask,
                )

            return self._heun_step(
                x=x_curr,
                t=t,
                dt=dt,
                cond1=cond1,
                cond2=cond2,
                cond2_mask=cond2_mask,
            )

        x_final = jax.lax.fori_loop(
            0,
            num_steps,
            body,
            x_init,
        )

        if self.clip_output:
            x_final = jnp.clip(
                x_final,
                -1.0,
                1.0,
            )

        return x_final

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def sample(
        self,
        shape: Tuple[int, int, int],
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        *,
        rng: jax.Array,
        num_steps: Optional[int] = None,
        solver: str = "euler",
    ) -> jnp.ndarray:
        """Generate trajectories from the standard Gaussian prior.

        Args
        ----
        shape:
            [B, D, T]

        num_steps:
            Number of ODE integration steps.

        solver:
            "euler" or "heun".
        """

        if num_steps is None:
            num_steps = self.default_num_steps

        x_init = jax.random.normal(
            rng,
            shape=shape,
            dtype=jnp.float32,
        )

        return self._integrate(
            x_init=x_init,
            start_t=1.0,
            num_steps=num_steps,
            cond1=cond1,
            cond2=cond2,
            cond2_mask=cond2_mask,
            solver=solver,
        )

    # ------------------------------------------------------------------
    # Proposal refinement
    # ------------------------------------------------------------------

    def resample(
        self,
        proposals: jnp.ndarray,
        cond1: jnp.ndarray,
        cond2: jnp.ndarray,
        cond2_mask: jnp.ndarray,
        *,
        rng: jax.Array,
        t_start: float = 0.2,
        num_steps: Optional[int] = None,
        solver: str = "euler",
    ) -> jnp.ndarray:
        """Corrupt proposals to t_start and integrate back to data.

        This is the Flow Matching analogue of partial diffusion denoising.

        t_start controls perturbation magnitude independently of the number
        of numerical ODE steps.

        Example
        -------
        t_start = 0.2

            proposal at t=0
                |
                | interpolate toward Gaussian noise
                v
              x_0.2
                |
                | learned ODE
                v
            refined sample at t=0
        """

        if not (0.0 < t_start <= 1.0):
            raise ValueError(
                f"t_start must be in (0, 1], got {t_start}"
            )

        if num_steps is None:
            num_steps = max(
                1,
                int(round(
                    self.default_num_steps * t_start
                )),
            )

        noise = jax.random.normal(
            rng,
            shape=proposals.shape,
            dtype=proposals.dtype,
        )

        batch_size = proposals.shape[0]

        t = jnp.full(
            (batch_size,),
            t_start,
            dtype=proposals.dtype,
        )

        x_init = self.interpolate(
            x0=proposals,
            noise=noise,
            t=t,
        )

        return self._integrate(
            x_init=x_init,
            start_t=t_start,
            num_steps=num_steps,
            cond1=cond1,
            cond2=cond2,
            cond2_mask=cond2_mask,
            solver=solver,
        )


# ======================================================================
# Policy feature container
# ======================================================================


@struct.dataclass
class PolicyFeatures:
    ego_state: jnp.ndarray
    goal_xy: jnp.ndarray
    subgoal_xy: jnp.ndarray
    remaining_timesteps: jnp.ndarray

    other_states: jnp.ndarray
    other_valid: jnp.ndarray

    map_features: jnp.ndarray
    map_valid: jnp.ndarray

    traffic_light_features: jnp.ndarray
    traffic_light_valid: jnp.ndarray

    inst_features: Optional[jnp.ndarray] = None
    inst_valid: Optional[jnp.ndarray] = None

    subgoal_valid: Optional[jnp.ndarray] = None
    ego_trajectory: Optional[jnp.ndarray] = None

    @staticmethod
    def from_mapping(
        features: Mapping[str, jnp.ndarray],
    ) -> "PolicyFeatures":

        required = (
            "ego_state",
            "goal_xy",
            "subgoal_xy",
            "subgoal_valid",
            "remaining_timesteps",
            "other_states",
            "other_valid",
            "map_features",
            "map_valid",
            "traffic_light_features",
            "traffic_light_valid",
            "inst_features",
            "inst_valid",
        )

        missing = [
            key
            for key in required
            if key not in features
        ]

        if missing:
            raise KeyError(
                f"Missing required feature keys: {missing}"
            )

        return PolicyFeatures(
            ego_state=features["ego_state"],
            goal_xy=features["goal_xy"],
            subgoal_xy=features["subgoal_xy"],
            subgoal_valid=features["subgoal_valid"],
            remaining_timesteps=features["remaining_timesteps"],
            other_states=features["other_states"],
            other_valid=features["other_valid"],
            map_features=features["map_features"],
            map_valid=features["map_valid"],
            traffic_light_features=features[
                "traffic_light_features"
            ],
            traffic_light_valid=features[
                "traffic_light_valid"
            ],
            inst_features=features.get(
                "inst_features"
            ),
            inst_valid=features.get(
                "inst_valid"
            ),
            ego_trajectory=features.get(
                "ego_trajectory"
            ),
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
        inst_attr_dim: int,
        ego_dim: int = 5,
        other_dim: int = 15,
        predict_horizon: int = 25,
        subgoal_conditioned: bool = False,
        default_num_steps: int = 100,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise TypeError(
                "Unexpected FlowMatchingPolicy arguments: "
                f"{sorted(kwargs.keys())}. "
                "Note: flow_num_steps / flow_time_embedding_scale are not "
                "constructor args; default_num_steps controls model_time = "
                "t * (default_num_steps - 1). Pass ODE step counts at "
                "sample/resample time via num_steps."
            )

        self.target_dim = target_dim
        self.predict_horizon = predict_horizon
        self.subgoal_conditioned = subgoal_conditioned

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
            rngs=rngs,
        )

        # --------------------------------------------------------------
        # Instruction conditioning
        # --------------------------------------------------------------

        self.instruction_encoder = MLP(
            [
                inst_attr_dim,
                hidden_dim,
                hidden_dim,
                cond_dim,
            ],
            rngs=rngs,
        )

        if subgoal_conditioned:
            self.subgoal_encoder = MLP(
                [
                    2,
                    hidden_dim,
                    hidden_dim,
                    cond_dim,
                ],
                rngs=rngs,
            )

        # --------------------------------------------------------------
        # Velocity field
        # --------------------------------------------------------------

        velocity_fn = UNet1DConditioned(
            in_ch=target_dim,
            cond_dim=cond_dim,
            cond2_dim=cond_dim,
            time_emb_dim=cond_dim,
            horizon=predict_horizon,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )

        self.flow = FlowMatching(
            velocity_fn=velocity_fn,
            default_num_steps=default_num_steps,
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

        return (
            self.scene_tokenizer
            ._tokenize_scene_features(
                input_features=features.__dict__,
                deterministic=True,
            )
        )

    def _instruction_condition_impl(
        self,
        features: PolicyFeatures,
    ) -> jnp.ndarray:

        x_inst = self.instruction_encoder(
            features.inst_features,
            deterministic=True,
        )

        if self.subgoal_conditioned:

            x_subgoal = self.subgoal_encoder(
                features.subgoal_xy,
                deterministic=True,
            )

            x_subgoal = (
                x_subgoal
                * features.subgoal_valid[..., None]
            )

        else:
            x_subgoal = 0.0

        return x_inst + x_subgoal

    def compute_condition(
        self,
        input_features: (
            PolicyFeatures
            | Mapping[str, jnp.ndarray]
        ),
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        features = self._as_features(
            input_features
        )

        return (
            self._condition_impl(features),
            self._instruction_condition_impl(
                features
            ),
        )

    def input_features_projection(
        self,
        input_features: (
            PolicyFeatures
            | Mapping[str, jnp.ndarray]
        ),
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        return self.compute_condition(
            input_features
        )

    # ==================================================================
    # Sampling from precomputed conditions
    # ==================================================================

    def _sample_from_condition_impl(
        self,
        cond: jnp.ndarray,
        inst_cond: jnp.ndarray,
        inst_cond_mask: jnp.ndarray,
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
            cond1=cond,
            cond2=inst_cond,
            cond2_mask=inst_cond_mask,
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
        inst_cond: jnp.ndarray,
        inst_cond_mask: jnp.ndarray,
        *,
        rng: jax.Array,
        data_parallel: bool = False,
        num_steps: Optional[int] = None,
        solver: str = "euler",
    ) -> jnp.ndarray:

        if data_parallel:
            cond = self.shard_condition(cond)
            inst_cond = self.shard_condition(
                inst_cond
            )
            inst_cond_mask = self.shard_mask(
                inst_cond_mask
            )

        return self._sample_from_condition_impl(
            cond=cond,
            inst_cond=inst_cond,
            inst_cond_mask=inst_cond_mask,
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
        inst_cond: jnp.ndarray,
        inst_cond_mask: jnp.ndarray,
        rng: jax.Array,
    ) -> jnp.ndarray:

        # Policy trajectory:
        # [B, T, D]
        #
        # Flow model:
        # [B, D, T]

        target_bdt = jnp.transpose(
            target_btd,
            (0, 2, 1),
        )

        return self.flow.loss(
            target_bdt,
            cond1=cond,
            cond2=inst_cond,
            cond2_mask=inst_cond_mask,
            rng=rng,
        )

    def loss_from_condition(
        self,
        target_btd: jnp.ndarray,
        cond: jnp.ndarray,
        inst_cond: jnp.ndarray,
        inst_cond_mask: jnp.ndarray,
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

            inst_cond = (
                inst_cond
                if pre_sharded
                else self.shard_condition(
                    inst_cond
                )
            )

            inst_cond_mask = (
                inst_cond_mask
                if pre_sharded
                else self.shard_mask(
                    inst_cond_mask
                )
            )

        return self._loss_from_condition_impl(
            target_btd=target_btd,
            cond=cond,
            inst_cond=inst_cond,
            inst_cond_mask=inst_cond_mask,
            rng=rng,
        )

    # ==================================================================
    # Proposal refinement
    # ==================================================================

    def _resample_from_condition_impl(
        self,
        cond: jnp.ndarray,
        inst_cond: jnp.ndarray,
        inst_cond_mask: jnp.ndarray,
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
            cond1=cond,
            cond2=inst_cond,
            cond2_mask=inst_cond_mask,
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
        inst_cond: jnp.ndarray,
        inst_cond_mask: jnp.ndarray,
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
            inst_cond = self.shard_condition(
                inst_cond
            )
            inst_cond_mask = self.shard_mask(
                inst_cond_mask
            )
            proposals_btd = (
                self.shard_trajectory(
                    proposals_btd
                )
            )

        return self._resample_from_condition_impl(
            cond=cond,
            inst_cond=inst_cond,
            inst_cond_mask=inst_cond_mask,
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

        cond, inst_cond = (
            self.compute_condition(features)
        )

        inst_cond_mask = jnp.asarray(
            features.inst_valid,
            dtype=bool,
        )

        return self.sample_from_condition(
            cond=cond,
            inst_cond=inst_cond,
            inst_cond_mask=inst_cond_mask,
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

        cond, inst_cond = (
            self.compute_condition(features)
        )

        inst_cond_mask = jnp.asarray(
            features.inst_valid,
            dtype=bool,
        )

        return self.loss_from_condition(
            target_btd=target,
            cond=cond,
            inst_cond=inst_cond,
            inst_cond_mask=inst_cond_mask,
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

        cond, inst_cond = (
            self.compute_condition(features)
        )

        inst_cond_mask = jnp.asarray(
            features.inst_valid,
            dtype=bool,
        )

        return self.resample_from_condition(
            cond=cond,
            inst_cond=inst_cond,
            inst_cond_mask=inst_cond_mask,
            proposals_btd=proposals,
            rng=rng,
            t_start=t_start,
            num_steps=num_steps,
            solver=solver,
            data_parallel=data_parallel,
        )