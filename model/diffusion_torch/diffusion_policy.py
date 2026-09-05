from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import torch
from torch import nn

from .diffusion import GaussianDiffusion, UNet1DConditioned
from .modules.mlp import MLP
from .modules.scene_tokenizer import SceneTokenizer


@dataclass
class PolicyFeatures:
    ego_state: torch.Tensor
    goal_xy: torch.Tensor
    remaining_timesteps: torch.Tensor
    other_states: torch.Tensor
    other_valid: torch.Tensor
    map_features: torch.Tensor
    map_valid: torch.Tensor
    traffic_light_features: torch.Tensor
    traffic_light_valid: torch.Tensor
    ego_trajectory: Optional[torch.Tensor] = None

    @staticmethod
    def from_mapping(features: Mapping[str, torch.Tensor]) -> "PolicyFeatures":
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
        missing = [k for k in required if k not in features]
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


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        target_dim: int,
        hidden_dim: int,
        cond_dim: int,
        map_attr_dim: int,
        tl_attr_dim: int,
        ego_dim: int = 5,
        other_dim: int = 15,
        predict_type: str = "v",
        predict_horizon: int = 25,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del kwargs
        self.target_dim = target_dim
        self.predict_horizon = predict_horizon

        self.scene_tokenizer = SceneTokenizer(
            ego_dim=ego_dim,
            other_dim=other_dim,
            map_attr_dim=map_attr_dim,
            tl_attr_dim=tl_attr_dim,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            token_num=32,
        )
        denoise_fn = UNet1DConditioned(
            in_ch=target_dim,
            cond_dim=cond_dim,
            cond2_dim=cond_dim,
            time_emb_dim=cond_dim,
            horizon=predict_horizon,
            hidden_dim=hidden_dim,
        )
        self.diffusion = GaussianDiffusion(
            denoise_fn=denoise_fn,
            timesteps=100,
            predict_type=predict_type,
        )

    @staticmethod
    def _as_features(features: PolicyFeatures | Mapping[str, torch.Tensor]) -> PolicyFeatures:
        if isinstance(features, PolicyFeatures):
            return features
        return PolicyFeatures.from_mapping(features)

    def compute_condition(self, input_features: PolicyFeatures | Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self._as_features(input_features)
        return self.scene_tokenizer._tokenize_scene_features(features.__dict__, deterministic=True)
        
    def input_features_projection(self, input_features: PolicyFeatures | Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.compute_condition(input_features)

    def sample_from_condition(
        self,
        cond: torch.Tensor,
        eta: float = 1.0,
        sampling_temp: float = 1.0,
        temp_mode: str = "uniform",
    ) -> torch.Tensor:
        batch_size = cond.shape[0]
        pred = self.diffusion.sample(
            shape=(batch_size, self.target_dim, self.predict_horizon),
            cond1=cond,
            eta=eta,
            sampling_temp=sampling_temp,
            temp_mode=temp_mode,
        )
        return pred.transpose(1, 2)

    def loss_from_condition(
        self,
        target_btd: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        return self.diffusion.loss(target_btd.transpose(1, 2), cond1=cond)

    def resample_from_condition(
        self,
        cond: torch.Tensor,
        proposals_btd: torch.Tensor,
        n_timesteps: int,
        noise_scale: float = 2.0,
        eta: float = 1.0,
        sampling_temp: float = 1.0,
        temp_mode: str = "uniform",
    ) -> torch.Tensor:
        x = self.diffusion.resample(
            proposals_btd.transpose(1, 2),
            cond1=cond,
            n_timesteps=n_timesteps,
            noise_scale=noise_scale,
            eta=eta,
            sampling_temp=sampling_temp,
            temp_mode=temp_mode,
        )
        return x.transpose(1, 2)

    def forward(self, input_features: PolicyFeatures | Mapping[str, torch.Tensor], eta: float = 1.0, sampling_temp: float = 1.0, temp_mode: str = "uniform") -> torch.Tensor:
        return self.sample(input_features, eta=eta, sampling_temp=sampling_temp, temp_mode=temp_mode)

    def sample(
        self,
        input_features: PolicyFeatures | Mapping[str, torch.Tensor],
        eta: float = 1.0,
        sampling_temp: float = 1.0,
        temp_mode: str = "uniform",
    ) -> torch.Tensor:
        features = self._as_features(input_features)
        cond = self.compute_condition(features)
        return self.sample_from_condition(cond, eta=eta, sampling_temp=sampling_temp, temp_mode=temp_mode)

    def loss(
        self, 
        input_features: PolicyFeatures | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        features = self._as_features(input_features)
        if features.ego_trajectory is None:
            raise ValueError("ego_trajectory is required for loss()")
        target = features.ego_trajectory[:, : self.predict_horizon, :]
        cond = self.compute_condition(features)
        return self.loss_from_condition(target, cond)

    def resample(
        self,
        input_features: PolicyFeatures | Mapping[str, torch.Tensor],
        proposals: torch.Tensor,
        n_timesteps: int,
        eta: float = 1.0,
        sampling_temp: float = 1.0,
        temp_mode: str = "uniform",
    ) -> torch.Tensor:
        del data_parallel
        features = self._as_features(input_features)
        cond = self.compute_condition(features)
        return self.resample_from_condition(cond, proposals, n_timesteps, eta=eta, sampling_temp=sampling_temp, temp_mode=temp_mode)