from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class WorldModelConfig:
    """Gaussian world model with optional ego-candidate conditioning.

    Training limitation
    -------------------
    When ego_conditioning=True we condition on the *logged* ego future, i.e. the
    single realized ego action in the WOMD log. There is no counterfactual
    other-agent GT for alternate ego plans. Optional tiny perturbations of the
    logged ego future are input-only; other-agent targets stay the logged ones.
    """

    history_steps: int = 10
    history_stride: int = 2
    future_steps: int = 25
    future_stride: int = 2
    world_dt: float = 0.1

    max_agents: int = 128
    max_range: float = 50.0
    max_velocity: float = 25.0
    max_width: float = 10.0
    num_object_types: int = 8
    max_map_segments: int = 64
    max_points_per_segment: int = 32
    max_tl_points: int = 16
    num_map_type_classes: int = 21
    map_unknown_type_index: int = 20

    hidden_dim: int = 256
    num_heads: int = 4
    num_scene_layers: int = 3
    num_future_layers: int = 2
    dropout: float = 0.1

    sigma_eps: float = 1e-3
    sigma_raw_bias: float = -4.0

    yaw_loss_weight: float = 0.5
    collision_focus_radius_m: float = 20.0
    collision_focus_weight: float = 2.0

    # Candidate-conditioned decoding. Scene encoder is never rerun per candidate.
    ego_conditioning: bool = True
    ego_future_dim: int = 6
    ego_aug_xy_m: float = 0.3
    ego_aug_yaw_deg: float = 2.0

    def __post_init__(self) -> None:
        self.ego_conditioning = bool(self.ego_conditioning)

    @property
    def agent_feature_dim(self) -> int:
        return 8 + int(self.num_object_types)

    @property
    def map_attr_dim(self) -> int:
        return 4 + int(self.num_map_type_classes)

    @property
    def tl_attr_dim(self) -> int:
        return 11

    @property
    def future_dt(self) -> float:
        return float(self.future_stride) * float(self.world_dt)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WorldModelConfig":
        fields = cls.__dataclass_fields__
        kwargs = {k: value[k] for k in fields if k in value}
        return cls(**kwargs)
