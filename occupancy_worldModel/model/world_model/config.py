from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class WorldModelConfig:
    """Configuration for the latent multi-agent dynamics world model.

    The model consumes past/current scene state and predicts future motion of
    surrounding dynamic agents. The decoded output is intentionally compatible
    with the existing FM-ES scorer interface: explicit future boxes are produced
    and the existing collision/off-road/goal functions remain unchanged.
    """

    # Temporal sampling from WOMD/Waymax logs. WOMD is normally 10 Hz.
    history_steps: int = 10
    history_stride: int = 2
    future_steps: int = 25
    future_stride: int = 2
    world_dt: float = 0.1

    # Scene limits / normalization.
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

    # Network.
    hidden_dim: int = 256
    num_heads: int = 4
    num_scene_layers: int = 3
    num_future_layers: int = 2
    dropout: float = 0.1

    # Diagonal Gaussian XY head: sigma = softplus(raw) + sigma_eps.
    sigma_eps: float = 1e-3
    sigma_raw_bias: float = -4.0

    # Ego-centric BEV occupancy rasterized from the trajectory head.
    # x is ego-forward, y is ego-left, same frame as current_local_xy.
    occ_x_min: float = -10.0
    occ_x_max: float = 40.0
    occ_y_min: float = -20.0
    occ_y_max: float = 20.0
    occ_cell_size: float = 0.5
    occ_tau: float = 0.25
    occ_sigma_scale: float = 1.0
    occ_window_m: float = 8.0
    occ_loss_weight: float = 1.0
    occ_focal_gamma: float = 2.0
    occ_focal_alpha: float = 0.25
    occ_boundary_boost: float = 1.0
    occ_threshold: float = 0.5

    # Loss. Nearby moving objects matter most for the CV -> GT overlap gap.
    yaw_loss_weight: float = 0.5
    collision_focus_radius_m: float = 20.0
    collision_focus_weight: float = 2.0

    @property
    def agent_feature_dim(self) -> int:
        # local xy (2), local velocity (2), sin/cos(relative yaw) (2),
        # length/width (2), object-type one hot (num_object_types)
        return 8 + int(self.num_object_types)

    @property
    def map_attr_dim(self) -> int:
        # Existing project map features: xy(2) + direction(2) + type one-hot.
        return 4 + int(self.num_map_type_classes)

    @property
    def tl_attr_dim(self) -> int:
        # Existing project TL features: xy(2) + state one-hot(9).
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
