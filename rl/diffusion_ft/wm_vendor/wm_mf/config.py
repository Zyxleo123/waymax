from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


# Independent-agent modes (mode k is NOT a joint scene future).
# Legacy aliases ``expected`` / ``worst_case`` canonicalize to the *_independent forms.
PLANNING_MODES = (
    "top1",
    "expected_independent",
    "worst_case_independent",
)
_PLANNING_MODE_ALIASES = {
    "expected": "expected_independent",
    "worst_case": "worst_case_independent",
}


def canonicalize_planning_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    return _PLANNING_MODE_ALIASES.get(m, m)


@dataclass
class WorldModelConfig:
    """Multimodal latent world model: K Gaussian futures per agent.

    Encoder / scene transformer / shared future dynamics match the unimodal
    Gaussian world model. Only the decoder is mode-conditioned.
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

    # K future modes, decoded in one pass from shared latents.
    num_modes: int = 6
    sigma_eps: float = 1e-3
    sigma_raw_bias: float = -4.0

    yaw_loss_weight: float = 0.5
    mode_cls_weight: float = 1.0
    collision_focus_radius_m: float = 20.0
    collision_focus_weight: float = 2.0

    # Planner-facing collision risk: top1 | expected_independent | worst_case_independent
    # (aliases: expected, worst_case). Modes are agent-wise, not scene-joint.
    planning_mode: str = "top1"

    def __post_init__(self) -> None:
        if int(self.num_modes) < 1:
            raise ValueError(f"num_modes must be >= 1, got {self.num_modes}")
        self.num_modes = int(self.num_modes)
        mode = canonicalize_planning_mode(self.planning_mode)
        if mode not in PLANNING_MODES:
            raise ValueError(
                f"planning_mode must be one of {PLANNING_MODES} "
                f"(aliases: expected, worst_case), got {self.planning_mode!r}"
            )
        self.planning_mode = mode

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
