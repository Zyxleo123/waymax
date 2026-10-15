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
DECODER_TYPES = ("per_step", "direct")
WTA_STRATEGIES = ("hard", "ewta")
STATIC_AGENT_MODES = ("compete", "all_modes", "fixed_mode")

_PLANNING_MODE_ALIASES = {
    "expected": "expected_independent",
    "worst_case": "worst_case_independent",
}


def canonicalize_planning_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    return _PLANNING_MODE_ALIASES.get(m, m)


def parse_ewta_schedule(schedule: str) -> list[tuple[int, int]]:
    """Parse ``"k:start_epoch,..."`` (e.g. ``"6:0,4:50,2:100,1:150"``) sorted by epoch."""
    stages = []
    for item in str(schedule).split(","):
        item = item.strip()
        if not item:
            continue
        k_str, sep, epoch_str = item.partition(":")
        if not sep:
            raise ValueError(f"ewta_schedule entries must be 'k:start_epoch', got {item!r}")
        k, start = int(k_str), int(epoch_str)
        if k < 1 or start < 0:
            raise ValueError(f"ewta_schedule needs k >= 1 and start_epoch >= 0, got {item!r}")
        stages.append((start, k))
    if not stages:
        raise ValueError("ewta_schedule is empty")
    stages.sort()
    if stages[0][0] != 0:
        raise ValueError(f"ewta_schedule must start at epoch 0, got {schedule!r}")
    return [(k, start) for start, k in stages]


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
    # Decode mu / sigma in each agent's current heading frame, then rotate mu
    # back to the ego frame. False keeps legacy (ego-frame) checkpoints loadable.
    agent_frame_decoding: bool = False
    # "per_step": future transformer over T, modes added per step (legacy).
    # "direct": one query per (agent, mode) -> whole [T, 6] trajectory in one MLP.
    decoder_type: str = "per_step"
    # Init norm of each mode embedding row. None -> sqrt(hidden_dim), i.e. the
    # norm of the LayerNorm'd latent it is added to; 1.0 reproduces legacy init,
    # where modes were a ~6% perturbation and decoded near-identical trajectories.
    mode_embed_scale: float | None = None

    yaw_loss_weight: float = 0.5
    mode_cls_weight: float = 1.0
    collision_focus_radius_m: float = 20.0
    collision_focus_weight: float = 2.0

    # Training-only mode assignment (inference is unaffected).
    # "hard": only the min-ADE mode is regressed. "ewta": evolving WTA, the
    # top-k closest modes are regressed with k from ewta_schedule. The mode
    # classifier target is always the single min-ADE mode.
    wta_strategy: str = "hard"
    ewta_schedule: str = "6:0,4:50,2:100,1:150"
    # Agents with current speed < static_speed_thresh (m/s) and max future
    # displacement < static_disp_thresh (m):
    #   compete    -> treated like any other agent (legacy)
    #   all_modes  -> all K modes regressed equally, no classifier loss
    #   fixed_mode -> always assigned to static_mode_index (regression + classifier)
    static_agents: str = "compete"
    static_speed_thresh: float = 0.5
    static_disp_thresh: float = 1.0
    static_mode_index: int = 0

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
        if self.decoder_type not in DECODER_TYPES:
            raise ValueError(f"decoder_type must be one of {DECODER_TYPES}, got {self.decoder_type!r}")
        if self.wta_strategy not in WTA_STRATEGIES:
            raise ValueError(f"wta_strategy must be one of {WTA_STRATEGIES}, got {self.wta_strategy!r}")
        parse_ewta_schedule(self.ewta_schedule)
        if self.static_agents not in STATIC_AGENT_MODES:
            raise ValueError(f"static_agents must be one of {STATIC_AGENT_MODES}, got {self.static_agents!r}")
        if not 0 <= int(self.static_mode_index) < self.num_modes:
            raise ValueError(
                f"static_mode_index must be in [0, {self.num_modes}), got {self.static_mode_index}"
            )

    def ewta_k(self, epoch: int) -> int:
        """Number of modes regressed per agent at ``epoch`` (1 for hard WTA)."""
        if self.wta_strategy != "ewta":
            return 1
        k = 1
        for stage_k, start in parse_ewta_schedule(self.ewta_schedule):
            if int(epoch) >= start:
                k = stage_k
        return max(1, min(int(k), self.num_modes))

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
