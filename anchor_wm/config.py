from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# Waymax ObjectTypeIds: 1 vehicle, 2 pedestrian, 3 cyclist. Anchor sets are per group;
# the SDC and every other type use the vehicle set.
DYNAMIC_TYPES = (1, 2, 3)
ANCHOR_GROUPS = ("vehicle", "pedestrian", "cyclist")
# Waymax MapElementIds ROAD_EDGE_BOUNDARY / ROAD_EDGE_MEDIAN.
ROAD_EDGE_TYPES = (15, 16)
SUB_SCORES = ("overlap", "offroad", "progress", "comfort")


@dataclass
class AnchorWMConfig:
    """Encoder / data fields match `multiFuture_GaussianWorldModel` WorldModelConfig, so the
    MF-Gaussian encoder weights load unchanged (`model.warm_start_encoder`)."""

    # ---- data / encoder (identical to MF-Gaussian) ----
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
    dropout: float = 0.1

    # ---- anchor decoder ----
    num_anchors: int = 64            # per anchor group; must match the anchor file
    num_out_modes: int = 6           # planner interface [N, 6, T]
    nms_dist_m: float = 1.5          # endpoint NMS when picking the planner modes
    sigma_eps: float = 1e-3
    sigma_raw_bias: float = -4.0
    yaw_loss_weight: float = 0.5
    cls_weight: float = 1.0
    cls_soft_temp_m: float = 1.0     # soft anchor-classification target exp(-ade / temp); 0 = one-hot
    collision_focus_radius_m: float = 20.0
    collision_focus_weight: float = 2.0
    # Per-agent forecasting weight: "radius" = 1 + w * [|x_now| <= radius] (A1 / A2); "kernel" =
    # 1 + w * exp(-d / tau), d = closest time-aligned approach of the logged future to the logged ego
    # future or a constant-velocity ego rollout (0 / 0.5 / 1 / 1.5 x speed), i.e. likely interactions.
    interaction_weight: str = "radius"
    interaction_tau_m: float = 5.0
    ego_loss_weight: float = 1.0     # the SDC is decoded too (its anchors are plan proposals)

    # ---- label geometry (online, log replay) ----
    num_label_edges: int = 2048      # road-edge points kept for offroad labels
    edge_box: tuple = (-30.0, 140.0, 60.0)   # behind, ahead, half-width (m) in the ego frame
    num_edge_tokens: int = 256       # scorer road-edge tokens (strided subset of the label edges)
    edge_token_box: tuple = (-25.0, 100.0, 40.0)

    # ---- scorer ----
    scorer_hidden: int = 128
    scorer_heads: int = 4
    scorer_agents: int = 16
    scorer_modes: int = 8
    scorer_pos_scale: float = 20.0
    scorer_vel_scale: float = 5.0
    # Fine-tuning additions (off for the A1 / A2 checkpoints; new modules start at zero output, so
    # a warm-started model scores exactly as before until they train).
    scorer_goal_features: bool = False    # per-step candidate->goal geometry + endpoint goal features
    scorer_route_features: bool = False   # per-step distance / heading error to the nearest lane centerline
    scorer_aux_heads: bool = False        # time-to-violation and goal-distance heads
    scorer_rule_features: bool = False    # exact box gaps to the forecast modes + Waymax-rule edge distance
    # Use the candidates' own headings when the caller has them (recorded ES candidates, closed-loop
    # populations) instead of the travel direction from xy, which is noise for slow / stopping plans.
    scorer_input_heading: bool = False
    # The `scorer_near_agents` nearest selected agents get `scorer_near_modes` modes, the rest
    # `scorer_modes` (0 = every agent gets `scorer_modes`).
    scorer_near_agents: int = 0
    scorer_near_modes: int = 12
    # Steps of the plan left before the episode ends (`plan_horizon`): an input, and every pooled /
    # summary / endpoint quantity of the scorer is taken over the steps within it.
    scorer_time_remaining: bool = False

    def __post_init__(self) -> None:
        self.edge_box = tuple(float(v) for v in self.edge_box)
        self.edge_token_box = tuple(float(v) for v in self.edge_token_box)
        if self.scorer_hidden % self.scorer_heads:
            raise ValueError("scorer_hidden must be divisible by scorer_heads")
        if self.num_out_modes > self.num_anchors or self.scorer_modes > self.num_anchors:
            raise ValueError("num_out_modes and scorer_modes must be <= num_anchors")
        if self.scorer_near_agents > 0 and not self.scorer_modes <= self.scorer_near_modes <= self.num_anchors:
            raise ValueError("scorer_near_modes must be in [scorer_modes, num_anchors]")
        if self.interaction_weight not in ("radius", "kernel"):
            raise ValueError(f"interaction_weight must be radius|kernel, got {self.interaction_weight!r}")

    @property
    def scorer_modes_max(self) -> int:
        return max(int(self.scorer_modes), int(self.scorer_near_modes) if self.scorer_near_agents > 0 else 0)

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
    def from_dict(cls, value: dict[str, Any]) -> "AnchorWMConfig":
        return cls(**{k: value[k] for k in cls.__dataclass_fields__ if k in value})
