"""Discriminator ablations. The Ego Encoder (candidate GRU) is kept in every model; only the flags
below change. Checkpoints without a flag in their model_kwargs have it on.

  A0  original (all flags on)
  A1  no direct goal embedding in the final fusion (goal token still in scene attention)
  A2  no generic map tokens in scene attention
  A3  A1 + A2
  A4  A3 + no historical agent tokens in scene attention (scene attention: lights + goal)

Second round, each one change from A4:
  A5  scene cross-attention replaced by direct conditioning: the goal embedding and the masked mean
      of the traffic-light embeddings enter the final fusion
  A6  no road-edge encoder / attention
  A7  no Gaussian WM encoder / cross-attention
  A8  WM and road-edge attention outputs pooled over time by mean only (A4: mean and max)
"""

from __future__ import annotations

FLAG_KEYS = ("goal_fusion", "map_tokens", "agent_tokens", "scene_attention", "edge_branch", "wm_branch",
             "max_pool")

_ON = dict.fromkeys(FLAG_KEYS, True)
_A4 = {**_ON, "goal_fusion": False, "map_tokens": False, "agent_tokens": False}

ABLATIONS: dict[str, dict[str, bool]] = {
    "A0": dict(_ON),
    "A1": {**_ON, "goal_fusion": False},
    "A2": {**_ON, "map_tokens": False},
    "A3": {**_ON, "goal_fusion": False, "map_tokens": False},
    "A4": dict(_A4),
    "A5": {**_A4, "scene_attention": False},
    "A6": {**_A4, "edge_branch": False},
    "A7": {**_A4, "wm_branch": False},
    "A8": {**_A4, "max_pool": False},
}

DESCRIPTIONS = {
    "A0": "original trained discriminator",
    "A1": "remove direct goal embedding from final fusion",
    "A2": "remove generic map tokens from scene attention",
    "A3": "remove both (A1 + A2)",
    "A4": "A3 + remove historical agent tokens from scene attention",
    "A5": "A4 + goal / traffic-light conditioning instead of scene cross-attention",
    "A6": "A4 + remove road-edge encoder and attention",
    "A7": "A4 + remove Gaussian WM encoder and cross-attention",
    "A8": "A4 + mean-only temporal pooling of WM / road-edge attention",
}


def flags(name: str) -> dict[str, bool]:
    if name not in ABLATIONS:
        raise ValueError(f"unknown ablation {name!r}; choose from {sorted(ABLATIONS)}")
    return dict(ABLATIONS[name])


def ablation_of(model_kwargs: dict) -> str:
    """Ablation name of a checkpoint's model_kwargs (checkpoints from before the flags are A0)."""
    got = {k: bool(model_kwargs.get(k, True)) for k in FLAG_KEYS}
    for name, f in ABLATIONS.items():
        if f == got:
            return name
    return "custom"
