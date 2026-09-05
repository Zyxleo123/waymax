from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from data.types import PreprocessConfig
from model.diffusion_torch.diffusion_policy import DiffusionPolicy


@dataclass
class InferenceBundleTorch:
    model: DiffusionPolicy
    preprocess_cfg: PreprocessConfig
    model_config: dict[str, Any]
    device: torch.device
    checkpoint_path: str


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _to_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def _resolve_checkpoint_file(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    if not path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    if path.is_file():
        return path

    candidates = sorted(path.glob("*.pt"))
    if not candidates:
        candidates = sorted(path.glob("**/*.pt"))
    if not candidates:
        raise ValueError(f"No .pt checkpoint file found under: {checkpoint_path}")

    # Prefer latest modified checkpoint when a directory is given.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_optional_metadata(checkpoint_file: Path, metadata_path: str | None) -> dict[str, Any]:
    if metadata_path is not None:
        meta_path = Path(metadata_path)
    else:
        direct = checkpoint_file.parent / "metadata.json"
        if direct.exists():
            meta_path = direct
        else:
            parent = checkpoint_file.parent.parent / "metadata.json"
            meta_path = parent if parent.exists() else None

    if meta_path is None or not meta_path.exists():
        return {}

    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_runtime_config(payload: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config", {})
    if not isinstance(config, dict):
        config = {}

    merged = dict(metadata)
    merged.update(config)
    return merged


def _build_preprocess_cfg(config: dict[str, Any]) -> PreprocessConfig:
    max_segments = config.get("max_num_segments", config.get("max_segments", 128))
    max_points_per_segment = config.get("max_points_per_segment", 128)

    return PreprocessConfig(
        model_dt=_to_float(config.get("model_dt"), 0.2),
        world_dt_fallback=0.1,
        max_range=_to_float(config.get("max_range"), 30.0),
        ego_range=_to_float(config.get("ego_range"), 100.0),
        max_velocity=_to_float(config.get("max_velocity"), 25.0),
        max_width=_to_float(config.get("max_width"), 10.0),
        max_tl_points=_to_int(config.get("max_tl_points"), 16),
        num_map_type_classes=_to_int(config.get("num_map_type_classes"), 21),
        predict_horizon=_to_int(config.get("predict_horizon"), 25),
        map_unknown_type_index=_to_int(config.get("map_unknown_type_index"), 20),
        max_segments=_to_int(max_segments, 128),
        max_points_per_segment=_to_int(max_points_per_segment, 128),
        num_object_types=_to_int(config.get("num_object_types"), 8),
        inst_dim=_to_int(config.get("inst_dim"), 768),
    )


def _build_model(config: dict[str, Any]) -> DiffusionPolicy:
    required = (
        "target_dim",
        "hidden_dim",
        "cond_dim",
        "map_attr_dim",
        "tl_attr_dim",
        "predict_horizon",
        "predict_type",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(
            "Missing required model config keys for torch diffusion inference: "
            f"{missing}. Expected these keys in checkpoint['config'] or metadata.json"
        )

    return DiffusionPolicy(
        target_dim=_to_int(config["target_dim"], 5),
        hidden_dim=_to_int(config["hidden_dim"], 512),
        cond_dim=_to_int(config["cond_dim"], 256),
        map_attr_dim=_to_int(config["map_attr_dim"], 25),
        tl_attr_dim=_to_int(config["tl_attr_dim"], 11),
        ego_dim=_to_int(config.get("ego_dim"), 5),
        other_dim=_to_int(config.get("other_dim"), 15),
        predict_type=str(config["predict_type"]),
        predict_horizon=_to_int(config["predict_horizon"], 25),
    )


def load_model_for_inference_torch(
    checkpoint_path: str,
    *,
    use_ema: bool = True,
    metadata_path: str | None = None,
    map_location: str | None = None,
) -> InferenceBundleTorch:
    checkpoint_file = _resolve_checkpoint_file(checkpoint_path)

    if map_location is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(map_location)

    payload = torch.load(checkpoint_file, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint payload format: {checkpoint_file}")

    metadata = _load_optional_metadata(checkpoint_file, metadata_path)
    runtime_config = _extract_runtime_config(payload, metadata)

    preprocess_cfg = _build_preprocess_cfg(runtime_config)
    model = _build_model(runtime_config)

    weight_key = "ema_model" if use_ema and "ema_model" in payload else "model"
    if weight_key not in payload:
        raise KeyError(
            f"Checkpoint {checkpoint_file} does not contain '{weight_key}' weights. "
            f"Available keys: {sorted(payload.keys())}"
        )

    model.load_state_dict(payload[weight_key], strict=True)
    model.to(device)
    model.eval()

    model_cfg = {
        "target_dim": model.target_dim,
        "predict_horizon": model.predict_horizon,
        "predict_type": runtime_config.get("predict_type", "v"),
    }

    return InferenceBundleTorch(
        model=model,
        preprocess_cfg=preprocess_cfg,
        model_config=model_cfg,
        device=device,
        checkpoint_path=str(checkpoint_file),
    )
