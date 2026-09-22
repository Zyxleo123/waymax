from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any
from flax import nnx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_diffusion.utils.infer import (
    load_model_for_inference,
)
from train_diffusion.utils.checkpoints import restore_checkpoint
from train_diffusion.utils.utils import coerce_tree_like
from simulation.runner import run

os.environ["PYTHONWARNINGS"] = "ignore"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diffusion_ckpt_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--fm_ckpt_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--vla_ckpt_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--tfrecord_dir",
        type=str,
        default="/zfsauton/scratch/eshau/womd/tf_example/",
    )
    parser.add_argument(
        "--split", type=str, default="training", choices=["training", "validation"]
    )
    parser.add_argument(
        "--lane_graph_dir",
        type=str, 
        default="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
    )
    parser.add_argument(
        "--exp_dir",
        type=str, 
        default="/zfsauton/scratch/mineuih/waymax_rs/exp"
    )
    parser.add_argument("--planner", type=str, default="diffusion",
                         choices=["diffusion", "es", "es_fm", "vla", "log-replay", "temporal-vla", "fm"])
    parser.add_argument("--start_timestep", type=int, default=10)
    parser.add_argument("--mask_goal", action="store_true", default=False)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_num_objects", type=int, default=128)
    parser.add_argument("--num_worlds", type=int, default=5)
    parser.add_argument("--num_scenarios", type=int, default=100000000)
    parser.add_argument("--replan_interval_steps", type=int, default=10)
    parser.add_argument("--instruction_interval_steps", type=int, default=10)
    parser.add_argument("--visualize_mode", type=str, default="none", choices=["all", "success", "failure", "none"])
    parser.add_argument("--save_trajectory", action="store_true", default=False)
    parser.add_argument("--save_instruction", action="store_true", default=False)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--scenario_indices", type=str, default=None)
    parser.add_argument("--dummy_instruction", action="store_true", default=False)
    parser.add_argument("--use_subgoal", action="store_true", default=False)
    parser.add_argument("--population_size", type=int, default=64)
    parser.add_argument("--resample_timesteps", type=int, default=5)
    parser.add_argument("--elite_size", type=int, default=8)
    parser.add_argument("--num_es_iterations", type=int, default=4)
    parser.add_argument("--resample_t_start", type=float, default=0.2,
                        help="FM-ES only: flow time to restart resampling from (0=data, 1=noise).")
    parser.add_argument("--resample_num_steps", type=int, default=20,
                        help="FM-ES only: ODE steps during resample (explicit; do not rely on default_num_steps).")
    parser.add_argument("--fm_num_steps", type=int, default=50,
                        help="FM / FM-ES only: ODE steps for full (noise→data) sampling. "
                        "Independent of model timestep conditioning (default_num_steps=100 → t*99).")
    parser.add_argument("--fm_solver", type=str, default="euler", choices=["euler", "heun"])
    parser.add_argument(
        "--other_agent_forecast",
        type=str,
        default="cv",
        choices=["cv", "gt", "wm"],
        help="ES scorers only: 'cv' = PDM constant-velocity; "
        "'gt' = privileged logged futures; "
        "'wm' = learned world-model forecast.",
    )
    parser.add_argument(
        "--world_model_ckpt",
        type=str,
        default=None,
        help="Checkpoint for --other_agent_forecast wm "
        "(e.g. .../epoch_0500.pt or a directory containing latest.pt).",
    )
    parser.add_argument(
        "--world_model_device",
        type=str,
        default="auto",
        help="Torch device for the world model: auto|cpu|cuda|cuda:N. "
        "'auto' prefers CUDA when available.",
    )
    parser.add_argument(
        "--wm_backend",
        type=str,
        default="production",
        choices=["production", "multifuture", "occupancy", "ego"],
        help="Which world-model package to load for --other_agent_forecast wm. "
        "'production' = model/world_model (det/gaussian). "
        "Others load the matching package under the repo root.",
    )
    parser.add_argument(
        "--wm_planning_mode",
        type=str,
        default=None,
        choices=[
            "top1",
            "expected_independent",
            "worst_case_independent",
            # Legacy aliases (canonicalize to *_independent inside multifuture).
            "expected",
            "worst_case",
        ],
        help="Multifuture WM only: agent-wise collision risk. "
        "top1 | expected_independent | worst_case_independent "
        "(aliases: expected, worst_case). Default: checkpoint config.",
    )
    parser.add_argument(
        "--wm_refresh_steps",
        type=int,
        default=10,
        help="WM absolute-sim-step refresh interval. "
        "50 = Exp1 (0,50,...), 20 = Exp2, 10 = Exp3 / every replan. "
        "Shortfall always re-queries at the current timestep. "
        "Only used by the production ScheduledWorldModelForecaster.",
    )
    parser.add_argument(
        "--num_collision_worlds",
        type=int,
        default=8,
        help="Gaussian WM only: Monte Carlo samples for collision scoring. "
        "1 disables sampling (mean-only). Ignored for deterministic WM / cv / gt.",
    )
    parser.add_argument(
        "--collision_reduce",
        type=str,
        default="mean",
        choices=["any", "mean"],
        help="How to reduce collision across WM sample worlds: "
        "'mean' = Monte Carlo P(safe); 'any' = collide if any sample overlaps.",
    )
    parser.add_argument(
        "--collision_sample_seed",
        type=int,
        default=0,
        help="RNG seed offset for Gaussian collision sampling (plus timestep).",
    )
    parser.add_argument("--save_cache", type=str, default="none", choices=["none", "all", "success", "failure"])
    parser.add_argument("--start_from", type=str, default=None)
    parser.add_argument("--target_scenario_path", type=str, default=None)
    return parser.parse_args()


def _resolve_world_model_device(device_arg: str) -> str:
    value = str(device_arg).strip().lower()
    if value in {"auto", ""}:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return str(device_arg).strip()


def _load_isolated_wm_forecaster(
    pkg_root: Path,
    *,
    checkpoint_path: str,
    device: str,
    planning_mode: str | None = None,
):
    """Load a package-local WorldModelForecaster without clashing with repo ``model``/``data``."""
    import importlib
    import types

    pkg_root = pkg_root.resolve()
    alias = f"_wmpkg_{pkg_root.name.replace('-', '_')}"

    def _register_namespace(fullname: str, path: Path) -> None:
        if fullname in sys.modules:
            return
        mod = types.ModuleType(fullname)
        mod.__path__ = [str(path)]  # type: ignore[attr-defined]
        init_py = path / "__init__.py"
        mod.__file__ = str(init_py if init_py.exists() else path)
        sys.modules[fullname] = mod

    _register_namespace(alias, pkg_root)
    _register_namespace(f"{alias}.data", pkg_root / "data")
    _register_namespace(f"{alias}.model", pkg_root / "model")
    _register_namespace(f"{alias}.model.world_model", pkg_root / "model" / "world_model")
    if (pkg_root / "occupancy").is_dir():
        _register_namespace(f"{alias}.occupancy", pkg_root / "occupancy")

    # Package code uses absolute ``from data...`` / ``from occupancy...``.
    saved_top = {
        key: sys.modules[key]
        for key in ("data", "occupancy")
        if key in sys.modules
    }
    sys.modules["data"] = sys.modules[f"{alias}.data"]
    if f"{alias}.occupancy" in sys.modules:
        sys.modules["occupancy"] = sys.modules[f"{alias}.occupancy"]

    try:
        inference = importlib.import_module(f"{alias}.model.world_model.inference")
        load_kwargs: dict[str, Any] = {"device": device}
        # Only multifuture accepts planning_mode today.
        if planning_mode is not None:
            load_kwargs["planning_mode"] = planning_mode
        try:
            forecaster = inference.WorldModelForecaster.from_checkpoint(
                checkpoint_path,
                **load_kwargs,
            )
        except TypeError:
            load_kwargs.pop("planning_mode", None)
            forecaster = inference.WorldModelForecaster.from_checkpoint(
                checkpoint_path,
                **load_kwargs,
            )
    finally:
        for key in ("data", "occupancy"):
            if key in saved_top:
                sys.modules[key] = saved_top[key]
            elif key in sys.modules and getattr(sys.modules[key], "__name__", "").startswith(alias):
                del sys.modules[key]

    return forecaster


def _load_world_model_forecaster(args: argparse.Namespace):
    if str(args.other_agent_forecast).strip().lower() != "wm":
        return None
    if not args.world_model_ckpt:
        raise ValueError("--world_model_ckpt is required when --other_agent_forecast wm.")

    device = _resolve_world_model_device(args.world_model_device)
    # Torch WM + JAX planner often share one GPU; avoid JAX grabbing all memory.
    if device.startswith("cuda"):
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    backend = str(getattr(args, "wm_backend", "production")).strip().lower()
    repo_root = Path(__file__).resolve().parents[1]
    backend_roots = {
        "production": None,
        "multifuture": repo_root / "multiFuture_GaussianWorldModel",
        "occupancy": repo_root / "occupancy_worldModel",
        "ego": repo_root / "egoConditioned_WM",
    }
    if backend not in backend_roots:
        raise ValueError(f"Unknown --wm_backend {backend!r}")

    if backend == "production":
        from model.world_model.inference import ScheduledWorldModelForecaster

        forecaster = ScheduledWorldModelForecaster.from_checkpoint(
            args.world_model_ckpt,
            device=device,
            refresh_steps=int(args.wm_refresh_steps),
            num_collision_worlds=int(args.num_collision_worlds),
            collision_reduce=str(args.collision_reduce),
            collision_sample_seed=int(args.collision_sample_seed),
        )
    else:
        pkg_root = backend_roots[backend]
        assert pkg_root is not None
        planning_mode = None
        if backend == "multifuture" and getattr(args, "wm_planning_mode", None):
            planning_mode = str(args.wm_planning_mode)
        forecaster = _load_isolated_wm_forecaster(
            pkg_root,
            checkpoint_path=str(args.world_model_ckpt),
            device=device,
            planning_mode=planning_mode,
        )
        if not hasattr(forecaster, "num_collision_worlds"):
            forecaster.num_collision_worlds = 1
        if not hasattr(forecaster, "collision_reduce"):
            forecaster.collision_reduce = "any"

    print(
        f"[wm] backend={backend} loaded {args.world_model_ckpt} on {device}; "
        f"wm_refresh_steps={int(args.wm_refresh_steps)}; "
        f"num_collision_worlds={int(getattr(forecaster, 'num_collision_worlds', 1))}; "
        f"collision_reduce={getattr(forecaster, 'collision_reduce', 'any')}"
    )
    return forecaster


def _fm_default_num_steps_from_metadata(metadata: dict) -> int:
    """Resolve FlowMatching time-embedding scale from checkpoint metadata.

    Never use ``flow_num_steps`` here — that field was an unused pretrain CLI
    leftover and caused the t*49 vs t*99 bug. Missing/null → 100 (trained default).
    """
    raw = metadata.get("default_num_steps", 100)
    if raw is None:
        return 100
    return int(raw)


def _load_fm_model_for_inference(
    checkpoint_path: str,
    *,
    use_ema: bool = True,
    metadata_path: str | None = None,
    seed: int = 0,
    skip_checkpoint_load: bool = False,
):
    from data.types import PreprocessConfig
    from model.diffusion.fm_policy import FlowMatchingPolicy

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    meta_path = Path(metadata_path) if metadata_path is not None else ckpt_path / "metadata.json"
    if not meta_path.exists():
        raise ValueError(f"Metadata file not found: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    preprocess_cfg_data = metadata.get("preprocess_cfg", {})
    if not isinstance(preprocess_cfg_data, dict):
        preprocess_cfg_data = {}
    preprocess_cfg = PreprocessConfig(
        model_dt=float(preprocess_cfg_data.get("model_dt", metadata.get("model_dt", 0.2))),
        world_dt_fallback=float(preprocess_cfg_data.get("world_dt_fallback", 0.1)),
        max_range=float(preprocess_cfg_data.get("max_range", metadata.get("max_range", 30.0))),
        ego_range=float(preprocess_cfg_data.get("ego_range", metadata.get("ego_range", 100.0))),
        max_velocity=float(preprocess_cfg_data.get("max_velocity", metadata.get("max_velocity", 25.0))),
        max_width=float(preprocess_cfg_data.get("max_width", metadata.get("max_width", 10.0))),
        max_tl_points=int(preprocess_cfg_data.get("max_tl_points", metadata.get("max_tl_points", 16))),
        num_map_type_classes=int(preprocess_cfg_data.get("num_map_type_classes", metadata.get("num_map_type_classes", 21))),
        predict_horizon=int(preprocess_cfg_data.get("predict_horizon", metadata.get("predict_horizon", 25))),
        map_unknown_type_index=int(preprocess_cfg_data.get("map_unknown_type_index", 20)),
        max_segments=int(preprocess_cfg_data.get("max_segments", metadata.get("max_num_segments", 128))),
        max_points_per_segment=int(preprocess_cfg_data.get("max_points_per_segment", metadata.get("max_points_per_segment", 128))),
        num_object_types=int(preprocess_cfg_data.get("num_object_types", 8)),
        inst_dim=int(preprocess_cfg_data.get("inst_dim", metadata.get("inst_dim", 768))),
    )

    model = FlowMatchingPolicy(
        target_dim=int(metadata.get("target_dim", 5)),
        hidden_dim=int(metadata.get("hidden_dim", 512)),
        cond_dim=int(metadata.get("cond_dim", 256)),
        map_attr_dim=int(metadata.get("map_attr_dim", 25)),
        tl_attr_dim=int(metadata.get("tl_attr_dim", 11)),
        inst_attr_dim=int(metadata.get("inst_dim", 768)),
        ego_dim=int(metadata.get("ego_dim", 5)),
        other_dim=int(metadata.get("other_dim", 15)),
        predict_horizon=int(metadata.get("predict_horizon", preprocess_cfg.predict_horizon)),
        subgoal_conditioned=bool(metadata.get("subgoal_conditioned", False)),
        # Prefer metadata default_num_steps when present. Never use flow_num_steps
        # (legacy unused training CLI). Existing ckpt has default_num_steps=null
        # and must still resolve to 100 → model_time = t * 99.
        default_num_steps=_fm_default_num_steps_from_metadata(metadata),
        rngs=nnx.Rngs(int(seed)),
    )

    graphdef, template_params_state, template_nonparam_state = nnx.split(model, nnx.Param, ...)
    params_state = template_params_state
    nonparam_state = template_nonparam_state

    if not skip_checkpoint_load:
        restored = restore_checkpoint(checkpoint_path)
        if "params_state" not in restored or "ema_params" not in restored:
            raise KeyError("Checkpoint is missing 'params_state' or 'ema_params'.")

        raw_params_state = restored["ema_params"] if use_ema else restored["params_state"]
        params_state = coerce_tree_like(template_params_state, raw_params_state)
        nonparam_state = coerce_tree_like(
            template_nonparam_state,
            restored.get("nonparam_state", template_nonparam_state),
        )

    policy = nnx.merge(
        graphdef,
        params_state,
        nonparam_state,
    )
    return policy, preprocess_cfg


def main(args: argparse.Namespace) -> None:
    args.tfrecord_dir = os.path.join(args.tfrecord_dir, args.split)
    
    exp_name = f"simulation_{args.planner}_{args.split}"
    if args.tag is not None:
        exp_name += f"_{args.tag}"
    exp_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # If `start_from` is provided, continue writing results into that directory
    # instead of creating a new experiment directory. If it's a relative path,
    # interpret it relative to `args.exp_dir`.
    if args.start_from:
        start_dir = os.path.expanduser(args.start_from)
        if not os.path.isabs(start_dir):
            start_dir = os.path.join(args.exp_dir, start_dir)
        args.output_dir = start_dir
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        args.output_dir = os.path.join(args.exp_dir, exp_name)
        os.makedirs(args.output_dir, exist_ok=True)
    
    if args.planner == "diffusion":
        inference = load_model_for_inference(
            checkpoint_path=args.diffusion_ckpt_path,
            use_ema=args.use_ema,
            metadata_path=None,
            seed=args.seed,
            skip_checkpoint_load=False,
        )
        policy = nnx.merge(
            inference.graphdef,
            inference.params_state,
            inference.nonparam_state,
        )
        from planner.diffusion_planner import DiffusionPlanner
        planner = DiffusionPlanner(
            policy=policy,
            preprocess_cfg=inference.preprocess_cfg,
            num_worlds=args.num_worlds,
        )
    elif args.planner == "es":
        inference = load_model_for_inference(
            checkpoint_path=args.diffusion_ckpt_path,
            use_ema=args.use_ema,
            metadata_path=None,
            seed=args.seed,
            skip_checkpoint_load=False,
        )
        policy = nnx.merge(
            inference.graphdef,
            inference.params_state,
            inference.nonparam_state,
        )
        from planner.diffusion_es_planner import DiffusionESPlanner
        planner = DiffusionESPlanner(
            policy=policy,
            preprocess_cfg=inference.preprocess_cfg,
            population_size=args.population_size,
            resample_timesteps=args.resample_timesteps,
            elite_size=args.elite_size,
            num_iterations=args.num_es_iterations,
            num_worlds=args.num_worlds,
            metrics=["collision", "offroad"],
            other_agent_forecast=args.other_agent_forecast,
            world_model_forecaster=_load_world_model_forecaster(args),
        )
    elif args.planner == "fm":
        fm_ckpt_path = args.fm_ckpt_path or args.diffusion_ckpt_path
        if fm_ckpt_path is None:
            raise ValueError("--fm_ckpt_path is required when --planner fm is selected.")

        policy, preprocess_cfg = _load_fm_model_for_inference(
            checkpoint_path=fm_ckpt_path,
            use_ema=args.use_ema,
            metadata_path=None,
            seed=args.seed,
            skip_checkpoint_load=False,
        )

        from planner.fm_planner import FMPlanner

        planner = FMPlanner(
            policy=policy,
            preprocess_cfg=preprocess_cfg,
            population_size=args.population_size,
            num_worlds=args.num_worlds,
            # ODE steps only; conditioning stays default_num_steps=100 → t*99.
            num_steps=int(args.fm_num_steps),
            solver=args.fm_solver,
        )
    elif args.planner == "es_fm":
        fm_ckpt_path = args.fm_ckpt_path or args.diffusion_ckpt_path
        if fm_ckpt_path is None:
            raise ValueError("--fm_ckpt_path is required when --planner es_fm is selected.")

        policy, preprocess_cfg = _load_fm_model_for_inference(
            checkpoint_path=fm_ckpt_path,
            use_ema=args.use_ema,
            metadata_path=None,
            seed=args.seed,
            skip_checkpoint_load=False,
        )

        from planner.fm_es_planner import FMESPlanner

        planner = FMESPlanner(
            policy=policy,
            preprocess_cfg=preprocess_cfg,
            population_size=args.population_size,
            elite_size=args.elite_size,
            num_iterations=args.num_es_iterations,
            num_worlds=args.num_worlds,
            metrics=["collision", "offroad"],
            resample_t_start=args.resample_t_start,
            resample_num_steps=args.resample_num_steps,
            # ODE steps only; conditioning stays default_num_steps=100 → t*99.
            num_steps=int(args.fm_num_steps),
            solver=args.fm_solver,
            other_agent_forecast=args.other_agent_forecast,
            world_model_forecaster=_load_world_model_forecaster(args),
        )
    elif args.planner == "vla":
        inference = load_model_for_inference(
            checkpoint_path=args.diffusion_ckpt_path,
            use_ema=args.use_ema,
            metadata_path=None,
            seed=args.seed,
            skip_checkpoint_load=False,
        )
        policy = nnx.merge(
            inference.graphdef,
            inference.params_state,
            inference.nonparam_state,
        )
        # from model.vla.gemma_vla import VecSceneGemmaVLA
        from model.vla.gemma_vla_v2 import VecSceneGemmaVLA
        from model.vla.embedding_gemma_encoder import EmbeddingGemmaEncoder
        from planner.vla_planner import VLAPlanner

        gemma_vla = VecSceneGemmaVLA.from_pretrained(args.vla_ckpt_path)
        instruction_encoder = EmbeddingGemmaEncoder(
            model_name="google/embeddinggemma-300m",
            device="cuda:1",
            max_length=128,
            target_dim=786,
        )
        planner = VLAPlanner(
            policy=policy,
            gemma_vla=gemma_vla,
            instruction_encoder=instruction_encoder,
            preprocess_cfg=inference.preprocess_cfg,
            num_worlds=args.num_worlds,
            dummy_instruction=args.dummy_instruction,
        )
    elif args.planner == "temporal-vla":
        inference = load_model_for_inference(
            checkpoint_path=args.diffusion_ckpt_path,
            use_ema=args.use_ema,
            metadata_path=None,
            seed=args.seed,
            skip_checkpoint_load=False,
        )
        policy = nnx.merge(
            inference.graphdef,
            inference.params_state,
            inference.nonparam_state,
        )
        from model.vla.temporal_vla import TemporalGemmaVLA
        from model.vla.embedding_gemma_encoder import EmbeddingGemmaEncoder
        from planner.temporal_vla_planner import TemporalVLAPlanner

        temporal_vla = TemporalGemmaVLA.from_pretrained(args.vla_ckpt_path)
        instruction_encoder = EmbeddingGemmaEncoder(
            model_name="google/embeddinggemma-300m",
            device="cuda:1",
            max_length=128,
            target_dim=786,
        )
        planner = TemporalVLAPlanner(
            policy=policy,
            temporal_vla=temporal_vla,
            instruction_encoder=instruction_encoder,
            preprocess_cfg=inference.preprocess_cfg,
            num_worlds=args.num_worlds,
            dummy_instruction=args.dummy_instruction,
            population_size=args.population_size,
            resample_timesteps=args.resample_timesteps,
            elite_size=args.elite_size,
            num_iterations=args.num_es_iterations,
        )
    elif args.planner == "log-replay":
        planner = None
    run(args, planner, log_replay=(args.planner == "log-replay"))


if __name__ == "__main__":
    args = _parse_args()
    main(args)
