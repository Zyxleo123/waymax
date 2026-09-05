from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
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
                         choices=["diffusion", "es", "vla", "log-replay", "temporal-vla", "fm"])
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
    parser.add_argument("--save_cache", type=str, default="none", choices=["none", "all", "success", "failure"])
    parser.add_argument("--start_from", type=str, default=None)
    parser.add_argument("--target_scenario_path", type=str, default=None)
    return parser.parse_args()


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
        default_num_steps=int(metadata.get("flow_num_steps", 100)),
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
            num_steps=None,
            solver="euler",
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
