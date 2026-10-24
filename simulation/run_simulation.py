from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any
from flax import nnx
import tensorflow as tf

# TensorFlow only reads the tfrecords here. A GPU build of it (babel's envs) otherwise claims the whole GPU on
# first use and JAX then fails to launch its kernels (CUDA_ERROR_OUT_OF_MEMORY).
tf.config.set_visible_devices([], "GPU")

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
        default="/zfsauton/scratch/mineuih/womd/tf_example/",
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
                         choices=["diffusion", "es", "es_fm", "vla", "log-replay", "temporal-vla", "fm", "fm-fd",
                                  "pdm_closed", "diffusion_planner"])
    parser.add_argument("--fd_vla_ckpt_path", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--num_denoising_steps", type=int, default=None)
    parser.add_argument("--goal_conditioned", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cfg_t_start", type=float, default=1.0)
    parser.add_argument("--cfg_correction_mode", type=str, default="resample", choices=["resample", "sample"])
    parser.add_argument("--fd_torch_device", type=str, default=None)
    parser.add_argument("--start_timestep", type=int, default=10)
    parser.add_argument("--mask_goal", action="store_true", default=False)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_num_objects", type=int, default=128)
    parser.add_argument("--num_worlds", type=int, default=5)
    parser.add_argument("--num_scenarios", type=int, default=100000000)
    parser.add_argument("--num_shards", type=int, default=None,
                        help="Use only the first N tfrecord shards (sorted, from index 0). Default: all.")
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
        choices=["production", "multifuture", "occupancy", "ego", "anchor"],
        help="Which world-model package to load for --other_agent_forecast wm. "
        "'production' = model/world_model (det/gaussian). "
        "'anchor' = anchor_wm (AnchorWM checkpoint; 6 planner modes, independent-agent collision). "
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
        "--offroad_mode",
        type=str,
        default="binary",
        choices=["binary", "margin_tiebreak"],
        help="ES offroad objective. binary = 0/1 road-edge gate; margin_tiebreak = gate x "
        "edge-clearance margin + small graded tiebreak for offroad plans.",
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
        "--wm_mode_temperature",
        type=float,
        default=1.0,
        help="multifuture WM only: mode probabilities used for collision scoring are "
        "softmax(mode_logits / T). 1.0 = as trained; <1 sharpens towards the top mode, >1 flattens. "
        "Mode-probability and ADE / FDE statistics are written to <output_dir>/wm_forecast_stats.json.",
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
    parser.add_argument(
        "--save_es_candidates",
        action="store_true",
        default=False,
        help="Diffusion-ES only: save every evaluated ES candidate per replan to <output_dir>/es_candidates "
        "(es_discriminator.es_candidate_recorder), discriminator training data. Planning is unchanged.",
    )
    parser.add_argument(
        "--label_in_loop",
        action="store_true",
        default=False,
        help="With --save_es_candidates: also write each batch's labels (.labels / .labels_v2 / .labels_waymax.npz) "
        "and AnchorWM bank cache (<output_dir>/anchor_cache) as it finishes (anchor_wm.label_in_loop), in place of "
        "the offline label / cache passes.",
    )
    parser.add_argument(
        "--posthoc_repair",
        action="store_true",
        help="Diffusion-ES only: repair the ES winner before executing it with yixiz's post-hoc optimization "
        "(arm o_am: safety barrier + comfort, constant-velocity neighbours, 2 s window; planner/posthoc_repair.py).",
    )
    parser.add_argument(
        "--discriminator_ckpt",
        type=str,
        default=None,
        help="Diffusion-ES only: replace the handcrafted reward at every ES iteration (elites and final "
        "selection) with this WM-conditioned discriminator (es_discriminator.real_es.closed_loop). "
        "Unset = handcrafted reward, unchanged.",
    )
    parser.add_argument(
        "--discriminator_wm_ckpt",
        type=str,
        default=None,
        help="Frozen multi-future Gaussian WM feeding the discriminator; must be the WM its training "
        "cache was built with. Default: the WM recorded in the discriminator checkpoint, else "
        "--world_model_ckpt (the planner's forecaster is reused when they are the same file).",
    )
    parser.add_argument("--discriminator_device", type=str, default="auto")
    parser.add_argument(
        "--anchor_scorer_ckpt",
        type=str,
        default=None,
        help="Diffusion-ES only: replace the handcrafted reward with the scorer head of this AnchorWM "
        "checkpoint (anchor_wm.closed_loop.AnchorLearnedScorer). Not with --discriminator_ckpt.",
    )
    parser.add_argument(
        "--anchor_scorer_rule_lambda",
        type=float,
        default=0.0,
        help="With --anchor_scorer_ckpt: hybrid reward = learned score + lambda * sum of log handcrafted "
        "factors in --anchor_scorer_rule_terms. 0 = learned score only.",
    )
    parser.add_argument("--anchor_scorer_rule_terms", type=str, default="collision,offroad")
    parser.add_argument(
        "--anchor_ego_seed",
        type=int,
        default=0,
        help="Diffusion-ES with --wm_backend anchor: append the planner WM's K ego modes to every initial "
        "population (anchor_wm.es_debug.AnchorEgoSeeder). 0 = off.",
    )
    parser.add_argument(
        "--es_debug_dir",
        type=str,
        default=None,
        help="Diffusion-ES with --wm_backend anchor: dump every replan (scene, AnchorWM ego modes, ES pools, "
        "chosen plan, seed lineage) to this dir for anchor_wm.viz_ego_seed.",
    )
    parser.add_argument("--start_from", type=str, default=None)
    parser.add_argument("--target_scenario_path", type=str, default=None)
    parser.add_argument(
        "--sim_agents",
        type=str,
        default="log",
        choices=["log", "idm"],
        help="How non-ego agents move during evaluation rollout: 'log' = log replay, "
        "'idm' = IDM along logged routes. Planning still sees logged agents.",
    )
    parser.add_argument(
        "--pdm_cv_velocity_ratios",
        type=str,
        default="1.0",
        help="PDM-Closed and --es_reward pdms: comma-separated constant-velocity ratios for the agent forecast. "
        "'1.0' = upstream tuplan_garage; '0.8,1.0' = the nuplan_rs fork.",
    )
    parser.add_argument(
        "--es_reward",
        type=str,
        default="rule",
        choices=["rule", "pdms"],
        help="Diffusion-ES reward: 'rule' = the handcrafted scorer (--metrics weights); 'pdms' = NAVSIM v1 PDM score "
        "with PDM-Closed's constant-velocity forecast (planner/pdm_score.py).",
    )
    parser.add_argument("--dp_repo", type=str, default=None,
                        help="Diffusion Planner only: path to the Diffusion-Planner checkout (a3a621f).")
    parser.add_argument("--dp_args_file", type=str, default=None,
                        help="Diffusion Planner only: args.json written by their train_predictor.py.")
    parser.add_argument("--dp_ckpt", type=str, default=None,
                        help="Diffusion Planner only: checkpoint (.pth) from their train_predictor.py.")
    parser.add_argument(
        "--other_agent",
        type=str,
        default="log",
        choices=["log", "idm"],
        help="new_model: 'idm' = non-ego agents are IDM-driven during planning as well (closed loop, "
        "simulation/idm_closed_loop.py). Independent of --sim_agents.",
    )
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

    # Package code uses absolute ``from data...`` / ``from occupancy...``. Set the repo's
    # modules (and their submodules) aside while it imports: otherwise an already imported
    # repo ``data.preprocess`` (e.g. by a planner) is reused and the WM gets the repo's map
    # features instead of its own. Afterwards the package's modules leave the top-level names.
    tops = ["data"] + (["occupancy"] if f"{alias}.occupancy" in sys.modules else [])

    def _owned(key: str) -> bool:
        return any(key == top or key.startswith(top + ".") for top in tops)

    saved_top = {key: sys.modules.pop(key) for key in list(sys.modules) if _owned(key)}
    for top in tops:
        sys.modules[top] = sys.modules[f"{alias}.{top}"]

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
        for key in [k for k in sys.modules if _owned(k)]:
            mod = sys.modules.pop(key)
            if "." in key:
                sys.modules.setdefault(f"{alias}.{key}", mod)
        sys.modules.update(saved_top)

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
    mode_temperature = float(getattr(args, "wm_mode_temperature", 1.0))
    repo_root = Path(__file__).resolve().parents[1]
    backend_roots = {
        "production": None,
        "multifuture": repo_root / "multiFuture_GaussianWorldModel",
        "occupancy": repo_root / "occupancy_worldModel",
        "ego": repo_root / "egoConditioned_WM",
        "anchor": None,
    }
    if backend not in backend_roots:
        raise ValueError(f"Unknown --wm_backend {backend!r}")

    if backend == "anchor":
        from anchor_wm.closed_loop import AnchorForecaster

        forecaster = AnchorForecaster.from_checkpoint(
            str(args.world_model_ckpt),
            device=device,
            planning_mode=getattr(args, "wm_planning_mode", None),
            mode_temperature=mode_temperature,
        )
        if getattr(args, "output_dir", None):
            forecaster.stats_path = os.path.join(args.output_dir, "wm_forecast_stats.json")
    elif backend == "production":
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
        if backend == "multifuture":
            forecaster.mode_temperature = mode_temperature
            if getattr(args, "output_dir", None):
                forecaster.stats_path = os.path.join(args.output_dir, "wm_forecast_stats.json")
    if mode_temperature != 1.0 and backend not in ("multifuture", "anchor"):
        raise ValueError("--wm_mode_temperature is only supported with --wm_backend multifuture / anchor")

    print(
        f"[wm] backend={backend} loaded {args.world_model_ckpt} on {device}; "
        f"wm_refresh_steps={int(args.wm_refresh_steps)}; "
        f"num_collision_worlds={int(getattr(forecaster, 'num_collision_worlds', 1))}; "
        f"collision_reduce={getattr(forecaster, 'collision_reduce', 'any')}; "
        f"mode_temperature={getattr(forecaster, 'mode_temperature', 1.0)}"
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


def _is_fm_v2_checkpoint(checkpoint_path: str, metadata_path: str | None = None) -> bool:
    """A flow-matching checkpoint of branch new_model (model/diffusion/fm_policy_v2.py): no instruction /
    subgoal condition, so its metadata has no ``inst_dim`` (same rule as train_diffusion.utils.infer)."""
    meta_path = Path(metadata_path) if metadata_path is not None else Path(checkpoint_path) / "metadata.json"
    with meta_path.open("r", encoding="utf-8") as f:
        return "inst_dim" not in json.load(f)


def _load_fm_v2_model_for_inference(
    checkpoint_path: str,
    *,
    use_ema: bool = True,
    metadata_path: str | None = None,
    seed: int = 0,
    skip_checkpoint_load: bool = False,
    goal_conditioned: bool | None = None,
    num_denoising_steps: int | None = None,
):
    from data.types import PreprocessConfig
    from model.diffusion.fm_policy_v2 import FlowMatchingPolicy

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
        # new_model's checkpoints were trained on its random per-segment point sampling.
        map_point_sampling="random",
    )

    model = FlowMatchingPolicy(
        target_dim=int(metadata.get("target_dim", 5)),
        hidden_dim=int(metadata.get("hidden_dim", 512)),
        cond_dim=int(metadata.get("cond_dim", 256)),
        map_attr_dim=int(metadata.get("map_attr_dim", 25)),
        tl_attr_dim=int(metadata.get("tl_attr_dim", 11)),
        ego_dim=int(metadata.get("ego_dim", 5)),
        other_dim=int(metadata.get("other_dim", 15)),
        predict_horizon=int(metadata.get("predict_horizon", preprocess_cfg.predict_horizon)),
        goal_conditioned=bool(
            metadata.get("goal_conditioned", True)
            if goal_conditioned is None
            else goal_conditioned
        ),
        num_denoising_steps=int(
            num_denoising_steps
            if num_denoising_steps is not None
            else metadata.get(
                "num_denoising_steps",
                metadata.get("flow_num_steps", metadata.get("default_num_steps", 50)),
            )
        ),
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
        ego_dim=int(metadata.get("ego_dim", 5)),
        other_dim=int(metadata.get("other_dim", 15)),
        inst_attr_dim=int(metadata.get("inst_dim", 768)),
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
    if args.sim_agents != "log":
        exp_name += f"_{args.sim_agents}"
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
            offroad_mode=args.offroad_mode,
        )
        if args.posthoc_repair:
            from planner.posthoc_repair import PosthocRepair
            planner.posthoc_repair = PosthocRepair()
    elif args.planner == "fm" and (args.fm_ckpt_path or args.diffusion_ckpt_path) and _is_fm_v2_checkpoint(
        args.fm_ckpt_path or args.diffusion_ckpt_path
    ):
        # new_model's flow-matching policy (scene condition only).
        policy, preprocess_cfg = _load_fm_v2_model_for_inference(
            checkpoint_path=args.fm_ckpt_path or args.diffusion_ckpt_path,
            use_ema=args.use_ema,
            metadata_path=None,
            seed=args.seed,
            skip_checkpoint_load=False,
            goal_conditioned=args.goal_conditioned,
            num_denoising_steps=args.num_denoising_steps,
        )

        from planner.fm_planner_v2 import FMPlanner

        planner = FMPlanner(
            policy=policy,
            preprocess_cfg=preprocess_cfg,
            population_size=args.population_size,
            num_worlds=args.num_worlds,
            num_steps=args.num_denoising_steps,
            solver="euler",
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
            offroad_mode=args.offroad_mode,
        )
    elif args.planner == "fm-fd":
        fm_ckpt_path = args.fm_ckpt_path or args.diffusion_ckpt_path
        if fm_ckpt_path is None:
            raise ValueError("--fm_ckpt_path is required when --planner fm-fd is selected.")
        if args.fd_vla_ckpt_path is None:
            raise ValueError("--fd_vla_ckpt_path is required when --planner fm-fd is selected.")

        policy, preprocess_cfg = _load_fm_v2_model_for_inference(
            checkpoint_path=fm_ckpt_path,
            use_ema=args.use_ema,
            metadata_path=None,
            seed=args.seed,
            skip_checkpoint_load=False,
            goal_conditioned=args.goal_conditioned,
            num_denoising_steps=args.num_denoising_steps,
        )

        from planner.fm_fd_planner import FMFDPlanner

        planner = FMFDPlanner(
            policy=policy,
            fd_vla=args.fd_vla_ckpt_path,
            preprocess_cfg=preprocess_cfg,
            population_size=1,
            num_worlds=args.num_worlds,
            num_steps=args.num_denoising_steps,
            solver="euler",
            guidance_scale=args.guidance_scale,
            cfg_t_start=args.cfg_t_start,
            correction_mode=args.cfg_correction_mode,
            torch_device=args.fd_torch_device,
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
    elif args.planner == "pdm_closed":
        from planner.pdm_closed_planner import PDMClosedPlanner

        planner = PDMClosedPlanner(
            num_worlds=args.num_worlds,
            cv_velocity_ratios=[float(r) for r in args.pdm_cv_velocity_ratios.split(",")],
        )
    elif args.planner == "diffusion_planner":
        from planner.diffusion_planner_baseline import DiffusionPlannerBaseline

        planner = DiffusionPlannerBaseline(
            args_file=args.dp_args_file,
            ckpt_path=args.dp_ckpt,
            num_worlds=args.num_worlds,
            repo_path=args.dp_repo,
            seed=args.seed,
        )
    elif args.planner == "log-replay":
        planner = None
    if getattr(args, "discriminator_ckpt", None):
        _attach_learned_reward(args, planner)
    if getattr(args, "anchor_scorer_ckpt", None):
        _attach_anchor_reward(args, planner)
    if args.es_reward == "pdms":
        _attach_pdms_reward(args, planner)
    if args.anchor_ego_seed > 0 or args.es_debug_dir:
        _attach_es_debug(args, planner)
    if args.label_in_loop and not args.save_es_candidates:
        raise ValueError("--label_in_loop needs --save_es_candidates.")
    run(args, planner, log_replay=(args.planner == "log-replay"))
    if args.es_reward == "pdms":
        print("[pdms] stats:", json.dumps(planner.learned_scorer.stats()))

    forecaster = getattr(planner, "world_model_forecaster", None)
    if forecaster is not None and hasattr(forecaster, "write_forecast_stats"):
        forecaster.write_forecast_stats(os.path.join(args.output_dir, "wm_forecast_stats.json"))
        print("[wm] forecast stats:", json.dumps(forecaster.forecast_stats_summary()))


def _attach_learned_reward(args: argparse.Namespace, planner) -> None:
    import atexit

    if args.planner != "es" or not hasattr(planner, "learned_scorer"):
        raise ValueError("--discriminator_ckpt requires --planner es (Diffusion-ES).")
    if getattr(args, "save_es_candidates", False):
        raise ValueError("--save_es_candidates records handcrafted components; not with --discriminator_ckpt.")
    wm_ckpt = args.discriminator_wm_ckpt
    if not wm_ckpt:
        import torch

        # The WM of the discriminator's training cache (recorded by es_discriminator's trainer).
        ck = torch.load(args.discriminator_ckpt, map_location="cpu", weights_only=False)
        wm_ckpt = ck.get("wm_checkpoint") or args.world_model_ckpt
        del ck
    if not wm_ckpt:
        raise ValueError("--discriminator_ckpt needs --discriminator_wm_ckpt or --world_model_ckpt.")
    from es_discriminator.real_es.closed_loop import LearnedESScorer

    forecaster = getattr(planner, "world_model_forecaster", None)
    scorer = LearnedESScorer(
        args.discriminator_ckpt,
        wm_checkpoint=wm_ckpt,
        planner_forecaster=forecaster,
        planner_wm_checkpoint=args.world_model_ckpt if forecaster is not None else None,
        device=_resolve_world_model_device(args.discriminator_device),
        expected_calls_per_replan=int(args.num_es_iterations) + 1,
        stats_path=os.path.join(args.output_dir, "discriminator_stats.json"),
    )
    planner.learned_scorer = scorer
    atexit.register(scorer.close)
    info = {**scorer.describe(), "other_agent_forecast": args.other_agent_forecast,
            "planner_world_model_ckpt": args.world_model_ckpt, "wm_backend": args.wm_backend,
            "wm_planning_mode": getattr(args, "wm_planning_mode", None),
            "handcrafted_scorer_used": False, "seed": int(args.seed),
            "es": {"population_size": args.population_size, "elite_size": args.elite_size,
                   "num_es_iterations": args.num_es_iterations, "resample_timesteps": args.resample_timesteps},
            "timestamp": datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(args.output_dir, "discriminator_config.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[discriminator] learned reward {args.discriminator_ckpt} ({scorer.scorer.variant}); "
          f"WM {wm_ckpt} source={scorer.wm_source}; forecast={args.other_agent_forecast}")


def _attach_anchor_reward(args: argparse.Namespace, planner) -> None:
    """AnchorWM scorer head as the Diffusion-ES reward (anchor_wm.closed_loop.AnchorLearnedScorer)."""
    import atexit

    if args.planner != "es" or not hasattr(planner, "learned_scorer"):
        raise ValueError("--anchor_scorer_ckpt requires --planner es (Diffusion-ES).")
    if getattr(args, "discriminator_ckpt", None):
        raise ValueError("--anchor_scorer_ckpt and --discriminator_ckpt are mutually exclusive.")
    from anchor_wm.closed_loop import AnchorForecaster, AnchorLearnedScorer

    forecaster = getattr(planner, "world_model_forecaster", None)
    shared = (isinstance(forecaster, AnchorForecaster) and forecaster.checkpoint
              and os.path.realpath(forecaster.checkpoint) == os.path.realpath(args.anchor_scorer_ckpt))
    scorer = AnchorLearnedScorer(
        args.anchor_scorer_ckpt,
        device=_resolve_world_model_device(args.discriminator_device),
        expected_calls_per_replan=int(args.num_es_iterations) + 1,
        stats_path=os.path.join(args.output_dir, "anchor_scorer_stats.json"),
        model=forecaster.model if shared else None,
        rule_scorers=planner.scorers if float(args.anchor_scorer_rule_lambda) > 0 else None,
        rule_lambda=float(args.anchor_scorer_rule_lambda),
        rule_terms=tuple(t.strip() for t in str(args.anchor_scorer_rule_terms).split(",") if t.strip()),
    )
    planner.learned_scorer = scorer
    atexit.register(scorer.close)
    info = {**scorer.describe(), "shares_planner_wm": bool(shared), "other_agent_forecast": args.other_agent_forecast,
            "planner_world_model_ckpt": args.world_model_ckpt, "wm_backend": args.wm_backend,
            "handcrafted_scorer_used": False, "seed": int(args.seed),
            "es": {"population_size": args.population_size, "elite_size": args.elite_size,
                   "num_es_iterations": args.num_es_iterations, "resample_timesteps": args.resample_timesteps},
            "timestamp": datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(args.output_dir, "anchor_scorer_config.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[anchor_scorer] learned reward {args.anchor_scorer_ckpt}; shares planner WM: {bool(shared)}; "
          f"forecast={args.other_agent_forecast}")


def _attach_es_debug(args: argparse.Namespace, planner) -> None:
    """AnchorWM ego-mode seeding of the ES initial population and / or per-replan dumps (anchor_wm.es_debug)."""
    from anchor_wm.closed_loop import AnchorForecaster
    from anchor_wm.es_debug import AnchorEgoSeeder, ESDebugDumper

    forecaster = getattr(planner, "world_model_forecaster", None)
    if args.planner != "es" or not isinstance(forecaster, AnchorForecaster):
        raise ValueError("--anchor_ego_seed / --es_debug_dir need --planner es with --wm_backend anchor.")
    if args.anchor_ego_seed > 0:
        planner.ego_seed_fn = AnchorEgoSeeder(forecaster.model, args.anchor_ego_seed)
    if args.es_debug_dir:
        planner.debug_dump_fn = ESDebugDumper(forecaster.model, args.es_debug_dir, k=max(6, args.anchor_ego_seed))
    print(f"[es_debug] ego seeds {args.anchor_ego_seed} from {forecaster.checkpoint}; dumps -> {args.es_debug_dir}")


def _attach_pdms_reward(args: argparse.Namespace, planner) -> None:
    """NAVSIM v1 PDM score as the Diffusion-ES reward (planner.pdm_score.PDMScoreReward)."""
    if args.planner != "es" or not hasattr(planner, "learned_scorer"):
        raise ValueError("--es_reward pdms requires --planner es (Diffusion-ES).")
    if getattr(args, "discriminator_ckpt", None) or getattr(args, "anchor_scorer_ckpt", None):
        raise ValueError("--es_reward pdms excludes --discriminator_ckpt and --anchor_scorer_ckpt.")
    from planner.pdm_score import PDMScoreReward

    ratios = tuple(float(r) for r in args.pdm_cv_velocity_ratios.split(","))
    scorer = PDMScoreReward(planner.scorers, num_worlds=args.num_worlds, cv_velocity_ratios=ratios)
    planner.learned_scorer = scorer
    info = {**scorer.describe(), "handcrafted_scorer_used": False, "seed": int(args.seed),
            "planner_other_agent_forecast_unused": args.other_agent_forecast,
            "es": {"population_size": args.population_size, "elite_size": args.elite_size,
                   "num_es_iterations": args.num_es_iterations, "resample_timesteps": args.resample_timesteps},
            "timestamp": datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(args.output_dir, "pdms_reward_config.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[pdms] Diffusion-ES reward: NAVSIM v1 PDMS, constant-velocity ratios {ratios}")


if __name__ == "__main__":
    args = _parse_args()
    main(args)
