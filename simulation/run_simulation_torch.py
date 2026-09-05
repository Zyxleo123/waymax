from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import torch

from planner.diffusion_torch_planner import DiffusionTorchPlanner
from simulation.runner import run
from train_diffusion_torch.utils.infer import load_model_for_inference_torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion_ckpt_path", type=str, required=True)
    parser.add_argument("--tfrecord_dir", type=str, default="/zfsauton/scratch/eshau/womd/tf_example/")
    parser.add_argument("--split", type=str, default="training", choices=["training", "validation"])
    parser.add_argument("--exp_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/exp")
    parser.add_argument("--planner", type=str, default="diffusion", choices=["diffusion", "log-replay"])
    parser.add_argument("--start_timestep", type=int, default=10)
    parser.add_argument("--mask_goal", action="store_true", default=False)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_num_objects", type=int, default=128)
    parser.add_argument("--num_worlds", type=int, default=1)
    parser.add_argument("--num_scenarios", type=int, default=10000)
    parser.add_argument("--replan_interval_steps", type=int, default=10)
    parser.add_argument("--instruction_interval_steps", type=int, default=10)
    parser.add_argument("--visualize_mode", type=str, default="all", choices=["all", "success", "failure", "none"])
    parser.add_argument("--save_trajectory", action="store_true", default=False)
    parser.add_argument("--save_instruction", action="store_true", default=False)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--scenario_indices", type=str, default=None)
    parser.add_argument("--use_subgoal", action="store_true", default=False)
    parser.add_argument("--population_size", type=int, default=1)
    parser.add_argument(
        "--jax_device",
        type=str,
        default="cuda:0",
        help="JAX device to use for Waymax preprocessing and simulation; default: cuda:0.",
    )
    parser.add_argument(
        "--torch_device",
        type=str,
        default="cuda:1" if torch.cuda.device_count() > 1 else "cuda:0",
        help="Torch device for diffusion policy inference (for example: cuda, cuda:0, cuda:1).",
    )
    return parser.parse_args()


def _select_jax_device(device_name: str) -> jax.Device:
    device = jax.devices(device_name.split(":", 1)[0])[0] if device_name.startswith("cuda") else jax.devices(device_name)[0]
    if device_name.startswith("cuda"):
        idx = int(device_name.split(":", 1)[1]) if ":" in device_name else 0
        if idx >= len(jax.devices("cuda")):
            raise ValueError(
                f"Requested JAX device {device_name} is unavailable; available CUDA devices: "
                f"{list(map(str, jax.devices('cuda')))}"
            )
        device = jax.devices("cuda")[idx]
    return device


def _set_device_for_process(args: argparse.Namespace) -> tuple[jax.Device, torch.device]:
    jax_device = _select_jax_device(args.jax_device)
    torch_device = torch.device(args.torch_device)

    if torch_device.type == "cuda":
        if torch.cuda.device_count() == 0:
            raise ValueError("Torch CUDA device requested but no CUDA device is available.")
        if torch_device.index is not None and torch_device.index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested Torch device {args.torch_device} is unavailable; "
                f"available devices: {[f'cuda:{i}' for i in range(torch.cuda.device_count())]}"
            )
        torch.cuda.set_device(torch_device)

    return jax_device, torch_device


def main(args: argparse.Namespace) -> None:
    args.tfrecord_dir = os.path.join(args.tfrecord_dir, args.split)
    exp_name = f"simulation_torch_{args.planner}_{args.split}"
    if args.tag is not None:
        exp_name += f"_{args.tag}"
    exp_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.output_dir = os.path.join(args.exp_dir, exp_name)

    jax_device, torch_device = _set_device_for_process(args)
    planner = None
    if args.planner == "diffusion":
        inference = load_model_for_inference_torch(
            checkpoint_path=args.diffusion_ckpt_path,
            use_ema=args.use_ema,
            metadata_path=None,
            map_location=args.torch_device,
        )
        planner = DiffusionTorchPlanner(
            policy=inference.model,
            preprocess_cfg=inference.preprocess_cfg,
            population_size=args.population_size,
            num_worlds=args.num_worlds,
            device=inference.device,
        )

    with jax.default_device(jax_device):
        run(args, planner, log_replay=(args.planner == "log-replay"))


if __name__ == "__main__":
    parsed_args = _parse_args()
    main(parsed_args)
