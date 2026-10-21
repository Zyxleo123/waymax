"""Runs Diffusion Planner's own train_predictor.py (unmodified) on the WOMD samples from convert_womd.py.

Their decoder reads ``config.guidance_fn``, which train_predictor.py's arguments never define, so their training
script fails as published (upstream issues #32, #42). This launcher sets it to None (no guidance, as in their
main results) and otherwise passes every argument through to their parser.

  torchrun --nproc-per-node <gpus> diffusion_planner_womd/train.py --train_set <data_root> \
      --train_set_list <data_root>/train_list.json --time_len 21 --future_len 50 ...

Their ddp_setup_universal sets MASTER_PORT to --port. Under torchrun (TORCHELASTIC_USE_AGENT_STORE=True) the
ranks then wait for torchrun's store on that port, where nothing listens, and hang; --port is therefore set to
torchrun's own MASTER_PORT, which makes their assignment a no-op.

Their logger imports torch.utils.tensorboard, which imports TensorFlow when it is installed (it is, for Waymax);
TensorFlow's CUDA 12 libraries then break torch's (CUDA 13) cuSOLVER (CUSOLVER_STATUS_INTERNAL_ERROR in their
augmentation's torch.linalg.inv). Training never needs TensorFlow, so its import is blocked and tensorboard
uses its TensorFlow-free stub.

WOMD has no speed limits and few static objects, so the speed-limit embedding and the static-object encoder
often get no gradient, which their DistributedDataParallel (built without find_unused_parameters) rejects;
DDP is created with find_unused_parameters=True.

Their checkpoints hold a numpy scalar (the loss), which torch >= 2.6 refuses to load by default, so resuming
(--resume_model_path) would fail; torch.load is given weights_only=False for these self-written checkpoints.

Their repo and diffusion_planner_womd/nuplan_shim must be on PYTHONPATH (the shim provides the one nuPlan
import on their training path); run from their repo root so the default normalization.json resolves.

``--goal_conditioned --goal_dir <data_root>/goals`` (ours, taken out before their parser) trains the
goal-conditioned variant (goal.py); both are saved in args.json, from which the closed-loop planner rebuilds it.
"""

import argparse
import functools
import os
import sys
from pathlib import Path

sys.modules["tensorflow"] = None  # see above: keeps TensorFlow's CUDA libraries out of this process
sys.path.append(str(Path(__file__).resolve().parents[1]))  # waymax_rs, after their repo

import torch  # noqa: E402

torch.load = functools.partial(torch.load, weights_only=False)

import train_predictor  # noqa: E402

from diffusion_planner_womd import goal  # noqa: E402

train_predictor.DDP = functools.partial(train_predictor.DDP, find_unused_parameters=True)
_get_args = train_predictor.get_args


def get_args():
    ours = argparse.ArgumentParser(add_help=False)
    ours.add_argument("--goal_conditioned", action="store_true")
    ours.add_argument("--goal_dir", default=None)
    own, sys.argv[1:] = ours.parse_known_args()
    args = _get_args()
    args.guidance_fn = None
    if "MASTER_PORT" in os.environ:
        args.port = os.environ["MASTER_PORT"]
    args.goal_conditioned = own.goal_conditioned
    if own.goal_conditioned:
        if not own.goal_dir:
            raise SystemExit("--goal_conditioned needs --goal_dir")
        args.goal_dir = own.goal_dir
        goal.extend_normalizer(args.observation_normalizer)
        train_predictor.DiffusionPlannerData = goal.goal_dataset(train_predictor.DiffusionPlannerData, own.goal_dir)
    return args


if __name__ == "__main__":
    goal.install()
    train_predictor.model_training(get_args())
