"""Closed-loop entry point for the Diffusion Planner baseline: simulation/run_simulation.py with GPU torch.

In the waymax_rs env TensorFlow (imported by the runner to read TFRecords; CPU only) breaks torch when it is
loaded first:
  - its CUDA 12 libraries break torch's (CUDA 13) cuSOLVER, so torch's CUDA libraries, which load lazily, are
    bound here by running the kinds of ops the model uses;
  - its bundled LLVM crashes Triton's on load (segfault), and Diffusion Planner's encoder imports timm, which
    imports torchvision -> torch._dynamo -> triton, so timm is imported here.
Both happen before anything imports TensorFlow. Arguments are those of run_simulation.py.

  python diffusion_planner_womd/run_closed_loop.py --planner diffusion_planner --dp_repo ... --dp_args_file ...
"""

import sys
from pathlib import Path

import timm  # noqa: F401  (loads triton before TensorFlow)
import torch

if torch.cuda.is_available():
    x = torch.randn(8, 4, 16, 32, device="cuda")
    torch.nn.functional.scaled_dot_product_attention(x, x, x)
    torch.nn.functional.layer_norm(x, (32,)) @ x.transpose(-1, -2)
    torch.linalg.inv(torch.eye(6, device="cuda"))
    torch.cuda.synchronize()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation import run_simulation  # noqa: E402  (imports TensorFlow via the runner)

if __name__ == "__main__":
    run_simulation.main(run_simulation._parse_args())
