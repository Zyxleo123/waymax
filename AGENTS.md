# waymax_rs

GPU/SLURM-oriented autonomous-driving ML research monorepo built on **Waymax**
(Waymo's JAX driving simulator) and the Waymo Open Motion Dataset (WOMD).
Sub-projects: a diffusion planner (`model/diffusion`, `train_diffusion`), an LLM
VLA planner (`model/vla`, `train_vla`, `vla`), a language annotation pipeline
(`annotation`, `scripts`), and shared sim/scoring/viz code (`simulation`,
`scores`, `lane_graph`, `viz`, `data`). There is **no web server, API, or
database** — everything is offline batch training/eval, normally launched on a
GPU cluster via the `slurm_*.sh` scripts.

## Cursor Cloud specific instructions

### Environment shape
- Dependencies live in a Python venv at `.venv` (gitignored). Activate with
  `source .venv/bin/activate` (Python 3.12). There is no `conda` here; `env.yml`
  is the upstream conda spec, but the cloud env uses pip into the venv.
- **This VM has no GPU.** The venv installs CPU builds of JAX, TensorFlow, and
  PyTorch. Always force CPU when running anything that touches JAX/TF/Waymax:
  `export WAYMAX_FORCE_CPU=1 JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES=""`.
  `TF_CPP_MIN_LOG_LEVEL=3` and `MPLBACKEND=Agg` quiet noisy logs / headless plots.
- The repo's Python packages (`viz`, `data`, `scores`, `model`, `simulation`,
  `simulation_utils`, `lane_graph`, `train_diffusion`, `train_vla`, `vla`,
  `annotation`) are **not pip-installed**; run from the repo root with
  `PYTHONPATH=/workspace` (or `cd /workspace`) so imports resolve.

### What runs here vs. what does not
- Runnable on this CPU VM: Waymax simulation/metrics, the repo's visualization
  code, and `viz/viz_test.py`. Use these for smoke testing.
- **Not runnable here:** full training, simulation rollouts, and goal-reaching
  (`scripts/main_goal_reaching.py`, `python -m train_*`, `python -m simulation.*`).
  These need a GPU, the proprietary WOMD tfrecords, pre-extracted lane graphs,
  and (for VLA) gated HuggingFace weights (Qwen/Gemma) — none of which are
  present. Don't treat their absence as an environment bug.
- The Gemini annotation pipeline (`annotation/`, uses `google-generativeai`) is
  **not usable here**: `google-generativeai` requires `protobuf<6`, but the core
  TensorFlow/Waymax stack requires `protobuf>=6.31.1`. The env keeps protobuf at
  6.x for the core, so importing `google.generativeai` will fail.

### Tests
- `python -m pytest viz/viz_test.py -q` (the only test suite). It needs Waymax's
  upstream `tfrecord_with_routes` test file, which is missing from the waymax pip
  wheel; the update script restores it into the installed package's
  `dataloader/testdata/` dir. If tests error with "No such file ...
  tfrecord_with_routes", re-run the update script to re-fetch it.
- There is no configured linter/formatter in the repo.

### Smoke test (hello-world)
`PYTHONPATH=/workspace python scripts/hello_world_viz.py out.png` builds a
synthetic Waymax `SimulatorState`, runs overlap/offroad metrics, and renders it
with the repo's `viz.plot_simulator_state` — a quick end-to-end check that the
sim + scoring + visualization path works without a GPU or dataset.
