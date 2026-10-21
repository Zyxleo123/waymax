# Babel paths for the AnchorWM v2 pipeline (sourced). Exported so sbatch hands them to every job, where
# common.sh picks them up in place of its rhea defaults. /data/user_data is only mounted on compute nodes.
export ROOT="${ROOT:-/home/mineuih/ishan/waymax_rs}"
export CONDA_ROOT="${CONDA_ROOT:-/home/mineuih/miniconda3}"
export WAYMAX_ENV="${WAYMAX_ENV:-/home/mineuih/miniconda3/envs/waymax_rs_ishan}"
export TF_ROOT="${TF_ROOT:-/data/datasets/waymo/waymo-open-dataset-v1.3.1/tf_example}"
export OUT_ROOT="${OUT_ROOT:-/data/user_data/mineuih/ishan/anchor_wm}"
export CACHE_DIR="${CACHE_DIR:-/data/user_data/mineuih/waymax_rs/scene_state_cache_npz}"
export ANCHORS="${ANCHORS:-${OUT_ROOT}/anchors/anchors_k64.npz}"
export MAX_STEPS_FILE="${MAX_STEPS_FILE:-${OUT_ROOT}/v2_max_steps.txt}"
export LOG_DIR="${LOG_DIR:-/home/mineuih/ishan/logs}"
export WANDB_PROJECT="${WANDB_PROJECT:-anchor_wm}"
export WANDB_ENTITY="${WANDB_ENTITY-scbellad2025-carnegie-mellon-university}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"   # the team's W&B host (~/.netrc)
export GPU_GRES="${GPU_GRES:-gpu:L40S:1}"        # same GPU type for profiling and training
