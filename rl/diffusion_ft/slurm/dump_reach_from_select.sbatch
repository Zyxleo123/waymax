#!/bin/bash
#SBATCH --job-name=dft_scenes
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Geometry-only scene bundle for real recovered / still-failing / preserved
# trajectories, plus the failure-cause breakdown of the whole failure split
# (rl/diffusion_ft/tools/dump_reach_from_select.py). Reloads roadgraph + agents +
# goal for a chosen scene set and pairs them with the paths the select run already
# dumped -- NO denoiser, NO rollout. Still needs a GPU because the diffusion env
# only initialises on a6000/a5000 (a5000=gpu28 excluded by the submitter).
#
# Config arrives as a positional env FILE. Never sbatch --export.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"
cd "$ROOT"; mkdir -p "$ROOT/logs/dft"
for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done

PY="${DFT_PYTHON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
DFT_CKPT="${DFT_CKPT:-/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
DFT_SHARD_GLOB="${DFT_SHARD_GLOB:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
DFT_INDEX_DIR="${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
DFT_SPLITS_JSON="${DFT_SPLITS_JSON:?set DFT_SPLITS_JSON}"
DFT_SELECT_DIR="${DFT_SELECT_DIR:?set DFT_SELECT_DIR (a select run scored with --dump_traj)}"
DFT_OUT_JSON="${DFT_OUT_JSON:?set DFT_OUT_JSON}"
DFT_VIZ_ARMS="${DFT_VIZ_ARMS:-base,c_r10,c_r20}"
DFT_RECOVERED="${DFT_RECOVERED:-}"
DFT_UNRECOVERED="${DFT_UNRECOVERED:-}"
DFT_PRESERVED="${DFT_PRESERVED:-}"
DFT_REGRESSED="${DFT_REGRESSED:-}"
# N>0: pick up to N scenes per kind from the run's own rows instead of the lists.
DFT_AUTO_PICK="${DFT_AUTO_PICK:-0}"
DFT_SOLVED_ARM="${DFT_SOLVED_ARM:-c_r10}"

echo "host=${SLURMD_NODENAME:-unknown}  select=$DFT_SELECT_DIR"
echo "arms=$DFT_VIZ_ARMS  recovered=$DFT_RECOVERED  preserved=$DFT_PRESERVED"
echo "unrecovered=$DFT_UNRECOVERED"

# Gate inside the job: explain and exit 0 rather than strand the CPU html step.
if ! compgen -G "$DFT_SELECT_DIR/tasks/*/feat/*.npz" > /dev/null; then
  echo "SCENE DUMP NOT RUN: no feat/*.npz under $DFT_SELECT_DIR"; exit 0
fi

srun "$PY" -u -m rl.diffusion_ft.tools.dump_reach_from_select \
  --ckpt "$DFT_CKPT" --shard_glob "$DFT_SHARD_GLOB" --index_dir "$DFT_INDEX_DIR" \
  --splits_json "$DFT_SPLITS_JSON" --out_dir "$(dirname "$DFT_OUT_JSON")" \
  --select_dir "$DFT_SELECT_DIR" --out_json "$DFT_OUT_JSON" \
  --viz_arms "$DFT_VIZ_ARMS" --solved_arm "$DFT_SOLVED_ARM" \
  --recovered "$DFT_RECOVERED" --unrecovered "$DFT_UNRECOVERED" \
  --preserved "$DFT_PRESERVED" --regressed "$DFT_REGRESSED" --auto_pick "$DFT_AUTO_PICK" \
  --eval_scenes 0 --prefetch 0 --probe_per_split 0 --no-wandb
echo "json -> $DFT_OUT_JSON"
