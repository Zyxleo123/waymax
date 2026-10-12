#!/bin/bash
# Submitter for the FlowMol3 arm evals. Only calls sbatch.
#
# Score the four arms plus the untrained base reference:
#   FM3_SAVE_DIR=/zfsauton/scratch/yixiz/waymax_rs/flowmol3/<stamp> \
#   DFT_SPLITS_JSON=<verified splits.json> \
#   DFT_SHARD_GLOB='<the glob splits.json was built against>' \
#   bash rl/diffusion_ft/slurm/submit_eval_flowmol3.sh
#
# Chain it behind the training array so each arm is scored as soon as ITS OWN
# task finishes (aftercorr maps array index to array index; a plain afterok
# would make every eval wait for the slowest arm):
#   FM3_TRAIN_JOBID=<jobid> ... bash rl/diffusion_ft/slurm/submit_eval_flowmol3.sh
#
# The base reference has no training task, so it is submitted as its own job
# with no dependency -- an aftercorr task with no counterpart would sit in
# DependencyNeverSatisfied forever.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
: "${FM3_SAVE_DIR:?set FM3_SAVE_DIR to the save dir of the training run}"
: "${DFT_SPLITS_JSON:?set DFT_SPLITS_JSON}"
: "${DFT_SHARD_GLOB:?set DFT_SHARD_GLOB}"

STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_DIR="$ROOT/logs/dft/env/fm3eval_${STAMP}"; mkdir -p "$ENV_DIR"
OUT_DIR="${FM3_EVAL_OUT:-$FM3_SAVE_DIR/eval_${STAMP}}"
BASE_CKPT="${FM3_INIT_FROM:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
ARMS=(base sc dist sc_dist)

emit_env() {  # $1 = file, $2 = name, $3 = ckpt glob
  {
    echo "FM3_EVAL_NAME='$2'"
    echo "FM3_EVAL_GLOB='$3'"
    echo "FM3_EVAL_OUT='${OUT_DIR}'"
    echo "DFT_SPLITS_JSON='${DFT_SPLITS_JSON}'"
    echo "DFT_SHARD_GLOB='${DFT_SHARD_GLOB}'"
    for v in DFT_SPLITS DFT_SPLIT_CAP DFT_EVAL_CHUNK DFT_EVAL_SEED DFT_INDEX_DIR DFT_PYTHON; do
      [[ -n "${!v:-}" ]] && echo "${v}='${!v}'"
    done
    true
  } > "$1"
  ( set -euo pipefail; set -a; . "$1"; set +a
    [[ "$FM3_EVAL_NAME" == "$2" ]] || { echo "FATAL: env round trip broken ($1)" >&2; exit 1; } )
}

for i in "${!ARMS[@]}"; do
  emit_env "$ENV_DIR/${i}.env" "${ARMS[$i]}" "$FM3_SAVE_DIR/${ARMS[$i]}/*/latest"
done
emit_env "$ENV_DIR/base_ckpt.env" "base_ckpt" "$BASE_CKPT"

echo "Env dir: $ENV_DIR"
echo "Out dir: $OUT_DIR"

DEP=()
if [[ -n "${FM3_TRAIN_JOBID:-}" ]]; then
  DEP=(--dependency="aftercorr:${FM3_TRAIN_JOBID}")
  echo "Chained aftercorr on train job ${FM3_TRAIN_JOBID}"
fi

env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch "${DEP[@]}" --array="${FM3_ARRAY:-0-3}%2" \
  rl/diffusion_ft/slurm/eval_flowmol3.sbatch "$ENV_DIR"

# The untrained base reference: no dependency, runs immediately.
env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch rl/diffusion_ft/slurm/eval_flowmol3.sbatch "$ENV_DIR/base_ckpt.env"
