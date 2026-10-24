#!/bin/bash
# Anatomy of what the safety fine-tune still fails. Only calls sbatch.
#
# QUESTION. rows.jsonl says the ft arm leaves ~68% of failure_heldout unsolved
# and splits it roughly 38% never-reached / 25% overlap / 36% off-road. Three
# readings of that split change what to build next, and none is decidable from
# booleans:
#
#   * NO-GOAL: is it pace (right line, too slow -- nothing a geometric
#     restriction can fix) or route (wrong line)? Measured by the ego's speed
#     and along-path lag against its OWN log, and by cross-track error.
#   * OVERLAP: is it downstream of the stall? Log-replayed traffic does not
#     yield, so an ego metres behind its log gets driven into. Measured by the
#     collider's bearing in the ego frame and the ego's lag at contact.
#   * OFF-ROAD: can the barrier lead the car out at all? Measured by sweeping
#     the ego box laterally at every excursion onset and recomputing the SAME
#     clearance waymax thresholds -- so "there is no on-road pose within +-4 m"
#     and "the loss pushes the wrong way" are ground truth, not inference.
#
# CPU only, over artefacts a past select run already wrote. The select run must
# have been submitted with DFT_DUMP_TRAJ=1, or there are no paths to score.
#
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/submit_failure_anatomy.sh                 # newest dir with traj
#   bash rl/diffusion_ft/slurm/submit_failure_anatomy.sh <select_dir>
#   DFT_ANAT_SPLIT=intermittent_heldout bash rl/diffusion_ft/slurm/submit_failure_anatomy.sh
#   DFT_ANAT_SMOKE=1 bash rl/diffusion_ft/slurm/submit_failure_anatomy.sh   # 8 scenes, 1 chunk
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"
SEL_ROOT=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select

# Arms whose `<arm>/r<rep>/traj` exists in a feat npz, as a comma-separated list.
# An .npz IS a zip and each array is one `<key>.npy` member, so this reads the
# key names with the STDLIB -- rhea's system python3 has no numpy, and the
# submitter has to run there (every partition sets AllocNodes=rhea).
command -v python3 >/dev/null || {
  echo "FATAL: no python3 on PATH; this submitter needs one only to read zip members" >&2
  exit 1; }
npz_traj_arms() {
  python3 - "$1" <<'PY'
import sys, zipfile
try:
    with zipfile.ZipFile(sys.argv[1]) as z:
        keys = [n[:-4] for n in z.namelist() if n.endswith(".npy")]
except Exception:
    raise SystemExit(0)
print(",".join(sorted({k.split("/")[0] for k in keys if k.endswith("/traj")})))
PY
}

SEL_DIR="${1:-${DFT_SELECT_DIR:-}}"
if [[ -z "$SEL_DIR" ]]; then
  # The newest dir that actually dumped TRAJECTORIES, not merely the newest dir:
  # feat/*.npz exists for every select run, `<arm>/r0/traj` only under --dump_traj.
  for d in $(ls -dt "$SEL_ROOT"/*/ 2>/dev/null); do
    f=$(ls "$d"tasks/*/feat/*.npz 2>/dev/null | head -1) || true
    [[ -n "${f:-}" ]] || continue
    [[ -n "$(npz_traj_arms "$f")" ]] || continue
    SEL_DIR="${d%/}"; break
  done
fi
[[ -n "$SEL_DIR" ]] || { echo "FATAL: no select dir with dumped traj under $SEL_ROOT" >&2; exit 1; }
[[ -d "$SEL_DIR" ]] || { echo "FATAL: '$SEL_DIR' is not a directory" >&2; exit 1; }

SPLIT="${DFT_ANAT_SPLIT:-failure_heldout}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${DFT_ANAT_OUT:-$SEL_DIR/anatomy_$STAMP}"
CHUNKS="${DFT_ANAT_CHUNKS:-6}"
CAP="${DFT_ANAT_CAP:--1}"
if [[ "${DFT_ANAT_SMOKE:-0}" == "1" ]]; then CHUNKS=1; CAP="${DFT_ANAT_CAP:-8}"; fi

# Which arms exist in the dump; asking for one that does not is a silent no-op
# (the tool skips a missing `<arm>/r<rep>/traj`), so it is checked here instead.
# A glob, not `ls | head -1`: under pipefail, head closing the pipe early kills
# ls with SIGPIPE once there are enough files, and the script exits 141 silently.
FEATS=("$SEL_DIR"/tasks/*/feat/*.npz)
FEAT="${FEATS[0]}"
[[ -f "$FEAT" ]] || { echo "FATAL: no feat/*.npz under $SEL_DIR" >&2; exit 1; }
HAVE=$(npz_traj_arms "$FEAT")
[[ -n "$HAVE" ]] || { echo "FATAL: $SEL_DIR has no dumped trajectories (run with DFT_DUMP_TRAJ=1)" >&2; exit 1; }
# Default to the three arms the question is about -- the frozen base, the safety
# fine-tune, and the shipping repair bank -- not to every arm in the dump. Cost
# is linear in the arm count and the ramp variants answer a different question.
WANT="base,ft,c_r10,b_l24,bf_w10"
DEFAULT=""
for a in ${WANT//,/ }; do
  [[ ",$HAVE," == *",$a,"* ]] && DEFAULT="${DEFAULT:+$DEFAULT,}$a"
done
ARMS="${DFT_ANAT_ARMS:-${DEFAULT:-$HAVE}}"
for a in ${ARMS//,/ }; do
  [[ ",$HAVE," == *",$a,"* ]] || { echo "FATAL: arm '$a' has no traj in the dump (have: $HAVE)" >&2; exit 1; }
done
SWEEP_ARM="${DFT_ANAT_SWEEP_ARM-ft}"
if [[ -n "$SWEEP_ARM" && ",$HAVE," != *",$SWEEP_ARM,"* ]]; then
  echo "FATAL: sweep arm '$SWEEP_ARM' has no traj in the dump (have: $HAVE)" >&2; exit 1
fi

mkdir -p "$OUT_DIR"
ENV_FILE="$OUT_DIR/anatomy.env"
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }
{ kv DFT_SELECT_DIR "$SEL_DIR"
  kv DFT_ANAT_OUT "$OUT_DIR"
  kv DFT_ANAT_SPLIT "$SPLIT"
  kv DFT_ANAT_ARMS "$ARMS"
  kv DFT_ANAT_REPEATS "${DFT_ANAT_REPEATS:-4}"
  kv DFT_ANAT_CHUNKS "$CHUNKS"
  kv DFT_ANAT_CAP "$CAP"
  kv DFT_ANAT_LOAD_BATCH "${DFT_ANAT_LOAD_BATCH:-8}"
  kv DFT_SPLITS_JSON "${DFT_SPLITS_JSON:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata.json}"
  kv DFT_SHARD_GLOB "${DFT_SHARD_GLOB:-/data/user_data/yixiz/autonlab/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-*}"
  kv DFT_INDEX_DIR "${DFT_INDEX_DIR:-/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets}"
  kv DFT_ANAT_SWEEP_ARM "$SWEEP_ARM"
  kv DFT_ANAT_SWEEP_MAX "${DFT_ANAT_SWEEP_MAX:-3}"
  kv DFT_ANAT_SWEEP_LAT "${DFT_ANAT_SWEEP_LAT:-4.0}"
  kv DFT_ANAT_SWEEP_STEP "${DFT_ANAT_SWEEP_STEP:-0.1}"; } > "$ENV_FILE"
MERGE_ENV="$OUT_DIR/anatomy_merge.env"
{ cat "$ENV_FILE"; kv DFT_ANAT_MERGE 1; } > "$MERGE_ENV"

# Same round-trip invariant the other submitters use: a value that does not
# survive `source` (an unquoted `;` or `*`) is caught here, not after the job runs.
for F in "$ENV_FILE" "$MERGE_ENV"; do
  while IFS= read -r k; do
    want="$(sed -n "s/^${k}='\(.*\)'$/\1/p" "$F" | tail -1)"
    got="$(env -i bash -c 'source "$1" >/dev/null 2>&1; printf "%s" "${!2}"' _ "$F" "$k")"
    [[ "$want" == "$got" ]] || {
      echo "FATAL: $F does not round-trip: $k wrote [$want] read [$got]" >&2; exit 1; }
  done < <(sed -n 's/^\(DFT_[A-Z_]*\)=.*/\1/p' "$F" | sort -u)
done

SB=(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch --parsable)
AID=$("${SB[@]}" --array=0-$((CHUNKS - 1)) \
  rl/diffusion_ft/slurm/failure_anatomy.sbatch "$ENV_FILE")
MID=$("${SB[@]}" --dependency=afterok:"$AID" --job-name=dft_anat_merge \
  rl/diffusion_ft/slurm/failure_anatomy.sbatch "$MERGE_ENV")

echo "anatomy -> array $AID (0-$((CHUNKS - 1)), CPU)   merge -> $MID"
echo "select  -> $SEL_DIR"
echo "split   -> $SPLIT   arms -> $ARMS   sweep -> ${SWEEP_ARM:-<none>}"
echo "out     -> $OUT_DIR/anatomy_$SPLIT.json"
echo "logs    -> $ROOT/logs/dft/slurm-dft_anat-$AID"'_*.out'
