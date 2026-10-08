#!/bin/bash
# Residual decomposition of an existing selection run. Only calls sbatch.
#
# QUESTION. The cons filter takes failure_heldout from 0.034 to 0.39. Of the ~61%
# it does NOT save, how much is each of: no_goal (never reached -- a restriction
# filter cannot invent progress), unseen (violation never predicted -- horizon),
# coverage (base predicted to violate and NO candidate was clean -- densify or
# enrich the edit family), prog_blocked (a clean candidate existed but lost more
# than tol_m of progress -- the tolerance binds), fired_unsaved (every executed
# candidate was predicted clean and it violated anyway -- compounding past the
# window)? Each bucket points at a different fix, and only one of them is the
# one a denser bank or a learned continuous edit can touch.
#
# It also reports, at coverage-limited replans, how far the BEST candidate fell
# short. That number PREDICTS the dense-bank sweep: near 0 means a larger edit of
# the same family clears it; strongly negative means the family is exhausted.
#
# CPU only, reads artefacts a past run already wrote, so it is free to re-run.
#
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/submit_residual_decomp.sh                  # newest select dir
#   bash rl/diffusion_ft/slurm/submit_residual_decomp.sh <select_dir>
#   DFT_RESID_SPLITS=failure_heldout bash rl/diffusion_ft/slurm/submit_residual_decomp.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"
SEL_ROOT=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select

SEL_DIR="${1:-${DFT_SELECT_DIR:-}}"
if [[ -z "$SEL_DIR" ]]; then
  # Newest dir that actually has per-replan features, not merely the newest dir.
  for d in $(ls -dt "$SEL_ROOT"/*/ 2>/dev/null); do
    if compgen -G "$d/tasks/*/feat/*.npz" > /dev/null; then SEL_DIR="${d%/}"; break; fi
  done
fi
[[ -n "$SEL_DIR" ]] || { echo "FATAL: no select dir with feat/*.npz under $SEL_ROOT" >&2; exit 1; }
[[ -d "$SEL_DIR" ]] || { echo "FATAL: '$SEL_DIR' is not a directory" >&2; exit 1; }
compgen -G "$SEL_DIR/tasks/*/feat/*.npz" > /dev/null || {
  echo "FATAL: no tasks/*/feat/*.npz under $SEL_DIR (nothing to decompose)" >&2; exit 1; }

ENV_FILE="$SEL_DIR/residual.env"
kv() { printf "%s='%s'\n" "$1" "${2//\'/\'\\\'\'}"; }
{ kv DFT_SELECT_DIR "$SEL_DIR"
  kv DFT_RESID_SPLITS "${DFT_RESID_SPLITS:-failure_heldout,intermittent_heldout}"
  kv DFT_RESID_ARMS "${DFT_RESID_ARMS:-}"
  kv DFT_RESID_WORLD_DT "${DFT_RESID_WORLD_DT:-0.1}"; } > "$ENV_FILE"
# Same round-trip invariant the selection submitter uses: a value that does not
# survive `source` is caught here, not after the job has started.
while IFS= read -r k; do
  want="$(sed -n "s/^${k}='\(.*\)'$/\1/p" "$ENV_FILE")"
  got="$(env -i bash -c 'source "$1" >/dev/null 2>&1; printf "%s" "${!2}"' _ "$ENV_FILE" "$k")"
  [[ "$want" == "$got" ]] || {
    echo "FATAL: $ENV_FILE does not round-trip: $k wrote [$want] read [$got]" >&2; exit 1; }
done < <(sed -n 's/^\(DFT_[A-Z_]*\)=.*/\1/p' "$ENV_FILE")

JID=$(env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch --parsable \
  rl/diffusion_ft/slurm/residual_decomp.sbatch "$ENV_FILE")
echo "decomp -> job $JID (CPU)"
echo "dir    -> $SEL_DIR"
echo "out    -> $SEL_DIR/residual_*.{json,png}"
