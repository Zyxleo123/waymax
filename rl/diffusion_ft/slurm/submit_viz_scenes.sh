#!/bin/bash
# Real trajectories of failure/success scenes for the deploy-time repair line.
# Only calls sbatch. Two stages chained afterok:
#   1. dump_reach_from_select (GPU, a6000): reload roadgraph/agents/goal for the
#      chosen scenes, pair with the select run's already-dumped paths -> scenes.json
#   2. build_scene_viz (CPU): scenes.json -> interactive scenes.html (toggle arms)
#
# Config travels as ONE timestamped env file passed positionally (never --export).
#
# Usage (from repo root, ON RHEA):
#   bash rl/diffusion_ft/slurm/submit_viz_scenes.sh
#   DFT_RECOVERED="59208,60665" DFT_PRESERVED="8844,50" bash rl/diffusion_ft/slurm/submit_viz_scenes.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"

SEL=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/select
# Ramp-sweep run scored with --dump_traj; has base + c_r10 (=ship l24 geometry) + c_r20.
DFT_SELECT_DIR="${DFT_SELECT_DIR:-$SEL/20260918_110957_3371380}"
DFT_SPLITS_JSON="${DFT_SPLITS_JSON:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/verified_seed777/eval_splits_3strata.json}"
DFT_VIZ_ARMS="${DFT_VIZ_ARMS:-base,c_r10,c_r20}"
DFT_SOLVED_ARM="${DFT_SOLVED_ARM:-c_r10}"
# Dump every arm, draw only these (empty = all). c_r20 is scored and kept in the
# bundle as the comfort A/B, but the page is the single-arm story.
DFT_SHOW_ARMS="${DFT_SHOW_ARMS:-base,c_r10}"
# Curated: offroad + overlap recoveries; preserved (2 touched-but-clean, 1 untouched).
DFT_RECOVERED="${DFT_RECOVERED:-59208,60665,86688,126300,12956,17880}"
DFT_PRESERVED="${DFT_PRESERVED:-8844,5536,50}"
# Curated residual -- scenes c_r10 solves in 0 of the 4 repeats, one per mode of
# failing, so the page cannot be read as a highlight reel:
#   7040,18412    pure no-goal, the filter never fires (nothing is predicted wrong)
#   383195,291317 base's overlap/offroad IS repaired, the car still never arrives
#   51493         the filter fires up to 5x and the off-road excursion survives
#   5757          a collision the violation predictor never flags
DFT_UNRECOVERED="${DFT_UNRECOVERED:-7040,18412,383195,291317,51493,5757}"

OUT_DIR="${DFT_SCENE_OUT_DIR:-$ROOT/logs/dft/scenes}"; mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_JSON="$OUT_DIR/scenes_${STAMP}.json"
OUT_HTML="$OUT_DIR/scenes_${STAMP}.html"

# Scrub SLURM_* so a submission from inside an allocation does not trip user-env retrieval.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
sb() { env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" sbatch "$@"; }

ENVF="$OUT_DIR/scenes_${STAMP}.env"
{
  echo "DFT_SELECT_DIR='$DFT_SELECT_DIR'"
  echo "DFT_SPLITS_JSON='$DFT_SPLITS_JSON'"
  echo "DFT_OUT_JSON='$OUT_JSON'"
  echo "DFT_OUT_HTML='$OUT_HTML'"
  echo "DFT_VIZ_ARMS='$DFT_VIZ_ARMS'"
  echo "DFT_SHOW_ARMS='$DFT_SHOW_ARMS'"
  echo "DFT_SOLVED_ARM='$DFT_SOLVED_ARM'"
  echo "DFT_RECOVERED='$DFT_RECOVERED'"
  echo "DFT_UNRECOVERED='$DFT_UNRECOVERED'"
  echo "DFT_PRESERVED='$DFT_PRESERVED'"
} > "$ENVF"

# Verify every key round-trips through a clean source before sbatch.
while read -r k; do
  want="$(sed -n "s/^${k}='\(.*\)'$/\1/p" "$ENVF")"
  got="$(env -i bash -c 'source "$1" >/dev/null 2>&1; printf "%s" "${!2}"' _ "$ENVF" "$k")"
  [[ "$want" == "$got" ]] || { echo "FATAL: $ENVF does not round-trip for $k" >&2; exit 1; }
done < <(sed -n 's/^\(DFT_[A-Z_]*\)=.*/\1/p' "$ENVF")

echo "env  -> $ENVF"
echo "json -> $OUT_JSON"
echo "html -> $OUT_HTML"

DID=$(sb --parsable --exclude=gpu28 rl/diffusion_ft/slurm/dump_reach_from_select.sbatch "$ENVF")
VID=$(sb --parsable --dependency="afterok:${DID}" rl/diffusion_ft/slurm/viz_scenes.sbatch "$ENVF")
echo "dump  -> job $DID (GPU a6000)"
echo "html  -> job $VID (CPU, afterok:$DID)"
