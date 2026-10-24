#!/bin/bash
# Dense-bank ladder: does the repair neighbourhood have more in it, or is the
# lat x tw family exhausted? Only calls sbatch (via submit_select_eval.sh).
#
# QUESTION. Eight fixed edits of the base's own proposal take failure_heldout
# from 0.034 to 0.39, and the lateral half carries nearly all of it (lat-only
# 0.348, tw-only 0.109). Those 8 are a coarse grid over a continuous 2-D edit
# space. This walks the LATERAL grid from 4 to 24 offsets, paired draw for draw
# in ONE job, and reads off the shape of the curve:
#
#   still climbing at 24  -> the grid was the limit, not the family. A learned
#                            CONTINUOUS edit head (distilling this search) has
#                            real headroom above 0.39, and the ladder's top is
#                            the number it has to beat.
#   saturates near 0.40   -> the 2-D family is exhausted. No learned head that
#                            outputs a lat/tw edit can beat it either, and the
#                            residual needs a richer edit space or a longer
#                            horizon. Worth knowing BEFORE writing training code.
#
# Pair this with submit_residual_decomp.sh, which predicts the answer from the
# run you already have: at replans where no candidate was clean, how far short
# did the best one fall?
#
# DENSER IS NOT MORE AGGRESSIVE. `bank_choice` executes the eligible candidate of
# SMALLEST magnitude (|offset| + 0.1 for lat), so a finer grid means the minimal
# repairing edit is found closer to the base, not further from it. And every
# candidate faces the same per-candidate `cons` test, so the do-no-harm property
# -- a clean base proposal is never touched, at any K -- holds verbatim. That is
# what success_heldout in this sweep is there to confirm.
#
# COST. Candidates cost NO denoiser evaluations, only a geometry score each, but
# `proposal_select.rollout` scores them in a Python loop over K: this ladder is
# 72 candidate-scorings per replan against the previous run's 32. Budget roughly
# 2x that run's wall (it stayed well inside the 8 h limit). Smoke first.
#
# Usage (from repo root, ON RHEA):
#   DFT_SELECT_SMOKE=1 bash rl/diffusion_ft/slurm/submit_dense_bank.sh   # ~20 min, 4 scenes/split
#   bash rl/diffusion_ft/slurm/submit_dense_bank.sh                      # the real ladder
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/home/yixiz/waymax_rs}"; cd "$ROOT"

L4="lat0.25,lat-0.25,lat0.5,lat-0.5"
L8="lat0.15,lat-0.15,lat0.35,lat-0.35,lat0.6,lat-0.6,lat0.9,lat-0.9"
L16="lat0.1,lat-0.1,lat0.2,lat-0.2,lat0.3,lat-0.3,lat0.4,lat-0.4,\
lat0.55,lat-0.55,lat0.7,lat-0.7,lat0.9,lat-0.9,lat1.2,lat-1.2"
L24="lat0.1,lat-0.1,lat0.175,lat-0.175,lat0.25,lat-0.25,lat0.325,lat-0.325,\
lat0.4,lat-0.4,lat0.5,lat-0.5,lat0.6,lat-0.6,lat0.75,lat-0.75,\
lat0.9,lat-0.9,lat1.1,lat-1.1,lat1.35,lat-1.35,lat1.6,lat-1.6"
TW4="tw0.85,tw1.15,tw0.7,tw1.3"

# 'name=spec;name=spec'. The submitter single-quotes every env-file value and
# verifies the round trip, so the ';' separators survive `source` (they did not
# before f9831ae -- jobs 55580/55642 silently lost everything after the first ';').
export DFT_SELECT_BANKS="l4=${L4};l8=${L8};l16=${L16};l24=${L24};dense=${L16},${TW4}"
export DFT_SELECT_BANK="$L4"          # default bank; every arm below names one
export DFT_BASE_ARMS="base,b_l4=bank@l4:cons:20,b_l8=bank@l8:cons:20,\
b_l16=bank@l16:cons:20,b_l24=bank@l24:cons:20,b_dense=bank@dense:cons:20"
# One checkpoint's fine-tune arm, kept in the SAME job so the learned alternative
# is paired against the ladder draw for draw -- it is the baseline the distilled
# edit head would have to beat, and it sat at 0.125-0.250 while 8 fixed edits sat
# at 0.39.
export DFT_FT_ARMS="${DFT_FT_ARMS:-bf_cons10=ft:cons:10}"
RUN=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/20260911_160658_962947
export DFT_SELECT_CKPTS="${DFT_SELECT_CKPTS:-$RUN/base_s0/iter_000110}"

echo "ladder:"
for n in l4 l8 l16 l24 dense; do
  spec="$(sed 's/.*'"$n"'=//; s/;.*//' <<< "$DFT_SELECT_BANKS")"
  printf '  %-6s K=%2d  %s\n' "$n" "$(tr -cd ',' <<< "$spec" | wc -c | awk '{print $1+1}')" "$spec"
done
echo
exec bash rl/diffusion_ft/slurm/submit_select_eval.sh
