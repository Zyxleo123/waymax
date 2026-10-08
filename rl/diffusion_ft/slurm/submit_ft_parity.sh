#!/bin/bash
# Window-matched fine-tune baseline. Only calls sbatch (via submit_select_eval.sh).
#
# QUESTION, and it is a reviewer's question before it is ours. Every headline so
# far compares `bf_cons10` (the fine-tune at selector window 10, ONE checkpoint,
# iter_000110 of a stopping sweep) against bank arms at window 20. The fine-tune
# scores 0.201 on failure_heldout and the l24 bank 0.555, and the honest reply to
# "you handicapped the learned baseline" is currently "probably not, but we did
# not check". This checks:
#
#   ft        the fine-tune driving alone, no filter at all
#   bf_w10    fine-tune as the alternative, window 10   <- the published number
#   bf_w20    fine-tune as the alternative, window 20   <- MATCHED to the bank arms
#   b_l24     the bank, for reference, in the same paired job
#
# across all THREE seeds of the stopping run, so the comparison also carries a
# seed spread rather than resting on one checkpoint. Recall the protocol's own
# resolution limit: seed spread is ~0.03, so a difference smaller than that is
# not a difference.
#
# The window is not a free parameter for the fine-tune the way it is for a bank:
# a longer window declares more replans violations (more chances to intervene)
# but each candidate must stay clean over a longer horizon (harder to qualify).
# Which way that nets out is exactly what is unmeasured.
#
# COST: 3 checkpoints x 2 shards; the first checkpoint's tasks also carry the
# checkpoint-independent arms. Budget like the l24 ladder, ~1 h per task.
#
# Usage (from repo root, ON RHEA):
#   DFT_SELECT_SMOKE=1 bash rl/diffusion_ft/slurm/submit_ft_parity.sh
#   bash rl/diffusion_ft/slurm/submit_ft_parity.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"; cd "$ROOT"

L24="lat0.1,lat-0.1,lat0.175,lat-0.175,lat0.25,lat-0.25,lat0.325,lat-0.325,\
lat0.4,lat-0.4,lat0.5,lat-0.5,lat0.6,lat-0.6,lat0.75,lat-0.75,\
lat0.9,lat-0.9,lat1.1,lat-1.1,lat1.35,lat-1.35,lat1.6,lat-1.6"

export DFT_SELECT_BANKS="l24=${L24}"
export DFT_SELECT_BANK="$L24"
# Checkpoint-independent arms run once, on the first checkpoint's tasks.
export DFT_BASE_ARMS="base,b_l24=bank@l24:cons:20"
# Per-checkpoint arms: the fine-tune alone, and as the alternative at BOTH windows.
export DFT_FT_ARMS="${DFT_FT_ARMS:-ft,bf_w10=ft:cons:10,bf_w20=ft:cons:20}"
RUN=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/stopping/20260911_160658_962947
export DFT_SELECT_CKPTS="${DFT_SELECT_CKPTS:-$RUN/base_s0/iter_000110 \
$RUN/base_s1/iter_000110 $RUN/base_s2/iter_000110}"

echo "fine-tune parity: ft alone | ft:cons:10 | ft:cons:20 | bank l24:cons:20"
echo "checkpoints: $DFT_SELECT_CKPTS"
echo
exec bash rl/diffusion_ft/slurm/submit_select_eval.sh
