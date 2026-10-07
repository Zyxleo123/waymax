#!/bin/bash
# ES simulations on WOMD training shards 00000+00001 and validation shards 00000+00001
# (both splits run in one job). Flags: see README.md. Comma lists expand.
#
#   bash slurm/sim/trainval.sh --planner diffusion --forecast cv,gt,wm_det,wm_mf
#   bash slurm/sim/trainval.sh --planner diffusion --forecast wm_mf --mf_mode top1,worst_case_independent
#   bash slurm/sim/trainval.sh ... --dry_run
exec bash "$(dirname "${BASH_SOURCE[0]}")/_submit.sh" --subset trainval "$@"
