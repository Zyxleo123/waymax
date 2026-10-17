#!/bin/bash
# ES simulations on WOMD validation shards 00000-00009 (2916 scenes; 2895 run).
# Flags: see `bash slurm/sim/val10.sh --help`. Comma lists expand.
#
#   bash slurm/sim/val10.sh --planner diffusion,fm --forecast gt,cv,wm_mf
#   bash slurm/sim/val10.sh ... --other_agents idm --dry_run
exec bash "$(dirname "${BASH_SOURCE[0]}")/_submit.sh" --subset val10 "$@"
