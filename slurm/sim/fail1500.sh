#!/bin/bash
# ES simulations on the fail1500 subset (1500 FM-failure scenes, 187 shards).
# Flags: see README.md or `bash slurm/sim/fail1500.sh --help`. Comma lists expand.
#
#   bash slurm/sim/fail1500.sh --planner fm --forecast cv,wm_det,wm_mf --it 6 --rs 40
#   bash slurm/sim/fail1500.sh --planner diffusion --forecast wm_mf --mf_collision gaussian_uniform --M 4,8,16
#   bash slurm/sim/fail1500.sh ... --dry_run
exec bash "$(dirname "${BASH_SOURCE[0]}")/_submit.sh" --subset fail1500 "$@"
