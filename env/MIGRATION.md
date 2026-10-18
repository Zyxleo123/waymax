# Moving waymax_rs from autonlab to Babel

Written 2026-10-18 on autonlab, before its outage. It covers two jobs: what yixiz runs on
autonlab before it goes down (section 0), and what a coding agent on Babel does afterwards
(sections 1-7). Every tool named here is in `env/` on branch `posthoc-opt-es`.

## Layout on Babel

| What | Where | Notes |
|---|---|---|
| Code: repo + 5 worktrees, and their `logs/` | `/home/yixiz/waymax_rs`, `/home/yixiz/waymax_rs_*` | `CODE_HOME=/home/yixiz` (100 GB quota) |
| Data, checkpoints, results, conda env | `NEWROOT=/data/user_data/yixiz/autonlab` | 500 GB quota; every autonlab path is kept under it |

Data is mirrored with its full autonlab path. For example,
`/zfsauton/scratch/yixiz/waymax_rs/...` becomes
`/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/...`. A single prefix rewrite
(`env/relocate_paths.sh`, section 5) then repoints ~400 files: code, `#SBATCH` lines, split
JSONs, run env files and symlinks. `/zfsauton2/home/yixiz/...` goes to `/home/yixiz/...`;
every other `/zfsauton*` path goes under `NEWROOT`.

What moves, ~26 GB in one rsync of `env/transfer_list.txt`:
- your scratch, about 16 GB: results, split JSONs, WOMD offset indexes, ES subsets and run roots
- `womd_compact/valfail1568`: the 1568 new validation failures as a compact copy (4.2 GB, see below)
- WOMD validation shards 0-9 (7.8 GB)
- policy checkpoints (mineuih), world models (sbellad, ishanj), the ES discriminator (sbellad)
- `logs/` (28 MB)

The code comes from GitHub, where `git@github.com:Zyxleo123/waymax.git` (private) holds every
branch. The conda env is rebuilt from `env/conda_env.yml`.

## Babel facts

From `sinfo` on login3, 2026-10-18:

- **`general`** has GPU nodes only: 6000Ada, A100_80GB, A6000, L40 and L40S. Every one is
  ≥48 GB and Ampere or Ada, so **`--gres=gpu:1` is safe on `general`**.
- **`preempt` is the default partition** (`*`). Always pass `-p general` or `-p preempt`
  explicitly. `preempt` adds A100_40GB, H100, H200, **RTX_PRO_6000 (Blackwell, untested with
  this jaxlib)** and some GPU-less nodes. Pin a type there, e.g. `--gres=gpu:A6000:1` or
  `gpu:L40S:1`, and pair it with `--requeue` on resumable jobs.
- **There is no `cpu`, `long` or `debug` partition.** CPU-only jobs (analysis, plots, checks)
  go to `general` without `--gres`. Confirm that is allowed, and the time limit, with
  `scontrol show partition general` (look for `MaxTime`). The ES job asks for `--time=2-00:00:00`.
- **GRES type names are case-sensitive.** Babel uses `A6000`, where autonlab scripts used
  `a6000`. The relocation step rewrites all 1-GPU `--gres` lines.
- The lab partitions (neulab, katef, dlab, ...) are not ours.
- Storage is AutoFS, so always use full paths: `ls /data/user_data` can come back empty, but
  `/data/user_data/yixiz` works.

## 0. On autonlab, before it goes down (yixiz)

Run from lov4, inside `tmux`:

```bash
cd /zfsauton2/home/yixiz/waymax_rs
NEW=yixiz@<babel login host>; NEWROOT=/data/user_data/yixiz/autonlab
ssh $NEW "mkdir -p $NEWROOT"
# ~26 GB. Rerun just before the outage: it then sends only what changed
# (the ES runs' latest per-scene results).
rsync -a -r -z --partial --info=progress2 --exclude=.jax_compilation_cache \
  --files-from=env/transfer_list.txt / $NEW:$NEWROOT/
rsync -a ~/.claude/CLAUDE.md ~/.claude/projects/-zfsauton2-home-yixiz-waymax-rs/memory $NEW:claude_carry/
```

---

The rest is for **the coding agent on Babel**. Set these once and use them throughout:

```bash
NEWROOT=/data/user_data/yixiz/autonlab
CODE_HOME=/home/yixiz
E=$NEWROOT/zfsauton/scratch/mineuih/conda_envs/waymax_rs      # the env path the code expects
```

**Rules.** yixiz's global instructions were written for autonlab. These still apply:
- All compute goes through `sbatch`, never the login node. That includes checks and plots.
- Split jobs: GPU work on GPUs, and anything that only reads saved artefacts in a CPU job.
- Draw figures from files on disk.
- Pass config as a positional env-file or `VAR=value` argument, never `sbatch --export`.
- Put `set -euo pipefail` in the job body and locate the repo via `$SLURM_SUBMIT_DIR`.
- Every job prints `hostname` and `nvidia-smi -L`.
- Decision gates live inside jobs; long jobs are resumable.

These do **not** apply:
- `myfree`: it doesn't exist here. Use `sinfo -p general -o "%n %G %t %C"`.
- Everything about rhea and lov4: submit from the Babel login node directly.
- The autonlab partition, QOS and node names.
- The claim that `sacct` doesn't work. That was autonlab; test it here before relying on it.

## 1. Preflight

Stop and ask yixiz if any of these fail:

```bash
ls $NEWROOT/zfsauton/scratch/yixiz/waymax_rs/womd_compact/valfail1568/target_scenarios.json   # the rsync landed
ls $NEWROOT/zfsauton/datasets/womd/tf_example/validation | wc -l                            # 10
ssh -T git@github.com            # "Hi Zyxleo123!" — otherwise a key must be added to GitHub
command -v conda || module avail 2>&1 | grep -i conda
```

## 2. Code

```bash
cd $CODE_HOME
git clone git@github.com:Zyxleo123/waymax.git waymax_rs && cd waymax_rs
git checkout posthoc-opt-es                                   # o_am / diffusion_ft work
git worktree add ../waymax_rs_es_newmodel es_wm_newmodel      # the live ES new-model work
git worktree add ../waymax_rs_es_disc     es_wm_disc
git worktree add ../waymax_rs_es_posthoc  es_wm_posthoc
git worktree add ../waymax_rs_es_idmfix   es_wm_idmfix
git worktree add ../waymax_rs_amfix       amfix
git remote rename origin myorigin
git remote add origin git@github.com:mini-hong/waymax_rs.git  # the shared repo
mkdir -p logs && rsync -a $NEWROOT/zfsauton2/home/yixiz/waymax_rs/logs/ logs/
```

## 3. Environment

The stack: Python 3.12.13, jax/jaxlib 0.10.0 with the cuda12 plugin, flax 0.12.7, torch 2.11,
tf 2.21, CUDA 12.8.1, and waymax from git at `a64dfec`. sbellad's env path is the same stack,
so link it to the one env:

```bash
conda env create -p $E -f env/conda_env.yml
mkdir -p $NEWROOT/zfsauton/scratch/sbellad/conda_envs && ln -s $E $NEWROOT/zfsauton/scratch/sbellad/conda_envs/waymax_rs
```

If the solve fails, fall back to a Python 3.12 venv at `$E` plus
`pip install -r env/pip_freeze.txt`; the jax and torch cu12 wheels bring their own CUDA. The
exact conda build list is in `env/conda_explicit.txt`.

Then check that jax sees a GPU, **in a job**:

```bash
sbatch -p general --gres=gpu:1 --time=10:00 --output=logs/jax_check-%j.out \
  --wrap "hostname; nvidia-smi -L; $E/bin/python -c 'import jax; print(jax.devices())'"
```

## 4. Check the data

Both checks are CPU jobs, run from `$CODE_HOME/waymax_rs`. Both must print PASS.

```bash
# val shards 0-9: sizes vs env/womd_sizes.tsv, record chains vs the cached offset indexes
sbatch -p general env/verify_womd.sbatch NEWROOT=$NEWROOT TRAIN_EXPECT=0 VAL_EXPECT=10
# the compact failure set: TF re-read, per-record sha256, placeholders empty
sbatch -p general --mem=8G --time=1:00:00 --output=logs/compact_verify-%j.out --wrap \
  "hostname; cd $PWD && CUDA_VISIBLE_DEVICES= $E/bin/python -m rl.diffusion_ft.tools.build_compact_shards verify --out_root $NEWROOT/zfsauton/scratch/yixiz/waymax_rs/womd_compact/valfail1568"
```

Scene ids resolve through offset indexes keyed by shard basename only. A WOMD copy whose bytes
differ is therefore read wrongly **with no error**.

If val shards 0-9 are missing, fetch them from Waymo: accept the licence at waymo.com/open,
run `gcloud auth login`, then
`gcloud storage cp -n "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord-0000[0-9]-of-00150" $NEWROOT/zfsauton/datasets/womd/tf_example/validation/`.
Then re-check. If a downloaded copy fails the check, don't use it, and don't rebuild the
offsets either: that would silently re-map scene ids.

**About the compact failure set.** The new policy's failures touch all 150 validation shards
(118 GB). So `tools/build_compact_shards.py` kept each shard's real name and record count:
failure records are byte-identical, and every other record is an empty 16-byte frame. The ES
runner's loader enumerates raw records and parses only the requested indices, so it sees the
same scene at the same index. This was checked on autonlab: 12 scenes loaded bit-identically,
and placeholders were refused. **Never point anything that parses every record of a shard at
that directory**, e.g. a splits JSON over "all scenes".

## 5. Relocate paths

```bash
cd $CODE_HOME/waymax_rs
C=". ../waymax_rs_es_newmodel ../waymax_rs_es_disc ../waymax_rs_es_posthoc ../waymax_rs_es_idmfix ../waymax_rs_amfix"
V="CODE_HOME=$CODE_HOME GPU_PARTITION=general CPU_PARTITION=general DROP_QOS=1 DROP_NODE_PINS=1 GPU_GRES=gpu:1"
env $V bash env/relocate_paths.sh $NEWROOT --dry-run $C
env $V bash env/relocate_paths.sh $NEWROOT $C
for d in $C; do git -C $d commit -qam "Relocate paths to Babel"; done   # revert this commit to go back
```

These exact values were tested on a mock tree on autonlab. Afterwards:
- no `/zfsauton` path was left
- code paths were in "home" and data paths in "NEWROOT"
- every `#SBATCH --partition` was `general` and every `--gres` was `gpu:1`
- a second run changed nothing, and every touched script parsed

The script skips `env/` and the byte-identical vendored trees. Check what's left by hand: the ES
submitter's optional preempt path hard-codes `--partition=preempt --qos=qos_preempt`, but it
only runs when `ES_PREEMPT_IDX` is set.

```bash
git grep -nE -- '--(exclude|nodelist|qos)=|--partition=|gres=gpu:[A-Za-z]' -- '*.sh' ':!es_baseline' ':!env'
```

Then test the preamble with `env -i`, and read the first job's `.out` before trusting any chain.

## 6. Resume and continue

**val10 ES run.** The new-model val10 run was part-done at the outage. Resubmitting the same
array skips every scene that already has a result (`--start_from`). Expect some arms to finish
on Ada GPUs where autonlab used A6000; that drift is mm-level, like run-to-run noise.

```bash
cd $CODE_HOME/waymax_rs_es_newmodel
R=$NEWROOT/zfsauton/scratch/yixiz/waymax_rs/es_posthoc/runs/nm_val10_20261018_034021_870941
J=$(sbatch --parsable -p general --cpus-per-task=16 --mem=40G --array=0-27 slurm/sim/es_disc.sbatch $R/env)
sbatch -p general --dependency=afterany:$J slurm/sim/es_disc_cpu.sbatch $R/compare.env
```

The fail1500 run cannot resume, because its training shards stayed on autonlab.

**The new eval set** is `womd_compact/valfail1568/`: shards in `validation/`, plus
`target_scenarios.json` in the ES runner's format and `manifest.json`. The source list is
`mineuih/waymax_rs/exp/validation_0_150_failure_scenarios.json`, where index k means
`validation_tfexample.tfrecord-<k:05d>-of-00150`. It is not wired in yet. It needs:
- a subset directory like `es_posthoc/fail1500_subset_g2`, per group: a `target_scenarios.json`
  and `tfrecords/validation/` symlinks into the compact shards
- an `ES_SUBSET` case in `slurm/sim/submit_es_disc.sh`

## 7. Claude memory

```bash
mkdir -p ~/.claude/projects/-home-yixiz-waymax-rs && cp -r ~/claude_carry/memory ~/.claude/projects/-home-yixiz-waymax-rs/
```

The folder name is the repo path with every non-alphanumeric character turned into `-`. The
memories cite autonlab paths. Read `/zfsauton2/home/yixiz/...` as `/home/yixiz/...` and any
other `/zfsauton*` path as `$NEWROOT/zfsauton*`; job IDs and node names are autonlab history.
`~/claude_carry/CLAUDE.md` is yixiz's autonlab-era global file. Adapt it using the rules list
above before it becomes `~/.claude/CLAUDE.md`.

## Deliberately not copied

- WOMD training shards (1.3 TB): too big for the quota and not needed for evaluation. If they
  are needed later, check `/data/datasets/` first, then verify any copy with `env/verify_womd.sbatch`.
- `mineuih/waymax_rs/lane_graphs` (58 GB): the ES runner's lane-graph loader is commented out.
- `mineuih/waymax_rs/sim_state_cache_npz` (120 GB): used only by annotation and legacy code.
- `yixiz/ScenarioMaxWaymo/training.tfrecord` (930 GB): from the SB3/V-Max era.
- The conda envs (rebuilt) and `.jax_compilation_cache` (it is GPU-specific).

The checkpoints in the transfer list belong to mineuih, sbellad and ishanj; tell them you took a copy.
