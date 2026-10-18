# Moving waymax_rs off autonlab

Written 2026-10-18 before the autonlab outage. All the tools named here are in `env/` on
branch `posthoc-opt-es`.

The scheme: copy every needed tree under **one root on the new cluster, keeping its full
autonlab path** (`$NEWROOT/zfsauton/scratch/...`, `$NEWROOT/zfsauton2/home/yixiz/...`).
Then a single prefix rewrite (`env/relocate_paths.sh`) fixes the hard-coded paths: about 400
files of code and #SBATCH lines, the split JSONs, the run env files and the ES-subset symlinks.

| What | Size | How it moves |
|---|---|---|
| Code, 10 branches + 5 worktrees | — | git: `Zyxleo123/waymax` has every branch |
| Your scratch, ckpts, WMs, disc (`env/transfer_list.txt`) | ~14 GB | rsync **now**, again just before shutdown |
| conda env | — | rebuild from `env/conda_env.yml` |
| WOMD v1.3.1 tf_example | 1.3 TB train + 7.8 GB val (shards 0-9) | GCS download, then `env/verify_womd.sbatch` |
| Claude memory | 1 MB | rsync |

## Target: Babel (CMU LTI)

Taken from the public LTI Slurm page and search snippets of the Babel wiki. The wiki itself
needs CMU login, so check these against `sinfo` / `scontrol show partition` on Babel.

- **Storage.** `/home/<andrewid>` holds 100 GB. `/data/user_data/<andrewid>` holds **500 GB by
  default**. `/data/group_data`, `/data/datasets` and `/data/models` are AutoFS mounts, so always
  use full paths: `ls /data/user_data` can come back empty on a compute node. Use
  `NEWROOT=/data/user_data/<andrewid>/autonlab`.
- **The quota is the main constraint.** Transfer list ~14 GB + conda env ~20 GB + the ES WOMD
  shards 251 GB ≈ 285 GB, which fits. Full WOMD for the o_am splits (1.3 TB) does **not** fit.
  First check `/data/datasets/` for a WOMD copy: if one exists, symlink its `tf_example` to
  `$NEWROOT/zfsauton/datasets/womd/tf_example` and verify it (section 3). Otherwise ask LTI help
  for more `user_data` or a `group_data` allocation before moving the o_am line.
- **Partitions.** `general`: 2-day max, 6 h default, sbatch only, normal QoS, up to 8 GPUs.
  `long`: 7-day max; use it for multi-day `train_safety` runs. `cpu`: 2-day max. There are also
  `preempt` and `debug` (short, no arrays). The names `general` and `cpu` match autonlab, but the
  autonlab QOS names (`qos_general`, `qos_cpu`, `qos_preempt`) do not exist, so drop them.
- **GPUs.** Request them as `--gres=gpu:<type>:1`. Get the type names with
  `sinfo -h -o "%P %G %N" | sort -u` and pick a ≥24 GB Ampere-or-newer type (section 2).
  Unpinned `--gres=gpu:1` may land on a card this env cannot use.
- **Relocation values** (section 4): `GPU_PARTITION=general CPU_PARTITION=cpu DROP_QOS=1 DROP_NODE_PINS=1 GPU_GRES=gpu:<type>:1`.
- The wiki lists Slurm 20.11; arrays, `--parsable` and `afterok`/`afterany` all work there.

## 0. Before autonlab goes down

Run these from lov4, inside `tmux`. `NEW=user@newhost` and `NEWROOT=/abs/path` are on the new cluster.

```bash
cd /zfsauton2/home/yixiz/waymax_rs
NEW=user@newhost NEWROOT=/abs/path
ssh $NEW "mkdir -p $NEWROOT"
# ~14 GB. Rerun the same line just before the outage: it then sends only what changed,
# which picks up the ES runs' latest per-scene results.
rsync -a -r -z --partial --info=progress2 --exclude=.jax_compilation_cache \
  --files-from=env/transfer_list.txt / $NEW:$NEWROOT/
# Claude memory and global instructions
rsync -a ~/.claude/CLAUDE.md ~/.claude/projects/-zfsauton2-home-yixiz-waymax-rs/memory $NEW:claude_carry/
```

**WOMD from here.** On Babel, copy the ES shards now: 251 GB fits the default quota, and both
clusters are on the CMU network. Leave the full training set until the quota question is
settled. ZFS stores WOMD at about 5x compression; if the copy runs slower than the link
allows, drop `-z`.

```bash
# just the shards the ES subsets use (fail1500: 187 training shards, 243 GB; val10: 10 val shards)
find -P /zfsauton/scratch/yixiz/waymax_rs/es_posthoc/{fail1500_subset_g2,val10_subset_x5} -type l -printf '%l\n' | sort -u > ~/es_shards.txt
rsync -a -z --partial --info=progress2 --files-from=$HOME/es_shards.txt / $NEW:$NEWROOT/
# all 1000 training shards (the o_am split JSONs draw scenes from every shard)
rsync -a -R -z --partial --info=progress2 /zfsauton/datasets/womd/tf_example/training $NEW:$NEWROOT/
```

## 1. Code

```bash
mkdir -p $NEWROOT/zfsauton2/home/yixiz && cd $NEWROOT/zfsauton2/home/yixiz
git clone git@github.com:Zyxleo123/waymax.git waymax_rs && cd waymax_rs
git checkout posthoc-opt-es                                       # o_am / diffusion_ft work
git worktree add ../waymax_rs_es_newmodel es_wm_newmodel          # live ES new-model runs
git worktree add ../waymax_rs_es_disc     es_wm_disc
git worktree add ../waymax_rs_es_posthoc  es_wm_posthoc
git worktree add ../waymax_rs_es_idmfix   es_wm_idmfix
git worktree add ../waymax_rs_amfix       amfix
git remote rename origin myorigin
git remote add origin git@github.com:mini-hong/waymax_rs.git     # shared repo, over SSH
```

## 2. Environment

The working env was `mineuih/conda_envs/waymax_rs`; `sbellad/conda_envs/waymax_rs` is the same
stack. Both are Python 3.12.13, jax/jaxlib 0.10.0 + cuda12 plugin, flax 0.12.7, torch 2.11,
tf 2.21, CUDA 12.8.1, and waymax from git at `a64dfec`. Build the env once at the path the
code expects, then point the second path at it:

```bash
E=$NEWROOT/zfsauton/scratch/mineuih/conda_envs/waymax_rs
conda env create -p $E -f env/conda_env.yml
mkdir -p $NEWROOT/zfsauton/scratch/sbellad/conda_envs && ln -s $E $NEWROOT/zfsauton/scratch/sbellad/conda_envs/waymax_rs
```

If the conda solve fails, there is a fallback: a Python 3.12 venv at `$E` plus
`pip install -r env/pip_freeze.txt`. The jax and torch cu12 wheels bring their own CUDA. The
exact conda build list is in `env/conda_explicit.txt`.

**GPU:** you need at least 24 GB and Ampere or newer (A100, H100, A6000, L40S). This jaxlib
fails on V100 at cuDNN autotune, and a 2080 Ti runs out of memory with no traceback. Check with
`python -c "import jax; print(jax.devices())"` **inside a GPU job**.

## 3. WOMD

The code parses WOMD as `WOD_1_3_1`. Accept the licence at waymo.com/open with your Google
account, run `gcloud auth login`, then:

```bash
D=$NEWROOT/zfsauton/datasets/womd/tf_example; mkdir -p $D/training $D/validation
B=gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example
gcloud storage cp "$B/training/*" $D/training/
gcloud storage cp "$B/validation/validation_tfexample.tfrecord-0000[0-9]-of-00150" $D/validation/
```

**Then verify. This is the one step you must not skip.** Scene ids resolve through the cached
offset indexes, which are keyed by shard basename only, so a copy that differs from the
autonlab one is read wrongly **with no error**.

```bash
mkdir -p logs && sbatch -p <cpu partition> env/verify_womd.sbatch NEWROOT=$NEWROOT
```

The job must print `verify_womd: PASS`. It compares each shard's size with `env/womd_sizes.tsv`
and checks each shard's record chain against `womd_offsets/`. If it fails, the GCS bytes are
not the autonlab bytes. In that case copy the shards from autonlab (section 0) rather than
rebuilding the offsets: rebuilt offsets would silently re-map every scene id in every split JSON.

## 4. Point the code at the new paths

```bash
cd $NEWROOT/zfsauton2/home/yixiz/waymax_rs
C=". ../waymax_rs_es_newmodel ../waymax_rs_es_disc ../waymax_rs_es_posthoc ../waymax_rs_es_idmfix ../waymax_rs_amfix"
bash env/relocate_paths.sh $NEWROOT --dry-run $C
# Babel: GPU_PARTITION=general CPU_PARTITION=cpu DROP_QOS=1 DROP_NODE_PINS=1 GPU_GRES=gpu:<type>:1
GPU_PARTITION=<gpu partition> CPU_PARTITION=<cpu partition> DROP_QOS=1 DROP_NODE_PINS=1 GPU_GRES=gpu:<type>:1 \
  bash env/relocate_paths.sh $NEWROOT $C
for d in $C; do git -C $d commit -qam "Relocate paths to <cluster>"; done   # one commit each; revert it to go back
```

The script is safe to run twice, and it skips `env/` and the byte-identical vendored trees.
It changes four kinds of lines:
- `#SBATCH --partition=general|cpu` lines, remapped
- `#SBATCH --qos` lines, deleted
- autonlab node pins, dropped (`--exclude=gpu28`, `${ES_EXCLUDE-gpu30}` and similar)
- 1-GPU `--gres` lines, rewritten to `GPU_GRES`

Check what's left by hand. The ES submitter's optional preempt path hard-codes
`--qos=qos_preempt`; it only runs when `ES_PREEMPT_IDX` is set.

```bash
git grep -nE -- '--(exclude|nodelist|qos)=|--partition=preempt|gres=gpu' -- '*.sh' '*.sbatch' ':!es_baseline' | grep -v '^env/'
```

Then follow the global rules again: test the preamble with `env -i`, and read the first job's `.out` before trusting any chain.

## 5. Resume what was running

The ES new-model runs (val10 88078/88079, fail1500 88081/88082, compares 88080/88083) were
part-way done at the outage. The runner gets `--start_from <out_dir>`, so resubmitting the same
array skips every scene that already has a result:

```bash
cd $NEWROOT/zfsauton2/home/yixiz/waymax_rs_es_newmodel
for R in $NEWROOT/zfsauton/scratch/yixiz/waymax_rs/es_posthoc/runs/nm_{val10_20261018_034021_870941,fail1500_20261018_034021_871142}; do
  J=$(sbatch --parsable --cpus-per-task=16 --mem=40G --array=0-27 slurm/sim/es_disc.sbatch $R/env)
  sbatch --dependency=afterany:$J slurm/sim/es_disc_cpu.sbatch $R/compare.env
done
```

## 6. Claude memory

Claude Code names a project's memory folder after the repo path, with every
non-alphanumeric character turned into `-`:

```bash
K=$(echo "$NEWROOT/zfsauton2/home/yixiz/waymax_rs" | sed 's|[^A-Za-z0-9]|-|g')
mkdir -p ~/.claude/projects/$K && cp -r ~/claude_carry/memory ~/.claude/projects/$K/
```

`~/claude_carry/CLAUDE.md` holds autonlab-specific rules (rhea, lov4, `myfree`, `sacct`, the
partition names). Edit it for the new cluster before copying it to `~/.claude/CLAUDE.md`.

## Deliberately not copied

- `mineuih/waymax_rs/lane_graphs` (58 GB). The ES runner's lane-graph loader is commented out.
- `mineuih/waymax_rs/sim_state_cache_npz` (120 GB). Used only by annotation and legacy code.
- `yixiz/ScenarioMaxWaymo/training.tfrecord` (930 GB). From the SB3/V-Max era.
- The conda envs (rebuilt in section 2) and `.jax_compilation_cache` (it is GPU-specific).
- Every pre-09-28 checkpoint under `$ZFS/waymax_rs`. Those were already deleted.

The checkpoints in the transfer list belong to mineuih, sbellad and ishanj; tell them you took a copy.
