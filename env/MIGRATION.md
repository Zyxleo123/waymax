# Moving waymax_rs off autonlab (via Backblaze B2)

Written 2026-10-18 on autonlab, before its outage. Everything the work needs is in Backblaze B2
at `b2:proteina/waymax_rs`, and the code is on GitHub. Any machine with `rclone` and `git` can
restore it. Section 0 is the upload, done on autonlab. Sections 1-7 are for **whoever (or
whichever coding agent) sets up the new machine**. Every tool named here is in `env/` on branch
`posthoc-opt-es`.

## Layout

Choose two directories on the new machine:

| Variable | What goes there | Example (Babel) |
|---|---|---|
| `CODE_HOME` | the repo + 5 worktrees, and their `logs/` (small) | `/home/yixiz` |
| `NEWROOT` | data, checkpoints, results, conda env (~30 GB, plus ~20 GB env) | `/data/user_data/yixiz/autonlab` |

Data keeps its full autonlab path under `NEWROOT`. For example,
`/zfsauton/scratch/yixiz/waymax_rs/x` becomes `$NEWROOT/zfsauton/scratch/yixiz/waymax_rs/x`.
A single prefix rewrite (`env/relocate_paths.sh`, section 5) then repoints ~400 files: code,
`#SBATCH` lines, split JSONs, run env files and symlinks. `/zfsauton2/home/yixiz/...` goes to
`$CODE_HOME/...`, and every other `/zfsauton*` path goes under `$NEWROOT`.

## What is in the bucket

| Object | Contents |
|---|---|
| `archives/scratch.tar.gz` | yixiz's scratch tree `zfsauton/scratch/yixiz/waymax_rs` (results, split JSONs, WOMD offset indexes, ES subsets and run roots; ~160k files), `ScenarioMaxWaymoFailures`, and `zfsauton2/home/yixiz/waymax_rs/logs` |
| `archives/claude.tar.gz` | `CLAUDE.md` and `projects/-zfsauton2-home-yixiz-waymax-rs/memory` from `~/.claude` |
| `archives/SHA256SUMS` | checksums of the two archives |
| `files/zfsauton/datasets/womd/tf_example/validation/` | WOMD v1.3.1 validation shards 0-9 (val10, 7.8 GB) |
| `files/zfsauton/scratch/yixiz/waymax_rs/womd_compact/valfail1568/` | the new goal-cond policy's 1568 validation failures as a compact copy (4.2 GB, see section 4) |
| `files/zfsauton/scratch/mineuih/...` | policy checkpoints: the 0616 base (o_am), `goal_cond_diffusion_policy/epoch_1000` (ES), and the 0711 ES policy; plus the failure list `exp/validation_0_150_failure_scenarios.json` |
| `files/zfsauton/scratch/{sbellad,ishanj}/...` | MF world models (Sagar's `latest.pt` and `epoch_0450.pt`, Ishan's `epoch_0450.pt`), the ES discriminator `best.pt`, and its split JSON |

## 0. Upload (autonlab)

```bash
cd /zfsauton2/home/yixiz/waymax_rs && bash env/b2_upload.sh     # DEST defaults to b2:proteina/waymax_rs
```

**Uploaded and verified 2026-10-19 00:36.** The archives' SHA256 checked out; `rclone check`
reported 3/3 archive objects and 434/434 bulk files matching; and the 5 files under other
users' unlistable scratch roots (world models, discriminator) were sent one by one and matched
by SHA1. The archives date from 00:14 and won't hold ES val10 results written after that.
**Before the outage, re-run `bash env/b2_upload.sh` without `REUSE_ARCHIVES`.**
The `proteina` bucket also holds ~20 GiB of the Proteina/CPSea project (`cpsea/`), and the B2
storage cap was raised for this upload. **Never `rclone sync`, `delete` or `cleanup` at the
bucket root:** that would touch the other project's data. Only ever write under `waymax_rs/`.

The script builds the archives on lov4's local disk and uploads them. It then uploads the plain
files: in bulk, except files under other users' unlistable scratch roots, which go one by one.
Finally it verifies everything (`rclone check`, plus SHA1 for the one-by-one files). On a re-run
the plain files are incremental and the archives are rebuilt. The bucket holds 442 objects,
19.1 GiB.

## 1. Download (new machine)

You need `rclone` with a B2 remote named `b2`. Create it with `rclone config`: a new remote, type
`b2`, using yixiz's B2 keyID and applicationKey. Or copy `~/.config/rclone/rclone.conf` over a
secure channel. That file holds the key, so never commit or upload it.

```bash
SRC=b2:proteina/waymax_rs
NEWROOT=/abs/data/dir/autonlab; CODE_HOME=/abs/code/dir      # see Layout
mkdir -p $NEWROOT/_b2
rclone copy  $SRC/files $NEWROOT/ --transfers 16 -P
rclone check $SRC/files $NEWROOT/ --one-way --fast-list         # must report 0 differences
rclone copy  $SRC/archives $NEWROOT/_b2 -P
(cd $NEWROOT/_b2 && sha256sum -c SHA256SUMS)                     # both must say OK
tar -C $NEWROOT -xzf $NEWROOT/_b2/scratch.tar.gz                  # lands as $NEWROOT/zfsauton/...
mkdir -p ~/claude_carry && tar -C ~/claude_carry -xzf $NEWROOT/_b2/claude.tar.gz
rm -r $NEWROOT/_b2                                                # once the extracts are in place
```

---

The rest is for **the coding agent on the new machine**. Set these once and use them throughout:

```bash
NEWROOT=...; CODE_HOME=...                                       # as in section 1
E=$NEWROOT/zfsauton/scratch/mineuih/conda_envs/waymax_rs        # the env path the code expects
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
- `myfree`: it doesn't exist elsewhere. Use `sinfo`.
- Everything about rhea and lov4.
- The autonlab partition, QOS and node names.
- The claim that `sacct` doesn't work. Test it on the new cluster before relying on it.

## 2. Code

```bash
cd $CODE_HOME
git clone git@github.com:Zyxleo123/waymax.git waymax_rs && cd waymax_rs   # needs a GitHub key: ssh -T git@github.com
git checkout posthoc-opt-es                                   # o_am / diffusion_ft work
git worktree add ../waymax_rs_es_newmodel es_wm_newmodel      # the live ES new-model work
git worktree add ../waymax_rs_es_disc     es_wm_disc
git worktree add ../waymax_rs_es_posthoc  es_wm_posthoc
git worktree add ../waymax_rs_es_idmfix   es_wm_idmfix
git worktree add ../waymax_rs_amfix       amfix
git remote rename origin myorigin
git remote add origin git@github.com:mini-hong/waymax_rs.git  # the shared repo
mkdir -p logs && cp -a $NEWROOT/zfsauton2/home/yixiz/waymax_rs/logs/. logs/
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

**GPU:** you need at least 24 GB and Ampere or newer. On autonlab this jaxlib failed on V100
(at cuDNN autotune) and ran out of memory on a 2080 Ti. Blackwell (e.g. RTX PRO 6000) is
untested. Check that jax sees a GPU **in a job**:

```bash
sbatch -p <gpu partition> --gres=gpu:1 --time=10:00 --output=logs/jax_check-%j.out \
  --wrap "hostname; nvidia-smi -L; $E/bin/python -c 'import jax; print(jax.devices())'"
```

## 4. Check the data

Both checks are CPU jobs, run from `$CODE_HOME/waymax_rs`. Both must print PASS. They are in
addition to the download checks in section 1.

```bash
# val shards 0-9: sizes vs env/womd_sizes.tsv, record chains vs the cached offset indexes
sbatch -p <partition> env/verify_womd.sbatch NEWROOT=$NEWROOT TRAIN_EXPECT=0 VAL_EXPECT=10
# the compact failure set: TF re-read, per-record sha256, placeholders empty
sbatch -p <partition> --mem=8G --time=1:00:00 --output=logs/compact_verify-%j.out --wrap \
  "hostname; cd $PWD && CUDA_VISIBLE_DEVICES= $E/bin/python -m rl.diffusion_ft.tools.build_compact_shards verify --out_root $NEWROOT/zfsauton/scratch/yixiz/waymax_rs/womd_compact/valfail1568"
```

Scene ids resolve through offset indexes keyed by shard basename only. A WOMD copy whose bytes
differ is therefore read wrongly **with no error**. Use the shards from the bucket. If you ever
take them from Waymo's GCS instead (`gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/`),
the check must pass before you use them, and never rebuild the offsets to make it pass.

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
V="CODE_HOME=$CODE_HOME GPU_PARTITION=<gpu partition> CPU_PARTITION=<cpu partition> DROP_QOS=1 DROP_NODE_PINS=1 GPU_GRES=gpu:1"
env $V bash env/relocate_paths.sh $NEWROOT --dry-run $C
env $V bash env/relocate_paths.sh $NEWROOT $C
for d in $C; do git -C $d commit -qam "Relocate paths to <cluster>"; done   # revert this commit to go back
```

What each switch does:
- `GPU_PARTITION` / `CPU_PARTITION` remap the `#SBATCH --partition=general|cpu` lines.
- `DROP_QOS` deletes the autonlab `#SBATCH --qos` lines.
- `DROP_NODE_PINS` removes autonlab node names (`--exclude=gpu28`, `${ES_EXCLUDE-gpu30}`).
- `GPU_GRES` rewrites every 1-GPU `--gres` line. GRES type names are case-sensitive and differ
  per cluster: autonlab said `a6000`, Babel says `A6000`. Use `gpu:1` when every GPU in the
  partition is ≥24 GB and Ampere or newer.

The Babel values were tested on a mock tree on autonlab. Afterwards:
- no `/zfsauton` path was left
- code paths were under `CODE_HOME` and data paths under `NEWROOT`
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
array skips every scene that already has a result (`--start_from`). Different GPU models cause
mm-level drift, the same size as run-to-run noise.

```bash
cd $CODE_HOME/waymax_rs_es_newmodel
R=$NEWROOT/zfsauton/scratch/yixiz/waymax_rs/es_posthoc/runs/nm_val10_20261018_034021_870941
J=$(sbatch --parsable -p <gpu partition> --cpus-per-task=16 --mem=40G --array=0-27 slurm/sim/es_disc.sbatch $R/env)
sbatch -p <cpu partition> --dependency=afterany:$J slurm/sim/es_disc_cpu.sbatch $R/compare.env
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

Claude Code names the memory folder after the repo path, with every non-alphanumeric character
turned into `-`:

```bash
K=$(echo "$CODE_HOME/waymax_rs" | sed 's|[^A-Za-z0-9]|-|g')      # /home/yixiz/waymax_rs -> -home-yixiz-waymax-rs
mkdir -p ~/.claude/projects/$K && cp -r ~/claude_carry/projects/-zfsauton2-home-yixiz-waymax-rs/memory ~/.claude/projects/$K/
```

The memories cite autonlab paths. Read `/zfsauton2/home/yixiz/...` as `$CODE_HOME/...` and any
other `/zfsauton*` path as `$NEWROOT/zfsauton*`; job IDs and node names are autonlab history.
`~/claude_carry/CLAUDE.md` is yixiz's autonlab-era global file. Adapt it using the rules list
above before it becomes `~/.claude/CLAUDE.md`.

## If the target is Babel (CMU LTI)

**Done on Babel 2026-10-19** (node babel-t9-32, RTX A6000). What turned out to be true there,
which corrects the notes written from `sinfo` on 10-18:

- Layout: `CODE_HOME=/home/yixiz` (100 GB) and `NEWROOT=/data/user_data/yixiz/autonlab` (500 GB).
  `/data/user_data/yixiz` is mounted only on compute nodes with an active job (not on login), and
  AutoFS needs the full path.
- **yixiz may only use `preempt`**, and it requires a QOS: GPU jobs
  `-p preempt --qos=preempt_qos --gres=gpu:1` (pin `gpu:A6000:1`: preempt also has Blackwell
  RTX_PRO_6000, untested), CPU jobs `-p preempt --qos=preempt_cpu_qos` with no `--gres`.
  `general` rejects GPU-less jobs, and it is not ours to use. `preempt` requeues on preemption.
- Queue waits are long, so work runs directly on a compute node yixiz already holds (the IDE's
  node-hold job) where it fits. The download, env build and every check below ran that way.
- conda: Miniforge at `/data/user_data/yixiz/miniforge3`. The env was built from
  `env/conda_env.yml` with the `defaults` channel removed (37 min).
- `sacct` works on Babel.
- Relocation values used:
  `CODE_HOME=/home/yixiz GPU_PARTITION=preempt CPU_PARTITION=preempt DROP_QOS=1 GPU_QOS=preempt_qos CPU_QOS=preempt_cpu_qos DROP_NODE_PINS=1 GPU_GRES=gpu:A6000:1`.
  After that, by hand: the `DFT_PARTITION`/`GPU_PARTITION`/`PARTITION` defaults in 4 submitters
  went to `preempt`, `--qos=qos_preempt` in `submit_es_disc.sh` became `preempt_qos`, and
  `V-Max/slurm/sdc_probe/probe_sdc_index.sbatch` got a preempt/`preempt_cpu_qos` header.
- Results: section 1 reported 0 differences on 439 files and both archive checksums OK; jax sees the A6000;
  `verify_womd` PASS (10/10); the compact set PASS (150/150 shards, 1568 scenes); after the
  relocation no `/zfsauton` path is left, and a second run changes nothing. The val10 run's env
  files resolve. Broken symlinks (~6.5k) all point at things that were not copied (training
  shards, wandb debug logs, V-Max-era SAC checkpoints).

## Deliberately not copied

- WOMD training shards (1.3 TB): not needed for evaluation. If they are needed later, take a
  copy and pass `env/verify_womd.sbatch` before using it.
- `mineuih/waymax_rs/lane_graphs` (58 GB): the ES runner's lane-graph loader is commented out.
- `mineuih/waymax_rs/sim_state_cache_npz` (120 GB): used only by annotation and legacy code.
- `yixiz/ScenarioMaxWaymo/training.tfrecord` (930 GB): from the SB3/V-Max era.
- The conda envs (rebuilt) and `.jax_compilation_cache` (it is GPU-specific).

The checkpoints in the bucket belong to mineuih, sbellad and ishanj; tell them you took a copy.
