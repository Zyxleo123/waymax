# `rl/diffusion_ft` — diffusion policy simulator-reward fine-tuning

Unified framework for fine-tuning the pretrained diffusion motion policy against
the **simulator** reward (filtered FT, DRWR, DAWR, preference, DPPO, hybrids,
ES + distillation). Deliberately separate from
`train_diffusion/train_diffusion_policy.py`, which only updates
instruction/subgoal/FiLM params. The outer action here is *one sampled
trajectory*.

## Phase 1 (this commit): one trustworthy environment

| Module          | What it is |
| --------------- | ---------- |
| `checkpoint.py` | Loads the architecture from checkpoint **metadata** (`metadata.json`), not config defaults (the dataclass/argparse defaults disagree). Exposes both raw params (collection/likelihoods) and EMA params (evaluation). |
| `reward.py`     | The single batched simulator-reward function. Reproduces `rl/waymax_env.py` step reward exactly, vectorized, pure. Every method consumes this scalar. |
| `env.py`        | `DiffusionWaymaxEnv`: batched, `StateDynamics` execution, real per-scene goal timestep, `log_trajectory` immutable (executed motion lives in `sim_trajectory`), trajectory-as-action (samples internally or accepts an external candidate for best-of-N / ES). |

Shared-code repairs that Phase 1 depends on:
- `data/preprocess.py`: `goal_step_override` now accepts a per-scene **array** (real goal step), not just a scalar.
- `planner/{diffusion,vla}_planner.py` + `simulation/planning_utils.py` + `simulation/runner.py`: removed hard-coded `goal_step_override=90`; threaded the real goal step.
- `simulation/runner.py`: fixed the post-goal metric window (was indexing a `[start_timestep:]`-sliced timeline with an absolute step) and the `video_paths` crash / mis-indexing when visualization is a subset or disabled.
- `scores/scorer.py`: `set_speed` weighted metric was dead code (scored under the name `speed`); fixed the name.

## Gate 1 (run before ANY fine-tuning)

```bash
bash rl/diffusion_ft/slurm/submit_gate1.sh          # from repo root, on rhea
# override scenes:  DFT_INDICES=0,1,2,3,4,5 bash rl/diffusion_ft/slurm/submit_gate1.sh
```

Six gates: expert-replay clean · batched==single · per-step sums to return ·
obs changes on ego deviation · permutation equivariance · frozen-ckpt
reproducible. Log lands in `logs/dft/slurm-dft_gate1-<jobid>.out`.

Interpreter: `/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python` (flax 0.12.7).
Base checkpoint: `.../pretrain_diffusion_without_subgoal_20260616_013642/latest`.

## Phase 2 — diffusion API extensions (Gate 2 PASS)

`model/diffusion/diffusion.py` (inference unchanged): `loss_per_example`,
`sample_with_trace` (frozen early / trainable late; records transitions), and
`transition_log_prob` (v→x0→posterior mean, floored std, averaged over channels +
executed prefix). `rl/diffusion_ft/rl_actor.py` = `DiffusionRLActor`: frozen base
denoiser + cloned trainable copy (structural gradient isolation), clipped-PPO
update with optional expert anchor. Tests: `test_gate2.py` (core, 4/4),
`test_gate2b.py` (RL-actor, 3/3) — tiny synthetic model, CPU only.

**Submit them; do not run them on a login node.** The model is small but the XLA
CPU compile is not, and the process gets killed (observed: exit 137).

```bash
bash rl/diffusion_ft/slurm/submit_gate2b.sh   # cpu partition (lov*), ~5 min
```

## Phase 3 — training (DPPO first)

`rl/diffusion_ft/train_dppo.py` (`DiffusionRLActor` + `ValueCritic` + GAE + env):
collect a batched episode, credit only the executed prefix, GAE with a state-only
critic on stopped-grad scene features, clipped-PPO on the trainable denoiser,
optional `--expert_weight` (DPPO + expert anchor). Unified CLI: `rl.diffusion_ft.train`.

```bash
# short GPU smoke first (shakes out integration shapes)
DFT_ITERS=2 DFT_INDICES=0,1 bash rl/diffusion_ft/slurm/submit_train_dppo.sh
# overfit test (8 scenes)
DFT_JOBNAME=dppo_pure bash rl/diffusion_ft/slurm/submit_train_dppo.sh
# DPPO + expert anchor -- keep the weight small, see below
DFT_JOBNAME=dppo_anchor DFT_EXPERT_WEIGHT=0.01 bash rl/diffusion_ft/slurm/submit_train_dppo.sh
```

Always set `DFT_JOBNAME` when submitting more than one arm: it names the log
(`%x`) *and* the env file, so two arms cannot silently share one config.

Logs/checkpoints under `--out_dir`; per-iter metrics in `train_log.jsonl`.
Planned next behind the same CLI (`--method`): `filtered`, `drwr`, `dawr`, `preference`.

### Status (2026-08-10): gradient fixed, learning not yet demonstrated

The first DPPO runs printed `pg=-0.0000 kl=0.0000 ratio=1.000` and looked frozen.
They were not — the console format string was rounding. `train_log.jsonl` had
`ratio=0.99999`, `kl=5e-6`, and a nonzero `grad_global_norm` every iteration.
**Always read the JSONL, not the console line, before concluding anything is
zero.** Four separate problems were behind the flat curve:

| Problem | Evidence | Fix |
| --- | --- | --- |
| Expert anchor swamped the PG term ~1000:1 | `loss = -1.4e-05 + 0.1*0.169`; PG was 0.08% of the objective | `--expert_weight` 0.1 → ~0.01 |
| `_reduce_logdensity` averages over `channels*prefix_len`, shrinking the PG gradient by that factor while the expert loss is unscaled | ratio pinned within 1e-5 of 1 after 4 epochs | `--pg_scale` (auto = `channels*prefix_len`) applied to `pg_loss` only; the clip still acts on the un-scaled ratio |
| Critic got one gradient step per iteration and could not track its own target | `value_loss` 409 → 1154 while return-to-go inflated 8 → 38 → 86, true return ~20 | `--value_epochs` 10, and `gae_lambda` 1.0 so `returns` is the observed discounted return instead of `adv + values` |
| Critic trained on returns from already-terminated envs | `adv` was masked by `alive` in `_merge`, `ret` was not | pass the mask; weight-normalized MSE |

After the fix the gradient is unambiguously live — `pg` -1.4e-05 → -4.4e-02,
`grad_global_norm` 0.02–0.09 → 0.11–1.8, `clip_frac` nonzero, `approx_kl`
approaching `target_kl`. Gate 2b re-passed 3/3 (job 27646).

**Episode return then got worse, not better** (16.2 → -5.7 over 13 iterations),
because the advantage the now-live gradient follows is still noise: pre-update
`value_loss` sat at 1000–5158 and jumped between iterations, i.e. critic error
exceeded the whole spread of episode returns. The `gae_lambda=1.0` /
`actor_lr=3e-5` retune is a response to that and is **untested** as of this
writing. If return still does not climb, the remaining suspect is advantage
*variance* (8 scenes × 1 rollout each), and the fix is a group baseline —
N rollouts per scene, advantage normalized within the scene — which removes the
value net from the critical path entirely. More tuning will not help there.

Scenes `0..7` are the argparser default: the first 8 of raw WOMD shard 0,
**not** the failure set. Fine as an overfit target, but do not read a result on
them as a result on the failure set.

### Operational notes

- **GPU class matters.** Five runs (27447/27448/27456/27457/27463) died in
  cuDNN init within seconds; all had landed on v100s (gpu20, gpu23) via
  `legacy`. Every run that reached iteration 0 was on an a6000 under `general`,
  which is the sbatch default. Check `host=` in the first line of the `.out`
  before theorizing about a failure.
- **`--out_dir` is on a quota'd filesystem.** Job 27279 died 27 iterations in on
  `OSError: [Errno 122] Disk quota exceeded` writing `train_log.jsonl`. The
  sbatch now write-probes `DFT_OUT_DIR` and exits non-zero in seconds instead.

## Phase 4 — the geometric safety loss (2026-08-31 / 09-01)

`safety_loss.py` scores geometry that already happened; `safety_objective.py`
scores a *proposed* candidate differentiably. Both terms are one-sided barriers
on a signed clearance, so the loss is flat when clear and an unbounded ramp once
in violation.

### What changed, and what it is now measured against

**The agent term measures the exact box gap** (`agent_geometry="exact"`, the
default; the old disc cover stays selectable as `"disc"`). Outside: the exact
distance between the two rectangles, a min over 32 vertex/edge clamped
projections. Inside: the exact penetration depth from a 4-axis SAT test. Both
branches go to 0 at contact, so the function is continuous, signed and unbounded
on the violating side — which the disc cover could not give, because two
overlapping boxes are at distance 0 there, flat, with no gradient exactly where
the policy is most wrong.

The trap, found by a unit gate and worth not re-discovering: the per-axis
quantity for a *depth* is `min(maxA - minB, maxB - minA)`, **not** waymax's
intersection length `min(maxA,maxB) - max(minA,minB)`. They agree on partial
overlap and always share a sign (so `has_overlap` is reproduced either way), but
under containment the intersection length returns the inner box's own extent —
position-independent, hence **zero gradient in the deepest-overlap case**. With
the wrong form a concentric pair reported a 2.0 m depth and `|grad| = 0`.

**Margins are calibrated per corpus, from the human demonstration.** They are
p01 (edge) / p05 (agent) of the expert's own clearance, so ~99%/95% of expert
steps stay on the plateau and the safety term does not fight whatever pulls the
policy toward the expert.

| | SMX failure set (39985) | **WOMD** (40089, the default) |
|---|---|---|
| expert-dirty scenes | 3/64 (4.7%) | 0/64 |
| edge p01 / p05 | +0.31 / +0.40 | +0.31 / +0.52 |
| agent p01 / p05 | +0.17 / +0.91 | +0.51 / **+0.76** |
| agent p50 | +4.96 | +1.90 |

Hence `edge_margin_m = 0.3` (p01 on *both* corpora) and `agent_margin_m = 0.75`.
Note WOMD traffic runs closer than the failure set does — those scenes are
geometrically hard, not crowded. Re-run `submit_calibrate_expert.sh` against the
corpus you will actually train on; these numbers do not transfer.

### Validation

- **Unit gates** (`tests/test_safety_loss.py`, CPU, seconds): exact gap ==
  an independent Minkowski-difference/hull computation to 9.5e-7 m over 400
  random pairs (135 overlapping); sign == `waymax.has_overlap` 400/400;
  1-Lipschitz continuity through contact; gradient unit-scale on both sides.
- **Probe** (`tests/probe_safety_loss.py`, GPU): on WOMD (40091) Gate A edge
  agreement is 100.0000% of 3,630 steps, C separation 1110x, D plateau 0.00000,
  E gradient 122x concentrated and buys real clearance. On the failure set
  (39987) Gate B is two-sided and exact: 0 of 2,940 steps disagree with waymax
  `overlap` across 368 overlaps.
- **Gate B is vacuous on easy WOMD scenes** — this policy had 0 overlap steps in
  3,630. The identity is corpus-independent (it *is* the SAT test `has_overlap`
  runs), but empirical coverage there needs scenes where the policy collides.

### The open blocker: the anchor is toothless, and sigma is why it may stay so

`gate_safety_grad.py` gates the analytic path. Gates 0, 0b, 1, 2, 3, 4a pass on
WOMD (40093); safety descends 2.91 -> 1.02 (-64.9%) in 30 steps. **4b fails**:
the anchor is 0.02% of the objective and 0.00% of the gradient at
`anchor_weight=1`. That is now a *scale* problem, not a wrong-target problem —
anchor loss median 9.55e-05 against a safety term of ~2.9 puts parity near
`anchor_weight` 1e4. `sweep_anchor_weight.sh` is the next run.

Before spending it, note a tension that has not been resolved. The DPPO sampler
runs with `sigma_sample_floor = 0.03` in *normalized* units, and ego positions
are normalized by `ego_range = 100 m` — so the floor is **~3 m of per-step
positional jitter** against ~0.87 m of per-step motion, and several times the
0.3 / 0.75 m margins the loss is calibrated to. Evidence it is jitter and not
speed: the gate's proposal arcs 27.79 m over 10 world steps while its mean
deviation from the log is only 2.87 m, and the *executed* rollout
(`deterministic=True`) is smooth at 0.91x expert speed with path efficiency
1.000 (`tests/diagnose_arc.py`, job 40319). Zero-mean jitter inflates arc length
without moving the mean. **Unconfirmed:** that `DFT_SIGMA_FLOOR=0.0` collapses
the arc to ~8-9 m. That run was blocked by a slurmdbd outage on 09-01
(every `sbatch` returning `Access/permission denied` while `scontrol ping` was
UP) and should be the first thing run when the scheduler is back.

Lowering sigma has previously frozen learning outright (the 08-11 sigma x KL
sweep never took a step), so this may not have a comfortable setting. Know it
before reading a fine-tune result, not after.

## Re-running DPPO on WOMD (2026-09-02)

Everything below the preprocessing note applies retroactively: **every DPPO run
taken on `failures.tfrecord` before 09-01 is unreadable**, because the ego frame
the policy was conditioned on was pinned ~1 s stale, not merely the BC target.
That includes `submit_dppo_entryclean.sh`, i.e. the reward-ordering test itself.

The re-run moves to raw WOMD rather than repeating on SMX: the timestamp fix is a
no-op there, and WOMD is the corpus the base checkpoint was pretrained on, which
is what makes the expert/BC term a coherent anchor.

```bash
# smoke first -- 2 iters, one seed, both regimes; check `route[train]: 32/32`
DFT_ITERS=2 DFT_EVAL_EVERY=1 DFT_SEED_GRID=0 \
  bash rl/diffusion_ft/slurm/submit_dppo_entryclean_womd.sh
# the run: 2 regimes x 3 seeds = 6 sibling jobs, 500 iters
bash rl/diffusion_ft/slurm/submit_dppo_entryclean_womd.sh
```

Two regimes, queued concurrently, answering different questions: `fixed` = 32
scenes of WOMD shard 0 loaded once (the like-for-like analogue of the invalidated
run, an overfit target), `corpus` = a fresh 32 scenes each iteration over all
1000 shards with a fixed held-out eval set (generalization; uses the byte-offset
indexes, already built 1000/1000).

The recipe is byte-identical to `entry_clean`. What changes is the corpus and
**`DFT_NUM_PATHS=45` / `DFT_NUM_POINTS_PER_PATH=800`** -- ScenarioMax's 10x300
against WOMD yields zero valid `sdc_paths`, so the whole route-based recipe would
be silently dead. Read `route[train]:` in the first `.out` before anything else.

Do not compare the return curve to any pre-09-01 run (corpus *and* ego frame
changed). WOMD is far easier than the failure set, so read the it=0 headroom
before reading a trend. The `sigma_floor=0.03` caveat below still stands.

### What the WOMD smoke actually said (jobs 40469/40470)

Both regimes ran clean and all three launch checks passed (`route[train]: 32/32`
at 45x800, entry_clean flags, a6000). The result is that the run **as configured
cannot answer its question**, for two reasons, and both are now tooled.

**No headroom.** `eval_solve_deterministic` at iteration 0 is **0.979** (fixed
shard-0 batch) and **0.948** (held-out corpus draw). The base checkpoint already
solves ~95-98% of uniformly drawn WOMD. There is nothing for a reward-ordering
fix to move.

**Collection and evaluation are different policies.** At `sigma_floor=0.03`:

| | deterministic | stochastic (what the gradient sees) |
|---|---|---|
| solve | 0.979 | **0.052** |
| collided | 0.000 | 0.750 |
| offroad | 0.021 | 0.750 |
| off-road steps | 0.31 / 57 | 15.0 / 57 |

DPPO computes every advantage from the stochastic rollouts (`collect`, not
`evaluate`), so the gradient is fitted to a policy that solves 5% of the time in
order to update one that solves 98%. It is live (`gnorm` 0.59 -> 0.86,
`clip_frac` 0.08-0.12) and it moves solve **down**: 0.979 -> 0.948 and 0.948 ->
0.906 after a single update each. One update against
`eval_solve_deterministic_std = 0.0147` is suggestive rather than conclusive, but
it falls in both regimes and in the direction the mechanism predicts.

So the binding constraint is collection noise, not reward ordering -- now
measured on WOMD with no ScenarioMax confound.

### Blocker 1: the sigma sweep, on WOMD, including 0

```bash
bash rl/diffusion_ft/slurm/submit_sweep_sigma_womd.sh
```

A thin wrapper on the tested `submit_sweep_sigma.sh` (WOMD data, 45x800 routes,
`sigma = 0` included -- the pending test the 09-01 slurmdbd outage blocked). It
measures the largest sigma whose stochastic driving still resembles the
deterministic policy's.

**It measures driving, not learning, and that is the trap.** Below ~0.03 the
candidate-KL trust region stops opening and no step is applied (the 08-11 sweep
froze outright), so a small sigma that drives beautifully and never takes a step
looks like a pass here. If every drivable sigma is untrainable, the fix is
structural -- a group baseline, N rollouts per scene with the advantage
normalized within the scene, which removes the value net from the critical path
-- not another sweep.

Fixed while pointing it at WOMD: `submit_sweep_sigma.sh` set `DFT_NODELIST`
with `:-`, so an explicitly empty override reverted to `gpu28` and produced
`--nodelist=gpu28 --exclude=gpu28`, an unsatisfiable job. Now no-colon `-`, the
idiom the other sweeps already use.

### Blocker 2: a WOMD pool with headroom

```bash
bash rl/diffusion_ft/slurm/submit_hard_womd.sh      # scan (GPU array) -> select (CPU)
DFT_SCENE_IDS_JSON=<scan_out>/hard_womd.json \
  bash rl/diffusion_ft/slurm/submit_dppo_entryclean_womd.sh
```

Keeps WOMD (real timestamps, the pretraining corpus) and recovers the failure
set's headroom by selecting on *this* policy's measured failures.

- `tools/scan_womd_headroom.py` (GPU, array-parallel, resumable) rolls three arms
  per scene and writes one JSONL row each: base **deterministic**, base
  **stochastic** at the collection sigma, and the **expert log replay**.
- `tools/select_hard_womd.py` (CPU) gates on expert-clean, keeps the base's
  failures, writes `hard_womd.json` + a summary.

**The expert gate is the point, not hygiene.** Selecting on policy failure
selects just as hard for dirty ground truth -- 22.6% of the diffusion policy's
failures on the curated set were scenes where the logged human is itself
off-road. WOMD measured 0/64 expert-dirty on a *uniform* draw, which says nothing
about the conditional rate among hard scenes; the summary reports that
conditional rate explicitly.

`SceneSampler` gained a `scene_ids=` pool (train and held-out eval both drawn
from it, still disjoint); `--scene_ids_json` wires it to the trainer and refuses
a pool scanned against a different `--shard_glob`, since the ids are positions in
that glob's concatenated shard order. Uniform sampling is unchanged and remains
the default -- a pool is a real distribution shift and is printed in
`describe()`.

Expect a yield around 5% of scenes scanned (base solve ~0.95, minus expert-dirty),
so `DFT_SCAN_SCENES=4096` gives roughly 150-200 pool scenes. For a pool that is a
corpus rather than an overfit target, scan ~16k (`DFT_SCAN_SCENES=16384
DFT_ARRAY=0-15`). `--include_marginal` widens a thin pool with solved-but-scraped
scenes, at the cost of a weaker definition of headroom.

### Preprocessing dependency (fixed 09-01, but know it exists)

Everything here reads `data/preprocess.py`. Every ScenarioMax-converted record
carries a constant `-1` timestamp column, which made `src_t` degenerate and
`searchsorted` return a meaningless bracket, so the ego frame
(`origin_xy`, `anchor_yaw`, `anchor_world_state` — all off
`ego_resampled_world[:, 0]`) was pinned to the log's **step 0** instead of the
current step. Fixed by `ego_source_time_seconds`: timestamps when strictly
increasing, else the step-index base. A no-op on WOMD, whose stamps are real.

Effect while it was live, measured: BC target round-trip 60.9 m -> 0.002 m, base
policy drift from the log 10.29 m -> 3.05 m. Weights were never affected — the
6/16 checkpoint trained on WOMD tf_example — but every *measurement* of policy
behaviour on SMX data predating 09-01 was taken with the policy seeing itself
~1 s behind where it was. `es_baseline/` holds an unpatched vendored copy by
design; its ES results carry the same caveat.

## Phase 5 — the teacher-forced unroll (`bondunroll`, 2026-09-02)

Ported from the `bondunroll` mechanic in `~/Proteina-Complexa`
(`training/unrolled_prediction.py` + `configs/example/training_cpsea_bondunroll_from_v4cfg.yaml`).
There it replaces a closing-bond loss taken on the one-step clean prediction
with the same loss taken on the endpoint of a short ODE unroll. Here the same
machinery lives in `unrolled_prediction.py` and gives the safety objective a
second, much cheaper way to produce the trajectory it scores.

**The mechanic.** Corrupt the *expert* future to a random low-noise timestep
(`q_sample(x0_expert, t)` — the teacher-forcing half), then run `K` eta-DDIM
reverse steps back down to `t_end` and score the clean prediction there. Only the
last `unroll_grad_steps` evaluations keep a gradient (DRaFT-K truncated
backprop). `--unroll_steps 0` (the default) keeps the original free-running path
and nothing about an existing run changes.

**Why it is not just an optimisation.** The one-step prediction the cheap
version of this loss would read is `E[x0 | x_t]` — the conditional *mean* — and
clearance is nonlinear in the coordinates, so `clearance(E[x0]) != E[clearance(x0)]`.
Averaging over plausible futures smooths the trajectory toward the middle of the
corridor, so the mean can sit inside the margin while a fair share of what the
sampler emits clips the edge. `K=1` reproduces that mean exactly and is the
control arm; `K>1` moves the penalised object toward a sample.

**What it costs and what it buys.** `K` network evaluations per gradient step
instead of the chain's 100. With `--anchor_states log` the collection rollout is
driven by the human's log window too, and then nothing in an iteration runs the
full chain.

**Two constraints that are easy to get wrong.**

- The unroll must stay **inside the trainable window**. Every step of it runs the
  trainable denoiser copy, but `sample_with_trace` only invokes that copy at
  `t <= k_trainable - 1`; a gradient taken above that trains behaviour the
  deployed sampler never uses except through weight sharing. `--unroll_t_start`
  defaults to `k_trainable - 1` and a larger explicit value is **rejected**.
- Teacher forcing does not fix exposure bias, and `--anchor_states log` gives up
  the half of it this package already had. The diffusion state
  `q_sample(x0_expert, t)` is a noised human trajectory, not what the frozen
  high-noise steps actually produce at that `t`.

```bash
# gate it first -- ~minutes, and gate 2 can tell you not to bother
bash rl/diffusion_ft/slurm/submit_gate_bond_unroll.sh
# then train
DFT_UNROLL_STEPS=3 bash rl/diffusion_ft/slurm/submit_train_safety.sh
# the cheap end of the sweep: log states, nothing samples the full chain
DFT_UNROLL_STEPS=3 DFT_ANCHOR_STATES=log bash rl/diffusion_ft/slurm/submit_train_safety.sh
```

### Gate results (job 40471, WOMD failures, 8 scenes, K=3)

Everything passes except the first version of gate 2, which asserted the wrong
*direction* and has been restated. The measured ordering at a fixed `x_t`
(`t_start=9`, 16.7 m of corruption):

| object scored | safety loss |
| --- | --- |
| human log | 0.2550 |
| free-running sample (what gets driven) | 0.2592 |
| **unrolled endpoint, K=3** | **0.2639 ± 0.0007** |
| one-step mean (`K=1`) | 0.2980 |

**The Jensen gap runs the opposite way here**, by 46 sem: the one-step mean
scores *worse* than the samples, not better. In a ring closure, averaging over
conformations pulls the two anchors together, so the mean closes when nothing
sampled does. Here the one-step prediction from a 16.7 m corrupted state is a
blurred, under-committed trajectory, and blur costs clearance rather than buying
it. The mechanic still does what it is for — the unrolled endpoint lands 8x
closer to the free-running sample than the one-step mean does — so gate 2 now
tests *that* (a distance to the driven trajectory) and records the gap without
gating on it.

**Transfer (gate 5) works.** 30 steps on the teacher-forced objective move the
*free-running* safety loss 0.2592 → 0.1072 (−58.7%), against a free-running
control's −68.0%. Cheaper steps, ~86% of the per-step effect.

**The speedup is small, and the reason matters.** 1.4x measured, not the ~30x the
eval count suggests, because the per-step cost is dominated by the **safety
evaluation, not the denoiser chain** — and the unroll shortens only the chain.
That reading was partly an artefact (the gate evaluated `safety_of` twice per
step, now fixed) and partly real; gate 3 now prints the forward-only
decomposition so the ceiling is visible. If the unroll is ever to be much
cheaper, the next thing to attack is the eager `candidate_safety` call in
`safety_update`, not `K`.

Arithmetic verified separately on CPU: `K=1` is bit-identical to
`p_mean_variance`'s `x0_pred`; the K-step endpoint matches a hand-rolled DDIM
chain exactly; at `t_end=0` the scored `x0_pred` is exactly what `p_sample`
emits; truncation moves the gradient norm 2.68 / 4.47 / 15.01 at `grad_steps`
1 / 2 / 8 with the forward unchanged.

### Logging (W&B added 2026-09-02)

`train_safety.py` had **no W&B integration until 2026-09-02** — only
`train_dppo.py` did. Every safety run before that date exists solely as
`train_log.jsonl`, which is why nothing from them appears in the dashboard. The
JSONL is still written unconditionally; W&B mirrors it and is off by default.

```bash
DFT_WANDB=1 DFT_UNROLL_STEPS=3 bash rl/diffusion_ft/slurm/submit_train_safety.sh
# push an older (or still-running) run's rows up after the fact -- CPU job
DFT_OUT_DIR=.../safety_train/20260901_234654 \
  bash rl/diffusion_ft/slurm/submit_wandb_backfill.sh
```

The run id is persisted to `<out_dir>/wandb_run_id.txt` and reused with
`resume="allow"`. That matters because this trainer resumes into a *fixed*
out_dir rather than opening a new timestamped one per launch — without the
pinned id, hitting the 12 h wall clock would split one training curve across two
charts at whatever iteration the limit happened to fall on. The backfill tool
writes the same file, so replaying an old log first and resuming that run later
with `--wandb` lands both halves on one chart.

**Baseline for comparison — job 40438 (free-running, 500/500 iters, complete):**
final eval `offroad 0.094`, `overlap 0.031`, `drift 1.64 m`. That is the control
any `--unroll_steps` arm should be read against.

## The Adam noise walk (2026-09-03)

The 09-02 safety:anchor sweep (`safety_anchor_sweep/20260902_200735_785143`,
6 arms × 200 iters, WOMD corpus, 48-scene named splits) degraded in **every**
arm, including both controls. Read as run means over its 21 eval points rather
than at the endpoints:

| arm | failure solved | success solved | success drift |
|---|---|---|---|
| `sw1_aw0` safety only | 0.354 | 0.806 | 1.10 → 2.86 m |
| `sw0.1_aw1` | 0.348 | 0.842 | → 1.85 m |
| `sw0.01_aw1` | 0.321 | 0.823 | → 1.51 m |
| `sw0.001_aw1` | 0.311 | 0.832 | → 1.25 m |
| `sw0.0003_aw1` | 0.300 | 0.846 | → 1.11 m |
| `sw0_aw1` anchor only | **0.191** | **0.792** | → 1.25 m |

So the safety term does **not** degrade the success split — that split is flat
across every arm and *lowest* in the anchor-only control — and failure solved
rises monotonically with safety weight. What rises monotonically with safety
weight is drift. The endpoint reading that suggested otherwise was noise: within
one arm, solved rate swings ±0.15 between adjacent eval points.

### Two defects, both fixed

**1. Eval drew from the training rng.** `rollout` split `self.rng`, which
collection and every update step also consumed, so an eval's sampler noise was a
function of how much training had preceded it. Arms with *identical* weights
disagreed at `iter 0` (success 0.69 vs 1.00). Evals now draw from `--eval_seed`
alone, average `--eval_repeats` of them (4 in the job scripts), and the frozen
base is scored once on the same seeds and carried in every row as `base_*`, so
each point is a paired comparison against a constant.

**2. One Adam step per replan context.** An Adam update is `lr·m/(√v+ε)` — O(lr)
per parameter *whatever the gradient norm is*. Gradient magnitude picks the
direction; the step **count** picks the distance. Nothing bounded it:
`safety_update` skips the DPPO trust region (correctly — no importance ratio to
protect — but that region was also the only leash on how far the weights moved),
and `clip_by_global_norm` sits *before* Adam, where a uniform pre-scale cancels
against Adam's own RMS normalization.

At P = 31,663,877, `lr = 1e-5`, N = 200 iters × 8 contexts = 1600 steps:

```
aligned      lr·N·√P      = 90
random walk  lr·√(N·P)    = 2.25
observed     pairwise ‖θᵢ − θⱼ‖ = 1.35 – 5.02      (‖θ‖ = 332.95 in all six arms)
```

`--grad_accum 0` (the new default) averages over every replan context and takes
one step per iteration — 8× fewer steps on 8× the data each. `--grad_accum 1`
restores the old schedule.

### Measured: it is coherent drift, not diffusion (job 41725)

`probe_param_distance` against the base checkpoint, on those same six arms:

| arm | ‖θ − θ₀‖ | ×walk | %aligned | failure solved |
|---|---|---|---|---|
| `sw0_aw1` | **4.352** | 1.93 | 4.83% | **0.191** |
| `sw0.0003_aw1` | 4.211 | 1.87 | 4.68% | 0.300 |
| `sw0.001_aw1` | 4.176 | 1.86 | 4.64% | 0.311 |
| `sw0.01_aw1` | 3.886 | 1.73 | 4.32% | 0.321 |
| `sw0.1_aw1` | 3.250 | 1.44 | 3.61% | 0.348 |
| `sw1_aw0` | **2.527** | 1.12 | 2.81% | **0.354** |

**Do not read "4.8% of aligned" as noise** — that was the first reading here and
it is wrong. `aligned` assumes perfect per-coordinate sign consistency for 1600
straight steps; a few percent of it is ordinary for real training with partial
cancellation, so magnitude alone cannot tell diffusion from slow coherent drift.
The displacement **cosines** settle it: they run 0.006–0.95 where two random
walks in 31.7M dimensions would be |cos| ~ 1e-4.

```
sw0.0003_aw1 | sw0_aw1     +0.9505     anchor-dominated arms: near-collinear
sw0.001_aw1  | sw0_aw1     +0.9003
sw0.01_aw1   | sw0.1_aw1   +0.7335
sw0.1_aw1    | sw1_aw0     +0.4602
sw0_aw1      | sw1_aw0     +0.0063     anchor vs safety: ORTHOGONAL
```

The arms sit on a clean 2-D structure — an anchor direction and a safety
direction at right angles, with the ratio arms interpolating monotonically
between them. Both terms move the policy coherently.

What replaced the noise story is sharper: **displacement rank-orders damage
perfectly and inversely** (Spearman −1.0 over all six arms), and the anchor is
what does the moving. Adding safety weight monotonically *shrinks* the distance,
which is what Adam does when two near-orthogonal terms disagree per coordinate —
`m̂/√v̂` falls, so safety brakes the anchor's march. And the two terms differ
wildly in weight-motion-per-metre-of-behaviour: safety-only moves the weights
least (2.53) and the car most (drift +1.76 m), anchor-only moves the weights most
(4.35) and the car least (+0.45 m). A large weight change buying almost no
behaviour is what a term spread over all 25 output columns looks like when only
~5 are ever executed.

**Caveat, and it is why `sweep_anchor_diagnosis.sh` exists:** all six arms ran at
`--seed 0` with the same data order, so their gradient noise is identical and
would also produce high cosines between *similar* objectives. The
anchor-vs-safety orthogonality rules that out as the whole story; it does not
rule it out for two arms sharing an objective. Only a seed replicate does.

### The null experiment

```bash
# anchor-only (the null objective) down an lr ladder, + a param-distance probe
bash rl/diffusion_ft/slurm/sweep_null_lr.sh
# just the measurement, against any finished sweep
DFT_ARMS='/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/safety_anchor_sweep/20260902_200735_785143/*/' \
  bash rl/diffusion_ft/slurm/submit_param_distance.sh
```

Arms are `lr:grad_accum`, defaulting to `1e-5:1 1e-6:1 1e-7:1 1e-5:0`. Given the
probe, the ladder now tests *damage ∝ distance* rather than *damage ∝ noise* —
still the right test, since distance scales with lr either way. The ladder
runs at `grad_accum=1` on purpose — that is the schedule that produced the
degradation being explained, and running it under the new default would confound
"lr does not matter" with "accumulation already fixed it". The fourth arm is the
controlled version of that: 200 steps instead of 1600 shrinks the predicted walk
by √8 = 2.8×, so it should land between the `1e-5` and `1e-6` arms. If it lands
on top of `1e-5`, the walk story is wrong.

`tests/probe_param_distance.py` reports `‖θ − θ₀‖` per arm against both
predictions, plus the **cosine between arms' displacement vectors** — two runs
sharing a real signal are correlated, two random walks are orthogonal (|cos| ~
1e-4 by chance in 31.7M dimensions). That is the difference between the safety
term steering and the safety term being a rumour on top of noise.

### Still open, and not explained by the walk

- The safety term genuinely trades drift for clearance: success drift rises to
  2.86 m in safety-only, far past the ~0.45 m the walk alone accounts for.
- **Support mismatch.** Safety scores world steps `t0+1..t0+10` (1.0 s), which the
  resample draws from roughly the first 5 of the model's 25 output columns; the
  anchor covers all 25. The two terms never compete on the same coordinates, so
  the effective ratio on the scored columns is ~5× the logged one.
- Failure-split `offroad_rate` rises over training in every arm (0.33 → 0.45+),
  and `safety_last` rises on the policy's own collected states.
- **No base number yet.** Every comparison above is arm-vs-arm, so "the anchor
  hurts" and "the anchor is neutral and safety helps +0.16" are both consistent
  with the data. `base_reference()` settles it and every run started after
  2026-09-03 prints it at its first eval.

### The anchor diagnosis

```bash
DFT_EXTRA_ARMS='/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/null_lr_sweep/<STAMP>/*/' \
  bash rl/diffusion_ft/slurm/sweep_anchor_diagnosis.sh
```

Arms are `anchor_states:seed`, defaulting to `policy:1 log:0`, both anchor-only.

- `policy:1` is the **seed replicate**: identical to the null ladder's
  `lr1e-5_ga1` arm except for `--seed`. Cosine ~0.95 against it means the anchor
  direction is objective signal; ~0.05 means it was one seed's noise, and the
  ladder's reading changes with it. The seed-0 side is not re-run — pass the
  ladder's directory as `DFT_EXTRA_ARMS` and the probe spans both sweeps
  (`--arms` takes comma-separated globs; run directory basenames are the arm
  labels and must stay distinct).
- `log:0` swaps `--anchor_states`. The default `policy` is DAgger-shaped: the
  contexts are the policy's own drifted states but the target is the human's
  future re-expressed from that drifted pose, which demands an instantaneous
  lateral correction a model then learns to emit. `log` trains the same objective
  where the target is feasible by construction. If the anchor's damage is the
  relabeling, `log` is harmless and `policy` is not.

### The base reference, measured (job 41716, abandoned at iter 17)

The first run with `base_reference()` gave the number every earlier claim was
missing — the frozen base, on the fixed eval seeds, on the random 32-scene
held-out batch:

```
base            solved 0.938   offroad 0.0625   drift 0.412 m
iter 0   (8 Adam steps)  0.922          0.078          0.717 m
iter 10  (88 steps)      0.922          0.070          0.810 m
```

This arm is the **null objective**: `safety_weight=0, anchor_weight=1`, a plain BC
continuation that optimizes nothing new. It is not a no-op. Solved 0.938 → 0.922
is one scene of 32 and means nothing alone, but **drift nearly doubles in the
first 8 Adam steps** (0.412 → 0.717 m), and drift is a continuous mean over 32
scenes, not a threshold rate — that is signal. Then it crawls: 0.717 → 0.810 m
over the next 80 steps.

That shape is the finding. Most of the displacement is spent in the first
iteration, before the run has seen anything, and **there is no lr warmup anywhere
in this trainer** — Adam's bias correction makes the earliest steps full-size
regardless of gradient magnitude. A warmup arm is probably worth more GPU time
than the ladder's `1e-7` rung, which will likely only show that nothing moves.

### Trap: the named splits must be pinned, or you measure the wrong thing

`DFT_SPLITS_JSON` defaulting to empty makes `_load_eval_splits` return `{}` and
the run scores **only** the random held-out batch — `failure_heldout` and
`success_heldout`, which every claim in this section is stated on, are silently
never evaluated. It fails quietly: the job runs, the curves look fine, and the
only tell is that the base-reference line prints one `eval_solved_rate` instead
of one per split. `sweep_null_lr.sh` shipped with this bug and burned an arm.

Both new sweeps now default it to the file the 09-02 sweep used:

```
/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/eval_splits.json
```

A *different* splits file would also be wrong — the scene ids are positions in
the scan's `shard_glob` order, so arms scored against another file are not
comparable to the 09-02 sweep even though nothing errors.

## The claim nobody had tested: does the loss fix the trajectory? (2026-09-04)

Short answer: **no run in this package has tested it.** Every safety arm to date
drew its training batch uniformly from the ~482k-scene WOMD corpus. A 48-scene
split named `failure_train` is therefore drawn about **0.3 times** in a
200-iteration run (`16 x 200 x 48 / 482153`), so `failure_train` and
`failure_heldout` were *both* held out and their curves agree for that reason and
no other. `pool=none` in every DPPO `.out` that logs it, and `train_safety.py`
had no pool flag at all. **Not one gradient step has been taken on a scene that
was then scored**, and nothing published from here shows a single off-road
trajectory being repaired -- only aggregate rates moving on scenes the gradient
never touched.

What the 09-02 sweep actually measured, per split, run over its 21 eval points
(`sw1_aw0` = safety only, `sw0_aw1` = anchor only, the nearest thing to a base):

| split | metric | `sw0_aw1` iter 0 | `sw1_aw0` iter 0 | `sw1_aw0` iter 190 |
|---|---|---|---|---|
| failure_heldout | solved | 0.125 | 0.354 | 0.375 |
| failure_heldout | **offroad** | 0.542 | 0.354 | **0.375** |
| failure_heldout | overlap | 0.229 | 0.250 | 0.229 |
| failure_heldout | reached | 0.854 | 0.896 | 0.938 |
| success_heldout | solved | **1.000** | 0.688 | 0.875 |

Three things to read off it, none of them "the loss fixes off-road":

- **Off-road does not fall.** On the failure split it *rises* over training in
  every arm (`sw1_aw0` 0.271 -> 0.375 on `failure_train`, 0.354 -> 0.375 on
  `failure_heldout`; §"Still open" records 0.33 -> 0.45+ elsewhere). Whatever
  `solved` gains come from `reached` and `overlap`, which is not the term the
  question was about.
- **The first eval is already downstream of the damage.** `sw0_aw1` at iter 0
  reads failure 0.125 / success 1.000 -- close to the split definitions. Every
  safety arm reads failure ~0.3 / success ~0.5-0.7 at the *same* iter 0, i.e.
  after 8 Adam steps. Comparing an arm to its own first point hides that.
- **`sw0_aw1` is the only arm whose iter 0 resembles the base**, and it is also
  the arm whose anchor was running on the frozen-rng bug. Neither the "safety
  helps" nor the "anchor hurts" reading survives that.

### What was added to test it

```bash
# both arms: trained ON the failure pool, probing a fixed 2x2 every 10 iters
bash rl/diffusion_ft/slurm/sweep_probe_anchor.sh
# smoke it first (~15 min, same code path)
DFT_ITERS=4 DFT_PROBE_EVERY=2 DFT_PROBE_PER_SPLIT=2 DFT_SPLIT_CAP=8 \
  DFT_EVAL_REPEATS=1 DFT_TRAIN_SPLIT_CAP=32,32 \
  bash rl/diffusion_ft/slurm/sweep_probe_anchor.sh
# failure-only pool (no do-no-harm pressure in the objective at all)
DFT_TRAIN_SPLITS=failure_train DFT_TRAIN_SPLIT_CAP=-1 \
  bash rl/diffusion_ft/slurm/sweep_probe_anchor.sh
# re-render the figures against a pair that already ran (CPU, seconds)
DFT_PROBE_ARMS="no-anchor=$OUT/noanchor anchor=$OUT/anchor" \
  sbatch rl/diffusion_ft/slurm/viz_probe_slider.sbatch
```

**`--train_splits` / `--train_split_cap`** (`DFT_TRAIN_SPLITS`,
`DFT_TRAIN_SPLIT_CAP`) build the training pool from named splits, so "trained on
the agent's own failures" is true rather than nominal. Empty is the old uniform
draw and stays the default; a pool is a real distribution shift and is printed at
startup. The caps matter: `success_train` is 23,254 scenes against
`failure_train`'s 564, so an uncapped union is 98% successes. The sweep defaults
to `256,256`, which is also what makes "trained on" mean something inside a few
hundred iterations -- 16 scenes/iter x 300 iters over 512 pool scenes is ~9 visits
each.

**`--probe_per_split`** pins a handful of scenes from each of the four splits and
dumps their *rollout* every probe point to `<out_dir>/probe/`: `static.npz`
(roadgraph, human log, goal, and the frozen-base rollout -- ~99% of the bytes,
written once), one small `iter_%06d.npz` per point, and `probe_log.jsonl`. The
rollout uses the fixed `--eval_seed`, so a difference between two frames is the
weights' doing and not the sampler's.

**A scene counts as `trained` only if it is in the pool AND not in
`SceneSampler`'s own held-out ids** -- the sampler removes those from the training
draw, and mislabelling one of them "trained on" is precisely the error this whole
probe exists to rule out. `tests/test_probe_selection.py` tests that without a
GPU (8/8, CPU, no corpus needed).

**`tools/viz_probe_slider.py`** (CPU) turns the dumps into
`probe_slider.html` -- a slider over optimization steps, rows = arms, columns =
failure/success x trained-on/held-out, grey = the frozen base and colour = the
current weights in every panel -- plus `probe_filmstrip.png` (the same thing as a
still, for slides) and `probe_curves.png` (the rates, with the base as a dashed
reference). It reads only the dumps, so it re-renders in seconds and can be run
against a *still-running* pair.

Failure columns lead with a scene the base actually drove off the road: ~15% of
the failure pool is `never_reached`, a failure with no off-road and no contact in
it, which the barrier has nothing to say about. `anchor_last` / `anchor_share`
are now logged per iteration for the same reason -- at `anchor_weight=1` the
anchor measured 0.66% of the objective, so "with anchor" at 1 is "without anchor"
with extra compute, and the sweep's default of 100 is a guess off a pre-rng-fix
gate ladder (30 -> 17%, 100 -> 40%, 300 -> 67% of the objective) that the first
iteration's `anchor_share` will correct.

**What the slider cannot settle.** It is a handful of scenes chosen by
`--probe_seed`; `probe_curves.png` and the `<split>/*` keys in `train_log.jsonl`
carry the population claim. A slider is evidence about a mechanism, not a rate.

## The 1000-scene rerun on verified splits (2026-09-06)

The 09-04 probe sweep answered the "does it fix anything" question, and one arm
came out ahead: **safety only (`safety_weight=1, anchor_weight=0`), gradient
taken on a mixed failure/success pool** (`probe_anchor/20260904_002609_25528/noanchor`).

| split | base | iter 299 |
|---|---|---|
| failure_heldout | 0.328 | **0.578** (peak 0.625) |
| failure_train | 0.219 | **0.562** |
| success_heldout | 0.984 | 0.922 |
| success_train | 0.953 | 1.000 |

The anchor arm was worse on failures at every point after iter 40, and the
failure-ONLY pool (sweep `..._25576`) bought the same failure gain by taking
`success_heldout` to 0.688 -- so the mixed pool is doing the do-no-harm work.

Three things were wrong with the *measurement*, and this section is what fixes
them. Nothing about the objective changed.

### 1. The splits were 48 scenes, which is the size of the effect

`DFT_SPLIT_CAP` defaulted to 48. A 48-scene split has a ~0.07 binomial floor on a
rate near 0.8 before any policy noise, against a measured effect of ~0.25. The
default is now **1000**, and `--eval_chunk` (default 32 — it was 64 until the
a5000 OOM below) rolls a split in chunks
and aggregates scene-weighted -- a single 1000-scene batched state is ~2 GB and
its rollout does not fit on a 24 GB a5000. The aggregate is exact, including the
ratio metric `arc_ratio_vs_log`, which is recomputed from summed arcs rather than
averaged. `tests/test_eval_chunking.py` pins all three of the changes in this
section (21/21, CPU, no corpus needed) -- chunk arithmetic against a ragged tail,
the per-batch quota, and the ambiguous-scene rule below.

Cost is linear in `cap x --eval_repeats x |--splits|` at every `--eval_every`:
at 1000 x 2 x 4 an eval point is ~25 min, so `submit_verified_safety.sh` uses
`DFT_EVAL_EVERY=25` (12 points over 300 iters). Raising the cadence back to 10
needs a `--time` bump, not just the flag.

### 2. A third of the "failure" split was not a failure

The splits were labelled by `scan_womd_headroom`, which rolls the base
**deterministically** at `sigma_floor=0.03` under its own reward config. The
training curve scores the same checkpoint through
`train_safety.rollout(train=False)`: stochastic sampler at `sigma_floor=1e-4`,
8 replans, expert-clean gate. Under *that* protocol the base scores
`solved=0.328` on `failure_heldout` -- a split defined as scenes it fails.

`train_safety --verify_only` re-measures the base per scene **through the eval
path itself** (not a parallel implementation that can drift) at
`--verify_seed 777`, deliberately disjoint from `--eval_seed 12345`: selecting
scenes on the draws they are later scored on would bake this sample's noise into
the split. `tools/rebuild_verified_splits.py` then keeps only

* **failures** the base solved in **0 of 4** rollouts and that are expert-clean;
* **held-out successes** it solved in **4 of 4**.

Scenes it solves *sometimes* are dropped from both pools rather than reassigned.
The success *training* pool is deliberately NOT filtered -- the gradient should
see all the data, and verifying 23k scenes would cost more than the fine-tune.

### 3. The pool was a capped sample, not the data

`--train_split_cap 256,256` balanced the mix by throwing away 22k successes. The
new `--train_split_ratio` (`DFT_TRAIN_SPLIT_RATIO`) is a **per-batch quota**
instead: `1,1` puts 8 failures and 8 successes in every 16-scene batch while the
pool stays uncapped. Quotas are largest-remainder so they sum to the batch size
exactly, batch composition is constant across iterations, and a group smaller
than its own quota raises at the first batch instead of looping forever.

```bash
# smoke all three paths first (~20 min: verify, chunked eval, balanced batches)
bash rl/diffusion_ft/slurm/submit_smoke_verified.sh
# the chain: verify (GPU) -> rebuild (CPU) -> train (GPU) -> figures (CPU)
bash rl/diffusion_ft/slurm/submit_verified_safety.sh
# once the verified splits exist they are a build-once asset
DFT_SKIP_VERIFY=1 bash rl/diffusion_ft/slurm/submit_verified_safety.sh
```

**The failure splits cannot reach 1000.** The scan covered 24,278 scenes at a
2.6% failure rate, so there are 628 failure candidates in total and verification
only removes more (expect ~380). `--heldout_failure -1` splits what survives
50/50 between the held-out set and the training pool; the success splits do get
their full 1000. Getting 1000 *held-out* failures means scanning ~100k more
scenes with `submit_build_splits.sh`, which is ~25 GPU-hours and was deliberately
not done here.

### Mixing in the pretraining loss (`--bc_weight`, 2026-09-07)

The safety-only arm still costs ~0.06 on `success_heldout`. The obvious leash is
the objective the checkpoint was pretrained with — but that is **not**
`--anchor_weight`, even though both call the same function
(`DiffusionRLActor._expert_loss` → `diffusion.loss_per_example`). What differs is
the state distribution the conditioning comes from:

| | conditioning states | target | shape |
|---|---|---|---|
| `--anchor_weight` | the **policy's** rollout | human future re-expressed from the drifted pose | DAgger relabeling |
| `--bc_weight` | the **human's** log states | that state's own future | the pretraining loss |

From a drifted state the anchor's target demands an instantaneous lateral
correction, and a model trained on it learns to emit exactly that — consistent
with the measured result that the anchor moves the weights *most* (4.35) and the
car *least* (+0.45 m), and that it was the worse arm on failures in the 09-04
sweep. `--anchor_states log` already trains on human states, but it drives the
*only* rollout from the log, so the safety term is then scored where there is
nothing to fix. The two were coupled; `--bc_weight` decouples them.

`collect_bc_contexts` is a **second, sampling-free pass over the same batch**:
the log supplies the trajectory, so it costs 8 env steps against the policy
pass's 8 × 100 denoiser evals — **measured at 9%** of an iteration
(`bc_collect_s` 3.9 s of 43 s), not 2×. Same batch on purpose: under `--train_split_ratio 1,1` half of
it is failure scenes, and there the human log *is* the trajectory the safety term
is searching for, so the term is a do-no-harm leash on the successes and a
demonstration on the failures.

**Sizing the weight — and read the MEDIAN, not iteration 0.** A weight is
meaningless without the size of the term. The ladder was predicted from the
anchor (`anchor_weight=1` measured 0.66% of the objective, and its gate ladder
read 30 → 17%, 100 → 40%, 300 → 67%) and the prediction held:

| | iter 0 | median over the run |
|---|---|---|
| `bc_last` | 0.058 | **0.0044** |
| `bc_share` at `bc_weight=300` | 0.95 | **0.57** |

The BC loss starts high and is minimized within a few iterations, so **iter 0
reads the term at ~13× its steady size** — the guidance to read it there was
wrong and is corrected here. On the median, weight 100 → 0.30 of the objective
and 300 → 0.57, so `0,100,300` is a well-spread ladder. ~0 would mean an arm is
the control with extra compute; ~1 would mean safety is switched off.

**First result** (bc300, 300 iterations, verified splits): `failure_heldout`
0.040 → 0.364 (peak 0.390) against `success_heldout` 0.998 → 0.962. A 9× gain on
scenes the base verifiably fails, for 0.036 on the do-no-harm set — but the bc0
and bc100 controls OOMed (below), so nothing is attributable to the BC term yet.
The BC pass itself cost **9%** of an iteration (`bc_collect_s` 3.9 s of 43 s).

```bash
# the ladder, sharing one verify/rebuild; arms fan out as siblings
DFT_SKIP_VERIFY=1 bash rl/diffusion_ft/slurm/sweep_bc_weight.sh
DFT_BC_LADDER="0,100" bash rl/diffusion_ft/slurm/sweep_bc_weight.sh
# or a single arm through the ordinary submitter
DFT_BC_WEIGHT=100 bash rl/diffusion_ft/slurm/submit_verified_safety.sh
```

Never run `--bc_weight` and `--anchor_weight` together in an arm you intend to
read: two BC-shaped terms on two state distributions make neither attributable.
`tests/test_bc_term.py` pins the routing (10/10, CPU) — that the term reads
`bc_ctx` and not `ctx`, that it uses `bc_ctx.alive`, and that the anchor and BC
keys are independent draws, all of which fail silently by producing a plausible
curve.

### The marginal (corpus-level) rate — and why every arm so far is a net loss

The eval splits are a **stratified** sample and the strata are wildly
disproportionate: verified failures are **1.5%** of the expert-clean corpus,
verified successes **98.5%**. So no per-split curve answers the deployment
question — did the overall solve rate go up — and worse, the per-split view gets
the *sign* wrong. bc300, 300 iterations, verified splits:

| iter | failure (w=0.0150) | success (w=0.9850) | marginal | vs base |
|---|---|---|---|---|
| base | 0.040 | 0.998 | 0.9842 | — |
| 0 | 0.090 | 0.992 | 0.9790 | −0.0052 |
| 100 | 0.338 | 0.909 | 0.9005 | −0.0837 |
| 250 | 0.390 | 0.918 | 0.9101 | −0.0741 |
| 299 | 0.364 | 0.962 | 0.9531 | −0.0311 |

**Every eval point is worse than the base**, and the iteration-250 point that
looks best on failures is nearly the worst marginally. The arithmetic is
unforgiving: a gain of `d` on the failure split is paid for only if the success
split stays within `d × w_f/w_s = d/65.8` of its base — a +0.30 failure gain
allows a success drop of **0.0046**, and bc300 gave up 0.036.

`marginal/solved_rate`, `marginal/delta_vs_base` and
`marginal/breakeven_success_rate` are now logged at every eval point (and go to
W&B with the rest of the row); the trainer also prints
`MARGINAL 0.9531 (-0.0311 vs base)` on the eval line. Weights live in the
`population` block that `rebuild_verified_splits` writes, so they are a property
of the splits file rather than a constant in the code — re-run it on the same
rows to add the block to an older splits file (deterministic; the split ids do
not change). A missing stratum **disables** the marginal rather than
renormalizing over whatever was measured.

For a run that already finished, `tools/marginal_rate.py` computes the same
numbers from `train_log.jsonl` (CPU), and `--in_place` writes the keys back so
`tools/wandb_backfill.py` can push them to the existing W&B run.

Scenes the base solves only *sometimes* belong to neither stratum, so the
marginal covers **96.5%** of the expert-clean corpus; `marginal/coverage` records
that. `tests/test_marginal_rate.py` (13/13, CPU) pins the weighted sum, the
break-even algebra, and the refusal to compute a partial marginal.

### Running it for days (2026-09-08)

`general` caps a job at 2 days (`MaxTime=2-00:00:00`), so a multi-day arm is a
CHAIN. `sweep_bc_long.sh` submits each arm as a **job array with `%1`**: tasks
run one at a time, in order, each resuming from `<out_dir>/latest`. One
submission per arm instead of N, and a task that dies just means the next one
picks up from the last checkpoint.

**Three things in the resume path had to be fixed first, and all three failed
silently** — the job restarts, the log continues, the curve looks smooth:

1. **`save` wrote weights only.** Every restart reset Adam's moments and step
   counter, and bias correction then makes the first steps after a restart
   full-size *whatever the gradient is* — this model moves drift 0.412 → 0.717 m
   in ONE such iteration, then only → 0.810 over the next 80. That is one
   uncontrolled kick per restart, in a package where displacement rank-orders
   damage (Spearman −1.0). `save` now also writes `opt_state.npz` and
   `train_state.json` (rng + `adam_steps`). It is a **separate file** because
   `load`, `eval_reach` and `probe_param_distance` all read `tparams.npz` via
   `sorted(z.files)`; an extra key there would permute the pytree in all three.
   An old checkpoint still loads, with a printed warning.
2. **The batch stream restarted.** The sampler is now seeded
   `data_seed + 100003 * resume_iter`, so a resumed run continues the stream
   instead of re-drawing the batches it already trained on. `_resume_iter` is
   read in `__init__` (the seed depends on it) and reused for the loop start, so
   the two can never disagree.
3. **`latest` is overwritten.** `--keep_every` writes un-overwritten
   `iter_%06d` snapshots — weights only (~121 MB), since a snapshot exists to be
   *scored*, not resumed. Over days the question "the success split was best
   2000 iterations ago, give me that model" is the one you will actually have.

**The a5000 will OOM at `--eval_chunk 64`.** The eval rollout is the job's peak
allocation (8 replans of the 100-step sampler over a whole chunk at once). Jobs
44862/44864 both died with `RESOURCE_EXHAUSTED: ... 1.84GiB` at their first eval
point on **gpu28 (a5000, 24 GB)** while the identical arm on gpu27 (a6000,
48 GB) ran to completion. Fixed two ways: `DFT_EVAL_CHUNK` defaults to **32**
(a third below the cap-48 batch that always worked) and the sbatch exports
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, since JAX otherwise preallocates 75% and
a transient beyond that fails on a card with room to spare.
`DFT_EXCLUDE=gpu28` is the fallback.

**Eval budget is the real scheduling decision.** An eval point is
`cap × repeats × |splits|` = 2346 scenes × repeats: ~13 min at repeats 1, ~27 at
2. At the 300-iteration default (`eval_every=25`) a multi-day run spends HALF its
wall clock evaluating. The long sweep uses repeats 1 every 200 iterations (~18%
overhead, ~36 points over 5 days). repeats=1 is not a compromise at this size —
the 4-repeat default was set when a split was 48 scenes with a ~0.07 binomial
floor; at 1000 scenes the split average already integrates sampler noise across
scenes.

```bash
DFT_SKIP_VERIFY=1 bash rl/diffusion_ft/slurm/sweep_bc_long.sh          # 6 d/arm
DFT_DAYS=10 DFT_ITERS=40000 bash rl/diffusion_ft/slurm/sweep_bc_long.sh
```

`scancel` at any point leaves a resumable `latest` and a complete curve.

### Where this stands (2026-09-08)

> Superseded by [the 09-09 → 09-13 sweeps](#geometry-stopping-and-gating-2026-09-09--09-13):
> the bc arms were measured (bc300 is decisively worse), and neither
> `k_trainable` nor any other objective knob is a lever.

**The binding constraint is do-no-harm, not the failure gain.** That is the one
sentence to carry forward. Verified failures are 1.5% of the corpus, so the
break-even budget for a +0.30 failure gain is a **0.0046** drop on the success
split, and every arm measured so far spends 5-10× that. The failure curve going
up is necessary and nowhere near sufficient; read `marginal/delta_vs_base`.

What is settled:

* the safety loss works and is validated against waymax (5 gates, 2026-08-30);
* it does move the failure split hard — 0.040 → 0.364 on scenes the base
  *verifiably* fails, a 9× gain;
* it pays for that with success-split regression far beyond the marginal budget;
* the denoising **anchor** is not the fix (policy-state relabeling, DAgger-shaped);
  the **BC term** on human states is the current candidate and is untested,
  because its two controls OOMed.

What is running or next:

| | |
|---|---|
| `sweep_bc_long.sh` | bc_weight 0 / 100 / 300 (0.00 / 0.30 / 0.57 of the objective), days per arm |
| read | `marginal/delta_vs_base` crossing zero — not the failure curve |
| if no arm crosses | the term is not enough; the next lever is where the gradient enters (`k_trainable`), or a per-scene margin (see Phase 4's open decision) |

CPU test inventory, all runnable without a GPU or the corpus:

```bash
for t in test_marginal_rate test_resume_state test_bc_term \
         test_eval_chunking test_probe_selection; do
  python -m rl.diffusion_ft.tests.$t; done      # 13 + 15 + 10 + 21 + 8 = 67 checks
```

## Geometry, stopping, and gating (2026-09-09 → 09-13)

**Result: the safety fine-tune cannot beat its base by an amount that matters.**
Ungated it is **−0.011** corpus solve rate (paired, 3 seeds). Gated to high-risk
scenes it is **+0.0005** (cross-validated), against a **+0.0039** ceiling for a
perfect gate. Five experiments, in order:

| date | experiment | array | outcome |
|---|---|---|---|
| 09-09 | one knob at a time, 12 arms × 300 it | 46733 | misread at iter 299; corrected below |
| 09-10 | leave-one-out combo, 7 arms × 200 it | 47510 | the "wins" do not compose |
| 09-11 | stopping: base, swin25 × 3 seeds × 150 it, eval every 10 | 47924 | no stable optimum; seed spread ~0.03 |
| 09-12 | bc300 × 3 seeds, same protocol | 49302 | decisively worse than the frozen-ckpt anchor |
| 09-13 | hardness gate, per-scene paired | 51307/51308 | sign flips, win ≈ 0 |

Everything below reads **`marginal/delta_vs_base`** (margΔ) =
`0.0150 × failure_heldout + 0.9850 × success_heldout − base`, on
`verified_seed777/eval_splits_verified.json` (base 0.017 / 0.997).

### Noise floors — read these before ranking anything

* `failure_heldout` is **173 scenes** (it cannot reach the 1000 cap; only ~628
  candidates exist). se ≈ 0.036, so arms **cannot be ranked on failure**.
* `success_heldout` is 1000 scenes, se ≈ 0.006.
* **Three seeds of an identical config, same scenes, same eval seed, differ by a
  median 0.031 margΔ at a fixed iteration** (up to 0.135). This is the real floor.
  Every single-run difference below ~0.03 in any sweep here is noise.

### New flags and plumbing

| flag / change | meaning |
|---|---|
| `--edge_tau`, `--agent_tau` | softplus knee width of the barrier (defaults 0.15 / 0.25) — previously hardcoded |
| `--safety_window` | WORLD steps of the candidate the safety term scores. `0` = executed window (10), `-1` = whole candidate (50), `N` |
| `--anchor_window` | MODEL steps the anchor/bc/base_anchor losses are charged on. `-1` = whole horizon (25), `0` = executed part (5), `N` |
| `GaussianDiffusion.loss_per_example(time_weight=)` | weighted-mean time mask; `None` is bit-for-bit the old loss |
| `train_safety.sbatch` / `verify_splits.sbatch` accept an env **directory** | array mode: each task sources `<dir>/<task_id>.env`, so `--array=0-N%3` caps concurrent GPUs |
| `--verify_only --load_ckpt DIR` | per-scene rows for a **fine-tuned** checkpoint (each row records `ckpt`) |

The two windows live in different time bases (world dt 0.1 s vs model dt 0.2 s);
`_init_windows` resolves `0`/`-1` and prints both in seconds at startup.

### 1. One knob at a time (09-09, `ablation/20260909_171022_586834`)

Centre: β=100 success-only frozen-ckpt anchor, k=10, τ 0.15/0.25, margin
0.3/0.75. Scored at iter 299:

| arm | change | fail | succ | margΔ | drift m |
|---|---|---|---|---|---|
| tau_lo | τ 0.075/0.125 | 0.335 | 0.979 | −0.0130 | 0.72 |
| swin_full | safety on 5 s | 0.318 | 0.975 | −0.0172 | 0.79 |
| awin_exec | anchor on 1 s | 0.358 | 0.972 | −0.0195 | 0.72 |
| swin_25 | safety on 2.5 s | 0.324 | 0.972 | −0.0200 | 0.77 |
| margin_lo | 0.15/0.375 | 0.347 | 0.970 | −0.0217 | 0.72 |
| tau_hi | τ 0.30/0.50 | 0.364 | 0.961 | −0.0303 | 0.73 |
| k25 | 25 trainable steps | 0.329 | 0.961 | −0.0308 | 0.71 |
| k3 | 3 trainable steps | 0.347 | 0.960 | −0.0315 | 0.73 |
| **base** | centre | 0.335 | 0.954 | −0.0376 | 0.72 |
| margin_hi | 0.60/1.50 | 0.376 | 0.949 | −0.0419 | 0.76 |
| anchor_bc | bc 100, no base anchor | 0.399 | 0.947 | −0.0435 | 1.58 |
| anchor_none | no anchor | 0.422 | 0.756 | −0.2313 | 3.12 |

**This ranking is a stopping-point artifact.** The centre itself was −0.0071 at
iter 200 and decayed to −0.0376 by 299; the "wins" were arms that decay more
slowly. Two results survive because they are far above the noise: no anchor
collapses success, and the log-target anchor is worse than the frozen-ckpt anchor
at **every** matched iteration (23× at iter 100), with drift climbing monotonically.

### 2. Do the wins compose? (09-10, `combo/20260910_223634_788955`)

Leave-one-out around all four wins ON (k3, 200 iters). At iter 199:

| arm | fail | succ | margΔ |
|---|---|---|---|
| **ref_k3** (all four off) | 0.324 | 0.986 | **−0.0063** |
| combo_noMargin | 0.347 | 0.987 | −0.0049 |
| combo_noSwin | 0.347 | 0.987 | −0.0049 |
| combo_noTau | 0.370 | 0.983 | −0.0085 |
| **combo** (all four on) | 0.364 | 0.978 | **−0.0135** |
| combo_noAwin | 0.347 | 0.975 | −0.0167 |
| combo_k10 | 0.376 | 0.967 | −0.0242 |

No. ref_k3 @199 reproduces the ablation centre @200 across sweeps and seeds.
Arm gaps are 1–3σ; do not rank them.

### 3. Where to stop (09-11/12, `stopping/20260911_160658_962947`, `stopping/20260912_191809_1099368`)

base, swin25 and bc300, each × 3 seeds (`DFT_SEED` and `DFT_DATA_SEED` varied,
eval seed pinned), 150 iters, eval and `keep_every` 10, held-out splits only.

| config | best 3-seed mean margΔ | at iter | fail | succ |
|---|---|---|---|---|
| base (β=100) | −0.0093 | 110 | 0.293 | 0.983 |
| swin25 | −0.0104 | 120 | 0.353 | 0.981 |
| bc300 (bc 300, β=0) | −0.0337 | 130 | 0.353 | 0.958 |

* All configs are negative at all 16 eval points.
* swin25 − base averages −0.0008 and changes sign 9/16 times: no difference.
* No iteration is best across seeds (s0 peaks at 90, s1 at 120, s2 at 130).
* **bc300 − base averages −0.049** and leads at 1/16 points. It collapses success
  to 0.77–0.80 over iters 10–30 and drifts to 1.13 m (base 0.65–0.75). The 09-07
  `bc_sweep` curve that looked better was scored on a splits file rewritten 14 h
  after that sweep started (its base was 0.040/0.998) and had no marginal computed.

### 4. The hardness gate (09-13, `gate/20260913_140701_1214568`)

Deploy the fine-tune only where `runs/hardness/risk.npz` `scenario_risk ≥ t`, the
frozen base elsewhere. Risk rows index identically to the sampler's scene ids
(checked); AUC on these labels **0.847** (train splits 0.853). Base and the three
`base` seeds @ iter 110 scored per scene by `--verify_only` at **verify seed 4242**
(not 777, which selected the splits; not 12345, which chose iter 110), 4 repeats,
so every scene's (ft, base) pair is paired.

| share of corpus gated | margΔ | 95% CI (scene bootstrap) |
|---|---|---|
| 0.5% | +0.0010 | [+0.0005, +0.0018] |
| 1–3% | +0.0007..+0.0009 | ~[−0.0009, +0.0024] |
| 5% | −0.0006 | [−0.0032, +0.0015] |
| 10% | −0.0038 | [−0.0078, −0.0004] |
| 100% (ungated) | −0.0114 | [−0.0170, −0.0066] |

**Cross-validated operating point** (threshold chosen on half the scenes, scored
on the other, 500×): **+0.0005**, 90% [−0.0011, +0.0018], positive in 84% of
splits, median gate 1.9% of the corpus. Oracle ceiling (gate = failure stratum
exactly) +0.0039.

Why it stops paying at ~3%: the fine-tune's damage to successes is concentrated
in exactly the high-risk scenes. Seed-mean ft − base by risk band:

| risk band | success Δ | failure Δ |
|---|---|---|
| bottom 50% | −0.003 | +0.163 |
| 50–80% | −0.017 | +0.341 |
| 80–90% | −0.017 | +0.382 |
| 90–95% | −0.062 | +0.352 |
| top 5% | −0.080 | +0.389 |

Gain saturates from the median up; damage keeps climbing.

```bash
# score + analyze (GPU array %3, then CPU afterok)
bash rl/diffusion_ft/slurm/submit_gate_analysis.sh            # DFT_GATE_ITER=110 default
# re-analyze / redraw without a GPU
sbatch rl/diffusion_ft/slurm/gate_analysis.sbatch <gate_dir>/analyze.env
# -> <gate_dir>/gate_curve.json, gate_curve.png
```

### Where this stands (2026-09-13)

Settled:

* **Objective geometry is inert** — τ, margin, k_trainable, both windows: all
  inside the ~0.03 seed spread. k3 is the cheap choice.
* **Only the frozen-checkpoint anchor works** — no anchor (−0.23), bc 100 and
  bc 300 are clearly worse and drift the policy.
* **No stopping point breaks even** — best 3-seed mean −0.009 @110.
* **A hardness gate buys ≈ +0.0005** of a +0.0039 ceiling.
* **Method:** on a non-monotonic curve a single-iteration comparison ranks decay
  rates, not configs; below ~0.03 use ≥3 seeds or a paired per-scene test
  (`--verify_only --load_ckpt`, then McNemar on discordant scenes).

The one open direction: a fine-tune that protects **hard successes** specifically
(e.g. a base-anchor weight that rises with scene risk). Retuning the current
objective, its stopping point, or where it is deployed is bounded near zero.

Scripts: `slurm/sweep_safety_{ablation,combo,stopping}.sh`,
`slurm/submit_smoke_ablation.sh`, `slurm/submit_gate_analysis.sh`,
`tools/gate_analysis.py`, `tools/plot_gate_curve.py`.
