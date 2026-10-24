# `rl/diffusion_ft` — diffusion policy simulator-reward fine-tuning

Unified framework for fine-tuning the pretrained diffusion motion policy against
the **simulator** reward (filtered FT, DRWR, DAWR, preference, DPPO, hybrids,
ES + distillation). Deliberately separate from
`train_diffusion/train_diffusion_policy.py`, which only updates
instruction/subgoal/FiLM params. The outer action here is *one sampled
trajectory*.

**Post-hoc trajectory optimization** (training-free repair of a sampled plan,
and how to add it to an ES planner): see [README_posthoc_opt.md](README_posthoc_opt.md).

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

Interpreter: `/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python` (flax 0.12.7).
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
DFT_ARMS='/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/safety_anchor_sweep/20260902_200735_785143/*/' \
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
DFT_EXTRA_ARMS='/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/null_lr_sweep/<STAMP>/*/' \
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
/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/hard_womd/scan_seed0/eval_splits.json
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

> Superseded 2026-09-14 by [the results below](#results-2026-09-14): protecting
> hard successes (matched-sample preservation) does remove the damage but only
> reaches parity, while making the base the DEFAULT and letting the fine-tune earn
> each replan is +0.0033 with nothing broken.

Scripts: `slurm/sweep_safety_{ablation,combo,stopping}.sh`,
`slurm/submit_smoke_ablation.sh`, `slurm/submit_gate_analysis.sh`,
`tools/gate_analysis.py`, `tools/plot_gate_curve.py`.

## Selection and matched-sample preservation (2026-09-13 → 09-14)

**Result: keeping the base as the default and letting the fine-tune earn each
replan is +0.0033 corpus solve rate (+0.0077 on three strata), against +0.0005
for the hardness gate and −0.0101 for deploying the fine-tune everywhere. It
breaks no success at all. Read [the results](#results-2026-09-14) first; the
sections below are how each piece works.**

Two experiments follow from the gate result. **The base stays the default.** A
second candidate has to earn each intervention, and training protects what the
base actually drives when it succeeds. Code is in; no GPU job has run yet.

### Experiment 1: training-free proposal selection (`proposal_select.py`)

At every replan the base draws its candidate and keeps the frozen prefix
`x_{k-1}` and its late key. The alternative is one of:

| alt | what | extra compute |
|---|---|---|
| `ft` | fine-tuned late chain, **same prefix, same late noise** | k denoiser evals |
| `base2` | base late chain, same prefix, fresh late key (**matched control**) | k evals |
| `basefull` | an independent full base sample | 100 evals |

Both candidates are scored on the exact clearances whose signs *are* waymax's
offroad/overlap (min road-edge clearance, min agent box gap) over `window` world
steps, plus progress toward the goal. The `cons` selector switches only if the
base candidate is predicted to violate, the alternative is predicted clean, and
the alternative gives up less than `tol` metres of progress. A clean base
candidate is never touched (pinned by a test). The `margin` selector also acts
on near-misses. There is no lookahead yet (execute, replan, then judge); that is
the next selector if single-step geometry turns out not to capture the benefit.

All arms run on the same scenes, seed and fixed chunks, so everything is paired
draw for draw. The same holds across checkpoint tasks: chunks are fixed slices of
the shard list, never of the not-yet-done remainder, because a scene's noise
depends on its batch.

**Third stratum.** `tools/add_intermittent_split.py` writes
`verified_seed777/eval_splits_3strata.json` with `intermittent_heldout`: the
expert-clean scenes the base solves 1-3 times out of 4. That is 223 of the 628
failure candidates plus 54 of the 2564 verified successes, about twice the size
of the verified-failure stratum, and none of them is in any training pool (the
tool checks this). `population.strata` is copied unchanged, so the trainer's
logged marginal stays comparable. The three-stratum weights are in `strata3`.

`tools/select_analysis.py` reports, per arm and per checkpoint mean:

* paired `recovered` / `broken` rates vs the base. Population-weighted,
  recovered − broken *is* the delta;
* `marginal2` and `marginal3` deltas with a stratified scene-bootstrap 95% CI;
* the oracles `any(base, ft)` (same seed) and `any(base_r, base_r')` (two
  independent base episodes), plus `selection_gap3` = oracle − selector. A large
  gap means the selector is the bottleneck. A small oracle gain means the
  proposals are.

```bash
DFT_SELECT_SMOKE=1 bash rl/diffusion_ft/slurm/submit_select_eval.sh   # 4 scenes/split first
bash rl/diffusion_ft/slurm/submit_select_eval.sh    # stopping base_s{0,1,2}@110, seed 4242
sbatch rl/diffusion_ft/slurm/select_analysis.sbatch <select_dir>/analyze.env   # redraw, CPU
```

Seed 4242 is the hardness gate's seed, so its base and ft rates should agree in
distribution with `gate/20260913_140701_1214568`. If the selector thresholds are
tuned on these results, the confirming run needs a fresh seed.

### Experiment 2: preservation on sampled trajectories (`train_safety --preserve_weight`)

The base anchor compares one-step posterior means on **noised human**
trajectories, at t drawn from the whole schedule, on the **fine-tune's** visited
states. Its logged share of the objective was about 1e-5 throughout the stopping
sweep. The new term instead:

1. rolls the **frozen base** out on 16 success-pool scenes per iteration: up to 8
   successes the current policy recently broke (replay: capacity 256, 10 draws
   each, re-breaking refreshes), half of the rest from the top risk quintile, the
   remainder uniform;
2. keeps each replan's prefix and late key, and only the rows where **the base
   solved the episode**;
3. penalizes the difference between the trainable late chain and the base late
   chain from that same prefix and noise, in world space (position / 1 m,
   wrapped heading / 0.1 rad, speed / 1 m/s) over the executed window. The term
   is exactly zero at the base weights.

`--safety_rows unsolved` removes the barrier from rows whose collection rollout
succeeded. Otherwise the softplus still pushes a tight successful manoeuvre
near its margin. Both flags default off, which is the old behaviour.

New log keys: `preserve_rms_m` (metres the fine-tune's executed segment has
moved off the base on base-solved rows, the number to read), `preserve_share`,
`preserve_base_solved`, `replay_size`, `replay_added`, `safety_rows`.

```bash
DFT_ARMS="pres3_s0" DFT_ITERS=2 DFT_EVAL_EVERY=1 DFT_SPLIT_CAP=8 bash rl/diffusion_ft/slurm/sweep_preserve.sh  # smoke
bash rl/diffusion_ft/slurm/sweep_preserve.sh    # srow / pres3 / pres30 x 3 seeds, stopping protocol
```

The control is the 09-11 stopping sweep's `base_s{0,1,2}`, run under the same
protocol. Add `ctrl_s*` through `DFT_ARMS` to rerun it under the new code. To
compare, read `marginal/delta_vs_base` at matched iterations, then score the
chosen snapshot per scene with `DFT_SELECT_CKPTS=... submit_select_eval.sh` for
the paired recovered/broken split.

### Counterfactual branching: repair search, learned and look-ahead selectors (2026-09-13)

`branching.py` snapshots a closed-loop episode at any replan and continues it
with the base. A snapshot is a set of references to the env's immutable pytrees.
Continuations reuse the original episode's per-replan sampler keys, so a row with
no override replays the original bit for bit, because it has the same state,
batch and noise tensor. Every repair search also runs an `id` branch and records
whether it reproduced the base (`id_consistent`). **Read that first**: below
1.0, the engine is not replaying the episode and nothing downstream is reliable.

**Repair search** (`counterfactual.py --mode repair`, on `failure_train`):

1. Take the base episode's first violating replan. A clean scene that never
   reaches the goal is charged to its last replan.
2. Rewind 0, 1, 2 replans before it. Execute a perturbed base candidate for
   `hold` replans, then resume the base with the original keys.
3. Keep a repair only if the whole continuation solves the scene. Per scene the
   kept repair is the latest rewind that works, and within it the smallest edit.

Candidate families:

| family | token | what it tests |
|---|---|---|
| time warp | `tw0.7`, `tw1.3h2` | same path at a×speed, eased in over 1 s: yield or arrive later / go earlier. Neighbours are log-replayed, so timing alone can dissolve a conflict |
| lateral | `lat±0.25`, `lat±0.5` | smoothstep shift of the line along its normal, continuous at the join |
| re-noise | `rn20s0`, `rn50s1` | partially re-noise the base candidate to step t and denoise with the base: a different manoeuvre |

The outputs are `repairs.jsonl`, plus `repair_data/*.npz` holding (frozen
condition, executed candidate in normalized space, covered-step mask) per applied
replan. `tools/repair_summary.py` reports the repaired rate by failure type, by
family and by rewind depth.

**The inverse is lossy past the scenario end, and that is not a bug to fix.**
`preprocess` clamps `world_t_seconds` at the end of the scenario: every world
sample beyond it repeats the last valid pose. The forward map can emit those
samples; `world_to_norm` cannot recover the model steps behind them, because the
information is gone. Measured on the first repair smoke (2026-09-13) as a **0.256
normalized / 25.6 m** round-trip error, entirely in the tail. So
`branching.model_step_valid` marks the covered model steps, targets carry the
BASE proposal on the rest (never executed, and a sane thing to be taught), and
`--repair_weight` weights the denoising loss by that mask. The job re-checks the
round trip on covered steps at startup: it warns above 10 cm and refuses to write
targets above 1 m. Measured on the fixed smoke: **2 cm** on covered steps (0.2563
normalized on the masked tail, 85% of steps covered). That 2 cm is expected --
`world_dt` is measured per scene and does not divide the 0.2 s model grid, so the
inverse interpolates between world samples rather than landing on a breakpoint --
and it is far below the 0.3 m edge margin. Repair data written before the mask existed has no `valid`
array and the trainer rejects it.

**Correction training** (`train_safety --repair_data --repair_weight`): the
trainable late copy learns the verified repairs through
`DiffusionRLActor.late_denoise_loss`, an x0-prediction loss at t < k_trainable
only, the steps that copy actually drives. The copy starts equal to the base,
so this is a zero residual at iteration 0. The trainer refuses repair data that
contains a held-out scene.

**Selector labels** (`--mode labels`): at each replan of a base episode, execute
the fine-tune's candidate (same prefix and noise) at that replan only, then
resume the base. Each (scene, replan) gets a label
`solved(switch) − solved(base)` next to the selector features.
`tools/fit_selector.py` fits a 3-class logistic regression with population
weights (each stratum at its corpus weight, each scene equal within it) and
picks the switch threshold θ ≥ 0 by scene-grouped cross-validation. It reports,
out of fold, the mean single-decision effect of the learned rule, `cons`, and
always-switch on the same rows. A single-decision effect is not a
multi-switch episode delta; the selection eval measures that.

**New selector arms** in `proposal_select`:

* `look`: where either candidate is within 0.3 m of a hazard, run each one plus
  `--look_replans − 1` base replans from a snapshot, restore, and switch only if
  that removes a violation without ending farther than `tol` from the goal. Both
  branches share their look-ahead keys.
* `learned`: applies `selector.json` (from `--selector_json`).

```bash
DFT_CF_SMOKE=1 bash rl/diffusion_ft/slurm/submit_counterfactual.sh   # smoke: every stage, tiny
bash rl/diffusion_ft/slurm/submit_counterfactual.sh
#  repair (GPU) -> train repair / repair_pres x 3 seeds (GPU)
#  labels (GPU) -> fit_selector (CPU) -> select base,bf_cons,bf_look,bf_learned (GPU) -> analyze (CPU)
```

First smoke (2026-09-13, 8 failure scenes): `id_consistent` 1.000 -- the
do-nothing branch reproduced the base episode on every scene -- and 3 of 8
repaired, one per failure mode and each by the family the mode predicts: a
collision by arriving EARLIER (`tw1.3`, rewind 2), an off-road by a 0.5 m lateral
shift, a never-reached goal by speeding up. Worth watching at full scale: that
smoke's 113 one-switch labels were ALL zero (no single switch changed an
episode's outcome on 16 scenes). If that holds on the full pool, the learned
selector has nothing to fit, and the per-chunk `+1 / -1` counts say so early.

Seeds: the counterfactual jobs refuse 777, 4242 and the eval seed, and use 2718.
The selection stage reuses 4242 with the same shards and chunk as
`submit_select_eval.sh`, so its `base` / `bf_cons` rows pair with an Experiment 1
run on `base_s0@110`. Point `select_analysis --rows` at both directories to merge.

CPU tests: `sbatch rl/diffusion_ft/slurm/test_proposal_select.sbatch` runs
`test_proposal_select`, `test_branching` and the existing trainer suites.

## Results (2026-09-14)

Everything below is ONE overnight batch: `select/20260913_234334_1261057`,
`counterfactual/20260913_234710_1263710`, `preserve/20260913_234938_1263938`.
The CPU suites (147 checks, 7 files) pass.

### 1. Selection beats the base, and nothing else here has

Paired per scene at verify seed 4242, 4 repeats, 1450 held-out scenes, three
fine-tuned checkpoints (stopping `base_s{0,1,2}` @110) averaged. `m2` is the
two-stratum marginal every earlier run reports; `m3` adds the intermittent stratum.

| arm | m2 Δ | m3 Δ (95% CI) | recovered | broken |
|---|---|---|---|---|
| **`bf_cons10`** base + ft, 1 s selector window | **+0.0033** | **+0.0077** [+0.0064, +0.0092] | 0.0077 | **0.0000** |
| `bf_cons` same, 2 s window | +0.0022 | +0.0058 [+0.0048, +0.0070] | 0.0058 | 0.0000 |
| `bf_margin` near-miss rule, 2 s | +0.0020 | +0.0051 | 0.0051 | 0.0000 |
| `bb_cons` **second base sample** (matched compute) | +0.0018 | +0.0039 [+0.0026, +0.0055] | 0.0039 | 0.0001 |
| `bfull_cons` independent base sample (2x compute) | +0.0015 | +0.0036 | 0.0037 | 0.0001 |
| `ft` the fine-tune everywhere | −0.0101 | −0.0062 | 0.0128 | 0.0190 |

Against the hardness gate's **+0.0005** (of a then-computed +0.0039 ceiling), the
conservative selector is **+0.0033 on the same measure** — about 7x the gate, at
what was thought to be the ceiling for routing this fine-tune.

Three things carry:

* **It breaks nothing.** `broken` is 0.0000 and the switch rate on the success
  stratum is 0.000: the rule fires on 4.8% of failure-scene replans, 3.0% of
  intermittent ones, and never on a clean base candidate. Do-no-harm — the
  constraint that sank every arm from 09-02 to 09-13 — stops binding when the base
  is the default and the burden of proof is on the intervention.
* **The fine-tune's candidate really is better than another base draw**: +0.0077 vs
  +0.0039, CIs disjoint. A full independent base sample at 2x the compute is no
  better than the cheap late-chain redraw, so this is not "more samples".
* **The intermittent stratum carries much of it**: 0.496 → 0.646, against
  failure 0.046 → 0.182 and success 0.997 → 0.998. That is the 3% of the corpus
  the two-stratum marginal excludes entirely, and it is where selection pays.

A SHORTER selector window wins (1 s over 2 s): scoring 2 s ahead condemns
violations the next replan would have removed, so the alternative looks dirty and
the switch does not happen (switch rate 0.048 at w10 vs 0.030 at w20).

### 2. Repair search: half of the failures are repairable, and timing is half the fix

`failure_train`, 168 searched failures, `id_consistent` **1.0** (the do-nothing
branch reproduced the base episode on every scene, so the rewind machinery is exact):

| | repaired | rate |
|---|---|---|
| off-road | 41 / 76 | 0.54 |
| collision | 33 / 62 | 0.53 |
| never reached the goal | 11 / 30 | 0.37 |
| **total** | **85 / 168** | **0.51** |

Kept repairs by family: **timing 42, lateral 40, re-noise 3**. Rewind depth: 34 at
the failing replan, 31 one back, 20 two back — rewinding earlier is worth a third
of the repairs, so "fix it where it broke" would have missed them.

That is the review's hypothesis, measured: a clearance gradient cannot express
*arrive later*, and half of these fixes are exactly that. 100 training targets.

### 3. Preservation removes the damage but does not buy a win

3-seed means at matched iterations — the honest statistic, since per-seed bests
select on noise and the seed spread is ~0.03:

| config | best 3-seed mean margΔ | mean over all 16 eval points |
|---|---|---|
| `base` (09-11 stopping sweep) | −0.0093 | −0.0322 |
| `srow` (safety on failed rollouts only) | −0.0024 | −0.0306 |
| `pres3` | +0.0019 | −0.0104 |
| **`pres30`** | **+0.0021** | **−0.0010** |
| `repair` (safety OFF, learn verified repairs) | +0.0004 | −0.0016 |
| `repair_pres` | +0.0009 | −0.0100 |

Averaged across the curve `pres30` sits at −0.001 where the old centre sat at
−0.032: **the preservation term removes essentially all of the success-split
damage**, which is a change in kind. It does not produce a gain — +0.002 against a
0.03 seed spread is not a win — and it costs failure recovery (`pres30` reaches
0.18-0.21 on the failure split where `srow` reaches 0.27-0.28). Read it as: the
do-no-harm problem is solvable in training, and solving it there buys parity, not
profit. Selection buys the profit.

`srow` alone is flat, so preservation is the active ingredient, not the row mask
(the two-term discipline: zero the other weight to attribute).

Learning from verified repairs (`repair`, safety term entirely off) is the first
objective in this package that is neither a loss nor a wash — flat at −0.002 with
success at 0.998-0.999 — but its failure split stays at 0.05-0.10. 100 targets from
85 scenes is a small dataset; the repair search is cheap enough to grow it.

### 4. The learned selector is worse than the hand-written one, and why

| selector (ckpt `base_s0` only) | m3 Δ | recovered | broken |
|---|---|---|---|
| `bf_cons10` (hand-written, 1 s) | +0.0083 | 0.0083 | 0.0000 |
| `bf_look` (execute + 1 base replan, then judge) | +0.0088 | 0.0088 | 0.0000 |
| `bf_learned` (fit to 12201 one-switch labels) | +0.0085 | 0.0134 | **0.0049** |

The labels are real (failure stratum +50/−4, success +21/−17 over 2173 scenes), but
**the threshold selection is flawed**: `fit_selector` picks θ by maximizing the
population-weighted mean *single-decision* effect, and that objective cannot
penalize switching often — a switch with zero expected effect is free under it. It
chose θ=0, which switches on **63% of decisions**, and in closed loop that damages
successes (0.0049) for no net gain over the conservative rule. The fix is to score
candidate θ by the closed-loop episode outcome (or to add an explicit switch-rate
cost / a floor on predicted benefit), not by a per-decision sum. Until then prefer
`cons`.

`look` is within noise of `cons10` at several times the cost, so single-step
geometry already captures the benefit and the look-ahead is not currently worth it.

### Where this leaves it

* **Deploy-time selection is the lever**, not the objective, not the stopping
  point, not the gate. The base drives; the fine-tune proposes; a two-line
  geometric rule decides; nothing regresses.
* **The measurement that made it visible** is the intermittent stratum plus paired
  per-scene scoring: the training curve's ~0.03 seed spread cannot resolve a 0.003
  effect, and the paired selection eval resolves it with CIs an order of magnitude
  tighter.
### The three follow-ups, run 2026-09-14

**1. θ by episode outcome -> the learned selector is dead.** `theta` is now the
4th arm field (`ft:learned:20:0.35`) and `fit_selector` picks its default under
`--max_switch_rate` (0.01) while reporting the whole curve and the unconstrained
argmax it no longer trusts. Two measurements killed the learned rule:

| θ | switch rate | per-decision effect (OOF) | closed loop (m3 Δ) |
|---|---|---|---|
| 0.00 | 63% | +0.00083 | +0.0085, **broke 0.0049** |
| 0.01 | 0.95% | +0.00026 | — |
| 0.02 | 0.30% | +0.00011 | — |
| 0.04 | 0.10% | +0.00005 | — |
| 0.10 | — | — | +0.0004 |
| >= 0.30 | ~0 | ~0 | **+0.0000** (never fires) |
| **`cons`** | **0.12%** | **+0.00046** | **+0.0083** |

At a MATCHED switch rate the two-line geometric rule beats the fitted score nine
to one on the fit's own metric (0.12% -> +0.00046 vs 0.10% -> +0.00005). The
learned selector only looked competitive at θ=0, where it switches constantly and
pays for it in broken successes. It is not ranking decisions better than
`clear_p < 0 and clear_a >= 0`; a better feature set would have to come first.
`look` (execute + 1 base replan, then judge) matched `cons10` at several times the
cost, so single-step geometry already captures the benefit.

**2. A bigger repair set -> more data does not move the correction arm.**
`DFT_REPAIR_SPLITS` / `DFT_REPAIR_CAPS` / `DFT_CANDS` / `DFT_REWINDS` are
submitter knobs now (held-out splits still refused). Pointing the search at the
success pool as well, with 19 candidates instead of 15:

| pool | scanned | base failures | repaired | rate |
|---|---|---|---|---|
| `failure_train` | 173 | 166 | 100 | 0.60 (was 0.51 at 15 candidates) |
| `success_train` | 4000 | 44 | 40 | **0.91** |
| total | 4173 | 210 | 140 | 0.67 |

160 targets, up from 100. The base's own failures inside the SUCCESS pool are
rare (1.1%) and easy -- 91% repairable -- which says they are marginal rather
than structural. But the arms trained on 160 targets land at best 3-seed mean
**−0.0001** (curve average −0.0022) against +0.0004 / −0.0016 on 100: unchanged.
**The correction objective is not dataset-size-limited**, so the next question is
what it IS limited by, not how to collect more.

```bash
DFT_CF_PARTS=repair DFT_NUM_SHARDS=4 \
  DFT_REPAIR_SPLITS=failure_train,success_train DFT_REPAIR_CAPS=-1,4000 \
  DFT_CANDS="id,tw0.85,tw1.15,lat0.25,lat-0.25,tw0.7,tw1.3,lat0.5,lat-0.5,lat0.75,lat-0.75,tw0.6h2,tw1.4h2,tw0.5h2,tw1.6h2,rn20s0,rn20s1,rn50s0,rn50s1" \
  bash rl/diffusion_ft/slurm/submit_counterfactual.sh
```

A chunk with no failure costs one base rollout and exits the rewind loop at once,
so scanning 4000 success scenes is far cheaper than 4000 searches.

**3. Selection on the checkpoints these experiments produced** -- the result is
the section below, and it is the most surprising thing here.

## The ranking inverts under selection (2026-09-14, `select/20260914_094848_1315921`)

**Training a fine-tune to do no harm makes it a WORSE proposal generator.** Same
selector (`cons`, 1 s window), same scenes, seeds and pairing; only the checkpoint
that supplies the alternative changes:

| alternative comes from | alone (m3 Δ) | broken alone | under `cons10` (m3 Δ) | broken |
|---|---|---|---|---|
| `srow_s0@110` (safety on failed rows, no preservation) | **−0.0026** | 0.0151 | **+0.0086** [+0.0071, +0.0102] | 0.0000 |
| `base_s0@110` (09-11 stopping centre) | −0.0062 | 0.0190 | +0.0083 [+0.0069, +0.0098] | 0.0000 |
| `pres30_s0@140` (damage removed by preservation) | +0.0017 | 0.0043 | +0.0027 [+0.0017, +0.0038] | 0.0000 |
| `repair_s0@120` (learned only from verified repairs) | +0.0021 | 0.0039 | +0.0006 [+0.0002, +0.0012] | 0.0000 |

Read the first two columns against the last two: the standalone ranking is
`repair > pres30 > srow > base`, and the ranking as a PROPOSAL GENERATOR is
exactly reversed. `srow` is the worst policy in this package to deploy on its own
(it breaks 1.5% of successes) and the best alternative to offer a selector.

Why: every mechanism that removes damage does it by making the fine-tune more
base-like, and an alternative that agrees with the base has nothing to contribute
when the base is already the default. It shows up in the switch rate — `cons`
fires on 5.5% of failure-scene replans with `srow`, 1.3% with `pres30` and 0.4%
with `repair`, because the repair-trained policy so rarely proposes anything
different that there is nothing to select. The failure split under selection
tracks it directly: 0.215 / 0.182 / 0.071 / 0.051.

**So do-no-harm training and good-alternative training are opposite objectives,
and the selector already supplies the safety.** `pres30`'s +0.0017 standalone is
the best any fine-tune has scored alone here, and it is still half of what the
aggressive checkpoint buys through a selector. Preservation was solving a problem
that deployment structure dissolves.

Caveat on the summary's `ckpt_mean` row: this run scored three DIFFERENT configs,
so that row averages apples with oranges (it is a seed mean only when the
checkpoints are seeds of one config). The key was renamed from `seed_mean` for
exactly this reason; read the per-policy rows here.

Open: `srow` was never swept for seeds or stopping point as a *proposal
generator* — it was only ever measured as a policy, where it looks bad. The axis
worth sweeping now is how aggressive the alternative may be before the selector
stops filtering it (`srow` at later iterations, or the unanchored arm).

## The repair bank as a selector alternative (2026-09-15)

**The asset the correction objective could not use.** `counterfactual --mode
repair` verifies 140/210 failures repairable, and the kept repairs are **timing
42, lateral 40, re-noise 3** — 82 of 85 are two geometric edit families. Training
on them does not transfer (best 3-seed mean +0.0004 at 100 targets, −0.0001 at
160: *not* dataset-size-limited). The ranking inversion above says why that is not
the end of it: the checkpoint is valuable as a **proposal generator**, not as a
policy, and a repair is a *decision* ("slow down at replan 3 here") whose
condition a denoising loss at `t ~ U[0, k_trainable)` throws away while it
imitates the shape.

So offer the same interventions at decision time. `proposal_select` gains a
fourth alternative, `bank`:

| alt | where the alternative comes from | denoiser evals per replan |
|---|---|---|
| `ft` | fine-tuned late chain, same prefix and late noise | `k_trainable` |
| `base2` | base late chain, fresh late key | `k_trainable` |
| `basefull` | independent full base sample | 100 |
| **`bank`** | **`tw`/`lat` edits of the base's OWN proposal** | **0** |

`time_warp` and `lateral_shift` are pure functions of a trajectory already drawn,
so a bank of K costs K jitted scorer calls and **no network**. A bank arm is
therefore cheaper than `bb_cons` and needs no checkpoint at all — it lives in
`DFT_BASE_ARMS`.

**What preserves the do-no-harm property.** Every candidate faces the *same*
per-candidate rule as a single alternative (`select_switch`), so if the base
proposal is predicted clean no candidate is eligible, however large the bank —
this is the property that made `broken` 0.0000, and `test_proposal_select`
pins it over 2000 random rows x 8 candidates. Among the eligible, the winner is
the **smallest** perturbation by `Candidate.magnitude`, the repair search's own
ordering. Ranking by magnitude rather than by clearance is deliberate: it makes
the bank a minimal intervention instead of a licence to take the most aggressive
edit that happens to score well (pinned: "smallest edit beats the clearest one").

**Scope cuts, and why.** `id` is refused (it *is* the base proposal — a switch
that changes nothing); `h<n>` is refused (the selector re-decides every replan,
so holding has no meaning); `rn` is refused (a denoiser eval per candidate, which
is the whole property this alternative has — and it was 3 of 85 kept repairs, so
it is not what the search says is worth buying). `bank` + `learned`/`look` is
also refused: neither has a defined meaning over K candidates, and silently
scoring only the chosen one would make the arm a different experiment than its
name says.

**Data kept for a re-rank.** The npz stores `choice` (which candidate fired, −1
= base) and `clear_k` (every candidate's clearance, `[B, R, K]`), so the bank can
be re-ranked or a selector fit offline without re-rolling anything.

**One analysis fix this forced.** `selection_gap3` was computed for every arm
against the `any(base, ft)` oracle. A bank arm selects over a different candidate
set, so that would report a gap against a ceiling it was never offered; the gap
is now emitted only for arms whose alt IS `ft`, read from the runner's own meta
(`analyze(..., alt_of=)`).

```bash
# default: two bank arms added to the base-only (checkpoint-free) set
bash rl/diffusion_ft/slurm/submit_select_eval.sh
# bank only, no fine-tune involved anywhere:
DFT_FT_ARMS=ft DFT_BASE_ARMS="base,bank_cons10=bank:cons:10" \
  DFT_SELECT_BANK="tw0.85,tw0.7,lat0.25,lat-0.25" bash rl/diffusion_ft/slurm/submit_select_eval.sh
DFT_SELECT_SMOKE=1 bash rl/diffusion_ft/slurm/submit_select_eval.sh   # 4 scenes/split first
```

### Result (`select/20260915_130443_1678068`, complete)

Same protocol as the 09-13 batch: verify seed 4242, 4 repeats, paired per scene,
stratified scene-bootstrap CIs. `bf_cons10` reproduces its published
+0.0033 / +0.0077 to the third decimal, so this is the same measurement.

| policy | m2 Δ | m3 Δ (95% CI) | recovered | broken |
|---|---|---|---|---|
| **`bank_cons`** (w20) | **+0.0069** | **+0.0178 [+0.0157, +0.0202]** | 0.0181 | 0.0003 |
| `bank_cons10` (w10) | +0.0055 | +0.0137 [+0.0117, +0.0160] | 0.0137 | **0.0000** |
| `bf_cons10` — the 09-14 winner | +0.0035 | +0.0083 [+0.0068, +0.0099] | 0.0083 | 0.0000 |
| `bb_cons` | +0.0018 | +0.0039 | 0.0039 | 0.0001 |
| `ft` alone | −0.0119 | −0.0082 | 0.0130 | 0.0212 |

Per split, `bank_cons`: failure 0.046 → **0.361**, intermittent 0.583 → **0.914**,
success 0.9968 → **0.9990** — it gains on all three, so it never spends the
66:1 budget at all. **2.1x the previous best, CIs disjoint, no checkpoint, no
gradient and no denoiser evaluation anywhere in the arm.**

**It clears the oracle every earlier selection result was measured against:**

```
any(base, ft) same seed    +0.0128
any(base, ft) independent  +0.0149
bank_cons                  +0.0178
```

Not a paradox — the bank offers candidates outside the fine-tune's set — but
`selection_gap3` against the ft oracle is no longer the right bound for this line,
which is why it is now emitted only for arms whose alt IS `ft`. Against the m3
ceiling for a perfect fixer (~+0.032) this is ~56%.

**The runtime invariant held on all 1490 switches: zero fired on a clean base
candidate.** Switch rate 2.9% (`bank_cons`) against `bf_cons10`'s 1.1%.

### Trap: a `;` in an env-file value is a shell statement separator (2026-09-16)

The submitter writes one env file per task and the batch script **sources** it, so
a bare `KEY=value` line is a shell assignment. `--select_banks` uses `;` between
named banks, and an unquoted `;` **ends the assignment**:

```
DFT_SELECT_BANKS=lat=lat0.25,lat-0.25;tw=tw0.85,tw1.15
  -> DFT_SELECT_BANKS='lat=lat0.25,lat-0.25'   and   tw='tw0.85,tw1.15'
```

The timing half was silently dropped into a stray variable, so the `tw` ablation
arm referenced a bank that did not exist. Jobs 55580 / 55642. Two things to keep:

* **The runner's startup guard is what caught it** -- `--arms reference bank(s)
  ['tw'] that --select_bank/--select_banks do not define (have ['', 'lat'])`,
  30 s in, before a scene was scored. Without it the run would have completed with
  `bank_tw` quietly scoring an empty bank (never switching, therefore identical to
  `base`) and the ablation would have "shown" that timing contributes nothing.
* **Inspecting the env FILE is not the check.** `grep` showed the full line,
  because the truncation happens in `source`, not on disk. The check is to source
  it in a clean shell and compare.

Fixed both ways: values are written single-quoted (`kv`), and the submitter now
sources each env file in an `env -i` shell and requires every `DFT_*` key to come
back byte-identical **before** any `sbatch`. This is the same family as the
`--export` trap in the root CLAUDE.md -- configuration that travels through a
shell needs its round trip tested, not assumed.

### Seed replicate (`select/20260916_175340`, seed 5150)

The bank arms had no seed replication in the first run. A second `select_seed`
on one checkpoint:

| policy | m2 Δ (4242 → 5150) | m3 Δ (4242 → 5150) |
|---|---|---|
| `bank_cons` | +0.0069 → **+0.0073** | +0.0178 → **+0.0187** |
| `bank_cons10` | +0.0055 → +0.0049 | +0.0137 → +0.0133 |
| `bf_cons10` | +0.0035 → +0.0036 | +0.0083 → +0.0082 |

Seed spread is ~0.001 on m3, an order of magnitude below the effect, and the
2.1x ratio over `bf_cons10` reproduces at 2.3x. The result is not one seed's luck.

### Two things this does NOT establish

**1. The mechanism is not the one the bank was built on.** The candidate mix is
**88% lateral / 12% timing** (1316 / 174), against the repair search's verified
50/50. So "half the fixes are arrive-later, which a clearance gradient cannot
express" — the argument for building this — is *not* what produced the gain. The
cause is the magnitude ladder mixing units (`lat0.25` = 0.35 ranks below
`tw1.15` = 0.38), combined with an admissibility test far weaker than the repair
search's: a one-step geometric prediction admits many more candidates than
"the whole continuation solves the scene", so the magnitude tie-break binds much
harder and the cheapest lateral almost always wins.

`--select_banks` exists for exactly this ablation: named banks referenced as
`bank@lat` / `bank@tw`, so lateral-only and timing-only run as arms of ONE job and
are paired draw for draw rather than compared across submissions.

**2. The window result inverts, unexplained.** For the `ft` alternative the
SHORTER window won (`cons10` > `cons`: scoring 2 s ahead condemns violations the
next replan would have removed). For the bank the LONGER window wins, and clearly
(0.361 vs 0.282 on failures) — and it is also where the only damage appears
(`broken` 0.0003 vs 0.0000). Different mechanism, no account of it yet.

Also open: the bank arms have **no seed replication**. They are checkpoint-free,
so they ran once on `select_seed` 4242 with 4 repeats; the CI is a scene
bootstrap, not a seed spread. The ft arms got three checkpoint seeds.

## Where the safety loss is evaluated (2026-09-14)

Three proposals now exist for the trajectory the geometric loss is charged on.
They differ in **which diffusion state the denoiser is corrected at**, and that
is the only axis — the environment states are the policy's own in all three
(`--anchor_states policy`, the default: reset, plan, execute, record the replan
contexts, differentiate them afterwards).

| flag | state the denoiser sees | cost/gradient | the defect |
|---|---|---|---|
| `--unroll_steps 0` (default) | free-running: 100 frozen+trainable steps from pure noise | 100 evals, 10 with gradient | none, and it is the expensive one |
| `--unroll_steps K` | `q_sample(x0_expert, t)` — a noised **human** trajectory | K evals | diffusion-state mismatch: the sampler's own `x_t` is not drawn from this |
| `--tb_weight w` | a stored `x_t` the sampler **actually passed through** | K evals (K=1 default) | staleness — an older policy produced it |

`--reuse_prefix` (on by default) is orthogonal and applies to the first row only:
the gradient replays the executed sample's late steps from its stored frozen
prefix, so what is scored is what was driven.

### Transition replay (`rl/diffusion_ft/transition_buffer.py`)

`sample_with_trace` already records `(x_t, t)` for each of the `k_trainable` late
steps on its way to the trajectory the env executes; the safety path had been
throwing that trace away (only DPPO consumed it). Keeping it costs one
device-to-host copy and no denoiser work, and each record is a genuine sampler
state at a known low-noise timestep. One datum is one transition:

    (scene_id, t0, x_t, t, cond1, cond2, cond2_mask, aux)

Nothing in it depends on the weights, which is what makes it replayable. The
geometry is recovered by reloading the scene by id — `candidate_safety` reads
only `log_trajectory`, `object_metadata` and `roadgraph_points`, all constant for
a whole episode — so a replayed minibatch mixes scenes and replan boundaries.
That is what the per-row `t0` path in `safety_objective._window_slice` is for.

Two ways to score a stored state, both reusing `unrolled_clean_prediction`:

* `--tb_unroll_steps 1` — the one-step clean prediction `x0_hat(x_t, t)`, one
  network evaluation, no unroll. The penalised object is a posterior **mean**, so
  it carries the Jensen gap `unrolled_prediction`'s docstring argues about.
* `--tb_unroll_steps K > 1` — unroll `K-1` eta-DDIM steps down to
  `--tb_unroll_t_end`, score the final clean prediction. `--tb_grad_steps 1`
  (default) keeps the gradient on exactly one transition however long the unroll.

The buffer is host memory, FIFO, `--tb_capacity` slots (~40 KB/slot, dominated by
`cond1`); `--tb_max_uses` caps draws per slot. It is **not** checkpointed — a
resume starts it empty and refills at
`scenes_per_iter x max_replans x tb_store_per_replan` per iteration. Watch
`tb_age` in `train_log.jsonl`: it is how stale, in iterations, the states the
gradient was taken at have become. `--safety_weight 0 --tb_weight 1` is the
pure-replay arm (the rollout still runs — it is what fills the buffer).

```bash
bash rl/diffusion_ft/slurm/sweep_proposal.sh              # 15 arms: ctrl/tf1/tf4/tb1/tb4 x 3 seeds
sbatch rl/diffusion_ft/slurm/test_transition_buffer.sbatch  # CPU gates first
```

`sweep_proposal.sh` also carries two arms outside the default set: `tbmix`
(buffer **and** on-policy term) and `tbfresh` (capacity == batch, so every drawn
transition is one this iteration produced — the on-policy control that isolates
staleness from everything else).

### Traps found building this (2026-09-13/14)

All four were silent in the sense that matters: nothing crashed at the point of
the mistake, and three of them would have produced plausible-looking numbers.

* **The world->normalized inverse is lossy past the scenario end.** 25.6 m of
  tail error in every repair target, because `preprocess` clamps
  `world_t_seconds` and the forward map's samples there are a frozen pose. Caught
  by a round-trip check written before any target was consumed; now masked
  (`branching.model_step_valid`), the mask ships WITH the data, and the trainer
  refuses repair data that predates it. The check warns above 10 cm and refuses
  above 1 m; ~2 cm is the expected discretization residual.
* **`env.batch_size` before the first reset is 0.** The env learns its batch size
  in `reset`, so sizing per-scene arrays on the line above the rollout gave empty
  arrays; the first repair job died on the first broadcast. Now taken from the
  finished episode.
* **`seed_mean` was a mean over whatever checkpoints were scored.** Fine for
  three seeds of one config, wrong for the three DIFFERENT configs of the
  checkpoint comparison. Renamed `ckpt_mean`, with the distinction in the
  docstring.
* **`test_bc_term` was already broken at `d640540`** -- its fake trainer predates
  the 09-09 anchor-window change and never set `_anchor_tw`. Unrelated to this
  work, found because the new suite runs the old tests alongside the new ones.

The pattern worth keeping: every new data path here (repair targets, labels,
branch replays) got a cheap invariant checked at runtime -- the round trip, the
`id` branch's `id_consistent`, the held-out-scene guard on `--repair_data`. Two
of the four above were caught by those, not by tests.

### The teacher-forced unroll was never actually cheap (fixed 2026-09-14)

Measured on the first proposal sweep, per-iteration wall time after the
first-iteration compile:

| arm | s/iter |
|---|---|
| `ctrl` — free-running, 100 evals under gradient | **5.96** |
| `tb1` / `tb4` — replayed transition | 6.3 – 7.9 |
| `tf1` / `tf4` — teacher-forced, 1–4 evals | **34.5** |

The proposal whose entire claim is `K` network evaluations instead of 100 ran 6x
SLOWER than the thing it was meant to undercut. Not a property of the method:
`_grads_for` routed `unroll_steps > 0` to `_loss_for`'s un-jitted closure, which
re-traced the unrolled sampler + geometry graph once per context per iteration
(8x), while `unroll_steps == 0` went through the cached `_jit_free_grad`. The
unroll now shares `_free_core` behind a static `has_unroll` branch;
`--unroll_eager` keeps the old closure as the equivalence reference, and
`tests/test_unroll_jit.py` pins the two key streams equal under every combination
of the terms that consume the stream ahead of the proposal (anchor / bc / base).
The paths are numerically equivalent, not bit-identical — XLA fusion reassociates
floats — so a `tf` arm from before the fix is still comparable to one after it.

**The second finding is the one that matters, and it survives the fix:** `tb`
costs the same as `ctrl` despite doing 1 network evaluation under gradient
instead of 8x100. **Collection dominates** — the 100-step sampler x 8 replans
runs in every arm, because policy-visited states require it. So a cheaper
gradient buys much less than `unrolled_prediction`'s docstring implies, whichever
cheap proposal you pick. If iteration cost is the target, the thing to attack is
collection (`--collect_steps`, the DDIM-strided prefix), not the gradient.

### Proposal sweep result (15 arms x 150 iters, 3 seeds, complete 2026-09-15)

Seed means at matched iterations; `base_marginal = 0.98234`. "best" is the eval
point with the best SEED-MEAN delta, so it still selects a stopping point but not
a lucky seed.

| arm | fail_held @149 | succ_held @149 | delta @149 | best delta (iter) | seeds > 0 |
|---|---|---|---|---|---|
| `ctrl` free-running | 0.341 ±0.012 | 0.951 ±0.005 | −0.0408 | −0.0057 (110) | 0/3 |
| `tf1` teacher-forced, x0_hat | 0.239 ±0.022 | 0.991 ±0.004 | **−0.0026** | **−0.0007 (100)** | 0/3 |
| `tf4` teacher-forced, unroll | 0.333 ±0.010 | 0.940 ±0.034 | −0.0518 | −0.0091 (110) | 0/3 |
| `tb1` replayed, x0_hat | 0.197 ±0.025 | 0.983 ±0.015 | −0.0108 | **+0.0007 (140)** | **3/3** |
| `tb4` replayed, unroll | 0.291 ±0.022 | 0.980 ±0.006 | −0.0123 | −0.0122 (140) | 0/3 |

**Three findings, in decreasing order of confidence.**

1. **`K=4` buys ~+0.09 failure-split gain over `K=1` in BOTH families** (tf 0.333 vs
   0.239, tb 0.291 vs 0.197) — comfortably outside the ±0.02 seed spread. That is
   the Jensen-gap prediction in `unrolled_prediction`'s docstring, made in advance
   and confirmed independently on teacher-forced and replayed states. **It does
   not convert into corpus value:** both `K=4` arms are worse at the margin than
   their `K=1` counterparts (−0.0091 vs −0.0007, −0.0122 vs +0.0007). `K=4` is not
   better, it is *more aggressive*, and it lands on the same failure↔success
   tradeoff every other knob does.
2. **`tf1` is the best proposal on this protocol.** It reaches 0.256 failure-split
   at its best point against `tb1`'s 0.198 — a +0.058 difference, outside the
   spread — for a corpus delta (−0.0007 vs +0.0007) that is not. More gain at
   indistinguishable cost.
3. **Transition replay works mechanically and does not win.** Removing the
   diffusion-state mismatch (the one defect teacher forcing admits to) changed the
   trade point but did not beat teacher forcing. `tb1` is the only arm positive in
   all 3 seeds, but at +0.0007 ±0.0003 that is *parity*, not a gain — and it is
   parity bought by being the weakest update in the sweep.

**No arm beats the frozen base at the corpus level.** Every family's seed-mean
delta is ≤ 0 at every one of the 16 eval points except `tb1`'s +0.0007. Read
together with the geometry and stopping sweeps, the proposal axis joins the list
of things that move the failure/success trade point without moving the margin.

## Does the barrier fire where nothing is wrong? (2026-09-15, built, not yet run)

Every number in this file about *where* the safety loss bites came from
`calibrate_expert.py`, which reports **clearance percentiles** — how close the
human gets. Percentiles cannot see the complementary question: how much of the
loss comes from geometry that is *already clear of the margin*, i.e. from terms
the barrier was supposed to have switched off.

The mechanism is structural, not a rounding error. The barrier

```
tau * softplus((margin - c) / tau)
```

has **infinite support**: at `c = margin + k*tau` it is still `~tau*exp(-k)`,
and its gradient is still `~exp(-k)`. Per term that is negligible — but the
barrier is *summed*, deliberately (`safety_loss.py` docstring: "applied per
corner and per disc pair and summed, rather than to an aggregated minimum"),
over 4 ego corners, over every valid other agent, and over every step. A 91-step
scene with 40 agents sums 364 edge terms and 3,640 agent terms, so a per-term
epsilon is multiplied by a factor that grows with **how crowded and how long the
scene is** — not with anything the policy did. An empty straight road and a
dense but perfectly safe queue of traffic are then not scored the same, and the
whole difference is traffic density. On the agent term the exposure is the
larger one: `agent_tau` is 0.25 m and WOMD's median expert gap is 1.90 m, only
~4.6 tau out.

### The probe: `tests/probe_barrier_tail.py` (`submit_probe_barrier_tail.sh`)

Splits every scene's loss **and its gradient** exactly two ways —

* **bite**, contributed by corners/pairs actually inside the margin (`c < margin`);
* **tail**, contributed by corners/pairs at or beyond it (`c >= margin`);

— because the loss split alone is cosmetic: a tail that is 30% of the loss but
3% of the gradient does not move the policy, and a tail that is 30% of the
*gradient* is a second, unintended objective ("stay away from everything,
always") riding along with the safety term. Scenes are then bucketed by whether
anything in them is even close:

* **trivial** — the expert log never comes within either margin at any step. The
  loss on these should be 0; whatever it is, is misfire, with no legitimate part
  to net off. This bucket is the answer to the question.
* **bites** — something is inside a margin at some step.
* **dirty** — the log is itself off-road, excluded as everywhere else.

It also correlates the trivial bucket's loss against the scene's mean agent
count (a safe scene charged for its traffic is exactly what that correlation
is), and re-scores the identical clearances under every barrier shape, so the
size of the fix is read off the same numbers rather than a second run.

Pure geometry over `log_trajectory`: no env, no checkpoint, no GPU, CPU
partition — same shape of job as `calibrate_expert`. Submitted against **both**
corpora as siblings: `failures.tfrecord` says how much of a *hard* scene's loss
is tail, and a WOMD shard — ordinary driving, where the trivial scene is the
common case and where the corpus-level regression is measured — says whether the
term misfires.

### The fix, wired behind a flag: `--barrier_shape` / `DFT_BARRIER_SHAPE`

| shape | value at margin | past the margin | deep ramp | smooth? |
|---|---|---|---|---|
| `softplus` (default) | `tau*ln2` | decays as `exp(-(c-m)/tau)`, never 0 | `m - c` | C^∞ |
| `hinge2` | **0** | **exactly 0, value and gradient** | `m - c - tau/2` | C^1 |
| `hinge` | 0 | exactly 0 | `m - c` | kink (reference only) |

`hinge2` is the quadratic-smoothed hinge: `z**2/(2*tau)` on the knee, linear
beyond it. It keeps everything the softplus was chosen for — `tau` still means
knee width, the ramp is still unbounded with gradient saturating at 1 so a
trajectory deep in violation is still pushed out, and it is still C^1 so there
is no `max` kink to differentiate through — and drops only the tail. The
quadratic branch clips its argument before squaring, so a broken record at
`c = -1e6` gives a finite value and a saturated gradient instead of a NaN
(asserted in `test_safety_loss.py::test_compact_barrier`).

**Default is unchanged.** Every margin in `safety_loss.py` and every run on
record was measured under `softplus`, and `hinge2` moves what a margin *means*
by removing ~2.7 tau of reach — the same reason `edge_geometry` and
`agent_geometry` are selectable rather than one silently replacing the other.
Run the probe, then re-run `calibrate_expert.py` under the new shape, before
flipping it.

### Taking it to a training arm: `sweep_barrier_shape.sh`

The A/B runs on the two arms with the best record under the current protocol,
with everything else at the preserve/stopping/proposal settings so the curves
compare across sweeps: `tf1` (teacher-forced K=1 — the proposal sweep's winner,
failure 0.239, marginal −0.0026) and `ctrl` (free-running — what every
geometry/stopping sweep used). 2 shapes × 2 proposals × 3 seeds = 12 tasks,
150 iters, eval+keep every 10, eval seed 12345.

Both proposals are in the default set because they put the safety term at very
different states. `tf1` scores a noised **human** trajectory, which is mostly
clear of both margins, so `hinge2` can zero far more of its loss there than on
the policy's own visited states. A result on one proposal alone could not
separate a barrier effect from a proposal interaction.

**`tail_share` is the deciding number, and it arrives at iter 0.** New
per-iteration metric (console line, `train_log.jsonl` and W&B): the fraction of
the safety loss charged to corners/agents already clear of their margin.
`safety_terms` now returns `edge_tail_loss` / `agent_tail_loss` / `tail_loss`
alongside the existing terms — reported, never differentiated. On a `hinge2`
arm it must read exactly `0.000`; that is the flag taking. On a `softplus` arm:

* ~0.02 → the two shapes are the same objective on this data, a null A/B is the
  *expected* result rather than a failed experiment, and the run is over after
  one iteration;
* 0.2+ → a fifth of the term is traffic density, and the arms should separate.

Note that `hinge2`'s `safety_last` is strictly lower than `softplus`'s by
construction — that is a different constant, not an improvement. Only the split
rates, read per seed at matched iterations against their own `base_*` rows,
settle it.

**The honest prior, recorded before the run.** Every geometry knob tried so far
(tau, margin, both windows) moved the marginal by ≤0.01, inside the noise, while
the stopping point moved it by 0.03 — and `barrier_shape` is a knob of that same
family. What makes it worth the GPU anyway is that it is not a
re-parameterisation of the same objective: it deletes a term that is present in
every scene whether or not anything is wrong. And either way the run returns a
measured `tail_share`, which no previous sweep reported.

## Reactive NPCs (2026-09-19/21) — 70% of the frozen-car collision rate is the rails

Every result above was measured against **log-replayed** other agents:
`controlled_object=SDC` on a `PlanningAgentEnvironment`, so each non-ego object
repeats its recorded pose and cannot see the ego, and `overlap` is symmetric box
contact with **no fault attribution**. The 08-30 probe localised what that costs:
lateral deviation is cheap (1 m off the log = one extra scene in 32, 0.031 →
0.062) but **delay** is not (5 steps 0.125, 20 steps 0.500, 40 steps — a car that
never moves — **0.719**). Rails punish being there at the wrong *time*, which is
exactly what a slow policy is. Reactive agents are the only thing that separates
"the policy stalls" from "a follower on rails drove through it."

### The result (job 58232, `idm_brake`, 32 failure scenes, all three gates PASS)

| delay (world steps late) | rails (`none`) | `idm_brake` | removed |
|---|---|---|---|
| 0 — **the control** | 0.031 | 0.031 | — |
| 2 | 0.062 | 0.031 | −0.031 |
| 5 | 0.125 | 0.062 | −0.062 |
| 10 | 0.250 | 0.156 | −0.094 |
| 20 | 0.500 | 0.250 | −0.250 |
| 40 — a car that never moves | **0.719** | **0.219** | **−0.500** |

Gate 1 PASS: the human demonstration is untouched (1 scene → 1 scene). Gate 2
PASS: the lateral column does not move (largest change −0.062 at 0.5 m, and it
*improves*), so the agents have not gone slack. Only then gate 3: **70% of the
frozen-car collision rate was the simulator**, and no reward, filter or fine-tune
could ever have removed it.

This converges with the failure-anatomy measurement (jobs 58135/58136,
`tools/failure_anatomy.py`), taken independently on
the policy's own rollouts: at first contact the ego is behind its own log and the
collider comes from **dead astern** (`c_r10`: 87% from behind, |bearing| p50
176°, paired speed deficit −1.81 m/s). A rear-ender by a car on rails is exactly
what `idm_brake` deletes.

**What this invalidates.** Every collision number above was measured under rails
and is inflated by an amount that grows with how far behind schedule the policy
is — so the arms that lag most are penalised most, which is a *ranking* hazard,
not just a level shift. `project_collision_is_downstream_of_stalling`'s "there
was never a collision problem to reward-shape" was derived from the 0.719, half
of which is now known to be exogenous. The failure splits themselves were built
under rails, so some `failure_heldout` membership is rails-induced.

### Post-mortem: waymax's stock `IDMRoutePolicy` does not test that (jobs 58075, 58076)

`IDMRoutePolicy`'s free-road term is `max_accel * (1 - (v/desired_vel)**delta)`
with defaults `max_accel=2.0`, `desired_vel=30.0` m/s, `delta=4`. An unobstructed
car at 10 m/s therefore accelerates at **1.98 m/s²** toward **108 km/h**, whatever
the human did — ~16 m/s and ~64 m of extra distance over an 8 s episode.

Mechanics gate (58075, 8 scenes, expert replay, so the ego is on the log and
every difference belongs to the NPCs): same-timestep deviation **p99 = 53.7 m**,
and 5% of poses **7.75 m off the agent's own logged polyline** — the latter
because an agent that races through its route runs out of waypoints and
`invalidate_on_end=False` extrapolates it straight past the end.

Paired probe (58076, 32 failure scenes, `none` → `idm`):

| case | none | idm | Δ |
|---|---|---|---|
| lateral 0 m (the control) | 0.031 | 0.062 | +0.031 |
| lateral 1 m | 0.062 | **0.188** | +0.125 |
| lateral 3 m | 0.250 | 0.344 | +0.094 |
| delay 2 / 3 / 5 steps | 0.062 / 0.062 / 0.125 | 0.062 / 0.062 / 0.125 | **0.000** |

It damages the demonstration, it *adds* lateral collisions, and it removes
**nothing** from the delay column — which is the only thing it was wanted for. A
speeding follower does not brake for a late ego, it arrives sooner.

Two caveats on that run, both since fixed: the delay sweep stopped at 5 steps and
never reached the region where the 0.719 lives (default is now `0,2,5,10,20,40`),
and the probe's own gate said PASS because its threshold was absolute (+0.06)
while the control's lift was a *doubling* of a 1-scene baseline. At n=32 one
scene is 0.031, so the gate now reports scene counts and reads three ordered
questions rather than one.

### `idm_brake`: keep the reactive half, delete the free-road half

`sim_agents.BrakingIDMPolicy` caps IDM's speed at the agent's own logged speed
for the step being produced:

    v = min(idm(v, lead_dist, lead_vel), v_log[t+1])

With nothing ahead, IDM's free-road term is above the logged speed almost
everywhere, the cap binds, and the agent reproduces its recording — so an
undisturbed scene is log replay and `d=0` is a control again. With the ego ahead,
the interaction term drops below the cap and the agent brakes. It also sets
`invalidate_on_end=True`, so an agent that finishes its route is invalidated
rather than extrapolated, which is what log replay does there.

**The cap is indexed by POSITION, not time**, and that was wrong in the first
implementation. Reading `v_log[t+1]` lets an agent that has fallen behind use a
*later, faster* part of the speed profile at an *earlier, slower* part of its
route — a car still queued at a light gets allowed the speed the human had after
it, sprints through and overshoots. Job 59763 caught it: **2.4% of poses ahead of
the recording, p99 = +15 steps**. Reading `v_log[j]` at the nearest waypoint to
where the agent actually *is* makes overshoot impossible: getting ahead would
require exceeding the human's own speed at that point.

The gate that catches this is the signed along-path schedule offset in
`tests/test_reactive_npc.py` (negative = braking, positive = racing). Hold any
future agent model to it. It excludes agents whose logged path is under 2 m —
a parked car's nearest-waypoint index is arbitrary and produced a spurious
p1 = −83 on the first reading.

The 58232 numbers above were produced by the time-indexed version, so they are
if anything *pessimistic* about `idm_brake`: 2.4% of NPCs were still racing.

`sim_agents="idm"` is kept so 58075/58076 stay reproducible, not because it is
usable.

### Two knobs, and they are not the same knob

| flag | what it changes |
|---|---|
| `--sim_agents none\|idm` (`DFT_SIM_AGENTS`) | how the other agents MOVE, i.e. what `overlap` means |
| `--neighbor_source log\|sim` (`DFT_NEIGHBOR_SOURCE`) | where the SAFETY LOSS reads their boxes |

Under `sim_agents=none` both sources hold the same motion and `neighbor_source`
is a no-op (kept as an equivalence control). Under `idm` they diverge, and `log`
charges the candidate for a hazard that braked its way out of existence.
`sim` reads `sim_trajectory` of the state the collection rollout ended in — the
NPCs' *executed* poses. That is one policy update stale, exactly like the
contexts themselves; making the NPCs react to the live candidate would mean
re-stepping the simulator inside the gradient, and the IDM's lead selection is an
argmin over an overlap indicator (zero gradient a.e.), so it would buy no signal.

`--neighbor_source sim` refuses to start with `--tb_weight>0`: `buffer_grads`
reloads its scenes at timestep 0 and cannot reconstruct the NPC motion its
transitions were recorded under, so the replay term would keep reading the log
while the on-policy term read the sim.

**Watch `n_neighbors`** (new, per iteration, console + `train_log.jsonl` + W&B):
valid ego-neighbour pairs per scored step. `sim_trajectory` is invalid past
`timestep`, `safety_terms` masks invalid pairs, and the agent barrier then reads
**0** — identical to an empty road. `neighbor_window` raises where the check is
concrete; `n_neighbors` is what catches it where it is traced.

### Run order

```bash
# 1. mechanics. The gate with teeth is "NPCs never run AHEAD of their
#    recording" -- that is the one stock IDM failed. CPU, minutes.
bash rl/diffusion_ft/slurm/submit_test_reactive_npc.sh
DFT_REACTIVE=idm bash rl/diffusion_ft/slurm/submit_test_reactive_npc.sh  # reproduce 58075

# 2. the paired probe, now reaching delay 40. CPU, ~45 min.
bash rl/diffusion_ft/slurm/submit_probe_logreplay_collision.sh
DFT_SIM_AGENTS=none,idm,idm_brake \
  bash rl/diffusion_ft/slurm/submit_probe_logreplay_collision.sh   # all three
```

**Read the gates before the curves, in order.** (1) Does the model damage the
human demonstration? A braking agent falls *behind* its logged schedule and can
end up where the recording never put it. (2) Did the lateral column go slack or
get worse? (3) Only then: how much of the delay column was rails. The probe
prints all three with scene counts, because at n=32 one scene is 0.031 and a
single-scene move is the resolution limit, not a result.

### The A/B (jobs 60192/60218 rails, 60194/60219 reactive; all 11 gates pass)

`submit_reactive_select.sh` runs `base` + `b_l24` as two sibling jobs, identical
but for the simulator.

| arm | m2 vs base, rails | m2 vs base, `idm_brake` |
|---|---|---|
| `b_l24_r20` | +0.0103 [+0.0083,+0.0127] | +0.0097 [+0.0078,+0.0122] |
| `b_l24_r10` | +0.0069 | +0.0066 |
| `ft` | −0.0119 | −0.0108 |
| `broken` (bank) | 0.0000 | 0.0002 |

**The selection result does not depend on the log-replay assumption.** The
prediction that the bank's margin would *grow* is refuted — it is unchanged
within overlapping CIs. `cons`'s do-no-harm weakens from structurally exact to
0.0002, which is the honest and expected consequence of `clear` becoming a
prediction rather than a certificate.

**But the base policy's measured failure rate was inflated by the rails**, and
that is the finding with consequences:

| split | base solved, rails | base solved, `idm_brake` |
|---|---|---|
| `failure_heldout` | 0.0462 | **0.1069** |
| `intermittent_heldout` | 0.4955 | **0.5523** |
| `success_heldout` | 0.9968 | 0.9950 |

Six points of `failure_heldout` and `intermittent_heldout` were the simulator.
The bank's own overlap on `failure_heldout` falls 0.1127 → 0.0318 (−72%), with
`offroad` and `reached` unmoved. Since `failure_heldout` was *built* under rails,
part of its membership is rails-induced — on top of the third that
`project_split_labels_protocol_mismatch` already flagged.

#### The two defects this A/B exposed, both fixed

**1. A 6% spurious collision floor on good driving.** The first run (59781/59812)
had `success_heldout` base fall 0.9968 → **0.9357**, entirely through overlap
(0.0000 → 0.0610), `reached`/`offroad` unchanged, bank switch rate 0.001 — the
simulator, not the intervention. Cause: stock IDM lets an NPC brake for *another
NPC* with generic constants (`min_spacing` 2 m, `safe_time_headway` 2 s) that
real logged traffic violates, so traffic halts where the human did not and the
ego rear-ends it. `BrakingIDMPolicy._get_accel` now masks the obstacle axis to
the **SDC only**: NPC-NPC motion replays as recorded (mutually consistent by
construction) and the ego is the only perturbation. The floor fell to 0.0018.
Note `failure_heldout` moved the *right* way throughout, which is why a
failure-set-only probe could not have caught this — **validate a simulator change
on the successes too.**

**2. The ego could not see the reactive NPCs.** `reached_rate` came back
bit-identical across simulators on all three splits (0.7327 / 0.9883 / 1.0000).
The preprocessor reads other agents off `log_trajectory`, and the observation
splice only replaced the *ego's* row — so NPCs reacted in the simulator and stood
still in the observation. Clean for an A/B that holds behaviour fixed and moves
only the metric; fatal for training against them, which is the
`project_sb3_route_reward_unobservable` failure mode. `_splice_sim_into_log_batched`
now splices every object. Under log replay that is an exact no-op (a replayed
object's simulated pose IS its logged pose — gate 1 measures 0.00e+00 over 19529
poses), and gate 1b asserts it both ways.

The A/B numbers above predate defect 2's fix, so they still describe a fixed-ego
comparison. That is the right design for *this* question; a run that lets the
policy respond to the traffic is a different and larger experiment.

Also reverted: the position-indexed speed cap. It deadlocks an agent that halts
in a stop cluster (cap reads ~0, it never leaves), and time-indexing cannot race
anyway — `∫v ≤ ∫v_log` bounds arclength, so an agent may catch up but never pass.
The racing gate now measures **arclength in metres**; waypoint *index* is
ill-conditioned in exactly those stop clusters, which produced the spurious
±30–80 step readings in jobs 59763/59778.

### If the gates pass

Nothing is comparable across the two simulators — `overlap` under `idm` is not
the same number as `overlap` under `none` — so every arm needs its own base arm
in the same job. `--sim_agents` is on `build_argparser`, so `proposal_select`
inherits it (`DFT_SIM_AGENTS` in `submit_select_eval.sh`), and re-scoring the
shipping `b_l24 @ ramp 20` arm under `idm_brake` alongside `base` is the direct
test of whether `cons`'s do-no-harm survives reactive agents. Note `make_scorer` keeps
reading the log deliberately: the selector commits at *plan* time, so the NPCs'
future is not available to it in any simulator. What changes is the status of
`clear` — exact under rails, a genuine prediction under reactive agents.

## FlowMol3: self-conditioning + late-stage distortion (inference-time OOD)

FlowMol3 ([arXiv:2508.12629](https://arxiv.org/abs/2508.12629), Dunn & Koes;
*Digital Discovery* `d5dd00363f`) names a pathology this package already
describes in `unrolled_prediction.py` under "Diffusion-state mismatch":
**transport models produce suboptimal geometries at inference despite never
observing suboptimal geometries at training time.** The reverse chain's own
`x_t` is not a sample from the forward path the denoiser was trained on, the gap
compounds step by step, and nothing in the training objective ever asked the
network what to do about it. Two of FlowMol3's three techniques port directly
(the third, fake atoms, does not — the horizon here is fixed at 25).

Both are **off by default**, and off they leave the parameter tree, the traced
programs and every existing result untouched (`tests/test_self_cond.py` gate 1
pins that against hand-written references, and the same job re-runs Gate 2).

**Self-conditioning** — the denoiser additionally sees its own clean prediction
from the previous reverse step, `g(x_t, x̂0(x_{t+1}))`, so it can notice that it
is revising and correct rather than compound. Implemented as a channel concat at
the U-Net stem, so it is the one change that alters the parameter tree: the
stem's fan-in doubles from `in_ch` to `2*in_ch`. Training follows FlowMol3's
50/50 split — half the examples are scored on the zero-self-cond prediction, half
get an input from a **stop-gradient** first pass — and inference always passes
the previous prediction.

**Late-stage geometry distortion** — at training time only,
`x_t += 1[t_min ≤ t ≤ t_max] · (M ⊙ ε)` with `M_i ~ Bernoulli(p)` over
**waypoints** (the waypoint is this model's "atom") and `ε ~ N(0, σ²)` on the
**position** channels only. FlowMol3's `t_distort = 0.5` is flow time
(`t=0` noise → `t=1` data), i.e. the low-noise half, which is `t ≤ 49` on this
100-step DDPM index. The regression **target is corrected** for the kick so the
network is asked to map the distorted state back to the *same* clean
trajectory; leaving the target alone would train it to *reproduce* the kick
(`x0_pred = x0 + √ᾱ·d`), which is exactly backwards — gate 5 catches that.

`--distort_mode relative` (the default) scales σ by `√(1-ᾱ_t)`, the forward noise
already present at that timestep. `absolute` matches FlowMol3's fixed σ, in
normalized units (`1.0 == ego_range == 100 m`). Relative is the default because
on a **v-parameterized** DDPM the corrected target for an absolute kick blows up
like `1/√(1-ᾱ_t)` — a factor of ~40 at `t=0`.

### What to run, in order

```bash
# 1. gates (CPU, minutes). Nothing else is worth submitting until these pass.
bash rl/diffusion_ft/slurm/submit_test_self_cond.sh

# 2. measure the drift, so sigma is not a transplant from a molecule (GPU, ~20 min)
bash rl/diffusion_ft/slurm/submit_probe_drift.sh
#    -> prints path_std (1.0 == on the forward path), the drift penalty `gap_m`,
#       and a suggested --distort_sigma. If gap_m is ~0 in the late window,
#       STOP: there is no inference-time OOD here for distortion to fix.

# 3. the 2x2 ablation, 20k steps each from the base checkpoint (GPU, %2 throttle)
FM3_DISTORT_SIGMA=<from step 2> bash rl/diffusion_ft/slurm/submit_train_flowmol3.sh

# 4. score every arm through the SAME harness/splits as every other arm here
FM3_SAVE_DIR=<the save dir step 3 printed> \
DFT_SPLITS_JSON=<the VERIFIED splits.json> \
DFT_SHARD_GLOB='<the glob splits.json was built against>' \
FM3_TRAIN_JOBID=<step 3 job id> \
bash rl/diffusion_ft/slurm/submit_eval_flowmol3.sh
```

Arms: `base` (same fine-tune, neither feature — **not** the base checkpoint, so
"trained longer" is controlled for), `sc`, `dist`, `sc_dist`. The eval also
scores the untrained base checkpoint as `base_ckpt`.

### Why a fine-tune and not a re-pretrain

Self-conditioning changes the parameter tree, which normally forces training from
scratch. `--init_from` instead **zero-expands** the stem
(`rl/diffusion_ft/self_cond_init.py`): the new input block starts at zero, so it
contributes nothing to the convolution and the expanded model computes the base's
function *exactly* at step 0 for any self-cond input, including the zeros the
first reverse step supplies. Gate 2 asserts that rather than assuming it. Every
other leaf is copied unchanged. `--init_from` is **not** `--resume_path`: it
loads parameters only and starts fresh optimizer state, schedule and counters.

To produce a standalone self-conditioned checkpoint (e.g. to point an existing
eval at):

```bash
python -m rl.diffusion_ft.self_cond_init --src <base>/latest --dst <new>/latest
```

The new `metadata.json` carries `self_cond: true`, which is what
`train_diffusion/utils/infer.py::_build_model_from_metadata` reads — the
architecture flag must come from the checkpoint, never from a caller's default,
or the restore silently shape-mismatches.

### Known limits of this port

* DPPO/safety machinery is threaded through (`sample_with_trace` records `x_sc`,
  `transition_log_prob` takes it, and the reusable prefix becomes the pair
  `(x_{k-1}, x̂0_k)` — still one opaque pytree argument, so no caller changed).
  It is *gated* for equivalence but has not been *run* under self-conditioning.
* The self-cond training branch costs a second denoiser evaluation on every step
  (the first pass runs for the whole batch and is then masked per example).
  FlowMol3's per-step coin flip would average 1.5; the per-example mask was
  chosen for lower variance and a static program shape under jit/pmap.

## Anatomy of the residual (2026-09-19/23) — no-goal is a pace bug, and it has a root cause

> **SUPERSEDED 2026-09-23 — read [The world-dt bug](#the-world-dt-bug-2026-09-23--the-never-reached-block-is-a-time-base)
> first.** The never-reached block below is the set of scenes whose world step
> was read off a short first timestamp pair (43 of its 44 scenes), and the
> 13–23% "velocity inflation" is the pooled signature of the same scenes. The
> "two heads, two constants" mechanism is not established. The off-road and
> collision anatomy stands.

**Result: the three things the safety fine-tune still fails on are one bug.** The
model's velocity head and its position head disagree by ~15%; the positions
under-travel; the goal is a fixed 3 m ball at the end of the human's path, so a
constant fractional shortfall is free on a short trip and fatal on a long one;
and the resulting lag is what the log-replayed traffic rear-ends. The off-road
term is the only part the barrier was ever the right tool for.

`tools/failure_anatomy.py` + `slurm/{failure_anatomy.sbatch,submit_failure_anatomy.sh}`.
CPU only: it reads the executed world paths a select run dumped under
`--dump_traj` (`<arm>/r<rep>/traj` in `feat/*.npz`) and recomputes every geometric
quantity with `safety_loss` itself, so a disagreement here is a disagreement with
the objective rather than with an approximation of it. No GPU, no checkpoint, no
re-rolling.

```bash
bash rl/diffusion_ft/slurm/submit_failure_anatomy.sh          # array of 6 + afterok merge
DFT_ANAT_SMOKE=1 bash rl/diffusion_ft/slurm/submit_failure_anatomy.sh
DFT_ANAT_SPLIT=intermittent_heldout bash rl/diffusion_ft/slurm/submit_failure_anatomy.sh
```

**Two alignment facts, both asserted at runtime rather than trusted.** Each replan
dumps `world[:, :n_exec]` — the anchor plus the first `n_exec-1` executed poses —
and the next replan's anchor IS the pose the previous one ended on, so
concatenating the replans gives a contiguous 0.1 s path; and index `i` of it is
absolute log step `T0 + i` with `T0 = init_steps - 1 = 10` (waymax's `reset`
leaves `state.timestep` one below `init_steps`). Measured residual against the
ego's logged pose at `T0`: **0.0000 m**, a hard exit above `--align_tol`. The
recomputed `solved`/`reached`/`offroad`/`overlap` reproduce the run's own flags at
0.99–1.00. Read both before reading anything below.

### What the residual is made of (jobs 58135/58136, `failure_heldout`, 692 episodes)

ft = `stopping/20260911_160658_962947/base_s0/iter_000110`, verify seed 4242.

| | base | ft | c_r10 (bank) |
|---|---|---|---|
| solved | 0.043 | 0.312 | 0.549 |
| never reached | 188 | **187** | 188 |
| off-road (any) | 354 | 187 | 100 |
| overlap (any) | 204 | 178 | 78 |

The fine-tune halves off-road, moves overlap 13%, and does **nothing** to
reaching. Of its 476 unsolved episodes: 39% never reached, 39% off-road, 37%
overlap (the buckets overlap).

### No-goal is trip length x pace, not geometric hardness

44 of 173 scenes never reach in ANY of 4 repeats, and it is the *same 44 scene
ids* under base, `b_l24` and `bf_w10` (46 under ft alone); ft rescues one. That
invariance is the tell — it is a property of the scenes. Why:

* the human's path length **equals** its goal distance (surplus **+0.03 m**), so
  the 3 m ball sits at the END of the demonstration with no slack;
* the fraction of the human's path the ego must cover is therefore
  `(goal_dist - 3)/s_log`, which rises with trip length: **0.921** on the reached
  scenes (44 m trips) against **0.979** on the never-reached ones (94 m);
* a roughly constant fractional shortfall is then free when short and fatal when
  long.

| trip length | <40 m | 40–60 | 60–80 | 80–120 | >120 m |
|---|---|---|---|---|---|
| never-reach rate | 0.03 | 0.19 | 0.40 | 0.40 | **0.58** |

**AUC of trip length alone = 0.793.** `stall_frac` is **0.00** — it never stops —
cross-track is 0.52 m and detours are 2.1%, so this is not a stall and not a
detour. It drives the right line short.

### The root cause: two heads, two constants, nothing coupling them

`data/postprocess.py::ego_normalized_to_world` denormalizes the model's two heads
with two independent constants — `pos * ego_range (100)` and
`vel * max_velocity (25)`. **Not config drift**: the checkpoint's `metadata.json`
carries exactly those values. Nothing forces `d(pos)/dt == vel`, and measured over
753 episodes the two disagree:

| arm | commanded / realised (ego) | same ratio for the LOG |
|---|---|---|
| base | 1.132 | 1.000 |
| ft | 1.149 | 1.000 |
| c_r10 | **1.229** | 1.000 |

The log column is the control and reads 1.000, so this is the policy's output,
not a dt or frame error in the comparison. **`c_r10`'s 1.229 is NOT explained**:
an earlier note here said the bank edits "move positions and leave velocities
alone", but `lateral_shift` and `time_warp` both end in `branching.fd_velocity`
(since the module was written), so a fired edit's velocities ARE consistent with
its positions and should pull the ratio toward 1, not away. Suspect episode
composition (a per-episode ratio of means over slow tails) before any mechanism.

**Which head is right: the velocity one.** Against the human, commanded speed is
**1.05–1.07x** and realised displacement **0.93x**. The model knows roughly the
right speed and emits positions that travel ~7–10% short — which is the pace
deficit above, to within its own error bar.

**And it is a closed loop.** `_splice_sim_into_log_batched` writes sim
`vel_x`/`vel_y` into `log_trajectory` (`env.py`, the `blend` call) and
`data/preprocess.py` builds the policy's own `ego_state` from exactly that field,
so at every replan the model is conditioned on a speed it is not doing, having
been pretrained on data where the two agree.

### The barrier is not the problem it was suspected of being

Ground-truth lateral sweep at 69 excursion onsets — the ego box is shifted along
its own normal and the clearance `is_offroad` thresholds are recomputed from
scratch at every offset, so this is the metric's own verdict on each shifted pose:

* an on-road offset exists within ±4 m at **100%** of onsets; both sides blocked **0%**;
* corner coherence p50 **0.99**; corners disagree at only 2.9%;
* the barrier's descent points at the nearer exit **89.9%** of the time under the
  default `edge_geometry="point"`, **97.1%** under `"segment"`.

So **"the car is trapped between two road edges" does not happen**, and the
wrong-direction case is a 10% minority rather than the mechanism. The number that
replaces it: the exit is **0.10 m away at p50** (0.22 m at p90) and the excursion
still lasts **21 steps (2.1 s)** at 0.64 m median depth. Two seconds off-road, a
hand's width from legal, with a correctly-pointing gradient. Consistent with the
bank's −0.69 m shortfall, which is measured over a 20-step window: the
instantaneous exit is trivial, staying clear for 2 s is not.

`edge_geometry="segment"` also removes a **+0.10 m** overstatement of depth — with
"point" the barrier charges the distance to a roadgraph SAMPLE, not to the
boundary.

### Collisions

At first contact the ego is behind its own log (ft 2.58 m / 6 steps, 53% lagging
>2 m, 74% still on the log line, collider from behind 54%, paired speed deficit
−0.38 m/s), and the signature sharpens as the other modes are removed: `c_r10` is
14.07 m / 11 steps behind, 87% from behind, |bearing| p50 **176°** — dead astern.
This is the same finding the reactive-NPC section reaches from the other side;
see [Reactive NPCs](#reactive-npcs-2026-09-1921--70-of-the-frozen-car-collision-rate-is-the-rails),
and re-score under `idm_brake` before spending anything on the agent barrier.

### Traps, all three silent

* **Speed read from the velocity field inverted the conclusion.** The first read
  had the ego at **1.18x** the human it was falling 7.7 m behind, which reads as
  "not a pace problem". Off positions it is 0.91x, matching the independently
  computed path-length ratio of 0.90. The tool now computes pace from POSITIONS
  and logs `speed_cmd_ratio` (ego) beside `log_speed_cmd_ratio`, which must read
  ~1.000 — the defect is now visible rather than remembered.
* **`jax.grad` on a closure built per call leaks its trace cache.** The escape
  test defines a differentiated function per excursion; `jax.grad` keys its cache
  on the function object, so a few hundred sweeps accumulated a few hundred
  never-evicted traces and the job was OOM-killed with nothing large anywhere in
  it. `_edge_grad_fn` builds it once per configuration.
* **The submitter must run on rhea, and rhea's system python3 has no numpy.** It
  inspects the dumped `.npz` with `zipfile` from the stdlib instead — an `.npz`
  IS a zip and each array is one `<key>.npy` member.

### Where this leaves it

* **The no-goal residual is not reachable by any restriction-shaped method** —
  not the barrier, not the bank, not a selector. It needs pace.
* **The first thing to try is the cheapest**: a `bank@pace:always` arm with a
  single `tw1.1` candidate (`bank`+`always` is permitted). It is not
  restriction-shaped, so `broken` will NOT be 0.0000 and the cost on
  `success_heldout` is the whole measurement.
* **Then couple the heads**: derive ego velocity in the splice from the blended
  positions (the observation stops lying — sign on pace genuinely unpredictable,
  since making the state self-consistent should return the model to its training
  manifold but that is the experiment), and/or rescale predicted displacement to
  the predicted speed in `ego_normalized_to_world`. Score on `reached_rate` in the
  LONG-TRIP decile; a corpus average washes it out.
* `edge_geometry="segment"` is a real but small A/B and does not touch the
  0.10 m / 2.1 s puzzle.
* The 09-19 artifact on disk predates the velocity fix: its two "speed ratio"
  lines read ~1.18 where they should read ~0.91. Everything else in it stands.

### Fix 1 x fix 2a: the pace 2x2 (built 2026-09-23; superseded before it ran)

> The smoke (jobs 62441–62446) ran the chain end to end; its gate failed G1, which is how
> the world-dt bug below was found. Both fixes divide by the same wrong step, so the full
> 2x2 was not run. The flags stay (defaults unchanged).

```bash
DFT_SELECT_SMOKE=1 bash rl/diffusion_ft/slurm/submit_pace.sh   # 4 scenes/split, 1 repeat
bash rl/diffusion_ft/slurm/submit_pace.sh
bash rl/diffusion_ft/slurm/submit_test_ego_vel.sh              # the gate on its own
```

**2a is `--ego_vel_source {commanded,realised}`** (env `DFT_EGO_VEL_SOURCE`).
`realised` replaces the ego's `vel_x/vel_y` in the observation splice with the
backward difference of the executed positions -- only the ego, only on steps the
policy executed (`init_steps..timestep`), never the replayed history, so the first
replan is bit-identical. Backward because at the anchor the blend holds the
HUMAN's next pose; a central difference would leak the demonstration into the
observation. `commanded` is the default and every earlier result.

**Why a 2x2.** The pace arm (`bank@p10:always`, one `tw1.1` candidate) is not a
pure speed-up: `time_warp` rewrites velocities from finite differences, so under
`commanded` it also hands the policy a self-consistent speed. The same arms under
`realised` separate the two effects (`realised:base - commanded:base` is 2a alone;
`realised:pace10 - realised:base` is the speed-up with the observation held
fixed). Pace arms run at **ramp 1**: at the bank's default ramp 10 the factor
eases 1.0 -> 1.1 inside each 10-step window and resets at the next anchor, so
"tw1.1" would execute 5.5%. `pace05` (tw1.05) is a dose point.

**Gates.** `tests/test_ego_vel_splice.py` (CPU, `test_ego_vel.sbatch`): the default
is the old splice bit for bit; the estimator on the HUMAN's positions reproduces
the human's logged speed within 1% (the log column still reads 1.000), also end
to end at the policy's anchors under expert replay; positions, yaw, every non-ego
row and the replayed history are untouched; under a synthetic under-travelling
override `commanded` shows the lie (~1/0.9) and `realised` reads 1.000 at every
anchor on both the jitted and eager paths. It always exits 0 and writes
`gate/verdict.txt`; the `realised` sibling depends on it and its tasks exit 0 with
the reason unless it says PASS.

**Live checks in every row** (`proposal_select.episode_pace`): `trip_m` (the
human's path to the goal), `goal_d_min` (closest approach, the continuous version
of `reached`), `pace_moving` (ego/human displacement while the human moves, from
positions), and `obs_v_sum`/`real_v_sum` -- the speed the policy was SHOWN vs the
speed it drove. `realised` cells must read 1.000; `commanded:base` shows the
defect (~1.13). `tools/pace_analysis.py` reads both siblings, refuses to pair
unless seed/chunk/shards match, checks `trip_m` agrees across every row of a
scene, and reports `reached`/`solved` per trip bin (40/60/80/120 m and each
split's top decile) with paired scene-bootstrap CIs, plus m2/m3 via
`select_analysis`. Headline: failure_heldout top-decile `reached`; cost:
success_heldout `solved`.

## The world-dt bug (2026-09-23) — the never-reached block is a time base

**Result.** `data/preprocess.py::compute_world_dt_seconds` read each scene's world
step off its FIRST valid timestamp pair. The preprocessor lays the policy's plan
onto a `k * world_dt` grid, and the env advances one log step per grid point.
WOMD tf_example timestamps here are real and slightly non-uniform, and in some
scenes the first interval is short — p50 **0.0748 s** (p5 0.059), where the median
step is 0.100 s and 80 steps span 8.0 s. There the plan executes at ~0.75x, while
its velocity field still reports full speed.

| `failure_heldout` (09-19 anatomy run + per-scene dt) | short first pair | the rest |
|---|---|---|
| scenes | 44 of 173 | 129 |
| never reach the goal (base) | **43 of 44** | 4 of 129 |
| reached rate, base / ft / c_r10 | 0.023 / 0.023 / 0.023 | 0.969 / 0.971 / 0.969 |
| commanded speed vs the human | 1.19 | 0.99 |

`success_heldout`: 0 of 120 sampled scenes have a short first pair, and
`intermittent_heldout` 0 of 96. The base fails every such scene, so the scan put
all of them in the failure split. A quarter of `failure_heldout` is selected by
this bug, and every restriction-shaped method measured on it had a ~25% ceiling
it could not see.

**The log is fine.** Dividing each step by its OWN timestamp gap, the human's
backward difference matches its logged speed at **1.0014** (central 1.0000;
p90 error 0.1 m/s), and every object's at 0.9991, on the 64 scenes where dividing
by the first-pair dt read 1.1085.

**The fix is `--world_dt_method median`** (env `DFT_WORLD_DT_METHOD`, default
`first` = every earlier result). Only the JAX preprocessor has it;
`data/preprocess_cached.py` and `vla/preprocess.py` keep their own `first`
copies. Training barely depends on it, because targets use the stamped time
base, and here it only bounds the anchor draw.

```bash
DFT_SELECT_SMOKE=1 bash rl/diffusion_ft/slurm/submit_world_dt.sh
bash rl/diffusion_ft/slurm/submit_world_dt.sh
```

The chain: `test_world_dt.sbatch` (CPU gate) → `first/` and `median/` select
siblings, `base` only, where `median/` reads the gate's verdict →
`pace_analysis.sbatch`, whose `dt_short`/`dt_normal` columns are the readout.
The gate (`tests/test_world_dt.py`):

* `first` reproduces the old function bit for bit;
* `median` matches the executed window's clock to 1% at p95;
* the flag reaches `aux["world_dt_seconds"]` on the jitted and eager paths;
* the human's own trajectory, executed as the plan, stays on the log under
  `median` (p95 < 0.25 m) and visibly lags on the short-dt scenes under `first`.

**Result (jobs 62629–62634, 2026-09-24; gate PASS, 1450 scenes x 4, paired by
common random numbers).** The base policy alone, with no weights changed:

| `median - first` (paired, 95% scene bootstrap) | n | reached | solved | broken |
|---|---|---|---|---|
| failure_heldout, short-dt | 44 | **0.03 → 1.00** (+0.972) | **0.01 → 0.95** (+0.949 [+0.886, +0.994]) | 0.000 |
| failure_heldout, normal-dt | 129 | +0.000 | +0.000 | 0.000 |
| intermittent, short-dt | 3 | +0.667 | +0.417 | 0.000 |
| intermittent, normal-dt | 274 | −0.001 | −0.004 [−0.007, −0.001] | 0.004 |
| success_heldout | 1000 | +0.000 | +0.000 | 0.000 |

The corpus-level **m3 is +0.0035 [+0.0026, +0.0045]** (rec 0.0036, brk 0.0001) —
a third of the shipping bank's +0.0103, from a clock fix. In the short-dt
scenes, pace goes 0.90 → 1.00 and the world dt used goes 0.0748 → 0.1000.
Every failure-split flip is a short-dt scene, and every one of them is 0 → 1.

The only breaks are 4 single episodes in 4 normal-dt intermittent scenes, whose
dt moved by at most 1.1% (the median move on normal scenes is 0.03%). Three of
the four flipped via a new off-road. Intermittent scenes are coin flips by
construction, so any perturbation diverges them, and 4-vs-0 has p ≈ 0.13
two-sided. Read it as noise.

**So the failure split was a quarter bug.** The 44 short-dt scenes are not hard:
once they can arrive, the base solves 0.95 of them. The genuine residual is the
129 normal-dt scenes, which the base reaches (0.98) and then fails on off-road
and overlap (solved 0.06). The "long trips never reach" result was this bug too:
the top-decile trips go 0.40 → 0.90 reached.

**Defaults since 2026-09-24:** `median` everywhere in this pipeline — the env
constructor, `train_safety --world_dt_method`, and new `submit_select_eval.sh`
env files, which write the key explicitly. `select_eval.sbatch`'s fallback stays
`first`, so an older env file without the key, requeued, finishes on the method
it started with. To resume a `train_safety` run started before then, pass
`--world_dt_method first`. `data/preprocess.py` keeps `first` as its own default,
because it is shared with pretraining; the env always passes the method.

**Retrained on the fixed clock (2026-09-25, `submit_retrain_worlddt.sh`, run
`retrain_worlddt/20260924_235925_2645687`).** The 09-11 recipe unchanged except for
the clock, 3 seeds, scored at iteration 110 on all three splits (1450 scenes x 4).
The two evals share their base draws (5800 of 5800 episodes, 1 outcome differs).

| solved (fixed clock) | failure | intermittent | success | m3 vs base (95% CI) |
|---|---|---|---|---|
| base | 0.288 | 0.496 | 0.997 | — |
| ship bank l24, ramp 20 | **0.773** | **0.870** | 0.999 | **+0.0206 [+0.0184, +0.0233]**, brk 0.0003 |
| old ft (trained on the old clock), 3-seed | 0.554 | 0.619 | 0.982 | −0.0062 [−0.0116, −0.0015] |
| new ft (retrained), 3-seed | 0.541 | 0.638 | 0.986 | −0.0023 [−0.0068, +0.0018] |

Paired, new minus old fine-tune: failure −0.013 [−0.024, −0.001], intermittent
+0.019 [+0.007, +0.031], success +0.0037 [+0.0013, +0.0062]. Retraining on the right
clock makes the fine-tune gentler on successes, taking the corpus effect from
significantly negative to indistinguishable from zero. It is still not a corpus
win. The fine-tune's gain on failures (+0.25) did not depend on the clock; the
training curve still peaks at iteration 110 (marginal −0.0056). The repair bank
remains the only corpus-positive intervention.

Why the base "solves" 0.29 of a failure split: the split was built on the old
clock, so 44 of its 173 scenes (25%) are clock-bug failures. On the fixed clock
the base solves 0.955 of those and **0.060** of the 129 genuine failures; the
fine-tune goes 0.953 on the bug scenes and **0.401** on the genuine ones — a
+0.34 real gain, diluted to +0.25 in the blended number. The m2/m3 estimates are
still UNBIASED: they are stratified estimates (weight = the stratum's corpus share
as scanned, rate = measured now), and the bug scenes contribute identically to
base and fine-tune. (An earlier note here claimed the weights over-count the
failure stratum by ~0.001–0.002; that was wrong.) The re-scan only makes the
labels mean what they say.

**Next:** re-scan the splits under `median` (a quarter of `failure_heldout` is
bug-selected, and `failure_train` presumably as well), then re-baseline the
shipping arms (`ft`, `b_l24_r20`) against the rebuilt splits. Every earlier
failure-set number carries a hidden ~25% ceiling.

## `edge_geometry="segment"` in training (2026-09-25, built, not yet run)

`--edge_geometry {point,segment}` on `train_safety` (env `DFT_EDGE_GEOMETRY`,
default `point` = every earlier run, bit for bit). `segment` measures a corner's
road-edge clearance to the polyline through the nearest roadgraph sample rather
than to the sample: same sign, so the same `offroad` boundary, but no ~0.10 m
depth overstatement, and descent toward the nearer exit 89.9% → 97.1% (see
[The barrier is not the problem](#the-barrier-is-not-the-problem-it-was-suspected-of-being)).
Expected to be a minority effect. It does not touch the 2.1 s excursions.

**The margin is re-measured, not carried over.** 0.3 m is the human's p01
clearance *under `point`* (0.314, job 40089). `segment` clearances are never
larger, so 0.3 there is a stricter barrier, and the A/B would confound geometry
with strictness. The arm keeps the **rule** instead (margin = expert p01, floored
to the cm): `calibrate_edge_geometry.sbatch` runs `calibrate_expert --edge_geometry`
under both geometries on the same 64 scenes, and `tools/decide_edge_margin.py`
writes `cal/edge_margin.env` only if these gates pass:

* the point control reproduces 0.314;
* both runs pool the same expert-clean steps;
* segment ≤ point on every step;
* segment p01 ≥ 0.10 m.

`train_safety.sbatch` sources that file via `DFT_EDGE_MARGIN_FILE`. If the file
is set but missing, it prints `TRAINING NOT RUN` and exits 0. `decision.json`
also reports the expert's mean barrier-gradient strength at `point@0.3`,
`segment@<new>` and `segment@0.3`. The first two should match, which is what
"same rule" means. The third is the literal flag flip, for scale.

```bash
bash rl/diffusion_ft/slurm/submit_edge_segment.sh                   # calibrated margin
DFT_SEG_MARGIN=0.3 bash rl/diffusion_ft/slurm/submit_edge_segment.sh  # literal flip
```

The chain:

1. `cal` (CPU).
2. `train` (afterok:cal): the 09-11 recipe on the median clock, `base_s0..s2`.
   Before anything is submitted, the env files are diffed against
   `retrain_worlddt/20260924_235925_2645687/train/env`: only the edge keys, the
   out dir and the wandb group may differ, otherwise FATAL.
3. `eval` (afterok:train): base + ft @ iter 110, seed 4242. Its env files are
   byte-identical to that run's `eval_new` apart from paths, so the base draws
   are shared.
4. `compare` (CPU): `tools/compare_select.py`, segment minus point. It takes the
   3-seed means paired per (scene, repeat), with the stratified scene bootstrap.
   It refuses if the two runs' base episodes disagree on more than 0.2%. A
   self-comparison of `eval_new` reproduces `select_analysis`'s ft means exactly
   (0.541 / 0.986 / 0.638, delta 0).

Read the curves first: the same eval seed and splits make `train_log.jsonl`
directly comparable with the reference's. Then read `eval/compare.json`. Iter 110
is pre-registered. If segment peaks elsewhere, record it; do not move the eval.

New plumbing: `sweep_safety_stopping.sh` forwards `DFT_EDGE_GEOMETRY` and
`DFT_EDGE_MARGIN_FILE`, and takes `DFT_DEPENDENCY` and `DFT_DRY_RUN=1`, which
writes the env files and submits nothing.

## Behaviour-cloning the repair bank (2026-09-28/30) — negative; full write-up in `docs/`

Distilling `b_l24_r20` (m3 +0.0209) into the base policy does not carry its gain
into the weights. Plain BC on the bank's executed plans gives m3 +0.0012: the
edits are under-fit at about 7% of a batch. Putting half of every batch on the
edits fits them (edit-row loss 0.40x base) but gives m3 -0.020: failures
0.29 -> 0.49, successes 0.997 -> 0.969, mostly off-road. A probe rules out the
frozen prefix as the limit (`d_free == d_tf`; noise at late step t=9 is about
17 m). The student drifts on rows the bank left alone, because it cannot see
which of its own draws the bank would have edited.

The setup, every number, the probe, what was measured vs hypothesized, and ranked
ideas (BC policy as a proposal under the selector, preference distillation on
(rejected draw, edit) pairs, DAgger, kept-row tether, a better teacher) are in
[`docs/bank_bc_distillation.md`](docs/bank_bc_distillation.md). Code is
`proposal_select --dump_bc`, `train_safety --repair_{steps,rows,split_ratio,switch_share}`,
`tools/probe_bc_transfer.py` and `slurm/submit_bank_bc.sh`.

## Diffusion-DPO on the bank's pairs (2026-10-01, built)

Idea 2 of [`docs/bank_bc_distillation.md`](docs/bank_bc_distillation.md): instead of
BC toward the edit, train on the pair (condition, **rejected** base draw, accepted
edit) with Diffusion-DPO, the frozen base as reference. A pair says "this edit over
THIS draw of yours", which BC on targets alone never shows the student.

**Objective** (`train_safety --pref_weight`, `SafetyFineTuner._init_pref` /
`pref_grads`, loss in `DiffusionRLActor.pref_loss`), on every Adam step:

    L = mean softplus(-z) + pref_anchor_weight * base_anchor_KL(success_train kept rows)
    z = beta * [(l_model_bad - l_ref_bad) - (l_model_good - l_ref_good)]

`l` is the eps-prediction MSE (converted exactly from the checkpoint's v output),
masked to covered steps, with ONE shared `(t, eps)` per pair for both trajectories
and both models. At step 0 the model is the reference, so `z = 0` and the loss is
`log 2`; the trainer checks this and exits if it does not hold.

**Choices that differ from a textbook Diffusion-DPO, and why:**

* `t ~ U[0, k_trainable)`, not the whole schedule. Only the last `k_trainable = 10`
  reverse steps run the trainable copy at deploy time; a preference taught at
  t >= 10 is taught to a step the frozen base drives. `--pref_t_max` narrows it
  (the low-noise arm uses 3). This is also the range BC trained on.
* The anchor is the existing frozen-base KL (`base_anchor_loss`), but charged at
  `t ~ U[0, k_trainable)` (new `t_max` argument; `None` keeps every earlier run
  bit-identical), on `success_train` rows the bank KEPT, with `x_t` formed from the
  base's own draw. Failure-scene pairs are never anchored.
* Pairs are every switched row of a solved episode, any split (572 failure + 39
  success in the bank-BC collection: BC's 611 edit rows exactly).
* `beta` is small here relative to the paper (2500-5000 there). With edits of ~0.3 m
  against 2.5-17 m of noise, `|z|` likely stays << 1, the sigmoid never saturates,
  and `beta` then acts as a gradient scale against the anchor. **Measured in the
  smoke run (job 70446):** at beta 1, z ~ 1e-4. After one step the DPO gradient
  norm was ~0.1 against an anchor gradient of 70-390 (times the weight of 100).
  Under the 1.0 global-norm clip the step is then all anchor, so the plan's
  beta 1 / 0.1 arms would re-measure the base. The run therefore uses beta 1e4
  (DPO:anchor near the safety recipe's ~1:20) and 1e3 (10x more anchored). Both
  are near the paper's 2500-5000. Read `pref/logit_z` and the two gradient norms
  (`pref/grad_norm`, `pref/anchor_grad_norm`) before interpreting beta.
* Edit weighting (arm B) up-weights model steps where `|good - bad| > delta`. The
  bank's edits are rigid lateral shifts after a 2 s ramp, so most of the plan past
  ~1 s is "edited": this arm mostly de-weights the ramp, not a local patch.

**Data.** `proposal_select --dump_bc` now also writes `rejected`: on switched rows
`world_to_norm(world_p)` on exactly the steps where `target = world_to_norm(edit)`,
`x_p` elsewhere. Both sides of a pair take the same normalization path, so
`target - rejected` is the edit and nothing else. `_init_pref` refuses data where
they differ anywhere the arm did not edit.

**Logged per iteration** (means over the iteration's steps): `pref/{loss, logit_z,
z_abs, accuracy, good_model_loss, good_ref_loss, bad_model_loss, bad_ref_loss,
ref_gap, init_gap, grad_norm, anchor_loss, anchor_grad_norm}`. Watch `good m-r` and
`bad m-r` (model minus ref) separately. DPO can raise `z` by getting worse at BOTH
trajectories, worse at the bad one faster. The CPU test reproduces this without an
anchor: +1.04 vs +1.40 after 60 steps.

**Run.** `slurm/submit_pref_dpo.sh`: recollect (bank-BC settings, seed 2718) ->
pair check -> train 4 arms x 3 seeds (`b1e4`, `b1e3`, `lowt` and `edit` at beta 1e4; 3020 steps,
16 pairs + 16 anchor rows/step, anchor weight 100) -> per-arm `select_eval` at
the pre-registered iterations 50 and 150 on the bank-BC draws -> paired compares
vs base / bank / bcd / bcsw50 -> `summary.md` -> probe (`d_rej_m`: did deploy
samples leave the rejected draw). Tests: `slurm/test_pref_loss.sbatch`.
The whole chain is 64 qos_general jobs against a 50-job per-user cap (array tasks
count individually), so it is submitted in stages: `DFT_PREF_STAGE=train` (16),
then `DFT_PREF_STAGE=eval DFT_PREF_DIR=<run> DFT_PREF_ARMS=<arm>` (12 per arm), each
refusing up front if it would not fit. The first full run
(`pref_dpo/20261001_230431_2632246`) died on the disk quota at collection. The
rerun is `pref_dpo/20261002_140352_3015119`. On 10-02 it was cut to the vanilla
arm b1e4 x 3 seeds, at most 2 GPUs, scored at iter 150 only. The low-noise and
edit arms run only if vanilla moves the failure split, as the plan sequenced them.
At beta 1e4, z climbs from 0 to about 20 by iteration 75, with accuracy about 0.95.
The init check used to compare the loss with log 2, which is not beta-free: at
beta 1e4 a 4.8e-6 float gap becomes |z| = 0.01. It now checks that the params are
bit-identical, plus the loss gap relative to the loss.

## Gated tail adapter: the repair optimizer inside the sampler (2026-10-04)

Full write-up, commands and how to switch it off: `docs/adapter_distill.md`.

The repair optimizer `o_lpA3` is distilled into two residual adapters, added
after the UNet's last two up blocks. They are active only in the last K = 5
of the 10 late sampling steps, scaled by `strength * gate`. There is one
learned gate per plan, computed from frozen features at the first adapted
step. Added parameters: 41,921, which is 0.13% of the denoiser. Latency is
372.9 vs 371.8 ms per replan call at batch 32.

The difference from bank BC: pairs store the frozen prefix output and the
per-row late noise. The student therefore replays its own draw, and the loss
is charged on the final sampled plan, not on a denoising target.

* **Built.** `adapter.py`, `train_adapter.py`, the `tail_hook` in the UNet,
  `proposal_select` arms `name=adapter:<learned|open|closed>[:strength]` with
  `--adapter_ckpt`, and `--dump_pairs`. Chain: `slurm/submit_adapter_pilot.sh`.
  Tests: `slurm/test_adapter.sbatch`.
* **Splits file.** `verified_seed777/eval_splits_3strata_womd.json` is the
  verified splits with `shard_glob` moved to `/data/user_data/yixiz/autonlab/zfsauton/datasets/womd`; the
  eshau copy is gone.
* **Run.** `adapter/20261004_005655_968306`. Results are in its `report.md`
  and in the doc's results section.
