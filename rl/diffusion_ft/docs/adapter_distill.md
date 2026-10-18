# Gated tail adapter: the repair optimizer, distilled into the sampler

Started 2026-10-03. This note covers the design, the exact commands, and how
to switch the adapter off. Results go in the last section once the pilot has
run.

## What it is

The deploy-time repair `o_lpA3` runs a lateral/longitudinal trajectory
optimizer on every replan whose base plan is tight or violating. Its corpus
gain is m3 +0.0235 with 0 successes broken. It costs about four optimizer
calls per replan, and it needs a scorer at deployment.

The adapter tries to make the diffusion policy produce the repaired plan in
its own sampling pass. Two pieces are added to the frozen denoiser (a
`UNet1DConditioned`):

* **Two residual adapters**, one after each of the last two up blocks. These
  are the trajectory-processing blocks at full horizon resolution, width
  d = 128. They are already cross-attended to the scene tokens and FiLM-ed by
  the instruction:

  ```
  adapter(h) = Linear(32 -> d)(SiLU(Linear(d -> 32)(LayerNorm(h))))
  ```

  The last `Linear` starts at zero.

* **One gate per sampled plan.** It is computed once, at the first adapted
  step, from the frozen hidden state just before the first residual is added:
  mean- and max-pooled over the 25 horizon tokens and concatenated with the
  UNet's own timestep feature. That is
  `Linear(384, 64) -> SiLU -> Linear(64, 1) -> sigmoid`. The scalar is shared
  by both adapters and the whole tail, and is recomputed at every replan. The
  tokens are already scene-conditioned, so no separate scene summary is added.

The update rule is:

```
h' = h + strength * gate * adapter(h)        strength in [0, 1], external
```

The adapters are active only in the **last K = 5 steps** of the deployed late
chain. The late chain is the k_trainable = 10 unit steps that
`DiffusionRLActor` runs after the frozen 90-step prefix. The sampler, noise
schedule, v-parameterization, conditioning and decoding are all the base's.
Added parameters: 2 x 8,608 for the adapters plus 24,705 for the gate, about
42k in total. For scale, the base denoiser has 31.7M.

Code: `rl/diffusion_ft/adapter.py`, plus `tail_hook` in
`model/diffusion/diffusion.py`. Without a hook, the traced graph is unchanged.

## Why this might work where bank BC did not

`docs/bank_bc_distillation.md` concluded the following. The teacher's edit
depends on the sampled draw, but a denoising loss on the targets never shows
the student which of its own draws were bad. Weak pressure moved nothing.
Strong pressure shifted every draw and broke successes.

Here the training example is the draw itself. Each pair stores two things:

* the frozen prefix output `x_{k-1}`;
* the per-row late noise. `sample_with_trace` draws one `[B, C, T]` tensor per
  late step from `key_late`, so a row's noise depends on its batch, and the
  key alone would not reproduce it.

The student replays the base's late chain up to the tail, runs the adapted
tail with that row's remaining noise, and is supervised on the **final
sampled trajectory**. That is the plan the policy would actually execute for
this draw. The gate sees the same draw's features, so "is this draw bad" is
an input, not something to be averaged over.

## Data: paired repairs

`proposal_select --dump_pairs` collects the base arm and `o_lpA3` on the
training pools only:

* failure_train: 173 scenes x 4 repeats;
* success_train: every 56th scene (about 400) x 2 repeats;
* seed 2718.

The teacher config is the reference run's exactly: bank `olp` with tokens
`0,lon0,prev,lonprev`, ramp 20, cubic easing, cv neighbours, and `proj` with
theta 0.3.

Each alive replan writes one row containing:

* the condition;
* `x_prefix` and `noise_late` (`[10, 5, 25]`);
* `x_base`, the base draw;
* `world_base` and `world_exec`, the base plan and the executed plan in world
  coordinates (`[52, 5]`);
* the replan's `aux` transform, `t0`, the window `n`, the goal;
* the base plan's scores (`feat`), and `switched`;
* the episode outcome, legacy and term-aware.

The trainer builds three pools, using the term-aware outcome:

| pool | rows | role |
|---|---|---|
| recovery | teacher arm, switched, episode solved | `L_match` target = executed plan; gate label 1 |
| keep | base-arm rows of solved episodes, plus teacher-kept rows **after** its first switch (earlier ones duplicate the base arm's) | `L_keep` target = base plan; gate label 0 |
| failed | teacher arm, switched, episode not solved | `L_repair` only; unlabeled |

`o_lpA3` switches in two cases: when the base plan is predicted to violate
(`clear_p < 0`), and when a clean plan is inside the 0.3 m margin (the
projection rule). Recovery rows of both kinds are kept; the trainer reports
them separately as `rec_violation` / `rec_tight`. Validation **scenes** (15%)
are held out before any sampling. The real evaluation uses the held-out
splits, which share no scenes with the training pools.

## Objective

Everything is charged on the final sampled plan, mapped to world coordinates
with the row's own transform:

```
L = L_match + keep_weight * L_keep + repair_weight * L_repair + gate_weight * L_gate
```

* **`L_match` and `L_keep`** use `adapter.traj_distance`. That is
  `train_safety.preserve_loss`'s metric (position m², wrapped heading at
  0.1 rad, speed at 1 m/s) over the scored window, world steps 1..n.
* **`L_repair`** is the optimizer's own objective evaluated on the student's
  plan (`adapter.make_repair_objective`). Its parts are:
  * the hinge2 barrier with segment road edges and 0.3 / 0.75 m margins, over
    20 steps, with cv neighbours read from the reloaded scene;
  * plus `0.5 x^T H x`, where x is the plan's edit against the base plan in
    lateral / along-path coordinates. The config is
    `lon=1, w_lon_prox=0.05, lon_back_m=1000`: o_lpA3's comfort terms plus the
    closeness term.

  It applies to recovery and failed rows. Scenes are reloaded by id with
  `sim_trajectory := log_trajectory`, because a freshly loaded state's sim
  trajectory is zero past t = 0. Before training, the trainer checks that the
  teacher's scorer reproduces its recorded base clearances on these reloaded
  scenes.
* **`L_gate`** is BCE on the labeled rows.

Defaults: `keep_weight` 1, `repair_weight` 1 (the optimizer's own scale),
`gate_weight` 0.1. Adam at lr 1e-4 with global-norm clip 1.0, and 16 recovery
+ 8 failed + 16 keep rows per batch. Half of the keep rows come from
failure_train scenes.

## Stages

0. **Preflight.** Training starts only if both checks pass; otherwise the job
   writes `preflight.json` and exits 0 with no checkpoint.
   1. **Identity.** Every row's stored randomness replays its base draw. The
      initial adapter reproduces it with the gate open, with the learned gate,
      and at strength 0. Tolerance 1e-4 normalized, i.e. 1 cm.
   2. **Gradient.** One backward pass with the gate open. The final
      projections get finite, nonzero gradients; the down-projections get zero
      gradients, as expected at init. The base parameters are unchanged by a
      step.
1. **A.** 500 updates, gate held at 1, adapters only, on `L_match` + `L_keep`.
   `L_repair` switches on at step 250 if validation `L_match` has fallen at
   least 10% by then.
2. **B.** 200 updates, adapters frozen, gate only: BCE on the bootstrap labels
   (recovery 1, keep 0).
3. **Relabel.** For every teacher-switched row, the learned adapter is held
   fully on and compared with fully off on the same draw, using the teacher's
   own scorer. A row is labeled positive when the student's plan is clean,
   gives up no more than 1 m of progress, and (for a margin-triggered row) is
   at least 1 cm clearer. The other teacher-switched rows are unlabeled; keep
   rows stay 0. The teacher-repairable rows the student cannot recover go to
   `student_unrecovered.jsonl`.
4. **C.** 500 updates of adapter + gate on the full objective with the
   refreshed labels. The softly gated plan is trained directly, and every 4th
   minibatch runs its recovery rows with the gate forced open.
5. **Verify and bench.** Strength 0 must reproduce the base draws again. An
   offline strength sweep (0, 0.25, 0.5, 0.75, 1) runs on the validation
   scenes, plus the gate's quantiles on recovery vs keep rows. Latency is
   timed at batch 32, deployed base sampler vs adapter policy (prefix + tail).

## Commands

From the repo root. On lov4, `sbatch` is local; elsewhere, submit from rhea.

```bash
# smoke: 8 scenes per training split, 20/10/20 steps, 4 eval scenes
DFT_ADP_SMOKE=1 DFT_ADP_TAG=smoke bash rl/diffusion_ft/slurm/submit_adapter_pilot.sh

# the pilot: collect -> train -> eval (+ strength sweep) -> analysis/compares/ride/report
bash rl/diffusion_ft/slurm/submit_adapter_pilot.sh

# retrain and re-evaluate on an earlier collection (e.g. the K=10 escalation)
DFT_ADP_PAIRS_FROM=/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/adapter/<stamp> \
  DFT_ADP_K=10 DFT_ADP_TAG=k10 bash rl/diffusion_ft/slurm/submit_adapter_pilot.sh

# re-render the report from what is on disk
sbatch rl/diffusion_ft/slurm/adapter_report.sbatch <run>/report.env

# CPU tests
sbatch rl/diffusion_ft/slurm/test_adapter.sbatch
```

The run dir is `/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/adapter/<stamp>/`,
and `logs/dft/adapter_pilot_latest.txt` holds the latest one. Its contents:

* `collect/tasks/*/pair_data/<arm>/*.npz`;
* `train/{preflight,verify,bench,summary}.json`, `train/train_log.jsonl` and
  `train/adapter_{A,B,C*,final}/`;
* `eval/` (select rows, `compare_*.json`, ride);
* `sweep/`;
* `report.{md,json}`.

To evaluate any checkpoint as a policy, add an arm to a select run:

```
--arms 'base,adp=adapter:learned:1' --adapter_ckpt <run>/train/adapter_final
```

The eval uses `seed 4242`, 4 repeats, 4 shards x chunk 32, on a6000. That is
what pairs it with `select/lateral_opt_proj_20260930_171718_1665097`, whose
`o_lpA3` rows serve as "base + optimizer". The report refuses that pairing if
the base rows disagree on more than 0.2% of episodes.

### Switching the adapter off

The adapter is off exactly in any of these cases:

* strength 0, e.g. arm `adapter:learned:0`, or `tail_from_prefix(..., strength=0.0)`;
* gate mode `closed`;
* no adapter at all: the `base` arm, or no `--adapter_ckpt`.

At strength 0 every residual adds exactly zero. The preflight and the
post-training check measure this against the stored base draws.

## Splits file

`/zfsauton/scratch/eshau/` vanished around 10-02. The verified splits file
still names its glob, and `train_safety` refuses a splits file whose
`shard_glob` resolves to a different file list. On 2026-10-04 I wrote
`eval_splits_3strata_womd.json` next to the original, identical except for
`shard_glob` (now `/zfsauton/datasets/womd/...`) and a `shard_glob_moved`
note. The scene ids are unchanged: the offset indices were verified on 10-03.
`select_eval.sbatch`'s default shard glob points at the same place.

## Results: pilot 1 (2026-10-04, `adapter/20261004_005655_968306`)

**Data.** 19,150 rows collected. Training used:

* 1,120 recovery rows from 140 scenes (665 violation-triggered, 641
  margin-triggered across train and val);
* 6,592 keep rows;
* 71 failed-teacher rows.

186 recovery rows from 27 scenes were held out for validation.

**Preflight: both checks PASS.**

* The adapter at initialization matches the base chain bit-exactly.
* Stored randomness replays the deployed draws to within 9.3e-5 normalized
  (the float32 floor).
* The reloaded-scene scorer reproduces the teacher's clearances exactly.

**Closed loop.** Held-out splits, seed 4242, 4 repeats, 5,800 episodes. All
four policies share their draws; this eval's base matches the reference run on
5,799 of 5,800 episodes. Outcomes are term-aware:

| policy | failure | intermittent | success | m3 (Δ vs base, 95% CI) | base failures recovered | base successes broken | latency ms (b=32) | added params |
|---|---|---|---|---|---|---|---|---|
| base | 0.306 | 0.538 | 0.9975 | 0.9737 | — | — | 371.8 | 0 |
| base + optimizer (o_lpA3) | 0.903 | 0.951 | 1.0000 | 0.9971 (+0.0234) | 888/1002 | 7/4798 | (4 optimizer calls/replan) | 0 |
| adapter fully on | 0.400 | 0.550 | 0.9607 | 0.9403 (−0.0334 [−0.043, −0.024]) | 289/1002 | 358/4798 | 372.9 | 41,921 |
| learned soft gate | 0.366 | 0.547 | 0.9962 | 0.9736 (−0.0001 [−0.0027, +0.0025]) | 174/1002 | 128/4798 | 372.9 | 41,921 |

Broken counts per split for the soft gate:

* success: 11/3,990;
* intermittent: 105/596;
* failure: 12/212.

The corpus-weighted broken rate is 0.87%. The fully-on adapter breaks 154 of
3,990 successes.

Under the legacy outcome, the soft gate is m3 −0.0096 and the fully-on adapter
−0.0893. The adapter leaves more post-termination violations than the base.

The soft gate against `o_lpA3` is −0.0235 [−0.0265, −0.0206].

**Gate.** Its value on alive replans, by the base outcome of the same episode:

| split | base-failure: mean / share > 0.5 | base-success: mean / share > 0.5 |
|---|---|---|
| failure | 0.45 / 0.38 | 0.34 / 0.09 |
| success | 0.45 / 0.41 | 0.35 / 0.09 |
| intermittent | 0.43 / 0.30 | 0.44 / 0.33 |

The gate separates failure scenes from success scenes. It does not separate
the two kinds of episode inside an intermittent scene, which is where the
breaks are.

**Ride.** The soft gate's ride is near the base's. Per-episode peak lateral
acceleration, p50/p90 (m/s²), success split:

* soft gate 1.66/3.62;
* base 1.60/3.55;
* teacher 1.61/3.59;
* fully-on adapter 1.99/4.07.

**Offline, on validation scenes.** Stage A closed only about 10% of the
student-to-teacher distance (validation `L_match` 0.918 → 0.830, flattening
after step 400). Validation `L_match` had fallen 7.7% by step 250, short of the
10% bar, so `L_repair` never switched on in Stage A.

At the relabel step, the fully-on student was useful on 544 of 1,306 recovery
rows:

* 395 of 641 margin-triggered rows;
* 149 of 665 violation-triggered rows.

The learned gate's AUC was 0.71. Strength 0 after training matches the base
chain exactly.

The closed-loop strength sweep covers 48 scenes per split, 1 repeat. Failure
solved goes 0.31 → 0.42 → 0.48 at strength 0, 0.25 and ≥ 0.5; successes stay
at 48/48; intermittent breaks go 0 → 3.

**Reading.**

* The tail adapter reproduces about a third of the teacher's recoveries when
  fully on (289 vs 888), and it shifts working plans enough to break 7.5% of
  base successes.
* The gate cuts the breaks to 128 but also halves the recoveries. The result
  is a corpus tie with the base.
* Both measured limits are present:
  * **fit:** the open adapter is far from the teacher, and imitation stalls at
    about 10%;
  * **selectivity:** the gate is weak inside intermittent scenes.

  The plan's order for a fit limit is K 5 → 10 first, then width, on the same
  pairs (`DFT_ADP_PAIRS_FROM=... DFT_ADP_K=10`).

## Results: K = 10 (2026-10-05, `adapter/20261005_203755_1213956_k10`)

The first escalation from the plan: the same pairs and schedule, with the
adapter active on all 10 late steps instead of the last 5. The checkpoint
config confirms `K=10`. The eval used the same draws; base mismatch against the
reference run is 4/5,800.

| policy | failure | intermittent | success | m3 Δ vs base (95% CI) | recovered | broken |
|---|---|---|---|---|---|---|
| base + optimizer (o_lpA3) | 0.903 | 0.951 | 1.0000 | +0.0234 | 887/1001 | 7/4799 |
| adapter fully on, K=10 | 0.403 | 0.553 | 0.9720 | −0.0225 [−0.031, −0.015] | 281/1001 | 300/4799 |
| learned soft gate, K=10 | 0.370 | 0.553 | 0.9965 | +0.0004 [−0.0022, +0.0029] | 170/1001 | 114/4799 |
| *(pilot 1, K=5, for reference)* fully on | 0.400 | 0.550 | 0.9607 | −0.0334 | 289/1002 | 358/4798 |
| *(pilot 1, K=5)* soft gate | 0.366 | 0.547 | 0.9962 | −0.0001 | 174/1002 | 128/4798 |

Under the legacy outcome, the K=10 soft gate is −0.0088 [−0.0144, −0.0038].

**K is not the limit.** Offline, the K=10 run tracks K=5 almost exactly:

| measure (validation scenes) | K=5 | K=10 |
|---|---|---|
| student-to-teacher distance closed by Stage A | 9.6% | 9.4% |
| fully-on adapter useful on recovery rows | 0.41 | 0.41 |
| recovery rows the student recovers at relabel | 544 | 542 |
| mean shift at strength 1 | 0.156 m | 0.159 m |

In closed loop, recoveries are unchanged (281 vs 289). Breaks fall somewhat,
mostly among successes (109 vs 154 of 3,990). The gate is no more selective
inside intermittent scenes: it is open on 0.33 of replans in base-failure
episodes vs 0.36 in base-success episodes.

Doubling the adapted span leaves both how far the adapter moves plans and
where it moves them unchanged. The next item in the plan's order is the adapter
width.

## Fit diagnostic (2026-10-06, `adapter/20261006_195051_3827310_overfit`, job 74278)

The question: did the pilot stall because the adapter cannot express the
repair, or because training is the problem?

The test fits a fixed set of 64 training recovery rows: 55 scenes, 30 of the
rows violation-triggered. Every step uses the full set, with the gate open and
the matching loss only, for 3,000 steps from a fresh init
(`train_adapter --overfit_rows 64`). On these rows the base plans are 0.38 m RMS
from the teacher's plans, and the teacher's plans are 100% "useful" under its
own scorer.

| width : lr | closed @ 500 / 1000 / 3000 steps | position RMS @ 3000 | useful @ 3000 | held-out val closed |
|---|---|---|---|---|
| 32 : 1e-4 (the pilot's) | 0.73 / 0.93 / 0.98 | 2.4 cm | 1.00 | −0.51 |
| 32 : 1e-3 | 0.97 / 0.98 / 0.98 | 1.4 cm | 1.00 | −0.48 |
| 32 : 3e-3 | 0.98 / 0.98 / 0.99 | 0.9 cm | 0.98 | −0.50 |
| 128 : 1e-3 | 0.98 / 0.99 / 0.99 | 0.7 cm | 0.98 | −0.56 |

**Expressivity and optimization are not the limit.** Even the pilot's width
and learning rate fit 64 repairs to a few centimetres. Every violation-triggered
row ends up clean under the teacher's scorer.

**Generalization is the limit.** Every fit makes the held-out recovery rows
about 50% *worse* than no adapter. What it memorizes is specific to its rows.
Applied elsewhere, it is a large shift in the wrong direction (up-projection
norms 3–6, against about 0.5 in the pilot).

The pilot's roughly 10% on held-out rows is the part that transfers from 1,120
rows in 140 scenes. Width (and, as already measured, K) cannot change that. The
levers are the amount and diversity of repair data, and regularization with a
validation stopping point.

## Data-scaling probe (2026-10-07, `adapter/20261007_132342_100112_scaling`, job 75341)

Training used nested subsets of the training recovery rows, each from a fresh
init: batch 128, lr 1e-3, width 32, gate open, matching loss only, 3,000 steps.
Every 100 steps it was scored on the 186 held-out recovery rows (27 scenes,
disjoint from the training scenes), with a scene-bootstrap 95% CI.

| training rows (scenes) | best held-out closed (step) [95% CI] | final held-out closed [CI] | final train closed | final drift on held-out keep rows |
|---|---|---|---|---|
| 64 (46) | 0.000 (step 0) | −0.98 [−1.61, −0.56] | 0.98 | 0.47 m |
| 256 (108) | 0.000 (step 0) | −0.86 [−1.27, −0.41] | 0.96 | 0.39 m |
| 512 (128) | +0.034 (100) [−0.10, +0.18] | −0.49 [−0.78, −0.16] | 0.91 | 0.35 m |
| 1,120 (140) | +0.073 (100) [−0.07, +0.23] | −0.47 [−0.91, −0.04] | 0.86 | 0.30 m |

Pilot 1 (lr 1e-4, which acts as an early stop) reached +0.10 at 1,120 rows.

**Reading.**

* Training rows are always memorized. The best held-out point comes within the
  first 100 steps; after that the adapter memorizes, and held-out rows get
  worse than no adapter.
* The transferable part grows with data:
  * best held-out: 0 → 0 → +0.03 → +0.07;
  * memorization damage: −0.98 → −0.47.
* It grows slowly: about +0.04 per doubling of rows, with CIs that include zero
  at every size.
* Extrapolating, reaching even a third of the teacher's effect would need tens
  of times more repair data.
* Violation-triggered repairs (the ones that recover failures) come almost
  entirely from the 173 failure_train scenes. Scaling them means scanning the
  corpus for new base-failure scenes. Success_train mostly yields margin
  (tight) repairs.

`val_useful` (0.2–0.4) is not a clean generalization measure. A large generic
shift sometimes clears a tight plan, so it stays positive even where held-out
distance is −0.9.
