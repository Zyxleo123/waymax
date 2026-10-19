# Behaviour-cloning the repair bank into the policy: a negative result

2026-09-28 → 09-30. Two training runs, three seeds each, one diagnostic probe.
This file is written so that someone who was not there can propose the next
attempt. Section 6 lists ideas, ranked against the evidence. Section 5 separates
what was measured from what is only a hypothesis.

## TL;DR

* **The teacher.** The deploy-time repair bank `b_l24_r20` takes the corpus
  solved rate from 0.9715 to **0.9924**: m3 **+0.0209**
  [+0.0185, +0.0236]. It breaks 0.0003 of successes.
* **Plain BC** of the bank's executed plans into the base policy recovers
  **~6%** of that: m3 +0.0012 over three seeds. The training curve is flat, and
  the edit rows are under-fit.
* **BC with half of every batch on the edited rows** fits the edits (loss 0.40×
  base) but is **corpus-negative**: m3 −0.020. Failures go up (0.29 → 0.49), and
  successes break (0.997 → 0.969), mostly by going off-road.
* **The frozen denoising prefix is not the bottleneck** (measured). The late
  trainable steps can express the edits.
* **Working explanation.** The bank's edit depends on the specific sampled plan
  in front of it, and on the other cars' recorded future. The student sees only
  the scene. Learning the edit therefore means shifting the whole distribution,
  which breaks successes. This is the same do-no-harm trade-off every
  weight-space fine-tune in this package has hit.

Vocabulary (`cons`, bank, arms, m2/m3) is defined in the main README. In short,
m3 is the corpus-level solved rate, re-weighted over the three held-out strata:
failure 1.45%, success 95.6% and intermittent 3.0% of the corpus. **Success
breaks dominate it**: a +0.20 failure gain is worth about +0.003, while a −0.025
success loss costs about −0.024.

---

## 1. The teacher

`b_l24_r20 = bank@l24r20:cons:20`. The base diffusion policy proposes a plan at
every replan. The bank then offers 24 rigid lateral offsets of that same plan:
±0.1, 0.175, 0.25, 0.325, 0.4, 0.5, 0.6, 0.75, 0.9, 1.1, 1.35 and 1.6 m. Each
is eased in over 20 world steps with a cubic profile.

The `cons` selector switches to the smallest-magnitude offset only when three
things hold over the next 20 steps (2 s):
* the base plan is predicted to violate (`clear_p < 0`);
* the offset is clean;
* the offset gives up at most 1 m of progress.

Other cars come from the **recorded log**. Under log replay that is an oracle of
their 2 s future, which is acceptable here because the bank only *labels*
training data. The sim uses the median clock (`world_dt_method=median`) and the
other agents replay the log (`sim_agents=none`).

Its scores, eval seed 4242, 1,450 held-out scenes × 4 repeats:

| | failure (173) | intermittent (277) | success (1000) | m3 |
|---|---|---|---|---|
| base | 0.288 | 0.496 | 0.9968 | 0.9715 |
| bank | 0.775 | 0.870 | 0.9995 | 0.9924 (+0.0209) |

The bank switches on 15.7% of failure replans and on **0.1%** of success
replans. It reproduced its earlier score (`retrain_worlddt/.../eval_old`:
0.9921, +0.0206) on identical base draws.

## 2. What was done

The student is the original pretrained policy
(`mineuih/.../pretrain_diffusion_without_subgoal_20260616_013642/latest`),
fine-tuned the way every `train_safety` arm is. A trainable copy drives the last
`k_trainable = 10` of 100 DDPM steps, and the first 90 stay frozen.

**Collection** (`proposal_select --dump_bc`, new). The teacher drove the base
policy on the *training* pools only:
* `failure_train`: all 173 scenes × 4 repeats.
* `success_train`: every 22nd scene, about 2,024 scenes × 1 repeat. The split
  file is sorted by id, so a stride spreads the sample across the corpus.

Sampler seed 2718. At every alive replan it stored:
* the frozen condition (`cond1` 32×256, `cond2`, `mask`);
* the plan the arm **executed**, in the denoiser's normalized `[25, 5]` layout.
  That is the base's own sample verbatim where the bank kept it, and
  `world_to_norm(edited world plan)` where it switched;
* a covered-steps mask, plus `switched`, the chosen offset and the episode
  outcome.

The world↔normalized round trip loses 0.6 mm at p50, 2.4 cm at p99 and 12 cm at
worst.

| collected | rows | edit rows | episodes solved by teacher |
|---|---|---|---|
| failure_train | 5,301 | 769 | 0.72 |
| success_train | 11,447 | 44 | 0.995 |

Training uses rows from solved episodes only: **611 edit rows** (572 failure +
39 success) out of 15,342, from **132 scenes**. Of the 173 `failure_train`
scenes, 107 contribute at least one edit row. The top 20% of scenes hold 41% of
the edit rows, and the most any single scene contributes is 14.

The chosen offsets are mostly small: median 0.33 m, p75 0.60 m, and 10% at
≥ 1.1 m. Edits are spread over replans 1–6, by index:
`[35, 79, 108, 134, 113, 80, 51, 11]`.

**Training** (`train_safety` with the safety term and the base anchor both off).
* The only objective is `late_denoise_loss`: x0-prediction MSE of the trainable
  copy at t ~ U[0, 10), on the targets above, masked to covered steps.
* Batch 32, half `failure_train` and half `success_train`.
* 20 Adam steps per iteration × 151 iterations = 3,020 steps, lr 1e-5.
* Eval every 10 iterations on failure/success held-out (seed 12345); checkpoints
  every 25.
* The success rows are the base's own samples. They were meant as the tether, so
  there is no separate anchor.

| run | batch composition | dir |
|---|---|---|
| `bcd` | uniform over solved rows → ~7% edit rows per batch | `bank_bc/20260928_121223_39589` |
| `bcsw50` | 16 edit + 8 failure-kept + 8 success-kept (`--repair_switch_share 0.5`) | `bank_bc/20260930_000014_1275746` |

**Scoring** is pre-registered at iteration 150. It uses `select_eval` on the
same draws as the bank and the retrained safety fine-tune, so the base arms
match on 5,799–5,800 of 5,800 episodes. Every comparison below is therefore
paired per (scene, repeat).

## 3. Results

| policy | failure | intermittent | success | success off-road / overlap | m3 vs base [95% CI] |
|---|---|---|---|---|---|
| base | 0.288 | 0.496 | 0.9968 | 0.0027 / 0.0008 | — |
| bank (teacher) | 0.775 | 0.870 | 0.9995 | 0.0005 / 0 | **+0.0209** [+0.0185, +0.0236] |
| retrained safety ft (`eval_new`) | 0.541 | 0.638 | 0.986 | — | −0.0023 |
| `bcd` s0 / s1 / s2 | 0.331 / 0.393 / 0.312 | 0.568 / 0.605 / 0.553 | 0.9948 / 0.9955 / 0.9938 | ~0.005 / ~0.0007 | +0.0009 / **+0.0036** / −0.0008 |
| `bcsw50` s0 / s1 / s2 | 0.507 / 0.493 / 0.480 | 0.634 / 0.606 / 0.615 | 0.9725 / 0.9635 / 0.9715 | 0.021–0.029 / ~0.008 | −0.0159 / −0.0255 / −0.0178 |

The reached-goal rate is unchanged against base on every split for every arm
(failure ≈ 0.98, intermittent ≥ 0.99, success ≈ 1.0). So nothing here is a pace
or stalling effect.

Paired m3 deltas, each the seed mean, in `eval/compare_vs_*.json`:

| comparison | m3 [95% CI] | per split |
|---|---|---|
| `bcd` − bank | −0.0197 [−0.0229, −0.0168] | failure −0.43, success −0.005 |
| `bcd` − safety ft | +0.0035 [−0.0005, +0.0075] | a tie: fewer fixes, fewer breaks |
| `bcsw50` − `bcd` | −0.0209 [−0.0278, −0.0149] | failure **+0.148**, success **−0.025**, intermittent +0.044 |
| `bcsw50` − safety ft | −0.0174 [−0.0240, −0.0111] | |
| `bcsw50` − bank | −0.0406 [−0.0482, −0.0335] | |

The break rate on intermittent scenes rises from 0.07–0.08 (`bcd`) to 0.11–0.12
(`bcsw50`).

**Training curves** (`train/<tag>_s*/train_log.jsonl`).
* `bcd`: every eval point of every seed is within ±0.004 of base. `repair_loss`
  sits at about 4e-5 throughout.
* `bcsw50`: failure solve jumps to 0.46–0.49 by iteration 10 and then plateaus
  around 0.45. The marginal delta drifts from about −0.005 (three-seed mean over
  iterations 0–30) to about −0.019 at iteration 150. No eval point is reliably
  positive, so a different stopping point would not rescue it.

## 4. Diagnostic probe

`tools/probe_bc_transfer.py`, output in `probe/bc_transfer.json`. It uses 512
edit rows (solved failure episodes) and 512 kept rows, with the same rows,
`(t, eps)` draws and sampler keys for every checkpoint.

* `loss` is the training objective itself, as a ratio to the base.
* `d_free` is the distance from a deploy-time sample (frozen prefix + late copy)
  to the target. It is a mean over covered steps, in metres.
* `d_tf` is the same late chain started from a **teacher-forced** prefix
  `q_sample(target, t = 9)`, which is the input the loss trains on.
* `closed` is `1 − d(ckpt) / d(base)`.

| checkpoint | edit rows: loss | edit rows: closed_free | edit rows: closed_tf | kept rows: loss | kept rows: closed_free |
|---|---|---|---|---|---|
| base (d_free 0.680 m edit / 0.450 m kept; d_tf 0.641 m) | 1 | 0 | 0 | 1 | 0 |
| `bcd` s0 @25 | 0.85 | +0.080 | +0.077 | 0.98 | +0.084 |
| `bcd` s0/s1/s2 @150 | 0.69 / 0.69 / 0.67 | +0.11 / +0.12 / +0.11 | +0.11 / +0.12 / +0.10 | 0.97–1.00 | +0.082–0.089 |
| `bcsw50` s0 @25 | 0.57 | +0.12 | +0.13 | **1.21** | +0.016 |
| `bcsw50` s0/s1/s2 @150 | 0.40 / 0.40 / 0.40 | **+0.23 / +0.22 / +0.22** | +0.23 / +0.22 / +0.22 | **1.10 / 1.14 / 1.11** | +0.03–0.06 |

What each column settles:

1. **The frozen prefix is not the limit.** For every checkpoint `closed_free`
   equals `closed_tf` within 0.01. Even teacher-forcing the prefix *from the
   target itself* lands the base only 4 cm closer (0.64 vs 0.68 m). The
   arithmetic agrees: noise std at late step t = 9 is
   `sqrt(1 − ᾱ_9) = 0.167` normalized, which is **16.7 m** at ego_range 100.
   At t = 4 it is 8.9 m and at t = 0 it is 2.5 m. The late steps have far more
   freedom than a 0.3–1.6 m edit needs. **Do not build a full-chain arm on
   expressivity grounds.**
2. **`bcd` under-fit the edits.** Its edit-row loss was still falling at 3,000
   steps (0.85 → 0.69). Its `closed_free` on edit rows (+0.11) is mostly the
   same generic sharpening it shows on kept rows (+0.085), so the
   edit-specific part is about 2.5 points.
3. **`bcsw50` fit the edits but drifted off the base.** It gets about 18
   edit-specific points (+0.22 on edit rows vs +0.04 on kept rows). But the
   loss on rows where the teacher *kept* the base sample **rose** 10–14% (21% at
   iteration 25). The student moves on scenes where the teacher did nothing.
   That matches the eval: successes break, mostly off-road (0.003 → 0.02–0.03).

The probe runs in-sample: it uses training rows, not held-out ones.

## 5. Why it fails: measured vs hypothesized

**Measured:**
* The edits are small next to the policy's own sampling spread. The median
  chosen offset is 0.33 m. A fresh base sample sits 0.45 m from an old base
  sample on kept rows, and 0.68 m from the edit target on edit rows.
* The teacher edits very rarely where the base is fine: 0.1% of success
  replans.
* Fitting the edits (run 2) moved kept-row behaviour too, and the successes
  broke.
* The prefix is not the limit; see section 4.

**Hypothesis (consistent with the above, not directly tested).** The teacher's
action is a function of (scene, **the realized draw x_p**, **the other cars'
recorded 2 s future**). The student sees only the scene.

At a state where the teacher edits some draws and keeps others, the BC target is
a mixture: {kept draws} ∪ {edit(bad draws)}. Reproducing it requires moving
probability mass off *bad draws specifically*. A denoising loss on the targets
alone teaches "where good plans are". It never shows the student its own bad
draws, so it cannot learn "which of my draws to avoid". With weak pressure (run
1) nothing moves; with strong pressure (run 2) the whole conditional
distribution shifts, which helps failures and hurts successes.

About a quarter of the bank's switches are triggered by a future-car conflict
(`bank_honest` attribution). That part is unlearnable from the scene in
principle.

**Also unexamined: covariate shift.** All training states come from teacher
rollouts, not from the student's own. DAgger was not tried.

## 6. Improvement ideas, ranked against the evidence

1. **Use the BC policy as a *proposal* under the selector.** This is deployment,
   not distillation. Under `cons` the burden of proof sits on the intervention,
   so the student's off-road drift on successes is filtered out. That is the
   README section "The ranking inverts under selection (2026-09-14)": the worst
   fine-tune on its own became the best alternative under `cons`. `bcsw50` alone solves about 0.49 of
   failures.
   *Needs:* an arm that offers the `bcsw50` sample *and* the bank candidates, or
   the bank on top of a `bcsw50` primary. `Arm.primary` exists, but
   `parse_arms` hardcodes `primary="base"` for every alt arm.
   *Read:* m3 vs the bank alone on the same draws.
2. **Preference / contrastive distillation instead of BC.** This attacks the
   mixture problem in section 5 directly. For each edit row, keep the base draw
   x_p that the teacher rejected. Then train a Diffusion-DPO-style objective:
   raise the edit's likelihood *relative to the rejected draw* under the same
   condition, with the frozen base as reference. That says "avoid this draw"
   instead of "go here".
   *Needs:* `_bc_rows` to also store `x_p` for switched rows. Today the data
   does not have it, so this needs recollection; `counterfactual.py` already
   stores a `base` field for exactly this reason. Then a pairwise loss in
   `rl_actor`.
3. **DAgger: let the teacher label the student's own draws.** Roll out the
   current student with the bank on top, dump the rows, aggregate them and
   retrain. This gives the student examples of *its own* bad draws next to their
   repairs, and removes the covariate shift. It needs the same primary=ft bank
   arm as idea 1, then `--dump_bc` on it. Cost is about one collection pass per
   round (~15 min GPU).
4. **Tether the kept rows explicitly.** Run 2's kept-row loss rose 10–21%. Add
   `base_anchor_loss` (KL to the frozen base, which already exists in
   `rl_actor`) on kept rows only, or weight them up. Prior: every tether in this
   package has bought parity, not profit, because it works by making the
   fine-tune more base-like.
5. **Better teacher, more edit data.**
   * An honest and stronger teacher now exists: the lateral-optimization repair,
     latest `o_lpA3` / always-on `proj`. It scores m3 about +0.0235 with
     constant-velocity neighbours, not the oracle. Because it is always on, it
     also nudges tight *success* plans, which gives the student edits on the
     kind of scene it breaks. Any selector arm can be dumped with `--dump_bc`.
   * The 611 edit rows come from 132 scenes. The hardness heuristic (AUC 0.86;
     the top decile holds half the failures) could pick `success_train` scenes
     where the teacher fires, to multiply the edit data.
6. **Finer sweeps** of switch share (0.15–0.3), lr or stopping point. These are
   cheap, but the `bcsw50` curve shows the success damage arriving with the
   failure gain at every point. Low prior.

## 7. Limits of these measurements

* One lr (1e-5), one batch size (32), one step budget (3,000). Switch share was
  tested only at about 0.07 (uniform) and 0.5.
* Late-copy only (`k_trainable = 10`). Full-chain training was not tried; the
  probe says it is not needed for expressivity, but it would change which steps
  learn the scene→edit map.
* Only iteration 150 was scored with `select_eval`. Earlier checkpoints
  (iterations 25–125) were seen only on the 1-repeat training curve. Those
  checkpoints are still on disk in both run dirs.
* The teacher is the oracle-neighbour bank. Its labels on agent-triggered
  switches use information the student cannot have.
* The probe is in-sample (512 + 512 training rows).

## 8. Reproduce, artifacts, code

```bash
# from the repo root on rhea or lov4 (lov4 submits directly)
DFT_BCD_SMOKE=1 bash rl/diffusion_ft/slurm/submit_bank_bc.sh        # smoke, ~8 scenes, 3 iters
bash rl/diffusion_ft/slurm/submit_bank_bc.sh                        # run 1 (collect + bcd)
DFT_BCD_DATA_FROM=/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/bank_bc/20260928_121223_39589 \
DFT_BCD_SWITCH_SHARE=0.5 DFT_BCD_TAG=bcsw50 \
  bash rl/diffusion_ft/slurm/submit_bank_bc.sh                      # run 2, reuses run 1's rows
```

The chain is collect → train (3 seeds) → `select_eval` @150 → analysis +
paired compares, plus the probe after training.

Artifacts live under `/data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/bank_bc/`:
* `20260928_121223_39589/` — run 1. `collect/tasks/*/{rows.jsonl, bc_data/b_l24_r20/*.npz}`
  (the 16,748 rows), `train/bcd_s*`, `eval/`, `probe/`.
* `20260930_000014_1275746/` — run 2. `train/bcsw50_s*`, `eval/`, `probe/`.
* `20260928_121219_39381/` — the smoke run.

Code (all new code is behind flags; defaults reproduce every earlier run):

| file | what |
|---|---|
| `proposal_select.py` | `--dump_bc`, `ProposalSelectEval._bc_rows`. Writes `bc_data/<arm>/<split>_s<shard>_<start>.npz` and refuses held-out splits. |
| `train_safety.py` | `--repair_steps`, `--repair_rows {all,solved}`, `--repair_split_ratio`, `--repair_switch_share`, in `_init_repair` / `repair_grads`. The loss is `rl_actor.late_denoise_loss`. |
| `tools/probe_bc_transfer.py`, `slurm/probe_bc_transfer.sbatch` | the section 4 probe |
| `tools/compare_select.py` | `--arm_a/--arm_b`: pair two *different* arms, e.g. BC ft vs the bank |
| `slurm/submit_bank_bc.sh` | the chain; options `DFT_BCD_{SMOKE,DATA_FROM,SWITCH_SHARE,TAG,DIR}` |
| `slurm/sweep_safety_stopping.sh` | `DFT_ARM_SPECS` (custom arm table) and `DFT_EXTRA_ENV_FILE` (shared overrides) |
| `slurm/select_eval.sbatch`, `slurm/train_safety.sbatch` | `DFT_DUMP_BC`; `DFT_REPAIR_{STEPS,ROWS,SPLIT_RATIO,SWITCH_SHARE}` |

`bc_data` npz schema, one row per (scene, repeat, alive replan):
* condition: `cond1 [32,256]`, `cond2 [256]`, `mask`;
* target: `target [25,5]` (normalized x, y, vx, vy, yaw/π), `valid [25]`
  (covered steps);
* edit: `switched`, `choice` (bank index, −1 = kept), `rt_err_m`;
* bookkeeping: `replan`, `t0`, `scene_id`, `repeat`, `split`;
* episode outcome: `ep_solved`, `ep_reached`, `ep_offroad`, `ep_overlap`.

The base draw is **not** stored for switched rows; see idea 2.
