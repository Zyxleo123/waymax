# Post-hoc trajectory optimization — integration guide

This is a training-free repair step that runs on a plan the planner has already
chosen. The diffusion policy (or your ES) proposes a plan as usual. Before it is
executed, the next 2 s are edited by minimizing **one loss**:

    L(d, s) = Safety(d, s) + Comfort(d) + Comfort(s)

Here `d_t` is a sideways offset and `s_t` an along-route offset, one of each per
world step. A plan already clear of every safety margin has `L = 0` at
`d = s = 0`. The optimizer then returns exactly zero and the plan runs
unchanged, so this step needs no trigger and no acceptance test.

This file covers only what you need to bolt the repair onto another planner,
such as the Diffusion-ES in `es_baseline/`. It describes one configuration, the
no-rule arm `o_am`. [lateral_opt.py](lateral_opt.py) and
[proposal_select.py](proposal_select.py) also hold earlier rule-based arms;
you need only the functions named below.

## What it buys

Base diffusion policy vs base + post-hoc optimization (arm `o_am`): held-out
splits, 1450 scenes × 4 seeds, constant-velocity neighbours, no oracle.

| solved rate        | failure (173) | intermittent (277) | success (1000) | corpus (marginal) |
|--------------------|---------------|--------------------|----------------|-------------------|
| base policy        | 0.306         | 0.545              | 0.9975         | 0.9739            |
| + post-hoc opt     | **0.903**     | **0.956**          | **0.9992**     | **0.9965**        |

* The corpus gain is +0.0227, with a 95% CI of [+0.020, +0.025].
* Off-road on the failure split falls from 0.487 to 0.038, and collision falls from 0.198 to 0.040.
* The success split is not harmed.
* Run: `select/lateral_opt_argmin_20261002_142129_3029435` (`select_summary_term.json`).
* About 44 of the 173 "failure" scenes were a clock bug, not a policy failure (see
  [Pitfall 2](#2-the-es-preprocess-still-has-the-first-pair-world-dt-bug)).

The integration is to run the repair once on whatever plan your planner
picks (see [the recipe](#recipe--repair-the-plan-your-es-picks)).

## How it works (the essentials)

**The edit.** The input is a world-frame plan `world[B, W, 5] = (x, y, yaw, vx, vy)`,
where index 0 is the current state (the anchor) and steps are 0.1 s apart. The
optimizer edits steps `1..n` (`n = 20`, i.e. 2 s):

    p'_t = route(S_t + s_t) + d_t * normal_t

1. [`branching.path_retime`](branching.py) moves each pose `s_t` metres along the plan's own path.
2. [`branching.lateral_offset`](branching.py) then moves it `d_t` metres along the left normal.
3. Heading and velocity are recomputed from the new positions.

At the boundaries of the edit window:

* At the anchor, the edit has zero offset, velocity and acceleration, so there is no jump.
* After the window, `d` is held constant, so the car arrives parallel.
* After the window, `s` continues at its last slope, so a speed change can persist.

**Safety.** The safety term is [`safety_objective.candidate_safety`](safety_objective.py),
a barrier on two exact clearances:

* `c_edge`: signed distance of each of the 4 box corners to the road edge. Its sign is
  identically waymax's `offroad`.
* `c_agent`: signed box–box gap to every other agent. Its sign is identically waymax's
  `overlap`.

The barrier is `hinge2`:

* It is exactly zero beyond the margin: 0.3 m for the road edge (`tau` 0.15) and
  0.75 m for other agents (`tau` 0.25). Both margins are near the low tail of the
  human drivers' measured clearance.
* It is quadratic through a knee of width `tau`.
* It is linear inside the knee.

Because the barrier is exactly zero beyond the margin, comfortable plans feel no
force at all.

**Comfort.** The comfort terms are quadratic in the *edit*, not in the plan:

* lateral acceleration and jerk of `d`;
* longitudinal acceleration and jerk of `s`;
* `w_lon_prox * mean(s_t^2)`, which says "stay on the policy's timing".

Together they form `0.5 xᵀ H x` with a constant matrix `H`
([`lateral_opt.edit_hessian`](lateral_opt.py)).

**Optimizer.** The optimizer is CHOMP-style preconditioned gradient descent,
`x ← x − 0.5 · H⁻¹ ∇L`:

* A per-row trust region of 0.25 m × 0.92^k bounds each step.
* It runs 32 iterations inside one `jax.lax.fori_loop`.
* It returns the lowest-loss iterate it has seen.

`H⁻¹` turns a barrier gradient that touches one corner at one step into a smooth
swerve in one iteration. Plain gradient descent would need hundreds.

**Starts (`o_am`).** The optimizer runs twice, keeping the lower final loss per
row:

* `init0`: start from no edit.
* `initprev`: start from the tail of the plan executed at the previous replan,
  projected onto the fresh plan's path with
  [`lateral_opt.project_onto_path`](lateral_opt.py). Both starts are scored by the
  same loss, so the comparison is fair.

A row whose winning `x` is all zeros keeps the original plan.

## Files you need

All of them are pure JAX/numpy plus `waymax`. Nothing here touches the denoiser.

| file | what you use |
|---|---|
| [lateral_opt.py](lateral_opt.py) | `LateralOptConfig`, `make_lateral_optimizer`, `edit_world`, `project_onto_path` |
| [branching.py](branching.py) | `lateral_offset`, `path_retime`, `fd_velocity` (called by `edit_world`) |
| [safety_objective.py](safety_objective.py) | `SafetyObjectiveConfig`, `candidate_safety`, `predicted_neighbor_window` (cv neighbours) |
| [safety_loss.py](safety_loss.py) | `SafetyLossConfig`, `safety_terms` (the clearances and barrier) |
| `data/preprocess.py` (repo root) | `compute_world_dt_seconds(state, 0.1, "median")`, called only when `neighbor_pred != "log"`. See [Pitfall 3](#3-module-name-collision-with-es_baselinedata) |

The tested reference for how `o_am` calls these is
`ProposalSelectEval._argmin` in [proposal_select.py](proposal_select.py) (around
line 858).

## The API

```python
opt = make_lateral_optimizer(objcfg, cfg, *, dt, neighbor_pred="log", world_dt_method="median")
x, loss_best, loss_init = opt(world_p, x0, state, t0, active_b, n)
```

| argument | shape / type | meaning |
|---|---|---|
| `objcfg` | `SafetyObjectiveConfig` | Only `.safety` (a `SafetyLossConfig`) is read. The defaults are the calibrated margins. `cfg.barrier_shape` / `cfg.edge_geometry` override it for the loss only. |
| `cfg` | `LateralOptConfig` | Weights and schedule. `LateralOptConfig.parse("k=v,k=v")`. Unknown keys raise. |
| `dt` | float | World step in seconds: **0.1**. The comfort operators and the recomputed velocities assume it. |
| `neighbor_pred` | `"log"` \| `"cv"` \| `"none"` | Where other agents' next `n` steps come from. **Use `"cv"`.** `"log"` reads the recorded future (see [Pitfall 1](#1-neighbor_predlog-is-an-oracle)). |
| `world_p` | `[B, W, 5]` float32 | The plan: `(x, y, yaw, vx, vy)` in world coordinates. Index 0 = current state. This is exactly `postprocess_predictions(...)["trajectory_world_world_dt"]`, i.e. ES's `trajectory_world_bt5`. |
| `x0` | `[B, n]`, or `[B, 2n]` when `cfg.lon=1` | Initial edit, laid out as `[d_1..d_n, s_1..s_n]`. Zeros = start from no edit. |
| `state` | waymax `SimulatorState`, batch `B` | At the current step. Read: `roadgraph_points`, `log_trajectory` (ego extents, z; neighbours under `"log"`), `sim_trajectory` (neighbours under `"cv"`), `object_metadata.is_sdc`. |
| `t0` | scalar int32 | The current absolute world step. **Must be a scalar** under `"cv"`. |
| `active_b` | `[B]` float32 | 1 = this row feels the barrier. A 0 row feels only the comfort pull, so from zero it stays exactly at zero. |
| `n` | Python int (static) | Number of steps to optimize: `min(20, horizon - 1 - t0, W - 1)`. Each distinct `n` compiles once. |

It returns `x` (the best edit), `loss_best [B]` and `loss_init [B]`. You build
the executed plan with `edit_world(world_p, x, n=n, dt=0.1, lon=cfg.lon)`.

**The shipped configuration** (`o_am`): `LateralOptConfig.parse("lon=1,w_lon_prox=0.05,lon_back_m=1000")`.
Everything else stays at its default:

* `iters=32`, `step_m=0.25`, `decay=0.92`, `damping=0.5`
* `w_acc=w_jerk=0.02` (lateral), `w_lon_acc=w_lon_jerk=0.02`
* `barrier_shape=hinge2`, `edge_geometry=segment`
* the margins inherited from `SafetyLossConfig`: 0.3 m edge, 0.75 m agent

The other arguments are `n=20`, `neighbor_pred="cv"`, and the starts `init0` plus `initprev`.

## Recipe — repair the plan your ES picks

This adds one line after ES selection: the ES chooses *which* manoeuvre, and the
optimizer makes it clear the margins. It is `o_am` applied to the ES winner
instead of to a single policy sample.

```python
import jax.numpy as jnp
import numpy as np

from rl.diffusion_ft.lateral_opt import (LateralOptConfig, edit_world,
                                         make_lateral_optimizer, project_onto_path)
from rl.diffusion_ft.safety_objective import SafetyObjectiveConfig

WINDOW, DT = 20, 0.1
CFG = LateralOptConfig.parse("lon=1,w_lon_prox=0.05,lon_back_m=1000")
OPT = make_lateral_optimizer(SafetyObjectiveConfig(), CFG, dt=DT,
                             neighbor_pred="cv", world_dt_method="median")


def prev_tail(prev_world, k_exec, W):
    """The plan executed at the previous replan, from the current state on,
    padded to W with its last pose. k_exec = steps executed since (replan interval)."""
    tail = prev_world[:, k_exec:]
    return jnp.concatenate([tail, jnp.repeat(tail[:, -1:], W - tail.shape[1], axis=1)], axis=1)


def repair(world, state, t0, world_prev=None):
    """world [B, W, 5] (ES's trajectory_world_bt5) -> the plan to execute [B, W, 5]."""
    B, W, _ = world.shape
    horizon = int(state.log_trajectory.x.shape[-1])
    n = int(min(WINDOW, horizon - 1 - t0, W - 1))
    if n <= 0:
        return world
    t0j, act = jnp.asarray(t0, jnp.int32), jnp.ones((B,), jnp.float32)
    starts = [np.zeros((B, 2 * n), np.float32)]                     # init0
    if world_prev is not None:
        starts.append(project_onto_path(world, world_prev, n))      # initprev
    best_x, best_l = None, np.full(B, np.inf)
    for x0 in starts:
        x, loss, _ = OPT(world, jnp.asarray(x0), state, t0j, act, n)
        x, loss = np.asarray(x), np.asarray(loss)
        take = loss < best_l                                        # ties keep the earlier start
        best_x = x if best_x is None else np.where(take[:, None], x, best_x)
        best_l = np.where(take, loss, best_l)
    moved = np.any(best_x != 0.0, axis=1)                           # zero loss -> x == 0 exactly
    edited = edit_world(world, jnp.asarray(best_x), n=n, dt=DT, lon=1)
    return jnp.where(jnp.asarray(moved)[:, None, None], edited, world)
```

To wire it into `es_baseline/planner/diffusion_es_planner.py`, call `repair` on
`best_world_bt15[:, 0]` before building the `PlannerResult`. The simulator
executes `trajectory_world_bt5` (`simulation/runner.py`, `planning_utils.py`),
so that is the only field you need to replace. Keep the previous replan's
executed plan on the planner object to feed `world_prev`.

This sketch is `ProposalSelectEval._argmin` ([proposal_select.py](proposal_select.py))
with the evaluation bookkeeping removed. That function is the tested version.

## Pitfalls

### 1. `neighbor_pred="log"` is an oracle
Under log replay, the recorded future is exactly where every other car will be:
2 s of privileged information that no deployed planner has. Quote only `"cv"`
numbers as deployable. The `o_am` results above are all `"cv"`.

### 2. The ES preprocess still has the first-pair world-dt bug
`es_baseline/data/preprocess.py::compute_world_dt_seconds` takes the dt of the
**first** timestamp pair. In ~25% of the failure scenes that pair is ~0.075 s,
so the plan executes at ~0.75× speed and the car never reaches the goal.

The fix is the `"median"` method in this repo's
[data/preprocess.py](../../data/preprocess.py) (around line 55). It took those
scenes from 0.01 to 0.95 solved with nothing else changed.

Port it to the ES before comparing numbers. It also matters to the repair,
which assumes 0.1 s per world step.

### 3. Module-name collision with `es_baseline/data`
`es_baseline` is vendored with top-level imports (`from data...`), and it puts
itself first on `sys.path`. To import `rl.diffusion_ft.*` from there, also
append the repo root. `rl.*` then resolves to this repo, but `data.*` still
resolves to the ES copy.

Its `compute_world_dt_seconds` has no `method` argument, so
`make_lateral_optimizer(..., neighbor_pred="cv")` raises a `TypeError` the
first time it traces.

Fix it by porting the median `compute_world_dt_seconds` into the ES tree, which
you want anyway (Pitfall 2). Alternatively, copy the four files above into your
tree and change that one import.

### 4. The simulator does not enforce comfort
The env executes through `StateDynamics`, which teleports the ego to whatever
states it is handed. Nothing downstream rejects an abrupt edit, so the comfort
terms are the only thing keeping it drivable.

Do not set both `w_acc` and `w_jerk` to 0. `H` is the preconditioner, and
`parse` refuses that combination.

### 5. Do not finite-difference raw diffusion output
Diffusion output carries ~4 mm/step of position noise, which turns into
~30 m/s³ of fake jerk. Evaluate ride quality with a Savitzky–Golay filter.
`edit_world` itself recomputes velocities by finite differences of the *edited*
positions, which is fine at 0.1 s.

### 6. The scored window is longer than what executes
The window is 20 steps (2 s), but the replan interval executes only 10. That is
intentional: the second second gives the barrier look-ahead.

`n` is a static argument, so near the end of a scenario every shorter `n`
compiles once.

### 7. Cost
Each call is one jitted program (32 iterations of a safety-loss gradient). In
the evaluation, the optimizer took 0.13–0.5 s per call on an a5000/a6000.
`o_am` has no trigger: it runs 2 starts on every alive row at every replan.

The waymax_rs env only fits on a6000/a5000 nodes. A v100 dies at cuDNN
autotune, and a 2080 Ti runs out of memory.

## Tests and how to reproduce

The CPU unit tests are [tests/test_lateral_opt.py](tests/test_lateral_opt.py).
They have no job script of their own: they run as the first step of
`slurm/gate_lateral_opt.sbatch` and inside `slurm/test_proposal_select.sbatch`.
They check, among other things:

* `H` is the Hessian of the stated comfort quadratic, and is positive definite;
* the accel-only optimum is the cubic smoothstep, and the jerk-only optimum is
  the quintic smootherstep;
* `project_onto_path` inverts `edit_world`.

The real-scene gate also checks that a row with zero loss comes back with an
edit of exactly 0, that the anchor pose is kept, and that the loss never ends
above its starting value.

| what | entry point |
|---|---|
| Real-scene gate (expert windows distorted into violation, then repaired) | [tests/gate_lateral_opt.py](tests/gate_lateral_opt.py), [slurm/gate_lateral_opt.sbatch](slurm/gate_lateral_opt.sbatch) |
| Full `o_am` evaluation vs base | `bash rl/diffusion_ft/slurm/submit_lateral_opt_argmin.sh` (smoke: `DFT_LOPT_SMOKE=1`). It also runs older rule-based comparison arms (`o_lpA3`, `o_amc`); only `o_am` is described here |
| Before/after figures | [tools/plot_fix_pairs.py](tools/plot_fix_pairs.py) via `slurm/submit_plot_fix_pairs.sh` |

In `proposal_select`, the evaluated arms are configured as:

* `--select_banks "am="`
* `--select_bank_opts "am=init0,initprev"`
* `--select_bank_opt_cfgs "am=lon=1,w_lon_prox=0.05,lon_back_m=1000"`
* `--select_bank_neighbors "am=cv"`
* arm `o_am=bank@am:argmin:20`
