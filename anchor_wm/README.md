# `anchor_wm`: joint world model + ego-plan scorer (1-day hypothesis test)

**Hypothesis.** A world model and a plan scorer trained *jointly* rank ego plans better than a
separately trained WM + discriminator. Here jointly means a shared encoder, with the scorer
reading the WM's anchor futures and its loss reaching the WM. The model should do this while
staying comparable to MF-Gaussian as a forecaster.

The package is self-contained. Training needs neither waymax nor the MF package. Evaluation
imports `es_discriminator` (bank loading, replan-state rebuild, `evaluate.full_report`) and the
MF network, read-only.

## Architecture

```
history, map, lights ──► MF-Gaussian scene encoder (warm-started, same module names)
                                   │ agent tokens            │ scene tokens
k-means anchors ──► embed ─┐       ▼                         │
goal, ego velocity ────────┴► query (agent, anchor) ──► MLP ─┤► per-step residual around the anchor,
                                                             │  sigma, yaw, logit
                                                             ▼
                  planner modes: top-6 per agent with endpoint NMS ([N,6,T], MF-compatible)
                                                             │
ego candidates [C,T] ──► scorer: GRU + cross-attention to scene tokens + time-aligned
                         geometric attention to (agent, mode) tokens + road-edge tokens
                         ──► total score (ranking) and sub-scores (overlap, offroad, progress, comfort)
```

| Part | File | What it does |
|---|---|---|
| Data | `data.py` | Same inputs and targets as MF's cache loader (map / light builders copied). Adds the SDC target, goal, logged boxes of all objects, road-edge points (with Waymax's prior-point info) and scorer edge tokens. `payload_from_sim_state` gives the same preprocessing for recorded ES replans |
| Anchors | `anchors.py` | k-means per group (vehicle incl. SDC / pedestrian / cyclist) on 5 waypoints of agent-frame futures. Anchor = cluster mean of the full 25 steps. Stationary futures are down-weighted. Reports oracle best-anchor ADE/FDE/miss rate |
| Decoder | `model.py` | Query = agent token + embed(anchor) (+ goal for the SDC). MF's `direct` head predicts residuals in the agent frame, so the initial prediction is the anchor. The classifier sees query + predicted waypoints |
| Scorer | `model.py` (`PlanScorer`) | Port of `WMTrajectoryDiscriminator`. WM tokens = per-step pose / velocity / size / log sigma features + **the decoder's query embedding** + log p, over the 16 most relevant agents × top-8 modes (wm_cache's selection rule) |
| Joint switch | `model.grad_scale` | Every WM output the scorer reads (scene tokens, queries, means, sigma, yaw, probabilities) passes through a gradient scale. 0 = detached (A1), > 0 = joint (A2). The scorer's own weights always train |
| Candidates | `candidates.py` | Per scene: the K decoded ego anchors, the logged future, 16 smooth perturbations of it (lateral ramps, speed scaling, stops, nudge toward the nearest agent, combinations). 48 scored per step (vocabulary dropout, logged future always included) |
| Labels | `candidates.py`, `geometry.py` | Log replay, online on GPU: SAT box overlap + signed clearance, Waymax-rule road-edge signed distance on box corners (no height), progress along the logged path, goal gain, smoothed comfort |
| Losses | `losses.py` | WM: nearest-*anchor* assignment (MTR), Gaussian NLL + yaw on that query, distance-softened anchor classification, MF's near-ego weighting. Scorer: soft-target BCE sub-scores, B0's pairwise preferences (safe > unsafe; among safe, ≥ 2 m more goal gain), listwise softmax to a soft reward |

Scenes whose logged future is incomplete or labelled unsafe are dropped from the scorer loss
(es_discriminator's expert filter).

## 1-day experiment

| Arm | How | Isolates |
|---|---|---|
| B0 | MF-Gaussian `20261005_092950/epoch_0450` + its safety_progress discriminator | reference |
| A1 separate | `ARM=separate`: same model, scorer inputs detached | anchors + self-generated data |
| A2 joint | `ARM=joint`: scorer gradient into the WM (ramped in after 3k steps) | **the hypothesis (A2 vs A1)** |

```bash
sbatch anchor_wm/slurm/prep.sh                               # tests, anchors, label agreement
PROFILE=1 sbatch --time=0-00:45:00 anchor_wm/slurm/train.sh  # s/step + memory: wm_only / separate / joint
ARM=separate sbatch anchor_wm/slurm/train.sh                 # ~11.5 h cap each, run in parallel
ARM=joint sbatch anchor_wm/slurm/train.sh
CKPTS="<a1>/final.pt <a2>/final.pt" sbatch anchor_wm/slurm/eval.sh
```

Readouts (`eval.sh`):
- `eval_bank.py`: B0's held-out **test groups** of the fail1500 bank. Replan states are rebuilt
  as `wm_cache.py` does, and all 320 recorded candidates are scored. `evaluate.full_report` runs
  for AnchorWM and for B0's saved scores. Output: `comparison.md` (picked unsafe / overlap /
  offroad / goal gain, pairwise accuracy, safety AUROC, over the filtered set and the ES-unsafe
  scenes). The frame check fails loudly if the rebuilt ego frame differs from the recorded group frame.
- `eval_wm.py`: WOMD validation shards (same split as MF's training), MF-Gaussian vs each
  checkpoint. Reports minADE/minFDE/miss@6, top-1, safety recall (interacting agents covered at
  closest approach), endpoint spread and top-1 jerk.

Decision rule:
- **Supported:** A2 beats A1 on bank picking (lower unsafe at equal goal gain, higher pairwise
  accuracy), and A2's minADE@6 is within about 5% of A1's.
- **Comparable to the current pipeline:** A2's minADE@6 and miss rate are on par with
  MF-Gaussian, and its bank picking is close to B0. B0 trained on these banks; A2 never saw a
  diffusion candidate.

## Tests

```bash
python -m unittest discover -s anchor_wm/tests -t . -v
```

The tests cover:
- geometry conventions (SAT gap; Waymax sign and prior-point rule)
- data shapes
- forward, losses and backward
- the A1/A2 gradient switch (zero encoder / decoder gradient from the scorer at scale 0)
- warm-start equality with a real MF network (identical latents)
- perturbations and k-means

## Known gaps

- Offroad labels ignore road height (Waymax uses z), and training labels use the 0.2 s grid.
  `label_check.py` quantifies the disagreement with `labels_waymax`.
- Pedestrians and cyclists use Cartesian anchors like vehicles (no kinematic model).
- Not in this version: joint scene futures, ego-conditioned reaction, closed-loop deployment.
