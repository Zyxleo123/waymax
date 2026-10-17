# `es_discriminator`: learned reward for Diffusion-ES

A scene-conditioned PyTorch discriminator that scores Diffusion-ES candidate
trajectories. It is trained only on real ES candidates collected in closed loop
(below) and can replace the handcrafted ES reward at inference
(`simulation/run_simulation.py --discriminator_ckpt`). Diffusion-ES and the world
model stay frozen.

## Real ES candidates from closed loop (WM + Diffusion-ES, fail1500)

`simulation/run_simulation.py --save_es_candidates` attaches
`es_candidate_recorder.ESCandidateRecorder` to the Diffusion-ES planner. At every
replan it stores all evaluated candidates (initial population and every mutation
round, including rejected ones) with candidate ID, stage, parent elite, elite
membership, the selected flag, and every scorer factor (total, collision, offroad,
and the WM's per-candidate max agent collision probability). Context, logged
futures and WM forecasts are stored once per replan. Without the flag, nothing
changes.

```bash
sbatch slurm/sim/collect_es_candidates_fail1500.sh --smoke   # 10 scenes, separate folder
sbatch slurm/sim/collect_es_candidates_fail1500.sh           # all 1,500, then labels; resubmit to resume
```

`label_es_candidates.py` adds GT-replay labels offline (`<batch>.labels.npz`):
the first collision and offroad step per candidate, plus progress, goal and
ADE/FDE metrics. The array reference is in the `es_candidate_recorder.py` docstring.

## More data: whole WOMD training shards

`slurm/sim/collect_es_candidates_womd_train.sh` runs the same planner on training shards
00002-00011 (one array task per shard). It skips scenes already in fail1500, using
`make_collection_targets.py`. Each task also writes all three label files and the WM and
road-edge caches. Shards 00000-00001 are left out because they are the trainval benchmark.
`build_training_set.py` then symlinks fail1500 and the finished shards into one training set.
The trainval batch is dropped and the invalid-at-start list is merged. Scenes keep their
hash-based train/val/test split.

```bash
sbatch --array=2 slurm/sim/collect_es_candidates_womd_train.sh --smoke   # 5 scenes
sbatch slurm/sim/collect_es_candidates_womd_train.sh                     # shards 2-11, 2 GPUs at a time
python -m es_discriminator.build_training_set --name fail1500_plus_womd_train_s02_11
sbatch slurm/train/train_wm_discriminator.sh --data fail1500_plus_womd_train_s02_11
```

Context keys and shapes (B=batch, A=16 agents, H=5 past steps, M=256 map points,
L=16 traffic lights):

| Key | Shape | Encoding |
|---|---|---|
| `agents` | `[B,A,H,8]` | ego-relative x,y / 100; vx,vy / 25; sin/cos relative yaw; length,width / 10 |
| `agent_mask` | `[B,A,H]` | history validity |
| `map` | `[B,M,5]` | x,y / 100; rotated direction x,y; type / 32. Every 6th road point, nearest 256 (median reach about 35 m) |
| `map_mask` | `[B,M]` | selected roadgraph points |
| `lights` | `[B,L,4]` | ego-relative x,y / 100; state / 16; validity 1 |
| `lights_mask` | `[B,L]` | selected traffic lights |
| `goal` | `[B,2]` | ego-relative goal (last valid logged ego position) / 100 |
| `ego_vel` | `[B,2]` | rotated ego vx,vy / 25 |

## Tests

```bash
python -m unittest discover -s es_discriminator/tests -t . -v
```
