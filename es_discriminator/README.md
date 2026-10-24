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

`slurm/train/discriminator_data.sh` runs the whole data pipeline in stages
(collect, label, decisions, waymax, wm_cache; `--stage S` runs one, resubmitting resumes):

```bash
sbatch slurm/train/discriminator_data.sh --source fail1500 --smoke   # 10 scenes, separate folder
sbatch slurm/train/discriminator_data.sh --source fail1500           # all 1,500 + labels + caches
```

`label_es_candidates.py` adds GT-replay labels offline (`<batch>.labels.npz`):
the first collision and offroad step per candidate, plus progress, goal and
ADE/FDE metrics. The array reference is in the `es_candidate_recorder.py` docstring.

## More data: whole WOMD training shards

`--source womd` runs the same planner on training shards 00002-00011 (one array task per
shard). It skips scenes already in fail1500, using `make_collection_targets.py`. Shards
00000-00001 are left out because they are the trainval benchmark.
`build_training_set.py` then symlinks fail1500 and the finished shards into one training set.
The trainval batch is dropped and the invalid-at-start list is merged. Scenes keep their
hash-based train/val/test split.

```bash
sbatch --array=2 slurm/train/discriminator_data.sh --source womd --smoke     # 5 scenes
sbatch --array=2-11%2 --gres=gpu:a6000:1 --time=16:00:00 \
       slurm/train/discriminator_data.sh --source womd                       # shards 2-11, 2 at a time
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

## Architecture ablations

`analyse_discriminator/` holds the A0-A8 ablations (`ablations.py`, read by the trainer), their
training and comparison scripts, and the unit tests:

```bash
bash es_discriminator/analyse_discriminator/submit_ablations.sh A5 A6 A7 A8
sbatch es_discriminator/analyse_discriminator/compare_ablations.sh --baseline A4
python -m unittest es_discriminator.analyse_discriminator.test_ablations -v
```
