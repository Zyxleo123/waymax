# World Model Extension - Change Summary

This package is the original baseline code plus a V1 learned latent multi-agent
world model integrated into the ES scorers.

## New forecast mode

`simulation/run_simulation.py` now accepts:

```text
--other_agent_forecast cv
--other_agent_forecast wm
--other_agent_forecast gt
```

`wm` requires `--world_model_ckpt`.

## Scientific design

- predicts surrounding-agent physical futures rather than learned metric scores
- encodes agent history, map polylines and current traffic lights
- creates an interaction-aware latent scene
- predicts a future latent sequence
- decodes future x/y/yaw
- preserves the existing hand-written collision/offroad/goal scoring functions
- deterministic V1 for a clean CV -> WM -> GT ablation

## Training/evaluation

- `train_world_model.py`: supervised training on existing cached Waymax NPZ shards
- `evaluate_world_model.py`: offline future-prediction comparison against constant velocity
- `WORLD_MODEL_INSTRUCTIONS.md`: end-to-end commands
- `slurm_train_world_model.sh`: cluster training template
- existing FM-ES baseline/failure-subset SLURM scripts now accept `wm`
