"""Open-loop safety labels for every candidate in a diffusion bank (log replay).

For each bank anchor, reloads the scene, puts the expert and the K diffusion
samples back in the world frame, and checks them against the *logged* futures
of all other agents with the repo's own scorer functions:

  collision  Scorer.compute_collision_score (SAT box overlap, other agents' GT futures)
  offroad    Scorer.compute_offroad_score_v2 (ego box corners vs road edges)

Labels are exact for log replay over the 5 s horizon, evaluated at the bank's
0.2 s steps (anchor + 25 points) rather than 0.1 s. Needs the JAX/Waymax env;
CPU is fine (no model is loaded).

  python -m es_discriminator.label_bank --bank diffusion_bank_val.npz \
      --tfrecord_glob '.../validation_tfexample.tfrecord-0000[01]-of-00150' --out bank_val_labels.npz

Output: collision [N,K+1], offroad [N,K+1] bool over the 25 future points
(True = unsafe; column 0 = expert), and anchor_collision [N], anchor_offroad [N]
for the shared anchor pose.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import os

import jax.numpy as jnp
import numpy as np
from waymax import config as waymax_config

from data.scenario_loader import load_scenario_state_batch_fast
from scores.helpers import forecast_other_agents_log_futures
from scores.scorer_lane_graph import Scorer

from es_discriminator.data import DataConfig

NW = 5


def _to_world(local_xyyaw: np.ndarray, origin: np.ndarray, heading: float) -> np.ndarray:
    c, s = np.cos(heading), np.sin(heading)
    x, y = local_xyyaw[..., 0], local_xyyaw[..., 1]
    wx = c * x - s * y + origin[0]
    wy = s * x + c * y + origin[1]
    return np.stack([wx, wy, local_xyyaw[..., 2] + heading], axis=-1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", required=True)
    p.add_argument("--tfrecord_glob", required=True, help="Same tfrecords the bank was collected from")
    p.add_argument("--max_batches_per_shard", type=int, default=20)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cfg = DataConfig()

    with np.load(a.bank) as z:
        experts, samples = z["expert_xyyaw"], z["samples_xyyaw"].astype(np.float32)
        shards, sids, anchors = z["shard"], z["scenario_id"], z["anchor_t"]
    n, k = samples.shape[:2]
    collision = np.zeros((n, k + 1), dtype=bool)
    offroad = np.zeros((n, k + 1), dtype=bool)
    anchor_collision = np.zeros(n, dtype=bool)
    anchor_offroad = np.zeros(n, dtype=bool)
    done = np.zeros(n, dtype=bool)
    scorer = Scorer(metrics=["collision", "offroad"], other_agent_forecast="gt")
    steps = cfg.future_stride * cfg.horizon + 1  # world steps incl. anchor

    for shard in sorted(glob.glob(a.tfrecord_glob)):
        name = os.path.basename(shard)
        if not np.any(shards == name):
            continue
        wcfg = dataclasses.replace(waymax_config.WOD_1_3_1_TRAINING, path=shard,
                                   max_num_objects=128, batch_dims=(NW,), shuffle_seed=0)
        for b in range(a.max_batches_per_shard):
            try:
                st, _, ids = load_scenario_state_batch_fast(wcfg, list(range(b * NW, b * NW + NW)),
                                                           return_scenario_ids=True)
            except Exception:
                break
            log = st.log_trajectory
            lx, ly, lyaw = np.asarray(log.x), np.asarray(log.y), np.asarray(log.yaw)
            llen, lwid = np.asarray(log.length), np.asarray(log.width)
            is_sdc = np.asarray(st.object_metadata.is_sdc)
            for w in range(NW):
                sid = ids[w].decode() if isinstance(ids[w], bytes) else str(ids[w])
                rows = np.flatnonzero((shards == name) & (sids == sid) & ~done)
                if len(rows) == 0:
                    continue
                e = int(np.argmax(is_sdc[w]))
                for r in rows:
                    t = int(anchors[r])
                    origin = np.array([lx[w, e, t], ly[w, e, t]], dtype=np.float64)
                    heading = float(lyaw[w, e, t])
                    local = np.concatenate([experts[r][None], samples[r]], axis=0)  # [K+1,25,3]
                    world = _to_world(local, origin, heading)
                    anchor_pose = np.broadcast_to(np.array([origin[0], origin[1], heading]), (k + 1, 1, 3))
                    world = np.concatenate([anchor_pose, world], axis=1)  # [K+1,26,3]
                    size = np.broadcast_to(np.array([llen[w, e, t], lwid[w, e, t]]), world.shape[:2] + (2,))
                    ego = jnp.asarray(np.concatenate([world, size], axis=-1), dtype=jnp.float32)
                    obj, mask, _, _ = forecast_other_agents_log_futures(st, t, steps, w)
                    obj, mask = obj[:, ::cfg.future_stride], mask[:, ::cfg.future_stride]
                    if obj.shape[1] != ego.shape[1]:
                        raise ValueError(f"Log future too short at t={t}: {obj.shape[1]} vs {ego.shape[1]}")
                    # Outcomes over the 25 future points only; the shared anchor pose is
                    # flagged separately since no candidate choice can change it.
                    fut, fobj, fmask = ego[:, 1:], obj[:, 1:], mask[:, 1:]
                    collision[r] = np.asarray(scorer.compute_collision_score(fut, fobj, fmask)) < 0.5
                    offroad[r] = np.asarray(scorer.compute_offroad_score_v2(fut, st, t, w)) < 0.5
                    now = ego[:1, :1]
                    anchor_collision[r] = bool(np.asarray(
                        scorer.compute_collision_score(now, obj[:, :1], mask[:, :1]))[0] < 0.5)
                    anchor_offroad[r] = bool(np.asarray(scorer.compute_offroad_score_v2(now, st, t, w))[0] < 0.5)
                    done[r] = True
            print(f"{name} batch {b}: labelled {int(done.sum())}/{n}", flush=True)

    if not done.all():
        raise RuntimeError(f"{int((~done).sum())} bank rows were not matched to a scene; "
                           "check --tfrecord_glob / --max_batches_per_shard")
    np.savez_compressed(a.out, collision=collision, offroad=offroad,
                        anchor_collision=anchor_collision, anchor_offroad=anchor_offroad)
    unsafe = collision | offroad
    print(f"Saved {a.out}")
    print(f"anchor pose already flagged: collision {anchor_collision.mean():.3f} offroad {anchor_offroad.mean():.3f}")
    clean = ~(anchor_collision | anchor_offroad)
    print(f"expert on clean anchors: collision {collision[clean, 0].mean():.3f} offroad {offroad[clean, 0].mean():.3f}")
    print(f"expert: collision {collision[:, 0].mean():.3f} offroad {offroad[:, 0].mean():.3f}")
    print(f"diffusion samples: collision {collision[:, 1:].mean():.3f} offroad {offroad[:, 1:].mean():.3f} "
          f"unsafe {unsafe[:, 1:].mean():.3f}")
    mixed = unsafe[:, 1:].any(1) & ~unsafe[:, 1:].all(1)
    print(f"anchors with both safe and unsafe samples: {int(mixed.sum())}/{n}")


if __name__ == "__main__":
    main()
