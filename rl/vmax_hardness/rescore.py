"""Stage A: re-score raw WOMD with the pretrained V-Max SAC, with provenance.

The hardness risk model was fit on *diffusion*-policy failures; reusing it to
reweight a V-Max SAC fine-tune would upweight scenes hard for the wrong policy.
This rolls the pretrained SAC (``model_best`` staged as ``model_final`` so
``get_model_path`` selects it) over a fixed slice of the WOMD *training* shards
and writes one ``<shard>.scenario_<idx>.json`` per scene -- the exact format
:mod:`train_diffusion.hardness.labels` scans -- so the risk model can be refit on
what THIS policy actually fails.

Unlike ``rl.eval_vmax_failures_lightweight`` (whose ``scenario_index`` is a
shuffled stream position with no shard provenance), scenes here are addressed by
``(shard, record)`` through the byte-offset reader, so every label keys straight
into the risk table.  Resumable: an existing JSON is skipped.

    JAX_PLATFORMS=cpu python -m rl.vmax_hardness.rescore \
        --run-dir /path/to/staged_best_run \
        --shard-glob '/.../training_tfexample.tfrecord-*-of-01000' \
        --index-dir /data/user_data/yixiz/autonlab/zfsauton/scratch/yixiz/waymax_rs/womd_offsets \
        --n-shards 200 --per-shard 40 --out-dir /.../rescore_labels
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_VMAX = _REPO / "V-Max"
for p in (_REPO, _VMAX):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import jax  # noqa: E402

from rl.diffusion_ft.scene_sampler import DEFAULT_INDEX_DIR, ShardIndex  # noqa: E402
from rl.vmax_hardness.hardness import _womd_cfg, load_stacked_states  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True,
                   help="Staged run dir: .hydra/config.yaml + model/model_final.pkl (=best).")
    p.add_argument("--shard-glob", required=True)
    p.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-shards", type=int, default=200, help="How many shards to cover.")
    p.add_argument("--per-shard", type=int, default=40, help="Records per shard (from 0).")
    p.add_argument("--shard-start", type=int, default=0, help="First shard id (chunking).")
    p.add_argument("--max-num-objects", type=int, default=64)
    p.add_argument("--num-paths", type=int, default=45)
    p.add_argument("--num-points-per-path", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # vmax.scripts.evaluate.utils imports the visualization package, which imports
    # seaborn at module load (only used inside plotting fns we never call here). The
    # waymax_rs env has no seaborn, so stub it -- cheaper than adding a viz dep to
    # someone else's conda env.
    import types
    sys.modules.setdefault("seaborn", types.ModuleType("seaborn"))

    from vmax.scripts.evaluate import utils  # noqa: E402
    from rl.eval_vmax_failures_lightweight import _rollout_metrics  # noqa: E402

    run_dir = Path(args.run_dir).resolve()
    src_dir, path_model = str(run_dir.parent), run_dir.name
    print(f"[rescore] devices={jax.devices()} model_dir={run_dir}", flush=True)
    env, step_fn, _eval_path, termination_keys = utils.setup_evaluation(
        "ai", path_model, src_dir, args.shard_glob,
        str(out_dir / "ai_setup"), args.max_num_objects, False, True,
    )
    jitted_step = jax.jit(step_fn)
    jitted_reset = jax.jit(env.reset)

    index = ShardIndex(args.shard_glob, index_dir=args.index_dir)
    # Prefix (1,) with objects sliced to max_num_objects, matching the env built by
    # setup_evaluation and the lightweight eval's per-scenario reset.
    cfg = _womd_cfg(
        batch_dims=(1,), num_paths=args.num_paths,
        num_points_per_path=args.num_points_per_path,
        max_num_objects=args.max_num_objects,
    )

    shard_ids = range(args.shard_start, min(args.shard_start + args.n_shards, len(index.paths)))
    rng = jax.random.PRNGKey(args.seed)
    t0 = time.time()
    n_done = n_new = 0
    accs: list[float] = []

    for shard_id in shard_ids:
        shard_path = index.paths[shard_id]
        base = os.path.basename(shard_path)
        n_rec = min(args.per_shard, int(index.counts[shard_id]))
        for rec in range(n_rec):
            out_path = out_dir / f"{base}.scenario_{rec}.json"
            n_done += 1
            if out_path.exists():
                continue
            records = [index.read((shard_id, rec))]
            scenario = load_stacked_states(records, cfg)  # prefix (1,)
            rng, scenario_key = jax.random.split(rng)
            metrics, steps = _rollout_metrics(
                env, jitted_step, jitted_reset, scenario, scenario_key, termination_keys
            )
            success = bool(metrics.get("accuracy", 0.0) >= 0.5)
            row = {
                "tfrecord": shard_path,
                "scenario_idx": int(rec),
                "success": success,
                "reached_goal": float(metrics.get("reached_goal", float("nan"))),
                "offroad": float(metrics.get("offroad", float("nan"))),
                "overlap": float(metrics.get("overlap", float("nan"))),
                "run_red_light": float(metrics.get("run_red_light", float("nan"))),
                "episode_length": int(steps),
            }
            out_path.write_text(json.dumps(row))
            accs.append(float(success))
            n_new += 1
            if n_new % 25 == 0:
                print(f"[rescore] shard {shard_id} rec {rec} | new={n_new} seen={n_done} "
                      f"success_rate={np.mean(accs):.3f} elapsed={time.time()-t0:.1f}s", flush=True)
        jax.clear_caches()

    print(f"[rescore] done: {n_new} new, {n_done} covered, "
          f"success_rate(new)={np.mean(accs) if accs else float('nan'):.3f}", flush=True)


if __name__ == "__main__":
    main()
