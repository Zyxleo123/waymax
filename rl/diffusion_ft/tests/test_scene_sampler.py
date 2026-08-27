"""Gate the full-corpus scene sampler before spending GPU time on it.

Three things have to be true before a DPPO run can trust this data path:

1. **Equivalence.** A batch assembled from bytes fetched by byte offset must be
   the same simulator state as one produced by the existing scan-based loader
   (:func:`data.scenario_loader.load_scenario_state_batch_fast`) for the same
   scenes. Anything else means the run is training on subtly different data than
   every gate and diagnostic before it.
2. **Constant shape.** Every draw must have the same object count, or the trainer
   rebuilds the env AND the actor mid-run and silently discards the fine-tuned
   weights and the Adam state.
3. **Held-out really held out.** No training draw may contain an eval scene, or
   the eval curve is measuring memorization.

Also reports the per-batch load wall-clock and the valid-route fraction at the
raw-WOMD path dims (45x800), the two numbers that decide whether the run is
viable: a slow loader eats the iteration, and a zero route count means the
progression reward is identically 0.

CPU only, no model, no GPU. Submit it -- do not run it on a login node.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

WOMD_GLOB = ("/zfsauton/scratch/eshau/womd/tf_example/training/"
             "training_tfexample.tfrecord-*-of-01000")
SHARD0 = ("/zfsauton/scratch/eshau/womd/tf_example/training/"
          "training_tfexample.tfrecord-00000-of-01000")


def _leaves(state):
    import jax

    return {"/".join(str(k) for k in path): np.asarray(v)
            for path, v in jax.tree_util.tree_flatten_with_path(state)[0]}


def test_equivalence(index_dir: str) -> bool:
    """Byte-offset assembly == the scan-based loader, leaf for leaf."""
    from data.scenario_loader import load_scenario_state_batch_fast
    from rl.diffusion_ft.scene_sampler import ShardIndex, _state_from_bytes, build_ds_cfg

    idx = list(range(6))
    ref_cfg = build_ds_cfg(batch_dims=(len(idx),), include_sdc_paths=True,
                           num_paths=45, num_points_per_path=800, path=SHARD0)
    ref, idx_map = load_scenario_state_batch_fast(ref_cfg, idx)

    # The scan loader returns records in scan (= sorted index) order; asking for
    # 0..5 makes that the identity, so the two batches are directly comparable and
    # a mismatch cannot be blamed on ordering.
    assert all(idx_map[i] == i for i in idx), f"unexpected scan order {idx_map}"

    shards = ShardIndex(SHARD0, index_dir=index_dir)
    records = [shards.read((0, i)) for i in idx]
    got = _state_from_bytes(records, ref_cfg)

    a, b = _leaves(ref), _leaves(got)
    if set(a) != set(b):
        print(f"  FAIL: leaf sets differ ({len(a)} vs {len(b)})")
        return False
    bad = []
    for k in sorted(a):
        x, y = a[k], b[k]
        if x.shape != y.shape:
            bad.append(f"{k}: shape {x.shape} vs {y.shape}")
            continue
        if not np.array_equal(np.nan_to_num(x), np.nan_to_num(y)):
            d = np.abs(np.nan_to_num(x) - np.nan_to_num(y)).max()
            bad.append(f"{k}: max|diff|={d}")
    if bad:
        print(f"  FAIL: {len(bad)} leaves differ; first 5:")
        for line in bad[:5]:
            print(f"    {line}")
        return False
    print(f"  PASS: all {len(a)} leaves identical to the scan loader "
          f"(6 scenes, sdc_paths on)")
    return True


def test_sampler(index_dir: str, n_batches: int = 4, batch_size: int = 32) -> bool:
    """Shape stability, train/eval disjointness, load cost, route validity."""
    from rl.diffusion_ft.scene_sampler import SceneSampler

    t0 = time.time()
    s = SceneSampler(WOMD_GLOB, batch_size=batch_size, eval_size=batch_size,
                     include_sdc_paths=True, num_paths=45, num_points_per_path=800,
                     seed=0, prefetch=False, index_dir=index_dir)
    print(f"  index: {s.describe()}  ({time.time() - t0:.1f}s to open)")

    ev = s.eval_batch()
    ev_valid = np.asarray(ev.sdc_paths.valid).reshape(batch_size, -1).any(axis=1)
    print(f"  eval batch: route {int(ev_valid.sum())}/{batch_size} valid")

    ok = True
    n_obj = None
    seen: set[int] = set()
    times = []
    route_valid = 0
    for i in range(n_batches):
        t = time.time()
        st = s.next_train()
        times.append(time.time() - t)
        ids = set(int(x) for x in s.last_ids.tolist())
        if ids & set(int(x) for x in s.eval_ids.tolist()):
            print(f"  FAIL: batch {i} overlaps the held-out eval set")
            ok = False
        seen |= ids
        o = int(np.asarray(st.log_trajectory.x).shape[1])
        if n_obj is None:
            n_obj = o
        elif o != n_obj:
            print(f"  FAIL: object count changed {n_obj} -> {o} at batch {i}")
            ok = False
        v = np.asarray(st.sdc_paths.valid).reshape(batch_size, -1).any(axis=1)
        route_valid += int(v.sum())

    total = n_batches * batch_size
    print(f"  {n_batches} batches x {batch_size}: {len(seen)}/{total} distinct scenes, "
          f"objects={n_obj}, route {route_valid}/{total} valid "
          f"({100.0 * route_valid / total:.0f}%)")
    print(f"  load: {np.mean(times):.2f}s/batch (min {min(times):.2f}, max {max(times):.2f})")
    if route_valid == 0:
        print("  FAIL: no scene has a valid sdc_path -- the 45x800 path dims are wrong")
        ok = False
    print(f"  {'PASS' if ok else 'FAIL'}: sampler invariants")
    return ok


def test_prefetch(index_dir: str, batch_size: int = 32) -> bool:
    """The background loader hands back usable batches and hides its cost."""
    from rl.diffusion_ft.scene_sampler import SceneSampler

    s = SceneSampler(WOMD_GLOB, batch_size=batch_size, eval_size=0,
                     include_sdc_paths=False, seed=1, prefetch=True,
                     index_dir=index_dir)
    s.next_train()          # first pop may wait for the very first load
    time.sleep(5.0)         # stand in for an iteration of GPU work
    t = time.time()
    s.next_train()
    dt = time.time() - t
    print(f"  prefetched pop after a 5s 'iteration': {dt:.2f}s")
    print(f"  {'PASS' if dt < 1.0 else 'WARN'}: prefetch "
          f"{'hides' if dt < 1.0 else 'does NOT hide'} the load")
    return True


def main() -> int:
    index_dir = (sys.argv[1] if len(sys.argv) > 1
                 else "/zfsauton/scratch/yixiz/waymax_rs/womd_offsets")
    print(f"index_dir={index_dir}")
    results = {}
    print("\n[1/3] byte-offset assembly == scan loader")
    results["equivalence"] = test_equivalence(index_dir)
    print("\n[2/3] sampler invariants")
    results["sampler"] = test_sampler(index_dir)
    print("\n[3/3] background prefetch")
    results["prefetch"] = test_prefetch(index_dir)
    print("\n" + "=" * 60)
    for k, v in results.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    ok = all(results.values())
    print(f"{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
