#!/usr/bin/env python3
"""Does the ES runner read the same scenes from a compact copy as from the originals?

Loads target scenes through the ES tree's ``data.scenario_loader.load_scenario_state_batch_fast``
(what ``simulation/runner.py`` calls) from the compact shard and from the source shard, and
requires bit-identical simulator states and equal ``scenario/id``s. It also checks that a
placeholder index fails to load instead of yielding a scene. CPU only.

    python rl/diffusion_ft/tests/check_compact_shards.py --compact_root .../valfail1568 \\
        --src_dir /data/user_data/yixiz/autonlab/zfsauton/datasets/womd/tf_example/validation \\
        --es_root /home/yixiz/waymax_rs_es_newmodel --n_shards 4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--compact_root", required=True)
    p.add_argument("--src_dir", required=True)
    p.add_argument("--es_root", required=True)
    p.add_argument("--n_shards", type=int, default=4)
    args = p.parse_args()
    sys.path.insert(0, args.es_root)

    import jax
    import numpy as np
    from waymax import config as waymax_config

    from data.scenario_loader import load_scenario_state_batch_fast

    root = Path(args.compact_root)
    manifest = json.load(open(root / "manifest.json"))
    shards = manifest["shards"]
    picks = [shards[round(i * (len(shards) - 1) / max(args.n_shards - 1, 1))] for i in range(args.n_shards)]
    bad = 0
    for rec in picks:
        idx = sorted({rec["targets"][0], rec["targets"][len(rec["targets"]) // 2], rec["targets"][-1]})

        def load(path: Path):
            cfg = dataclasses.replace(waymax_config.WOD_1_3_1_TRAINING, path=str(path), max_num_objects=128,
                                      batch_dims=(len(idx),), shuffle_seed=0)
            return load_scenario_state_batch_fast(cfg, idx, return_scenario_ids=True)

        s_c, map_c, ids_c = load(root / manifest["split"] / rec["basename"])
        s_o, map_o, ids_o = load(Path(args.src_dir) / rec["basename"])
        leaves_c, leaves_o = jax.tree_util.tree_leaves(s_c), jax.tree_util.tree_leaves(s_o)
        same = (map_c == map_o and ids_c == ids_o and len(leaves_c) == len(leaves_o)
                and all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(leaves_c, leaves_o)))
        print(f"{rec['basename']} scenes {idx}: {'identical' if same else 'DIFFERENT'} ({len(leaves_c)} leaves, ids {ids_c})")
        bad += not same

    rec = picks[0]
    placeholder = next(i for i in range(rec["n_records"]) if i not in set(rec["targets"]))
    cfg = dataclasses.replace(waymax_config.WOD_1_3_1_TRAINING, max_num_objects=128, batch_dims=(1,),
                              path=str(root / manifest["split"] / rec["basename"]), shuffle_seed=0)
    try:
        load_scenario_state_batch_fast(cfg, [placeholder])
        print(f"placeholder {rec['basename']}:{placeholder} LOADED as a scene (expected an error)")
        bad += 1
    except Exception as e:
        print(f"placeholder {rec['basename']}:{placeholder} refused as expected ({type(e).__name__})")
    print("PASS" if bad == 0 else f"FAIL ({bad})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
