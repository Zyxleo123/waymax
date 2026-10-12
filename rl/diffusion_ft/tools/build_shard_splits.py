#!/usr/bin/env python3
"""A one-split splits JSON that names EVERY scene of a shard glob, unstratified.

The held-out splits (``build_eval_splits.py`` + ``add_intermittent_split``) are
labelled strata of the training corpus, reweighted to it. A plain slice of
another corpus (e.g. the first 10 WOMD validation shards) has no labels to
stratify by: every scene is scored, so the solved rate IS the corpus rate. This
writes that as one stratum of weight 1, which ``proposal_select`` and
``select_analysis`` read like any other splits file (``marginal2`` is then the
plain rate, with a scene-bootstrap CI).

Scene ids are positions in the glob's sorted, concatenated record order -- the
same convention as the scan -- so the byte-offset indexes must exist first
(``build_offsets.py``).

    python -m rl.diffusion_ft.tools.build_shard_splits \\
        --shard_glob '.../validation_tfexample.tfrecord-0000[0-9]-of-00150' \\
        --index_dir .../womd_offsets --split val10 --out val10_splits.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl.diffusion_ft.scene_sampler import ShardIndex


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard_glob", required=True)
    p.add_argument("--index_dir", required=True)
    p.add_argument("--split", required=True, help="the split's name, e.g. val10")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    idx = ShardIndex(a.shard_glob, a.index_dir)
    n = len(idx)
    spec = {
        "shard_glob": a.shard_glob, "index_dir": a.index_dir,
        "shards": [Path(s).name for s in idx.paths],
        "records_per_shard": [int(c) for c in idx.counts],
        "counts": {a.split: n},
        "splits": {a.split: {"criteria": f"every record of {len(idx.paths)} shards, "
                                         f"unlabelled and unstratified",
                             "scene_ids": list(range(n))}},
        # One stratum of weight 1: the analysis' population-weighted rate is the
        # plain mean over the scenes.
        "population": {"strata": {a.split: {"n": float(n), "weight": 1.0}},
                       "note": "unstratified shard slice; weight 1 = plain corpus rate"},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(spec, indent=1))
    print(f"{a.split}: {n} scenes from {len(idx.paths)} shards "
          f"({', '.join(map(str, idx.counts))}) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
