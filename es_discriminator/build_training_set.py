"""Combine collected ES candidate runs into one discriminator training set (symlinks only).

Every source is a folder with es_candidates/, wm_predictions/, road_edges/ (wm_cache output)
and either analysis/scenes.npz or an existing invalid_at_start.json. A batch is linked only
when it has all three label files and both caches, so the set can be rebuilt while a
collection is still running. Batch names carry the tfrecord and record indices, so files of
different sources never collide; a scene present in two sources is an error.

Output layout (what slurm/train/train_wm_discriminator.sh --data NAME reads):
    OUT/es_candidates/   batch .npz + .labels / .labels_v2 / .labels_waymax
    OUT/wm_predictions/  .wm.npz (+ manifest.json)
    OUT/road_edges/      .edges.npz
    OUT/invalid_at_start.json, OUT/sources.json

    python -m es_discriminator.build_training_set --name fail1500_plus_womd_train_s02_11
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import numpy as np

from .analyze_es_decisions import START_T, batch_files

SCRATCH = "/zfsauton/scratch/sbellad/es_discriminator"
FAIL1500 = "fail1500_es_diffusion_wm_mf_epoch0450_expected_independent_seed0"
WOMD_TRAIN = "womd_train_s02_11_es_diffusion_wm_mf_epoch0450_expected_independent_seed0"
TRAINVAL_TFRECORDS = ("training_tfexample.tfrecord-00000-of-01000", "training_tfexample.tfrecord-00001-of-01000")
LABEL_SUFFIXES = (".labels.npz", ".labels_v2.npz", ".labels_waymax.npz")


def default_sources() -> list[dict]:
    sources = [dict(name=FAIL1500, data=f"{SCRATCH}/es_candidate_data/{FAIL1500}/es_candidates",
                    wm=f"{SCRATCH}/wm_predictions/{FAIL1500}", edges=f"{SCRATCH}/road_edges/{FAIL1500}",
                    invalid=f"{SCRATCH}/decision_analysis/invalid_at_start.json")]
    for shard in sorted(glob.glob(f"{SCRATCH}/es_candidate_data/{WOMD_TRAIN}/shard_*")):
        sources.append(dict(name=f"{WOMD_TRAIN}/{os.path.basename(shard)}", data=f"{shard}/es_candidates",
                            wm=f"{shard}/wm_predictions", edges=f"{shard}/road_edges",
                            scenes=f"{shard}/analysis/scenes.npz"))
    return sources


def invalid_from_scenes(path: str) -> list[dict]:
    """Same rule as analyze_es_decisions `report`: ego overlapping or offroad at the start step."""
    s = dict(np.load(path))
    bad = s["overlap_t"][:, START_T] | s["offroad_t"][:, START_T]
    out = []
    for i in np.flatnonzero(bad):
        kind = "+".join(k for k in ("overlap", "offroad") if s[f"{k}_t"][i, START_T])
        out.append({"tfrecord": str(s["scen_tfrecord"][i]), "scenario_idx": int(s["scen_index"][i]), "type": kind})
    return out


def _link(src: str, dst_dir: str) -> None:
    os.symlink(os.path.abspath(src), os.path.join(dst_dir, os.path.basename(src)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", required=True, help="training set name under training_sets/")
    p.add_argument("--root", default=f"{SCRATCH}/training_sets")
    p.add_argument("--exclude_tfrecord", nargs="*", default=list(TRAINVAL_TFRECORDS),
                   help="drop batches from these tfrecords (default: the trainval benchmark shards)")
    a = p.parse_args()

    out = os.path.join(a.root, a.name)
    dirs = {k: os.path.join(out, k) for k in ("es_candidates", "wm_predictions", "road_edges")}
    for d in dirs.values():
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)

    seen: dict[tuple[str, int], str] = {}
    invalid, report, manifest = [], [], None
    for src in default_sources():
        files = batch_files(src["data"]) if os.path.isdir(src["data"]) else []
        n_excluded = n_incomplete = n_linked = 0
        for f in files:
            stem, base = f[:-4], os.path.basename(f)[:-4]
            if base.split(".scen_")[0] in a.exclude_tfrecord:
                n_excluded += 1
                continue
            wm, edges = os.path.join(src["wm"], base + ".wm.npz"), os.path.join(src["edges"], base + ".edges.npz")
            if not all(os.path.exists(stem + s) for s in LABEL_SUFFIXES) or not (os.path.exists(wm) and os.path.exists(edges)):
                n_incomplete += 1
                continue
            with np.load(f) as z:
                keys = list(zip(z["scen_tfrecord"].astype(str), z["scen_index"].astype(int).tolist()))
            for key in keys:
                if key in seen:
                    raise ValueError(f"scene {key} in both {seen[key]} and {src['name']}")
                seen[key] = src["name"]
            for path in (f, *(stem + s for s in LABEL_SUFFIXES)):
                _link(path, dirs["es_candidates"])
            _link(wm, dirs["wm_predictions"])
            _link(edges, dirs["road_edges"])
            n_linked += 1
        if src.get("invalid") and os.path.exists(src["invalid"]):
            invalid += json.load(open(src["invalid"]))
        elif src.get("scenes") and os.path.exists(src["scenes"]):
            invalid += invalid_from_scenes(src["scenes"])
        elif n_linked:
            raise FileNotFoundError(f"{src['name']}: no invalid_at_start source (analysis/scenes.npz missing)")
        m_path = os.path.join(src["wm"], "manifest.json")
        if n_linked and os.path.exists(m_path):
            m = json.load(open(m_path))
            if manifest is None:
                manifest = m
            elif m["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
                raise ValueError(f"{src['name']}: WM cache from a different checkpoint")
        report.append(dict(source=src["name"], batch_files=len(files), linked=n_linked,
                           incomplete=n_incomplete, excluded_trainval=n_excluded))

    if manifest is not None:
        with open(os.path.join(dirs["wm_predictions"], "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
    invalid = [r for r in invalid if (r["tfrecord"], int(r["scenario_idx"])) in seen]
    with open(os.path.join(out, "invalid_at_start.json"), "w") as f:
        json.dump(invalid, f, indent=1)
    total = dict(batch_files=sum(r["linked"] for r in report), scenes=len(seen), invalid_at_start=len(invalid))
    with open(os.path.join(out, "sources.json"), "w") as f:
        json.dump({"sources": report, "total": total}, f, indent=1)
    print(json.dumps({"out": out, "total": total, "sources": [r for r in report if r["batch_files"]]}, indent=1))


if __name__ == "__main__":
    main()
