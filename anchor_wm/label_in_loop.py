"""Label and cache a recorded Diffusion-ES batch inside the closed-loop run (`run_simulation.py
--save_es_candidates --label_in_loop`), in place of the four offline passes over the bank:

  es_discriminator.label_es_candidates       -> <batch>.labels.npz          (log-replay collision / offroad,
                                                                             progress, goal gain, horizon)
  es_discriminator.analyze_es_decisions      -> <batch>.labels_v2.npz       (Waymax-exact offroad)
    (relabel stage)
  es_discriminator.real_es.waymax_labels     -> <batch>.labels_waymax.npz   (Waymax overlap, expert checks)
  anchor_wm.cache_bank                       -> <cache_dir>/<batch>.anchor_cache.npz
                                                                            (AnchorWM inputs + per-candidate
                                                                             geometry, for bank_data / finetune)

The same functions run on the same inputs; the only difference is the scenarios' original simulator state,
which the runner already holds instead of each pass re-reading the tfrecord. The files are therefore
interchangeable with the offline ones (`--check_against` compares them). The cache is written last, so a
batch whose cache exists is complete.
"""

from __future__ import annotations

import os
import time

import jax
import numpy as np

from anchor_wm.cache_bank import process_file
from anchor_wm.config import AnchorWMConfig
from es_discriminator.analyze_es_decisions import compute_labels_v2
from es_discriminator.label_es_candidates import compute_labels
from es_discriminator.real_es.waymax_labels import compute_waymax_labels


def _save(path: str, arrays: dict) -> None:
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def cache_path(cache_dir: str, batch_path: str) -> str:
    return os.path.join(cache_dir, os.path.basename(batch_path)[:-4] + ".anchor_cache.npz")


def label_and_cache(batch_path: str, state, cache_dir: str, *, cfg: AnchorWMConfig | None = None,
                    offroad_chunk: int = 2048) -> dict:
    """Write the batch's three label files next to it and its AnchorWM cache to `cache_dir`.

    `state`: the batch's scenarios' original simulator state (as loaded by the runner, one world per recorded
    scenario, i.e. without the runner's padding copies). Returns timings and the cache's overlap agreement
    with the Waymax labels (the offline `cache_bank --check`)."""
    t0 = time.time()
    d = dict(np.load(batch_path))
    st = state                     # as the offline passes get it from load_batch_state (device arrays)
    n = len(d["scen_index"])
    if int(st.log_trajectory.x.shape[0]) != n:
        raise ValueError(f"{batch_path}: state has {st.log_trajectory.x.shape[0]} worlds, batch {n} scenarios")
    base = batch_path[:-4]
    v1 = compute_labels(d, st, offroad_chunk)
    _save(base + ".labels.npz", v1)
    v2 = compute_labels_v2(d, v1, st)
    _save(base + ".labels_v2.npz", v2)
    lw = compute_waymax_labels(d, v1, v2, st)
    _save(base + ".labels_waymax.npz", lw)
    t_labels = time.time() - t0
    res = process_file(batch_path, "", cfg or AnchorWMConfig(), check=True, state=st, d=d, lab=lw)
    os.makedirs(cache_dir, exist_ok=True)
    _save(cache_path(cache_dir, batch_path), res["arrays"])
    return {"labels_s": round(t_labels, 1), "cache_s": round(time.time() - t0 - t_labels, 1),
            "groups": int(len(d["group_t"])), "candidates": int(len(d["cand_id"])), "overlap_agree": res["agree"]}


def compare_batches(batch_path: str, cache_dir: str, ref_dir: str, ref_cache_dir: str) -> dict:
    """Max abs difference per array between this batch's in-loop files and offline files for the same batch
    (`ref_dir` holds the offline label files, `ref_cache_dir` the offline cache)."""
    name = os.path.basename(batch_path)[:-4]
    pairs = [(f"{batch_path[:-4]}{s}", os.path.join(ref_dir, name + s))
             for s in (".labels.npz", ".labels_v2.npz", ".labels_waymax.npz")]
    pairs.append((cache_path(cache_dir, batch_path), cache_path(ref_cache_dir, batch_path)))
    out = {}
    for ours, ref in pairs:
        a, b = np.load(ours), np.load(ref)
        diffs = {}
        for key in sorted(set(a.files) | set(b.files)):
            if key not in a.files or key not in b.files:
                diffs[key] = "missing"
                continue
            x, y = a[key], b[key]
            if x.shape != y.shape:
                diffs[key] = f"shape {x.shape} vs {y.shape}"
            elif x.dtype.kind in "fc":
                both_nan = np.isnan(x) & np.isnan(y)
                d = np.abs(np.where(both_nan, 0, x.astype(np.float64) - y.astype(np.float64)))
                if d.size and float(np.nanmax(d)) > 0:
                    diffs[key] = float(np.nanmax(d))
            elif not np.array_equal(x, y):
                diffs[key] = f"{int((x != y).sum())} of {x.size} differ"
        out[os.path.basename(ours)] = diffs or "identical"
    return out
