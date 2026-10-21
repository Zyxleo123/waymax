"""Recorded Diffusion-ES bank (+ cache_bank output) -> scorer fine-tuning batches.

Groups are filtered as es_discriminator.real_es.dataset.training_mask (identical to B0 / v3):
group_valid, scene not invalid at start, replan pose clean (no Waymax overlap / offroad), logged
expert future clean, finite ADE; then kept by scenario split (the bank's scenario_split json).

Each yielded batch stacks B groups:
  model inputs   the cache_bank "A" arrays (agent_history, map_features, ...) + goal_xy, as float32 / bool
  cand_xy [B,K,25,2], cand_yaw [B,K,25]   candidates at 0.2 s (world steps t+2 .. t+50; cand_traj[:, 1:50:2])
  labels [B,K]   overlap_step / offroad_step (Waymax, 0.1 s steps from t+1, 0 = none), goal_gain, progress,
                 final_goal_dist, path_len, ade, es_total, es_collision, es_offroad, selected, stage
  horizon [B]    label horizon (0.1 s steps)
  b_gap / b_type / b_offroad [B,K,50]   cache_bank "B" per-step geometry (0.1 s, world steps t+1 .. t+50)
  lane_xy / lane_dir / lane_id / lane_valid   cache_bank "C"
K candidates per group are sampled at random during training (`cands_per_group`, at most `max_unsafe_frac`
of them unsafe, as B0's stratified sampling), or all 320 for evaluation (`cands_per_group=0`).
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from anchor_wm.cache_bank import A_KEYS

LABEL_KEYS = ("overlap_step", "offroad_step", "goal_gain", "progress", "final_goal_dist", "path_len", "ade",
              "es_total", "es_collision", "es_offroad", "selected", "stage")
GEO_KEYS = ("b_gap", "b_type", "b_offroad")
LANE_KEYS = ("lane_xy", "lane_dir", "lane_id", "lane_valid")


def _batch_files(bank_dir: str) -> list[str]:
    return sorted(f for f in glob.glob(os.path.join(bank_dir, "*.npz")) if re.search(r"\.scen_[\d_]+\.npz$", f))


def _cache_path(cache_dir: str, batch: str) -> str:
    return os.path.join(cache_dir, os.path.basename(batch)[:-4] + ".anchor_cache.npz")


def build_index(bank_dir: str, cache_dir: str, split_json: str, invalid_json: str, splits: tuple[str, ...],
                filter_expert: bool = True) -> tuple[list[tuple[str, np.ndarray]], dict]:
    """[(batch file, usable group indices)] for files that have a cache, and filter counts."""
    with open(split_json) as fh:
        assign = json.load(fh)["assignment"]
    invalid = set()
    if invalid_json and os.path.exists(invalid_json):
        with open(invalid_json) as fh:
            invalid = {(r["tfrecord"], int(r["scenario_idx"])) for r in json.load(fh)}
    index, counts = [], {"files": 0, "files_without_cache": 0, "groups": 0, "kept": 0}
    for f in _batch_files(bank_dir):
        if not os.path.exists(_cache_path(cache_dir, f)):
            counts["files_without_cache"] += 1
            continue
        with np.load(f) as z:
            g_valid = z["group_valid"].astype(bool)
            g_scen = z["group_scen"].astype(int)
            tf, idx, sid = z["scen_tfrecord"].astype(str), z["scen_index"], z["scen_id"].astype(str)
            cand_group = z["cand_group"]
        with np.load(f[:-4] + ".labels_waymax.npz") as z:
            clean = ~(z["anchor_overlap"].astype(bool) | z["anchor_offroad"].astype(bool))
            expert_ok = (z["expert_overlap_step"] == 0) & (z["expert_offroad_step"] == 0)
        with np.load(f[:-4] + ".labels.npz") as z:
            ade = z["gt_ade_m"]
        finite = np.array([np.isfinite(ade[cand_group == g]).any() for g in range(len(g_valid))])
        scene_ok = np.array([(tf[s], int(idx[s])) not in invalid for s in range(len(sid))])
        m = g_valid & scene_ok[g_scen] & clean & finite
        if filter_expert:
            m &= expert_ok
        m &= np.isin(np.array([assign.get(sid[s]) for s in g_scen], dtype=object), list(splits))
        counts["files"] += 1
        counts["groups"] += len(g_valid)
        counts["kept"] += int(m.sum())
        if m.any():
            index.append((f, np.flatnonzero(m)))
    return index, counts


def _sample_cands(unsafe: np.ndarray, k: int, max_unsafe_frac: float, rng: np.random.Generator) -> np.ndarray:
    n = len(unsafe)
    if k <= 0 or k >= n:
        return np.arange(n)
    bad, good = np.flatnonzero(unsafe), np.flatnonzero(~unsafe)
    n_bad = min(len(bad), int(round(max_unsafe_frac * k)), k)
    n_good = min(len(good), k - n_bad)
    n_bad = min(len(bad), k - n_good)
    pick = np.concatenate([rng.choice(bad, n_bad, replace=False), rng.choice(good, n_good, replace=False)])
    return np.sort(pick)


def load_file_groups(batch: str, cache_dir: str, groups: np.ndarray, *, cands_per_group: int,
                     max_unsafe_frac: float, rng: np.random.Generator) -> list[dict]:
    d = np.load(batch)
    c = np.load(_cache_path(cache_dir, batch))
    lw = np.load(batch[:-4] + ".labels_waymax.npz")
    l1 = np.load(batch[:-4] + ".labels.npz")
    rows_all = c["cand_rows"]
    per_cand = {
        "overlap_step": lw["gt_overlap_step"], "offroad_step": lw["gt_offroad_step"],
        "goal_gain": l1["gt_goal_gain_m"], "progress": l1["gt_progress_m"],
        "final_goal_dist": l1["gt_final_goal_dist_m"], "path_len": l1["gt_path_len_m"], "ade": l1["gt_ade_m"],
        "es_total": d["cand_total"], "es_collision": d["cand_collision"], "es_offroad": d["cand_offroad"],
        "selected": d["cand_selected"], "stage": d["cand_stage"],
    }
    cand_traj = d["cand_traj"]
    out = []
    for g in groups:
        rows = rows_all[g]
        unsafe = (per_cand["overlap_step"][rows] > 0) | (per_cand["offroad_step"][rows] > 0)
        pick = _sample_cands(unsafe, cands_per_group, max_unsafe_frac, rng)
        r = rows[pick]
        traj = np.nan_to_num(cand_traj[r][:, 1:50:2]).astype(np.float32)                # [K,25,3]
        rec = {key: np.asarray(c[key][g]) for key in A_KEYS + ("goal_xy",) + LANE_KEYS}
        rec.update({key: np.asarray(c[key][g][pick]) for key in GEO_KEYS})
        rec.update({key: np.asarray(v[r]) for key, v in per_cand.items()})
        rec["cand_xy"], rec["cand_yaw"] = traj[..., :2], traj[..., 2]
        rec["horizon"] = np.int64(lw["label_horizon"][g])
        out.append(rec)
    return out


def _to_tensor(v: np.ndarray) -> torch.Tensor:
    v = np.asarray(v)
    if v.dtype == np.float16 or v.dtype == np.float64:
        v = v.astype(np.float32)
    elif v.dtype == np.int8 or v.dtype == np.int16 or v.dtype == np.uint8:
        v = v.astype(np.int64)
    return torch.from_numpy(np.ascontiguousarray(v))


def collate(recs: list[dict]) -> dict[str, torch.Tensor]:
    return {key: _to_tensor(np.stack([r[key] for r in recs])) for key in recs[0]}


class BankGroupDataset(IterableDataset):
    """Shuffled files -> shuffled usable groups -> batches of `batch_groups`. Files split across workers."""

    def __init__(self, index: list[tuple[str, np.ndarray]], cache_dir: str, *, batch_groups: int,
                 cands_per_group: int, max_unsafe_frac: float = 0.5, shuffle: bool = True, seed: int = 0) -> None:
        super().__init__()
        self.index, self.cache_dir = index, cache_dir
        self.batch_groups, self.cands_per_group = int(batch_groups), int(cands_per_group)
        self.max_unsafe_frac, self.shuffle, self.seed = float(max_unsafe_frac), bool(shuffle), int(seed)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        order = list(range(len(self.index)))
        rng = random.Random(self.seed)
        if self.shuffle:
            rng.shuffle(order)
        np_rng = np.random.default_rng(self.seed * 1000 + wid)
        buf: list[dict] = []
        for i in order[wid::nw]:
            f, groups = self.index[i]
            recs = load_file_groups(f, self.cache_dir, groups, cands_per_group=self.cands_per_group,
                                    max_unsafe_frac=self.max_unsafe_frac, rng=np_rng)
            if self.shuffle:
                np_rng.shuffle(recs)
            buf.extend(recs)
            while len(buf) >= self.batch_groups:
                yield collate(buf[:self.batch_groups])
                buf = buf[self.batch_groups:]
        if buf and not self.shuffle:
            yield collate(buf)


def make_bank_loader(index, cache_dir: str, *, batch_groups: int, cands_per_group: int, shuffle: bool, seed: int,
                     num_workers: int, max_unsafe_frac: float = 0.5) -> DataLoader:
    ds = BankGroupDataset(index, cache_dir, batch_groups=batch_groups, cands_per_group=cands_per_group,
                          max_unsafe_frac=max_unsafe_frac, shuffle=shuffle, seed=seed)
    return DataLoader(ds, batch_size=None, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      prefetch_factor=2 if num_workers > 0 else None)
