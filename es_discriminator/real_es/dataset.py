"""In-memory bank of recorded ES replan groups, shared filters and the scenario split.

`load_bank` reads every batch file (+ `.labels.npz`, `.labels_v2.npz`,
`.labels_waymax.npz`) into flat arrays:

  groups      context keys, scenario key/id, group_t, label horizon, safety flags of the
              replan pose and of the logged expert future, candidate offsets
  candidates  ego-frame trajectories at 0.2 s ([C,25,3] = world steps t+2 .. t+50, the
              discriminator format), ES selection, ADE to the logged expert, goal gain,
              route progress, path length, Waymax overlap/offroad steps, and the v2
              ("report") unsafe flag used to define the 285 ES-unsafe scenes
  scenes      key, id, runner success

Every group holds the same number of candidates (320 for the fail1500 run), so the
candidate block of group g is rows [g*K, (g+1)*K).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

from es_discriminator.analyze_es_decisions import batch_files
from es_discriminator.data import CONTEXT_KEYS, DataConfig

WAYMAX_SUFFIX = ".labels_waymax.npz"
DEFAULT_RUN = ("/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/es_discriminator/es_candidate_data/"
               "fail1500_es_diffusion_wm_mf_epoch0450_expected_independent_seed0/es_candidates")
DEFAULT_SPLIT = "/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/es_discriminator/splits/fail1500_scenario_split_seed0.json"
DEFAULT_INVALID = "/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/es_discriminator/decision_analysis/invalid_at_start.json"
_RUN_NAME = "fail1500_es_diffusion_wm_mf_epoch0450_expected_independent_seed0"
DEFAULT_WM_DIR = f"/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/es_discriminator/wm_predictions/{_RUN_NAME}"
DEFAULT_EDGE_DIR = f"/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/es_discriminator/road_edges/{_RUN_NAME}"
WM_KEYS = ("wm_mean", "wm_cov", "wm_probs", "wm_valid", "wm_size", "wm_current")
EDGE_KEYS = ("edge_xy", "edge_dir", "edge_mask")
SPLITS = ("train", "val", "test")


@dataclass
class Bank:
    k: int                      # candidates per group
    groups: dict                # [G, ...]
    cands: dict                 # [G*K, ...]
    scenes: dict                # [S, ...]
    files: list

    @property
    def num_groups(self) -> int:
        return len(self.groups["group_t"])

    def rows(self, g: int) -> slice:
        return slice(g * self.k, (g + 1) * self.k)


def _cache_path(cache_dir: str, batch: str, suffix: str) -> str:
    return os.path.join(cache_dir, os.path.basename(batch)[:-4] + suffix)


def load_bank(data_dir: str = DEFAULT_RUN, *, max_files: int = 0, cfg: DataConfig = DataConfig(),
              verbose: bool = True, wm_dir: str | None = None, edge_dir: str | None = None) -> Bank:
    """`wm_dir` / `edge_dir` (from `wm_cache`) attach the world-model mixture and road-edge tokens
    to the groups; batch files without both caches are then skipped."""
    files = [f for f in batch_files(data_dir) if os.path.exists(f[:-4] + WAYMAX_SUFFIX)]
    if wm_dir:
        files = [f for f in files if os.path.exists(_cache_path(wm_dir, f, ".wm.npz"))
                 and os.path.exists(_cache_path(edge_dir, f, ".edges.npz"))]
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"no batch files with {WAYMAX_SUFFIX} (and WM/edge caches) under {data_dir}")
    stride, horizon = cfg.future_stride, cfg.horizon
    G, C, S = [], [], []
    k = None
    scen_offset = 0
    for i, f in enumerate(files):
        with np.load(f) as z:
            d = {key: z[key] for key in (*CONTEXT_KEYS, "group_scen", "group_t", "group_valid", "cand_group",
                                         "cand_selected", "cand_traj", "cand_total", "scen_tfrecord", "scen_index",
                                         "scen_id", "scen_success", "group_ego_state")}
        v1 = dict(np.load(f[:-4] + ".labels.npz"))
        v2 = dict(np.load(f[:-4] + ".labels_v2.npz"))
        wm = dict(np.load(f[:-4] + WAYMAX_SUFFIX))
        ng = len(d["group_t"])
        order = np.argsort(d["cand_group"], kind="stable")
        counts = np.bincount(d["cand_group"], minlength=ng)
        if k is None:
            k = int(counts[0])
        if not np.all(counts == k):
            raise ValueError(f"{f}: groups with != {k} candidates: {np.unique(counts)}")
        g = {key: d[key] for key in CONTEXT_KEYS}
        g.update(
            scen=d["group_scen"].astype(np.int32) + scen_offset, group_t=d["group_t"].astype(np.int16),
            group_valid=d["group_valid"].astype(bool), label_horizon=wm["label_horizon"].astype(np.int16),
            anchor_overlap=wm["anchor_overlap"], anchor_offroad=wm["anchor_offroad"],
            expert_overlap_step=wm["expert_overlap_step"], expert_offroad_step=wm["expert_offroad_step"],
            v2_flagged=v2["anchor_collision"] | v2["anchor_offroad"],
            ego_size=d["group_ego_state"][:, 5:7].astype(np.float32),
        )
        if wm_dir:
            with np.load(_cache_path(wm_dir, f, ".wm.npz")) as z:
                g.update({key: z[key] for key in WM_KEYS})
            with np.load(_cache_path(edge_dir, f, ".edges.npz")) as z:
                g.update({key: z[key] for key in EDGE_KEYS})
            if len(g["wm_mean"]) != ng or len(g["edge_xy"]) != ng:
                raise ValueError(f"{f}: WM/edge cache rows do not match {ng} groups")
        G.append(g)
        traj = d["cand_traj"][order][:, stride - 1:stride * horizon:stride]
        c = {
            "traj": np.nan_to_num(traj).astype(np.float32),
            "selected": d["cand_selected"][order].astype(bool),
            "es_total": d["cand_total"][order].astype(np.float32),
            "overlap_step": wm["gt_overlap_step"][order],
            "offroad_step": wm["gt_offroad_step"][order],
            "v2_unsafe": ((v2["gt_collision_step"] > 0) | (v2["gt_offroad_step"] > 0))[order],
        }
        for key in ("gt_ade_m", "gt_goal_gain_m", "gt_progress_m", "gt_path_len_m"):
            c[key[3:-2]] = v1[key][order].astype(np.float32)   # ade, goal_gain, progress, path_len
        C.append(c)
        S.append({"tfrecord": d["scen_tfrecord"].astype(str), "index": d["scen_index"].astype(np.int32),
                  "scen_id": d["scen_id"].astype(str), "success": d["scen_success"].astype(bool)})
        scen_offset += len(d["scen_id"])
        if verbose and ((i + 1) % 50 == 0 or i + 1 == len(files)):
            print(f"loaded {i + 1}/{len(files)} batch files", flush=True)

    cat = lambda parts: {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}  # noqa: E731
    bank = Bank(k=k, groups=cat(G), cands=cat(C), scenes=cat(S), files=files)
    gs = bank.groups
    gs["scen_id"] = bank.scenes["scen_id"][gs["scen"]]
    h = np.repeat(gs["label_horizon"].astype(np.float32), k)
    bank.cands["speed"] = (bank.cands["path_len"] / np.maximum(h, 1) / 0.1).astype(np.float32)
    bank.cands["unsafe"] = (bank.cands["overlap_step"] > 0) | (bank.cands["offroad_step"] > 0)
    ade = bank.cands["ade"].reshape(-1, k)
    gs["min_ade"] = np.where(np.isfinite(ade).any(1), np.nanmin(np.where(np.isfinite(ade), ade, np.inf), 1),
                             np.nan).astype(np.float32)
    gs["expert_unsafe"] = (gs["expert_overlap_step"] > 0) | (gs["expert_offroad_step"] > 0)
    return bank


# ---- split ---------------------------------------------------------------------------

def make_split(scen_ids, seed: int = 0, fractions=(0.70, 0.15, 0.15)) -> dict[str, str]:
    """Deterministic scenario -> split assignment from a hash of the scenario id."""
    edges = np.cumsum(fractions)
    out = {}
    for sid in sorted(set(map(str, scen_ids))):
        u = int(hashlib.sha1(f"{seed}:{sid}".encode()).hexdigest()[:12], 16) / float(16 ** 12)
        out[sid] = SPLITS[int(np.searchsorted(edges, u, side="right"))] if u < edges[-1] else SPLITS[-1]
    return out


def all_scenario_ids(data_dir: str) -> list[str]:
    """Scenario ids of every recorded batch (labelled or not), so the split never depends on
    which subset is loaded."""
    ids = []
    for f in batch_files(data_dir):
        with np.load(f) as z:
            ids.extend(z["scen_id"].astype(str).tolist())
    return ids


def load_or_create_split(path: str, scen_ids, seed: int = 0) -> dict[str, str]:
    ids = set(map(str, scen_ids))
    if os.path.exists(path):
        split = json.load(open(path))["assignment"]
        missing = ids - set(split)
        if missing:
            raise ValueError(f"{path} lacks {len(missing)} scenarios (e.g. {sorted(missing)[:3]})")
        return split
    split = make_split(ids, seed)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    counts = {s: sum(v == s for v in split.values()) for s in SPLITS}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"seed": seed, "unit": "scenario_id", "counts": counts, "assignment": split}, f, indent=1)
    os.replace(tmp, path)
    return split


def group_split(bank: Bank, split: dict[str, str]) -> np.ndarray:
    return np.array([split[s] for s in bank.groups["scen_id"]])


# ---- filters ---------------------------------------------------------------------------

def invalid_scene_keys(path: str = DEFAULT_INVALID) -> set[tuple[str, int]]:
    if not path or not os.path.exists(path):
        return set()
    return {(r["tfrecord"], int(r["scenario_idx"])) for r in json.load(open(path))}


def scene_invalid_mask(bank: Bank, invalid: set) -> np.ndarray:
    sc = bank.scenes
    return np.array([(t, int(i)) in invalid for t, i in zip(sc["tfrecord"], sc["index"])], bool)


def training_mask(bank: Bank, invalid: set, *, filter_expert: bool = True) -> tuple[np.ndarray, dict]:
    """Groups usable for training/evaluation, identical for every model."""
    gs = bank.groups
    steps = {}
    m = gs["group_valid"].copy()
    steps["group_valid"] = int(m.sum())
    m &= ~scene_invalid_mask(bank, invalid)[gs["scen"]]
    steps["not_invalid_at_start_scene"] = int(m.sum())
    m &= ~(gs["anchor_overlap"] | gs["anchor_offroad"])
    steps["replan_pose_clean"] = int(m.sum())
    if filter_expert:
        m &= ~gs["expert_unsafe"]
        steps["logged_expert_future_clean"] = int(m.sum())
    m &= np.isfinite(gs["min_ade"])
    steps["has_finite_ade"] = int(m.sum())
    return m, {"total_groups": int(bank.num_groups), "remaining_after": steps}


def es_unsafe_scenes(bank: Bank, invalid: set) -> tuple[np.ndarray, np.ndarray]:
    """Scenes whose first bad replan is "ES picked unsafe although a safe candidate existed",
    with the decision report's rule (v2 labels, controllable failed scenes). Returns the
    scene mask [S] and, per such scene, the group index of that first bad replan [S] (-1)."""
    gs, sc, k = bank.groups, bank.scenes, bank.k
    inval = scene_invalid_mask(bank, invalid)
    unsafe = bank.cands["v2_unsafe"].reshape(-1, k)
    sel = bank.cands["selected"].reshape(-1, k)
    mask = np.zeros(len(sc["scen_id"]), bool)
    first = np.full(len(sc["scen_id"]), -1, np.int64)
    by_scene: dict[int, list[int]] = {}
    for g, s in enumerate(gs["scen"]):
        by_scene.setdefault(int(s), []).append(g)
    for s, glist in by_scene.items():
        if inval[s] or sc["success"][s]:
            continue
        for g in sorted(glist, key=lambda x: int(gs["group_t"][x])):
            u = unsafe[g]
            if gs["v2_flagged"][g] or u.all():
                break
            if u.any() and u[np.argmax(sel[g])]:
                mask[s] = True
                first[s] = g
                break
    return mask, first


# ---- batching ---------------------------------------------------------------------------

def context_batch(bank: Bank, gidx: np.ndarray) -> dict[str, np.ndarray]:
    return {key: bank.groups[key][gidx] for key in CONTEXT_KEYS}


def cand_batch(bank: Bank, gidx: np.ndarray, keys) -> dict[str, np.ndarray]:
    k = bank.k
    rows = (gidx[:, None] * k + np.arange(k)[None]).reshape(-1)
    return {key: bank.cands[key][rows].reshape((len(gidx), k) + bank.cands[key].shape[1:]) for key in keys}


def wm_batch(bank: Bank, gidx: np.ndarray) -> dict[str, np.ndarray]:
    """World-model inputs (all modes with probabilities and std)."""
    return {k[3:]: bank.groups[k][gidx] for k in WM_KEYS}


def edge_batch(bank: Bank, gidx: np.ndarray) -> dict[str, np.ndarray]:
    return {k[5:]: bank.groups[k][gidx] for k in EDGE_KEYS}


def iterate_batches(gidx: np.ndarray, batch_groups: int, rng: np.random.Generator | None = None):
    order = rng.permutation(gidx) if rng is not None else np.asarray(gidx)
    for i in range(0, len(order), batch_groups):
        yield order[i:i + batch_groups]
