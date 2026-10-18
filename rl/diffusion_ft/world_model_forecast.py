"""Learned world-model forecasts of the OTHER agents, for the post-hoc repair.

The honest neighbour source the repair shipped with is constant velocity
(``safety_objective.predicted_neighbor_window``, ``neighbor_pred="cv"``): every
agent extrapolated from its state at ``t0``. This module swaps in the learned
world models of mini-hong/waymax_rs@es_WM (Sagar Bellad), vendored byte-identical
under ``wm_vendor/`` (see its PROVENANCE.md). Three neighbour sources:

* ``wm_det``     -- the deterministic WM: one predicted future per agent.
* ``wm_mf``      -- the multi-future Gaussian WM's K mode means (K=6), each a
  pseudo-agent whose barrier is weighted by its mode probability. The agent term
  is a SUM over agents, so this is exactly the expected barrier over each agent's
  modes, independently per agent -- the analogue of the ES's
  ``expected_independent`` scoring, their best non-oracle forecast on fail1500.
* ``wm_mf_top1`` -- the same model's most probable mode only, weight 1.

Like cv, nothing here reads past ``t0``. The WM's agent history is the last 2 s
of ``sim_trajectory`` -- where every agent actually WAS, the driven ego included,
reactive NPCs included. Their ES runner feeds ``log_trajectory`` instead, which in
closed loop shows the WM the LOGGED ego's past, not the driven one: a ghost car
that the log-replayed NPCs were recorded reacting to. We do not.

The WM is PyTorch. It runs eagerly, once per (state, t0), outside the jitted
optimizer and scorer; its boxes enter their jitted programs as an argument
(``safety_objective.candidate_safety(neighbor_boxes=...)``). The forecast is
decoded at each scene's own world clock ``k * dt_b`` (k = 1..n), as cv is.

What it returns is the ``candidate_safety`` neighbour contract plus a weight:
``(box [B, n, M, 5] = (x, y, length, width, yaw), valid [B, n, M], weight [B, n, M])``
with ``M = N * K`` (pseudo-agent ``i * K + k`` is agent ``i``'s mode ``k``).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

from rl.diffusion_ft.safety_objective import WM_PREDICTIONS

_CKPT_ROOT = "/zfsauton/scratch/sbellad/world_model"
# det: es_WM README section 8's default. mf: the run's latest.pt, as handed over
# 2026-10-11 (the README's fail1500 rows used epoch_0450.pt of the same run; the
# two files differ, latest.pt was written 7 min after it).
DEFAULT_CKPTS = {
    "det": f"{_CKPT_ROOT}/ckpts/world_model_20260912_170741/epoch_0500.pt",
    "mf": (f"{_CKPT_ROOT}/ckpts_multifuture_gaussian/"
           "multiFuture_GaussianWorldModel_20260916_200620/latest.pt"),
}
_PKG = {"wm_det": "det", "wm_mf": "mf", "wm_mf_top1": "mf"}
_VENDOR = Path(__file__).resolve().parent / "wm_vendor"
# Top-level names the vendored code imports absolutely, and that this repo also
# has (with a different `_build_map_features` signature, see PROVENANCE.md).
_SHADOWED = ("data", "scores")


def _is_shadowed(name: str) -> bool:
    return name.split(".")[0] in _SHADOWED


def _import_inference(pkg: str):
    """The vendored ``wm_<pkg>.inference`` module, imported against the VENDORED
    ``data`` / ``scores`` and leaving this repo's in ``sys.modules`` untouched.

    The vendored modules bind what they import at import time, so swapping the
    top-level names only for the duration of the import is enough. Every
    ``data.*`` / ``scores.*`` entry is moved aside, not just the top-level one:
    an already-imported ``data.preprocess`` would otherwise win the lookup.
    """
    name = f"wm_{pkg}.inference"
    if name in sys.modules:
        return sys.modules[name]
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if _is_shadowed(k)}
    sys.path.insert(0, str(_VENDOR))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(_VENDOR))
        for k in [k for k in sys.modules if _is_shadowed(k)]:
            del sys.modules[k]
        sys.modules.update(saved)


def _resolve_device(device: str) -> str:
    d = str(device or "auto").strip().lower()
    if d != "auto":
        return d
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _rot(xy, heading):
    c, s = np.cos(heading), np.sin(heading)
    return np.stack([xy[..., 0] * c - xy[..., 1] * s, xy[..., 0] * s + xy[..., 1] * c], -1)


class WMNeighborForecaster:
    """``f(state, t0, n, dt_b) -> (box, valid, weight)`` from a vendored WM checkpoint.

    Cached on (state object, t0): the optimizer's starts and the scorer all ask
    about the same replan, and the WM must run once for them. The cached state is
    held, so its ``id`` cannot be recycled while it is the key.
    """

    def __init__(self, source: str, ckpt: str, device: str = "auto"):
        if source not in WM_PREDICTIONS:
            raise ValueError(f"source must be one of {WM_PREDICTIONS}, got {source!r}")
        inf = _import_inference(_PKG[source])
        self.source, self.ckpt, self.device = source, str(ckpt), _resolve_device(device)
        self.base = inf.WorldModelForecaster.from_checkpoint(self.ckpt, device=self.device)
        self.cfg = self.base.config
        self.dynamic_types = np.asarray(inf._DYNAMIC_TYPES)
        self.n_queries = 0
        self._key_state, self._key_t0, self._raw = None, None, None
        self._decoded = {}

    # ------------------------------------------------------------------ #
    def _predict(self, state, t0: int) -> dict:
        """One WM forward on the driven history. Delta / yaw per mode, ``[B, N, K, T, 2]``."""
        import torch

        hist = state.replace(log_trajectory=state.sim_trajectory)
        features, meta = self.base._prepare_features(hist, int(t0))
        with torch.inference_mode():
            pred = self.base.model(features)

        def np_(k):
            return pred[k].detach().float().cpu().numpy()

        if self.source == "wm_mf":
            delta, yawv, probs = (np_("pred_delta_xy_norm_k"), np_("pred_yaw_vec_k"),
                                  np_("mode_probs"))
        else:                         # wm_det, or wm_mf_top1 (the MF model's top-1 gather)
            delta, yawv = np_("pred_delta_xy_norm")[:, :, None], np_("pred_yaw_vec")[:, :, None]
            probs = np.ones(delta.shape[:3], np.float32)
        self.n_queries += 1
        return {"delta": delta, "yawv": yawv, "probs": probs, "meta": meta}

    def decode(self, raw: dict, tq_bn: np.ndarray):
        """Boxes at query times ``tq_bn`` [B, n] (seconds after t0).

        Vectorized ``WorldModelForecaster._decode_boxes``: the same frame (the
        anchor ego's), the same t=0 knot at the current pose, the same linear
        interpolation clamped at the horizon, yaw through sin/cos, and the same
        hold-in-place for every agent that is not a valid dynamic non-ego agent
        inside the modeled set. ``tests/test_wm_forecast.py`` pins it to theirs.
        """
        cfg, m = self.cfg, raw["meta"]
        t = int(m["timestep"])
        n_mod = int(m["modeled_agents"])
        B, N = m["x"].shape[:2]
        K, T = raw["delta"].shape[2], raw["delta"].shape[3]
        n = tq_bn.shape[1]

        cur_xy = np.stack([m["x"][:, :, t], m["y"][:, :, t]], -1)          # [B, N, 2]
        cur_yaw = m["yaw"][:, :, t]
        cur_valid = m["valid"][:, :, t]
        is_sdc = m["is_sdc"]
        moving = (cur_valid & np.isin(m["object_types"], self.dynamic_types) & ~is_sdc)
        moving[:, n_mod:] = False

        d_world = _rot(raw["delta"][:, :n_mod] * float(cfg.max_range),
                       m["ego_yaw"][:, None, None, None])                 # [B, n_mod, K, T, 2]
        fut_xy = cur_xy[:, :n_mod, None, None, :] + d_world
        fut_yaw = _wrap(cur_yaw[:, :n_mod, None, None]
                        + np.arctan2(raw["yawv"][:, :n_mod, ..., 0], raw["yawv"][:, :n_mod, ..., 1]))
        src_xy = np.concatenate(
            [np.broadcast_to(cur_xy[:, :n_mod, None, None, :], (B, n_mod, K, 1, 2)), fut_xy], 3)
        src_yaw = np.concatenate(
            [np.broadcast_to(cur_yaw[:, :n_mod, None, None], (B, n_mod, K, 1)), fut_yaw], 3)

        # np.interp on knots 0, dt_m, .., T dt_m: clamp, then lerp between i0 and i0+1.
        u = np.clip(np.asarray(tq_bn, np.float64) / float(cfg.future_dt), 0.0, float(T))
        i0 = np.minimum(np.floor(u).astype(np.int64), T - 1)               # [B, n]
        w = (u - i0).astype(np.float32)[:, None, None, :]                   # [B, 1, 1, n]
        idx = np.broadcast_to(i0[:, None, None, :], (B, n_mod, K, n))

        def lerp(src):                                                      # [B, n_mod, K, T+1]
            a = np.take_along_axis(src, idx, axis=3)
            b = np.take_along_axis(src, idx + 1, axis=3)
            return (1.0 - w) * a + w * b

        qx, qy = lerp(src_xy[..., 0]), lerp(src_xy[..., 1])
        qyaw = np.arctan2(lerp(np.sin(src_yaw)), lerp(np.cos(src_yaw)))

        out = np.empty((B, N, K, n, 3), np.float32)
        out[..., 0] = cur_xy[:, :, None, None, 0]
        out[..., 1] = cur_xy[:, :, None, None, 1]
        out[..., 2] = cur_yaw[:, :, None, None]
        mv = moving[:, :n_mod, None, None]
        sub = out[:, :n_mod]
        sub[..., 0] = np.where(mv, qx, sub[..., 0])
        sub[..., 1] = np.where(mv, qy, sub[..., 1])
        sub[..., 2] = np.where(mv, qyaw, sub[..., 2])

        ln = np.broadcast_to(m["length"][:, :, t, None, None], (B, N, K, n))
        wd = np.broadcast_to(m["width"][:, :, t, None, None], (B, N, K, n))
        box = np.stack([out[..., 0], out[..., 1], ln, wd, out[..., 2]], -1)  # candidate_safety order
        valid = np.broadcast_to((cur_valid & ~is_sdc)[:, :, None, None], (B, N, K, n))
        probs = np.zeros((B, N, K), np.float32)
        probs[:, :n_mod] = raw["probs"][:, :n_mod]
        probs[:, n_mod:, 0] = 1.0           # beyond the modeled set: held, one copy
        weight = np.broadcast_to(probs[..., None], (B, N, K, n))

        def btm(a):                                                         # [B,N,K,n,..] -> [B,n,N*K,..]
            a = np.moveaxis(a, 3, 1)
            return np.ascontiguousarray(a.reshape(B, n, N * K, *a.shape[4:]))

        return btm(box).astype(np.float32), btm(valid), btm(weight).astype(np.float32)

    # ------------------------------------------------------------------ #
    def __call__(self, state, t0, n: int, dt_b):
        """``(box [B,n,M,5], valid [B,n,M], weight [B,n,M])`` as jnp arrays."""
        import jax.numpy as jnp

        t = int(t0)
        if self._key_state is not state or self._key_t0 != t:
            self._raw = self._predict(state, t)
            self._key_state, self._key_t0, self._decoded = state, t, {}
        dt = np.asarray(dt_b, np.float64).reshape(-1)
        key = (int(n), dt.tobytes())
        if key not in self._decoded:
            tq = dt[:, None] * np.arange(1, int(n) + 1, dtype=np.float64)[None, :]
            box, valid, weight = self.decode(self._raw, tq)
            self._decoded[key] = (jnp.asarray(box), jnp.asarray(valid), jnp.asarray(weight))
        return self._decoded[key]


# --------------------------------------------------------------------------- #
# One forecaster per source per process, shared by every scorer and optimizer
# that names it (so a replan runs the WM once, not once per consumer).
_CONFIG = {"det": DEFAULT_CKPTS["det"], "mf": DEFAULT_CKPTS["mf"], "device": "auto"}
_INSTANCES: dict[str, WMNeighborForecaster] = {}


def configure(*, det_ckpt: str = "", mf_ckpt: str = "", device: str = "") -> None:
    """Override the checkpoints / device before the first ``get_forecaster``."""
    new = {k: str(v) for k, v in (("det", det_ckpt), ("mf", mf_ckpt), ("device", device)) if v}
    if _INSTANCES and any(_CONFIG[k] != v for k, v in new.items()):
        raise RuntimeError(f"world-model config changed after {sorted(_INSTANCES)} were built")
    _CONFIG.update(new)


def get_forecaster(source: str) -> WMNeighborForecaster:
    if source not in _INSTANCES:
        _INSTANCES[source] = WMNeighborForecaster(source, _CONFIG[_PKG[source]],
                                                  _CONFIG["device"])
        f = _INSTANCES[source]
        print(f"[wm] {source}: {f.ckpt} on {f.device} (K={f.cfg.num_modes if source == 'wm_mf' else 1}"
              f", history {f.cfg.history_steps}x{f.cfg.history_stride} steps, "
              f"horizon {f.cfg.future_steps * f.cfg.future_dt:.1f}s)", flush=True)
    return _INSTANCES[source]


def describe() -> dict:
    """What the run used, for the select config dump."""
    return {"ckpts": {k: _CONFIG[k] for k in ("det", "mf")}, "device": _CONFIG["device"],
            "built": sorted(_INSTANCES), "history": "sim_trajectory",
            "queries": {k: v.n_queries for k, v in _INSTANCES.items()}}
