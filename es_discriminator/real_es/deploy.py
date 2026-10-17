"""Score Diffusion-ES populations with a trained discriminator, from the planner's own inputs.

    scorer = DiscriminatorScorer("/.../best.pt")            # also loads the frozen WM
    scores_bk = scorer.score(sim_state, timestep, goal_xy_b2, pop_world_bkt5)   # higher = better

`sim_state` is the state the planner receives at a replan (`replaced_state`, whose
`log_trajectory` holds the executed ego), `pop_world_bkt5` the world-frame population
[x, y, yaw, vx, vy] at 0.1 s with index 0 = anchor. Context (one frozen-WM forward, the
agent selection and the road-edge tokens, built by the same functions as the offline cache)
is computed once per (state, timestep) and shared by every ES iteration of that replan; each
call then only encodes the candidates. The planner uses it through `closed_loop.LearnedESScorer`.

Consistency check against the offline training path on recorded groups:

  python -m es_discriminator.real_es.deploy --checkpoint .../best.pt --data_dir .../es_candidates \
      --tfrecord_dir .../tfrecords/training --max_files 1
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from es_discriminator.data import CONTEXT_KEYS, DataConfig
from es_discriminator.es_adapter import context_from_sim_state, pool_world_to_ego
from es_discriminator.model import make_candidate_features_torch

from .dataset import EDGE_KEYS, WM_KEYS


def check_forecaster(forecaster) -> None:
    """The WM must produce the inputs the discriminator was trained on (wm_cache manifest)."""
    cfg = forecaster.config
    want = {"num_modes": 6, "future_steps": 25, "future_stride": 2, "max_agents": 128}
    got = {k: int(getattr(cfg, k)) for k in want}
    if got != want or not hasattr(forecaster, "_prepare_features"):
        raise ValueError(f"world model does not match the discriminator's training inputs: {got} != {want}")


class DiscriminatorScorer:
    def __init__(self, checkpoint: str, device: str = "auto", wm_checkpoint: str | None = None,
                 forecaster=None) -> None:
        """`forecaster`: an already-loaded multi-future Gaussian WM (e.g. the planner's) to reuse
        instead of loading `wm_checkpoint`; its configuration must match the cached training inputs."""
        from .trainer import build_model

        dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                           ("cpu" if device == "auto" else device))
        from .wm_cache import DEFAULT_CKPT, load_forecaster

        ck = torch.load(checkpoint, map_location=dev, weights_only=False)
        self.model = build_model(ck["model_kwargs"]).to(dev).eval()
        self.model.load_state_dict(ck["model"])
        self.variant = self.model.variant
        self.cfg = DataConfig(**ck["data_config"])
        self.feat = dict(ck["candidate_features"])
        self.method = ck.get("method", "")
        self.device = dev
        self.forecaster = forecaster or load_forecaster(wm_checkpoint or DEFAULT_CKPT, str(dev))
        check_forecaster(self.forecaster)
        self._cache_key = None
        self._cache = None
        self.wm_forward_s: list[float] = []

    def contexts(self, sim_state, timestep: int, goal_xy_b2: np.ndarray):
        key = (id(sim_state), int(timestep))
        if self._cache_key == key:
            return self._cache
        import jax
        st = jax.device_get(sim_state)
        goal = np.asarray(goal_xy_b2, dtype=np.float64)
        nw = goal.shape[0]
        ctxs, origin, heading, valid = [], np.zeros((nw, 2)), np.zeros(nw), np.zeros(nw, bool)
        empty = None
        for w in range(nw):
            c = context_from_sim_state(st, w, int(timestep), goal[w], self.cfg, use_sim_trajectory=False)
            if c is None:
                ctxs.append(None)
                continue
            ctxs.append(c)
            origin[w], heading[w], valid[w] = c["origin"], c["heading"], True
            empty = empty or {k: np.zeros_like(c[k]) for k in CONTEXT_KEYS}
        if empty is None:
            raise ValueError(f"ego invalid in every world at t={timestep}")
        batch = {k: np.stack([(c or empty)[k] for c in ctxs]) for k in CONTEXT_KEYS}
        batch.update(self._wm_inputs(sim_state, st, int(timestep), origin, heading))
        self._cache_key, self._cache = key, (batch, origin, heading, valid)
        return self._cache

    def _wm_inputs(self, sim_state, st, t: int, origin: np.ndarray, heading: np.ndarray) -> dict:
        from .wm_cache import _roadgraph_np, edge_tokens, group_inputs, jax_state, predict_step
        t0 = time.perf_counter()
        p = predict_step(self.forecaster, jax_state(sim_state), t)
        self.wm_forward_s.append(time.perf_counter() - t0)
        log = st.log_trajectory
        sdc = np.argmax(np.asarray(st.object_metadata.is_sdc).astype(bool), axis=1)
        rg = _roadgraph_np(st)
        dt = 0.1 * int(self.forecaster.config.future_stride)
        rows, edges, size = [], [], []
        for w in range(len(origin)):
            e = int(sdc[w])
            speed = float(np.hypot(np.asarray(log.vel_x)[w, e, t], np.asarray(log.vel_y)[w, e, t]))
            rows.append(group_inputs(p, w, origin[w], float(heading[w]), speed, dt))
            edges.append(edge_tokens(rg, w, origin[w], float(heading[w])))
            size.append([np.asarray(log.length)[w, e, t], np.asarray(log.width)[w, e, t]])
        out = {k: np.stack([r[k] for r in rows]) for k in WM_KEYS}
        out.update({k: np.stack([r[k] for r in edges]) for k in EDGE_KEYS})
        out["ego_size"] = np.asarray(size, np.float32)
        return out

    @torch.no_grad()
    def score_ego(self, context: dict[str, np.ndarray], cand_ego_bkt3: np.ndarray) -> np.ndarray:
        """Context batch [B,...] + WM / edge arrays + ego-frame candidates [B,K,horizon,3] at the
        training dt -> [B,K]."""
        from .dataset import Bank, edge_batch, wm_batch
        to = lambda v: torch.from_numpy(np.ascontiguousarray(v)).to(self.device)  # noqa: E731
        ctx = {k: to(context[k]) for k in CONTEXT_KEYS}
        traj = to(np.asarray(cand_ego_bkt3, np.float32))
        feats = make_candidate_features_torch(traj, **self.feat)
        view = Bank(k=0, groups=context, cands={}, scenes={}, files=[])
        idx = np.arange(len(context["goal"]))
        wm = {k: to(v) for k, v in wm_batch(view, idx).items()}
        edges = {k: to(v) for k, v in edge_batch(view, idx).items()}
        return self.model.score_groups(ctx, feats, traj, wm=wm, edges=edges,
                                       ego_size=to(context["ego_size"])).float().cpu().numpy()

    def score(self, sim_state, timestep: int, goal_xy_b2, pop_world_bkt5) -> np.ndarray:
        batch, origin, heading, valid = self.contexts(sim_state, timestep, goal_xy_b2)
        pop = np.asarray(pop_world_bkt5, dtype=np.float64)
        ego = np.stack([pool_world_to_ego(pop[w], origin[w], heading[w], self.cfg) for w in range(len(pop))])
        scores = self.score_ego(batch, ego)
        scores[~valid] = 0.0
        return scores


def main() -> None:
    import jax

    from es_discriminator.analyze_es_decisions import load_batch_state
    from es_discriminator.label_bank import _to_world
    from .check_alignment import replaced_state
    from .dataset import DEFAULT_EDGE_DIR, DEFAULT_WM_DIR, load_bank

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--tfrecord_dir", required=True)
    p.add_argument("--wm_dir", default=DEFAULT_WM_DIR)
    p.add_argument("--edge_dir", default=DEFAULT_EDGE_DIR)
    p.add_argument("--max_files", type=int, default=1)
    p.add_argument("--tol", type=float, default=1e-3)
    a = p.parse_args()
    scorer = DiscriminatorScorer(a.checkpoint, device="cpu")
    bank = load_bank(a.data_dir, max_files=a.max_files, cfg=scorer.cfg, verbose=False,
                     wm_dir=a.wm_dir, edge_dir=a.edge_dir)
    keys = CONTEXT_KEYS + (*WM_KEYS, *EDGE_KEYS, "ego_size")
    worst, n = 0.0, 0
    g0 = 0
    for f in bank.files:
        d = dict(np.load(f))
        rs = jax.device_get(replaced_state(load_batch_state(d, a.tfrecord_dir), d))
        ng = len(d["group_t"])
        for t in np.unique(d["group_t"]):
            gl = np.flatnonzero((d["group_t"] == t) & d["group_valid"].astype(bool))
            if len(gl) == 0:
                continue
            nw = len(d["scen_id"])
            pops = np.zeros((nw, bank.k, d["cand_traj"].shape[1] + 1, 5))
            for g in gl:
                w = int(d["group_scen"][g])
                rows = np.flatnonzero(d["cand_group"] == g)
                o, hd = d["group_origin"][g], float(d["group_heading"][g])
                pops[w, :, 0, :3] = [o[0], o[1], hd]
                pops[w, :, 1:, :3] = _to_world(np.nan_to_num(d["cand_traj"][rows].astype(np.float64)), o, hd)
            online = scorer.score(rs, int(t), d["scen_goal_world"], pops)
            gidx = g0 + gl
            offline = scorer.score_ego({k: bank.groups[k][gidx] for k in keys},
                                       bank.cands["traj"].reshape(-1, bank.k, *bank.cands["traj"].shape[1:])[gidx])
            on = online[d["group_scen"][gl].astype(int)]
            worst = max(worst, float(np.abs(on - offline).max()))
            n += len(gl)
        g0 += ng
    print(json.dumps({"groups": n, "variant": scorer.variant, "max_abs_score_diff_online_vs_offline": worst,
                      "tol": a.tol}))
    if not worst <= a.tol:
        raise SystemExit("deploy consistency check FAILED")


if __name__ == "__main__":
    main()
