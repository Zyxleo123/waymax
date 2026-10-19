"""Closed-loop adapters against the model run directly (needs jax; skipped otherwise).

  python -m unittest discover -s anchor_wm/tests -t . -v
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

try:
    import jax  # noqa: F401
    HAVE_JAX = True
except ImportError:
    HAVE_JAX = False

from anchor_wm.data import build_arrays, rotate_xy_np
from anchor_wm.model import AnchorWorldModel, gather_k, select_modes
from anchor_wm.tests.test_anchor_wm import random_anchors, synthetic_payload, tiny_cfg

T_NOW = 30


def fake_sim_state(payload: dict) -> SimpleNamespace:
    g = lambda prefix, names: SimpleNamespace(**{n: payload[f"{prefix}{n}"] for n in names})  # noqa: E731
    return SimpleNamespace(
        log_trajectory=g("log_trajectory_", ("x", "y", "yaw", "valid", "vel_x", "vel_y", "length", "width")),
        object_metadata=g("object_metadata_", ("is_sdc", "object_types")),
        roadgraph_points=g("roadgraph_points_", ("x", "y", "dir_x", "dir_y", "types", "ids", "valid")),
        log_traffic_light=g("log_traffic_light_", ("state", "x", "y", "valid")),
    )


@unittest.skipUnless(HAVE_JAX, "jax not installed")
class ClosedLoopAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = tiny_cfg()
        self.model = AnchorWorldModel(self.cfg, random_anchors(self.cfg)).eval()
        self.payload = synthetic_payload(s=2, n=8)
        self.state = fake_sim_state(self.payload)
        arrays = build_arrays(self.payload, anchor_step=T_NOW, cfg=self.cfg, with_targets=False)
        self.origin, self.heading = arrays["origin_xy"], arrays["origin_yaw"]
        self.f = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in arrays.items()
                  if k not in ("origin_xy", "origin_yaw")}

    def test_forecaster_matches_model(self):
        from anchor_wm.closed_loop import AnchorForecaster
        fc = AnchorForecaster(self.model)
        h = 51
        modes = fc.forecast_modes(self.state, timestep=T_NOW, traj_length=h, world_idx=1, dt=0.1)
        traj_k, probs, masks = modes["traj_k"], modes["mode_probs"], modes["masks"]
        n_total = self.payload["log_trajectory_x"].shape[1]
        self.assertEqual(traj_k.shape, (n_total, self.cfg.num_out_modes, h, 5))
        with torch.no_grad():
            _, dec = self.model(self.f)
        sel = select_modes(dec, self.cfg.num_out_modes, self.cfg.nms_dist_m, self.cfg.max_range)
        delta = gather_k(dec["pred_xy_k"], sel)[1].numpy() * self.cfg.max_range              # [n,6,T,2] ego frame
        cur = np.stack([self.payload["log_trajectory_x"][1, :, T_NOW], self.payload["log_trajectory_y"][1, :, T_NOW]], -1)
        world = cur[:, None, None] + rotate_xy_np(delta, -self.heading[1])
        agent = 1                                                       # a valid, dynamic, non-SDC vehicle
        np.testing.assert_allclose(traj_k[agent, :, 0, :2], np.repeat(cur[agent][None], self.cfg.num_out_modes, 0),
                                   atol=1e-4)
        np.testing.assert_allclose(traj_k[agent, :, 2::2, :2], world[agent].astype(np.float32), atol=1e-3)
        self.assertTrue(np.allclose(probs[agent].sum(), 1.0, atol=1e-5))
        self.assertFalse(masks[0].any())                                # the SDC is never an obstacle
        # Interpolated odd steps lie halfway between neighbouring model steps.
        np.testing.assert_allclose(traj_k[agent, :, 3, :2], 0.5 * (traj_k[agent, :, 2, :2] + traj_k[agent, :, 4, :2]),
                                   atol=1e-3)
        # MF's independent-agent collision scoring runs on the packed futures.
        ego = np.zeros((4, h, 5), np.float32)
        ego[..., 3:] = [4.5, 2.0]
        ego[..., :2] = cur[0]
        fc.forecast(self.state, timestep=T_NOW, traj_length=h, world_idx=1, dt=0.1)
        score, debug = fc.score_independent_mode_collision(ego, world_idx=1)
        self.assertEqual(score.shape, (4,))
        self.assertTrue(np.all((score >= 0) & (score <= 1)))

    def test_learned_scorer_matches_direct_scoring(self):
        from anchor_wm.closed_loop import AnchorLearnedScorer
        sc = AnchorLearnedScorer("", model=self.model)
        k, t = 5, self.cfg.future_steps
        rng = np.random.default_rng(0)
        tt = np.arange(1, t + 1) * self.cfg.future_dt
        cand = np.stack([rng.uniform(3, 12, (2, k, 1)) * tt, rng.uniform(-0.3, 0.3, (2, k, 1)) * tt ** 2], -1)
        yaw = np.arctan2(np.gradient(cand[..., 1], axis=-1), np.gradient(cand[..., 0], axis=-1))
        # World population at 0.1 s: step 0 = current ego pose, odd steps interpolated.
        steps = 2 * t + 1
        pop = np.zeros((2, k, steps, 5))
        local = np.concatenate([np.zeros((2, k, 1, 2)), cand], 2)                         # [B,K,T+1,2]
        lyaw = np.concatenate([np.zeros((2, k, 1)), yaw], 2)
        src = np.arange(t + 1) * 2
        for b in range(2):
            for i in range(k):
                for d in range(2):
                    pop[b, i, :, d] = np.interp(np.arange(steps), src, local[b, i, :, d])
                pop[b, i, :, 2] = np.interp(np.arange(steps), src, lyaw[b, i])
        pop[..., :2] = rotate_xy_np(pop[..., :2], -self.heading[:, None, None]) + self.origin[:, None, None]
        pop[..., 2] = pop[..., 2] + self.heading[:, None, None]
        goal_world = rotate_xy_np(np.array([[40.0, 0.0], [40.0, 0.0]]), -self.heading) + self.origin
        got = np.asarray(sc(self.state, T_NOW, goal_world, pop))

        f = dict(self.f)
        f["goal_xy"] = torch.tensor([[40.0, 0.0], [40.0, 0.0]])
        with torch.no_grad():
            enc, dec = self.model(f)
            ctx = self.model.scorer.context(enc, dec, f, 0.0)
            want, _ = self.model.scorer(ctx, torch.from_numpy(cand).float(), torch.from_numpy(yaw).float())
        np.testing.assert_allclose(got, want.numpy(), atol=1e-4)
        self.assertLess(sc.anchor_err_m, 1e-6)
        # A second call in the same replan reuses the context; a mis-anchored population is rejected.
        sc(self.state, T_NOW, goal_world, pop)
        self.assertEqual(sc._calls, 2)
        bad = pop.copy()
        bad[:, :, 0, 0] += 1.0
        sc2 = AnchorLearnedScorer("", model=self.model)
        with self.assertRaises(RuntimeError):
            sc2(self.state, T_NOW, goal_world, bad)

        # Hybrid: learned + lambda * sum log(clip(rule factor)), factors from the rule scorers' components.
        class FakeRule:
            def __init__(self, coll, off):
                self.coll, self.off = coll, off
            def compute_score(self, trajectories, sim_state, timestep, world_idx):
                self.last_components = {"collision": self.coll, "offroad": self.off}
        coll = np.array([1.0, 0.5, 0.0, 1.0, 0.9])
        off = np.array([1.0, 1.0, 1.0, 0.0, 1.0])
        sc3 = AnchorLearnedScorer("", model=self.model, rule_scorers=[FakeRule(coll, off), FakeRule(coll, off)],
                                  rule_lambda=2.0)
        hyb = np.asarray(sc3(self.state, T_NOW, goal_world, pop))
        pen = np.log(np.clip(coll, 1e-3, 1)) + np.log(np.clip(off, 1e-3, 1))
        np.testing.assert_allclose(hyb, want.numpy() + 2.0 * pen[None], atol=1e-3)


if __name__ == "__main__":
    unittest.main()
