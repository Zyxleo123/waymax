"""v2 recipe: near-agent modes, time remaining / horizon masking, horizon-aware candidates and labels,
the interaction-kernel forecasting weight and the simplified scorer loss (CPU, synthetic payload).

  python -m unittest discover -s anchor_wm/tests -t . -v
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch

from anchor_wm.candidates import build_candidates, label_candidates
from anchor_wm.losses import interaction_distance, simple_score_loss, wm_loss
from anchor_wm.model import AnchorWorldModel
from anchor_wm.tests.test_anchor_wm import batch, random_anchors, tiny_cfg
from anchor_wm.train import scorer_step

ALL_INPUTS = dict(scorer_goal_features=True, scorer_route_features=True, scorer_rule_features=True,
                  scorer_input_heading=True, scorer_aux_heads=True)


def _targs(**kw):
    a = dict(train_cands=20, score_chunk=0, progress_key="goal_gain", progress_margin=2.0, w_sub=1.0, w_pair=1.0,
             w_list=0.5, w_aux=0.5, scorer_loss="current", simple_w_safe=1.0, simple_temp=1.0)
    a.update(kw)
    return SimpleNamespace(**a)


def _scores(model, f, xy, yaw):
    with torch.no_grad():
        enc, dec = model(f)
        ctx = model.scorer.context(enc, dec, f, 0.0)
        total, sub = model.scorer(ctx, xy, yaw, heading=yaw)
    return total, sub, ctx


class NearModesTest(unittest.TestCase):
    def test_all_near_equals_uniform_modes(self):
        # Every selected agent near -> the slot layout keeps all slots and must match plain top-5 modes.
        torch.manual_seed(0)
        cfg_a = tiny_cfg(scorer_modes=5, **ALL_INPUTS)
        cfg_b = tiny_cfg(scorer_modes=3, scorer_near_agents=4, scorer_near_modes=5, **ALL_INPUTS)
        a = AnchorWorldModel(cfg_a, random_anchors(cfg_a)).eval()
        b = AnchorWorldModel(cfg_b, random_anchors(cfg_b)).eval()
        b.load_state_dict(a.state_dict())
        f = batch(cfg_a)
        _, dec = a(f)
        cands = build_candidates(f, dec, cfg_a.max_range, cfg_a.future_dt)
        ta, _, _ = _scores(a, f, cands["xy"], cands["yaw"])
        tb, _, _ = _scores(b, f, cands["xy"], cands["yaw"])
        torch.testing.assert_close(ta, tb, atol=1e-5, rtol=1e-5)

    def test_far_agents_keep_fewer_modes(self):
        torch.manual_seed(0)
        cfg = tiny_cfg(scorer_modes=3, scorer_near_agents=2, scorer_near_modes=5, **ALL_INPUTS)
        model = AnchorWorldModel(cfg, random_anchors(cfg)).eval()
        f = batch(cfg)
        _, dec = model(f)
        cands = build_candidates(f, dec, cfg.max_range, cfg.future_dt)
        total, sub, ctx = _scores(model, f, cands["xy"], cands["yaw"])
        self.assertTrue(bool(torch.isfinite(total).all() and torch.isfinite(sub).all()))
        p = ctx["wm"]["probs"]                                                  # [B,A,5]
        self.assertEqual(float(p[:, 2:, 3:].abs().sum()), 0.0)                  # far agents: top 3 only
        self.assertEqual(ctx["wm_tok"].shape[2], 2 * 5 + 2 * 3)
        valid = ctx["wm"]["valid"]
        torch.testing.assert_close(p.sum(-1)[valid], torch.ones_like(p.sum(-1)[valid]))


class TimeRemainingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = tiny_cfg(scorer_time_remaining=True, **ALL_INPUTS)
        self.model = AnchorWorldModel(self.cfg, random_anchors(self.cfg)).eval()
        for p in self.model.scorer.parameters():        # make the zero-initialised additions matter
            if p.dim() >= 2:
                torch.nn.init.normal_(p, std=0.05)

    def test_plan_horizon(self):
        self.assertEqual(batch(self.cfg, anchor=30)["plan_horizon"].tolist(), [25, 25])
        self.assertEqual(batch(self.cfg, anchor=70)["plan_horizon"].tolist(), [10, 10])

    def test_steps_past_horizon_do_not_change_scores(self):
        f = batch(self.cfg, anchor=70)
        _, dec = self.model(f)
        cands = build_candidates(f, dec, self.cfg.max_range, self.cfg.future_dt, horizon=f["plan_horizon"])
        xy, yaw = cands["xy"].clone(), cands["yaw"].clone()
        t0, _, _ = _scores(self.model, f, xy, yaw)
        late = xy.clone()
        late[:, :, 10:] += torch.tensor([3.0, -7.0])                            # e.g. drive offroad after the end
        t1, _, _ = _scores(self.model, f, late, yaw)
        torch.testing.assert_close(t0, t1, atol=1e-5, rtol=1e-5)
        early = xy.clone()
        early[:, :, 5:] += torch.tensor([3.0, -7.0])
        t2, _, _ = _scores(self.model, f, early, yaw)
        self.assertGreater(float((t0 - t2).abs().max()), 1e-4)

    def test_candidates_and_labels_within_horizon(self):
        f = batch(self.cfg, anchor=80)                                          # 5 plan steps left, log ends
        self.assertFalse(bool(f["ego_future_valid"].all()))
        _, dec = self.model(f)
        hz = f["plan_horizon"]
        gt = self.cfg.num_anchors                                               # anchors first, then the log
        self.assertFalse(bool(build_candidates(f, dec, self.cfg.max_range, self.cfg.future_dt)["valid"][:, gt].any()))
        cands = build_candidates(f, dec, self.cfg.max_range, self.cfg.future_dt, horizon=hz)
        self.assertTrue(bool(cands["valid"][:, gt].all()))
        lab = label_candidates(f, cands["xy"], cands["yaw"], self.cfg.future_dt, horizon=hz)
        self.assertFalse(bool((lab["overlap"] | lab["offroad"])[:, gt].any()))
        self.assertTrue(bool((lab["first_overlap"][:, gt] == self.cfg.future_steps + 1).all()))
        self.assertEqual(lab["horizon"].tolist(), hz.tolist())
        torch.testing.assert_close(lab["goal_dist"][:, gt], torch.zeros(2), atol=0.5, rtol=0)   # log ends at the goal

    def test_scorer_step_both_losses(self):
        f = batch(self.cfg, anchor=70)
        for loss_name in ("current", "simple"):
            self.model.train()
            self.model.zero_grad(set_to_none=True)
            enc, dec = self.model(f)
            dec = {k: v.float() for k, v in dec.items()}
            loss, m = scorer_step(self.model, enc, dec, f, _targs(scorer_loss=loss_name), self.cfg, 1.0,
                                  torch.Generator().manual_seed(0))
            loss.backward()
            self.assertTrue(math.isfinite(float(loss)), loss_name)
            self.assertIn("score_aux", m)
            self.assertGreater(float(self.model.scorer.total_head.weight.grad.abs().sum()), 0.0)
            self.assertGreater(float(self.model.scorer.time_end[-1].weight.grad.abs().sum()), 0.0)


class SimpleLossTest(unittest.TestCase):
    def test_tiers(self):
        # One group: safe far from goal, safe at goal, unsafe late, unsafe early. Utility order: 1 > 0 > 2 > 3.
        labels = {"overlap": torch.tensor([[False, False, True, True]]), "offroad": torch.zeros(1, 4, dtype=torch.bool),
                  "first_overlap": torch.tensor([[26.0, 26.0, 20.0, 2.0]]), "first_offroad": torch.full((1, 4), 26.0),
                  "clearance": torch.tensor([[2.0, 2.0, -0.2, -0.2]]), "offroad_sd": torch.full((1, 4), -1.0),
                  "goal_gain": torch.tensor([[20.0, 30.0, 30.0, 30.0]]), "goal_dist": torch.tensor([[10.0, 0.1, 0.1, 0.1]]),
                  "horizon": torch.tensor([25])}
        valid = torch.ones(1, 4, dtype=torch.bool)
        good = torch.tensor([[1.0, 2.0, 0.0, -1.0]], requires_grad=True)
        bad = torch.tensor([[-1.0, 0.0, 2.0, 1.0]], requires_grad=True)
        sub = torch.zeros(1, 4, 4)
        l_good, m = simple_score_loss(good, sub, labels, valid)
        l_bad, _ = simple_score_loss(bad, sub, labels, valid)
        self.assertLess(float(l_good), float(l_bad))
        self.assertEqual(m["picked_unsafe"], 0.0)
        l_bad.backward()
        self.assertTrue(bool(torch.isfinite(bad.grad).all()))


class InteractionWeightTest(unittest.TestCase):
    def test_kernel(self):
        torch.manual_seed(0)
        cfg = tiny_cfg(interaction_weight="kernel")
        model = AnchorWorldModel(cfg, random_anchors(cfg))
        f = batch(cfg)
        d = interaction_distance(f, cfg)
        self.assertEqual(tuple(d.shape), (2, 8))
        torch.testing.assert_close(d[:, 0], torch.zeros(2), atol=1e-4, rtol=0)  # the SDC meets its own log
        self.assertTrue(bool(torch.isfinite(d[f["agent_valid"]]).all()))
        _, dec = model(f)
        loss, m = wm_loss(dec, f, cfg)
        loss.backward()
        self.assertTrue(math.isfinite(float(loss)))
        self.assertGreaterEqual(m["agent_weight_mean"], 1.0)
        self.assertTrue(math.isfinite(m["sigma_median_m"]))


def _ego_frame_geometry(cov_ego, d, cand_yaw):
    """The earlier _wm_geometry covariance features, formed from the ego-frame (sxx, syy, sxy)."""
    from anchor_wm.model import _rotate_cov
    cov = cov_ego + torch.tensor([0.05, 0.05, 0.0])
    det = (cov[..., 0] * cov[..., 1] - cov[..., 2] ** 2).clamp(min=1e-6)
    maha = (cov[..., 1] * d[..., 0] ** 2 - 2 * cov[..., 2] * d[..., 0] * d[..., 1] + cov[..., 0] * d[..., 1] ** 2) / det
    cand_cov = _rotate_cov(cov, cand_yaw)
    return torch.stack([torch.log1p(maha), 0.5 * torch.log(cand_cov[..., 0].clamp(min=1e-4)),
                        0.5 * torch.log(cand_cov[..., 1].clamp(min=1e-4)), 0.25 * torch.log(det).expand_as(maha)], -1)


class WmGeometryTest(unittest.TestCase):
    """The Mahalanobis / determinant / candidate-variance features (last 4 geometry channels)."""

    def _geometry(self, sigma, cov_yaw, agent_xy, cand_xy, cand_yaw):
        from anchor_wm.model import _rotate_cov
        model = AnchorWorldModel(tiny_cfg(), random_anchors(tiny_cfg()))
        b, a, m, t = sigma.shape[:4]
        var = sigma.square()
        cov = _rotate_cov(torch.stack([var[..., 0], var[..., 1], torch.zeros_like(var[..., 0])], -1),
                          -cov_yaw[:, :, None, None])
        wm = {"xy": agent_xy, "yaw": torch.zeros(b, a, m, t), "size": torch.ones(b, a, 2), "cov": cov,
              "sigma": sigma, "cov_yaw": cov_yaw, "slots": None}
        geo = model.scorer._wm_geometry(cand_xy, cand_yaw, torch.tensor([[4.5, 2.0]]).expand(b, 2), wm)[..., -4:]
        d = agent_xy.permute(0, 3, 1, 2, 4).reshape(b, 1, t, a * m, 2) - cand_xy.unsqueeze(3)
        ref = _ego_frame_geometry(cov.permute(0, 3, 1, 2, 4).reshape(b, 1, t, a * m, 3), d, cand_yaw.unsqueeze(-1))
        return geo, ref

    def test_matches_ego_frame_formula(self):
        g = torch.Generator().manual_seed(0)
        b, a, m, t, k = 2, 3, 2, 4, 5
        sigma = 0.2 + 3.0 * torch.rand(b, a, m, t, 2, generator=g)
        cov_yaw = 6.0 * torch.rand(b, a, generator=g) - 3.0
        agent_xy = 20.0 * torch.randn(b, a, m, t, 2, generator=g)
        cand_xy = 20.0 * torch.randn(b, k, t, 2, generator=g)
        cand_yaw = 6.0 * torch.rand(b, k, t, generator=g) - 3.0
        geo, ref = self._geometry(sigma, cov_yaw, agent_xy, cand_xy, cand_yaw)
        torch.testing.assert_close(geo, ref, rtol=1e-3, atol=1e-3)

    def test_thin_long_gaussian_stays_finite(self):
        # nonfinite_step27744: an unused mode with sigma 0.1 m x 1300 m, 8.7 km from the candidate.
        sigma = torch.tensor([0.1008, 1300.1]).view(1, 1, 1, 1, 2)
        cov_yaw = torch.tensor([[2.1]])
        agent_xy = torch.tensor([-7288.7, -4669.3]).view(1, 1, 1, 1, 2)
        cand_xy, cand_yaw = torch.zeros(1, 1, 1, 2), torch.tensor([[[0.3]]])
        geo, ref = self._geometry(sigma, cov_yaw, agent_xy, cand_xy, cand_yaw)
        self.assertTrue(torch.isfinite(geo).all(), geo)
        self.assertGreaterEqual(float(geo[..., 0].min()), 0.0)          # log1p(maha), maha >= 0
        var = sigma.square() + 0.05
        self.assertAlmostEqual(float(geo[..., 3]), 0.25 * math.log(float(var[..., 0] * var[..., 1])), places=4)


if __name__ == "__main__":
    unittest.main()
