"""CPU unit tests on a synthetic WOMD-like payload (no waymax / data needed).

  python -m unittest discover -s anchor_wm/tests -t . -v
"""

from __future__ import annotations

import importlib
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from anchor_wm.anchors import kmeans
from anchor_wm.candidates import NUM_PERTURB, build_candidates, candidate_headings, label_candidates, perturb_gt
from anchor_wm.config import AnchorWMConfig
from anchor_wm.data import build_arrays
from anchor_wm.geometry import box_corners, edge_signed_distance, sat_gap
from anchor_wm.losses import score_loss, wm_loss
from anchor_wm.model import ENCODER_PREFIXES, AnchorWorldModel, select_modes, warm_start_encoder

REPO = Path(__file__).resolve().parents[2]
T_ALL = 91


def tiny_cfg(**kw) -> AnchorWMConfig:
    base = dict(hidden_dim=32, num_heads=4, num_scene_layers=1, max_agents=8, max_map_segments=6,
                max_points_per_segment=8, num_anchors=8, num_out_modes=3, scorer_hidden=16, scorer_heads=4,
                scorer_agents=4, scorer_modes=3, num_label_edges=64, num_edge_tokens=16, dropout=0.0)
    base.update(kw)
    return AnchorWMConfig(**base)


def synthetic_payload(s: int = 2, n: int = 8, seed: int = 0) -> dict:
    """Agents driving along +x on a straight road (edges at y = +-6 m); object 0 is the SDC."""
    rng = np.random.default_rng(seed)
    t = np.arange(T_ALL) * 0.1
    speed = rng.uniform(2, 12, (s, n))
    y0 = rng.choice([-3.5, 3.5], (s, n))           # neighbours in adjacent lanes: the log is collision-free
    x0 = rng.uniform(-30, 30, (s, n))
    x0[:, 0], y0[:, 0] = 0.0, 0.0
    x = x0[..., None] + speed[..., None] * t
    y = np.broadcast_to(y0[..., None], x.shape).copy()
    types_ = np.ones((s, n), np.int32)
    types_[:, -1] = 2
    is_sdc = np.zeros((s, n), bool)
    is_sdc[:, 0] = True
    valid = np.ones((s, n, T_ALL), bool)
    valid[:, -2, 60:] = False
    # Road edges: right edge along +x at y = -6, left edge along -x at y = +6 (road on the left of travel).
    px = np.arange(-60, 200, 1.0)
    rx = np.concatenate([px, px[::-1]])
    ry = np.concatenate([np.full_like(px, -6.0), np.full_like(px, 6.0)])
    dx = np.concatenate([np.ones_like(px), -np.ones_like(px)])
    ids = np.concatenate([np.zeros_like(px), np.ones_like(px)]).astype(np.int32)
    lane_x = px
    rgx = np.concatenate([rx, lane_x])
    rgy = np.concatenate([ry, np.zeros_like(lane_x)])
    rgdx = np.concatenate([dx, np.ones_like(lane_x)])
    rgt = np.concatenate([np.full(len(rx), 15), np.full(len(lane_x), 2)]).astype(np.int32)
    rgid = np.concatenate([ids, np.full(len(lane_x), 2)]).astype(np.int32)
    tile = lambda a: np.broadcast_to(a, (s,) + a.shape).copy()  # noqa: E731
    return {
        "object_metadata_is_sdc": is_sdc, "object_metadata_object_types": types_,
        "log_trajectory_x": x.astype(np.float32), "log_trajectory_y": y.astype(np.float32),
        "log_trajectory_yaw": np.zeros((s, n, T_ALL), np.float32), "log_trajectory_valid": valid,
        "log_trajectory_vel_x": np.broadcast_to(speed[..., None], x.shape).astype(np.float32),
        "log_trajectory_vel_y": np.zeros((s, n, T_ALL), np.float32),
        "log_trajectory_length": np.full((s, n, T_ALL), 4.5, np.float32),
        "log_trajectory_width": np.full((s, n, T_ALL), 2.0, np.float32),
        "log_traffic_light_state": np.zeros((s, 4, T_ALL), np.int32),
        "log_traffic_light_x": np.zeros((s, 4, T_ALL), np.float32),
        "log_traffic_light_y": np.zeros((s, 4, T_ALL), np.float32),
        "log_traffic_light_valid": np.zeros((s, 4, T_ALL), bool),
        "roadgraph_points_x": tile(rgx.astype(np.float32)), "roadgraph_points_y": tile(rgy.astype(np.float32)),
        "roadgraph_points_dir_x": tile(rgdx.astype(np.float32)),
        "roadgraph_points_dir_y": tile(np.zeros_like(rgx, dtype=np.float32)),
        "roadgraph_points_types": tile(rgt), "roadgraph_points_ids": tile(rgid),
        "roadgraph_points_valid": tile(np.ones(len(rgx), bool)),
    }


def batch(cfg, anchor=30, **kw) -> dict:
    arrays = build_arrays(synthetic_payload(**kw), anchor_step=anchor, cfg=cfg)
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in arrays.items()}


def random_anchors(cfg, seed=0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(1, cfg.future_steps + 1) * cfg.future_dt
    speed = torch.rand(3, cfg.num_anchors, 1, generator=g) * 15
    lat = (torch.rand(3, cfg.num_anchors, 1, generator=g) - 0.5) * 0.4
    return torch.stack([speed * t, lat * t ** 2], -1)


class GeometryTest(unittest.TestCase):
    def test_sat_gap(self):
        lw = torch.tensor([2.0, 2.0])
        a = box_corners(torch.tensor([0.0, 0.0]), torch.tensor(0.0), lw)
        b = box_corners(torch.tensor([3.0, 0.0]), torch.tensor(0.0), lw)
        c = box_corners(torch.tensor([1.5, 0.5]), torch.tensor(0.7), lw)
        self.assertAlmostEqual(float(sat_gap(a, torch.tensor(0.0), b, torch.tensor(0.0))), 1.0, places=5)
        self.assertLess(float(sat_gap(a, torch.tensor(0.0), c, torch.tensor(0.7))), 0.0)

    def test_signed_distance_sign_convention(self):
        # Edge along +x at y = 0: Waymax's cross product makes the right side (y < 0) offroad (> 0).
        xs = torch.arange(-10, 11, 1.0)
        edges = {"edge_xy": torch.stack([xs, torch.zeros_like(xs)], -1)[None],
                 "edge_dir": torch.tensor([1.0, 0.0]).expand(1, len(xs), 2),
                 "edge_prev_dir": torch.tensor([1.0, 0.0]).expand(1, len(xs), 2),
                 "edge_same_prev": torch.ones(1, len(xs), dtype=torch.bool),
                 "edge_valid": torch.ones(1, len(xs), dtype=torch.bool)}
        sd = edge_signed_distance(torch.tensor([[[0.2, -1.0], [0.2, 1.0]]]), edges)
        self.assertAlmostEqual(float(sd[0, 0]), math.hypot(0.2, 1.0), places=4)
        self.assertLess(float(sd[0, 1]), 0.0)

    def test_prior_point_rule(self):
        # Convex corner: edge goes +x then turns +y at (0, 0). A point at (1, -1) lies right of both
        # tangents -> offroad; a point at (-1, 1) is left of both -> on road.
        pts = torch.tensor([[-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        dirs = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        prev = torch.cat([dirs[:1], dirs[:-1]])
        edges = {"edge_xy": pts[None], "edge_dir": dirs[None], "edge_prev_dir": prev[None],
                 "edge_same_prev": torch.tensor([[False, True, True, True, True]]),
                 "edge_valid": torch.ones(1, 5, dtype=torch.bool)}
        sd = edge_signed_distance(torch.tensor([[[0.3, -0.3], [-0.3, 0.3]]]), edges)
        self.assertGreater(float(sd[0, 0]), 0.0)
        self.assertLess(float(sd[0, 1]), 0.0)


class DataTest(unittest.TestCase):
    def test_build_arrays(self):
        cfg = tiny_cfg()
        f = batch(cfg)
        s, n, t = 2, 8, cfg.future_steps
        self.assertEqual(tuple(f["agent_history"].shape), (s, n, cfg.history_steps, cfg.agent_feature_dim))
        self.assertEqual(tuple(f["fut_xy"].shape), (s, n, t, 2))
        self.assertTrue(bool(f["target_valid_all"][:, 0].all()))        # SDC targets kept ...
        self.assertFalse(bool(f["target_valid"][:, 0].any()))           # ... but not in MF's mask
        self.assertFalse(bool(f["fut_valid"][:, 0].any()))
        torch.testing.assert_close(f["goal_xy"][:, 1], torch.zeros(s), atol=1e-4, rtol=0)
        self.assertGreater(int(f["edge_valid"].sum()), 0)
        self.assertGreater(int(f["tok_edge_valid"].sum()), 0)


class ModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = tiny_cfg()
        self.model = AnchorWorldModel(self.cfg, random_anchors(self.cfg))
        self.f = batch(self.cfg)

    def _score(self, scale):
        enc, dec = self.model(self.f)
        cands = build_candidates(self.f, dec, self.cfg.max_range, self.cfg.future_dt)
        labels = label_candidates(self.f, cands["xy"], candidate_headings(cands["xy"], cands["yaw_fallback"]),
                                  self.cfg.future_dt)
        ctx = self.model.scorer.context(enc, dec, self.f, scale)
        total, sub = self.model.scorer(ctx, cands["xy"], cands["yaw_fallback"], chunk=7)
        return dec, cands, labels, total, sub

    def test_forward_losses_backward(self):
        enc, dec = self.model(self.f)
        k, t = self.cfg.num_anchors, self.cfg.future_steps
        self.assertEqual(tuple(dec["pred_xy_k"].shape), (2, 8, k, t, 2))
        sel = select_modes(dec, self.cfg.num_out_modes, self.cfg.nms_dist_m, self.cfg.max_range)
        self.assertEqual(tuple(sel.shape), (2, 8, self.cfg.num_out_modes))
        self.assertTrue(all(len(set(r.tolist())) == len(r) for r in sel.reshape(-1, sel.shape[-1])))
        loss, m = wm_loss(dec, self.f, self.cfg)
        self.assertTrue(math.isfinite(float(loss.detach())) and math.isfinite(m["min_ade_6"]))
        dec, cands, labels, total, sub = self._score(1.0)
        c = k + 1 + NUM_PERTURB
        self.assertEqual(tuple(total.shape), (2, c))
        self.assertEqual(tuple(sub.shape), (2, c, 4))
        gt = int((cands["kinds"] == 1).nonzero()[0])
        self.assertFalse(bool(labels["overlap"][:, gt].any()))          # the log does not collide with itself
        self.assertFalse(bool(labels["offroad"][:, gt].any()))
        l_s, ms = score_loss(total, sub, labels, cands["valid"])
        (loss + l_s).backward()
        self.assertTrue(math.isfinite(float(l_s.detach())))
        # Some decoded anchors / perturbations are unsafe, so the ranking loss has safety pairs.
        self.assertTrue(bool((labels["overlap"] | labels["offroad"]).any()))

    def test_initial_predictions_are_anchors(self):
        _, dec = self.model(self.f)
        diff = (dec["pred_xy_agent_k"] - dec["anchors_agent"]).abs().max() * self.cfg.max_range
        self.assertLess(float(diff), 1.0)

    def test_grad_scale_controls_joint_gradient(self):
        for scale, expect in ((0.0, False), (1.0, True)):
            self.model.zero_grad(set_to_none=True)
            _, cands, labels, total, sub = self._score(scale)
            l_s, _ = score_loss(total, sub, labels, cands["valid"])
            l_s.backward()
            enc_grad = sum(float(p.grad.abs().sum()) for n, p in self.model.named_parameters()
                           if n.startswith(ENCODER_PREFIXES) and p.grad is not None)
            dec_grad = self.model.traj_head.net[-1].weight.grad
            dec_grad = 0.0 if dec_grad is None else float(dec_grad.abs().sum())
            self.assertEqual(enc_grad > 0, expect, f"encoder grad at scale {scale}: {enc_grad}")
            self.assertEqual(dec_grad > 0, expect, f"decoder grad at scale {scale}: {dec_grad}")
            self.assertGreater(float(self.model.scorer.total_head.weight.grad.abs().sum()), 0)

    def test_offroad_label_on_synthetic_road(self):
        f = self.f
        t = torch.arange(1, self.cfg.future_steps + 1) * self.cfg.future_dt
        on = torch.stack([10 * t, torch.zeros_like(t)], -1)
        off = torch.stack([10 * t, -1.5 * t ** 2], -1)                   # drifts right past y = -6
        xy = torch.stack([on, off])[None].expand(2, 2, -1, -1)
        lab = label_candidates(f, xy, candidate_headings(xy, torch.zeros(xy.shape[:-1])), self.cfg.future_dt)
        self.assertFalse(bool(lab["offroad"][:, 0].any()))
        self.assertTrue(bool(lab["offroad"][:, 1].all()))


class MidpointLabelTest(unittest.TestCase):
    def test_overlap_between_steps(self):
        # Ego drives +x at 10 m/s; a crossing car passes x = 3 m (only between the 0.2 s and 0.4 s
        # samples, where the ego is at x = 3). Both grid samples are clear; the midpoint overlaps.
        cfg = tiny_cfg()
        f = batch(cfg)
        t = torch.arange(1, cfg.future_steps + 1) * cfg.future_dt
        ego = torch.stack([10 * t, torch.zeros_like(t)], -1)[None, None].expand(2, 1, -1, -1).contiguous()
        cross_y = 60.0 * (t - 0.3)                                    # y = -6 at 0.2 s, +6 at 0.4 s
        f = dict(f)
        f["fut_xy"] = torch.zeros_like(f["fut_xy"])
        f["fut_xy"][:, 1, :, 0] = 3.0
        f["fut_xy"][:, 1, :, 1] = cross_y
        f["fut_yaw"] = torch.zeros_like(f["fut_yaw"])
        f["fut_yaw"][:, 1] = math.pi / 2
        f["fut_valid"] = torch.zeros_like(f["fut_valid"])
        f["fut_valid"][:, 1] = True
        yaw = torch.zeros(ego.shape[:-1])
        coarse = label_candidates(f, ego, yaw, cfg.future_dt, midpoints=False)
        fine = label_candidates(f, ego, yaw, cfg.future_dt, midpoints=True)
        self.assertFalse(bool(coarse["overlap"].any()))
        self.assertTrue(bool(fine["overlap"].all()))


class WarmStartTest(unittest.TestCase):
    def test_loads_mf_encoder(self):
        alias = "_anchor_wm_mf_test"
        pkg = types.ModuleType(alias)
        pkg.__path__ = [str(REPO / "multiFuture_GaussianWorldModel" / "model" / "world_model")]
        sys.modules[alias] = pkg
        mf_cfg_mod = importlib.import_module(f"{alias}.config")
        mf_net = importlib.import_module(f"{alias}.network")
        cfg = tiny_cfg()
        mf_cfg = mf_cfg_mod.WorldModelConfig(hidden_dim=32, num_heads=4, num_scene_layers=1, max_agents=8,
                                             max_map_segments=6, max_points_per_segment=8, decoder_type="direct",
                                             agent_frame_decoding=True)
        mf = mf_net.LatentMultiAgentWorldModel(mf_cfg)
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "mf.pt")
            torch.save({"model_state_dict": mf.state_dict(), "config": mf_cfg.to_dict(), "epoch": 1}, path)
            model = AnchorWorldModel(cfg, random_anchors(cfg))
            rep = warm_start_encoder(model, path)
        self.assertGreater(rep["tensors"], 10)
        own = model.state_dict()
        for k, v in mf.state_dict().items():
            if k.startswith(ENCODER_PREFIXES):
                torch.testing.assert_close(own[k], v)
        # Same encoder output as MF's latent for the same inputs.
        f = batch(cfg)
        model.eval(), mf.eval()
        with torch.no_grad():
            torch.testing.assert_close(model.encode(f)["latent_agent"], mf(f)["latent_scene"], atol=1e-5, rtol=1e-4)


class CandidateTest(unittest.TestCase):
    def test_perturbations(self):
        t = torch.arange(1, 26) * 0.2
        gt = torch.stack([10 * t, torch.zeros_like(t)], -1)[None]
        pert = perturb_gt(gt, torch.tensor([10.0]), torch.tensor([1.0]), 0.2)
        self.assertEqual(tuple(pert.shape), (1, NUM_PERTURB, 25, 2))
        self.assertAlmostEqual(float(pert[0, 0, -1, 1]), -3.0, places=3)     # lateral -3 m after the ramp
        self.assertAlmostEqual(float(pert[0, 6, -1, 0]), 25.0, places=3)     # speed x0.5
        stop = pert[0, 9]                                                    # 2.5 m/s^2 stop from 10 m/s
        self.assertAlmostEqual(float(stop[-1, 0]), 20.0, places=3)


class AnchorFitTest(unittest.TestCase):
    def test_kmeans_recovers_clusters(self):
        g = torch.Generator().manual_seed(0)
        centers = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        x = torch.cat([c + 0.1 * torch.randn(200, 2, generator=g) for c in centers])
        assign = kmeans(x, 3, 20, 0)
        for i in range(3):
            self.assertEqual(len(set(assign[i * 200:(i + 1) * 200].tolist())), 1)


if __name__ == "__main__":
    unittest.main()
