"""Fine-tuning: warm-start equivalence, bank losses / gradients, and a 3-step end-to-end run on a fake bank.

  python -m unittest discover -s anchor_wm/tests -t . -v
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from anchor_wm import finetune
from anchor_wm.bank_data import build_index, make_bank_loader
from anchor_wm.config import AnchorWMConfig
from anchor_wm.losses import bank_eval_metrics, bank_score_loss, bank_targets
from anchor_wm.model import RULE_END_DIM, RULE_STEP_DIM, AnchorWorldModel, rule_features
from anchor_wm.tests.test_anchor_wm import random_anchors, tiny_cfg
from anchor_wm.tests.test_bank import make_fake_bank


def _save_base(path: str, cfg: AnchorWMConfig) -> AnchorWorldModel:
    torch.manual_seed(0)
    model = AnchorWorldModel(cfg, random_anchors(cfg)).eval()
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": 7}, path)
    return model


class FinetuneTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.cfg = tiny_cfg()
        self.bank, self.cache, self.split, self.invalid = make_fake_bank(self.root, self.cfg)
        self.ckpt = os.path.join(self.root, "base.pt")
        self.base = _save_base(self.ckpt, self.cfg)
        idx, _ = build_index(self.bank, self.cache, self.split, self.invalid, ("train", "test"))
        loader = make_bank_loader(idx, self.cache, batch_groups=3, cands_per_group=0, shuffle=False, seed=0,
                                  num_workers=0)
        self.batch = next(iter(loader))

    def tearDown(self):
        self.tmp.cleanup()

    def _args(self, **kw):
        a = dict(init_ckpt=self.ckpt, no_goal_features=False, no_route_features=False, no_aux_heads=False,
                 rule_features=False, scorer_agents=self.cfg.scorer_agents, scorer_modes=self.cfg.scorer_modes)
        a.update(kw)
        return type("A", (), a)()

    def test_warm_start_scores_unchanged(self):
        model, info = finetune.build_model(self._args(), torch.device("cpu"))
        self.assertGreater(info["new_tensors"], 0)
        model.eval()
        with torch.no_grad():
            new, _, aux, _, _ = finetune.score_bank(model, self.batch, 0.0, False)
            old, _, _, _, _ = finetune.score_bank(self.base, self.batch, 0.0, False)
        torch.testing.assert_close(new, old, atol=1e-5, rtol=1e-5)          # zero-init additions
        self.assertEqual(set(aux), {"ttv", "goaldist"})

    def test_bank_loss_trains_scorer_only(self):
        model, _ = finetune.build_model(self._args(), torch.device("cpu"))
        finetune.set_trainable(model, unfreeze_decoder=False)
        finetune.train_mode(model)
        total, sub, aux, _, _ = finetune.score_bank(model, self.batch, 0.0, False)
        tg = bank_targets(self.batch)
        loss, m = bank_score_loss(total, sub, aux, tg, self.batch, w_distill=0.1)
        loss.backward()
        self.assertTrue(math.isfinite(float(loss.detach())))
        enc_grads = [p.grad for n, p in model.named_parameters() if not n.startswith("scorer.")]
        self.assertTrue(all(g is None for g in enc_grads))
        self.assertGreater(float(model.scorer.total_head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.scorer.goal_step[-1].weight.grad.abs().sum()), 0)   # new modules learn
        self.assertGreater(m["pairs_safety"], 0)
        ev = bank_eval_metrics(total.detach(), tg)
        self.assertTrue(0.0 <= ev["picked_unsafe"] <= 1.0)

    def test_rule_features_warm_start_and_grad(self):
        model, _ = finetune.build_model(self._args(rule_features=True), torch.device("cpu"))
        self.assertTrue(model.config.scorer_rule_features)
        model.eval()
        with torch.no_grad():
            new, _, _, _, _ = finetune.score_bank(model, self.batch, 0.0, False)
            old, _, _, _, _ = finetune.score_bank(self.base, self.batch, 0.0, False)
        torch.testing.assert_close(new, old, atol=1e-5, rtol=1e-5)
        finetune.set_trainable(model, unfreeze_decoder=False)
        finetune.train_mode(model)
        total, sub, aux, _, _ = finetune.score_bank(model, self.batch, 0.0, False)
        loss, _ = bank_score_loss(total, sub, aux, bank_targets(self.batch), self.batch)
        loss.backward()
        self.assertGreater(float(model.scorer.rule_step[-1].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.scorer.rule_end[-1].weight.grad.abs().sum()), 0)

    def test_input_heading(self):
        # Off (A1 / A2 checkpoints): a passed heading is ignored. On: it replaces the travel direction.
        model, _ = finetune.build_model(self._args(), torch.device("cpu"))
        model.eval()
        f = dict(self.batch)
        twisted = f["cand_yaw"] + 0.7
        with torch.no_grad():
            enc, dec = model(f)
            ctx = model.scorer.context(enc, dec, f, 0.0)
            base, _ = model.scorer(ctx, f["cand_xy"], f["cand_yaw"])
            off, _ = model.scorer(ctx, f["cand_xy"], f["cand_yaw"], heading=twisted)
            torch.testing.assert_close(off, base)
            model.scorer.cfg.scorer_input_heading = True
            on, _ = model.scorer(ctx, f["cand_xy"], f["cand_yaw"], heading=twisted)
        self.assertFalse(torch.allclose(on, base))

    def test_targets(self):
        tg = bank_targets(self.batch)
        unsafe = (self.batch["overlap_step"] > 0) | (self.batch["offroad_step"] > 0)
        torch.testing.assert_close(tg["unsafe"], unsafe)
        self.assertTrue(bool((tg["ttv"][~unsafe] == 1.0).all()))
        self.assertTrue(bool((tg["ttv"][unsafe] < 1.0).all()))
        self.assertTrue(bool((tg["soft_o"][self.batch["overlap_step"] > 0] > 0.5).all()))
        self.assertTrue(bool((tg["soft_o"][self.batch["overlap_step"] == 0] <= 0.5).all()))

    def test_end_to_end(self):
        argv = ["finetune", "--init_ckpt", self.ckpt, "--bank_dir", self.bank, "--cache_dir", self.cache,
                "--split_json", self.split, "--invalid_json", self.invalid, "--save_dir", os.path.join(self.root, "ft"),
                "--device", "cpu", "--max_steps", "3", "--batch_groups", "1", "--cands_per_group", "4",
                "--num_workers", "0", "--val_every", "3", "--val_groups", "2", "--log_every", "1",
                "--wandb_mode", "disabled", "--scorer_agents", str(self.cfg.scorer_agents),
                "--scorer_modes", str(self.cfg.scorer_modes), "--rule_features"]
        # The fake bank has train groups only in scene A; use it for validation too.
        orig = finetune.build_index
        with mock.patch.object(finetune, "build_index",
                               side_effect=lambda *a, **k: orig(*a[:4], ("train",), **k)), \
                mock.patch.object(sys, "argv", argv):
            finetune.main()
        runs = os.listdir(os.path.join(self.root, "ft"))
        files = os.listdir(os.path.join(self.root, "ft", runs[0]))
        self.assertIn("final.pt", files)
        ck = torch.load(os.path.join(self.root, "ft", runs[0], "final.pt"), weights_only=False)
        self.assertTrue(ck["config"]["scorer_goal_features"])
        self.assertTrue(ck["config"]["scorer_rule_features"])
        self.assertEqual(ck["step"], 3)
        self.assertIsNotNone(ck["opt"])
        # Resume: continues from step 3 to 5 with the saved weights and optimizer.
        final = os.path.join(self.root, "ft", runs[0], "final.pt")
        argv2 = [x if x != "3" else "5" for x in argv] + ["--resume", final, "--run_name", "ft_resumed"]
        with mock.patch.object(finetune, "build_index",
                               side_effect=lambda *a, **k: orig(*a[:4], ("train",), **k)), \
                mock.patch.object(sys, "argv", argv2):
            finetune.main()
        runs2 = sorted(os.listdir(os.path.join(self.root, "ft")))
        self.assertEqual(len(runs2), 2)
        new = [r for r in runs2 if r != runs[0]][0]
        ck2 = torch.load(os.path.join(self.root, "ft", new, "final.pt"), weights_only=False)
        self.assertEqual(ck2["step"], 5)


class RuleFeaturesTest(unittest.TestCase):
    def test_overlap_and_offroad(self):
        t = 3
        # One agent, one mode: a 4 x 2 m box parked at (5, 0).
        wm = {"xy": torch.tensor([5.0, 0.0]).expand(1, 1, 1, t, 2).clone(), "yaw": torch.zeros(1, 1, 1, t),
              "probs": torch.ones(1, 1, 1), "valid": torch.ones(1, 1, dtype=torch.bool),
              "size": torch.tensor([[[4.0, 2.0]]])}
        # Road edge along y = -10 heading +x: the road is on its left (y > -10).
        ex = torch.linspace(-50, 50, 101)
        edges = {"edge_xy": torch.stack([ex, torch.full_like(ex, -10.0)], -1)[None],
                 "edge_dir": torch.tensor([1.0, 0.0]).expand(1, 101, 2).clone(),
                 "edge_prev_dir": torch.tensor([1.0, 0.0]).expand(1, 101, 2).clone(),
                 "edge_same_prev": torch.ones(1, 101, dtype=torch.bool), "edge_valid": torch.ones(1, 101, dtype=torch.bool)}
        # Candidate 0 drives through the agent on the road; candidate 1 is clear of it but off the road.
        xs = torch.tensor([3.0, 5.0, 7.0])
        cand = torch.stack([torch.stack([xs, torch.zeros(t)], -1), torch.stack([xs, torch.full((t,), -20.0)], -1)])[None]
        step, end = rule_features(cand, torch.zeros(1, 2, t), torch.tensor([[4.5, 2.0]]), wm, edges)
        self.assertEqual(step.shape, (1, 2, t, RULE_STEP_DIM))
        self.assertEqual(end.shape, (1, 2, RULE_END_DIM))
        self.assertTrue(bool((step[0, 0, :, 0] < 0).all()))          # gap < 0: overlap
        self.assertTrue(bool((step[0, 0, :, 3] == 1).all()))         # hard collision probability 1
        self.assertTrue(bool((step[0, 0, :, 6] < 0.01).all()))       # on the road
        self.assertTrue(bool((step[0, 1, :, 3] == 0).all()))         # no collision
        self.assertTrue(bool((step[0, 1, :, 5] > 0).all()))          # signed distance > 0: offroad
        self.assertTrue(bool((step[0, 1, :, 6] > 0.99).all()))
        self.assertAlmostEqual(float(end[0, 0, 6]), 0.0)             # violation from the first step
        self.assertAlmostEqual(float(end[0, 1, 6]), 0.0)


if __name__ == "__main__":
    unittest.main()
