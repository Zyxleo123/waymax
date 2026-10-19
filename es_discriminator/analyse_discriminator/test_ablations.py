"""A0-A8 discriminator ablations: shapes, gradients, removed inputs have no effect, checkpoint kwargs.

  PYTHONPATH=. python -m unittest es_discriminator.analyse_discriminator.test_ablations -v
"""

from __future__ import annotations

import unittest

import torch

from es_discriminator.analyse_discriminator.ablations import ABLATIONS, FLAG_KEYS, ablation_of, flags
from es_discriminator.real_es.trainer import build_model
from es_discriminator.real_es.wm_model import WMTrajectoryDiscriminator

B, K, T, A, M, E = 2, 7, 25, 5, 6, 12
H = 32
MODULES = {  # flag -> modules that exist only with the flag on
    "map_tokens": ("map_embed",),
    "agent_tokens": ("agent_embed",),
    "scene_attention": ("cross_attn",),
    "edge_branch": ("edge_embed", "edge_attn"),
    "wm_branch": ("mode_embed", "mode_temporal", "mode_index", "prob_embed", "wm_attn"),
}


def _inputs(seed: int = 0):
    """Random scene context, candidates [B,K,T], WM mixture and road edges (some invalid)."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    ctx = {"agents": r(B, 4, 3, 8), "agent_mask": torch.ones(B, 4, 3, dtype=torch.bool),
           "map": r(B, 9, 5), "map_mask": torch.ones(B, 9, dtype=torch.bool),
           "lights": r(B, 2, 4), "lights_mask": torch.ones(B, 2, dtype=torch.bool),
           "goal": r(B, 2), "ego_vel": r(B, 2)}
    xy = torch.cumsum(torch.rand(B, K, T, 2, generator=g) * torch.tensor([2.0, 0.3]), dim=2)
    yaw = torch.atan2(torch.diff(xy[..., 1], dim=2, prepend=torch.zeros(B, K, 1)),
                      torch.diff(xy[..., 0], dim=2, prepend=torch.zeros(B, K, 1)))
    raw = torch.cat([xy, yaw[..., None]], -1)
    feats = torch.cat([xy / 20, torch.zeros(B, K, T, 2), torch.sin(yaw)[..., None], torch.cos(yaw)[..., None]], -1)
    sig = torch.rand(B, A, M, T, 2, generator=g) + 0.2
    cov = torch.stack([sig[..., 0] ** 2, sig[..., 1] ** 2, 0.1 * sig[..., 0] * sig[..., 1]], -1)
    probs = torch.softmax(r(B, A, M), -1)
    wm = {"mean": r(B, A, M, T, 3) * 10, "cov": cov, "probs": probs, "current": r(B, A, 3) * 10,
          "size": torch.rand(B, A, 2, generator=g) * 4 + 1, "valid": torch.ones(B, A, dtype=torch.bool)}
    wm["valid"][1, -2:] = False
    wm["probs"][1, -2:] = 0
    edges = {"xy": r(B, E, 2) * 20, "dir": torch.nn.functional.normalize(r(B, E, 2), dim=-1),
             "mask": torch.ones(B, E, dtype=torch.bool)}
    edges["mask"][0, -3:] = False
    return ctx, feats, raw, wm, edges, torch.tensor([[4.8, 2.0]] * B)


def _model(name: str, seed: int = 0) -> WMTrajectoryDiscriminator:
    torch.manual_seed(seed)
    return WMTrajectoryDiscriminator(hidden=H, heads=4, dropout=0.0, **flags(name)).eval()


def _score(model, ctx, feats, raw, wm, edges, size, chunk=0):
    with torch.no_grad():
        return model.score_groups(ctx, feats, raw, wm=wm, edges=edges, ego_size=size, chunk=chunk)


def _n_params(m) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _perturbed(ctx, wm, edges):
    """Inputs with one source replaced by noise: name -> (ctx, wm, edges, flag that must be on to see it)."""
    g = torch.Generator().manual_seed(1)
    r = lambda t: torch.randn(t.shape, generator=g) * 3  # noqa: E731
    return {
        "map": (dict(ctx, map=r(ctx["map"])), wm, edges, "map_tokens"),
        "agents": (dict(ctx, agents=r(ctx["agents"])), wm, edges, "agent_tokens"),
        "edges": (ctx, wm, dict(edges, xy=r(edges["xy"]), dir=r(edges["dir"])), "edge_branch"),
        "wm": (ctx, dict(wm, mean=r(wm["mean"]), probs=torch.softmax(r(wm["probs"]), -1)), edges, "wm_branch"),
        "lights": (dict(ctx, lights=r(ctx["lights"])), wm, edges, None),
        "goal": (dict(ctx, goal=ctx["goal"] + 2.0), wm, edges, None),
    }


class TestAblations(unittest.TestCase):
    def test_shapes_chunking_and_encode_once(self):
        ctx, feats, raw, wm, edges, size = _inputs()
        for name in ABLATIONS:
            with self.subTest(name=name):
                m = _model(name)
                s = _score(m, ctx, feats, raw, wm, edges, size)
                self.assertEqual(tuple(s.shape), (B, K))
                self.assertTrue(torch.isfinite(s).all())
                torch.testing.assert_close(s, _score(m, ctx, feats, raw, wm, edges, size, chunk=3))
                with torch.no_grad():
                    enc = m.encode_context(ctx, wm, edges, size)
                    torch.testing.assert_close(s, m.score_encoded(enc, feats, raw))

    def test_every_parameter_gets_gradient(self):
        ctx, feats, raw, wm, edges, size = _inputs()
        for name in ABLATIONS:
            with self.subTest(name=name):
                m = _model(name).train()
                m.score_groups(ctx, feats, raw, wm=wm, edges=edges, ego_size=size).sum().backward()
                missing = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
                self.assertEqual(missing, [])

    def test_removed_modules_and_head_width(self):
        for name, f in ABLATIONS.items():
            with self.subTest(name=name):
                m = _model(name)
                for flag, mods in MODULES.items():
                    for mod in mods:
                        self.assertEqual(hasattr(m, mod), f[flag], mod)
                for kept in ("traj_embed", "traj_gru", "goal_embed", "light_embed", "head"):
                    self.assertTrue(hasattr(m, kept), kept)
                width = (1 + (1 if f["scene_attention"] else 2) + f["goal_fusion"]
                         + (2 if f["max_pool"] else 1) * (f["edge_branch"] + f["wm_branch"]))
                self.assertEqual(m.head[1].in_features, width * H)

    def test_head_width_per_ablation(self):
        want = {"A0": 7, "A1": 6, "A2": 7, "A3": 6, "A4": 6, "A5": 7, "A6": 4, "A7": 4, "A8": 4}
        self.assertEqual({n: _model(n).head[1].in_features // H for n in ABLATIONS}, want)

    def test_removed_inputs_have_no_effect(self):
        ctx, feats, raw, wm, edges, size = _inputs()
        cases = _perturbed(ctx, wm, edges)
        for name, f in ABLATIONS.items():
            m = _model(name)
            base = _score(m, ctx, feats, raw, wm, edges, size)
            for key, (c, w, e, flag) in cases.items():
                with self.subTest(name=name, input=key):
                    s = _score(m, c, feats, raw, w, e, size)
                    if flag is None or f[flag]:
                        self.assertGreater(float((s - base).abs().max()), 1e-5)
                    else:
                        torch.testing.assert_close(s, base, atol=0.0, rtol=0.0)

    def test_a5_lights_masked_mean(self):
        ctx, feats, raw, wm, edges, size = _inputs()
        m = _model("A5")
        base = _score(m, ctx, feats, raw, wm, edges, size)
        masked = dict(ctx, lights_mask=torch.tensor([[True, False]] * B))
        changed = dict(masked, lights=torch.cat([ctx["lights"][:, :1], ctx["lights"][:, 1:] + 5.0], 1))
        torch.testing.assert_close(_score(m, changed, feats, raw, wm, edges, size),
                                   _score(m, masked, feats, raw, wm, edges, size), atol=0.0, rtol=0.0)
        none = dict(ctx, lights_mask=torch.zeros(B, 2, dtype=torch.bool))
        self.assertTrue(torch.isfinite(_score(m, none, feats, raw, wm, edges, size)).all())
        self.assertGreater(float((_score(m, masked, feats, raw, wm, edges, size) - base).abs().max()), 1e-5)

    def test_a7_needs_no_wm_steps(self):
        ctx, feats, raw, wm, edges, size = _inputs()
        short = dict(wm, mean=wm["mean"][:, :, :, :5], cov=wm["cov"][:, :, :, :5])
        torch.testing.assert_close(_score(_model("A7"), ctx, feats, raw, short, edges, size),
                                   _score(_model("A7"), ctx, feats, raw, wm, edges, size), atol=0.0, rtol=0.0)
        with self.assertRaises(ValueError):
            _score(_model("A4"), ctx, feats, raw, short, edges, size)

    def test_scene_attention_off_requires_lights_and_goal_only(self):
        with self.assertRaises(ValueError):
            WMTrajectoryDiscriminator(hidden=H, heads=4, **{**flags("A0"), "scene_attention": False})

    def test_parameter_counts_shrink(self):
        n = {name: _n_params(_model(name)) for name in ABLATIONS}
        self.assertLess(n["A1"], n["A0"])
        self.assertLess(n["A2"], n["A0"])
        self.assertLess(n["A3"], min(n["A1"], n["A2"]))
        self.assertLess(n["A4"], n["A3"])
        for name in ("A5", "A6", "A7", "A8"):
            self.assertLess(n[name], n["A4"], name)

    def test_build_model_and_old_checkpoint_kwargs(self):
        base = {"hidden": H, "heads": 4, "dropout": 0.0, "variant": "full_gaussian", "pos_scale": 20.0,
                "vel_scale": 5.0}
        self.assertEqual(ablation_of(base), "A0")
        torch.manual_seed(0)
        old = build_model(base)
        self.assertEqual(set(old.state_dict()), set(_model("A0").state_dict()))
        old_a4 = {**base, "ablation": "A4", "goal_fusion": False, "map_tokens": False, "agent_tokens": False}
        self.assertEqual(ablation_of(old_a4), "A4")
        self.assertEqual(set(build_model(old_a4).state_dict()), set(_model("A4").state_dict()))
        for name in ABLATIONS:
            with self.subTest(name=name):
                kw = {**base, "ablation": name, **flags(name)}
                m = build_model(kw)
                self.assertEqual(ablation_of(kw), name)
                self.assertEqual({k: getattr(m, k) for k in FLAG_KEYS}, flags(name))
                m.load_state_dict(_model(name).state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
