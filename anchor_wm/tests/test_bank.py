"""cache_bank geometry and bank_data loader on synthetic data (no waymax / tfrecords needed).

  python -m unittest discover -s anchor_wm/tests -t . -v
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from anchor_wm.bank_data import build_index, make_bank_loader
from anchor_wm.cache_bank import A_KEYS, NUM_LANE_POINTS, STEPS, lane_points, step_geometry
from anchor_wm.config import AnchorWMConfig
from anchor_wm.data import build_arrays
from anchor_wm.tests.test_anchor_wm import synthetic_payload

K = 6


def _fut_one_box(x: float, y: float, typ: int = 1) -> dict:
    s = STEPS
    return {"xy": np.tile(np.array([[[x, y]]], np.float32), (1, 1, s, 1)),
            "yaw": np.zeros((1, 1, s), np.float32), "lw": np.tile(np.array([[[4.5, 2.0]]], np.float32), (1, 1, s, 1)),
            "valid": np.ones((1, 1, s), bool), "type": np.array([typ])}


def _edges() -> dict:
    xs = np.arange(-20, 80, 1.0, dtype=np.float32)
    n = len(xs)
    xy = np.concatenate([np.stack([xs, np.full(n, -6.0, np.float32)], -1), np.stack([xs[::-1], np.full(n, 6.0, np.float32)], -1)])
    d = np.concatenate([np.tile([[1.0, 0.0]], (n, 1)), np.tile([[-1.0, 0.0]], (n, 1))]).astype(np.float32)
    same = np.ones(2 * n, bool)
    same[0] = same[n] = False
    return {"edge_xy": xy, "edge_dir": d, "edge_prev_dir": np.concatenate([d[:1], d[:-1]]), "edge_same_prev": same,
            "edge_valid": np.ones(2 * n, bool)}


class GeometryTest(unittest.TestCase):
    def test_step_geometry(self):
        t = np.arange(1, STEPS + 1) * 0.1
        through = np.stack([4.0 * t, np.zeros_like(t), np.zeros_like(t)], -1)          # drives through x = 10
        beside = np.stack([4.0 * t, np.full_like(t, 3.0), np.zeros_like(t)], -1)       # passes 3 m to the left
        off = np.stack([4.0 * t, -0.2 * t * 10, np.zeros_like(t)], -1)                 # drifts right past y = -6
        cand = np.stack([through, beside, off]).astype(np.float32)
        gap, typ, sd = step_geometry(cand, np.array([4.5, 2.0]), _fut_one_box(10.0, 0.0, typ=2), _edges(), horizon=40)
        gap = gap.astype(np.float32)
        x = 4.0 * t
        expect = (np.abs(x - 10.0) < 4.5) & (np.arange(1, STEPS + 1) <= 40)
        np.testing.assert_array_equal(gap[0] <= 0, expect)
        self.assertTrue(np.all(typ[0][expect] == 2))
        self.assertTrue(np.all(gap[1][:40] > 0))
        self.assertTrue(np.all(gap[:, 40:] == 50))                                     # past the horizon
        self.assertTrue((sd[2][:40].astype(np.float32) > 0).any())
        self.assertFalse((sd[1][:40].astype(np.float32) > 0).any())

    def test_lane_points(self):
        p = synthetic_payload(s=1)
        lp = lane_points(p, 0, np.zeros(2), 0.0)
        self.assertEqual(lp["lane_xy"].shape, (NUM_LANE_POINTS, 2))
        self.assertGreater(int(lp["lane_valid"].sum()), 10)
        self.assertTrue(np.allclose(lp["lane_xy"][lp["lane_valid"]][:, 1], 0.0, atol=1e-2))  # lane at y = 0


def make_fake_bank(root: str, cfg: AnchorWMConfig | None = None) -> tuple[str, str, str, str]:
    """One batch file (2 scenes, 4 groups x K candidates) + labels + an anchor cache built with `cfg`."""
    bank, cache = os.path.join(root, "es_candidates"), os.path.join(root, "cache")
    os.makedirs(bank)
    os.makedirs(cache)
    cfg = cfg or AnchorWMConfig(max_agents=8, max_map_segments=6, max_points_per_segment=8, num_label_edges=64,
                                num_edge_tokens=16)
    payload = synthetic_payload(s=2, n=8)
    g_n, rng = 4, np.random.default_rng(0)
    name = "training_tfexample.tfrecord-00003-of-01000.scen_000_001"
    group_scen = np.array([0, 0, 1, 1])
    cand_group = np.repeat(np.arange(g_n), K)[rng.permutation(g_n * K)]
    n = g_n * K
    np.savez(os.path.join(bank, name + ".npz"),
             group_valid=np.ones(g_n, bool), group_scen=group_scen.astype(np.int16), group_t=np.array([10, 20, 10, 20]),
             scen_tfrecord=np.array(["training_tfexample.tfrecord-00003-of-01000"] * 2), scen_index=np.array([0, 1]),
             scen_id=np.array(["sceneA", "sceneB"]), cand_group=cand_group, cand_total=rng.random(n),
             cand_collision=rng.random(n), cand_offroad=np.ones(n), cand_selected=np.zeros(n, bool),
             cand_stage=np.zeros(n, np.int8), cand_traj=rng.normal(size=(n, 51, 3)).astype(np.float32))
    overlap = np.zeros(n, np.int16)
    overlap[:K // 2] = 7
    np.savez(os.path.join(bank, name + ".labels_waymax.npz"),
             gt_overlap_step=overlap, gt_offroad_step=np.zeros(n, np.int16), label_horizon=np.full(g_n, 50),
             anchor_overlap=np.array([False, True, False, False]), anchor_offroad=np.zeros(g_n, bool),
             expert_overlap_step=np.zeros(g_n, np.int16), expert_offroad_step=np.zeros(g_n, np.int16))
    np.savez(os.path.join(bank, name + ".labels.npz"), gt_ade_m=rng.random(n), gt_goal_gain_m=rng.random(n),
             gt_progress_m=rng.random(n), gt_final_goal_dist_m=rng.random(n), gt_path_len_m=rng.random(n))
    arrs = build_arrays(payload, anchor_step=10, cfg=cfg, with_targets=False)
    cache_arrays = {key: np.stack([arrs[key][int(group_scen[g])] for g in range(g_n)]) for key in A_KEYS}
    cache_arrays.update(goal_xy=np.zeros((g_n, 2), np.float32),
                        lane_xy=np.zeros((g_n, NUM_LANE_POINTS, 2), np.float16),
                        lane_dir=np.zeros((g_n, NUM_LANE_POINTS, 2), np.float16),
                        lane_id=np.zeros((g_n, NUM_LANE_POINTS), np.int32), lane_valid=np.zeros((g_n, NUM_LANE_POINTS), bool),
                        b_gap=np.ones((g_n, K, STEPS), np.float16), b_type=np.zeros((g_n, K, STEPS), np.int8),
                        b_offroad=-np.ones((g_n, K, STEPS), np.float16),
                        cand_rows=np.argsort(cand_group, kind="stable").reshape(g_n, K).astype(np.int32))
    np.savez(os.path.join(cache, name + ".anchor_cache.npz"), **cache_arrays)
    split = os.path.join(root, "split.json")
    json.dump({"assignment": {"sceneA": "train", "sceneB": "test"}}, open(split, "w"))
    invalid = os.path.join(root, "invalid.json")
    json.dump([], open(invalid, "w"))
    return bank, cache, split, invalid


class LoaderTest(unittest.TestCase):
    def test_filters_split_and_batches(self):
        with tempfile.TemporaryDirectory() as root:
            bank, cache, split, invalid = make_fake_bank(root)
            idx, counts = build_index(bank, cache, split, invalid, ("train",))
            self.assertEqual(counts["kept"], 1)                     # scene A: group 0 (group 1 has anchor overlap)
            self.assertEqual(list(idx[0][1]), [0])
            idx_all, counts_all = build_index(bank, cache, split, invalid, ("train", "test"))
            self.assertEqual(counts_all["kept"], 3)
            loader = make_bank_loader(idx_all, cache, batch_groups=3, cands_per_group=4, shuffle=False, seed=0,
                                      num_workers=0)
            batch = next(iter(loader))
            self.assertEqual(tuple(batch["cand_xy"].shape), (3, 4, 25, 2))
            self.assertEqual(tuple(batch["b_gap"].shape), (3, 4, STEPS))
            self.assertEqual(tuple(batch["agent_history"].shape)[0], 3)
            # At most half of the sampled candidates are unsafe.
            unsafe = (batch["overlap_step"] > 0) | (batch["offroad_step"] > 0)
            self.assertTrue(bool((unsafe.sum(1) <= 2).all()))
            # Candidates are the 0.2 s subsample of the recorded trajectories of that group's rows.
            d = np.load(os.path.join(bank, os.listdir(bank)[0].split(".labels")[0].split(".npz")[0] + ".npz"))
            rows = np.load(os.path.join(cache, os.listdir(cache)[0]))["cand_rows"][0]
            sub = d["cand_traj"][rows][:, 1:50:2, :2]
            got = batch["cand_xy"][0].numpy()
            self.assertTrue(all(any(np.allclose(g, s) for s in sub) for g in got))


if __name__ == "__main__":
    unittest.main()


class PlanUnitsTest(unittest.TestCase):
    def test_grouping_and_split(self):
        from anchor_wm.cache_bank import plan_units, tfrecord_of
        a = [f"/b/training_tfexample.tfrecord-00002-of-01000.scen_{i:03d}_{i + 1:03d}.npz" for i in range(10)]
        b = [f"/b/training_tfexample.tfrecord-00003-of-01000.scen_{i:03d}_{i + 1:03d}.npz" for i in range(3)]
        done = {os.path.basename(a[0])}
        units = plan_units(a + b, done, files_per_unit=4)
        self.assertEqual(tfrecord_of(a[1]), "training_tfexample.tfrecord-00002-of-01000")
        flat = [f for u in units for f in u]
        self.assertEqual(sorted(flat), sorted(a[1:] + b))                 # every remaining file exactly once
        self.assertTrue(all(len({tfrecord_of(f) for f in u}) == 1 for u in units))   # one tfrecord per unit
        self.assertTrue(all(len(u) <= 4 for u in units))
        self.assertEqual(len(units), 3 + 1)                               # 9 files of 00002 -> 3 units, 00003 -> 1


class PrefilterTest(unittest.TestCase):
    def test_prefilter_matches_full(self):
        rng = np.random.default_rng(1)
        k, s, n = 40, STEPS, 60
        cand = np.zeros((k, s, 3), np.float32)
        cand[..., 0] = np.linspace(0, 30, s)[None] * rng.uniform(0.3, 1.2, (k, 1))
        cand[..., 1] = rng.normal(0, 0.3, (k, s)).cumsum(1)
        pos = rng.uniform(-80, 150, (1, n, 1, 2)).astype(np.float32).repeat(s, 2)
        pos[0, :8] = rng.uniform(0, 30, (8, 1, 2)).repeat(s, 1)                 # some objects near the path
        fut = {"xy": pos, "yaw": rng.uniform(-3, 3, (1, n, s)).astype(np.float32),
               "lw": np.tile(np.array([4.5, 2.0], np.float32), (1, n, s, 1)),
               "valid": rng.random((1, n, s)) > 0.2, "type": rng.integers(1, 4, n)}
        e = _edges()
        full = step_geometry(cand, np.array([4.5, 2.0]), fut, e, 40, prefilter=False)
        fast = step_geometry(cand, np.array([4.5, 2.0]), fut, e, 40, prefilter=True)
        gf, gp = full[0].astype(np.float32), fast[0].astype(np.float32)
        np.testing.assert_array_equal(gf <= 0, gp <= 0)                         # overlap flags identical
        close = gf < 10.0
        np.testing.assert_array_equal(gf[close], gp[close])                     # near gaps identical
        np.testing.assert_array_equal(full[1][close], fast[1][close])
        np.testing.assert_array_equal(full[2], fast[2])                         # offroad identical here

    def test_no_objects_and_no_near_edges(self):
        t = np.arange(1, STEPS + 1) * 0.1
        cand = np.stack([np.stack([4 * t, 0 * t, 0 * t], -1)]).astype(np.float32)
        fut = _fut_one_box(500.0, 500.0)                                        # far away -> filtered out
        far_edges = {k: v.copy() for k, v in _edges().items()}
        far_edges["edge_xy"] = far_edges["edge_xy"] + np.array([0.0, 200.0], np.float32)
        gap, typ, sd = step_geometry(cand, np.array([4.5, 2.0]), fut, far_edges, 50)
        self.assertTrue(np.all(gap == 50))
        full = step_geometry(cand, np.array([4.5, 2.0]), fut, far_edges, 50, prefilter=False)
        np.testing.assert_array_equal(sd, full[2])                              # falls back to all edges


class ReadRecordsTest(unittest.TestCase):
    def test_seek_reader(self):
        import struct
        from anchor_wm.cache_bank import read_records
        payloads = [os.urandom(np.random.default_rng(i).integers(10, 5000)) for i in range(30)]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.tfrecord")
            with open(path, "wb") as fh:
                for p in payloads:
                    fh.write(struct.pack("<Q", len(p)) + b"\0\0\0\0" + p + b"\0\0\0\0")
            got = read_records(path, {0, 7, 29})
            self.assertEqual(set(got), {0, 7, 29})
            for i in got:
                self.assertEqual(got[i], payloads[i])
            self.assertEqual(set(read_records(path, {3, 99})), {3})          # missing index -> not returned
