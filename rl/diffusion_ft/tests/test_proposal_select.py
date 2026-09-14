"""Pure-logic tests for proposal selection and matched-sample preservation.

No model, no corpus, no GPU. What each block pins, and how it would lie if wrong:

1. the arm grammar -- a typo silently becoming a different arm;
2. the selector -- `cons` must NEVER touch a clean base candidate, however
   tight, or it is not conservative;
3. `preserve_loss` -- zero with zero gradient at the base, heading wrapped (a
   +pi/-pi pair is the same heading, not a 2pi error), zero-weight rows inert;
4. the intermittent stratum -- the two existing strata's counts are reused, not
   re-counted, so the old marginal stays comparable;
5. the analysis -- recovered - broken, population-weighted, IS the delta (the
   identity the figure's right panel relies on); the oracle is >= both policies;
6. replay bookkeeping -- deterministic ids per iteration, expiry, refresh.

    python -m rl.diffusion_ft.tests.test_proposal_select
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from rl.diffusion_ft.proposal_select import parse_arms, select_switch
from rl.diffusion_ft.tools.add_intermittent_split import intermittent_strata
from rl.diffusion_ft.tools.select_analysis import analyze
from rl.diffusion_ft.train_safety import SafetyFineTuner, preserve_loss

FAILED = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)


def raises(fn):
    try:
        fn()
    except SystemExit:
        return True
    return False


def t_arms():
    arms = parse_arms("base,ft,bf=ft:cons:10,bb=base2:margin", default_window=20)
    check([a.name for a in arms] == ["base", "ft", "bf", "bb"], "1 names in order")
    check(arms[0].alt is None and arms[1].primary == "ft", "1 single-policy arms")
    check(arms[2].window == 10 and arms[3].window == 20, "1 explicit and default window")
    check(arms[2].needs_ft and not arms[3].needs_ft and arms[1].needs_ft, "1 needs_ft")
    check(raises(lambda: parse_arms("x=ftt:cons")), "1 unknown alt refused")
    check(raises(lambda: parse_arms("x=ft:best")), "1 unknown selector refused")
    check(raises(lambda: parse_arms("a=ft:cons,a=base2:cons")), "1 duplicate names refused")
    check(raises(lambda: parse_arms("oops")), "1 bare token refused")
    # theta is the 4th field and belongs to `learned` alone: the fitted theta
    # optimizes a per-decision objective that cannot price over-switching, so the
    # deployable value is swept in closed loop.
    a = parse_arms("l=ft:learned:20:0.35")[0]
    check(a.selector == "learned" and a.window == 20 and abs(a.theta - 0.35) < 1e-9,
          f"1 theta parsed ({a.theta})")
    check(parse_arms("c=ft:cons:10")[0].theta is None, "1 no theta means the model's own")
    check(raises(lambda: parse_arms("c=ft:cons:10:0.3")), "1 theta on a non-learned arm refused")


def t_selector():
    kw = dict(tol_m=1.0, margin_m=0.3, delta_m=0.2)
    # rows: base violates/alt clean; base clean but tight; base violates/alt violates;
    #       base violates/alt clean but gives up progress
    cp = np.array([-0.2, 0.05, -0.5, -0.1]); ca = np.array([0.4, 0.9, -0.1, 0.5])
    pp = np.array([10., 10., 10., 10.]);    pa = np.array([9.5, 10., 10., 7.0])
    sw = select_switch("cons", clear_p=cp, clear_a=ca, prog_p=pp, prog_a=pa, **kw)
    check(sw.tolist() == [True, False, False, False], f"2 cons rule {sw.tolist()}")
    sw = select_switch("margin", clear_p=cp, clear_a=ca, prog_p=pp, prog_a=pa, **kw)
    check(sw.tolist() == [True, True, False, False], f"2 margin rule {sw.tolist()}")
    rng = np.random.default_rng(0)
    cp = rng.uniform(0, 5, 1000); ca = rng.uniform(-5, 5, 1000)
    sw = select_switch("cons", clear_p=cp, clear_a=ca, prog_p=np.zeros(1000),
                       prog_a=np.full(1000, 50.0), **kw)
    check(not sw.any(), "2 cons never switches away from a clean base candidate")


def t_preserve():
    rng = np.random.default_rng(1)
    tgt = jnp.asarray(rng.normal(size=(3, 12, 5)), jnp.float32)
    w = jnp.asarray([1.0, 1.0, 0.0])
    loss, aux = preserve_loss(tgt, tgt, w, n=10)
    check(float(loss) == 0.0, "3 zero at the base")
    g = jax.grad(lambda x: preserve_loss(x, tgt, w, n=10)[0])(tgt)
    check(float(jnp.max(jnp.abs(g))) < 1e-6, "3 zero gradient at the base")
    a = tgt.at[:, :, 2].set(np.pi - 1e-3)
    b = tgt.at[:, :, 2].set(-np.pi + 1e-3)
    l_wrap, _ = preserve_loss(a, b, w, n=10, s_yaw=0.1)
    check(float(l_wrap) < 1e-3, f"3 heading wrapped ({float(l_wrap):.2e})")
    moved = tgt.at[:, 1:11, 0].add(0.5)
    _, aux = preserve_loss(moved, tgt, w, n=10)
    check(abs(float(aux["preserve_rms_m"]) - 0.5) < 1e-5, "3 rms reads metres")
    moved2 = moved.at[2, :, 0].add(100.0)
    l1, _ = preserve_loss(moved, tgt, w, n=10)
    l2, _ = preserve_loss(moved2, tgt, w, n=10)
    check(abs(float(l1) - float(l2)) < 1e-6, "3 zero-weight row is inert")
    beyond = tgt.at[:, 11, 0].add(9.0)
    l3, _ = preserve_loss(beyond, tgt, w, n=10)
    check(float(l3) == 0.0, "3 steps past the window are not scored")


def t_intermittent():
    rows = {1: {"solved_frac": 0.0}, 2: {"solved_frac": 0.5}, 3: {"solved_frac": 0.25},
            4: {"solved_frac": 0.75, "clean_frac": 0.5},       # dirty: excluded
            10: {"solved_frac": 1.0}, 11: {"solved_frac": 0.5}, 12: {"solved_frac": 1.0},
            13: {"solved_frac": 1.0}}
    fail_src = [1, 2, 3, 4]
    succ_src = [10, 11, 12, 13, 14, 15, 16, 17]                # 4 of 8 verified
    s2 = {"failure_heldout": {"n": 1.0}, "success_heldout": {"n": 6.0}}
    ids, s3, c = intermittent_strata(rows, fail_src, succ_src, s2)
    check(ids == [2, 3, 11], f"4 ids {ids}")
    # 2 failure-source + 8 * (1/4) extrapolated = 4
    check(abs(s3["intermittent_heldout"]["n"] - 4.0) < 1e-9, "4 extrapolated count")
    check(s3["failure_heldout"]["n"] == 1.0 and s3["success_heldout"]["n"] == 6.0,
          "4 existing strata reused verbatim")
    check(abs(sum(v["weight"] for v in s3.values()) - 1.0) < 1e-12, "4 weights sum to 1")


def _rows(split, scenes, arm, outcomes, ckpt=""):
    out = []
    for s, reps in zip(scenes, outcomes):
        for r, v in enumerate(reps):
            out.append({"scene_id": s, "split": split, "arm": arm, "repeat": r, "seed": 4242,
                        "ckpt": ckpt, "solved": bool(v), "offroad": False, "overlap": False,
                        "reached": bool(v), "clean": True, "n_replans": 8,
                        "n_switch": 2 if arm != "base" else 0})
    return out


def t_analysis():
    from collections import defaultdict
    rng = np.random.default_rng(3)
    spec = {"population": {"strata": {"failure_heldout": {"weight": 0.02},
                                      "success_heldout": {"weight": 0.98}},
                           "strata3": {"failure_heldout": {"weight": 0.02},
                                       "success_heldout": {"weight": 0.95},
                                       "intermittent_heldout": {"weight": 0.03}}}}
    flat = []
    base_p = {"failure_heldout": 0.05, "success_heldout": 0.99, "intermittent_heldout": 0.5}
    for si, (sp, p) in enumerate(base_p.items()):
        sc = list(range(1000 * si, 1000 * si + 40))
        b = rng.random((40, 4)) < p
        f1 = np.where(rng.random((40, 4)) < 0.3, ~b, b)
        f2 = np.where(rng.random((40, 4)) < 0.3, ~b, b)
        flat += _rows(sp, sc, "base", b)
        flat += _rows(sp, sc, "ft", f1, ckpt="/r/s0/iter_000110")
        flat += _rows(sp, sc, "ft", f2, ckpt="/r/s1/iter_000110")
        flat += _rows(sp, sc, "bf", np.maximum(b, f1), ckpt="/r/s0/iter_000110")
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in flat:
        rows[(r["arm"], r["ckpt"])][r["split"]][r["scene_id"]][r["repeat"]] = r
    res = analyze(rows, spec, repeats=4, n_boot=200)
    for k, pol in list(res["policies"].items()) + list(res["ckpt_mean"].items()):
        for mt in ("marginal2", "marginal3"):
            m = pol[mt]
            check(abs(m["delta"] - (m["recovered"] - m["broken"])) < 1e-9,
                  f"5 {k} {mt}: recovered - broken == delta")
            check(m["ci95"][0] <= m["delta"] <= m["ci95"][1] + 1e-12, f"5 {k} {mt}: CI brackets")
    bf = res["policies"]["bf@s0/iter_000110"]["marginal3"]
    check(bf["broken"] == 0.0, "5 any(base, ft) never breaks a base success")
    orc = res["oracle"]["base_or_ft_same_seed"]["marginal3"]["gain"]
    check(orc >= res["ckpt_mean"]["ft"]["marginal3"]["delta"] - 1e-12, "5 oracle >= ft")
    check(orc >= 0.0, "5 oracle >= base")


def t_replay():
    t = SafetyFineTuner.__new__(SafetyFineTuner)
    t.args = SimpleNamespace(preserve_scenes=6, data_seed=0, preserve_replay_per_iter=2,
                             preserve_hard_frac=0.5, preserve_replay=3,
                             preserve_replay_uses=2)
    t.preserve_weight = 1.0
    t._replay = {}
    t._preserve_pool = np.arange(100, 200)
    t._preserve_hard = np.arange(100, 110)
    t._success_groups = [1]
    lab = {5: 1, 6: 1, 7: 0, 8: 1, 9: 1}
    t.sampler = SimpleNamespace(last_ids=np.array([5, 6, 7, 8]),
                                group_labels=lambda ids: np.array([lab[int(i)] for i in ids]))
    n = t.note_broken(np.array([0, 1, 0, 0]), np.array([1, 1, 1, 0]))
    check(n == 1 and list(t._replay) == [5], f"6 only broken clean success rows ({t._replay})")
    ids, k = t.preserve_ids(0)
    check(len(ids) == 6 and k == 1 and ids[0] == 5, f"6 replay first, fixed size ({ids})")
    check(sum(100 <= i < 110 for i in ids[1:]) >= 2, "6 hard share drawn")
    ids2, _ = t.preserve_ids(0)
    check(5 not in t._replay, "6 expires after its uses")
    t.sampler.last_ids = np.array([5, 6, 8, 9])
    t.note_broken(np.zeros(4), np.ones(4))
    check(len(t._replay) == 3 and 5 not in t._replay, f"6 capacity evicts oldest ({t._replay})")
    t2 = SafetyFineTuner.__new__(SafetyFineTuner)
    t2.__dict__.update({k: v for k, v in t.__dict__.items() if k != "_replay"})
    t2._replay = {}
    a, _ = t2.preserve_ids(7); b, _ = t2.preserve_ids(7)
    check(a == b, "6 same iteration -> same draw (resume-safe)")


def main():
    for fn in (t_arms, t_selector, t_preserve, t_intermittent, t_analysis, t_replay):
        fn()
    print(f"\n{'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
