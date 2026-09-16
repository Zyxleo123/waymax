"""Pure-logic tests for proposal selection and matched-sample preservation.

No model, no corpus, no GPU. What each block pins, and how it would lie if wrong:

1. the arm grammar -- a typo silently becoming a different arm;
2. the selector -- `cons` must NEVER touch a clean base candidate, however
   tight, or it is not conservative;
2b. the candidate BANK -- `cons`'s do-no-harm property must survive a bank of
   K (a clean base candidate is untouchable however many alternatives are on
   the table), the winner must be the SMALLEST admissible perturbation, and
   the perturbations must be continuous at the anchor or the car teleports;
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

from rl.diffusion_ft.branching import lateral_shift, time_warp
from rl.diffusion_ft.proposal_select import (bank_choice, make_bank, make_banks,
                                             parse_arms, select_switch)
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


def t_bank():
    kw = dict(tol_m=1.0, margin_m=0.3, delta_m=0.2)
    mag = [0.5, 1.0, 2.0]                       # ascending: smallest edit first
    # scene 0: only the LARGEST candidate is clean -> it must still be chosen;
    # scene 1: candidates 1 and 2 are both clean -> the smaller magnitude wins;
    # scene 2: base is clean -> nothing may fire, however clean the candidates;
    # scene 3: every candidate violates -> keep the base.
    cp = np.array([-0.3, -0.3, 0.9, -0.3])
    ck = np.array([[-0.1, -0.1, 5.0, -0.2],      # cand 0 (mag 0.5)
                   [-0.1, 0.4, 5.0, -0.2],       # cand 1 (mag 1.0)
                   [0.6, 0.4, 5.0, -0.2]])       # cand 2 (mag 2.0)
    pp = np.full(4, 10.0); pk = np.full((3, 4), 10.0)
    ch = bank_choice("cons", clear_p=cp, prog_p=pp, clear_k=ck, prog_k=pk,
                     magnitude=mag, **kw)
    check(ch.tolist() == [2, 1, -1, -1], f"2b bank picks smallest admissible {ch.tolist()}")
    # The do-no-harm property is the whole reason `broken` was 0.0000; a bank must
    # not weaken it by offering more chances to fire on a clean base candidate.
    rng = np.random.default_rng(3)
    n, K = 2000, 8
    ch = bank_choice("cons", clear_p=rng.uniform(0.001, 5, n), prog_p=np.zeros(n),
                     clear_k=rng.uniform(-5, 5, (K, n)), prog_k=np.full((K, n), 50.0),
                     magnitude=np.arange(K, dtype=float), **kw)
    check((ch < 0).all(), "2b bank never switches away from a clean base candidate")
    # A candidate that gives up too much progress is inadmissible even when clean.
    ch = bank_choice("cons", clear_p=np.array([-0.3]), prog_p=np.array([10.0]),
                     clear_k=np.array([[5.0]]), prog_k=np.array([[2.0]]),
                     magnitude=[0.5], **kw)
    check(ch.tolist() == [-1], "2b progress tolerance still binds inside a bank")
    # Ranking is by magnitude, NOT by clearance: the bank is a minimal
    # intervention, not a licence to take the most aggressive edit that scores well.
    ch = bank_choice("cons", clear_p=np.array([-0.3]), prog_p=np.array([10.0]),
                     clear_k=np.array([[0.1], [9.0]]), prog_k=np.full((2, 1), 10.0),
                     magnitude=[0.5, 1.0], **kw)
    check(ch.tolist() == [0], "2b smallest edit beats the clearest one")

    b = make_bank("tw0.85,lat-0.25")
    check([c.family for c in b] == ["tw", "lat"], "2b bank spec parsed")
    check(raises(lambda: make_bank("id")), "2b id refused (it IS the base proposal)")
    check(raises(lambda: make_bank("rn20s0")), "2b rn refused (costs a denoiser eval)")
    check(raises(lambda: make_bank("tw0.7h2")), "2b hold refused (selector re-decides)")
    check(raises(lambda: parse_arms("x=bank:learned:20:0.3")), "2b bank+learned refused")
    check(raises(lambda: parse_arms("x=bank:look")), "2b bank+look refused")
    check(parse_arms("x=bank:cons:10")[0].alt == "bank", "2b bank arm parses")

    # Named banks: the lateral-only / timing-only ablation must be able to run as
    # two arms of ONE job, or it is compared across submissions instead of paired.
    a = parse_arms("all=bank:cons:20,lo=bank@lat:cons:20,to=bank@tw:cons:20")
    check([x.alt for x in a] == ["bank"] * 3, "2b bank@name keeps alt 'bank'")
    check([x.bank for x in a] == ["", "lat", "tw"], f"2b bank names {[x.bank for x in a]}")
    check(raises(lambda: parse_arms("x=bank@:cons")), "2b empty bank name refused")
    bs = make_banks("tw0.85,lat0.25", "lat=lat0.25,lat-0.5;tw=tw0.85,tw1.3")
    check(sorted(bs) == ["", "lat", "tw"], f"2b make_banks keys {sorted(bs)}")
    check([c.name for c in bs["lat"]] == ["lat0.25", "lat-0.5"], "2b named bank parsed")
    check(all(c.family == "tw" for c in bs["tw"]), "2b timing-only bank is timing-only")
    check(raises(lambda: make_banks("tw0.85", "lat0.25")), "2b bank without a name refused")
    check(raises(lambda: make_banks("tw0.85", "lat=lat0.25;lat=lat0.5")),
          "2b duplicate bank name refused")
    # Every named bank faces the same family restriction as the default one.
    check(raises(lambda: make_banks("tw0.85", "bad=rn20s0")), "2b rn refused in a named bank")

    # Continuity at the anchor: index 0 is the CURRENT pose and both families must
    # leave it untouched, or the switch teleports the car at the join.
    rng = np.random.default_rng(4)
    w = np.zeros((2, 16, 5), np.float32)
    w[..., 0] = np.cumsum(rng.uniform(0.4, 0.6, (2, 16)), axis=1)
    w[..., 3] = 5.0
    w = jnp.asarray(w)
    for name, out in (("time_warp", time_warp(w, 0.7, ramp_steps=10, dt=0.1)),
                      ("lateral_shift", lateral_shift(w, 0.5, ramp_steps=10, dt=0.1))):
        d = float(jnp.max(jnp.abs(out[:, 0] - w[:, 0])))
        check(d < 1e-5, f"2b {name} leaves the anchor pose fixed ({d:.2e})")
    tw1 = time_warp(w, 1.0, ramp_steps=10, dt=0.1)
    check(float(jnp.max(jnp.abs(tw1[..., :3] - w[..., :3]))) < 1e-4,
          "2b time_warp(alpha=1) is the identity on the path")


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
    for fn in (t_arms, t_selector, t_bank, t_preserve, t_intermittent, t_analysis, t_replay):
        fn()
    print(f"\n{'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
