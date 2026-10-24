"""Counterfactual rollouts: repair search and per-decision selector labels.

Both modes run the frozen base closed-loop once per chunk, snapshot every
replan, then branch from those snapshots (:mod:`rl.diffusion_ft.branching`).

``--mode repair`` -- WHY. A clearance gradient can push the car away from an
obstacle; it cannot discover that the car should yield earlier, arrive later or
take a different line. So search those directly, around the base's own failing
episode, and keep only what verifiably works:

1. locate the failure: the first replan whose executed steps went off-road or
   overlapped (a scene that is clean but never reaches the goal is charged to its
   last replan -- that is a progress failure, and the time-warp speed-ups exist
   for it);
2. rewind ``j = 0, 1, ..`` replans before it; at that replan, execute a perturbed
   base candidate for ``hold`` replans, then resume the BASE with the original
   sampler keys to the end of the episode;
3. a repair counts only if the WHOLE continuation solves the scene. Per scene the
   kept repair is the latest rewind that works, and within it the smallest
   perturbation (identity < time warp / lateral shift by size < re-noise).

Candidate families (``branching.Candidate``): ``tw<a>`` drive the same path at a x
speed (arrive later / earlier -- other agents are log-replayed, so timing alone
can dissolve a conflict); ``lat<m>`` smooth lateral shift of the line;
``rn<t>s<seed>`` partially re-noise the base candidate to diffusion step t and
denoise with the base (a different manoeuvre from the same intent). ``id`` is
always run: at rewind 0 it must reproduce the base outcome on every scene, and
``id_consistent`` in the rows is that check.

Outputs: ``repairs.jsonl`` (one row per scene: failure type, repaired or not,
which candidate at which rewind, per-candidate success counts) and
``repair_data/*.npz`` -- for every kept repair and each replan it was applied on,
the frozen condition (cond1/cond2/mask) and the executed candidate in the
denoiser's normalized layout. ``train_safety --repair_data`` learns from those.

``--mode labels`` -- WHY. ``proposal_select`` measures whole episodes under a
selector; it cannot say what ONE switch was worth. Here, at each replan ``r`` of
the base episode, the alternative (the fine-tune's late chain from the same
prefix and noise, or a second base late chain) is executed at ``r`` only and the
base resumes. ``label = solved(branch) - solved(base)`` in {-1, 0, +1} per
(scene, replan), next to the same geometric features the selector arms compute.
``tools/fit_selector.py`` fits a selector to these. Caveat: a label prices one
switch with the base deciding afterwards, not a sequence of switches.

Scene pools are TRAINING splits by default (failure_train, success_train), so
nothing here touches the held-out strata ``proposal_select`` scores. Resumable
per chunk (``done/`` markers). Submitted via ``slurm/submit_counterfactual.sh``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np


def _fail_type(ep, b):
    if ep.overlap[b]:
        return "overlap"
    if ep.offroad[b]:
        return "offroad"
    return "noreach"


class Counterfactual:
    def __init__(self, tuner, *, ft_ckpt: str = "", cands=None, rewinds: int = 3,
                 ramp_steps: int = 10, label_alt: str = "ft", window: int = 20):
        import jax
        import jax.numpy as jnp

        from rl.diffusion_ft.branching import Brancher, model_step_valid, world_to_norm
        from rl.diffusion_ft.proposal_select import FEAT_COLS, make_scorer

        self.jax, self.jnp = jax, jnp
        self.t = tuner
        actor = tuner.actor
        actor.reset_trainable_to_frozen()
        self.base_params = actor._tparams
        self.ft_params = None
        if ft_ckpt:
            tuner.load(ft_ckpt)
            self.ft_params = actor._tparams
            actor._tparams = self.base_params
        self.ft_ckpt = str(ft_ckpt or "")
        self.br = Brancher(tuner, self.base_params)
        self.cands = list(cands or [])
        self.rewinds = int(rewinds)
        self.ramp = int(ramp_steps)
        self.label_alt = label_alt
        self.window = int(window)
        self.cfg = tuner.ckpt.preprocess_cfg
        self._w2n = lambda w, aux: world_to_norm(w, aux, self.cfg)
        self._valid = lambda aux: model_step_valid(
            aux, int(tuner.actor.diffusion.denoise_fn.horizon), float(self.cfg.model_dt))
        self._score = make_scorer(tuner.objcfg)
        self.feat_cols = FEAT_COLS
        self._rn = {}
        self._rt_checked = False

    # ------------------------------------------------------------------ #
    def _renoise(self, ts: int):
        fn = self._rn.get(ts)
        if fn is None:
            diff = self.t.actor.diffusion
            fn = self._rn[ts] = self.jax.jit(
                lambda x, c1, c2, m, key: diff.resample(x, c1, c2, m, ts, rng=key))
        return fn

    def _perturb(self, cand, ctx):
        from rl.diffusion_ft.branching import lateral_shift, time_warp
        jax, jnp = self.jax, self.jnp
        w = ctx["world_p"]
        if cand.family == "tw":
            return time_warp(w, cand.param, ramp_steps=self.ramp, dt=self.br.dt)
        if cand.family == "lat":
            return lateral_shift(w, cand.param, ramp_steps=self.ramp, dt=self.br.dt)
        if cand.family == "rn":
            key = jax.random.fold_in(ctx["rk"], 7919 + 31 * cand.seed + int(cand.param))
            x = self._renoise(int(cand.param))(ctx["x_norm"], ctx["cond1"], ctx["cond2"],
                                               ctx["mask"], key)
            return self.t.env.norm_to_world_with(jnp.transpose(x, (0, 2, 1)), ctx["aux"])
        return w

    def _roundtrip_check(self, ctx):
        """Once per job: world_to_norm(norm_to_world(x)) must return x ON THE
        COVERED STEPS, or every repair target this job writes is a DIFFERENT
        trajectory than the one that was verified.

        Scored under `model_step_valid`; the uncovered tail is reported separately
        because it is not a bug but lost information (see that function), and it is
        why targets carry the base's own proposal there."""
        if self._rt_checked:
            return
        jnp = self.jnp
        back = self._w2n(ctx["world_p"], ctx["aux"])
        ref = jnp.transpose(ctx["x_norm"], (0, 2, 1))
        err = np.asarray(jnp.abs(back - ref))
        m = np.asarray(self._valid(ctx["aux"]))[..., None]
        def mx(sl, mask):
            e = np.where(np.broadcast_to(mask, err.shape)[..., sl], err[..., sl], -np.inf)
            return float(e.max()) if np.isfinite(e).any() else 0.0
        self.rt_err = {"pos": mx(slice(0, 2), m), "vel": mx(slice(2, 4), m),
                       "yaw": mx(slice(4, 5), m)}
        tail = {"pos": mx(slice(0, 2), ~m), "steps_covered": float(m.mean())}
        print(f"[cf] world_to_norm round trip on COVERED steps, max |err| (normalized): "
              f"{self.rt_err}  (pos x ego_range={self.cfg.ego_range} m); "
              f"uncovered tail pos err {tail['pos']:.4f}, covered fraction "
              f"{tail['steps_covered']:.3f} (tail targets fall back to the base proposal)",
              flush=True)
        # Two thresholds, because the consequences differ by orders of magnitude.
        # A ~2 cm residual is EXPECTED and benign: the world samples are the model
        # polyline resampled at world_dt, and world_dt (measured per scene) does not
        # divide the 0.2 s model grid exactly, so the inverse interpolates between
        # two world samples instead of landing on a breakpoint. The residual is that
        # segment's own bend, plus float32. Against a 0.3 m edge margin and a 0.77 m
        # median off-road excursion it is noise. Decimetres would mean a time base
        # that does not line up; metres mean the targets are a different trajectory
        # and would poison `--repair_data` silently.
        if self.rt_err["pos"] > 1e-2:
            raise SystemExit(
                f"[cf] world<->norm round trip is lossy on COVERED steps "
                f"({self.rt_err['pos'] * self.cfg.ego_range:.2f} m): repair targets would "
                f"not be the trajectories that were verified. Refusing to write them.")
        if self.rt_err["pos"] > 1e-3:
            print(f"[cf] WARNING covered-step round trip is "
                  f"{self.rt_err['pos'] * self.cfg.ego_range:.3f} m, not ~0 -- check the "
                  f"time bases before trusting the targets", flush=True)
        self._rt_checked = True

    # ------------------------------------------------------------------ #
    def repair_chunk(self, state, keys):
        jnp = self.jnp
        base = self.br.run(state, keys=keys, record=True)
        # B comes from the finished episode, NOT from `env.batch_size` before it:
        # the env only learns its batch size in `reset`, which the line above is
        # what triggers. Read too early it is 0, and every per-scene array here is
        # empty (the first broadcast then fails).
        B = int(base.solved.shape[0])
        fail_r = np.where(base.fail_replan >= 0, base.fail_replan,
                          max(base.n_replans - 1, 0)).astype(np.int32)
        todo = base.clean & ~base.solved
        repaired = np.zeros(B, bool)
        chosen = [None] * B                    # (j, r_b, cand, magnitude)
        chosen_data = [None] * B
        succ_count = [dict() for _ in range(B)]
        id_consistent = np.ones(B, bool)
        for j in range(self.rewinds):
            r_b = fail_r - j
            active = todo & ~repaired & (r_b >= 0)
            if not active.any():
                break
            r_start = int(r_b[active].min())
            snap = base.snapshots.get(r_start)
            if snap is None:
                break
            best_mag = np.full(B, np.inf)
            best = [None] * B
            for cand in self.cands:
                def override(r, ctx, cand=cand):
                    self._roundtrip_check(ctx)
                    on = active & (r >= r_b) & (r < r_b + cand.hold) & (np.asarray(ctx["alive"]) > 0.5)
                    if cand.family == "id" or not on.any():
                        return None
                    new = self._perturb(cand, ctx)
                    base_norm = jnp.transpose(ctx["x_norm"], (0, 2, 1))
                    # Past the scenario end the inverse cannot recover the model
                    # steps (see branching.model_step_valid), so the target keeps
                    # the BASE proposal there: never executed, and a sane thing for
                    # the denoiser to be taught.
                    vmask = self._valid(ctx["aux"])[..., None]
                    tgt = np.asarray(jnp.where(vmask, self._w2n(new, ctx["aux"]), base_norm))
                    vnp = np.asarray(jnp.broadcast_to(vmask[..., 0], tgt.shape[:2]))
                    idx = np.nonzero(on)[0]
                    ctx["stash"].append({"replan": r, "rows": idx,
                                         "cond1": np.asarray(ctx["cond1"])[idx],
                                         "cond2": np.asarray(ctx["cond2"])[idx],
                                         "mask": np.asarray(ctx["mask"])[idx],
                                         "target": tgt[idx],
                                         "valid": vnp[idx],
                                         "base": np.asarray(base_norm)[idx]})
                    return jnp.where(jnp.asarray(on)[:, None, None], new, ctx["world_p"])

                ep = self.br.run(keys=keys, snapshot=snap, override=override)
                if cand.family == "id" and j == 0:
                    id_consistent &= ~active | (ep.solved == base.solved) & (ep.fail_replan == base.fail_replan)
                    continue
                if cand.family == "id":
                    continue
                win = ep.solved & active
                for b in np.nonzero(win)[0]:
                    succ_count[b][cand.name] = succ_count[b].get(cand.name, 0) + 1
                    if cand.magnitude < best_mag[b]:
                        best_mag[b] = cand.magnitude
                        rows = [{"replan": s["replan"],
                                 **{k: s[k][list(s["rows"]).index(b)] for k in
                                    ("cond1", "cond2", "mask", "target", "valid", "base")}}
                                for s in ep.stash if b in set(s["rows"].tolist())]
                        best[b] = ((j, int(r_b[b]), cand.name, cand.magnitude), rows)
            for b in range(B):
                if best[b] is not None:
                    chosen[b], chosen_data[b] = best[b]
                    repaired[b] = True
        return base, fail_r, todo, repaired, chosen, chosen_data, succ_count, id_consistent

    # ------------------------------------------------------------------ #
    def labels_chunk(self, state, keys):
        jax, jnp = self.jax, self.jnp
        env, actor = self.t.env, self.t.actor
        base = self.br.run(state, keys=keys, record=True)
        out = []
        for r, snap in sorted(base.snapshots.items()):
            alive_r = np.asarray(snap.carry["alive"]) > 0.5
            if not (alive_r & base.clean).any():
                continue
            feats = {}

            def override(rr, ctx):
                if rr != r:
                    return None
                if self.label_alt == "ft":
                    x_a = actor.sample_late_for(self.ft_params, ctx["cond1"], ctx["cond2"],
                                                ctx["mask"], ctx["x_prefix"], ctx["key_late"])
                else:
                    x_a = actor.sample_late_for(self.base_params, ctx["cond1"], ctx["cond2"],
                                                ctx["mask"], ctx["x_prefix"],
                                                jax.random.fold_in(ctx["key_late"], 1))
                world_a = env.norm_to_world_with(jnp.transpose(x_a, (0, 2, 1)), ctx["aux"])
                t0 = ctx["t0"]
                horizon = int(jnp.asarray(env.sim_state.log_trajectory.x).shape[-1])
                n = int(min(self.window, horizon - 1 - t0, world_a.shape[1] - 1))
                t0j = jnp.asarray(t0, jnp.int32)
                sp = [np.asarray(v) for v in self._score(ctx["world_p"], env.sim_state, t0j,
                                                         env._goal_xy_b2, n)]
                sa = [np.asarray(v) for v in self._score(world_a, env.sim_state, t0j,
                                                         env._goal_xy_b2, n)]
                feats["F"] = np.stack(sp + sa, axis=1).astype(np.float32)   # FEAT_COLS order
                return world_a

            ep = self.br.run(keys=keys, snapshot=snap, override=override)
            if "F" not in feats:            # the branch ended before replan r executed
                continue
            label = ep.solved.astype(int) - base.solved.astype(int)
            out.append((r, alive_r & base.clean, feats["F"], label, base.solved, ep.solved))
        return base, out

    # ------------------------------------------------------------------ #
    def run(self, out_dir: Path, *, mode, names, caps, seed, chunk, shard_index, num_shards):
        jax = self.jax
        t = self.t
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "done").mkdir(exist_ok=True)
        (out_dir / "repair_data").mkdir(exist_ok=True)
        spec = t._read_splits_spec(t.args.splits_json)
        key = jax.random.PRNGKey(int(seed))
        t_start = time.time()
        meta = {"mode": mode, "seed": seed, "chunk": chunk, "shard": [shard_index, num_shards],
                "cands": [c.__dict__ for c in self.cands], "rewinds": self.rewinds,
                "ramp_steps": self.ramp, "label_alt": self.label_alt, "window": self.window,
                "ft_ckpt": self.ft_ckpt, "feat_cols": list(self.feat_cols),
                "splits": list(names), "caps": list(caps)}
        (out_dir / f"meta_shard{shard_index}.json").write_text(json.dumps(meta, indent=1))
        for name, cap in zip(names, caps):
            ids = t._split_ids(spec, name)
            ids = (ids if cap < 0 else ids[:cap])[shard_index::num_shards]
            print(f"[cf:{mode}] {name}: {len(ids)} scenes (shard {shard_index}/{num_shards})", flush=True)
            for start in range(0, len(ids), chunk):
                marker = out_dir / "done" / f"{mode}_{name}_s{shard_index}_{start:05d}.ok"
                if marker.exists():
                    continue
                sub = ids[start:start + chunk]
                state = t.sampler.load_explicit(sub)
                keys = self.br.keys_for(jax.random.fold_in(key, 0))
                if mode == "repair":
                    self._write_repair(out_dir, name, sub, start, shard_index,
                                       *self.repair_chunk(state, keys))
                else:
                    self._write_labels(out_dir, name, sub, *self.labels_chunk(state, keys))
                marker.write_text("ok\n")
                print(f"[cf:{mode}] {name} {min(start + chunk, len(ids))}/{len(ids)}  "
                      f"{time.time() - t_start:.0f}s", flush=True)
        print(f"[cf:{mode}] done -> {out_dir}", flush=True)

    def _write_repair(self, out_dir, name, sub, start, shard, base, fail_r, todo, repaired,
                      chosen, chosen_data, succ_count, id_consistent):
        rows, data = [], {k: [] for k in ("scene_id", "replan", "rewind", "cand",
                                           "cond1", "cond2", "mask", "target", "valid", "base")}
        for b, sid in enumerate(sub):
            rec = {"scene_id": int(sid), "split": name, "clean": bool(base.clean[b]),
                   "base_solved": bool(base.solved[b]), "todo": bool(todo[b]),
                   "fail_type": None if base.solved[b] else _fail_type(base, b),
                   "fail_replan": int(fail_r[b]), "repaired": bool(repaired[b]),
                   "id_consistent": bool(id_consistent[b]), "success_by_cand": succ_count[b]}
            if chosen[b] is not None:
                j, r_b, cname, mag = chosen[b]
                rec.update({"rewind": j, "replan": r_b, "cand": cname, "magnitude": mag,
                            "n_targets": len(chosen_data[b])})
                for d in chosen_data[b]:
                    data["scene_id"].append(int(sid)); data["replan"].append(d["replan"])
                    data["rewind"].append(j); data["cand"].append(cname)
                    for k in ("cond1", "cond2", "mask", "target", "valid", "base"):
                        data[k].append(d[k])
            rows.append(rec)
        with (out_dir / "repairs.jsonl").open("a") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in rows))
        if data["scene_id"]:
            np.savez_compressed(
                out_dir / "repair_data" / f"{name}_s{shard}_{start:05d}.npz",
                **{k: (np.asarray(v) if k != "cand" else np.asarray(v, dtype="U32"))
                   for k, v in data.items()},
                rt_err=np.asarray([self.rt_err["pos"], self.rt_err["vel"], self.rt_err["yaw"]]))
        n_todo = int(todo.sum())
        print(f"[cf:repair] chunk: {n_todo} failures, {int(repaired.sum())} repaired, "
              f"id_consistent {float(id_consistent[todo].mean()) if n_todo else 1.0:.3f}", flush=True)

    def _write_labels(self, out_dir, name, sub, base, out):
        rows = []
        for r, valid, F, label, bsolved, asolved in out:
            for b in np.nonzero(valid)[0]:
                rows.append({"scene_id": int(sub[b]), "split": name, "replan": int(r),
                             "label": int(label[b]), "base_solved": bool(bsolved[b]),
                             "alt_solved": bool(asolved[b]),
                             "feat": [float(x) for x in F[b]]})
        with (out_dir / "labels.jsonl").open("a") as fh:
            fh.write("".join(json.dumps(x) + "\n" for x in rows))
        n = len(rows)
        pos = sum(1 for x in rows if x["label"] > 0)
        neg = sum(1 for x in rows if x["label"] < 0)
        print(f"[cf:labels] chunk: {n} decisions  +1 {pos}  -1 {neg}", flush=True)


def main():
    from rl.diffusion_ft.branching import parse_candidates
    from rl.diffusion_ft.train_safety import SafetyFineTuner, build_argparser

    p = build_argparser()
    p.description = __doc__
    p.add_argument("--mode", choices=("repair", "labels"), required=True)
    p.add_argument("--cf_out", default="", help="default <out_dir>")
    p.add_argument("--cf_splits", default="", help="default: failure_train (repair), "
                   "failure_train,success_train (labels)")
    p.add_argument("--cf_caps", default="-1", help="per-split scene cap, one value or a list")
    p.add_argument("--cf_seed", type=int, default=2718)
    p.add_argument("--cands", default="id,tw0.85,tw1.15,lat0.25,lat-0.25,tw0.7,tw1.3,"
                                       "lat0.5,lat-0.5,tw0.6h2,tw1.4h2,rn20s0,rn20s1,rn50s0,rn50s1")
    p.add_argument("--rewinds", type=int, default=3)
    p.add_argument("--ramp_steps", type=int, default=10)
    p.add_argument("--ft_ckpt", default="")
    p.add_argument("--label_alt", choices=("ft", "base2"), default="ft")
    p.add_argument("--label_window", type=int, default=20)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    args = p.parse_args()
    if not args.splits_json or not args.shard_glob:
        raise SystemExit("--splits_json and --shard_glob are required")
    if int(args.cf_seed) in (777, int(args.eval_seed), 4242):
        raise SystemExit(f"--cf_seed {args.cf_seed} reuses a selection/eval seed (777, eval, 4242)")
    names = [n.strip() for n in (args.cf_splits or (
        "failure_train" if args.mode == "repair" else "failure_train,success_train")).split(",")
        if n.strip()]
    held = [n for n in names if n.endswith("_heldout")]
    if held:
        raise SystemExit(f"--cf_splits {held}: counterfactual data trains a policy/selector; "
                         f"held-out strata must stay unseen")
    caps = [int(c) for c in args.cf_caps.split(",") if c.strip()]
    caps = caps * len(names) if len(caps) == 1 else caps
    if len(caps) != len(names):
        raise SystemExit("--cf_caps must be one value or one per split")
    if args.mode == "labels" and args.label_alt == "ft" and not args.ft_ckpt:
        raise SystemExit("--mode labels --label_alt ft needs --ft_ckpt")
    cands = parse_candidates(args.cands) if args.mode == "repair" else []
    if args.mode == "repair" and (not cands or cands[0].family != "id"):
        raise SystemExit("--cands must start with 'id' (the reproduction check)")
    args.splits = ""
    args.probe_per_split = 0
    tuner = SafetyFineTuner(args)
    cf = Counterfactual(tuner, ft_ckpt=args.ft_ckpt, cands=cands, rewinds=args.rewinds,
                        ramp_steps=args.ramp_steps, label_alt=args.label_alt,
                        window=args.label_window)
    cf.rt_err = {"pos": float("nan"), "vel": float("nan"), "yaw": float("nan")}
    cf.run(Path(args.cf_out or args.out_dir), mode=args.mode, names=names, caps=caps,
           seed=int(args.cf_seed), chunk=max(1, int(args.eval_chunk)),
           shard_index=int(args.shard_index), num_shards=int(args.num_shards))


if __name__ == "__main__":
    main()
