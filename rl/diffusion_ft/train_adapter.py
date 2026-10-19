"""Distill the deploy-time repair into a gated tail adapter (rl/diffusion_ft/adapter.py).

INPUT. ``proposal_select --dump_pairs`` rows from a collection that ran the base
arm and the teacher arm (default ``o_lpA3``: the lateral/longitudinal repair
optimizer under the always-on projection, cv neighbours) on the *_train splits.
Every row is one alive replan: the frozen condition, the frozen prefix output
``x_{k-1}`` and the per-row late noise (so the base draw is REPLAYED exactly), the
base plan and the executed plan in world coordinates, the replan's world transform,
the base plan's scores, and the episode outcome.

POOLS (term-aware outcome, ``ep_solved_term``):
* recovery  -- teacher arm, the teacher SWITCHED, the episode solved: the repaired
  plan is the matching target (L_match), gate label 1;
* keep      -- base arm rows of solved episodes, and teacher-arm rows the teacher
  KEPT in solved episodes after its first switch (before it, they duplicate the
  base arm's): the base plan is the target (L_keep), gate label 0;
* failed    -- teacher arm, switched, episode NOT solved: no matching target, still
  charged the differentiable repair objective (L_repair), gate unlabeled.
Rows are split train/val by SCENE before anything is sampled.

STUDENT. From the cached state entering the last K steps (the base's own late
chain replayed through timesteps k-1..K, no adapter there) the adapted tail is run
with the row's own remaining noise, and every loss is charged on the FINAL sampled
trajectory, mapped to world coordinates with the row's transform. Gradients flow
through the K frozen-denoiser evaluations into the adapters and the gate only.

    L = L_match + keep_weight * L_keep + repair_weight * L_repair + gate_weight * L_gate

* L_match / L_keep: ``adapter.traj_distance`` (train_safety.preserve_loss's per-step
  metric: position m^2 + wrapped heading / 0.1 rad + speed m/s) over the scored
  window (world steps 1..n);
* L_repair: the repair optimizer's own objective on the student's plan -- barrier
  (hinge2, segment edges, 0.3 m / 0.75 m) + comfort/closeness of the edit against
  the base plan (``adapter.make_repair_objective``), on recovery + failed rows;
* L_gate: BCE on labeled rows.

STAGES. A: gate held open, adapters only, L_match + L_keep, L_repair switched on
part-way if imitation is making progress. B: adapters frozen, gate only, BCE on
bootstrap labels. Relabel: per teacher-switched row, the learned adapter held fully
on vs off on the same draw, scored by the teacher's own scorer -- positive where the
student's plan is clean (and clearer, for a margin-triggered row) at no more than
``tol_m`` of progress; unresolved rows unlabeled; keep rows stay 0. C: adapter +
gate jointly on the full objective with the refreshed labels, the softly gated
trajectory trained directly, and a forced-open recovery minibatch every few steps.

PREFLIGHT (before any update; both must PASS or nothing trains):
1. identity -- the adapter at initialization (gate open, strength 1; learned gate;
   strength 0) reproduces the base late chain run on the same rows in the same
   program, within ``--identity_tol`` (normalized; 1e-5 = 0.1 mm); and every row's
   stored randomness replays the DEPLOYED draw it was collected with, within
   ``--replay_tol`` (max) / ``--replay_tol_p99``. The second bound is looser
   because GPU float32 kernels (TF32 on Ampere) depend on the batch shape: the
   smoke measured 6.8e-5 max / 5.7e-5 p99 between batch-8 collection and batch-64
   replay (~0.1 mm in position; the rest is velocity and heading channels);
2. gradient -- one backward pass with the gate open: the adapters' final projection
   gets finite, nonzero gradients; the base parameters are unchanged by a step.

Gates follow the cluster convention: a failed check or a missing input prints why,
writes ``preflight.json`` and exits 0 with no checkpoint, so downstream jobs report
"not run" instead of stranding.

    python -m rl.diffusion_ft.train_adapter --out_dir DIR --pairs 'COLLECT/tasks/*/pair_data/*/*.npz' \\
        --shard_glob '/data/user_data/yixiz/autonlab/zfsauton/datasets/womd/tf_example/training/training_tfexample.tfrecord-*' \\
        --splits_json SPLITS3 --edge_margin 0.3 --agent_margin 0.75
"""

from __future__ import annotations

import dataclasses
import glob
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

# Fields a pair file must carry (proposal_select._pair_rows + run()).
PAIR_KEYS = ("cond1", "cond2", "mask", "x_prefix", "noise_late", "x_base", "world_base",
             "world_exec", "goal", "t0", "n", "switched", "feat", "scene_id", "repeat",
             "replan", "split", "arm", "ep_solved_term")
# feat columns (proposal_select.FEAT_COLS): base plan clear / edge / agent / progress.
CP, PP = 0, 3


def _now():
    return time.strftime("%H:%M:%S")


def param_group(path) -> str:
    """'adapters' or 'gate': the top-level key of a GatedTailAdapter param path."""
    top = path[0]
    return str(getattr(top, "key", getattr(top, "name", top)))


class AdapterTrainer:
    def __init__(self, args):
        import jax

        from rl.diffusion_ft.adapter import AdapterConfig, GatedTailAdapter, split_params
        from rl.diffusion_ft.train_safety import SafetyFineTuner

        self.args = args
        if args.sim_agents != "none":
            raise SystemExit("train_adapter reloads scenes and reads the other agents off the "
                             "log: it needs the log-replay collection (--sim_agents none)")
        self.out = Path(args.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out / "train_log.jsonl"
        self.jax = jax
        args.splits = ""              # nothing to preload: training reads pair files
        args.probe_per_split = 0
        self.tuner = SafetyFineTuner(args)
        self.diff = self.tuner.actor.diffusion
        if getattr(self.diff, "self_cond", False):
            raise SystemExit("the base checkpoint is self-conditioned; the tail adapter is not")
        self.k = int(args.k_trainable)
        self.floor = float(args.sigma_floor)
        self.pcfg = self.tuner.ckpt.preprocess_cfg
        self.acfg = AdapterConfig.for_unet(self.diff.denoise_fn, K=int(args.adapter_K),
                                           k_trainable=self.k, width=int(args.adapter_width),
                                           gate_hidden=int(args.gate_hidden), seed=int(args.seed))
        self.module = GatedTailAdapter(self.acfg)
        self.gdef, self.params, self.rest = split_params(self.module)
        from rl.diffusion_ft.lateral_opt import LateralOptConfig
        self.opt_cfg = LateralOptConfig.parse(args.repair_opt_cfg)
        self.dt = float(self.pcfg.world_dt_fallback)
        self.rng = np.random.default_rng(int(args.seed) * 7919 + 1)
        self.verdict = {"stage": "init"}
        print(f"[adapter] {self.acfg}  repair objective: {self.opt_cfg}", flush=True)

    # ------------------------------------------------------------------ #
    def log(self, row):
        row = {"time": _now(), **row}
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print("[adapter] " + "  ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
            flush=True)

    def gate_out(self, ok: bool, reason: str, **extra):
        """Write the preflight verdict; on failure print why and exit 0 (no checkpoint)."""
        self.verdict.update({"pass": bool(ok), "reason": reason, **extra})
        (self.out / "preflight.json").write_text(json.dumps(self.verdict, indent=1, default=float))
        if not ok:
            print(f"[adapter] NOT TRAINED: {reason}", flush=True)
            sys.exit(0)

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    def load_pairs(self):
        a = self.args
        files = sorted(glob.glob(a.pairs))
        if not files:
            self.gate_out(False, f"--pairs matches no files: {a.pairs}")
        parts, aux_keys = {}, None
        for f in files:
            z = np.load(f)
            missing = [k for k in PAIR_KEYS if k not in z.files]
            if missing:
                self.gate_out(False, f"{f} lacks {missing}: not a proposal_select --dump_pairs file")
            ak = sorted(k for k in z.files if k.startswith("aux/"))
            if aux_keys is None:
                aux_keys = ak
            elif ak != aux_keys:
                self.gate_out(False, f"{f}: aux keys {ak} differ from {aux_keys}")
            for k in list(PAIR_KEYS) + ak:
                parts.setdefault(k, []).append(z[k])
        D = {k: np.concatenate(v, 0) for k, v in parts.items()}
        self.aux_keys = aux_keys
        N = len(D["t0"])
        if D["noise_late"].shape[1] != self.k:
            self.gate_out(False, f"pairs carry a {D['noise_late'].shape[1]}-step late chain, "
                                 f"--k_trainable is {self.k}")
        arm = D["arm"].astype(str)
        teacher, base = a.teacher_arm, a.base_arm
        for nm in (teacher, base):
            if nm not in set(arm.tolist()):
                self.gate_out(False, f"arm '{nm}' has no rows (have {sorted(set(arm.tolist()))})")
        sw = D["switched"].astype(bool)
        solved = D["ep_solved_term"].astype(bool)
        is_t, is_b = arm == teacher, arm == base
        rec = is_t & sw & solved
        fail = is_t & sw & ~solved
        # The two arms share every draw until the teacher's first switch, so a
        # teacher-kept row before it duplicates the base arm's row. Only kept rows
        # AFTER the episode diverged add states the base arm never visited.
        ep = np.char.add(np.char.add(D["split"].astype(str), ":"),
                         np.char.add(D["scene_id"].astype(str), np.char.add(":", D["repeat"].astype(str))))
        first_sw = {}
        for e, r in zip(ep[is_t & sw], D["replan"][is_t & sw]):
            first_sw[e] = min(first_sw.get(e, 10 ** 6), int(r))
        diverged = np.array([int(r) > first_sw.get(e, 10 ** 6) for e, r in zip(ep, D["replan"])])
        keep = (is_b & solved) | (is_t & ~sw & solved & diverged)
        # Held-out validation SCENES, fixed before any sampling.
        scenes = np.unique(D["scene_id"])
        perm = np.random.default_rng(int(a.seed) + 17).permutation(scenes)
        val_scenes = set(perm[:int(round(float(a.val_frac) * len(scenes)))].tolist())
        is_val = np.isin(D["scene_id"], list(val_scenes))
        use = rec | fail | keep
        idx = np.nonzero(use)[0]
        self.D = {k: v[idx] for k, v in D.items()}
        self.N = len(idx)
        self.pool = {"rec": rec[idx], "fail": fail[idx], "keep": keep[idx]}
        self.is_val = is_val[idx]
        split = self.D["split"].astype(str)
        self.from_fail_split = np.char.startswith(split, "failure")
        cp = self.D["feat"][:, CP]
        self.trigger_violation = cp < 0.0
        # Bootstrap gate labels: 1 recovery, 0 keep, -1 (unlabeled) failed teacher.
        self.label = np.where(self.pool["rec"], 1, np.where(self.pool["keep"], 0, -1)).astype(np.int8)
        W = self.D["world_base"].shape[1]
        n = self.D["n"].astype(np.int64)
        steps = np.arange(W)[None, :]
        self.step_w = ((steps >= 1) & (steps <= n[:, None])).astype(np.float32)
        counts = {}
        for p in ("rec", "fail", "keep"):
            for nm, m in (("train", ~self.is_val), ("val", self.is_val)):
                sel = self.pool[p] & m
                counts[f"{p}_{nm}"] = int(sel.sum())
                counts[f"{p}_{nm}_scenes"] = int(len(np.unique(self.D["scene_id"][sel])))
        counts["rec_violation"] = int((self.pool["rec"] & self.trigger_violation).sum())
        counts["rec_tight"] = int((self.pool["rec"] & ~self.trigger_violation).sum())
        counts["keep_from_base_arm"] = int((self.pool["keep"] & (arm[idx] == base)).sum())
        counts["rows_loaded"] = int(N)
        counts["files"] = len(files)
        counts["val_scenes"] = len(val_scenes)
        self.counts = counts
        print(f"[adapter] pairs: {json.dumps(counts)}", flush=True)
        for p in ("rec", "keep"):
            if counts[f"{p}_train"] == 0:
                self.gate_out(False, f"no '{p}' rows in the training scenes", counts=counts)

    def rows(self, idx):
        """Host -> device batch for row indices ``idx`` (numpy int array)."""
        jnp = self.jax.numpy
        D = self.D
        b = {"cond1": D["cond1"][idx], "cond2": D["cond2"][idx], "mask": D["mask"][idx],
             "aux": {k[4:]: jnp.asarray(D[k][idx]) for k in self.aux_keys}}
        out = {k: jnp.asarray(v) for k, v in b.items() if k != "aux"}
        out["aux"] = b["aux"]
        return out

    def precompute(self):
        """Per row: the state entering the tail (base late steps k-1..K), the frozen
        gate features at the first adapted step, and the replay of the whole late
        chain from the stored prefix -- which must land on the stored base draw."""
        jax, jnp = self.jax, self.jax.numpy
        from rl.diffusion_ft.adapter import base_steps, first_step_gate_features

        diff, acfg, k, K, floor = self.diff, self.acfg, self.k, self.acfg.K, self.floor

        @jax.jit
        def prep(c1, c2, m, x_prefix, noise_bkct):
            noise = jnp.swapaxes(noise_bkct, 0, 1)
            x_tail = base_steps(diff, c1, c2, m, x_prefix, noise[:k - K], t_hi=k - 1,
                                sigma_floor=floor)
            x_rep = base_steps(diff, c1, c2, m, x_tail, noise[k - K:], t_hi=K - 1,
                               sigma_floor=floor)
            return x_tail, x_rep, first_step_gate_features(diff, acfg, c1, c2, m, x_tail)

        C = int(self.args.chunk)
        x_tail, err, feat = [], [], []
        t0 = time.time()
        for s in range(0, self.N, C):
            i = np.arange(s, min(s + C, self.N))
            pad = C - len(i)
            j = np.concatenate([i, np.full(pad, i[-1])]) if pad else i
            b = self.rows(j)
            xt, xr, f = prep(b["cond1"], b["cond2"], b["mask"],
                             jnp.asarray(self.D["x_prefix"][j]), jnp.asarray(self.D["noise_late"][j]))
            xt, xr, f = (np.asarray(v)[:len(i)] for v in (xt, xr, f))
            x_tail.append(xt); feat.append(f)
            err.append(np.max(np.abs(xr - self.D["x_base"][i]), axis=(1, 2)))
        self.x_tail = np.concatenate(x_tail, 0)
        self.feat = np.concatenate(feat, 0)
        self.replay_err = np.concatenate(err, 0)
        self.noise_tail = np.ascontiguousarray(self.D["noise_late"][:, k - K:])
        print(f"[adapter] precomputed {self.N} rows in {time.time() - t0:.0f}s; replay of the "
              f"stored late chain vs the stored base draw: max {self.replay_err.max():.2e} "
              f"p99 {np.quantile(self.replay_err, 0.99):.2e} (normalized; x100 = metres)",
              flush=True)

    # ------------------------------------------------------------------ #
    # Scenes for the repair objective and the teacher's scorer
    # ------------------------------------------------------------------ #
    def load_scene_cache(self):
        """Every scene a teacher-switched row came from, loaded once and held on the
        host; a minibatch gathers its rows' scenes. ``candidate_safety`` reads only
        the log, the object metadata, the roadgraph and (cv) the agents' state at
        t0, which under log replay is the log -- so a reloaded scene scores a stored
        replan as the live state did. Checked: the teacher's own scorer on the stored
        base plans must reproduce the scores it recorded.

        A loaded state's ``sim_trajectory`` is zeros past timestep 0 (waymax fills it
        as the episode runs), and the cv predictor reads the agents' pose at t0 from
        it. Under log replay (``--sim_agents none``, the collection's) every non-ego
        agent's simulated state IS its log, and the ego is masked out of its own
        neighbour set, so the cache sets ``sim_trajectory = log_trajectory``."""
        jax = self.jax
        need = self.pool["rec"] | self.pool["fail"]
        ids = np.unique(self.D["scene_id"][need])
        self.scene_pos = {int(s): i for i, s in enumerate(ids)}
        chunks = []
        t0 = time.time()
        for s in range(0, len(ids), 32):
            st = self.tuner.sampler.load_explicit([int(x) for x in ids[s:s + 32]])
            st = st.replace(sim_trajectory=st.log_trajectory)
            chunks.append(jax.tree_util.tree_map(np.asarray, st))
        n_c = [len(ids[s:s + 32]) for s in range(0, len(ids), 32)]

        def cat(*leaves):
            if all(np.ndim(l) >= 1 and np.shape(l)[0] == c for l, c in zip(leaves, n_c)):
                return np.concatenate(leaves, 0)
            return leaves[0]

        self.scenes = jax.tree_util.tree_map(cat, *chunks) if len(chunks) > 1 else chunks[0]
        self.n_scenes = len(ids)
        print(f"[adapter] scene cache: {len(ids)} scenes in {time.time() - t0:.0f}s", flush=True)
        from rl.diffusion_ft.proposal_select import make_scorer
        self._scorer = make_scorer(self.tuner.objcfg, "cv", self.tuner.env.world_dt_method)
        rows = np.nonzero(need)[0]
        sc = self.score_worlds(self.D["world_base"][rows], rows)
        d = np.abs(sc[0] - self.D["feat"][rows, CP])
        ok = bool(np.quantile(d, 0.99) < 1e-3 and d.max() < 0.05)
        print(f"[adapter] reloaded-scene scorer vs the teacher's recorded base clearance on "
              f"{len(rows)} rows: max {d.max():.2e} m, p99 {np.quantile(d, 0.99):.2e} m", flush=True)
        self.verdict["scene_reload_err_m"] = {"max": float(d.max()), "p99": float(np.quantile(d, 0.99))}
        if not ok:
            self.gate_out(False, "the reloaded scenes do not reproduce the teacher's scores; "
                                 "L_repair and the relabel would score a different scene")

    def scene_batch(self, rows):
        jax = self.jax
        pos = np.asarray([self.scene_pos[int(s)] for s in self.D["scene_id"][rows]])
        return jax.tree_util.tree_map(
            lambda x: jax.numpy.asarray(x[pos]) if np.ndim(x) >= 1 and np.shape(x)[0] == self.n_scenes
            else jax.numpy.asarray(x), self.scenes)

    def score_worlds(self, worlds, rows):
        """Teacher's scorer (cv neighbours) on world plans ``[R, W, 5]`` of ``rows``:
        ``(clear, edge, agent, prog)``, each ``[R]``. Grouped by window length."""
        jnp = self.jax.numpy
        out = np.full((4, len(rows)), np.nan, np.float64)
        n = self.D["n"][rows].astype(int)
        C = int(self.args.chunk)
        for nv in np.unique(n):
            sel = np.nonzero(n == nv)[0]
            for s in range(0, len(sel), C):
                j = sel[s:s + C]
                r = rows[j]
                v = self._scorer(jnp.asarray(worlds[j]), self.scene_batch(r),
                                 jnp.asarray(self.D["t0"][r], jnp.int32),
                                 jnp.asarray(self.D["goal"][r]), int(nv))
                out[:, j] = np.stack([np.asarray(x) for x in v])
        return out

    # ------------------------------------------------------------------ #
    # Student
    # ------------------------------------------------------------------ #
    def build_fns(self):
        jax, jnp = self.jax, self.jax.numpy
        import optax
        from flax import nnx

        from data.postprocess import postprocess_predictions
        from rl.diffusion_ft.adapter import adapted_steps, make_repair_objective, traj_distance

        a = self.args
        diff, acfg, floor, pcfg = self.diff, self.acfg, self.floor, self.pcfg
        gdef, rest = self.gdef, self.rest
        n_rep = int(a.repair_window)
        self.repair_obj = make_repair_objective(
            self.tuner.objcfg, self.opt_cfg, n=n_rep, dt=self.dt, neighbor_pred="cv",
            world_dt_method=self.tuner.env.world_dt_method)
        kw, rw, gw = float(a.keep_weight), float(a.adapter_repair_weight), float(a.gate_weight)

        def to_world(x_bct, aux):
            post = postprocess_predictions(jnp.transpose(x_bct, (0, 2, 1)), aux, pcfg)
            return post["trajectory_world_world_dt"]

        def student(params, b, gate_policy, strength):
            """Final sampled plan in world coords, the gate used and its logit."""
            module = nnx.merge(gdef, params, rest)
            logit = module.gate(b["feat"])
            if gate_policy == "open":
                gate = jnp.ones_like(logit)
            elif gate_policy == "closed":
                gate = jnp.zeros_like(logit)
            else:
                gate = jnp.where(b["force_open"] > 0.5, 1.0, jax.nn.sigmoid(logit))
            x0, _, _ = adapted_steps(diff, module, acfg, b["cond1"], b["cond2"], b["mask"],
                                     b["x_tail"], jnp.swapaxes(b["noise_tail"], 0, 1),
                                     strength=strength, gate=gate, sigma_floor=floor)
            return to_world(x0, b["aux"]), gate, logit, x0

        def wmean(v, w):
            return jnp.sum(v * w) / jnp.maximum(jnp.sum(w), 1.0)

        def loss(params, b, state, *, gate_policy, use_repair, use_gate, n_rep_rows):
            world, gate, logit, _ = student(params, b, gate_policy, 1.0)
            d = traj_distance(world, b["target"], b["step_w"])
            L_match, L_keep = wmean(d, b["w_match"]), wmean(d, b["w_keep"])
            total = L_match + kw * L_keep
            m = {"match": L_match, "keep": L_keep,
                 "gate_rec": wmean(gate, b["w_match"]), "gate_keep": wmean(gate, b["w_keep"])}
            if use_repair:
                r = n_rep_rows
                per, parts = self.repair_obj(world[:r], b["world_base"][:r], state, b["t0"][:r])
                wr = b["w_rep"][:r]
                L_rep = wmean(per, wr)
                total = total + rw * L_rep
                m.update({"repair": L_rep, "barrier": wmean(parts["barrier"], wr),
                          "comfort": wmean(parts["comfort"], wr)})
            if use_gate:
                lab = b["label"]
                lm = (lab >= 0).astype(jnp.float32)
                bce = optax.sigmoid_binary_cross_entropy(logit, jnp.clip(lab, 0, 1).astype(jnp.float32))
                L_gate = wmean(bce, lm)
                total = total + gw * L_gate
                m["gate_bce"] = L_gate
            m["total"] = total
            return total, m

        self._loss_fns = {}

        def loss_fn(gate_policy, use_repair, use_gate, n_rep_rows):
            key = (gate_policy, use_repair, use_gate, n_rep_rows)
            if key not in self._loss_fns:
                self._loss_fns[key] = jax.jit(jax.value_and_grad(
                    lambda p, b, s: loss(p, b, s, gate_policy=gate_policy, use_repair=use_repair,
                                         use_gate=use_gate, n_rep_rows=n_rep_rows),
                    has_aux=True))
            return self._loss_fns[key]

        def gate_bce(params, feat, label):
            module = nnx.merge(gdef, params, rest)
            logit = module.gate(feat)
            lab = label.astype(jnp.float32)
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logit, lab)), logit

        self.loss_fn = loss_fn
        self.gate_grad = jax.jit(jax.value_and_grad(gate_bce, has_aux=True))
        self.gate_logit = jax.jit(lambda p, f: nnx.merge(gdef, p, rest).gate(f))
        self.student_eval = jax.jit(lambda p, b, s, policy: student(p, b, policy, s),
                                    static_argnums=(3,))
        self.to_world = jax.jit(to_world)

    def batch(self, idx, *, target, w_match, w_keep, w_rep, force_open):
        """Device batch: rows ``idx`` with per-row targets and loss weights."""
        jnp = self.jax.numpy
        b = self.rows(idx)
        b.update({"x_tail": jnp.asarray(self.x_tail[idx]),
                  "noise_tail": jnp.asarray(self.noise_tail[idx]),
                  "feat": jnp.asarray(self.feat[idx]),
                  "target": jnp.asarray(target), "step_w": jnp.asarray(self.step_w[idx]),
                  "world_base": jnp.asarray(self.D["world_base"][idx]),
                  "t0": jnp.asarray(self.D["t0"][idx], jnp.int32),
                  "label": jnp.asarray(self.label[idx], jnp.int32),
                  "w_match": jnp.asarray(w_match, jnp.float32),
                  "w_keep": jnp.asarray(w_keep, jnp.float32),
                  "w_rep": jnp.asarray(w_rep, jnp.float32),
                  "force_open": jnp.asarray(force_open, jnp.float32)})
        return b

    def train_batch(self, *, force_open_rec: bool):
        """[recovery | failed | keep] rows from the TRAINING scenes; returns
        (batch, rows, n_rep_rows). Failed rows carry only the repair objective."""
        a = self.args
        tr = ~self.is_val
        pick = lambda m, c: (self.rng.choice(np.nonzero(m)[0], size=c, replace=int(m.sum()) < c)
                             if c > 0 and m.any() else np.zeros(0, np.int64))
        r_rec = pick(self.pool["rec"] & tr, int(a.batch_rec))
        r_fail = pick(self.pool["fail"] & tr, int(a.batch_fail))
        nk = int(a.batch_keep)
        kf = self.pool["keep"] & tr & self.from_fail_split
        ks = self.pool["keep"] & tr & ~self.from_fail_split
        if kf.any() and ks.any():
            r_keep = np.concatenate([pick(kf, nk // 2), pick(ks, nk - nk // 2)])
        else:
            r_keep = pick(self.pool["keep"] & tr, nk)
        rows = np.concatenate([r_rec, r_fail, r_keep]).astype(np.int64)
        nr, nf = len(r_rec), len(r_fail)
        is_rec = np.arange(len(rows)) < nr
        is_keep = np.arange(len(rows)) >= nr + nf
        target = np.where(is_keep[:, None, None], self.D["world_base"][rows], self.D["world_exec"][rows])
        full = self.D["n"][rows] >= int(a.repair_window)
        b = self.batch(rows, target=target, w_match=is_rec, w_keep=is_keep,
                       w_rep=(~is_keep) & full, force_open=is_rec & force_open_rec)
        return b, rows, nr + nf

    # ------------------------------------------------------------------ #
    def eval_plans(self, params, rows, *, policy, strength=1.0):
        """``(world [R, W, 5], gate [R], x0 [R, C, T])`` for ``rows``, chunked."""
        C = int(self.args.chunk)
        W, G, X = [], [], []
        for s in range(0, len(rows), C):
            i = rows[s:s + C]
            pad = C - len(i)
            j = np.concatenate([i, np.full(pad, i[-1])]) if pad else i
            b = self.batch(j, target=self.D["world_base"][j], w_match=np.zeros(len(j)),
                           w_keep=np.zeros(len(j)), w_rep=np.zeros(len(j)),
                           force_open=np.zeros(len(j)))
            w, g, _, x0 = self.student_eval(params, b, float(strength), policy)
            W.append(np.asarray(w)[:len(i)]); G.append(np.asarray(g)[:len(i)])
            X.append(np.asarray(x0)[:len(i)])
        return np.concatenate(W), np.concatenate(G), np.concatenate(X)

    def dist(self, worlds, rows, target):
        """Per-row ``traj_distance`` and position RMS (m) over the scored window."""
        jnp = self.jax.numpy
        from rl.diffusion_ft.adapter import traj_distance
        d = np.asarray(traj_distance(jnp.asarray(worlds), jnp.asarray(target),
                                     jnp.asarray(self.step_w[rows])))
        w = self.step_w[rows]
        dp2 = np.sum((worlds[..., :2] - target[..., :2]) ** 2, -1)
        rms = np.sqrt(np.sum(dp2 * w, 1) / np.maximum(w.sum(1), 1))
        return d, rms

    def useful(self, rows, sc_on):
        """Did the student's plan intervene usefully at these teacher-switched rows?
        1 = clean and within tol of the base's progress (and, where the base was
        clean but inside the margin, at least PROJ_MIN_GAIN_M clearer -- the
        teacher's own projection rule); 0 = not."""
        from rl.diffusion_ft.proposal_select import PROJ_MIN_GAIN_M
        cp, pp = self.D["feat"][rows, CP], self.D["feat"][rows, PP]
        clean = sc_on[0] >= 0.0
        prog_ok = sc_on[3] >= pp - float(self.args.tol_m)
        gain = (cp < 0.0) | (sc_on[0] >= cp + PROJ_MIN_GAIN_M)
        return clean & prog_ok & gain

    def validate(self, params, tag, *, policies=("open", "learned")):
        """Offline metrics on the VALIDATION scenes (fixed subsample)."""
        out = {"stage": "val", "tag": tag}
        rec, keep = self.val_rows["rec"], self.val_rows["keep"]
        for pol in policies:
            if len(rec):
                w, g, _ = self.eval_plans(params, rec, policy=pol)
                d, rms = self.dist(w, rec, self.D["world_exec"][rec])
                db, _ = self.dist(self.D["world_base"][rec], rec, self.D["world_exec"][rec])
                sc = self.score_worlds(w, rec)
                u = self.useful(rec, sc)
                out.update({f"{pol}/rec_match": float(d.mean()), f"{pol}/rec_rms_m": float(rms.mean()),
                            f"{pol}/rec_closed": float(1.0 - d.mean() / max(db.mean(), 1e-9)),
                            f"{pol}/rec_useful": float(u.mean()), f"{pol}/gate_rec": float(g.mean())})
            if len(keep):
                w, g, _ = self.eval_plans(params, keep, policy=pol)
                d, rms = self.dist(w, keep, self.D["world_base"][keep])
                out.update({f"{pol}/keep": float(d.mean()), f"{pol}/keep_rms_m": float(rms.mean()),
                            f"{pol}/keep_p95_m": float(np.quantile(rms, 0.95)),
                            f"{pol}/gate_keep": float(g.mean())})
        lab = self.label[self.val_rows["labeled"]]
        if len(lab) and 0 < lab.mean() < 1:
            lg = np.asarray(self.gate_logit(params, self.jax.numpy.asarray(self.feat[self.val_rows["labeled"]])))
            out["gate_auc"] = _auc(lg, lab)
        self.log(out)
        return out

    def pick_val(self):
        cap = int(self.args.val_rows)
        rng = np.random.default_rng(int(self.args.seed) + 3)
        take = lambda m: np.sort(rng.choice(np.nonzero(m)[0], size=min(cap, int(m.sum())), replace=False)
                                 ) if m.any() else np.zeros(0, np.int64)
        v = self.is_val
        self.val_rows = {"rec": take(self.pool["rec"] & v), "keep": take(self.pool["keep"] & v)}
        self.val_rows["labeled"] = np.concatenate([self.val_rows["rec"], self.val_rows["keep"]])

    # ------------------------------------------------------------------ #
    # Preflight
    # ------------------------------------------------------------------ #
    def identity(self, rows, arms):
        """Max |x - base| per (name, gate policy, strength) in ``arms``, against the
        base late chain recomputed here (gate closed: the same program, residual 0)
        and against the stored deployed draws. Returns (result dict, pass)."""
        a = self.args
        _, _, x_ref = self.eval_plans(self.params, rows, policy="closed", strength=1.0)
        stored = np.max(np.abs(x_ref - self.D["x_base"][rows]), axis=(1, 2))
        res = {"rows": int(len(rows)), "tol": float(a.identity_tol),
               "replay_tol": float(a.replay_tol), "replay_tol_p99": float(a.replay_tol_p99),
               "replay_max": float(self.replay_err.max()),
               "replay_p99": float(np.quantile(self.replay_err, 0.99)),
               "closed_vs_stored_max": float(stored.max())}
        ok = (res["replay_max"] <= res["replay_tol"] and res["replay_p99"] <= res["replay_tol_p99"]
              and res["closed_vs_stored_max"] <= res["replay_tol"])
        for nm, pol, s in arms:
            _, _, x0 = self.eval_plans(self.params, rows, policy=pol, strength=s)
            res[f"{nm}_max"] = float(np.max(np.abs(x0 - x_ref)))
            res[f"{nm}_vs_stored_max"] = float(np.max(np.abs(x0 - self.D["x_base"][rows])))
            ok = ok and res[f"{nm}_max"] <= res["tol"]
        res["pass"] = bool(ok)
        return res, bool(ok)

    def preflight(self):
        jax, jnp = self.jax, self.jax.numpy
        # 1. identity
        rows = np.arange(self.N)
        if len(rows) > int(self.args.identity_rows):
            rows = np.sort(self.rng.choice(rows, int(self.args.identity_rows), replace=False))
        res, ok1 = self.identity(rows, (("init_open", "open", 1.0), ("init_learned", "learned", 1.0),
                                        ("init_s0", "learned", 0.0)))
        self.verdict["identity"] = res
        print(f"[adapter] PREFLIGHT 1 identity ({len(rows)} rows, tol {res['tol']:g} vs the base "
              f"chain; replay tol {res['replay_tol']:g} / p99 {res['replay_tol_p99']:g} vs the "
              f"deployed draws): " + "  ".join(f"{k}={v:.2e}" for k, v in res.items()
                                               if k.endswith(("_max", "_p99")) and "tol" not in k)
              + f"  -> {'PASS' if ok1 else 'FAIL'}", flush=True)
        # 2. gradient, gate held open, matching + preservation (no repair term yet)
        b, _, nrep = self.train_batch(force_open_rec=True)
        (val, m), g = self.loss_fn("open", False, False, nrep)(self.params, b, None)
        flat = {"/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path): np.asarray(x)
                for path, x in jax.tree_util.tree_flatten_with_path(g)[0]}
        up = {k: float(np.linalg.norm(v)) for k, v in flat.items() if "up" in k}
        down = {k: float(np.linalg.norm(v)) for k, v in flat.items() if "down" in k}
        finite = all(np.all(np.isfinite(v)) for v in flat.values())
        up_ok = all(v > 0 for v in up.values()) and len(up) >= 2 * self.acfg.n_blocks
        before = self.tuner.actor.frozen_state()
        p2 = self.adam_step(self.params, g, ("adapters",), self._fresh_opt())[0]
        frozen_ok = self.tuner.actor.frozen_params_unchanged(before)
        moved = any(not np.array_equal(np.asarray(x), np.asarray(y)) for x, y in
                    zip(jax.tree_util.tree_leaves(p2), jax.tree_util.tree_leaves(self.params)))
        ok2 = bool(finite and up_ok and frozen_ok and moved and np.isfinite(float(val)))
        self.verdict["gradient"] = {"loss": float(val), "match": float(m["match"]),
                                    "keep": float(m["keep"]), "finite": bool(finite),
                                    "up_grad_norms": up, "down_grad_norms": down,
                                    "base_frozen": bool(frozen_ok), "adapter_moved": bool(moved),
                                    "pass": ok2}
        print(f"[adapter] PREFLIGHT 2 gradient: loss {float(val):.4g} (match {float(m['match']):.4g}, "
              f"keep {float(m['keep']):.2e}); final projection grad norms "
              + ", ".join(f"{v:.3g}" for v in up.values())
              + "; down-projection " + ", ".join(f"{v:.1e}" for v in down.values())
              + f" (zero is expected at init); finite={finite} base_frozen={frozen_ok} "
              f"-> {'PASS' if ok2 else 'FAIL'}", flush=True)
        self.verdict["stage"] = "preflight"
        self.gate_out(ok1 and ok2, "preflight passed" if ok1 and ok2 else
                      "preflight failed: " + ", ".join(n for n, o in (("identity", ok1), ("gradient", ok2))
                                                     if not o))

    # ------------------------------------------------------------------ #
    # Optimizer
    # ------------------------------------------------------------------ #
    def _fresh_opt(self):
        import optax
        self.tx = optax.chain(optax.clip_by_global_norm(float(self.args.clip)),
                              optax.adam(float(self.args.lr)))
        return self.tx.init(self.params)

    def adam_step(self, params, grads, train_groups, opt_state):
        """One clipped Adam step on the named groups ('adapters', 'gate'); the other
        group's gradient is zeroed (fresh optimizer state per stage, so a zero
        gradient is a zero update)."""
        import optax
        jax = self.jax

        def keep(path, g):
            return g if param_group(path) in train_groups else jax.numpy.zeros_like(g)

        grads = jax.tree_util.tree_map_with_path(keep, grads)
        gn = float(optax.global_norm(grads))
        upd, opt_state = self.tx.update(grads, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, gn

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #
    def save(self, tag, **extra):
        from rl.diffusion_ft.adapter import save_adapter
        save_adapter(self.out / tag, self.acfg, self.params,
                     {"base_ckpt": self.args.ckpt, "teacher_arm": self.args.teacher_arm,
                      "pairs": self.args.pairs, "stage": tag, **extra})

    def stage_a(self):
        a = self.args
        steps = int(a.stage_a_steps)
        opt = self._fresh_opt()
        v0 = self.validate(self.params, "A0", policies=("open",))
        base_match = v0.get("open/rec_match", np.nan)
        use_rep = False
        start = int(round(float(a.repair_start_frac) * steps))
        last_v = (0, v0)
        for it in range(steps):
            if it == start and float(a.adapter_repair_weight) > 0:
                v = (last_v[1] if last_v[0] == it
                     else self.validate(self.params, f"A{it}", policies=("open",)))
                prog = 1.0 - v.get("open/rec_match", np.nan) / max(base_match, 1e-9)
                use_rep = bool(prog >= float(a.repair_min_progress))
                self.log({"stage": "A", "step": it, "event": "repair_on" if use_rep else "repair_off",
                          "match_progress": float(prog), "need": float(a.repair_min_progress)})
            b, rows, nrep = self.train_batch(force_open_rec=True)
            state = self.scene_batch(rows[:nrep]) if use_rep else None
            (val, m), g = self.loss_fn("open", use_rep, False, nrep)(self.params, b, state)
            self.params, opt, gn = self.adam_step(self.params, g, ("adapters",), opt)
            if it % int(a.log_every) == 0 or it == steps - 1:
                self.log({"stage": "A", "step": it, "grad_norm": gn,
                          **{k: float(v) for k, v in m.items()}})
            if (it + 1) % int(a.val_every) == 0 and it + 1 < steps:
                last_v = (it + 1, self.validate(self.params, f"A{it + 1}", policies=("open",)))
        self.repair_used_in_a = use_rep
        self.validate(self.params, "A_end", policies=("open",))
        self.save("adapter_A", repair_in_stage_a=use_rep)

    def stage_b(self, tag="B"):
        a = self.args
        jnp = self.jax.numpy
        opt = self._fresh_opt()
        tr = ~self.is_val
        pos = np.nonzero((self.label == 1) & tr)[0]
        neg = np.nonzero((self.label == 0) & tr)[0]
        if not len(pos) or not len(neg):
            self.log({"stage": tag, "event": "skipped", "pos": int(len(pos)), "neg": int(len(neg))})
            return
        half = int(a.gate_batch) // 2
        for it in range(int(a.stage_b_steps)):
            r = np.concatenate([self.rng.choice(pos, half, replace=len(pos) < half),
                                self.rng.choice(neg, half, replace=len(neg) < half)])
            (val, _), g = self.gate_grad(self.params, jnp.asarray(self.feat[r]),
                                         jnp.asarray(self.label[r]))
            self.params, opt, gn = self.adam_step(self.params, g, ("gate",), opt)
            if it % int(a.log_every) == 0:
                self.log({"stage": tag, "step": it, "gate_bce": float(val), "grad_norm": gn,
                          "pos": int(len(pos)), "neg": int(len(neg))})
        self.validate(self.params, f"{tag}_end")
        self.save(f"adapter_{tag}")

    def relabel(self):
        """Refresh labels with the LEARNED adapter, fully on vs off, same draws."""
        rows = np.nonzero(self.pool["rec"] | self.pool["fail"])[0]
        w_on, _, _ = self.eval_plans(self.params, rows, policy="open")
        sc = self.score_worlds(w_on, rows)
        u = self.useful(rows, sc)
        old = self.label.copy()
        self.label[rows] = np.where(u, 1, -1).astype(np.int8)
        rec = self.pool["rec"][rows]
        stats = {"stage": "relabel", "teacher_switched": int(len(rows)),
                 "rec_rows": int(rec.sum()), "rec_student_useful": int((u & rec).sum()),
                 "rec_unrecovered": int((~u & rec).sum()),
                 "fail_rows": int((~rec).sum()), "fail_student_useful": int((u & ~rec).sum()),
                 "pos_before": int((old == 1).sum()), "pos_after": int((self.label == 1).sum()),
                 "neg": int((self.label == 0).sum()), "unlabeled": int((self.label < 0).sum())}
        for nm, m in (("violation", self.trigger_violation[rows]), ("tight", ~self.trigger_violation[rows])):
            stats[f"rec_{nm}"] = int((rec & m).sum())
            stats[f"rec_{nm}_useful"] = int((rec & m & u).sum())
        self.log(stats)
        with (self.out / "student_unrecovered.jsonl").open("w") as fh:
            for i, r in enumerate(rows):
                if rec[i] and not u[i]:
                    fh.write(json.dumps({
                        "scene_id": int(self.D["scene_id"][r]), "repeat": int(self.D["repeat"][r]),
                        "replan": int(self.D["replan"][r]), "split": str(self.D["split"][r]),
                        "val": bool(self.is_val[r]), "base_clear": float(self.D["feat"][r, CP]),
                        "student_clear": float(sc[0, i]), "student_prog": float(sc[3, i]),
                        "base_prog": float(self.D["feat"][r, PP])}) + "\n")
        self.relabel_stats = stats

    def stage_c(self):
        a = self.args
        steps = int(a.stage_c_steps)
        opt = self._fresh_opt()
        use_rep = float(a.adapter_repair_weight) > 0
        every = max(int(a.force_open_every), 0)
        for it in range(steps):
            force = every > 0 and it % every == 0
            b, rows, nrep = self.train_batch(force_open_rec=force)
            state = self.scene_batch(rows[:nrep]) if use_rep else None
            (val, m), g = self.loss_fn("learned", use_rep, float(a.gate_weight) > 0, nrep)(
                self.params, b, state)
            self.params, opt, gn = self.adam_step(self.params, g, ("adapters", "gate"), opt)
            if it % int(a.log_every) == 0 or it == steps - 1:
                self.log({"stage": "C", "step": it, "grad_norm": gn, "forced_open": bool(force),
                          **{k: float(v) for k, v in m.items()}})
            if (it + 1) % int(a.val_every) == 0:
                self.validate(self.params, f"C{it + 1}")
                if it + 1 < steps:
                    self.save(f"adapter_C{it + 1:04d}")
        self.save("adapter_final", relabel=getattr(self, "relabel_stats", None))

    # ------------------------------------------------------------------ #
    # Diagnostic: can it fit a fixed small set at all?
    # ------------------------------------------------------------------ #
    def overfit(self):
        """Fit a FIXED set of ``--overfit_rows`` training recovery rows, full batch
        (no minibatch noise), gate open, L_match only, from a fresh init per
        ``width:lr`` config in ``--overfit_configs``. Separates "the tail adapter
        cannot express the repair" (the loss stalls even here) from "the pilot's
        optimization stalled" (it fits here). Logs the exact loss on those rows every
        ``--overfit_log_every`` steps; the teacher-scorer `useful` rate and position
        RMS every ``--overfit_eval_every``; the held-out validation match at the end.
        Writes ``overfit.json``; no checkpoint is kept."""
        a = self.args
        from rl.diffusion_ft.adapter import GatedTailAdapter, split_params
        rng = np.random.default_rng(int(a.seed) + 29)
        pool = np.nonzero(self.pool["rec"] & ~self.is_val)[0]
        rows = np.sort(rng.choice(pool, size=min(int(a.overfit_rows), len(pool)), replace=False))
        n = len(rows)
        target = self.D["world_exec"][rows]
        d_base, rms_base = self.dist(self.D["world_base"][rows], rows, target)
        viol = self.trigger_violation[rows]
        u_teacher = self.useful(rows, self.score_worlds(target, rows))
        b = self.batch(rows, target=target, w_match=np.ones(n), w_keep=np.zeros(n),
                       w_rep=np.zeros(n), force_open=np.ones(n))
        vr = self.val_rows["rec"]
        dvb, _ = self.dist(self.D["world_base"][vr], vr, self.D["world_exec"][vr])
        out = {"rows": int(n), "scenes": int(len(np.unique(self.D["scene_id"][rows]))),
               "violation_rows": int(viol.sum()), "base_match": float(d_base.mean()),
               "base_rms_m": float(rms_base.mean()), "teacher_useful": float(u_teacher.mean()),
               "steps": int(a.overfit_steps), "configs": {}}
        self.log({"stage": "overfit", "event": "set", **{k: v for k, v in out.items() if k != "configs"}})
        steps, le, ee = int(a.overfit_steps), int(a.overfit_log_every), int(a.overfit_eval_every)
        for spec in [x.strip() for x in a.overfit_configs.split(",") if x.strip()]:
            w, lr = spec.split(":")
            self.acfg = dataclasses.replace(self.acfg, width=int(w))
            self.module = GatedTailAdapter(self.acfg)
            self.gdef, self.params, self.rest = split_params(self.module)
            self.build_fns()
            a.lr = float(lr)
            opt = self._fresh_opt()
            fn = self.loss_fn("open", False, False, n)
            curve, t0 = [], time.time()
            for it in range(steps + 1):
                (_, m), g = fn(self.params, b, None)
                if it % le == 0 or it == steps:
                    rec = {"step": it, "match": float(m["match"]),
                           "closed": 1.0 - float(m["match"]) / float(d_base.mean())}
                    if it % ee == 0 or it == steps:
                        w_on, _, _ = self.eval_plans(self.params, rows, policy="open")
                        _, rms = self.dist(w_on, rows, target)
                        u = self.useful(rows, self.score_worlds(w_on, rows))
                        rec.update({"rms_m": float(rms.mean()), "useful": float(u.mean()),
                                    "useful_violation": float(u[viol].mean()) if viol.any() else None,
                                    "useful_tight": float(u[~viol].mean()) if (~viol).any() else None})
                    curve.append(rec)
                    self.log({"stage": "overfit", "config": spec, **rec})
                if it == steps:
                    break
                self.params, opt, _ = self.adam_step(self.params, g, ("adapters",), opt)
            w_v, _, _ = self.eval_plans(self.params, vr, policy="open")
            dv, _ = self.dist(w_v, vr, self.D["world_exec"][vr])
            ups = [float(np.linalg.norm(np.asarray(ad.up.kernel[...])))
                   for ad in nnx_merge(self.gdef, self.params, self.rest).adapters]
            out["configs"][spec] = {"curve": curve, "val_closed": float(1.0 - dv.mean() / dvb.mean()),
                                    "up_kernel_norms": ups, "minutes": (time.time() - t0) / 60.0}
            self.log({"stage": "overfit", "config": spec, "event": "done",
                      "val_closed": out["configs"][spec]["val_closed"], "up_norms": str(ups),
                      "minutes": out["configs"][spec]["minutes"]})
            (self.out / "overfit.json").write_text(json.dumps(out, indent=1))
        return out

    def scaling(self):
        """Does the transferable part grow with data? Train on NESTED subsets of the
        training recovery rows (``--scaling_rows``, e.g. 64,256,512,all), each from a
        fresh init, minibatch ``--scaling_batch`` at ``--scaling_lr``, gate open,
        L_match only (the same objective as :meth:`overfit`), and track the HELD-OUT
        recovery rows every ``--scaling_eval_every`` steps: distance closed (with a
        scene-bootstrap CI), teacher-scorer `useful`, and the drift the adapter causes
        on held-out keep rows. Validation scenes are disjoint from training scenes, so
        this is cross-scene generalization. Writes ``scaling.json``."""
        a = self.args
        from rl.diffusion_ft.adapter import GatedTailAdapter, split_params
        rng = np.random.default_rng(int(a.seed) + 31)
        pool = rng.permutation(np.nonzero(self.pool["rec"] & ~self.is_val)[0])
        sizes = [len(pool) if x.strip() == "all" else min(int(x), len(pool))
                 for x in a.scaling_rows.split(",") if x.strip()]
        vr, vk = self.val_rows["rec"], self.val_rows["keep"]
        tgt_v = self.D["world_exec"][vr]
        dvb, _ = self.dist(self.D["world_base"][vr], vr, tgt_v)
        sc_v = self.D["scene_id"][vr]
        scenes = np.unique(sc_v)
        boot = [np.concatenate([np.nonzero(sc_v == s)[0] for s in rng.choice(scenes, len(scenes))])
                for _ in range(1000)]

        def closed_ci(dv):
            vals = [1.0 - dv[i].mean() / dvb[i].mean() for i in boot]
            return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]

        out = {"val_rec_rows": int(len(vr)), "val_scenes": int(len(scenes)),
               "val_keep_rows": int(len(vk)), "pool": int(len(pool)), "lr": float(a.scaling_lr),
               "width": int(a.scaling_width), "batch": int(a.scaling_batch),
               "steps": int(a.scaling_steps), "runs": {}}
        steps, ev, B = int(a.scaling_steps), int(a.scaling_eval_every), int(a.scaling_batch)
        for N in sizes:
            train_rows = np.sort(pool[:N])
            n_tr = min(N, 256)
            tr_eval = train_rows[:n_tr]
            dtb, _ = self.dist(self.D["world_base"][tr_eval], tr_eval, self.D["world_exec"][tr_eval])
            self.acfg = dataclasses.replace(self.acfg, width=int(a.scaling_width))
            self.module = GatedTailAdapter(self.acfg)
            self.gdef, self.params, self.rest = split_params(self.module)
            self.build_fns()
            a.lr = float(a.scaling_lr)
            opt = self._fresh_opt()
            bsz = min(B, N)
            fn = self.loss_fn("open", False, False, bsz)
            curve, t0 = [], time.time()
            n_sc = int(len(np.unique(self.D["scene_id"][train_rows])))
            for it in range(steps + 1):
                if it % ev == 0 or it == steps:
                    w_v, _, _ = self.eval_plans(self.params, vr, policy="open")
                    dv, _ = self.dist(w_v, vr, tgt_v)
                    u = self.useful(vr, self.score_worlds(w_v, vr))
                    w_t, _, _ = self.eval_plans(self.params, tr_eval, policy="open")
                    dt, _ = self.dist(w_t, tr_eval, self.D["world_exec"][tr_eval])
                    w_k, _, _ = self.eval_plans(self.params, vk, policy="open")
                    _, rk = self.dist(w_k, vk, self.D["world_base"][vk])
                    rec = {"step": it, "val_closed": float(1.0 - dv.mean() / dvb.mean()),
                           "val_useful": float(u.mean()),
                           "train_closed": float(1.0 - dt.mean() / dtb.mean()),
                           "val_keep_rms_m": float(rk.mean())}
                    rec["_dv"] = dv
                    curve.append(rec)
                    self.log({"stage": "scaling", "N": N, **{k: v for k, v in rec.items() if k != "_dv"}})
                if it == steps:
                    break
                r = rng.choice(train_rows, size=bsz, replace=False)
                b = self.batch(r, target=self.D["world_exec"][r], w_match=np.ones(bsz),
                               w_keep=np.zeros(bsz), w_rep=np.zeros(bsz), force_open=np.ones(bsz))
                (_, _), g = fn(self.params, b, None)
                self.params, opt, _ = self.adam_step(self.params, g, ("adapters",), opt)
            best = max(curve, key=lambda c: c["val_closed"])
            res = {"rows": int(N), "scenes": n_sc, "minutes": (time.time() - t0) / 60.0,
                   "best": {k: v for k, v in best.items() if k != "_dv"},
                   "best_ci": closed_ci(best["_dv"]),
                   "final": {k: v for k, v in curve[-1].items() if k != "_dv"},
                   "final_ci": closed_ci(curve[-1]["_dv"]),
                   "curve": [{k: v for k, v in c.items() if k != "_dv"} for c in curve]}
            out["runs"][str(N)] = res
            self.log({"stage": "scaling", "N": N, "event": "done", "scenes": n_sc,
                      "best_step": res["best"]["step"], "best_val_closed": res["best"]["val_closed"],
                      "best_ci": str([round(x, 3) for x in res["best_ci"]]),
                      "final_val_closed": res["final"]["val_closed"],
                      "final_ci": str([round(x, 3) for x in res["final_ci"]]),
                      "minutes": res["minutes"]})
            (self.out / "scaling.json").write_text(json.dumps(out, indent=1))
        return out

    # ------------------------------------------------------------------ #
    # After training
    # ------------------------------------------------------------------ #
    def verify(self):
        """Strength 0 must be the base again; a few strengths with real sampler runs."""
        rows = np.arange(self.N)
        if len(rows) > int(self.args.identity_rows):
            rows = np.sort(self.rng.choice(rows, int(self.args.identity_rows), replace=False))
        res, ok = self.identity(rows, (("trained_s0", "learned", 0.0), ("trained_open_s0", "open", 0.0)))
        e0 = res["trained_s0_max"]
        out = {"strength0_max_err": e0, "strength0_vs_stored_max": res["trained_s0_vs_stored_max"],
               "strength0_pass": ok, "identity": res, "rows": int(len(rows)), "strengths": {}}
        rec, keep = self.val_rows["rec"], self.val_rows["keep"]
        for s in [float(x) for x in self.args.verify_strengths.split(",")]:
            r = {}
            if len(rec):
                w, g, _ = self.eval_plans(self.params, rec, policy="learned", strength=s)
                sh = np.max(np.linalg.norm(w[:, 1:, :2] - self.D["world_base"][rec][:, 1:, :2], axis=-1)
                            * self.step_w[rec][:, 1:], axis=1)
                r.update({"rec_useful": float(self.useful(rec, self.score_worlds(w, rec)).mean()),
                          "rec_shift_m": float(sh.mean()), "rec_gate": float(g.mean())})
            if len(keep):
                w, g, _ = self.eval_plans(self.params, keep, policy="learned", strength=s)
                _, rms = self.dist(w, keep, self.D["world_base"][keep])
                r.update({"keep_rms_m": float(rms.mean()), "keep_p95_m": float(np.quantile(rms, 0.95)),
                          "keep_gate": float(g.mean())})
            out["strengths"][f"{s:g}"] = r
        lab_rows = self.val_rows["labeled"]
        lg = np.asarray(self.gate_logit(self.params, self.jax.numpy.asarray(self.feat[lab_rows])))
        gate = 1.0 / (1.0 + np.exp(-lg))
        for nm, m in (("rec", self.pool["rec"][lab_rows]), ("keep", self.pool["keep"][lab_rows])):
            if m.any():
                out[f"gate_{nm}_quantiles"] = [float(q) for q in np.quantile(gate[m], [0.1, 0.25, 0.5, 0.75, 0.9])]
                out[f"gate_{nm}_open_frac"] = float((gate[m] > 0.5).mean())
        (self.out / "verify.json").write_text(json.dumps(out, indent=1))
        self.log({"stage": "verify", "strength0_max_err": e0,
                  **{f"s{k}/{kk}": v for k, d in out["strengths"].items() for kk, v in d.items()}})
        return out

    def bench(self):
        """Wall-clock per replan call, jitted and warm, batch ``--bench_batch``:
        the deployed base sampler vs the adapter policy (frozen prefix + tail), and
        the parameter counts."""
        jax, jnp = self.jax, self.jax.numpy
        from flax import nnx

        from rl.diffusion_ft.adapter import late_noise, param_count, tail_from_prefix
        B = int(self.args.bench_batch)
        rows = self.rng.choice(self.N, B, replace=self.N < B)
        b = self.rows(rows)
        c1, c2, m = b["cond1"], b["cond2"], b["mask"]
        actor, diff, acfg, floor = self.tuner.actor, self.diff, self.acfg, self.floor
        base_p = actor._tparams
        shape = (B, int(diff.denoise_fn.in_ch), int(diff.denoise_fn.horizon))
        module = nnx.merge(self.gdef, self.params, self.rest)

        def adapter_policy(c1, c2, m, rng):
            # The deployed adapter policy end to end: the same frozen prefix, then the
            # late chain with the tail adapter. sample_with_trace's own late chain is
            # unused here and dead-code-eliminated.
            _, _, (x_prefix, key_late) = diff.sample_with_trace(
                shape, c1, c2, m, rng=rng, k_trainable=self.k, rl_mode=True,
                sigma_sample_floor=floor, sigma_logprob_floor=floor,
                prefix_len=int(self.args.prefix_len), return_prefix=True)
            noise = late_noise(key_late, shape, self.k)
            return tail_from_prefix(diff, module, acfg, c1, c2, m, x_prefix, noise, strength=1.0,
                                    gate_mode="learned", sigma_floor=floor)[0]

        fns = {"base": lambda r: actor.sample_world_with_prefix(base_p, c1, c2, m, r)[0],
               "adapter": (lambda f: (lambda r: f(c1, c2, m, r)))(jax.jit(adapter_policy))}
        dev = jax.devices()[0]
        out = {"batch": B, "device": f"{dev} ({getattr(dev, 'device_kind', '?')})", "ms": {}}
        for nm, f in fns.items():
            for i in range(3):
                jax.block_until_ready(f(jax.random.PRNGKey(i)))
            ts = []
            for i in range(int(self.args.bench_reps)):
                t0 = time.perf_counter()
                jax.block_until_ready(f(jax.random.PRNGKey(100 + i)))
                ts.append(1e3 * (time.perf_counter() - t0))
            out["ms"][nm] = {"median": float(np.median(ts)), "p90": float(np.quantile(ts, 0.9))}
        base_n = param_count(nnx.split(diff.denoise_fn, nnx.Param, ...)[1])
        ad = param_count(self.params)
        gate_n = param_count(self.params["gate"])
        out["params"] = {"adapter_total": ad, "gate": gate_n, "adapters": ad - gate_n,
                         "base_denoiser": base_n, "added_fraction": ad / base_n}
        out["ms_overhead_frac"] = out["ms"]["adapter"]["median"] / out["ms"]["base"]["median"] - 1.0
        (self.out / "bench.json").write_text(json.dumps(out, indent=1))
        self.log({"stage": "bench", **{f"{k}_ms": v["median"] for k, v in out["ms"].items()},
                  "added_params": ad, "added_fraction": ad / base_n})
        return out


def nnx_merge(gdef, params, rest):
    from flax import nnx
    return nnx.merge(gdef, params, rest)


def _auc(score, label):
    """Mann-Whitney AUC of ``score`` for ``label`` (1 positive, 0 negative)."""
    from scipy.stats import rankdata
    r = rankdata(score)
    pos = label == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if not n1 or not n0:
        return float("nan")
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def build_parser():
    from rl.diffusion_ft.train_safety import build_argparser
    p = build_argparser()
    p.description = __doc__
    p.add_argument("--pairs", required=True, help="glob of proposal_select --dump_pairs npz files")
    p.add_argument("--teacher_arm", default="o_lpA3")
    p.add_argument("--base_arm", default="base")
    p.add_argument("--adapter_K", type=int, default=5,
                   help="adapted sampling steps (the last K of the k_trainable late chain)")
    p.add_argument("--adapter_width", type=int, default=32)
    p.add_argument("--gate_hidden", type=int, default=64)
    p.add_argument("--stage_a_steps", type=int, default=500)
    p.add_argument("--stage_b_steps", type=int, default=200)
    p.add_argument("--stage_c_steps", type=int, default=500)
    p.add_argument("--adapter_lr", type=float, default=1e-4,
                   help="Adam lr for the adapter/gate (the base stays frozen); the plan's pilot default")
    p.add_argument("--clip", type=float, default=1.0, help="global-norm gradient clip (rl_actor's)")
    p.add_argument("--batch_rec", type=int, default=16)
    p.add_argument("--batch_keep", type=int, default=16)
    p.add_argument("--batch_fail", type=int, default=8)
    p.add_argument("--gate_batch", type=int, default=64)
    p.add_argument("--keep_weight", type=float, default=1.0)
    p.add_argument("--adapter_repair_weight", type=float, default=1.0,
                   help="outer weight on L_repair (named apart from train_safety's BC "
                        "--repair_weight); 1 = the optimizer's own scale (barrier/n + comfort, "
                        "as lateral_opt and train_safety's safety_weight)")
    p.add_argument("--gate_weight", type=float, default=0.1)
    p.add_argument("--repair_opt_cfg", default="lon=1,w_lon_prox=0.05,lon_back_m=1000",
                   help="lateral_opt config whose objective L_repair evaluates: the teacher's "
                        "barrier + lateral/along-path comfort, plus the closeness term")
    p.add_argument("--repair_window", type=int, default=20,
                   help="world steps L_repair scores (the teacher's window); shorter rows are "
                        "left out of it")
    p.add_argument("--repair_start_frac", type=float, default=0.5,
                   help="Stage A switches L_repair on at this fraction of its steps...")
    p.add_argument("--repair_min_progress", type=float, default=0.1,
                   help="...if validation L_match has fallen at least this fraction by then")
    p.add_argument("--force_open_every", type=int, default=4,
                   help="Stage C: every Nth minibatch runs its recovery rows with the gate "
                        "forced open (0 = never)")
    p.add_argument("--tol_m", type=float, default=1.0,
                   help="progress the student's plan may give up and still count as useful "
                        "(the teacher's cons tolerance)")
    p.add_argument("--val_frac", type=float, default=0.15, help="share of training SCENES held out")
    p.add_argument("--val_rows", type=int, default=256, help="validation rows per pool (cap)")
    p.add_argument("--val_every", type=int, default=100)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--chunk", type=int, default=64, help="rows per jitted eval/precompute call")
    p.add_argument("--identity_tol", type=float, default=1e-5,
                   help="adapter at init / at strength 0 vs the base late chain on the same "
                        "rows in the same program: max |dx| (normalized; 1e-5 = 0.1 mm)")
    p.add_argument("--replay_tol", type=float, default=5e-4,
                   help="stored randomness vs the deployed draw it was collected with, max |dx| "
                        "(normalized): the GPU float32 batch-shape floor, not an adapter effect")
    p.add_argument("--replay_tol_p99", type=float, default=1e-4)
    p.add_argument("--identity_rows", type=int, default=2048)
    p.add_argument("--verify_strengths", default="0,0.25,0.5,0.75,1")
    p.add_argument("--bench_batch", type=int, default=32)
    p.add_argument("--bench_reps", type=int, default=20)
    p.add_argument("--preflight_only", action="store_true")
    p.add_argument("--overfit_rows", type=int, default=0,
                   help="> 0: after preflight, run the fit diagnostic (AdapterTrainer.overfit) "
                        "on this many training recovery rows instead of the staged pilot")
    p.add_argument("--overfit_configs", default="32:1e-4,32:1e-3,32:3e-3,128:1e-3",
                   help="comma list of adapter_width:lr, each from a fresh init")
    p.add_argument("--overfit_steps", type=int, default=3000)
    p.add_argument("--overfit_log_every", type=int, default=50)
    p.add_argument("--overfit_eval_every", type=int, default=500)
    p.add_argument("--scaling_rows", default="",
                   help="non-empty: after preflight, run the data-scaling probe "
                        "(AdapterTrainer.scaling) on these nested subset sizes, e.g. "
                        "'64,256,512,all', instead of the staged pilot")
    p.add_argument("--scaling_lr", type=float, default=1e-3)
    p.add_argument("--scaling_width", type=int, default=32)
    p.add_argument("--scaling_batch", type=int, default=128)
    p.add_argument("--scaling_steps", type=int, default=3000)
    p.add_argument("--scaling_eval_every", type=int, default=100)
    return p


def main():
    args = build_parser().parse_args()
    if not args.shard_glob:
        raise SystemExit("--shard_glob is required (scenes are reloaded by id)")
    args.lr = args.adapter_lr
    tr = AdapterTrainer(args)
    tr.load_pairs()
    tr.precompute()
    tr.build_fns()
    tr.load_scene_cache()
    tr.pick_val()
    tr.preflight()
    if args.preflight_only:
        return
    if args.overfit_rows > 0:
        tr.overfit()
        return
    if args.scaling_rows:
        tr.scaling()
        return
    t0 = time.time()
    tr.stage_a()
    tr.stage_b()
    tr.relabel()
    tr.stage_c()
    tr.verify()
    tr.bench()
    summary = {"counts": tr.counts, "relabel": getattr(tr, "relabel_stats", None),
               "repair_in_stage_a": getattr(tr, "repair_used_in_a", None),
               "minutes": (time.time() - t0) / 60.0, "config": dataclasses.asdict(tr.acfg),
               "args": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))}}
    (tr.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    import resource
    print(f"[adapter] done in {summary['minutes']:.1f} min; peak host RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
