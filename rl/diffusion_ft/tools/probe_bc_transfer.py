"""Why did distilling the bank into the weights move so little? (submit_bank_bc.sh)

The BC fine-tune (bank_bc/20260928_121223_39589) recovered ~6% of the bank's
corpus gain, and its loss was flat. Two readings predict different fixes:

  H1  NOT LEARNED. The edits are ~7% of a batch whose other rows are the base's
      own samples; the late copy never fit them. -> reweight / more steps.
  H2  LEARNED, NOT EXPRESSIBLE. The loss teaches the late copy to denoise a
      NOISED EDIT, but at deploy time the frozen prefix hands it a noised BASE
      plan. -> train the whole chain (k_trainable = timesteps), or train from the
      sampler's own prefix.

Measured on BC rows (proposal_select --dump_bc), paired across checkpoints
(same rows, same (t, eps) draws, same sampler keys):

  loss_sw / loss_keep   late-step denoising loss on switched (edit) rows vs kept
                        rows -- the training objective itself. H1: the fine-tune's
                        loss_sw is ~the base's.
  d_free_m              deploy-time sample (frozen prefix + late copy) to the edit
                        target, mean over covered steps, metres, switched rows.
  d_tf_m                the same late chain started from a TEACHER-FORCED prefix
                        q_sample(target, k-1): the input the loss trained on.
                        H2: d_tf falls with training while d_free does not.
  closed_free           1 - d_free(ckpt) / d_free(base): the fraction of the
                        base-to-edit distance the fine-tune closes at deploy time.

  d_rej_m               (rows carrying `rejected`, 2026-10-01+) the same deploy
                        sample to the base draw the bank REJECTED. A preference
                        fine-tune should raise it on switched rows while
                        closing d_free; BC has no reason to.

CPU-light, GPU for the sampler; reads only saved rows and checkpoints.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np


def _load_rows(pattern, n, seed):
    """``n`` switched rows (solved failure episodes -- the edits BC was taught) and
    ``n`` kept rows (solved episodes, any split), sampled once with ``seed``."""
    keys = ("cond1", "cond2", "mask", "target", "valid", "switched", "ep_solved", "split")
    parts = {k: [] for k in keys}
    files = sorted(glob.glob(pattern))
    if files and "rejected" in np.load(files[0]).files:
        keys = keys + ("rejected",)
        parts["rejected"] = []
    for f in files:
        z = np.load(f)
        for k in keys:
            parts[k].append(z[k])
    if not parts["target"]:
        raise SystemExit(f"no rows match {pattern}")
    R = {k: np.concatenate(v, 0) for k, v in parts.items()}
    rng = np.random.default_rng(seed)
    sw = np.nonzero(R["switched"] & R["ep_solved"] & (R["split"] == "failure_train"))[0]
    kp = np.nonzero(~R["switched"] & R["ep_solved"])[0]
    pick = {"sw": rng.choice(sw, min(n, len(sw)), replace=False),
            "keep": rng.choice(kp, min(n, len(kp)), replace=False)}
    print(f"[bcprobe] rows: {len(R['target'])} total, {len(sw)} switched-solved-failure, "
          f"{len(kp)} kept-solved; probing {len(pick['sw'])} + {len(pick['keep'])}", flush=True)
    out_k = ("cond1", "cond2", "mask", "target", "valid") + (
        ("rejected",) if "rejected" in R else ())
    return {s: {k: R[k][ix] for k in out_k} for s, ix in pick.items()}


def _params_from(path, template):
    z = np.load(Path(path) / "tparams.npz")
    leaves, treedef = jax.tree_util.tree_flatten(template)
    vals = [jnp.asarray(z[k]) for k in sorted(z.files)]
    if len(vals) != len(leaves):
        raise SystemExit(f"{path}: {len(vals)} leaves, model wants {len(leaves)}")
    return jax.tree_util.tree_unflatten(treedef, vals)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="base diffusion checkpoint")
    p.add_argument("--bc_data", required=True, help="glob of bc_data/<arm>/*.npz")
    p.add_argument("--ft", nargs="+", required=True, help="fine-tuned iter_* dirs")
    p.add_argument("--n_rows", type=int, default=512)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--loss_draws", type=int, default=8)
    p.add_argument("--k_trainable", type=int, default=10)
    p.add_argument("--prefix_len", type=int, default=5)
    p.add_argument("--sigma_floor", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    from rl.diffusion_ft.checkpoint import load_diffusion_checkpoint
    from rl.diffusion_ft.env import DiffusionWaymaxEnv
    from rl.diffusion_ft.rl_actor import DiffusionRLActor

    ck = load_diffusion_checkpoint(a.ckpt)
    env = DiffusionWaymaxEnv(ck, seed=a.seed, max_replans=1)      # train_safety's defaults
    diff = env.policy.diffusion
    actor = DiffusionRLActor(diff, k_trainable=a.k_trainable, prefix_len=a.prefix_len,
                             sigma_sample_floor=a.sigma_floor,
                             sigma_logprob_floor=a.sigma_floor, seed=a.seed)
    actor.reset_trainable_to_frozen()
    base = actor._tparams
    ego = float(ck.preprocess_cfg.ego_range)
    k = int(a.k_trainable)
    print(f"[bcprobe] self_cond={diff.self_cond} timesteps={diff.timesteps} k={k} "
          f"ego_range={ego}", flush=True)

    rows = _load_rows(a.bc_data, a.n_rows, a.seed)
    ckpts = {"base": base, **{f"{Path(f).parent.name}/{Path(f).name}": _params_from(f, base)
                              for f in a.ft}}
    loss_fn = jax.jit(actor.late_denoise_loss)

    def dist_m(x_bct, tgt_btd, valid_bt):
        d = jnp.linalg.norm(jnp.transpose(x_bct, (0, 2, 1))[..., :2] - tgt_btd[..., :2], axis=-1)
        v = valid_bt.astype(d.dtype)
        return np.asarray(jnp.sum(d * v, 1) / jnp.maximum(jnp.sum(v, 1), 1.0)) * ego

    res = {"n_rows": {s: int(len(r["target"])) for s, r in rows.items()}, "ckpts": {}}
    for name, params in ckpts.items():
        out = {}
        for s, R in rows.items():
            n = len(R["target"])
            losses, d_free, d_tf, d_rej = [], [], [], []
            for c0 in range(0, n, a.chunk):
                sl = slice(c0, c0 + a.chunk)
                c1, c2 = jnp.asarray(R["cond1"][sl]), jnp.asarray(R["cond2"][sl])
                m = jnp.asarray(R["mask"][sl])
                tgt = jnp.asarray(R["target"][sl], jnp.float32)
                val = jnp.asarray(R["valid"][sl])
                x0 = jnp.transpose(tgt, (0, 2, 1))
                B = x0.shape[0]
                for j in range(a.loss_draws):
                    rk = jax.random.PRNGKey(10_000 * a.seed + 100 * c0 + j)
                    losses.append(float(np.asarray(loss_fn(
                        params, x0, c1, c2, m, jnp.ones(B, jnp.float32), rk,
                        val.astype(jnp.float32)))))
                key = jax.random.PRNGKey(7919 + c0)
                x_free, (_, key_late) = actor.sample_world_with_prefix(params, c1, c2, m, key)
                d_free.append(dist_m(x_free, tgt, val))
                if "rejected" in R:
                    d_rej.append(dist_m(x_free, jnp.asarray(R["rejected"][sl], jnp.float32), val))
                # The state the loss trained on: the edit itself, noised to the
                # level the late chain starts from.
                eps = jax.random.normal(jax.random.PRNGKey(31 + c0), x0.shape)
                x_pre = diff.q_sample(x0, jnp.full((B,), k - 1, jnp.int32), eps)
                if diff.self_cond:
                    x_pre = (x_pre, jnp.zeros_like(x_pre))
                x_tf = actor.sample_late_for(params, c1, c2, m, x_pre, key_late)
                d_tf.append(dist_m(x_tf, tgt, val))
            out[s] = {"loss": float(np.mean(losses)),
                      "d_free_m": float(np.mean(np.concatenate(d_free))),
                      "d_free_p50_m": float(np.median(np.concatenate(d_free))),
                      "d_tf_m": float(np.mean(np.concatenate(d_tf))),
                      **({"d_rej_m": float(np.mean(np.concatenate(d_rej)))} if d_rej else {})}
        res["ckpts"][name] = out
        print(f"[bcprobe] {name:<28} " + "  ".join(
            f"{s}: loss {v['loss']:.3e} d_free {v['d_free_m']:.3f} m (p50 {v['d_free_p50_m']:.3f}) "
            f"d_tf {v['d_tf_m']:.3f} m" for s, v in out.items()), flush=True)

    b = res["ckpts"]["base"]
    for name, v in res["ckpts"].items():
        for s in v:
            v[s]["loss_vs_base"] = v[s]["loss"] / max(b[s]["loss"], 1e-12)
            v[s]["closed_free"] = 1.0 - v[s]["d_free_m"] / max(b[s]["d_free_m"], 1e-9)
            v[s]["closed_tf"] = 1.0 - v[s]["d_tf_m"] / max(b[s]["d_tf_m"], 1e-9)
            if "d_rej_m" in v[s]:
                # > 0: the deploy sample moved AWAY from the rejected draw.
                v[s]["left_rej"] = v[s]["d_rej_m"] / max(b[s]["d_rej_m"], 1e-9) - 1.0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("\n  ckpt                          sw loss/base  closed_free  closed_tf | "
          "keep loss/base  closed_free")
    for name, v in res["ckpts"].items():
        print(f"  {name:<28}  {v['sw']['loss_vs_base']:11.3f}  {v['sw']['closed_free']:+10.3f}  "
              f"{v['sw']['closed_tf']:+9.3f} | {v['keep']['loss_vs_base']:13.3f}  "
              f"{v['keep']['closed_free']:+10.3f}")
    if "left_rej" in res["ckpts"]["base"]["sw"]:
        print("\n  ckpt                          sw d_free m  d_rej m  left_rej (switched rows)")
        for name, v in res["ckpts"].items():
            print(f"  {name:<28}  {v['sw']['d_free_m']:9.3f}  {v['sw']['d_rej_m']:7.3f}  "
                  f"{v['sw']['left_rej']:+8.3f}")
    print(f"[bcprobe] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
