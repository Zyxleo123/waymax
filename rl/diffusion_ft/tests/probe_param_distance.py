"""How far did each run actually move from the pretrained weights?

The 09-02 safety:anchor sweep degraded in every arm, including the two that were
supposed to be controls (``sw=0`` pure anchor and ``aw=0`` pure safety). That
rules out any objective-specific story and points at the optimizer: an Adam
update is ``lr * m/(sqrt(v)+eps)``, which is O(lr) per parameter *whatever the
gradient norm is*, so the gradient chooses the direction and the STEP COUNT
chooses the distance. Nothing in ``train_safety`` bounded the step count.

This probe measures the distance directly against the base checkpoint and
compares it to the two regimes it could be in, for P parameters and N steps:

* ``aligned``     ``lr * N * sqrt(P)``   -- every step pulls the same way
* ``random walk`` ``lr * sqrt(N * P)``   -- zero-mean gradient, no signal

(the walk figure follows because Adam's EMA is a low-pass filter with unit DC
gain, so the accumulated update tracks the accumulated gradient divided by its
RMS -- i.e. ``sqrt(N)`` per coordinate, not ``N``.)

A run whose displacement lands near the walk figure spent its budget diffusing.
The ``cosine`` block separates that further: displacement vectors of two arms
with a real shared signal are correlated, while two random walks are orthogonal
in 31.7M dimensions (|cos| ~ 1e-4). That is the difference between "the safety
term is steering" and "the safety term is a rumour on top of noise".

CPU only -- it reads saved weights, restores the base, and does linear algebra.

Usage::

    python -m rl.diffusion_ft.tests.probe_param_distance \\
        --arms '/path/to/sweep/*/' --out /path/to/param_distance.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import numpy as np
from flax import nnx

from rl.diffusion_ft.checkpoint import DEFAULT_CHECKPOINT_PATH, load_diffusion_checkpoint


def base_flat(ckpt_path: str) -> np.ndarray:
    """The base denoiser's trainable params, flattened in ``save()``'s order.

    ``DiffusionRLActor`` clones ``diffusion.denoise_fn`` and optimizes the
    ``nnx.Param`` split of the clone; ``SafetyTrainer.save`` then flattens that
    pytree and writes the leaves as ``p00000..``. Splitting the *frozen* module
    the same way reproduces that order exactly, so leaf i here is leaf i there.
    ``use_ema=False`` matches ``DiffusionWaymaxEnv``'s own default -- the EMA
    weights are a different point and would report a spurious displacement.
    """
    policy = load_diffusion_checkpoint(ckpt_path).merge(use_ema=False)
    _, params, _ = nnx.split(policy.diffusion.denoise_fn, nnx.Param, ...)
    leaves = jax.tree_util.tree_flatten(params)[0]
    return np.concatenate([np.asarray(x).ravel().astype(np.float32) for x in leaves])


def arm_flat(npz_path: Path, n_expect: int) -> np.ndarray:
    z = np.load(npz_path)
    keys = sorted(z.files)                      # p00000.. sorts lexically == pytree order
    v = np.concatenate([z[k].ravel().astype(np.float32) for k in keys])
    if v.size != n_expect:
        raise SystemExit(f"{npz_path}: {v.size} params, base has {n_expect}")
    return v


def adam_steps(run_dir: Path) -> tuple[int, float | None]:
    """Optimizer steps and lr actually used, read from the run's own artifacts.

    ``n_steps`` became the count of *applied* steps when gradient accumulation
    landed (2026-09-03); before that it was one per replan context. Older logs
    therefore report the pre-accumulation schedule, which is the point.
    """
    log = run_dir / "train_log.jsonl"
    n = 0
    if log.exists():
        for line in log.read_text().splitlines():
            if line.strip():
                n += int(json.loads(line).get("n_steps", 0))
    lr = None
    meta = run_dir / "latest" / "meta.json"
    if meta.exists():
        lr = json.loads(meta.read_text()).get("lr")
    return n, lr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--arms", required=True,
                   help="glob(s) of run directories, comma-separated. Several "
                        "globs let one probe span sweeps, so a shared arm (same "
                        "objective, different seed) need not be re-run to be "
                        "compared -- run directory basenames are the arm names "
                        "and must therefore stay distinct across the globs.")
    p.add_argument("--lr", type=float, default=1e-5,
                   help="fallback when a run's meta.json predates the lr field")
    p.add_argument("--out", default="")
    a = p.parse_args()

    seen, runs = set(), []
    for pattern in [g.strip() for g in a.arms.split(",") if g.strip()]:
        for d in sorted(glob.glob(pattern)):
            r = Path(d)
            if not (r / "latest" / "tparams.npz").exists():
                continue
            if r.name in seen:
                raise SystemExit(
                    f"duplicate arm name '{r.name}' ({r}); basenames are the "
                    f"arm labels and a collision would silently overwrite one")
            seen.add(r.name)
            runs.append(r)
    if not runs:
        raise SystemExit(f"no runs with latest/tparams.npz under {a.arms}")

    print(f"[dist] restoring base from {a.ckpt}", flush=True)
    theta0 = base_flat(a.ckpt)
    P = int(theta0.size)
    n0 = float(np.linalg.norm(theta0.astype(np.float64)))
    print(f"[dist] base: P={P:,}  ||theta0||={n0:.3f}  "
          f"rms/param={n0 / math.sqrt(P):.5f}", flush=True)

    rows, deltas = {}, {}
    for r in runs:
        theta = arm_flat(r / "latest" / "tparams.npz", P)
        d = theta - theta0
        deltas[r.name] = d
        n_steps, lr_meta = adam_steps(r)
        lr = float(lr_meta) if lr_meta else float(a.lr)
        dist = float(np.linalg.norm(d.astype(np.float64)))
        walk = lr * math.sqrt(max(n_steps, 1) * P)
        aligned = lr * max(n_steps, 1) * math.sqrt(P)
        rows[r.name] = {
            "adam_steps": n_steps, "lr": lr,
            "distance": dist,
            "relative_to_theta0": dist / n0,
            "predicted_random_walk": walk,
            "predicted_aligned": aligned,
            "frac_of_walk": dist / walk if walk else None,
            "frac_of_aligned": dist / aligned if aligned else None,
            "max_abs_delta": float(np.abs(d).max()),
        }
        print(f"[dist] {r.name:<22s} N={n_steps:<6d} ||dtheta||={dist:8.3f}  "
              f"walk={walk:7.3f} ({dist / walk:5.2f}x)  "
              f"aligned={aligned:8.1f} ({100 * dist / aligned:5.2f}%)", flush=True)

    names = list(deltas)
    cos = {}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            dx, dy = deltas[x].astype(np.float64), deltas[y].astype(np.float64)
            denom = np.linalg.norm(dx) * np.linalg.norm(dy)
            cos[f"{x}|{y}"] = float(dx @ dy / denom) if denom else 0.0

    out = {"base_ckpt": a.ckpt, "n_params": P, "base_norm": n0,
           "arms": rows, "displacement_cosine": cos,
           "note": ("cosine ~0 (|cos| ~ 1e-4 is chance in 31.7M dims) means the "
                    "two runs moved in unrelated directions -- no shared signal.")}
    print("\n[dist] displacement cosine between arms:")
    for k, v in sorted(cos.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:<48s} {v:+.4f}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(f"\n[dist] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
