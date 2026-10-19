"""Method B: safety + progress discriminator on recorded Diffusion-ES candidates.

Preference pairs inside each replan group, from Waymax-exact labels (log-replay
agents, label horizon <= 5 s):
  1. a safe candidate beats an unsafe one (unsafe = Waymax overlap or offroad);
  2. between two safe candidates, the one with >= `progress_margin` metres more
     goal-directed progress wins, so safe but needlessly slow plans rank below safe
     plans that make progress.
Unsafe-vs-unsafe pairs are not used. Loss: softplus(-(s_better - s_worse)).
Normally launched by slurm/train/train_wm_discriminator.sh.

  python -m es_discriminator.safety_progress.train --output-dir .../safety_progress/<run> \
      --wandb-name safety_progress_full_gaussian_wm --train-cands 64 --eval-chunk 64
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from es_discriminator.real_es.trainer import Method, common_parser, run


def sample_pairs(unsafe: np.ndarray, progress: np.ndarray, *, margin: float, max_pairs: int,
                 rng: np.random.Generator, safety_frac: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """(better, worse) indices for one group: a `safety_frac` share of safety pairs, the rest progress
    pairs; whichever kind runs out is topped up with the other."""
    safe = np.flatnonzero(~unsafe)
    bad = np.flatnonzero(unsafe)
    n_safety = len(safe) * len(bad)
    prog = np.nan_to_num(progress[safe].astype(np.float64), nan=-np.inf)
    diff = prog[:, None] - prog[None, :]
    pi, pj = np.nonzero(diff >= margin)
    n_prog = len(pi)
    if n_safety == 0 and n_prog == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    want_s = min(n_safety, max(round(safety_frac * max_pairs), max_pairs - n_prog))
    want_p = min(n_prog, max_pairs - want_s)
    better, worse = [], []
    if want_s:
        flat = rng.choice(n_safety, want_s, replace=False)
        better.append(safe[flat // len(bad)])
        worse.append(bad[flat % len(bad)])
    if want_p:
        pick = rng.choice(n_prog, want_p, replace=False)
        better.append(safe[pi[pick]])
        worse.append(safe[pj[pick]])
    return np.concatenate(better).astype(np.int64), np.concatenate(worse).astype(np.int64)


def stratified_candidates(unsafe: np.ndarray, progress: np.ndarray, k: int, rng: np.random.Generator,
                          unsafe_frac: float = 0.5) -> np.ndarray:
    """k candidate indices of one group: up to `unsafe_frac` unsafe (random), the rest safe drawn
    one per equal-count progress bin, so subsets keep both safety pairs and progress pairs."""
    n = len(unsafe)
    if k >= n:
        return np.arange(n)
    bad = np.flatnonzero(unsafe)
    safe = np.flatnonzero(~unsafe)
    n_bad = min(len(bad), int(round(unsafe_frac * k)))
    n_safe = min(len(safe), k - n_bad)
    n_bad = min(len(bad), k - n_safe)
    pick = [rng.choice(bad, n_bad, replace=False)] if n_bad else []
    if n_safe:
        prog = np.nan_to_num(progress[safe].astype(np.float64), nan=-np.inf)
        order = safe[np.argsort(prog, kind="stable")]
        bins = np.array_split(order, n_safe)
        pick.append(np.array([rng.choice(b_) for b_ in bins]))
    return np.sort(np.concatenate(pick)).astype(np.int64)


def make_sampler(progress_key: str):
    def sampler(cands: dict, rng: np.random.Generator, k: int) -> np.ndarray:
        return np.stack([stratified_candidates(cands["unsafe"][b], cands[progress_key][b], k, rng)
                         for b in range(cands["unsafe"].shape[0])])
    return sampler


def make_loss(progress_key: str, margin: float, max_pairs: int, safety_frac: float):
    def loss_fn(scores: torch.Tensor, cands: dict, groups: dict, rng: np.random.Generator):
        bi, ii, jj = [], [], []
        n_safety = 0
        for b in range(scores.shape[0]):
            i, j = sample_pairs(cands["unsafe"][b], cands[progress_key][b], margin=margin,
                                max_pairs=max_pairs, rng=rng, safety_frac=safety_frac)
            n_safety += int((cands["unsafe"][b][j] & ~cands["unsafe"][b][i]).sum())
            bi.append(np.full(len(i), b))
            ii.append(i)
            jj.append(j)
        b_idx = np.concatenate(bi)
        if len(b_idx) == 0:
            return None, {}
        dev = scores.device
        b_t, i_t, j_t = (torch.from_numpy(x).to(dev) for x in (b_idx, np.concatenate(ii), np.concatenate(jj)))
        d = scores[b_t, i_t] - scores[b_t, j_t]
        loss = F.softplus(-d).mean()
        return loss, {"pair_accuracy": float((d > 0).float().mean()), "pairs": float(len(b_idx)),
                      "safety_pair_fraction": n_safety / len(b_idx)}
    return loss_fn


def main() -> None:
    p = common_parser(__doc__, default_out="/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/es_discriminator/safety_progress/manual",
                      default_name="safety_progress_v1")
    p.add_argument("--max-pairs", type=int, default=512, help="preference pairs per replan group per step")
    p.add_argument("--safety-frac", type=float, default=0.8,
                   help="share of each group's pairs that are safe-vs-unsafe (v1 used 0.5 and learned mostly "
                        "progress)")
    args = p.parse_args()
    run(args, Method(
        name="safety_progress",
        loss_fn=make_loss(args.progress_key, args.progress_margin, args.max_pairs, args.safety_frac),
        cand_keys=("unsafe", args.progress_key), group_keys=(),
        select_key="discriminator/unsafe", select_mode="min",
        sampler=make_sampler(args.progress_key),
    ))


if __name__ == "__main__":
    main()
