"""Closed-loop check of diffusion-ES with different initial pools / rewards on a few scenarios.

Same setup as ``es_diversity.ipynb`` (e450 WM, log agents, replan every 10 steps). For each
scenario x variant writes, under ``--out``:
  summary.json             success / overlap / offroad / goal per run
  <scen>_<variant>.csv     one row per replan (plan source, chosen plan terms, pool safety, MCMC stats)
  <scen>_<variant>.png     replan overview (plot_closed_loop_replans)
  <scen>_speed.png         speed / accel / jerk of all variants

Usage:
  python notebooks/es_mcmc_eval.py --scenarios 00017:173 --variants baseline mcmc_lex --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import tensorflow as tf  # noqa: E402

tf.config.set_visible_devices([], "GPU")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import es_debug as esd  # noqa: E402

SUBSET_DIR = "/zfsauton/scratch/sbellad/Diffusion_ES_sim/es_failure_subset_n1500_seed0"
TFRECORD_DIR = f"{SUBSET_DIR}/tfrecords/training"
LANE_GRAPH_DIR = "/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
DIFFUSION_CKPT = "/zfsauton/scratch/sbellad/train_diffusion/diffusion_baseline_baseline_20260810_144559/epoch_0500"
WM_CKPT = ("/zfsauton/scratch/ishanj/world_model/ckpts_multifuture_gaussian/"
           "multiFuture_GaussianWorldModel_20261005_092950/epoch_0450.pt")
ES = dict(population_size=64, elite_size=8, num_iterations=4, resample_timesteps=5)


def build_variants(base: esd.DebugESPlanner) -> dict:
    mcmc = esd.mcmc_pool()
    lex = esd.lexicographic_reward()
    return {
        "baseline": lambda: base,
        # diffusion pool, lexicographic reward: what does the reward alone do?
        "lex": lambda: base.variant(label="lex", reward=lex, resample_noise_scale=1.0),
        # MCMC safe-diverse pool + lexicographic reward
        "mcmc_lex": lambda: base.variant(label="mcmc_lex", initial_pool=mcmc, reward=lex, resample_noise_scale=1.0),
        # same, but ES only re-ranks (no diffusion re-noising of the MCMC plans)
        "mcmc_lex_noes": lambda: base.variant(label="mcmc_lex_noes", initial_pool=mcmc, reward=lex, num_iterations=0),
        # ES mutates MCMC-born elites in the kinematic space (smooth, stays near the safe region),
        # diffusion-born elites with the diffusion model
        "mcmc_lex_hybrid": lambda: base.variant(label="mcmc_lex_hybrid", initial_pool=mcmc, reward=lex,
                                                resample_noise_scale=1.0, es_mutation="hybrid"),
    }


def replan_rows(result: esd.ClosedLoopResult) -> list[dict]:
    traj, ego = result.rollout_traj, result.scenario.ego_idx
    v = np.hypot(np.asarray(traj.vel_x[ego]), np.asarray(traj.vel_y[ego]))
    rows = []
    for s in result.replan_steps:
        tr = result.traces[s]
        last = tr.iterations[-1]
        p0, pl = esd.pool_stats(tr, 0), esd.pool_stats(tr, -1)
        row = {
            "t": s, "v": round(float(v[s]), 2), "source": tr.source(), "best_score": round(tr.best_score, 4),
            "n_safe_init": p0["n_safe"], "n_safe_final": pl["n_safe"],
            "plan_v_end": round(float(np.hypot(*result.plans[s][-1, 3:5])), 2),
            "plan_jerk": round(float(esd.plan_jerk(result.plans[s][None])[0]), 2),
            "fail_in_interval": bool((result.overlap_t[s:s + 10] | result.offroad_t[s:s + 10]).any()),
        }
        for k, vals in last.breakdown.items():
            row[f"best_{k}"] = round(float(vals[tr.best_index]), 3)
        if tr.pool_info is not None:
            row.update({f"mcmc_{k}": (round(x, 3) if isinstance(x, float) else x)
                        for k, x in esd.mcmc_summary(tr.pool_info).items()})
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["00017:173"], help="shard:idx")
    ap.add_argument("--variants", nargs="+", default=["baseline", "mcmc_lex"])
    ap.add_argument("--sim_agents", default="log")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    policy, cfg = esd.load_diffusion_policy(DIFFUSION_CKPT, seed=args.seed)
    wm = esd.load_world_model(WM_CKPT, backend="multifuture", planning_mode="expected_independent")
    base = esd.DebugESPlanner(policy, cfg, other_agent_forecast="wm", world_model_forecaster=wm,
                              reward=esd.margin_tiebreak_reward(), label="baseline", **ES)
    factories = build_variants(base)

    summary = []
    for spec in args.scenarios:
        shard, idx = spec.split(":")
        scen = esd.load_scenario(f"{TFRECORD_DIR}/training_tfexample.tfrecord-{shard}-of-01000", int(idx),
                                 lane_graph_dir=LANE_GRAPH_DIR)
        runs = {}
        for name in args.variants:
            t0 = time.time()
            r = esd.run_closed_loop(factories[name](), scen, start_timestep=10, replan_interval_steps=10,
                                    sim_agents=args.sim_agents, seed=args.seed, verbose=False)
            runs[name] = r
            print(r.report(), f"({time.time() - t0:.0f}s)", flush=True)
            tag = f"{shard}_{idx}_{name}"
            df = pd.DataFrame(replan_rows(r))
            df.to_csv(out / f"{tag}.csv", index=False)
            print(df.to_string(index=False), flush=True)
            fig = esd.plot_closed_loop_replans(r)
            fig.savefig(out / f"{tag}.png", dpi=80)
            plt.close(fig)
            summary.append({"scenario": f"{shard}#{idx}", "variant": name, "success": r.success,
                            "goal_reached": r.goal_reached, "overlap": r.overlap, "offroad": r.offroad,
                            "min_goal_dist_m": round(r.min_goal_distance_m, 2),
                            "first_fail_t": esd.first_failure_t(r), "seconds": round(time.time() - t0)})
        fig = esd.plot_speed_profile(list(runs.values()))
        fig.savefig(out / f"{shard}_{idx}_speed.png", dpi=80)
        plt.close(fig)
        json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
