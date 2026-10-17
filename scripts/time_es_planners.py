"""Time Diffusion-ES / FM-ES planning only (no data loading, sim stepping or I/O).

Takes the same flags as simulation/run_simulation.py, plus --warmup_scenarios N.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import numpy as np

import simulation.run_simulation as rs
import planner.diffusion_es_planner as des
import planner.fm_es_planner as fes
from scores.scorer_lane_graph import Scorer

try:
    import torch
except ImportError:
    torch = None


def _pop_flag(name: str, default: int) -> int:
    if name in sys.argv:
        i = sys.argv.index(name)
        value = int(sys.argv[i + 1])
        del sys.argv[i : i + 2]
        return value
    return default


WARMUP = _pop_flag("--warmup_scenarios", 1)
TIMES: dict[str, list[tuple[int, float]]] = defaultdict(list)
STATE = {"scenario": -1, "start_t": 10}


def _sync(tree=None) -> None:
    if tree is not None:
        for leaf in jax.tree_util.tree_leaves(tree):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(key: str, fn, *, new_scenario_on_start: bool = False):
    def wrapper(*args, **kwargs):
        if new_scenario_on_start and int(kwargs.get("timestep", -1)) == STATE["start_t"]:
            STATE["scenario"] += 1
        _sync()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        if key == "plan":
            _sync((out.trajectory_world_bt5, out.aux))
        else:
            _sync(out)
        TIMES[key].append((STATE["scenario"], time.perf_counter() - t0))
        return out
    return wrapper


def _install_timers() -> None:
    des.DiffusionESPlanner.plan_trajectory = _timed(
        "plan", des.DiffusionESPlanner.plan_trajectory, new_scenario_on_start=True
    )
    fes.FMESPlanner.plan_trajectory = _timed(
        "plan", fes.FMESPlanner.plan_trajectory, new_scenario_on_start=True
    )
    des.preprocess_simulator_state = _timed("preprocess", des.preprocess_simulator_state)
    fes.preprocess_simulator_state = _timed("preprocess", fes.preprocess_simulator_state)
    Scorer.compute_score = _timed("score_forecast", Scorer.compute_score)


def _summarize(args) -> dict:
    kept = {k: [(s, t) for s, t in v if s >= WARMUP] for k, v in TIMES.items()}
    scenarios = sorted({s for s, _ in kept.get("plan", [])})
    if not scenarios:
        raise RuntimeError("No timed scenarios; increase --num_scenarios above --warmup_scenarios.")

    def per_scenario(key: str) -> list[float]:
        return [sum(t for s, t in kept.get(key, []) if s == sc) for sc in scenarios]

    plan_sc = per_scenario("plan")
    pre_sc = per_scenario("preprocess")
    score_sc = per_scenario("score_forecast")
    model_sc = [p - a - b for p, a, b in zip(plan_sc, pre_sc, score_sc)]
    return {
        "planner": args.planner,
        "forecast": args.other_agent_forecast,
        "wm_backend": getattr(args, "wm_backend", None),
        "num_worlds": args.num_worlds,
        "num_es_iterations": args.num_es_iterations,
        "resample_num_steps": args.resample_num_steps if args.planner == "es_fm" else None,
        "resample_timesteps": args.resample_timesteps if args.planner == "es" else None,
        "timed_scenarios": len(scenarios),
        "plan_calls": len(kept["plan"]),
        "per_call_ms": float(np.mean([t for _, t in kept["plan"]]) * 1e3),
        "per_scenario_s": {
            "total_planning": float(np.mean(plan_sc)),
            "preprocess": float(np.mean(pre_sc)),
            "score_incl_forecast": float(np.mean(score_sc)),
            "policy_sampling_and_es": float(np.mean(model_sc)),
        },
        "per_scenario_total_s_each": [round(x, 3) for x in plan_sc],
    }


def main() -> None:
    _install_timers()
    args = rs._parse_args()
    STATE["start_t"] = int(args.start_timestep)
    rs.main(args)
    summary = _summarize(args)
    print("TIMING " + json.dumps(summary), flush=True)
    with open(Path(args.output_dir) / "timing.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
