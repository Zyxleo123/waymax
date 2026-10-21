"""Diffusion Planner (Zheng et al., ICLR 2025) in the waymax_rs closed loop.

Runs their unmodified model (github.com/ZhengYinan-AIR/Diffusion-Planner @ a3a621f, put on PYTHONPATH or passed
as ``repo_path``) on inputs built by ``diffusion_planner_womd.features``, the same code that made its WOMD
training data. As in their nuPlan planner, the ego current state is given as pose only (velocity etc. zero),
the model samples without guidance, and the ego plan is the first predicted track. The plan (5 s at 10 Hz) is
replayed like the other waymax_rs planners'.

A checkpoint trained in goal mode (``goal_conditioned`` in args.json, diffusion_planner_womd/goal.py) is also
given the episode goal and the time left to it, clipped to at least one step once the goal step has passed.
"""

from __future__ import annotations

import sys
from typing import Any

import jax.numpy as jnp
import numpy as np

from diffusion_planner_womd import goal as dp_goal
from diffusion_planner_womd.features import FUTURE_LEN, ScenarioMap, WorldLog, build_inputs, ego_pose
from planner.abstract_planner import AbstractPlanner, PlannerResult
from scores.scorer_lane_graph import Scorer

DT = 0.1


class DiffusionPlannerBaseline(AbstractPlanner):
    def __init__(self, args_file: str, ckpt_path: str, num_worlds: int = 1, repo_path: str | None = None,
                 device: str | None = None, use_ema: bool = True, seed: int = 0, **kwargs: Any) -> None:
        if repo_path and repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        import torch
        from diffusion_planner.model.diffusion_planner import Diffusion_Planner
        from diffusion_planner.utils.config import Config

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        dp_goal.install(augmentation=False)  # their augmentation module needs nuplan
        config = Config(args_file, guidance_fn=None)
        self.goal_conditioned = bool(getattr(config, "goal_conditioned", False))
        self.normalizer = config.observation_normalizer
        self.model = Diffusion_Planner(config)
        # their checkpoints hold a numpy scalar (the loss), which torch >= 2.6 only loads with weights_only=False
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state = state["ema_state_dict"] if use_ema else state.get("model", state)
        # their checkpoints are saved from the DDP-wrapped model
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)

        self.num_worlds = int(num_worlds)
        self.scorers = [Scorer(metrics=[]) for _ in range(self.num_worlds)]  # lane-graph holders for the runner
        self._maps: list[tuple | None] = [None] * self.num_worlds

    def _scenario_map(self, sim_state, w, t, goal_xy):
        """Built once per scenario at its first replan, while ego's logged future is still in the state."""
        lane_graph = self.scorers[w]._lane_graph
        key = (id(lane_graph), float(goal_xy[0]), float(goal_xy[1]))
        cached = self._maps[w]
        if cached is None or cached[0] != key or t <= cached[1]:
            cached = (key, t, ScenarioMap(lane_graph, WorldLog.from_state(sim_state, w)))
        self._maps[w] = (key, t, cached[2])
        return cached[2]

    def plan_trajectory(self, sim_state, goal, *, rng=None, timestep: int = 0, **kwargs: Any) -> PlannerResult:
        torch = self.torch
        t = int(timestep)
        goal = np.asarray(goal, dtype=np.float64).reshape(-1, 2)
        logs, batch = [], []
        for w in range(self.num_worlds):
            smap = self._scenario_map(sim_state, w, t, goal[w])
            log = WorldLog.from_state(sim_state, w)
            ego_state = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)  # pose only
            if self.goal_conditioned:
                g = dp_goal.episode_goal(log, t)  # the runner's goal (ego's last valid log position)
                g[2] = max(g[2], DT)
                ego_state = np.concatenate([ego_state, g])
            inputs, _ = build_inputs(log, smap, t, ego_state=ego_state)
            logs.append(log)
            batch.append(inputs)
        inputs = {k: torch.as_tensor(np.stack([b[k] for b in batch])).to(self.device) for k in batch[0]}
        with torch.no_grad():
            _, outputs = self.model(self.normalizer(inputs))
        plan = outputs["prediction"][:, 0].float().cpu().numpy()  # [B, FUTURE_LEN, 4] x, y, cos, sin (ego frame)

        trajectories = []
        for w, log in enumerate(logs):
            origin = ego_pose(log, t)
            c, s = np.cos(origin[2]), np.sin(origin[2])
            xy = np.stack([plan[w, :, 0] * c - plan[w, :, 1] * s, plan[w, :, 0] * s + plan[w, :, 1] * c], -1) + origin[:2]
            yaw = np.arctan2(plan[w, :, 3], plan[w, :, 2]) + origin[2]
            xy = np.vstack([origin[:2], xy])
            yaw = np.r_[origin[2], yaw]
            vel = np.gradient(xy, DT, axis=0)
            vel[0] = [log.vx[log.ego, t], log.vy[log.ego, t]]
            trajectories.append(np.concatenate([xy, np.arctan2(np.sin(yaw), np.cos(yaw))[:, None], vel], -1))
        traj = jnp.asarray(np.stack(trajectories), dtype=jnp.float32)
        t_seconds = jnp.broadcast_to(jnp.arange(FUTURE_LEN + 1, dtype=jnp.float32) * DT, traj.shape[:2])
        return PlannerResult(
            start_t_b=jnp.full((self.num_worlds,), t, dtype=jnp.int32),
            trajectory_norm_btd=traj,
            trajectory_world_bt5=traj,
            world_t_seconds_bt=t_seconds,
            world_t_valid_bt=jnp.ones(traj.shape[:2], dtype=bool),
            aux={"world_dt_seconds": jnp.full((self.num_worlds,), DT, dtype=jnp.float32)},
        )
