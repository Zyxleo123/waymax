"""Goal-conditioned mode of the Diffusion Planner baseline (ours, not in their paper or code).

Their model sees the route (``route_lanes``) but not where along it the episode's goal is, nor by when, while the
waymax_rs closed loop scores goal reaching (the logged ego's last valid position, within 2 m). Goal mode gives
the model the goal the way the waymax_rs diffusion policy gets it: goal x, y in ego's frame and the time left
until the goal step, embedded by an MLP as one extra token of their encoder (next to the agent, static-object
and lane tokens), which the denoiser cross-attends to.

The goal rides as three extra columns of ``ego_current_state``, [..., 10:13] = (x, y, seconds to goal): their
data loader, training loop and loss pass that tensor through and only read columns :10, so only their
normalizer (three more mean / std entries), their augmentation (the goal is moved into the perturbed ego frame
like everything else) and their encoder (the goal token) change. ``install`` patches the last two; a model
whose args.json has no ``goal_conditioned`` is unchanged.

Training goals: training anchors are at most step 40 (the 5 s targets must be logged), so there the episode
goal is always >= 5 s away, while in closed loop it gets closer than that from step 41 on. Each training sample
therefore takes, with probability FINAL_GOAL_PROB, the episode goal (from goals_womd.py), and otherwise the
logged ego position k ~ U{1..FUTURE_LEN} steps ahead with k / 10 s to go, so goals inside the horizon are seen
too.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

GOAL_DIM = 3  # x, y (m, ego frame), seconds to the goal step
GOAL_MEAN = [0.0, 0.0, 0.0]
GOAL_STD = [40.0, 40.0, 4.0]
FINAL_GOAL_PROB = 0.5
DT = 0.1
STATE_DIM = 10  # their ego_current_state


def episode_goal(log, t: int) -> np.ndarray:
    """Goal input at step t: the logged ego's last valid position (the closed-loop goal, as in
    simulation.evaluation_utils.infer_goals_from_sim_state) in ego's frame at t, and seconds from t to it."""
    from diffusion_planner_womd.features import _to_local_xy, ego_pose

    e = log.ego
    g = int(np.flatnonzero(log.valid[e])[-1])
    xy = _to_local_xy(np.array([log.x[e, g], log.y[e, g]]), ego_pose(log, t))
    return np.array([xy[0], xy[1], (g - t) * DT], dtype=np.float32)


def extend_normalizer(observation_normalizer) -> None:
    """Their ObservationNormalizer, with the goal columns appended to ego_current_state's mean / std."""
    import torch

    stats = observation_normalizer._normalization_dict["ego_current_state"]
    if stats["mean"].shape[-1] == STATE_DIM:
        stats["mean"] = torch.cat([stats["mean"], torch.tensor(GOAL_MEAN, dtype=stats["mean"].dtype)])
        stats["std"] = torch.cat([stats["std"], torch.tensor(GOAL_STD, dtype=stats["std"].dtype)])


def goal_dataset(base, goal_dir: str):
    """Their DiffusionPlannerData with the goal appended to each sample's ego_current_state."""

    class GoalData(base):
        def __init__(self, data_dir, data_list, *args, **kwargs):
            super().__init__(data_dir, data_list, *args, **kwargs)
            goals = {}
            for f in sorted(Path(goal_dir).glob("*.json")):
                with open(f) as fh:
                    goals.update(json.load(fh))
            missing = [n for n in self.data_list if n not in goals]
            if missing:
                raise ValueError(f"{len(missing)} samples have no goal in {goal_dir}, e.g. {missing[0]}")
            self._goals = np.array([goals[n] for n in self.data_list], dtype=np.float32)  # [N, GOAL_DIM]

        def __getitem__(self, idx):
            sample = list(super().__getitem__(idx))
            ego_future = sample[1]  # [FUTURE_LEN, 3] x, y, heading in ego's frame
            # `random` is seeded per DataLoader worker by torch (numpy's global generator is not)
            if random.random() < FINAL_GOAL_PROB:
                goal = self._goals[idx]
            else:
                k = random.randint(1, len(ego_future))
                goal = np.array([ego_future[k - 1, 0], ego_future[k - 1, 1], k * DT], dtype=np.float32)
            sample[0] = np.concatenate([sample[0], goal]).astype(np.float32)
            return tuple(sample)

    return GoalData


def install(augmentation: bool = True) -> None:
    """Patches their Encoder (goal token, when config.goal_conditioned) and, for training, their augmentation (goal
    columns moved into the perturbed ego frame; it imports nuplan, so needs the nuplan shim). Their repo must be
    importable."""
    import torch
    import torch.nn as nn
    from timm.models.layers import Mlp

    from diffusion_planner.model.module import encoder as enc

    if augmentation:
        _install_augmentation()
    if getattr(enc.Encoder, "_goal_installed", False):
        return
    their_init, their_forward = enc.Encoder.__init__, enc.Encoder.forward

    def __init__(self, config):
        their_init(self, config)
        if getattr(config, "goal_conditioned", False):
            self.goal_encoder = Mlp(in_features=GOAL_DIM, hidden_features=config.hidden_dim,
                                    out_features=config.hidden_dim, act_layer=nn.GELU, drop=0.0)

    def forward(self, inputs):
        if not hasattr(self, "goal_encoder"):
            return their_forward(self, inputs)
        # their Encoder.forward, plus the goal token after the lane tokens (no position embedding)
        neighbors = inputs["neighbor_agents_past"]
        B = neighbors.shape[0]
        enc_n, mask_n, pos_n = self.neighbor_encoder(neighbors)
        enc_s, mask_s, pos_s = self.static_encoder(inputs["static_objects"])
        enc_l, mask_l, pos_l = self.lane_encoder(inputs["lanes"], inputs["lanes_speed_limit"], inputs["lanes_has_speed_limit"])
        encoding = torch.cat([enc_n, enc_s, enc_l], dim=1)
        pos = torch.cat([pos_n, pos_s, pos_l], dim=1).view(B * self.token_num, -1)
        mask = torch.cat([mask_n, mask_s, mask_l], dim=1).view(-1)
        pos_emb = torch.zeros((B * self.token_num, self.hidden_dim), device=encoding.device)
        pos_emb[~mask] = self.pos_emb(pos[~mask])
        encoding = encoding + pos_emb.view(B, self.token_num, -1)

        goal = self.goal_encoder(inputs["ego_current_state"][:, STATE_DIM : STATE_DIM + GOAL_DIM])
        encoding = torch.cat([encoding, goal[:, None]], dim=1)
        mask = torch.cat([mask.view(B, self.token_num), torch.zeros((B, 1), dtype=torch.bool, device=mask.device)], dim=1)
        return {"encoding": self.fusion(encoding, mask)}

    enc.Encoder.__init__ = __init__
    enc.Encoder.forward = forward
    enc.Encoder._goal_installed = True


def _install_augmentation() -> None:
    from diffusion_planner.utils import data_augmentation as aug

    if getattr(aug.StatePerturbation, "_goal_installed", False):
        return
    their_centric = aug.StatePerturbation.centric_transform

    def centric_transform(self, inputs, ego_future, neighbors_future):
        state = inputs["ego_current_state"]
        if state.shape[-1] > STATE_DIM:
            cur = state.clone()
            state[:, STATE_DIM : STATE_DIM + 2] = aug.vector_transform(
                state[:, STATE_DIM : STATE_DIM + 2], self.get_transform_matrix_batch(cur), cur[:, :2])
        return their_centric(self, inputs, ego_future, neighbors_future)

    aug.StatePerturbation.centric_transform = centric_transform
    aug.StatePerturbation._goal_installed = True
