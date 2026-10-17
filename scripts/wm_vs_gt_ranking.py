"""Rank a fixed set of diffusion candidates by GT, CV and WM collision scores and
measure how badly CV/WM rank the GT-best candidate. Log-replay setting (t <= 50)."""
import argparse, dataclasses, glob, json, os
import jax, jax.numpy as jnp, numpy as np
from flax import nnx
from scipy.stats import rankdata
from waymax import config as waymax_config

from data.scenario_loader import load_scenario_state_batch_fast
from planner.diffusion_es_planner import DiffusionESPlanner
from scores.scorer_lane_graph import Scorer
from scores import helpers as H
from simulation.evaluation_utils import infer_goals_from_sim_state
from simulation.run_simulation import _load_world_model_forecaster
from data.preprocess import preprocess_simulator_state
from train_diffusion.utils.infer import load_model_for_inference

p = argparse.ArgumentParser()
p.add_argument("--tfrecord_glob", required=True)
p.add_argument("--diffusion_ckpt", required=True)
p.add_argument("--wm_ckpt", required=True)
p.add_argument("--timesteps", default="10,30,50")
p.add_argument("--num_candidates", type=int, default=64)
p.add_argument("--max_batches_per_shard", type=int, default=20)
p.add_argument("--out", required=True)
a = p.parse_args()
NW = 5

inf = load_model_for_inference(checkpoint_path=a.diffusion_ckpt, use_ema=True,
                               metadata_path=None, seed=0, skip_checkpoint_load=False)
policy = nnx.merge(inf.graphdef, inf.params_state, inf.nonparam_state)
planner = DiffusionESPlanner(policy=policy, preprocess_cfg=inf.preprocess_cfg,
                             population_size=a.num_candidates, resample_timesteps=5,
                             elite_size=8, num_iterations=0, num_worlds=NW,
                             metrics=["collision"], other_agent_forecast="gt",
                             world_model_forecaster=None)
wm = _load_world_model_forecaster(argparse.Namespace(
    other_agent_forecast="wm", world_model_ckpt=a.wm_ckpt, world_model_device="auto",
    wm_backend="multifuture", wm_refresh_steps=10, num_collision_worlds=1,
    collision_reduce="any", collision_sample_seed=0,
    wm_planning_mode="expected_independent", mf_collision_method=None, output_dir=a.out))
scorers = {m: Scorer(metrics=["collision"], other_agent_forecast=m,
                     world_model_forecaster=wm if m == "wm" else None)
           for m in ("gt", "cv", "wm")}

def candidates(sim_state, goal, t, key):
    pre, _ = preprocess_simulator_state(sim_state, key, planner.preprocess_cfg,
                                        anchor_step_override=t, goal_step_override=90,
                                        goal_xy_override=goal)
    f = pre.features
    f["inst_features"] = jnp.zeros((NW, planner.preprocess_cfg.inst_dim), jnp.float32)
    f["inst_valid"] = jnp.zeros((NW,), jnp.float32)
    f["subgoal_xy"] = jnp.zeros((NW, 2), jnp.float32)
    f["subgoal_valid"] = jnp.zeros((NW,), jnp.float32)
    cond, inst = planner._compute_condition_jit(f)
    norm = planner._sample_population_jit(cond_bf=cond, inst_cond_bf=inst,
                                          inst_cond_mask_bf=f["inst_valid"], rng=key)
    world, _, _ = planner._postprocess_population(norm, pre)
    return np.asarray(world)                                   # [NW, K, T, 5]

def gt_clearance(sim_state, t, w, cand):                       # [K] min centre distance
    obj, mask, _, _ = H.forecast_other_agents_log_futures(sim_state, t, cand.shape[1], w)
    obj, mask = np.asarray(obj), np.asarray(mask)              # [O,T,5], [O,T]
    T = obj.shape[1]
    d = np.linalg.norm(cand[:, None, :T, :2] - obj[None, :, :, :2], axis=-1)  # [K,O,T]
    d = np.where(mask[None], d, np.inf)
    return d.min(axis=(1, 2))

def eval_ranking(score, safe, best):
    r = rankdata(-score, method="average")                     # 1 = top
    pct = (r[best] - 1) / (len(score) - 1)                     # 0 = top, 1 = bottom
    top = score == score.max()
    p_top_collide = float((~safe[top]).mean())
    pos, neg = score[safe], score[~safe]
    auroc = float(((pos[:, None] > neg[None]).mean()
                   + 0.5 * (pos[:, None] == neg[None]).mean()))
    return pct, p_top_collide, auroc

rows, n_dp = [], 0
key = jax.random.PRNGKey(0)
for shard in sorted(glob.glob(a.tfrecord_glob)):
    cfg = dataclasses.replace(waymax_config.WOD_1_3_1_TRAINING, path=shard,
                              max_num_objects=128, batch_dims=(NW,), shuffle_seed=0)
    for b in range(a.max_batches_per_shard):
        try:
            st, _, ids = load_scenario_state_batch_fast(cfg, list(range(b*NW, b*NW+NW)),
                                                       return_scenario_ids=True)
        except Exception:
            break
        wm.set_scene_ids(ids)
        goal, _, _ = infer_goals_from_sim_state(st)
        for t in map(int, a.timesteps.split(",")):
            key, k = jax.random.split(key)
            cand = candidates(st, goal, t, k)
            for w in range(NW):
                s = {m: np.asarray(scorers[m].compute_score(cand[w], st, t, world_idx=w))
                     for m in scorers}
                n_dp += 1
                safe = s["gt"] > 0.5
                if safe.all() or (~safe).all():
                    continue
                clr = np.where(safe, gt_clearance(st, t, w, cand[w]), -np.inf)
                best = int(np.argmax(clr))
                row = {"shard": os.path.basename(shard), "batch": b, "w": w, "t": t,
                       "n_safe": int(safe.sum())}
                for m in ("wm", "cv"):
                    row[f"{m}_best_pct"], row[f"{m}_top_collide"], row[f"{m}_auroc"] = \
                        eval_ranking(s[m], safe, best)
                rows.append(row)

os.makedirs(a.out, exist_ok=True)
json.dump(rows, open(f"{a.out}/rows.json", "w"))
print(f"decision points {n_dp}, informative {len(rows)} ({len(rows)/max(n_dp,1):.1%})")
for m in ("wm", "cv"):
    pct = np.array([r[f"{m}_best_pct"] for r in rows])
    print(f"{m.upper()}: GT-best median pct {np.median(pct):.2f}  "
          f"bottom-half {np.mean(pct >= .5):.1%}  bottom-quartile {np.mean(pct >= .75):.1%}  "
          f"P(top collides) {np.mean([r[f'{m}_top_collide'] for r in rows]):.1%}  "
          f"AUROC {np.mean([r[f'{m}_auroc'] for r in rows]):.3f}")