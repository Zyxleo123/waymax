"""What action does each multifuture WM mode usually indicate?

For every valid agent, take the Gaussian mean μ of each of the K modes at a
fixed horizon and compare it to the present state (current heading and speed).
Displacement is rotated into the agent frame: +forward along current heading,
+left of current heading. Constant-velocity (CV) is v_now × horizon.

Per agent group (all / moving / near ego / moving-and-near):
  mean/median Δ (fwd, lat)   direction of the mode mean vs now
  implied speed, Δv vs now   |Δ| / horizon compared with present speed
  vsCV fwd                   extra along-track metres vs coasting
  |lat|>2 m / |dθ|>15°       how often the mode actually turns
  action mix                 stay / straight / left / right / brake / accel / reverse

Run from multiFuture_GaussianWorldModel/ with it and its parent repo on PYTHONPATH::

  PYTHONPATH=$PWD:$PWD/.. python analysis/characterize_mf_modes.py \
      --cache_dir <cache> --world_model_ckpt <ckpt.pt>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data.utils import resolve_cache_paths, split_cache_paths
from data.world_model_cache_loader import build_world_model_cache_loader
from model.world_model import LatentMultiAgentWorldModel, WorldModelConfig

_DEFAULT_CKPT = (
    "/zfsauton/scratch/sbellad/world_model/ckpts_multifuture_gaussian/"
    "multiFuture_GaussianWorldModel_20260916_200620/epoch_0450.pt"
)
_DEFAULT_CACHE = "/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz"


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multifuture WM mode actions vs present state.")
    p.add_argument("--cache_dir", default=_DEFAULT_CACHE)
    p.add_argument("--world_model_ckpt", default=_DEFAULT_CKPT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_scenes", type=int, default=96, help="Per anchor; same scenes for every anchor.")
    p.add_argument("--anchor_steps", type=_parse_int_tuple, default=(20, 40))
    p.add_argument("--horizon_s", type=float, default=5.0)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--moving_mps", type=float, default=1.0)
    p.add_argument("--near_m", type=float, default=20.0)
    return p.parse_args()


def rotate_to_agent(xy, yaw):
    """xy [..., 2] in ego frame -> agent heading frame. +x forward, +y left."""
    c = np.cos(yaw)
    s = np.sin(yaw)
    x, y = xy[..., 0], xy[..., 1]
    return np.stack([x * c + y * s, -x * s + y * c], axis=-1)


def action_label(fwd, lat, ddeg, disp, speed0, impl_speed):
    stay = disp < 1.5
    reverse = fwd < -1.0
    left = (lat > 2.0) | ((ddeg > 12.0) & (lat > 0.5))
    right = (lat < -2.0) | ((ddeg < -12.0) & (lat < -0.5))
    brake = (~stay) & (speed0 > 2.0) & (impl_speed < 0.6 * speed0) & (np.abs(lat) < 2.0)
    accel = (~stay) & (impl_speed > speed0 + 1.5) & (np.abs(lat) < 2.0) & (fwd > 1.0)
    # priority: stay, reverse, left/right, then brake/accel, else straight
    lab = np.full(fwd.shape, "straight", dtype=object)
    lab[accel] = "accel/go"
    lab[brake] = "brake"
    lab[left] = "left"
    lab[right] = "right"
    lab[reverse] = "reverse"
    lab[stay] = "stay"
    return lab


@torch.inference_mode()
def collect(model, loader, cfg, device, max_scenes, horizon_s):
    scale = float(cfg.max_range)
    vmax = float(cfg.max_velocity)
    t_idx = int(round(horizon_s / float(cfg.future_dt))) - 1
    t_idx = int(np.clip(t_idx, 0, int(cfg.future_steps) - 1))
    rows = []
    scenes = 0
    for batch in loader:
        if max_scenes > 0 and scenes >= max_scenes:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        scenes += int(batch["target_valid"].shape[0])
        out = model(batch)
        hist = batch["agent_history"].float()
        valid = batch["target_valid"].bool()
        keep = batch["agent_valid"].bool() & valid.any(dim=-1)
        if not keep.any():
            continue

        pred = out["pred_delta_xy_norm_k"].float() * scale  # [B,N,K,T,2] metres, ego frame, vs present
        yaw_vec = out["pred_yaw_vec_k"].float()  # sin, cos of d-yaw vs present heading
        target = batch["target_delta_xy"].float() * scale
        vel_ego = hist[:, :, -1, 2:4] * vmax
        yaw = torch.atan2(hist[:, :, -1, 4], hist[:, :, -1, 5])
        speed = torch.linalg.vector_norm(vel_ego, dim=-1)
        dist = torch.linalg.vector_norm(batch["current_local_xy"].float(), dim=-1)

        times = (t_idx + 1) * float(cfg.future_dt)
        cv = vel_ego[:, :, None, :] * times

        keep_np = keep.cpu().numpy()
        pred_h = pred[:, :, :, t_idx].cpu().numpy()
        dsin = yaw_vec[:, :, :, t_idx, 0].cpu().numpy()
        dcos = yaw_vec[:, :, :, t_idx, 1].cpu().numpy()
        tgt_h = target[:, :, t_idx].cpu().numpy()
        tgt_ok = valid[:, :, t_idx].cpu().numpy()
        yaw_np = yaw.cpu().numpy()
        vel_np = vel_ego.cpu().numpy()
        speed_np = speed.cpu().numpy()
        dist_np = dist.cpu().numpy()
        cv_np = cv.cpu().numpy()
        probs = out["mode_probs"].float().cpu().numpy()

        b, n = keep_np.shape
        for i in range(b):
            for j in range(n):
                if not keep_np[i, j]:
                    continue
                y = yaw_np[i, j]
                pred_a = rotate_to_agent(pred_h[i, j], y)
                cv_a = rotate_to_agent(np.asarray(cv_np[i, j]).reshape(-1), y).reshape(-1)
                vel_a = rotate_to_agent(np.asarray(vel_np[i, j]).reshape(-1), y).reshape(-1)
                tgt_a = rotate_to_agent(tgt_h[i, j], y) if tgt_ok[i, j] else None
                ddeg = np.degrees(np.arctan2(dsin[i, j], dcos[i, j]))
                rows.append(
                    dict(
                        speed=float(speed_np[i, j]),
                        dist=float(dist_np[i, j]),
                        vel_fwd=float(vel_a[0]),
                        vel_lat=float(vel_a[1]),
                        pred=pred_a,
                        cv=cv_a,
                        ddeg=ddeg,
                        probs=probs[i, j],
                        tgt=tgt_a,
                    )
                )
        print(f"  scenes={scenes} agents={len(rows)}", flush=True)
    return rows, scenes


def summarize(rows, mask, k, horizon_s):
    idx = np.where(mask)[0]
    n = len(idx)
    if n == 0:
        return None
    pred = np.stack([rows[i]["pred"] for i in idx])
    cv = np.stack([rows[i]["cv"] for i in idx])
    ddeg = np.stack([rows[i]["ddeg"] for i in idx])
    speed = np.array([rows[i]["speed"] for i in idx])
    vel_fwd = np.array([rows[i]["vel_fwd"] for i in idx])
    probs = np.stack([rows[i]["probs"] for i in idx])
    fwd, lat = pred[..., 0], pred[..., 1]
    disp = np.linalg.norm(pred, axis=-1)
    impl = disp / horizon_s
    extra = pred - cv[:, None, :]
    out = {"n": n, "modes": []}
    for m in range(k):
        lab = action_label(fwd[:, m], lat[:, m], ddeg[:, m], disp[:, m], speed, impl[:, m])
        counts = {name: float((lab == name).mean()) for name in
                  ("stay", "straight", "left", "right", "brake", "accel/go", "reverse")}
        out["modes"].append(
            dict(
                mode=m,
                mean_p=float(probs[:, m].mean()),
                mean_fwd=float(fwd[:, m].mean()),
                mean_lat=float(lat[:, m].mean()),
                med_fwd=float(np.median(fwd[:, m])),
                med_lat=float(np.median(lat[:, m])),
                p10_fwd=float(np.percentile(fwd[:, m], 10)),
                p90_fwd=float(np.percentile(fwd[:, m], 90)),
                p90_abs_lat=float(np.percentile(np.abs(lat[:, m]), 90)),
                frac_lat_gt2=float((np.abs(lat[:, m]) > 2.0).mean()),
                frac_turn15=float((np.abs(ddeg[:, m]) > 15.0).mean()),
                mean_disp=float(disp[:, m].mean()),
                mean_ddeg=float(ddeg[:, m].mean()),
                med_ddeg=float(np.median(ddeg[:, m])),
                mean_impl_mps=float(impl[:, m].mean()),
                mean_speed0=float(speed.mean()),
                mean_dv=float((impl[:, m] - speed).mean()),
                mean_extra_fwd=float(extra[:, m, 0].mean()),
                mean_extra_lat=float(extra[:, m, 1].mean()),
                frac=counts,
            )
        )
    tgts = [rows[i]["tgt"] for i in idx if rows[i]["tgt"] is not None]
    if tgts:
        tgt = np.stack(tgts)
        out["gt"] = dict(
            n=len(tgts),
            mean_fwd=float(tgt[:, 0].mean()),
            mean_lat=float(tgt[:, 1].mean()),
            med_fwd=float(np.median(tgt[:, 0])),
            med_lat=float(np.median(tgt[:, 1])),
            mean_disp=float(np.linalg.norm(tgt, axis=-1).mean()),
        )
    out["present"] = dict(
        mean_speed=float(speed.mean()),
        mean_vel_fwd=float(vel_fwd.mean()),
        med_speed=float(np.median(speed)),
        frac_moving=float((speed > 1.0).mean()),
    )
    return out


def print_block(title, s, horizon_s):
    print(f"\n=== {title}  n={s['n']} ===")
    p = s["present"]
    print(
        f"present: mean speed {p['mean_speed']:.2f} m/s  median {p['med_speed']:.2f}  "
        f"along-heading vel {p['mean_vel_fwd']:.2f}  moving>1m/s {p['frac_moving']:.0%}"
    )
    if "gt" in s:
        g = s["gt"]
        print(
            f"GT @ {horizon_s:.0f}s: mean Δ (fwd {g['mean_fwd']:+.2f} m, lat {g['mean_lat']:+.2f} m)  "
            f"|Δ| {g['mean_disp']:.2f} m  n={g['n']}"
        )
    print(
        f"{'mode':>4} {'p':>5} {'fwd m':>8} {'lat m':>8} {'|Δ| m':>7} {'dθ°':>7} "
        f"{'v̂ m/s':>7} {'Δv vs now':>10} {'vsCV fwd':>9} {'|lat|>2m':>8} {'|dθ|>15':>8}  action mix"
    )
    for m in s["modes"]:
        frac = m["frac"]
        mix = "  ".join(f"{k} {v:.0%}" for k, v in frac.items() if v >= 0.08)
        print(
            f"{m['mode']:4d} {m['mean_p']:5.2f} {m['mean_fwd']:+8.2f} {m['mean_lat']:+8.2f} "
            f"{m['mean_disp']:7.2f} {m['mean_ddeg']:+7.1f} {m['mean_impl_mps']:7.2f} "
            f"{m['mean_dv']:+10.2f} {m['mean_extra_fwd']:+9.2f} {m['frac_lat_gt2']:8.0%} "
            f"{m['frac_turn15']:8.0%}  {mix}"
        )
        print(
            f"{'':4} {'med':5} {m['med_fwd']:+8.2f} {m['med_lat']:+8.2f} "
            f"p10/90 fwd {m['p10_fwd']:+.1f}/{m['p90_fwd']:+.1f}  "
            f"p90 |lat| {m['p90_abs_lat']:.1f}m  med dθ {m['med_ddeg']:+.1f}°"
        )


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.world_model_ckpt)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "latest.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = WorldModelConfig.from_dict(dict(ckpt["config"]))
    model = LatentMultiAgentWorldModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device(args.device)
    model.to(device).eval()
    print(
        f"ckpt={ckpt_path}  K={cfg.num_modes}  future_dt={cfg.future_dt}  "
        f"T={cfg.future_steps}  horizon={args.horizon_s}s  device={device}"
    )

    paths = resolve_cache_paths(args.cache_dir, None)
    _, val_paths = split_cache_paths(paths, val_fraction=args.val_fraction)
    all_rows = []
    for anchor in args.anchor_steps:
        print(f"\nanchor {anchor}", flush=True)
        loader = build_world_model_cache_loader(
            args.cache_dir, cfg, cache_paths=val_paths, batch_size=args.batch_size,
            anchor_steps=(int(anchor),), shuffle=False,
        )
        rows, scenes = collect(
            model, loader, cfg, device, args.max_scenes, args.horizon_s
        )
        print(f"anchor {anchor}: {scenes} scenes, {len(rows)} agents")
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No valid agents were collected.")

    k = int(cfg.num_modes)
    speed = np.array([r["speed"] for r in all_rows])
    dist = np.array([r["dist"] for r in all_rows])
    h = args.horizon_s
    print_block(
        f"ALL agents @ {h:.0f}s, anchors {args.anchor_steps}",
        summarize(all_rows, np.ones(len(all_rows), bool), k, h),
        h,
    )
    print_block(
        f"MOVING >{args.moving_mps:g} m/s @ {h:.0f}s",
        summarize(all_rows, speed > args.moving_mps, k, h),
        h,
    )
    print_block(
        f"NEAR ego ≤{args.near_m:g} m @ {h:.0f}s",
        summarize(all_rows, dist <= args.near_m, k, h),
        h,
    )
    print_block(
        f"MOVING and NEAR @ {h:.0f}s",
        summarize(all_rows, (speed > args.moving_mps) & (dist <= args.near_m), k, h),
        h,
    )


if __name__ == "__main__":
    main()
