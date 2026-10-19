"""BEV overlays of the six mode means vs the current heading/speed arrow.

Left panel: ego-centric scene (ego faces up). Nearby boxes, map, and the six
mode-mean polylines for moving agents.

Right panel: agent-centric zoom on one moving neighbour. Current velocity
arrow is the 5 s constant-velocity coast; dashed gray is that same CV path;
black is logged GT; coloured solid lines are the K Gaussian means.

Run from multiFuture_GaussianWorldModel/::

  PYTHONPATH=$PWD:$PWD/.. python analysis/viz_mf_mode_fan.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from data.utils import resolve_cache_paths, split_cache_paths
from data.world_model_cache_loader import build_world_model_cache_loader
from model.world_model import LatentMultiAgentWorldModel, WorldModelConfig

_DEFAULT_CKPT = (
    "/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/world_model/ckpts_multifuture_gaussian/"
    "multiFuture_GaussianWorldModel_20260916_200620/epoch_0450.pt"
)
_DEFAULT_CACHE = "/data/user_data/yixiz/autonlab/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz"
_DEFAULT_OUT = "/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/world_model/eval_per_anchor/mode_fan_bev"

MODE_COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#B279A2",
    "#9D755D",
)
COLOR_GT = "#111111"
COLOR_CV = "#6B6B6B"
COLOR_EGO = "#2E7D32"
COLOR_OTHER = np.array([0.55, 0.55, 0.55])
COLOR_FOCUS = np.array([0.12, 0.12, 0.12])
COLOR_MAP = "#C8C8C8"


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BEV overlays of multifuture mode means.")
    p.add_argument("--cache_dir", default=_DEFAULT_CACHE)
    p.add_argument("--world_model_ckpt", default=_DEFAULT_CKPT)
    p.add_argument("--out_dir", default=_DEFAULT_OUT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_scenes", type=int, default=128, help="Scenes to scan per anchor.")
    p.add_argument("--n_scenes", type=int, default=6, help="How many example PNGs to write.")
    p.add_argument("--anchor_steps", type=_parse_int_tuple, default=(20,))
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--horizon_s", type=float, default=5.0)
    p.add_argument("--moving_mps", type=float, default=3.0)
    p.add_argument("--near_m", type=float, default=25.0)
    p.add_argument("--min_spread_m", type=float, default=6.0, help="Min pairwise 5s endpoint spread.")
    return p.parse_args()


def _upright(xy: np.ndarray) -> np.ndarray:
    """Ego/agent frame (+x forward, +y left) -> plot frame (y up = forward)."""
    x = np.asarray(xy)[..., 0]
    y = np.asarray(xy)[..., 1]
    return np.stack([-y, x], axis=-1)


def rotate_about(xy: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    """Translate to origin, rotate by -yaw so heading becomes +x."""
    rel = np.asarray(xy) - np.asarray(origin)
    c, s = np.cos(-yaw), np.sin(-yaw)
    x, y = rel[..., 0], rel[..., 1]
    return np.stack([x * c - y * s, x * s + y * c], axis=-1)


def _draw_box(ax, xy, length, width, yaw, color, alpha=1.0, lw=1.2, z=5):
    c, s = np.cos(yaw), np.sin(yaw)
    pt = np.array([xy[0], xy[1]])
    u = np.array([c, s])
    ut = np.array([s, -c])
    l2, w2 = 0.5 * float(length), 0.5 * float(width)
    tl = pt + l2 * u - w2 * ut
    tr = pt + l2 * u + w2 * ut
    br = pt - l2 * u + w2 * ut
    bl = pt - l2 * u - w2 * ut
    cf = pt + l2 * u
    cl = pt - w2 * ut
    cr = pt + w2 * ut
    corners = np.stack([tl, tr, br, bl, tl], axis=0)
    ax.plot(corners[:, 0], corners[:, 1], color=color, lw=lw, alpha=alpha, zorder=z)
    head = np.stack([cl, cr, cf, cl], axis=0)
    ax.plot(head[:, 0], head[:, 1], color=color, lw=lw, alpha=alpha, zorder=z)


def _draw_poly(ax, xy, *, color, lw, ls="-", alpha=1.0, z=8):
    pts = np.asarray(xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls=ls, alpha=alpha, zorder=z, solid_capstyle="round")


def _draw_map(ax, map_xy: np.ndarray, map_valid: np.ndarray) -> None:
    n_seg = map_valid.shape[0]
    for s in range(n_seg):
        m = map_valid[s]
        if int(m.sum()) < 2:
            continue
        _draw_poly(ax, map_xy[s, m], color=COLOR_MAP, lw=0.7, alpha=0.9, z=1)


def _arrow(ax, start, end, color, lw=2.2):
    arr = FancyArrowPatch(
        tuple(start),
        tuple(end),
        arrowstyle="-|>",
        mutation_scale=14,
        lw=lw,
        color=color,
        zorder=20,
    )
    ax.add_patch(arr)


def _style_ax(ax, half_x: float, y_lo: float, y_hi: float, title: str) -> None:
    ax.set_xlim(-half_x, half_x)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("m")
    ax.set_ylabel("m")
    ax.grid(True, alpha=0.2, lw=0.4)
    ax.set_clip_on(True)


def _unpack_scene(batch, out, cfg, bi: int, horizon_s: float) -> dict:
    scale = float(cfg.max_range)
    vmax = float(cfg.max_velocity)
    t_idx = int(np.clip(round(horizon_s / float(cfg.future_dt)) - 1, 0, cfg.future_steps - 1))
    hist = batch["agent_history"][bi].numpy()
    valid_now = batch["agent_valid"][bi].numpy().astype(bool)
    is_sdc = batch["is_sdc"][bi].numpy().astype(bool)
    tgt_valid = batch["target_valid"][bi].numpy().astype(bool)
    xy = batch["current_local_xy"][bi].numpy()
    vel = hist[:, -1, 2:4] * vmax
    yaw = np.arctan2(hist[:, -1, 4], hist[:, -1, 5])
    length = hist[:, -1, 6] * float(cfg.max_width)
    width = hist[:, -1, 7] * float(cfg.max_width)
    speed = np.linalg.norm(vel, axis=-1)
    dist = np.linalg.norm(xy, axis=-1)

    pred = out["pred_delta_xy_norm_k"][bi].numpy() * scale  # [N,K,T,2]
    probs = out["mode_probs"][bi].numpy()
    target = batch["target_delta_xy"][bi].numpy() * scale
    times = (np.arange(1, cfg.future_steps + 1, dtype=np.float32) * float(cfg.future_dt))
    mode_xy = xy[:, None, None, :] + pred
    gt_xy = xy[:, None, :] + target
    cv_xy = xy[:, None, :] + vel[:, None, :] * times[None, :, None]
    end = mode_xy[:, :, t_idx]
    # pairwise spread of 5s endpoints
    d = end[:, None, :, :] - end[:, :, None, :]
    spread = np.sqrt((d ** 2).sum(-1)).max(axis=(-1, -2))

    map_xy = batch["map_features"][bi].numpy()[..., :2] * scale
    map_valid = batch["map_valid"][bi].numpy().astype(bool)
    return dict(
        xy=xy, vel=vel, yaw=yaw, length=length, width=width, speed=speed, dist=dist,
        valid=valid_now, is_sdc=is_sdc, tgt_ok=tgt_valid, probs=probs,
        mode_xy=mode_xy, gt_xy=gt_xy, cv_xy=cv_xy, spread=spread,
        map_xy=map_xy, map_valid=map_valid, t_idx=t_idx, times=times,
    )


def _pick_focus(sc: dict, moving_mps: float, near_m: float, min_spread_m: float) -> int | None:
    keep = (
        sc["valid"]
        & (~sc["is_sdc"])
        & sc["tgt_ok"].any(axis=-1)
        & (sc["speed"] >= moving_mps)
        & (sc["dist"] <= near_m)
        & (sc["spread"] >= min_spread_m)
    )
    if not keep.any():
        return None
    idx = np.where(keep)[0]
    # Prefer nearby, high-spread movers.
    score = sc["spread"][idx] + 0.15 * sc["speed"][idx] - 0.03 * sc["dist"][idx]
    return int(idx[int(np.argmax(score))])


def render_scene(sc: dict, focus: int, out_png: Path, horizon_s: float) -> None:
    k = int(sc["mode_xy"].shape[1])
    t_idx = int(sc["t_idx"])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 9.2), dpi=130)

    # ---- left: scene BEV, ego-up ----
    ax = axes[0]
    _draw_map(ax, _upright(sc["map_xy"]), sc["map_valid"])
    n = sc["xy"].shape[0]
    for i in range(n):
        if not sc["valid"][i]:
            continue
        xy_p = _upright(sc["xy"][i])
        yaw_p = float(sc["yaw"][i]) + 0.5 * np.pi
        is_ego = bool(sc["is_sdc"][i])
        is_focus = i == focus
        if is_ego:
            color, alpha, lw = COLOR_EGO, 1.0, 1.8
        elif is_focus:
            color, alpha, lw = COLOR_FOCUS, 1.0, 1.8
        else:
            color, alpha, lw = COLOR_OTHER, 0.55, 0.9
        _draw_box(ax, xy_p, sc["length"][i], sc["width"][i], yaw_p, color, alpha=alpha, lw=lw, z=6)
        moving = sc["speed"][i] > 1.0 and (not is_ego) and sc["dist"][i] <= 30.0
        if moving:
            a = 1.0 if is_focus else 0.35
            lw_m = 2.0 if is_focus else 0.9
            for m in range(k):
                _draw_poly(
                    ax,
                    _upright(sc["mode_xy"][i, m, : t_idx + 1]),
                    color=MODE_COLORS[m % len(MODE_COLORS)],
                    lw=lw_m,
                    alpha=a,
                    z=10 if is_focus else 7,
                )
    # ego arrow (current velocity, 5 s)
    ego = int(np.argmax(sc["is_sdc"].astype(np.int32)))
    if sc["valid"][ego]:
        start = _upright(sc["xy"][ego])
        end = _upright(sc["xy"][ego] + sc["vel"][ego] * horizon_s)
        _arrow(ax, start, end, COLOR_EGO, lw=2.0)
    corners = [
        _upright(sc["xy"][ego]).reshape(1, 2),
        _upright(sc["xy"][focus]).reshape(1, 2),
        _upright(sc["mode_xy"][focus, :, t_idx]).reshape(-1, 2),
        _upright(sc["cv_xy"][focus, t_idx]).reshape(1, 2),
    ]
    hull = np.concatenate(corners, axis=0)
    pad = 10.0
    ax.set_xlim(float(hull[:, 0].min()) - pad, float(hull[:, 0].max()) + pad)
    ax.set_ylim(float(hull[:, 1].min()) - pad, float(hull[:, 1].max()) + pad)
    ax.set_aspect("equal")
    ax.set_title("scene (ego up)  ·  focus agent in bold", fontsize=10)
    ax.set_xlabel("m")
    ax.set_ylabel("m")
    ax.grid(True, alpha=0.2, lw=0.4)

    # ---- right: agent-centric fan ----
    ax = axes[1]
    origin = sc["xy"][focus]
    yaw = float(sc["yaw"][focus])
    def to_agent_up(xy):
        return _upright(rotate_about(xy, origin, yaw))

    # local map around the agent
    map_a = rotate_about(sc["map_xy"], origin, yaw)
    _draw_map(ax, _upright(map_a), sc["map_valid"])

    _draw_box(
        ax,
        np.array([0.0, 0.0]),
        sc["length"][focus],
        sc["width"][focus],
        0.5 * np.pi,
        COLOR_FOCUS,
        lw=2.0,
        z=12,
    )
    # CV path + arrow
    cv = to_agent_up(sc["cv_xy"][focus, : t_idx + 1])
    _draw_poly(ax, cv, color=COLOR_CV, lw=2.0, ls="--", z=11)
    _arrow(ax, np.array([0.0, 0.0]), cv[-1], COLOR_CV, lw=2.4)
    # GT
    gt_m = sc["tgt_ok"][focus]
    if gt_m.any():
        _draw_poly(ax, to_agent_up(sc["gt_xy"][focus, gt_m]), color=COLOR_GT, lw=2.4, ls="-", z=12)
    # modes
    for m in range(k):
        poly = to_agent_up(sc["mode_xy"][focus, m, : t_idx + 1])
        col = MODE_COLORS[m % len(MODE_COLORS)]
        _draw_poly(ax, poly, color=col, lw=2.2, z=13)
        ax.scatter(
            poly[-1, 0], poly[-1, 1], s=28, color=col, zorder=14,
            edgecolors="white", linewidths=0.4, clip_on=True,
        )
        ax.text(
            poly[-1, 0] + 0.8, poly[-1, 1], str(m), color=col, fontsize=8,
            fontweight="bold", zorder=15, clip_on=True, va="center",
        )
    ends = [to_agent_up(sc["mode_xy"][focus, m, t_idx])[1] for m in range(k)]
    ends.append(float(cv[-1, 1]))
    y_hi = max(40.0, float(np.max(ends)) + 6.0)
    _style_ax(ax, 16.0, -8.0, y_hi, "focus agent  ·  heading up  ·  arrow = CV @ 5s")

    spd = sc["speed"][focus]
    dist = sc["dist"][focus]
    p = sc["probs"][focus]
    ptxt = "  ".join(f"p{m}={p[m]:.2f}" for m in range(k))
    fig.suptitle(
        f"mode-mean fan vs present  ·  focus speed {spd:.1f} m/s  dist {dist:.1f} m  "
        f"5s spread {sc['spread'][focus]:.1f} m\n{ptxt}",
        fontsize=10,
    )
    legend = [
        Line2D([0], [0], color=MODE_COLORS[m], lw=2.2, label=f"mode {m}")
        for m in range(k)
    ] + [
        Line2D([0], [0], color=COLOR_CV, lw=2, ls="--", label="CV (present v × 5s)"),
        Line2D([0], [0], color=COLOR_GT, lw=2.4, label="logged GT"),
        Line2D([0], [0], color=COLOR_EGO, lw=2, label="ego"),
    ]
    axes[0].legend(handles=legend, loc="lower left", fontsize=7.5, framealpha=0.92, ncol=2)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.08, wspace=0.18)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def render_gallery(picks: list[tuple[dict, int]], out_png: Path, horizon_s: float) -> None:
    n = len(picks)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 7.2 * rows), dpi=130)
    axes = np.atleast_1d(axes).ravel()
    k = int(picks[0][0]["mode_xy"].shape[1])
    for ax in axes:
        ax.set_visible(False)
    for ax, (sc, focus) in zip(axes, picks):
        ax.set_visible(True)
        origin = sc["xy"][focus]
        yaw = float(sc["yaw"][focus])
        t_idx = int(sc["t_idx"])

        def to_agent_up(xy, origin=origin, yaw=yaw):
            return _upright(rotate_about(xy, origin, yaw))

        _draw_map(ax, _upright(rotate_about(sc["map_xy"], origin, yaw)), sc["map_valid"])
        _draw_box(ax, np.array([0.0, 0.0]), sc["length"][focus], sc["width"][focus], 0.5 * np.pi, COLOR_FOCUS, lw=1.8, z=12)
        cv = to_agent_up(sc["cv_xy"][focus, : t_idx + 1])
        _draw_poly(ax, cv, color=COLOR_CV, lw=1.6, ls="--", z=11)
        _arrow(ax, np.array([0.0, 0.0]), cv[-1], COLOR_CV, lw=1.8)
        gt_m = sc["tgt_ok"][focus]
        if gt_m.any():
            _draw_poly(ax, to_agent_up(sc["gt_xy"][focus, gt_m]), color=COLOR_GT, lw=2.0, z=12)
        for m in range(k):
            poly = to_agent_up(sc["mode_xy"][focus, m, : t_idx + 1])
            col = MODE_COLORS[m % len(MODE_COLORS)]
            _draw_poly(ax, poly, color=col, lw=1.8, z=13)
            ax.scatter(
                poly[-1, 0], poly[-1, 1], s=18, color=col, zorder=14,
                edgecolors="white", linewidths=0.3, clip_on=True,
            )
            ax.text(
                poly[-1, 0] + 0.7, poly[-1, 1], str(m), color=col, fontsize=7,
                fontweight="bold", zorder=15, clip_on=True, va="center",
            )
        ends = [to_agent_up(sc["mode_xy"][focus, m, t_idx])[1] for m in range(k)]
        ends.append(float(cv[-1, 1]))
        y_hi = max(40.0, float(np.max(ends)) + 6.0)
        _style_ax(ax, 14.0, -8.0, y_hi, f"{sc['speed'][focus]:.1f} m/s  ·  {sc['dist'][focus]:.0f} m from ego")
    legend = [
        Line2D([0], [0], color=MODE_COLORS[m], lw=2, label=f"m{m}") for m in range(k)
    ] + [
        Line2D([0], [0], color=COLOR_CV, lw=1.6, ls="--", label="CV"),
        Line2D([0], [0], color=COLOR_GT, lw=2, label="GT"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=k + 2, fontsize=8, frameon=False)
    fig.suptitle("Agent-centric mode means vs present velocity (arrow = CV @ 5s)", fontsize=11)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.08, wspace=0.22, hspace=0.22)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


@torch.inference_mode()
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

    paths = resolve_cache_paths(args.cache_dir, None)
    _, val_paths = split_cache_paths(paths, val_fraction=args.val_fraction)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    picks: list[tuple[dict, int]] = []
    seen = 0
    for anchor in args.anchor_steps:
        if len(picks) >= args.n_scenes:
            break
        loader = build_world_model_cache_loader(
            args.cache_dir, cfg, cache_paths=val_paths, batch_size=args.batch_size,
            anchor_steps=(int(anchor),), shuffle=False,
        )
        for batch in loader:
            if seen >= args.max_scenes or len(picks) >= args.n_scenes:
                break
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            out = model(batch_dev)
            out_cpu = {k: v.detach().cpu() for k, v in out.items()}
            batch_cpu = {k: v.cpu() for k, v in batch.items()}
            b = int(batch_cpu["agent_valid"].shape[0])
            for bi in range(b):
                if seen >= args.max_scenes or len(picks) >= args.n_scenes:
                    break
                seen += 1
                sc = _unpack_scene(batch_cpu, out_cpu, cfg, bi, args.horizon_s)
                focus = _pick_focus(sc, args.moving_mps, args.near_m, args.min_spread_m)
                if focus is None:
                    continue
                picks.append((sc, focus))
                png = out_dir / f"mode_fan_a{anchor}_s{seen:04d}_n{focus:03d}.png"
                render_scene(sc, focus, png, args.horizon_s)
                print(f"wrote {png}  speed={sc['speed'][focus]:.1f}  spread={sc['spread'][focus]:.1f}", flush=True)

    if not picks:
        raise RuntimeError("No moving nearby agents with a visible mode fan were found.")
    gallery = out_dir / "mode_fan_gallery.png"
    render_gallery(picks, gallery, args.horizon_s)
    print(f"wrote {gallery}  ({len(picks)} agents from {seen} scanned scenes)")


if __name__ == "__main__":
    main()
