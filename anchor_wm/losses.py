"""Forecasting loss (nearest-anchor assignment), scoring loss on self-generated candidates, and the
fine-tuning loss on recorded ES banks (Waymax first-violation steps + cached per-step geometry)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from anchor_wm.config import AnchorWMConfig
from anchor_wm.geometry import comfort_excess, rotate_xy
from anchor_wm.model import gather_k, select_modes


def _masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def gaussian_nll_xy(pred_xy: torch.Tensor, target_xy: torch.Tensor, sigma_xy: torch.Tensor) -> torch.Tensor:
    """Same form as MF-Gaussian's `_gaussian_nll_xy`."""
    sigma = sigma_xy.clamp_min(1e-6)
    z = (pred_xy - target_xy) / sigma
    return 0.5 * z.square().sum(-1) + sigma.log().sum(-1)


def ade_fde(pred_xy_k: torch.Tensor, target_xy: torch.Tensor, valid: torch.Tensor, scale: float):
    """Per-mode ADE / FDE (m): pred [B,N,K,T,2], target [B,N,T,2], valid [B,N,T]."""
    err = (pred_xy_k - target_xy[:, :, None]).norm(dim=-1) * scale                          # [B,N,K,T]
    vf = valid.float()[:, :, None]
    ade = (err * vf).sum(-1) / vf.sum(-1).clamp_min(1.0)
    t = err.shape[-1]
    idx = torch.arange(t, device=err.device).view(1, 1, t)
    last = torch.where(valid, idx, torch.full_like(idx, -1)).amax(-1)                       # [B,N]
    fde = err.gather(-1, last.clamp_min(0)[:, :, None, None].expand(-1, -1, err.shape[2], 1)).squeeze(-1)
    return ade, fde, last >= 0


def wm_loss(dec: dict, f: dict, cfg: AnchorWMConfig) -> tuple[torch.Tensor, dict]:
    rng = float(cfg.max_range)
    valid = f["target_valid_all"].bool()                                                    # [B,N,T]
    sin, cos = dec["heading_sc"][..., 0, None], dec["heading_sc"][..., 1, None]
    target_agent = rotate_xy(f["target_delta_xy"].float(), -sin, cos)                       # normalized
    yaw_t = F.normalize(f["target_yaw_vec"].float(), dim=-1, eps=1e-6)

    # Nearest *anchor* (not nearest prediction) owns the target.
    anchor_ade, _, has = ade_fde(dec["anchors_agent"], target_agent, valid, rng)            # [B,N,K]
    k_star = anchor_ade.argmin(-1)
    pick = lambda x: gather_k(x, k_star[..., None]).squeeze(2)  # noqa: E731
    xy = pick(dec["pred_xy_agent_k"])
    sigma = pick(dec["sigma_k"])
    yaw = pick(dec["yaw_vec_k"])

    near = (f["current_local_xy"].float().norm(dim=-1) <= cfg.collision_focus_radius_m).float()
    agent_w = 1.0 + cfg.collision_focus_weight * near
    agent_w = torch.where(f["is_sdc"].bool(), torch.full_like(agent_w, cfg.ego_loss_weight), agent_w)
    time_w = valid.float() * agent_w[..., None]
    denom = time_w.sum().clamp_min(1.0)
    nll = (gaussian_nll_xy(xy, target_agent, sigma) * time_w).sum() / denom
    yaw_loss = ((1.0 - (yaw * yaw_t).sum(-1).clamp(-1.0, 1.0)) * time_w).sum() / denom

    logp = F.log_softmax(dec["logits"].float(), dim=-1)
    if cfg.cls_soft_temp_m > 0:
        target_p = torch.softmax(-anchor_ade / cfg.cls_soft_temp_m, dim=-1)
        ce = -(target_p * logp).sum(-1)
    else:
        ce = -logp.gather(-1, k_star[..., None]).squeeze(-1)
    aw = has.float() * agent_w
    cls = (ce * aw).sum() / aw.sum().clamp_min(1.0)
    loss = nll + cfg.yaw_loss_weight * yaw_loss + cfg.cls_weight * cls

    with torch.no_grad():
        others = has & ~f["is_sdc"].bool()
        hf = others.float()
        ade_k, fde_k, _ = ade_fde(dec["pred_xy_k"], f["target_delta_xy"].float(), valid, rng)
        sel = select_modes(dec, cfg.num_out_modes, cfg.nms_dist_m, rng)
        top1 = dec["probs"].argmax(-1, keepdim=True)
        ent = -(dec["probs"] * dec["probs"].clamp_min(1e-8).log()).sum(-1)
        usage = torch.bincount(top1[others].flatten(), minlength=cfg.num_anchors).float()
        usage = usage / usage.sum().clamp_min(1.0)
        ego = has & f["is_sdc"].bool()
        metrics = {
            "loss_wm": float(loss), "nll": float(nll), "yaw_loss": float(yaw_loss), "cls_loss": float(cls),
            "min_ade_6": float(_masked_mean(ade_k.gather(-1, sel).amin(-1), hf)),
            "min_fde_6": float(_masked_mean(fde_k.gather(-1, sel).amin(-1), hf)),
            "miss_rate_6": float(_masked_mean((fde_k.gather(-1, sel).amin(-1) > 2.0).float(), hf)),
            "min_ade_all": float(_masked_mean(ade_k.amin(-1), hf)),
            "min_fde_all": float(_masked_mean(fde_k.amin(-1), hf)),
            "top1_ade": float(_masked_mean(ade_k.gather(-1, top1).squeeze(-1), hf)),
            "anchor_oracle_ade": float(_masked_mean(anchor_ade.amin(-1), hf)),
            "mode_entropy": float(_masked_mean(ent, hf)),
            "anchor_usage_entropy": float(-(usage * usage.clamp_min(1e-12).log()).sum()),
            "ego_min_ade_all": float(_masked_mean(ade_k.amin(-1), ego.float())),
        }
    return loss, metrics


def _bce_soft(logit: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return _masked_mean(F.binary_cross_entropy_with_logits(logit.float(), target, reduction="none"), w)


def score_loss(total: torch.Tensor, sub: torch.Tensor, labels: dict, valid: torch.Tensor, *,
               progress_key: str = "goal_gain", progress_margin: float = 2.0,
               w_sub: float = 1.0, w_pair: float = 1.0, w_list: float = 0.5,
               tau_overlap: float = 0.5, tau_offroad: float = 0.5, tau_comfort: float = 0.5,
               list_temp: float = 1.0) -> tuple[torch.Tensor, dict]:
    """total [B,C], sub [B,C,4] (overlap, offroad, progress, comfort logits), valid [B,C]."""
    w = valid.float()
    unsafe = labels["overlap"] | labels["offroad"]
    gt_prog = labels["gt_progress"].clamp_min(1.0)[:, None]
    prog_norm = (labels["progress"] / gt_prog).clamp(0.0, 1.5)

    # Soft sub-score targets (guidance, not a rule order).
    t_overlap = torch.sigmoid(-labels["clearance"] / tau_overlap)
    t_offroad = torch.sigmoid(labels["offroad_sd"] / tau_offroad)
    t_comfort = torch.sigmoid(labels["comfort"] / tau_comfort)
    l_sub = (_bce_soft(sub[..., 0], t_overlap, w) + _bce_soft(sub[..., 1], t_offroad, w)
             + _masked_mean(F.smooth_l1_loss(sub[..., 2].float(), prog_norm, reduction="none", beta=0.1), w)
             + 0.5 * _bce_soft(sub[..., 3], t_comfort, w))

    # Pairwise preferences, as the current discriminator: safe > unsafe; among safe, >= margin more
    # progress (`progress_key`: goal_gain, as B0 trains and is evaluated, or route progress).
    vv = valid[:, :, None] & valid[:, None, :]
    safe = ~unsafe
    p = labels[progress_key]
    safety_pairs = safe[:, :, None] & unsafe[:, None, :] & vv
    prog_pairs = safe[:, :, None] & safe[:, None, :] & ((p[:, :, None] - p[:, None, :]) >= progress_margin) & vv
    diff = total.float()[:, :, None] - total.float()[:, None, :]
    pl = F.softplus(-diff)
    l_safety = _masked_mean(pl, safety_pairs.float())
    l_prog = _masked_mean(pl, prog_pairs.float())
    has_s, has_p = safety_pairs.any(), prog_pairs.any()
    l_pair = (l_safety * has_s + l_prog * has_p) / (has_s.float() + has_p.float()).clamp_min(1.0)

    # Listwise: softmax over the group toward softmax(soft reward / T).
    reward = -5.0 * t_overlap - 5.0 * t_offroad + prog_norm - 0.5 * t_comfort
    neg = torch.finfo(torch.float32).min
    target = torch.softmax(reward.masked_fill(~valid, neg) / list_temp, -1)
    logp = F.log_softmax(total.float().masked_fill(~valid, neg), -1)
    l_list = -(target * logp.masked_fill(~valid, 0.0)).sum(-1).mean()

    loss = w_sub * l_sub + w_pair * l_pair + w_list * l_list
    with torch.no_grad():
        acc_pairs = safety_pairs | prog_pairs
        pair_acc = _masked_mean((diff > 0).float() + 0.5 * (diff == 0).float(), acc_pairs.float())
        pick = total.float().masked_fill(~valid, neg).argmax(-1)
        ar = torch.arange(total.shape[0], device=total.device)
        metrics = {"loss_score": float(loss), "score_sub": float(l_sub), "score_pair": float(l_pair),
                   "score_list": float(l_list), "score_pair_acc": float(pair_acc),
                   "picked_unsafe": float(unsafe[ar, pick].float().mean()),
                   "cand_unsafe_rate": float(_masked_mean(unsafe.float(), w)),
                   "picked_progress": float(p[ar, pick].mean())}
    return loss, metrics


# ---- fine-tuning on recorded ES banks -----------------------------------------------------------

def bank_targets(batch: dict, *, tau: float = 0.5, late_weight: float = 0.5, dt: float = 0.2) -> dict:
    """Per-candidate targets [B,K] from Waymax first-violation steps and the cached 0.1 s geometry.

    Time weight w(t) = 1 - late_weight * t / S (earlier violations count more). Soft overlap / offroad =
    max_t w(t) * sigmoid(-gap_t / tau) / sigmoid(sd_t / tau), forced up to w(first step) where Waymax
    flags the candidate and capped at 0.5 where it does not (2-D geometry vs Waymax's 3-D offroad)."""
    ov, off = batch["overlap_step"].long(), batch["offroad_step"].long()
    h = batch["horizon"].long().clamp_min(1)[:, None]
    gap, sd = batch["b_gap"].float(), batch["b_offroad"].float()
    s = gap.shape[-1]
    w_t = 1.0 - late_weight * torch.arange(1, s + 1, device=gap.device, dtype=gap.dtype) / s
    w_at = lambda step: 1.0 - late_weight * step.clamp(1, s).to(gap.dtype) / s  # noqa: E731
    soft_o = (w_t * torch.sigmoid(-gap / tau)).amax(-1)
    soft_f = (w_t * torch.sigmoid(sd / tau)).amax(-1)
    soft_o = torch.where(ov > 0, torch.maximum(soft_o, w_at(ov)), soft_o.clamp(max=0.5))
    soft_f = torch.where(off > 0, torch.maximum(soft_f, w_at(off)), soft_f.clamp(max=0.5))
    unsafe = (ov > 0) | (off > 0)
    big = (h + 1).expand_as(ov)
    first = torch.minimum(torch.where(ov > 0, ov, big), torch.where(off > 0, off, big))
    ttv = (first.to(gap.dtype) / h).clamp(max=1.0)                                # 1 = no violation
    severity = (-gap).amax(-1).clamp(min=0) + sd.amax(-1).clamp(min=0)              # overlap depth + offroad depth
    gg = batch["goal_gain"].float()
    gg_norm = gg / gg.amax(-1, keepdim=True).clamp_min(1.0)
    gd = batch["final_goal_dist"].float()
    v0 = batch["ego_vel"].float()[:, None].expand(-1, gg.shape[1], 2)
    comfort = torch.sigmoid(comfort_excess(batch["cand_xy"].float(), v0, dt) / tau)
    return {"unsafe": unsafe, "first": first, "ttv": ttv, "soft_o": soft_o, "soft_f": soft_f, "severity": severity,
            "gg": gg, "gg_norm": gg_norm, "gd": gd, "gd_norm": (gd / 10.0).clamp(0.0, 2.0), "comfort": comfort}


def _pair_mean(diff: torch.Tensor, pairs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    loss = (F.softplus(-diff) * pairs).sum() / pairs.sum().clamp_min(1)
    return loss, pairs.any()


def bank_score_loss(total: torch.Tensor, sub: torch.Tensor, aux: dict, tg: dict, batch: dict, *,
                    progress_margin: float = 2.0, goal_margin: float = 1.0, time_margin_steps: int = 5,
                    severity_margin: float = 0.3, w_pairs: dict | None = None, w_sub: float = 1.0,
                    w_list: float = 0.5, w_aux: float = 0.5, w_distill: float = 0.0,
                    reward_weights: tuple = (5.0, 5.0, 2.0, 1.0, 0.5)) -> tuple[torch.Tensor, dict]:
    """total / sub [B,K(,4)], aux {"ttv","goaldist"} [B,K] (may be empty), tg = bank_targets(batch)."""
    wp = {"safety": 1.0, "progress": 1.0, "goal": 0.5, "unsafe_time": 0.5, "unsafe_severity": 0.25}
    wp.update(w_pairs or {})
    s = total.float()
    diff = s[:, :, None] - s[:, None, :]
    unsafe, safe = tg["unsafe"], ~tg["unsafe"]
    both_safe = safe[:, :, None] & safe[:, None, :]
    both_unsafe = unsafe[:, :, None] & unsafe[:, None, :]
    gg, gd, first, sev = tg["gg"], tg["gd"], tg["first"], tg["severity"]
    dgg = gg[:, :, None] - gg[:, None, :]
    pairs = {
        "safety": safe[:, :, None] & unsafe[:, None, :],
        "progress": both_safe & (dgg >= progress_margin),
        "goal": both_safe & (dgg.abs() < progress_margin) & ((gd[:, None, :] - gd[:, :, None]) >= goal_margin),
        "unsafe_time": both_unsafe & ((first[:, :, None] - first[:, None, :]) >= time_margin_steps),
        "unsafe_severity": both_unsafe & ((first[:, :, None] - first[:, None, :]).abs() < time_margin_steps)
                           & ((sev[:, None, :] - sev[:, :, None]) >= severity_margin),
    }
    num, den = s.new_zeros(()), s.new_zeros(())
    metrics = {}
    for name, p in pairs.items():
        loss_p, has = _pair_mean(diff, p.float())
        num = num + wp[name] * loss_p * has
        den = den + wp[name] * has.float()
        with torch.no_grad():
            metrics[f"pair_acc_{name}"] = float(((diff > 0).float() * p).sum() / p.float().sum().clamp_min(1))
            metrics[f"pairs_{name}"] = float(p.float().sum() / s.shape[0])
    l_pair = num / den.clamp_min(1e-6)

    bce = lambda logit, y: F.binary_cross_entropy_with_logits(logit.float(), y.float())  # noqa: E731
    l_sub = (bce(sub[..., 0], tg["soft_o"]) + bce(sub[..., 1], tg["soft_f"])
             + F.smooth_l1_loss(sub[..., 2].float(), tg["gg_norm"], beta=0.1) + 0.5 * bce(sub[..., 3], tg["comfort"]))
    l_aux = s.new_zeros(())
    if aux:
        l_aux = (F.smooth_l1_loss(aux["ttv"].float(), tg["ttv"], beta=0.1)
                 + F.smooth_l1_loss(aux["goaldist"].float(), tg["gd_norm"], beta=0.1))
    a_o, a_f, a_p, a_g, a_c = reward_weights
    reward = -a_o * tg["soft_o"] - a_f * tg["soft_f"] + a_p * tg["gg_norm"] - a_g * tg["gd_norm"] - a_c * tg["comfort"]
    l_list = -(torch.softmax(reward, -1) * torch.log_softmax(s, -1)).sum(-1).mean()
    l_distill = s.new_zeros(())
    if w_distill > 0:
        rule = torch.log((batch["es_collision"].float() * batch["es_offroad"].float()).clamp(1e-3, 1.0))
        l_distill = -(torch.softmax(rule, -1) * torch.log_softmax(s, -1)).sum(-1).mean()
    loss = l_pair + w_sub * l_sub + w_list * l_list + w_aux * l_aux + w_distill * l_distill
    with torch.no_grad():
        pick = s.argmax(-1)
        ar = torch.arange(s.shape[0], device=s.device)
        metrics.update(loss=float(loss), l_pair=float(l_pair), l_sub=float(l_sub), l_list=float(l_list),
                       l_aux=float(l_aux), l_distill=float(l_distill),
                       picked_unsafe=float(unsafe[ar, pick].float().mean()),
                       picked_overlap=float((batch["overlap_step"][ar, pick] > 0).float().mean()),
                       picked_goal_gain=float(gg[ar, pick].mean()), picked_goal_dist=float(gd[ar, pick].mean()),
                       cand_unsafe_rate=float(unsafe.float().mean()))
    return loss, metrics


@torch.no_grad()
def bank_eval_metrics(total: torch.Tensor, tg: dict, progress_margin: float = 2.0) -> dict:
    """B0-comparable metrics over all candidates: pairwise accuracy on (safe > unsafe; >= margin more goal
    gain among safe) pairs, safety AUROC, and the picked candidate's unsafe rate / goal gain / goal distance."""
    s = total.float()
    diff = s[:, :, None] - s[:, None, :]
    unsafe, safe = tg["unsafe"], ~tg["unsafe"]
    dgg = tg["gg"][:, :, None] - tg["gg"][:, None, :]
    pref = (safe[:, :, None] & unsafe[:, None, :]) | (safe[:, :, None] & safe[:, None, :] & (dgg >= progress_margin))
    correct = (diff > 0).float() + 0.5 * (diff == 0).float()
    per_group = (correct * pref).sum((1, 2)) / pref.float().sum((1, 2)).clamp_min(1)
    has = pref.any(-1).any(-1)
    ranks = s.argsort(-1).argsort(-1).float()
    n_pos, n_neg = safe.sum(-1).float(), unsafe.sum(-1).float()
    auroc = ((ranks * safe).sum(-1) - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg).clamp_min(1)
    both = (n_pos > 0) & (n_neg > 0)
    pick = s.argmax(-1)
    ar = torch.arange(s.shape[0], device=s.device)
    return {"pairwise_accuracy": float(per_group[has].mean()) if has.any() else float("nan"),
            "safety_auroc": float(auroc[both].mean()) if both.any() else float("nan"),
            "picked_unsafe": float(unsafe[ar, pick].float().mean()),
            "picked_goal_gain": float(tg["gg"][ar, pick].mean()),
            "picked_goal_dist": float(tg["gd"][ar, pick].mean()),
            "picked_ttv": float(tg["ttv"][ar, pick].mean())}
