"""Redraw the hardness-gate curve from gate_analysis.py's JSON (CPU, no model)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_png", required=True)
    a = ap.parse_args()
    r = json.loads(Path(a.in_json).read_text())
    c = r["curve"]
    x = [100 * v for v in c["gate_frac"]]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.fill_between(x, c["ci95_lo"], c["ci95_hi"], alpha=0.2, color="C0", label="95% CI (scene bootstrap)")
    for i, s in enumerate(c["delta_per_seed"]):
        ax.plot(x, s, color="C0", lw=0.8, alpha=0.45, label="per checkpoint" if i == 0 else None)
    ax.plot(x, c["delta_seed_mean"], color="C0", lw=2, marker="o", ms=3, label="seed mean")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(r["oracle_ceiling"], color="C2", ls="--", lw=1, label=f"oracle ceiling {r['oracle_ceiling']:+.4f}")
    cv = r["cv"]
    ax.axhspan(cv["delta_p05"], cv["delta_p95"], color="C3", alpha=0.12,
               label=f"cross-validated pick {cv['delta_mean']:+.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("share of corpus running the fine-tune (%)")
    ax.set_ylabel("corpus solve rate vs base")
    ax.set_title(f"Hardness-gated safety fine-tune (risk AUC {r['auc_failure_vs_success']:.3f})")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    Path(a.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out_png, dpi=150)
    print(f"[plot] wrote {a.out_png}")


if __name__ == "__main__":
    main()
