#!/usr/bin/env python3
"""Streamlit viewer: Diff-ES OK vs FM-ES fail — full scenario videos + FM diagnostics.

Expects data from::

  python scripts/export_diff_ok_fm_fail_viewer.py --n 20 --seed 0

Run::

  streamlit run apps/diff_vs_fm_fail_viewer/app.py \\
    -- --data_dir /data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/diff_ok_fm_fail_viewer_gt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import streamlit as st

DEFAULT_DATA_DIR = "/data/user_data/yixiz/autonlab/zfsauton/scratch/sbellad/immediate_exp/diff_ok_fm_fail_viewer_gt"


def parse_known_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    args, _ = p.parse_known_args()
    return args


def load_index(data_dir: Path) -> dict:
    cases_path = data_dir / "cases.json"
    if not cases_path.exists():
        return {}
    return json.loads(cases_path.read_text())


def load_diagnostics(case_dir: Path) -> dict | None:
    p = case_dir / "diagnostics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> None:
    args = parse_known_args()
    data_dir = Path(args.data_dir)

    st.set_page_config(page_title="Diff-ES OK vs FM-ES Fail", layout="wide")
    st.title("Diffusion-ES success / FM-ES fail — scenario videos")
    st.caption(f"Data: `{data_dir}`")

    status = data_dir / "STATUS.txt"
    if status.exists():
        st.info(f"Export status: `{status.read_text().strip()}`")

    index = load_index(data_dir)
    if not index:
        st.error(
            "No `cases.json` found. First run on a GPU node:\n\n"
            "`sbatch slurm_export_diff_ok_fm_fail_viewer.sh`"
        )
        return

    hp = index.get("hyperparams", {})
    with st.expander("ES hyperparams used for these runs (FM noising)", expanded=True):
        st.write(hp.get("note_on_noising", ""))
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Diffusion-ES")
            st.json(hp.get("diffusion_es", {}))
        with c2:
            st.subheader("FM-ES")
            st.json(hp.get("fm_es", {}))
            st.caption(
                "`resample_t_start` larger ⇒ more noise toward the prior. "
                "fail1500 used **0.2** (mild). If FM looks under-exploratory, try 0.4–0.8."
            )

    cases = index.get("cases", [])
    st.sidebar.header("Filters")
    fail_types = sorted({c["fm"]["fail_type"] for c in cases})
    selected_types = st.sidebar.multiselect(
        "FM fail type", fail_types, default=fail_types
    )
    filtered = [c for c in cases if c["fm"]["fail_type"] in selected_types]
    st.sidebar.write(f"Showing {len(filtered)} / {len(cases)} cases")
    st.sidebar.write(f"Mode: **{index.get('mode', '?')}** (other_agent_forecast)")
    st.sidebar.write(f"Mix: {index.get('fm_fail_type_mix', {})}")

    labels = [
        f"case_{c['case_id']:02d} | {c['tfrecord'][-20:]} #{c['scenario_idx']} | FM:{c['fm']['fail_type']}"
        for c in filtered
    ]
    if not labels:
        st.warning("No cases match the filter.")
        return

    choice = st.selectbox("Case", labels, index=0)
    case = filtered[labels.index(choice)]
    case_dir = data_dir / f"case_{case['case_id']:02d}"

    c1, c2, c3 = st.columns(3)
    c1.metric("Diffusion-ES", "SUCCESS" if case["diff"]["success"] else "FAIL")
    c2.metric("FM-ES", f"FAIL ({case['fm']['fail_type']})")
    c3.metric("Forecast mode", case.get("mode", index.get("mode", "?")))

    st.subheader("Side-by-side closed-loop video (left Diff-ES, right FM-ES)")
    st.caption(
        "Overlay colors: **magenta** = non-elite pop (~64), "
        "**orange** = other elites (7), **green** = best/selected elite. Labels black."
    )
    sbs = case_dir / "side_by_side.mp4"
    if sbs.exists():
        st.video(str(sbs))
    else:
        st.warning(
            f"Missing `{sbs}`. Re-run GPU export:\n"
            "`sbatch slurm_export_diff_ok_fm_fail_viewer.sh`"
        )

    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**Diffusion-ES**")
        dp = case_dir / "diff.mp4"
        if dp.exists():
            st.video(str(dp))
        else:
            st.warning(f"Missing `{dp.name}`")
    with v2:
        st.markdown("**FM-ES**")
        fp = case_dir / "fm.mp4"
        if fp.exists():
            st.video(str(fp))
        else:
            st.warning(f"Missing `{fp.name}`")

    diag = load_diagnostics(case_dir)
    if diag:
        st.subheader("Diagnostics for FM hyperparam tuning")
        summary = diag.get("summary", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("max |FM−Diff| lateral (m)", f"{summary.get('max_lateral_fm_vs_diff_m', float('nan')):.2f}")
        m2.metric("mean FM speed (m/s)", f"{summary.get('mean_speed_fm_mps', float('nan')):.2f}")
        m3.metric("mean Diff speed (m/s)", f"{summary.get('mean_speed_diff_mps', float('nan')):.2f}")
        m4.metric("final goal dist FM (m)", f"{summary.get('final_goal_dist_fm_m', float('nan')):.2f}")

        series = diag.get("series", {})
        t = series.get("t", [])
        chart_rows = {
            "lateral_err_fm_vs_diff_m": series.get("lateral_err_fm_vs_diff_m", []),
            "goal_dist_fm_m": series.get("goal_dist_fm_m", []),
            "goal_dist_diff_m": series.get("goal_dist_diff_m", []),
            "fm_speed_mps": series.get("fm_speed_mps", []),
            "diff_speed_mps": series.get("diff_speed_mps", []),
        }
        # Streamlit line chart wants a dataframe-like dict keyed by column.
        if t and chart_rows["lateral_err_fm_vs_diff_m"]:
            st.line_chart(
                {
                    "lateral_err_m": chart_rows["lateral_err_fm_vs_diff_m"],
                    "goal_dist_fm_m": chart_rows["goal_dist_fm_m"],
                    "goal_dist_diff_m": chart_rows["goal_dist_diff_m"],
                }
            )
            st.line_chart(
                {
                    "fm_speed_mps": chart_rows["fm_speed_mps"],
                    "diff_speed_mps": chart_rows["diff_speed_mps"],
                }
            )

        with st.expander("Full diagnostics.json"):
            st.json(diag)
        st.caption(
            diag.get("hyperparams", {}).get(
                "note_on_noising",
                "Larger FM resample_t_start ⇒ more noising.",
            )
        )
    else:
        st.warning(f"Missing `{case_dir / 'diagnostics.json'}`")

    m1, m2 = st.columns(2)
    with m1:
        st.subheader("Diffusion-ES metrics (from fail1500 run)")
        st.json(
            {
                "success": case["diff"]["success"],
                "overlap": case["diff"]["overlap"],
                "offroad": case["diff"]["offroad"],
                "goal_reached": case["diff"]["goal_reached"],
                "min_goal_distance_m": case["diff"]["min_goal_distance_m"],
            }
        )
    with m2:
        st.subheader("FM-ES metrics (from fail1500 run)")
        st.json(
            {
                "success": case["fm"]["success"],
                "overlap": case["fm"]["overlap"],
                "offroad": case["fm"]["offroad"],
                "goal_reached": case["fm"]["goal_reached"],
                "min_goal_distance_m": case["fm"]["min_goal_distance_m"],
                "fail_type": case["fm"]["fail_type"],
            }
        )

    st.subheader("IDs")
    st.code(
        f"tfrecord: {case['tfrecord']}\n"
        f"scenario_idx: {case['scenario_idx']}\n"
        f"case_dir: {case_dir}\n"
        f"files: diff.mp4, fm.mp4, side_by_side.mp4, diagnostics.json, trajectories.npz"
    )


if __name__ == "__main__":
    main()
