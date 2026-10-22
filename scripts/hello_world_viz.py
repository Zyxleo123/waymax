"""Hello-world smoke test for the waymax_rs dev environment (CPU).

Builds a synthetic Waymax SimulatorState (no WOMD dataset / no GPU required),
runs a Waymax simulation metric on it, and renders it with this repo's own
`viz.plot_simulator_state` visualization code. Saves a PNG to prove the
end-to-end sim + scoring + visualization pipeline works.
"""

import os
import sys

import numpy as np
from PIL import Image

from waymax import metrics as waymax_metrics
from waymax.utils import test_utils

import viz


def main(out_path: str) -> None:
    # 1. Synthetic SimulatorState (overlapping + offroad SDC), bundled with waymax.
    state = test_utils.simulator_state_with_overlap()
    print("Built synthetic SimulatorState:", type(state).__name__,
          "num_objects=", state.sim_trajectory.num_objects)

    # The repo's visualizer expects traffic lights with a time axis
    # (num_tl, num_steps); the synthetic helper ships a 1-D tensor, so add one.
    tl = state.log_traffic_light
    tl_2d = tl.replace(**{f: np.asarray(getattr(tl, f)).reshape(1, 1)
                          for f in ("x", "y", "z", "state", "lane_ids", "valid")})
    state = state.replace(log_traffic_light=tl_2d)

    # 2. Run real Waymax simulation metrics (scoring) on the state, on CPU.
    overlap = waymax_metrics.OverlapMetric().compute(state)
    offroad = waymax_metrics.OffroadMetric().compute(state)
    print("Overlap metric value:", np.asarray(overlap.value).tolist())
    print("Offroad metric value:", np.asarray(offroad.value).tolist())

    # 3. Render the scene using THIS repo's visualization code.
    img = viz.plot_simulator_state(state, use_log_traj=True)
    print("Rendered image shape:", img.shape)

    Image.fromarray(np.asarray(img)).save(out_path)
    print("Saved visualization to:", out_path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "hello_world_viz.png"
    main(out)
    print("HELLO_WORLD_OK")
