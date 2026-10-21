"""Episode goals of the Diffusion Planner training samples, for goal mode (goal.py).

  python diffusion_planner_womd/goals_womd.py --tfrecord .../training_tfexample.tfrecord-00000-of-01000 \
      --data_root <data_root>

reads <data_root>/lists/<shard>.json (from convert_womd.py) and writes <data_root>/goals/<shard>.json,
{sample: [x, y, seconds to goal]} with goal.episode_goal at the sample's step.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_planner_womd.convert_womd import _states  # noqa: E402
from diffusion_planner_womd.features import WorldLog  # noqa: E402
from diffusion_planner_womd.goal import episode_goal  # noqa: E402

SAMPLE_RE = re.compile(r"/(\d+)_t(\d+)\.npz$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--max_num_objects", type=int, default=128)  # as convert_womd.py
    args = parser.parse_args()

    shard = os.path.basename(args.tfrecord)
    with open(Path(args.data_root) / "lists" / f"{shard}.json") as f:
        names = json.load(f)
    wanted = defaultdict(list)
    for name in names:
        i, t = SAMPLE_RE.search(name).groups()
        wanted[int(i)].append((name, int(t)))

    goals = {}
    last = max(wanted) if wanted else -1
    for indices, state in _states(args.tfrecord, args.max_num_objects):
        for w, i in enumerate(indices):
            if i in wanted:
                log = WorldLog.from_state(state, w)
                for name, t in wanted[i]:
                    goals[name] = episode_goal(log, t).tolist()
        if indices[-1] >= last:
            break
    if len(goals) != len(names):
        raise RuntimeError(f"{shard}: goals for {len(goals)} of {len(names)} samples")
    out = Path(args.data_root) / "goals"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{shard}.json", "w") as f:
        json.dump(goals, f)
    print(f"{shard}: {len(goals)} goals")


if __name__ == "__main__":
    main()
