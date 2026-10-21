"""Merges the per-shard data lists written by convert_womd.py into the single list train_predictor.py reads.

  python diffusion_planner_womd/merge_lists.py <data_root>   ->   <data_root>/train_list.json
"""

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = []
for path in sorted((root / "lists").glob("*.json")):
    files += json.loads(path.read_text())
(root / "train_list.json").write_text(json.dumps(files))
print(f"{len(files)} samples from {len(list((root / 'lists').glob('*.json')))} shards -> {root / 'train_list.json'}")
