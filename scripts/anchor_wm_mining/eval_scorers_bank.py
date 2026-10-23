"""Score several AnchorWM checkpoints on one ES bank's val split with finetune.validate (name=ckpt args)."""
import json
import sys
from types import SimpleNamespace

import torch

from anchor_wm.bank_data import build_index
from anchor_wm.eval_bank import load_model
from anchor_wm.finetune import validate

bank = sys.argv[1]
dev = torch.device("cuda")
val_index, counts = build_index(f"{bank}/es_candidates", f"{bank}/anchor_cache", f"{bank}/scenario_split_seed0.json", "", ("val",))
print("val index", counts, flush=True)
args = SimpleNamespace(cache_dir=f"{bank}/anchor_cache", val_groups=10 ** 9)
for arg in sys.argv[2:]:
    name, path = arg.split("=", 1)
    model = load_model(path, dev)
    m = validate(model, val_index, args, dev, torch.cuda.is_bf16_supported())
    print(json.dumps({"name": name, **{k: round(v, 4) for k, v in sorted(m.items())}}), flush=True)
