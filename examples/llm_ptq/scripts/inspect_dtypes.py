"""Inspect actual tensor dtypes in a HF model directory."""
import json
import sys
from collections import Counter
from pathlib import Path

import safetensors

model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/disk1/guohaoran/models/Qwen3.5-0.8B")
cfg = json.loads((model_dir / "config.json").read_text())
print("top dtype:", cfg.get("torch_dtype") or cfg.get("dtype"))
for sub in ("text_config", "vision_config"):
    s = cfg.get(sub, {})
    print(sub, "dtype:", s.get("torch_dtype") or s.get("dtype"))

safes = sorted(model_dir.glob("*.safetensors")) + sorted(model_dir.glob("*.safetensors-*"))
dtypes: Counter = Counter()
sample: dict[str, str] = {}
for p in safes:
    with safetensors.safe_open(str(p), "pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            dtypes[str(t.dtype)] += t.numel()
            sample.setdefault(str(t.dtype), f"{p.name}::{k}")

total = sum(dtypes.values()) or 1
for d, n in dtypes.most_common():
    print(f"{d}: {n / 1e6:.2f}M params ({100 * n / total:.1f}%)  e.g. {sample[d]}")
