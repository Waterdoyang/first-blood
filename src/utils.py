from __future__ import annotations

import json
import os
import random
from datetime import datetime

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_output_dir(base: str = "outputs") -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(base, ts)
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    return out


def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
