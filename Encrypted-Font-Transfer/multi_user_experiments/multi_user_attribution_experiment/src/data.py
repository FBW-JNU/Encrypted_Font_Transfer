from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_tensor_split(path: Path, user_count: int, training: bool = False) -> Dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _load(path)
    if not isinstance(payload, dict) or set(payload) < {"images", "labels"}:
        raise ValueError(f"{path} must contain images and labels")
    images = torch.as_tensor(payload["images"], dtype=torch.float32)
    labels = torch.as_tensor(payload["labels"], dtype=torch.long)
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError(f"images must have shape [N,1,H,W], got {tuple(images.shape)}")
    if labels.ndim != 1 or labels.shape[0] != images.shape[0] or not labels.numel():
        raise ValueError("labels must be a non-empty vector aligned with images")
    maximum = user_count + int(training)
    if int(labels.min()) < 0 or int(labels.max()) > maximum:
        raise ValueError(f"labels must be in [0,{maximum}]")
    return {"images": images.contiguous(), "labels": labels.contiguous()}


def batch_for_step(split: Dict[str, torch.Tensor], batch_size: int, seed: int, step: int):
    count = split["labels"].numel()
    generator = torch.Generator().manual_seed(int(seed) + int(step))
    indices = torch.randint(count, (batch_size,), generator=generator)
    return split["images"][indices], split["labels"][indices]

