from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"experiment_name", "user_ids", "paths", "model", "training", "decision"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"configuration is missing: {', '.join(missing)}")
    if len(config["user_ids"]) < 2 or len(set(config["user_ids"])) != len(config["user_ids"]):
        raise ValueError("user_ids must contain at least two unique names")

    base = path.parent
    for name in ("train", "validation", "test", "output"):
        if name not in config["paths"]:
            raise ValueError(f"paths.{name} is required")
        value = Path(config["paths"][name]).expanduser()
        config["paths"][name] = str(value if value.is_absolute() else (base / value).resolve())

    training = config["training"]
    for name in ("steps", "batch_size", "validation_interval"):
        if int(training[name]) <= 0:
            raise ValueError(f"training.{name} must be positive")
    maximum_fpr = float(config["decision"]["maximum_validation_unknown_fpr"])
    if not 0.0 <= maximum_fpr <= 1.0:
        raise ValueError("maximum_validation_unknown_fpr must be in [0, 1]")
    return config


def fingerprint(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

