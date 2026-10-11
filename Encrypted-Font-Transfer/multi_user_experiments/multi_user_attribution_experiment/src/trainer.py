from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .config import fingerprint, write_json
from .data import batch_for_step, load_tensor_split
from .metrics import calibrate_threshold, summarize
from .model import build_model
from .objective import attribution_loss


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _random_state() -> Dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_random_state(state: Dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_checkpoint(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(path: Path) -> Dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _probabilities(model, split, device, batch_size: int) -> torch.Tensor:
    prior_training = model.training
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, split["labels"].numel(), batch_size):
            images = split["images"][start : start + batch_size].to(device)
            rows.append(torch.sigmoid(model(images)).cpu())
    model.train(prior_training)
    return torch.cat(rows)


def _validate(model, split, config, device) -> Tuple[float, Dict]:
    probabilities = _probabilities(
        model, split, device, int(config["training"]["batch_size"])
    )
    return calibrate_threshold(
        probabilities,
        split["labels"],
        config["user_ids"],
        float(config["decision"]["maximum_validation_unknown_fpr"]),
    )


def _rank(report: Dict) -> Tuple[float, float, float]:
    return (
        report["watermarked_top1_accuracy"],
        report["closed_set_top1_accuracy"],
        -report["unknown_false_attribution_rate"],
    )


def _check_evaluation_split(split, user_count: int, name: str) -> None:
    labels = split["labels"]
    if not bool((labels < user_count).any()) or not bool((labels >= user_count).any()):
        raise ValueError(f"{name} must contain registered-user and unknown samples")


def train(config: Dict) -> None:
    user_count = len(config["user_ids"])
    training = config["training"]
    seed = int(training["seed"])
    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(config["paths"]["output"])
    output.mkdir(parents=True, exist_ok=True)

    train_split = load_tensor_split(Path(config["paths"]["train"]), user_count, training=True)
    validation_split = load_tensor_split(Path(config["paths"]["validation"]), user_count)
    _check_evaluation_split(validation_split, user_count, "validation split")

    model = build_model(config["model"], user_count).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    config_fingerprint = fingerprint(config)
    last_path = output / "last.pt"
    best_path = output / "best.pt"
    start_step = 0

    if last_path.exists():
        saved = _load_checkpoint(last_path)
        if saved["config_fingerprint"] != config_fingerprint:
            raise RuntimeError("the saved checkpoint belongs to a different configuration")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        _restore_random_state(saved["random_state"])
        start_step = int(saved["step"])
        best_rank = tuple(saved["best_rank"])
        best_step = int(saved["best_step"])
    else:
        threshold, initial_report = _validate(model, validation_split, config, device)
        best_rank = _rank(initial_report)
        best_step = 0
        _save_checkpoint(
            best_path,
            {
                "step": 0,
                "model": model.state_dict(),
                "threshold": threshold,
                "validation": initial_report,
                "config_fingerprint": config_fingerprint,
            },
        )

    total_steps = int(training["steps"])
    interval = int(training["validation_interval"])
    progress = tqdm(
        total=total_steps,
        initial=start_step,
        desc="training",
        unit="step",
        dynamic_ncols=True,
    )
    try:
        for step in range(start_step + 1, total_steps + 1):
            images, labels = batch_for_step(
                train_split, int(training["batch_size"]), seed, step
            )
            images, labels = images.to(device), labels.to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, parts = attribution_loss(model(images), labels, user_count)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError("non-finite training update")
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}", refresh=False)
            progress.update(1)

            if step % interval == 0 or step == total_steps:
                threshold, report = _validate(model, validation_split, config, device)
                rank = _rank(report)
                if rank > best_rank:
                    best_rank, best_step = rank, step
                    _save_checkpoint(
                        best_path,
                        {
                            "step": step,
                            "model": model.state_dict(),
                            "threshold": threshold,
                            "validation": report,
                            "config_fingerprint": config_fingerprint,
                        },
                    )
                _save_checkpoint(
                    last_path,
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "random_state": _random_state(),
                        "best_rank": best_rank,
                        "best_step": best_step,
                        "config_fingerprint": config_fingerprint,
                    },
                )
    finally:
        progress.close()

    best = _load_checkpoint(best_path)
    model.load_state_dict(best["model"])
    rule = {
        "selected_only_on_validation": True,
        "threshold": float(best["threshold"]),
        "best_step": int(best["step"]),
        "validation": best["validation"],
    }
    write_json(output / "validation_rule.json", rule)

    test_split = load_tensor_split(Path(config["paths"]["test"]), user_count)
    _check_evaluation_split(test_split, user_count, "test split")
    test_probabilities = _probabilities(
        model, test_split, device, int(training["batch_size"])
    )
    summary = {
        "experiment_name": config["experiment_name"],
        "user_count": user_count,
        "test_count": int(test_split["labels"].numel()),
        "threshold_selected_only_on_validation": True,
        "best_step": int(best["step"]),
        "test": summarize(
            test_probabilities,
            test_split["labels"],
            float(best["threshold"]),
            config["user_ids"],
        ),
    }
    write_json(output / "summary.json", summary)


def evaluate(config: Dict) -> Dict:
    user_count = len(config["user_ids"])
    output = Path(config["paths"]["output"])
    best = _load_checkpoint(output / "best.pt")
    if best["config_fingerprint"] != fingerprint(config):
        raise RuntimeError("the saved checkpoint belongs to a different configuration")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"], user_count).to(device)
    model.load_state_dict(best["model"])
    test_split = load_tensor_split(Path(config["paths"]["test"]), user_count)
    _check_evaluation_split(test_split, user_count, "test split")
    probabilities = _probabilities(
        model, test_split, device, int(config["training"]["batch_size"])
    )
    return summarize(probabilities, test_split["labels"], best["threshold"], config["user_ids"])
