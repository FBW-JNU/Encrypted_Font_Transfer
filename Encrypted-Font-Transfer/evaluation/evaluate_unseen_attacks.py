#!/usr/bin/env python3
"""Evaluate Gamma, ordered combined attacks, and decision-based HopSkipJump."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.common import evaluate_runs, load_config
from evaluation.methods import extract_ber
from evaluation.runner import evaluate_ber


class QueryBudgetExceeded(RuntimeError):
    pass


def hopskipjump(method, clean, config, sample, condition):
    before = extract_ber(method, clean)
    budget = int(config.get("hopskipjump", {}).get("max_queries", 156))
    if budget != 156:
        raise ValueError("this protocol requires exactly 156 total decision queries")
    import torch
    count = 0

    def decision(batch):
        nonlocal count
        batch = np.asarray(batch, dtype=np.float32)
        if count + len(batch) > budget:
            raise QueryBudgetExceeded(f"156-query budget exhausted at {count}")
        values = []
        for image in batch:
            accepted = bool(method.verify(torch.from_numpy(image).unsqueeze(0).to(clean.device)))
            count += 1
            values.append([0.0, 1.0] if accepted else [1.0, 0.0])
        return np.asarray(values, dtype=np.float32)

    try:
        original = clean.detach().cpu().numpy().astype(np.float32)
        initial_decision = int(decision(original).argmax(1)[0])
    except (NotImplementedError, AttributeError) as exc:
        return {"status": "N/A", "reason": f"binary verifier unavailable: {exc}",
                "ber_before": before["ber"], "ber_after": None, "queries": count}
    try:
        from art.attacks.evasion import HopSkipJump
        from art.estimators.classification import BlackBoxClassifier
    except ImportError:
        return {"status": "N/A", "reason": "adversarial-robustness-toolbox is not installed",
                "ber_before": before["ber"], "ber_after": None, "queries": count}
    estimator = BlackBoxClassifier(predict_fn=decision, input_shape=(1, 32, 192),
                                   nb_classes=2, clip_values=(-1.0, 1.0))
    attack = HopSkipJump(classifier=estimator, targeted=False,
                         max_iter=int(config.get("hopskipjump", {}).get("max_iter", 10)),
                         max_eval=budget, init_eval=10, verbose=False)
    try:
        candidate = attack.generate(x=original, y=np.eye(2, dtype=np.float32)[[initial_decision]])
        final_decision = int(decision(candidate).argmax(1)[0])
        attacked = torch.from_numpy(candidate).to(clean.device)
        after = extract_ber(method, attacked)
        delta = candidate - original
        return {"status": "ok", "ber": after["ber"], "error_bits": after["error_bits"],
                "total_bits": 128, "ber_before": before["ber"], "ber_after": after["ber"],
                "queries": count, "decision_before": initial_decision,
                "decision_after": final_decision, "decision_changed": initial_decision != final_decision,
                "l2": float(np.linalg.norm(delta)), "linf": float(np.max(np.abs(delta)))}
    except QueryBudgetExceeded as exc:
        return {"status": "failed", "reason": str(exc), "ber_before": before["ber"],
                "ber_after": None, "queries": count, "decision_before": initial_decision}
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                "ber_before": before["ber"], "ber_after": None,
                "queries": count, "decision_before": initial_decision}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    configured_attacks = config.get("training_attacks")
    if config.get("training_config"):
        training = json.loads(Path(config["training_config"]).read_text(encoding="utf-8"))
        configured_attacks = training.get("attacks", training.get("noise_layers"))
    if isinstance(configured_attacks, (list, tuple)):
        listed = {str(value).lower() for value in configured_attacks}
        config["training_attack_exposure"] = {
            name: ("listed" if name in listed else "not_listed; unseen_status_unverified")
            for name in ("gamma", "jpeg", "resize", "rotation", "hopskipjump")
        }
    else:
        config["training_attack_exposure"] = "not_verifiable"
    conditions = [("clean", None), ("gamma", 1.4), ("combined", None), ("hopskipjump", None)]
    evaluate_runs(config, lambda run: evaluate_ber(run, conditions, hsj=hopskipjump))


if __name__ == "__main__":
    main()
