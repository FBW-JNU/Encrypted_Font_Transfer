from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch


def _predictions(probabilities: torch.Tensor, threshold: float) -> torch.Tensor:
    scores, users = probabilities.max(dim=1)
    unknown = torch.full_like(users, probabilities.shape[1])
    return torch.where(scores >= float(threshold), users, unknown)


def summarize(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
    user_ids: Iterable[str],
) -> Dict:
    user_ids = list(user_ids)
    user_count = len(user_ids)
    truth = torch.where(labels < user_count, labels, torch.full_like(labels, user_count))
    predicted = _predictions(probabilities, threshold)
    matrix = torch.zeros((user_count + 1, user_count + 1), dtype=torch.long)
    for expected, actual in zip(truth.tolist(), predicted.tolist()):
        matrix[expected, actual] += 1
    known = truth < user_count
    unknown = ~known
    closed = probabilities.argmax(dim=1)
    per_user = {}
    for index, name in enumerate(user_ids):
        mask = truth == index
        per_user[name] = float((predicted[mask] == index).float().mean()) if bool(mask.any()) else 0.0
    return {
        "threshold": float(threshold),
        "watermarked_top1_accuracy": float((predicted[known] == truth[known]).float().mean()),
        "closed_set_top1_accuracy": float((closed[known] == truth[known]).float().mean()),
        "unknown_recall": float((predicted[unknown] == user_count).float().mean()),
        "unknown_false_attribution_rate": float((predicted[unknown] != user_count).float().mean()),
        "overall_accuracy": float((predicted == truth).float().mean()),
        "per_user_accuracy": per_user,
        "labels": user_ids + ["unknown"],
        "confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
    }


def calibrate_threshold(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    user_ids: Iterable[str],
    maximum_unknown_fpr: float,
) -> Tuple[float, Dict]:
    user_ids = list(user_ids)
    user_count = len(user_ids)
    maxima = probabilities.max(dim=1).values
    candidates: List[float] = sorted(set([0.0, 1.000001] + maxima.tolist()))
    eligible = []
    for threshold in candidates:
        report = summarize(probabilities, labels, threshold, user_ids)
        if report["unknown_false_attribution_rate"] <= float(maximum_unknown_fpr) + 1e-12:
            rank = (
                report["watermarked_top1_accuracy"],
                report["closed_set_top1_accuracy"],
                -report["unknown_false_attribution_rate"],
                threshold,
            )
            eligible.append((rank, threshold, report))
    if not eligible:
        raise RuntimeError("no validation threshold satisfies the unknown false-attribution limit")
    _, threshold, report = max(eligible, key=lambda row: row[0])
    return threshold, report

