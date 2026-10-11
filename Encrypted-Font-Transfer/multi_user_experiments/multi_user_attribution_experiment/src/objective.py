from __future__ import annotations

import torch
import torch.nn.functional as F


def _binary_term(values: torch.Tensor, positive: bool) -> torch.Tensor:
    if not values.numel():
        return values.sum()
    targets = torch.ones_like(values) if positive else torch.zeros_like(values)
    return F.binary_cross_entropy_with_logits(values, targets)


def attribution_loss(logits: torch.Tensor, labels: torch.Tensor, user_count: int):
    if logits.ndim != 2 or logits.shape[1] != user_count or labels.shape != logits.shape[:1]:
        raise ValueError("logits and labels have incompatible shapes")
    known = labels < user_count
    classification = (
        F.cross_entropy(logits[known], labels[known])
        if bool(known.any())
        else logits.sum() * 0.0
    )
    detector_terms = []
    for user_index in range(user_count):
        own = labels == user_index
        other = known & ~own
        real = labels == user_count
        wrong_key = labels > user_count
        detector_terms.append(
            _binary_term(logits[own, user_index], True)
            + _binary_term(logits[other, user_index], False)
            + 0.5 * _binary_term(logits[real, user_index], False)
            + 0.5 * _binary_term(logits[wrong_key, user_index], False)
        )
    detection = torch.stack(detector_terms).mean()
    total = classification + detection
    return total, {"classification": classification.detach(), "detection": detection.detach()}
