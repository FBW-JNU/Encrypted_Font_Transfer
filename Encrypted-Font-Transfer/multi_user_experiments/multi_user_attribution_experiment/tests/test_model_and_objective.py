import unittest

import torch

from src.model import MultiUserAttributionNetwork
from src.objective import attribution_loss


class ModelAndObjectiveTest(unittest.TestCase):
    def test_forward_and_backward(self):
        model = MultiUserAttributionNetwork(
            user_count=3, channels=[8, 16], embedding_dim=24, dropout=0.0
        )
        images = torch.randn(8, 1, 32, 64)
        labels = torch.tensor([0, 1, 2, 0, 3, 3, 4, 4])
        logits = model(images)
        loss, parts = attribution_loss(logits, labels, user_count=3)
        loss.backward()
        self.assertEqual(tuple(logits.shape), (8, 3))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {"classification", "detection"})

    def test_unknown_only_batch_remains_finite(self):
        logits = torch.randn(4, 2, requires_grad=True)
        labels = torch.tensor([2, 2, 3, 3])
        loss, _ = attribution_loss(logits, labels, user_count=2)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()

