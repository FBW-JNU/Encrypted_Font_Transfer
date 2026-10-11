import unittest

import torch

from src.metrics import calibrate_threshold, summarize


class MetricsTest(unittest.TestCase):
    def test_validation_threshold_respects_unknown_limit(self):
        probabilities = torch.tensor(
            [
                [0.90, 0.10],
                [0.20, 0.85],
                [0.55, 0.20],
                [0.10, 0.15],
            ]
        )
        labels = torch.tensor([0, 1, 2, 2])
        threshold, report = calibrate_threshold(
            probabilities, labels, ["alice", "bob"], maximum_unknown_fpr=0.0
        )
        self.assertGreater(threshold, 0.55)
        self.assertEqual(report["watermarked_top1_accuracy"], 1.0)
        self.assertEqual(report["unknown_false_attribution_rate"], 0.0)

    def test_summary_uses_one_unknown_class(self):
        probabilities = torch.tensor([[0.8, 0.1], [0.2, 0.7], [0.3, 0.4]])
        labels = torch.tensor([0, 1, 3])
        report = summarize(probabilities, labels, 0.5, ["alice", "bob"])
        self.assertEqual(report["overall_accuracy"], 1.0)
        self.assertEqual(report["confusion_matrix_true_rows_predicted_columns"][2][2], 1)


if __name__ == "__main__":
    unittest.main()

