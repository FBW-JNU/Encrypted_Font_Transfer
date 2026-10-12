#!/usr/bin/env python3
"""Measure 128-bit watermark BER under common image attacks."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.attacks import standard_conditions
from evaluation.common import evaluate_runs, load_config
from evaluation.runner import evaluate_ber


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    evaluate_runs(config, lambda run: evaluate_ber(run, standard_conditions(run)))


if __name__ == "__main__":
    main()
