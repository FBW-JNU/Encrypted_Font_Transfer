#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.trainer import evaluate, train


def command_train(args) -> None:
    train(load_config(args.config))


def command_evaluate(args) -> None:
    report = evaluate(load_config(args.config))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_results(args) -> None:
    config = load_config(args.config)
    path = Path(config["paths"]["output"]) / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    print(path.read_text(encoding="utf-8"), end="")


def main() -> None:
    parser = argparse.ArgumentParser(description="Open-set multi-user attribution experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text, function in (
        ("train", "train and evaluate on the test split", command_train),
        ("evaluate", "evaluate with the frozen validation threshold", command_evaluate),
        ("results", "display the saved metrics summary", command_results),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, required=True)
        command.set_defaults(function=function)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
