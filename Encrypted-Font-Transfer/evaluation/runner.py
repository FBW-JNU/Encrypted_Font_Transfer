"""Shared fixed-sample BER evaluation loop and reporting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .attacks import apply_attack
from .common import (append_row, completed_rows, fixed_samples, load_dataset,
                     load_key, read_png, sample_tensors, save_png, start_output,
                     write_reports)
from .methods import METHODS, available_methods, extract_ber


def condition_name(name, value):
    return name if value is None else f"{name}_{value}"


def attack_seed(base_seed, sample_id, condition):
    value = f"{base_seed}:{sample_id}:{condition}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**31)


def evaluate_ber(config, conditions, hsj=None):
    output = start_output(config)
    data = load_dataset(config["dataset"])
    samples = fixed_samples(data, config, output)
    key = load_key(config["key"])
    methods, unavailable = available_methods(config, key)
    rows_path = output / "per_image.jsonl"
    rows = completed_rows(rows_path)
    selected = ["InstanceWmk" if x == "InstancWmk" else x
                for x in config.get("methods", list(METHODS))]
    for name in selected:
        method = methods.get(name)
        if method is None:
            for attack, value in conditions:
                condition = condition_name(attack, value)
                row_id = f"{name}:{condition}:N/A"
                if row_id not in rows:
                    row = {"row_id": row_id, "method": name, "condition": condition,
                           "status": "N/A", "reason": unavailable[name], "evaluated": 0}
                    append_row(rows_path, row)
                    rows[row_id] = row
            continue
        for index, sample in enumerate(samples):
            clean_path = output / "images" / name / f"{sample['sample_id']}.png"
            try:
                if clean_path.exists():
                    clean = read_png(clean_path).unsqueeze(0).to(config["device"])
                else:
                    references, _real = sample_tensors(data, sample)
                    clean = method.embed(references, sample["text"], config["seed"] + index)
                    if clean.shape != (1, 1, 32, 192):
                        raise ValueError(f"embed returned shape {tuple(clean.shape)}, expected (1,1,32,192)")
                    clean = clean.to(config["device"])
                    save_png(clean, clean_path)
                    clean = read_png(clean_path).unsqueeze(0).to(config["device"])
                embed_failure = None
            except Exception as exc:
                clean = None
                embed_failure = f"{type(exc).__name__}: {exc}"
            for attack, value in conditions:
                condition = condition_name(attack, value)
                row_id = f"{name}:{condition}:{sample['sample_id']}"
                if row_id in rows and rows[row_id].get("status") in ("ok", "N/A"):
                    continue
                row = {"row_id": row_id, "method": name, "sample_id": sample["sample_id"],
                       "condition": condition, "attack": attack, "parameter": value,
                       "total_bits": 128}
                if attack == "hopskipjump":
                    row.update(queries=None, ber_before=None, ber_after=None,
                               decision_changed=None, l2=None, linf=None)
                if embed_failure:
                    row.update(status="failed", reason=f"embed: {embed_failure}")
                else:
                    try:
                        if attack == "hopskipjump":
                            result = hsj(method, clean, config, sample, condition)
                            row.update(result)
                        else:
                            seed = attack_seed(config["seed"], sample["sample_id"], condition)
                            attacked = apply_attack(clean, attack, value, seed,
                                                    config.get("attack_settings"))
                            row.update(extract_ber(method, attacked))
                            row["status"] = "ok"
                    except Exception as exc:
                        row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                append_row(rows_path, row)
                rows[row_id] = row
        del method
    summary = write_reports(output, rows)
    if hsj:
        for item in summary:
            if item["condition"] != "hopskipjump":
                continue
            subset = [row for row in rows.values() if row.get("method") == item["method"]
                      and row.get("condition") == "hopskipjump" and row.get("status") == "ok"]
            item["attack_success_rate"] = (sum(bool(row.get("decision_changed")) for row in subset) / len(subset)
                                           if subset else None)
            item["mean_queries"] = (sum(row["queries"] for row in subset) / len(subset)
                                    if subset else None)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        import csv
        with (output / "summary.csv").open("w", newline="") as stream:
            columns = list(dict.fromkeys(key for item in summary for key in item))
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(summary)
    return summary
