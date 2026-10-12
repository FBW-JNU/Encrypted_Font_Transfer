#!/usr/bin/env python3
"""Evaluate generation quality as the fraction of correct key bits changes."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.common import (completed_rows, fixed_samples, generate, load_config,
                               load_dataset, load_key, load_model, read_png,
                               sample_tensors, save_png, start_output,
                               training_writer_order, evaluate_runs)
from models.inception import calculate_frechet_distance


RATIOS = (0, 0.25, 0.5, 0.75, 1)


def controlled_key(key, ratio, seed):
    count = round(512 * ratio)
    generator = torch.Generator().manual_seed(seed)
    positions = torch.randperm(512, generator=generator)
    changed = 1 - key.clone()
    changed[positions[:count]] = key[positions[:count]]
    actual = float((changed == key).float().mean())
    assert actual == ratio
    return changed, actual


@torch.no_grad()
def quality_metrics(model, image, writer_index, text, device):
    style = (float(model.netW(image, torch.tensor([writer_index], device=device)).item())
             if writer_index is not None else None)
    target, lengths = model.netconverter.encode([text.encode("utf-8")])
    prediction = model.netOCR(image)
    pred_lengths = torch.IntTensor([prediction.shape[0]])
    loss = model.OCR_criterion(prediction, target, pred_lengths, lengths)
    ocr = float(loss.mean().item())
    if (style is not None and not math.isfinite(style)) or not math.isfinite(ocr):
        raise ValueError("nonfinite quality metric")
    return style, ocr


@torch.no_grad()
def features(model, images, batch_size):
    result = []
    batch = []
    for image in images:
        batch.append(image)
        if len(batch) == batch_size:
            unit_rgb = torch.cat(batch).clamp(-1, 1).add(1).div(2).repeat(1, 3, 1, 1)
            result.extend(model.inception(unit_rgb)[0].flatten(1).cpu().numpy())
            batch = []
    if batch:
        unit_rgb = torch.cat(batch).clamp(-1, 1).add(1).div(2).repeat(1, 3, 1, 1)
        result.extend(model.inception(unit_rgb)[0].flatten(1).cpu().numpy())
    return np.stack(result)


def fid(model, generated_paths, real_images, device, batch_size):
    generated = (read_png(path).unsqueeze(0).to(device) for path in generated_paths)
    real = (image.unsqueeze(0).to(device) for image in real_images)
    fake_features = features(model, generated, batch_size)
    real_features = features(model, real, batch_size)
    if len(fake_features) < 2:
        return None
    if len(fake_features) > fake_features.shape[1] and len(real_features) > real_features.shape[1]:
        return float(calculate_frechet_distance(
            fake_features.mean(0), np.cov(fake_features, rowvar=False),
            real_features.mean(0), np.cov(real_features, rowvar=False)))
    x = fake_features - fake_features.mean(0)
    y = real_features - real_features.mean(0)
    trace_x = float((x * x).sum() / (len(x) - 1))
    trace_y = float((y * y).sum() / (len(y) - 1))
    cross = x @ y.T / math.sqrt((len(x) - 1) * (len(y) - 1))
    score = float(np.square(fake_features.mean(0) - real_features.mean(0)).sum()
                  + trace_x + trace_y - 2 * np.linalg.svd(cross, compute_uv=False).sum())
    return max(score, 0.0)


def evaluate(config):
    output = start_output(config)
    data = load_dataset(config["dataset"])
    samples = fixed_samples(data, config, output)
    key = load_key(config["key"])
    model = load_model(config)
    device = config["device"]
    writer_order = training_writer_order(config["dataset"])
    rows_path = output / "per_image.jsonl"
    rows = completed_rows(rows_path)
    real_images = [sample_tensors(data, sample)[1] for sample in samples]
    for ratio in RATIOS:
        condition = f"correct_{int(ratio * 100):03d}"
        for index, sample in enumerate(samples):
            row_id = f"{condition}:{sample['sample_id']}"
            if row_id in rows and rows[row_id].get("status") == "ok":
                continue
            row = {"row_id": row_id, "sample_id": sample["sample_id"],
                   "condition": condition, "correct_fraction_requested": ratio}
            image_path = output / "images" / condition / f"{sample['sample_id']}.png"
            try:
                changed, actual = controlled_key(key, ratio, config["seed"] + index)
                row["correct_fraction_actual"] = actual
                refs, _real = sample_tensors(data, sample)
                if image_path.exists():
                    image = read_png(image_path).unsqueeze(0).to(device)
                else:
                    image = generate(model, refs, sample["text"], changed,
                                     device, config["seed"] + index)
                    save_png(image, image_path)
                    image = read_png(image_path).unsqueeze(0).to(device)
                row["l_cls"], row["l_ocr"] = quality_metrics(
                    model, image, writer_order.get(sample["writer"]), sample["text"], device)
                row["l_cls_status"] = ("ok" if row["l_cls"] is not None else
                                       "N/A: test writer is absent from training classifier labels")
                row["image"] = str(image_path.relative_to(output))
                row["status"] = "ok"
            except Exception as exc:
                row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            with rows_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows[row_id] = row
    summary = []
    for ratio in RATIOS:
        condition = f"correct_{int(ratio * 100):03d}"
        subset = [r for r in rows.values() if r["condition"] == condition]
        valid = [r for r in subset if r["status"] == "ok"]
        paths = [output / r["image"] for r in valid]
        real_for_fid = [real_images[int(r["sample_id"].split("_")[-1])] for r in valid]
        score = fid(model, paths, real_for_fid, device, config["batch_size"]) if len(valid) >= 2 else None
        summary.append({"condition": condition, "correct_fraction": ratio, "evaluated": len(subset),
                        "failures": len(subset) - len(valid), "fid": score,
                        "l_cls": (float(np.mean([r["l_cls"] for r in valid if r["l_cls"] is not None]))
                                  if any(r["l_cls"] is not None for r in valid) else None),
                        "l_ocr": float(np.mean([r["l_ocr"] for r in valid])) if valid else None})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = [r"\begin{tabular}{lrrrr}", r"\hline", r"Correct key bits & FID & $L_{cls}$ & $L_{ocr}$ & Failures \\", r"\hline"]
    for item in summary:
        values = ["N/A" if item[name] is None else f"{item[name]:.4f}" for name in ("fid", "l_cls", "l_ocr")]
        lines.append(f"{int(item['correct_fraction'] * 100)}\\% & {' & '.join(values)} & {item['failures']}" + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    (output / "summary.tex").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    evaluate_runs(load_config(args.config), evaluate)


if __name__ == "__main__":
    main()
