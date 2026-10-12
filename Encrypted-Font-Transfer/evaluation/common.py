from __future__ import annotations

import csv
import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("dataset", "key", "output", "english_words"):
        if key not in config:
            raise ValueError(f"missing config field: {key}")
        value = Path(config[key]).expanduser()
        config[key] = str(value if value.is_absolute() else (path.parent / value).resolve())
    if "checkpoint" in config:
        value = Path(config["checkpoint"]).expanduser()
        config["checkpoint"] = str(value if value.is_absolute() else (path.parent / value).resolve())
    if "runs" in config:
        for run in config["runs"]:
            for key in ("checkpoint", "dataset", "key", "sample_manifest", "english_words", "training_config"):
                if key in run:
                    value = Path(run[key]).expanduser()
                    run[key] = str(value if value.is_absolute() else (path.parent / value).resolve())
    if "sample_manifest" in config:
        value = Path(config["sample_manifest"]).expanduser()
        config["sample_manifest"] = str(value if value.is_absolute() else (path.parent / value).resolve())
    if "training_config" in config:
        value = Path(config["training_config"]).expanduser()
        config["training_config"] = str(value if value.is_absolute() else (path.parent / value).resolve())
    config.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    config.setdefault("batch_size", 8)
    config.setdefault("sample_count", 1000)
    config.setdefault("seed", 20261012)
    if int(config["sample_count"]) <= 0 or int(config["batch_size"]) <= 0:
        raise ValueError("sample_count and batch_size must be positive")
    return config


def seeded(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def identity_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def start_output(config, output=None):
    output = Path(output or config["output"])
    output.mkdir(parents=True, exist_ok=True)
    saved = output / "config.json"
    canonical = json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if saved.exists() and saved.read_text(encoding="utf-8") != canonical:
        raise RuntimeError(f"configuration changed in resumed output: {saved}")
    saved.write_text(canonical, encoding="utf-8")
    return output


def load_dataset(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        data = pickle.load(stream)
    test = data["test"]
    return {str(writer): rows for writer, rows in test.items() if str(writer) != "None"}


def training_writer_order(path):
    with Path(path).open("rb") as stream:
        data = pickle.load(stream)
    return {str(writer): index for index, writer in enumerate(
        name for name in data["train"] if str(name) != "None")}


def fixed_samples(data, config, output):
    path = Path(config.get("sample_manifest", output / "samples.json"))
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["dataset_sha256"] != identity_digest(config["dataset"]):
            raise RuntimeError("sample manifest belongs to a different dataset")
        if len(payload["samples"]) < config["sample_count"]:
            raise RuntimeError("sample manifest has too few entries")
        return payload["samples"][: config["sample_count"]]
    rng = random.Random(config["seed"])
    candidates = []
    for writer, rows in sorted(data.items()):
        for index, row in enumerate(rows):
            label = row["label"]
            if isinstance(label, bytes):
                label = label.decode("utf-8")
            if 1 <= len(str(label)) <= 12:
                candidates.append((writer, index))
    if len(candidates) < config["sample_count"]:
        raise ValueError("IAM test set has fewer distinct target images than sample_count")
    rng.shuffle(candidates)
    samples = []
    for writer, index in candidates[: config["sample_count"]]:
        references = [i for i in range(len(data[writer])) if i != index]
        if not references:
            raise ValueError(f"writer {writer} has no reference images")
        refs = [rng.choice(references) for _ in range(15)]
        label = data[writer][index]["label"]
        if isinstance(label, bytes):
            label = label.decode("utf-8")
        samples.append({"sample_id": f"sample_{len(samples):05d}", "writer": writer,
                        "target_index": index, "reference_indices": refs, "text": str(label)})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"dataset_sha256": identity_digest(config["dataset"]),
                                "seed": config["seed"], "samples": samples}, indent=2) + "\n")
    return samples


def normalized_image(image):
    image = image.convert("L")
    if image.height != 32:
        raise ValueError(f"expected height 32, got {image.height}")
    raw = np.asarray(image, dtype=np.float32)[:, :192] / 255.0
    padded = np.ones((32, 192), dtype=np.float32)
    padded[:, :raw.shape[1]] = raw
    return torch.from_numpy(padded).unsqueeze(0).mul(2).sub(1)


def sample_tensors(data, sample):
    rows = data[sample["writer"]]
    refs = torch.cat([normalized_image(rows[i]["img"]) for i in sample["reference_indices"]], dim=0)
    real = normalized_image(rows[sample["target_index"]]["img"])
    return refs, real


def save_png(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().cpu().squeeze().clamp(-1, 1).add(1).div(2).mul(255).round().byte().numpy()
    Image.fromarray(array, mode="L").save(path)


def read_png(path):
    return normalized_image(Image.open(path))


def load_key(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    key = torch_load(path) if path.suffix in (".pt", ".pth") else torch.tensor(json.loads(path.read_text()))
    key = torch.as_tensor(key, dtype=torch.float32).reshape(-1)
    if key.numel() != 512 or not bool(((key == 0) | (key == 1)).all()):
        raise ValueError("user key must contain exactly 512 binary bits")
    return key


def load_model(config):
    checkpoint = Path(config["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not Path(config["key"]).is_file():
        raise FileNotFoundError(config["key"])
    import params
    params.DEVICE = torch.device(config["device"])
    params.ENGLISH_WORDS_PATH = config["english_words"]
    from models.model import TRGAN
    model = TRGAN(load_pretrained=False)
    model.load_state_dict(torch_load(checkpoint), strict=True)
    model.eval()
    return model


def projected_bits(model, key, device):
    encoded = (key.to(device) * 6 - 3).unsqueeze(0)
    with torch.no_grad():
        bits = torch.round(model.keyHashNet.sig(model.keyHashNet.fc(encoded))).reshape(-1)
    if bits.numel() != 128:
        raise RuntimeError(f"projection returned {bits.numel()} bits, expected 128")
    return bits


def generate(model, references, text, key, device, seed):
    seeded(seed)
    label, _ = model.netconverter.encode([text.encode("utf-8")])
    with torch.no_grad():
        image, _ = model.netG(references.unsqueeze(0).to(device), label.to(device),
                              (key.to(device) * 6 - 3).unsqueeze(0),
                              key.to(device).reshape(1, 1, 512), [])
    if image.shape[-2:] != (32, 192):
        image = pad_image(image)
    return image.detach()


def pad_image(image):
    if image.shape[-2] != 32 or image.shape[-1] > 192:
        raise ValueError(f"generated image is not compatible with 32x192: {tuple(image.shape)}")
    return torch.nn.functional.pad(image, (0, 192 - image.shape[-1]), value=1.0)


def ber_result(extracted, target):
    extracted = torch.as_tensor(extracted).reshape(-1)
    target = torch.as_tensor(target).reshape(-1)
    if extracted.numel() != 128 or target.numel() != 128:
        raise ValueError("extractor and target must both contain 128 bits")
    if not bool(torch.isfinite(extracted).all()):
        raise ValueError("extractor returned nonfinite values")
    errors = int(((extracted >= 0.5) != target.bool()).sum().item())
    return {"error_bits": errors, "total_bits": 128, "ber": errors / 128}


def completed_rows(path):
    rows = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                rows[row["row_id"]] = row
    return rows


def append_row(path, row):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def write_reports(output, rows, metric="ber"):
    groups = {}
    for row in rows.values():
        key = (row.get("method", "ours"), row["condition"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (method, condition), items in sorted(groups.items()):
        valid = [row[metric] for row in items if row.get("status") == "ok" and isinstance(row.get(metric), (int, float))]
        summary.append({"method": method, "condition": condition, "status": "ok" if valid else "N/A",
                        "evaluated": sum(item.get("status") != "N/A" for item in items),
                        "successes": len(valid),
                        "failures": sum(item.get("status") == "failed" for item in items),
                        f"mean_{metric}": float(np.mean(valid)) if valid else None})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]) if summary else ["method", "condition"])
        writer.writeheader()
        writer.writerows(summary)
    lines = [r"\begin{tabular}{llrrr}", r"\hline", r"Method & Condition & N & Failures & Mean \\", r"\hline"]
    for item in summary:
        value = item[f"mean_{metric}"]
        display = f"{value:.4f}" if value is not None else "N/A"
        lines.append(f"{item['method']} & {item['condition'].replace('_', ' ')} & {item['evaluated']} & {item['failures']} & {display}" + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    (output / "summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def evaluate_runs(config, evaluator):
    runs = config.get("runs")
    if not runs:
        return evaluator(config)
    import copy
    base_output = Path(config["output"])
    reports = []
    for index, run in enumerate(runs):
        run_config = copy.deepcopy(config)
        run_config.pop("runs", None)
        run_config.update(run)
        run_config["output"] = str(base_output / f"run_{index + 1:02d}")
        reports.append(evaluator(run_config))
    checkpoint_ids = [identity_digest(run["checkpoint"]) for run in runs]
    split_ids = [run.get("split_id") for run in runs]
    split_hashes = [identity_digest(run["sample_manifest"]) if run.get("sample_manifest")
                    and Path(run["sample_manifest"]).is_file() else None for run in runs]
    independent = (len(runs) >= 5 and all(run.get("independent_training_run") is True for run in runs)
                   and len(set(checkpoint_ids)) == len(runs)
                   and None not in split_ids and len(set(split_ids)) == len(runs)
                   and None not in split_hashes and len(set(split_hashes)) == len(runs))
    aggregate = {"independent_training_ci_available": independent, "run_count": len(runs)}
    if independent:
        from scipy.stats import t
        by_key = {}
        for report in reports:
            for row in report:
                key = (row.get("method", "ours"), row["condition"])
                for metric in ("mean_ber", "fid", "l_cls", "l_ocr"):
                    value = row.get(metric)
                    if value is not None:
                        by_key.setdefault((*key, metric), []).append(value)
        values = []
        for (method, condition, metric), samples in sorted(by_key.items()):
            if len(samples) < 5:
                continue
            mean = float(np.mean(samples))
            margin = float(t.ppf(0.975, len(samples) - 1) * np.std(samples, ddof=1) / np.sqrt(len(samples)))
            values.append({"method": method, "condition": condition, "metric": metric,
                           "mean": mean, "ci95_low": mean - margin, "ci95_high": mean + margin,
                           "training_runs": len(samples)})
        aggregate["metrics"] = values
        with (base_output / "across_runs.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]) if values else ["method", "condition", "metric"])
            writer.writeheader()
            writer.writerows(values)
        latex = [r"\begin{tabular}{lllrr}", r"\hline",
                 r"Method & Condition & Metric & Mean & 95\% CI \\", r"\hline"]
        for item in values:
            interval = f"[{item['ci95_low']:.4f}, {item['ci95_high']:.4f}]"
            latex.append(f"{item['method']} & {item['condition'].replace('_', ' ')} & {item['metric'].replace('_', ' ')} & {item['mean']:.4f} & {interval}" + r" \\")
        latex.extend([r"\hline", r"\end{tabular}"])
        (base_output / "across_runs.tex").write_text("\n".join(latex) + "\n")
    (base_output / "across_runs.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    return reports
