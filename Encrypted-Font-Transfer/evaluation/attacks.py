"""Shared image attacks; inputs and outputs use grayscale [-1, 1]."""

from __future__ import annotations

import io
import random

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from models.noise_layer import MotionBlur


def standard_conditions(config):
    values = config.get("attacks", {})
    defaults = {
        "awgn": [0.1, 0.5, 0.7],
        "salt_pepper": [0.1, 0.2, 0.3],
        "gaussian_blur": [0.5, 1.0, 2.0],
        "motion_blur": [1, 5, 9],
        "jpeg": [75, 50, 25],
        "rotation": [5, 15, 25, 35, 50, 75, 180],
        "warping": [0.2, 0.3, 0.4],
        "resize": [0.98, 0.95, 0.90],
    }
    return [("clean", None)] + [(name, value) for name, parameters in defaults.items()
                                    for value in values.get(name, parameters)]


def to_unit(images):
    return images.clamp(-1, 1).add(1).div(2)


def to_model(images):
    return images.clamp(0, 1).mul(2).sub(1)


def _jpeg(image, quality):
    array = to_unit(image).squeeze().mul(255).round().byte().cpu().numpy()
    buffer = io.BytesIO()
    Image.fromarray(array, mode="L").save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    restored = torch.from_numpy(__import__("numpy").asarray(Image.open(buffer).convert("L")).copy())
    return to_model(restored.to(image.device, dtype=torch.float32).reshape_as(image).div(255))


def apply_attack(image, name, value, seed, settings=None):
    settings = settings or {}
    image = image.clone()
    rng = random.Random(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    unit = to_unit(image)
    if name == "clean":
        return image
    if name == "awgn":
        noise = torch.randn(unit.shape, generator=generator).to(unit.device) * float(value)
        return to_model(unit + noise)
    if name == "salt_pepper":
        mask = torch.rand(unit.shape, generator=generator).to(unit.device)
        density = float(value)
        return to_model(torch.where(mask < density / 2, torch.zeros_like(unit),
                                    torch.where(mask > 1 - density / 2, torch.ones_like(unit), unit)))
    if name == "gaussian_blur":
        return to_model(TF.gaussian_blur(unit, int(settings.get("gaussian_kernel", 7)), float(value)))
    if name == "motion_blur":
        return to_model(MotionBlur(int(value), float(settings.get("motion_angle", 25)), unit.device)(unit).clamp(0, 1))
    if name == "jpeg":
        return _jpeg(image, value)
    if name == "rotation":
        sign = -1 if rng.randrange(2) else 1
        rotated = TF.rotate(unit, sign * float(value), interpolation=InterpolationMode.BILINEAR, fill=1.0)
        return to_model(rotated)
    if name == "warping":
        height, width = unit.shape[-2:]
        strength = float(value)
        start = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        end = [[max(0, min(width - 1, x + rng.uniform(-strength, strength) * width)),
                max(0, min(height - 1, y + rng.uniform(-strength, strength) * height))]
               for x, y in start]
        return to_model(TF.perspective(unit, start, end, interpolation=InterpolationMode.BILINEAR, fill=1.0))
    if name == "resize":
        height, width = unit.shape[-2:]
        small = F.interpolate(unit, size=(max(1, round(height * float(value))),
                                          max(1, round(width * float(value)))),
                              mode="bilinear", align_corners=False)
        return to_model(F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False))
    if name == "gamma":
        return to_model(unit.clamp(0, 1).pow(float(value)))
    if name == "combined":
        result = apply_attack(image, "jpeg", 50, seed, settings)
        result = apply_attack(result, "resize", 0.75, seed, settings)
        return apply_attack(result, "rotation", 5, seed, settings)
    raise ValueError(f"unknown attack: {name}")
