"""Watermark method adapters, including optional external baselines."""

from __future__ import annotations

import importlib

import torch

from .common import ber_result, generate, load_model, projected_bits


METHODS = ("StegaStamp", "RiHoop", "InstanceWmk", "SepMark", "QuatMom", "MultiHist", "Ours")


class Ours:
    def __init__(self, config, key):
        self.device = config["device"]
        self.model = load_model(config)
        self.key = key
        self.bits = projected_bits(self.model, key, self.device)

    def embed(self, references, text, sample_seed):
        return generate(self.model, references, text, self.key, self.device, sample_seed)

    def extract(self, image):
        with torch.no_grad():
            return self.model.netExt(image.to(self.device))[0].detach().cpu()

    def verify(self, image):
        raise NotImplementedError("no calibrated binary verification rule is supplied for the main checkpoint")


class External:
    def __init__(self, entry, config, key, bits):
        module_name, object_name = entry["factory"].split(":", 1)
        factory = getattr(importlib.import_module(module_name), object_name)
        self.impl = factory(config=config, key=key, bits=bits)
        self.bits = bits

    def embed(self, references, text, sample_seed):
        return self.impl.embed(references=references, text=text, bits=self.bits, seed=sample_seed)

    def extract(self, image):
        return self.impl.extract(image=image)

    def verify(self, image):
        return self.impl.verify(image=image)


def available_methods(config, key):
    chosen = config.get("methods", list(METHODS))
    plugins = config.get("external_methods", {})
    methods, unavailable = {}, {}
    for name in chosen:
        if name == "InstancWmk":
            name = "InstanceWmk"
        if name not in METHODS:
            raise ValueError(f"unknown method: {name}")
        if name == "Ours":
            try:
                methods[name] = Ours(config, key)
            except (FileNotFoundError, ImportError) as exc:
                unavailable[name] = str(exc)
            continue
        if name not in plugins:
            unavailable[name] = "no external method factory configured"
            continue
        try:
            if "Ours" not in methods:
                ours = Ours(config, key)
                methods["Ours"] = ours
            methods[name] = External(plugins[name], config, key, methods["Ours"].bits)
        except (FileNotFoundError, ImportError, AttributeError, KeyError) as exc:
            unavailable[name] = str(exc)
    return methods, unavailable


def extract_ber(method, image):
    return ber_result(method.extract(image), method.bits)
