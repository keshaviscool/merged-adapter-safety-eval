"""Server-only reconstruction and streaming weight-space measurements."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import torch

from pairwise_experiment.common import digest, multiplier, write_csv, write_json


class AdapterBank:
    """Keep small LoRA factors on CPU; materialize only the current module on GPU."""

    def __init__(self, adapters, device="cuda"):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        import json

        self.device, self.factors, self.shapes, self.metadata = device, {}, {}, {}
        for adapter in adapters:
            name, path = adapter["name"], adapter["path"]

            def fetch(filename):
                local = Path(path) / filename
                return local if local.exists() else Path(hf_hub_download(path, filename))

            config_path, weights_path = fetch("adapter_config.json"), fetch("adapter_model.safetensors")
            cfg = json.loads(config_path.read_text())
            validate_adapter_config(cfg)
            scale = cfg["lora_alpha"] / (math.sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["r"])
            state = load_file(str(weights_path), device="cpu")
            factors, consumed = {}, set()
            for key, a in state.items():
                if ".lora_A." not in key:
                    continue
                b_key = key.replace(".lora_A.", ".lora_B.")
                b = state[b_key]
                module = key.split(".lora_A.")[0].removeprefix("base_model.model.")
                if a.ndim != 2 or b.ndim != 2 or a.shape[0] != cfg["r"] or b.shape[1] != cfg["r"]:
                    raise ValueError(f"Invalid LoRA shapes for {name}/{module}")
                shape = (b.shape[0], a.shape[1])
                if module in self.shapes and self.shapes[module] != shape:
                    raise ValueError(f"Incompatible adapter shapes: {module}")
                if module in factors:
                    raise ValueError(f"Multiple LoRA entries for {name}/{module}")
                self.shapes[module] = shape
                factors[module] = (a, b, scale)
                consumed.update((key, b_key))
            if not factors or set(state) != consumed:
                raise ValueError(f"{name}: only plain LoRA A/B tensors are supported")
            self.factors[name] = factors
            self.metadata[name] = {
                "path": path, "rank": cfg["r"], "alpha": cfg["lora_alpha"],
                "scale": scale, "use_rslora": cfg.get("use_rslora", False),
                "config_sha256": digest(config_path), "weights_sha256": digest(weights_path),
            }

    def delta(self, name, module):
        if module not in self.factors[name]:
            return torch.zeros(self.shapes[module], device=self.device, dtype=torch.float32)
        a, b, scale = self.factors[name][module]
        result = scale * (b.to(self.device).float() @ a.to(self.device).float())
        if not torch.isfinite(result).all():
            raise ValueError(f"Nonfinite delta: {name}/{module}")
        return result


def validate_adapter_config(cfg):
    """Reject metadata that the dense-delta implementation does not implement."""
    if cfg.get("peft_type", "LORA") != "LORA":
        raise ValueError("Only LORA adapters are supported")
    for key in ("use_dora", "fan_in_fan_out", "rank_pattern", "alpha_pattern", "modules_to_save", "lora_bias", "use_qalora"):
        if cfg.get(key):
            raise ValueError(f"Unsupported adapter metadata: {key}")
    if cfg.get("bias", "none") != "none":
        raise ValueError("Adapter bias updates are unsupported")
    if cfg.get("r", 0) <= 0 or not math.isfinite(cfg.get("lora_alpha", float("nan"))):
        raise ValueError("Invalid rank/alpha")


def top_mask(x, density):
    """Same per-matrix threshold as ordering TIES; exclude zero coordinates."""
    k = max(1, int(density * x.numel()))
    threshold = x.abs().flatten().kthvalue(x.numel() - k + 1).values
    return (x.abs() >= threshold) & (x != 0)


def sufficient_stats(x, y, base, density):
    mx, my = top_mask(x, density), top_mask(y, density)
    joint = (x != 0) & (y != 0)
    return {
        "x_sq": x.double().square().sum().item(),
        "y_sq": y.double().square().sum().item(),
        "dot": (x.double() * y.double()).sum().item(),
        "base_sq": base.double().square().sum().item(),
        "n_coordinates": x.numel(), "joint_nonzero": joint.sum().item(),
        "sign_conflicts": (joint & (torch.sign(x) != torch.sign(y))).sum().item(),
        "top_intersection": (mx & my).sum().item(), "top_union": (mx | my).sum().item(),
        "top_x": mx.sum().item(), "top_y": my.sum().item(),
    }


def derived(stats):
    nx, ny, nw = (math.sqrt(stats[k]) for k in ("x_sq", "y_sq", "base_sq"))
    return {
        **stats, "norm_i": nx, "norm_j": ny,
        "cosine": max(-1., min(1., stats["dot"] / (nx * ny))) if nx and ny else None,
        "norm_ratio_i_over_j": nx / ny if nx and ny else None,
        "norm_balance": min(nx, ny) / max(nx, ny) if nx and ny else None,
        "relative_to_base_i": nx / nw if nw else None,
        "relative_to_base_j": ny / nw if nw else None,
        "sign_conflict": stats["sign_conflicts"] / stats["joint_nonzero"] if stats["joint_nonzero"] else None,
        "top_jaccard": stats["top_intersection"] / stats["top_union"] if stats["top_union"] else None,
        "retained_fraction_i": stats["top_x"] / stats["n_coordinates"],
        "retained_fraction_j": stats["top_y"] / stats["n_coordinates"],
    }


@torch.inference_mode()
def measure(run_dir, raw):
    from transformers import AutoModelForCausalLM

    bank = AdapterBank(raw["model"]["adapters"])
    # No generation model is present in this process. Only these base weights and factors.
    model = AutoModelForCausalLM.from_pretrained(
        raw["model"]["base_model"], torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=raw["model"].get("trust_remote_code", False),
    )
    params = dict(model.named_parameters())
    names = list(bank.factors)
    rows, aggregates = [], defaultdict(lambda: defaultdict(float))
    norms_sq = dict.fromkeys(names, 0.)
    for module in sorted(bank.shapes):
        print(f"Geometry: {module}", flush=True)
        weight = params[module + ".weight"]
        if tuple(weight.shape) != bank.shapes[module]:
            raise ValueError(f"Base weight shape mismatch: {module}")
        deltas = {name: bank.delta(name, module) for name in names}
        for name, delta in deltas.items():
            norms_sq[name] += delta.double().square().sum().item()
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", module)
        block = f"layer_{match.group(1)}" if match else "other"
        for a, b in combinations(names, 2):
            stats = sufficient_stats(deltas[a], deltas[b], weight, raw["pairwise"]["overlap_density"])
            rows.append({"adapter_i": a, "adapter_j": b, "level": "module", "location": module, **derived(stats)})
            for level, location in (("block", block), ("global", "all")):
                for field, value in stats.items():
                    aggregates[(a, b, level, location)][field] += value
        del deltas
    for (a, b, level, location), stats in aggregates.items():
        rows.append({"adapter_i": a, "adapter_j": b, "level": level, "location": location, **derived(stats)})
    write_csv(run_dir / "geometry.csv", rows)
    for name in names:
        bank.metadata[name]["frobenius_norm"] = math.sqrt(norms_sq[name])
    coefficients = []
    from pairwise_experiment.common import METHODS
    for a, b in combinations(names, 2):
        for method in METHODS:
            c = None if method == "ties" else multiplier(
                method, [bank.metadata[n]["scale"] for n in (a, b)],
                [bank.metadata[n]["frobenius_norm"] for n in (a, b)])
            coefficients.append({"adapter_i": a, "adapter_j": b, "method": method,
                                 "weight_i": c, "weight_j": c})
    write_csv(run_dir / "coefficients.csv", coefficients)
    # Last file is the completion marker; partial geometry is recomputed on resume.
    write_json(run_dir / "adapter_norms.json", bank.metadata)
