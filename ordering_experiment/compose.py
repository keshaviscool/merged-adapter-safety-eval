"""Builds merged weight updates from LoRA adapters, one module at a time.

A merge "expression" is a nested tuple:
    ("leaf", "a1")               one adapter's delta
    ("sum", ("a1", "a3"))        LINEAR: plain sum of those deltas (order never matters)
    ("ties", expr_x, expr_y)     TIES of two expressions (symmetric: children are stored sorted)

delta of one adapter on one module = (alpha / rank) * B @ A   (alpha / sqrt(rank) for rsLoRA), in fp32.
The model for an expression is   W = W0 + delta(expression)   (cast to the model dtype once).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sequential_merging.delta_norms import load_adapter  # noqa: E402


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------
def key(expr) -> str:
    """Canonical text of an expression; equal key <=> same model."""
    if expr[0] == "leaf":
        return expr[1]
    if expr[0] == "sum":
        return "sum(" + ",".join(expr[1]) + ")"
    return f"ties({key(expr[1])},{key(expr[2])})"


def make_sum(names: list[str], all_names: list[str]):
    names = sorted(names, key=all_names.index)  # fixed order -> bit-identical fp32 sum for the same set
    return ("leaf", names[0]) if len(names) == 1 else ("sum", tuple(names))


def make_ties(x, y):
    x, y = sorted([x, y], key=key)  # TIES(x, y) == TIES(y, x), so store one canonical form
    return ("ties", x, y)


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------
class Adapters:
    """LoRA factors of every adapter, kept on the GPU (they are small)."""

    def __init__(self, adapter_cfgs, device: str):
        self.device = device
        self.lora = {}  # lora[adapter][module] = (A, B, scale)
        self.shapes = {}  # shapes[module] = shape of the weight update (out_features, in_features)
        for ad in adapter_cfgs:
            cfg, state = load_adapter(ad.path)
            if cfg.get("use_dora"):
                raise ValueError(f"{ad.name}: DoRA adapters are not supported")
            rank, alpha = cfg["r"], cfg["lora_alpha"]
            scale = alpha / math.sqrt(rank) if cfg.get("use_rslora") else alpha / rank
            self.lora[ad.name] = {}
            for k, a in state.items():
                if ".lora_A." in k:
                    module = k.split(".lora_A.")[0].removeprefix("base_model.model.")
                    b = state[k.replace(".lora_A.", ".lora_B.")]
                    self.lora[ad.name][module] = (a.to(device), b.to(device), scale)
                    self.shapes[module] = (b.shape[0], a.shape[1])
                elif ".lora_B." not in k:
                    raise ValueError(f"{ad.name}: unsupported tensor '{k}' (only plain LoRA A/B pairs are handled)")

    def modules(self) -> list[str]:
        return sorted({m for per_adapter in self.lora.values() for m in per_adapter})

    def delta(self, adapter: str, module: str) -> torch.Tensor:
        if module not in self.lora[adapter]:
            return torch.zeros(self.shapes[module], device=self.device)  # adapter does not touch this module
        a, b, scale = self.lora[adapter][module]
        return scale * (b.float() @ a.float())


# ---------------------------------------------------------------------------
# Merge operators
# ---------------------------------------------------------------------------
def trim(x: torch.Tensor, density: float) -> torch.Tensor:
    """Keep the `density` fraction of entries with the largest magnitude, zero the rest."""
    k = max(1, int(density * x.numel()))
    threshold = x.abs().flatten().kthvalue(x.numel() - k + 1).values
    return torch.where(x.abs() >= threshold, x, torch.zeros_like(x))


def ties(x: torch.Tensor, y: torch.Tensor, density: float) -> torch.Tensor:
    """TIES merge of two update matrices: trim, elect sign, average the entries that agree."""
    x, y = trim(x, density), trim(y, density)
    sign = torch.sign(x + y)  # sign with the larger total magnitude
    keep_x = (torch.sign(x) == sign) & (x != 0)
    keep_y = (torch.sign(y) == sign) & (y != 0)
    count = keep_x.float() + keep_y.float()
    return (torch.where(keep_x, x, torch.zeros_like(x)) + torch.where(keep_y, y, torch.zeros_like(y))) / count.clamp(min=1)


def evaluate(expr, module: str, adapters: Adapters, density: float) -> torch.Tensor:
    if expr[0] == "leaf":
        return adapters.delta(expr[1], module)
    if expr[0] == "sum":
        total = adapters.delta(expr[1][0], module)
        for name in expr[1][1:]:
            total = total + adapters.delta(name, module)
        return total
    return ties(
        evaluate(expr[1], module, adapters, density),
        evaluate(expr[2], module, adapters, density),
        density,
    )
