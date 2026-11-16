"""Delta-weight norm of each LoRA adapter (needs only the adapter files, no base model).

For every adapted layer: delta = (alpha / rank) * B @ A   (alpha / sqrt(rank) if use_rslora).
Per adapter we report the Frobenius norm of delta: total over all layers, mean per layer,
and per module type (q_proj, v_proj, ...).

    python sequential_merging/delta_norms.py        # adapters from sequential_merging/config.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent


def load_adapter(path: str) -> tuple[dict, dict]:
    """Returns (adapter_config, state_dict) from a local folder or a hub id."""
    def fetch(name: str) -> str:
        local = Path(path) / name
        return str(local) if local.exists() else hf_hub_download(path, name)

    with open(fetch("adapter_config.json")) as f:
        cfg = json.load(f)
    return cfg, load_file(fetch("adapter_model.safetensors"))


def layer_norms(cfg: dict, state: dict) -> dict[str, float]:
    """Frobenius norm of (alpha/rank) * B @ A for every adapted layer."""
    rank, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / math.sqrt(rank) if cfg.get("use_rslora") else alpha / rank
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norms = {}
    for key, a in state.items():
        if ".lora_A." not in key:
            continue
        b = state[key.replace(".lora_A.", ".lora_B.")]
        name = key.split(".lora_A.")[0].removeprefix("base_model.model.")
        norms[name] = (scale * (b.to(device).float() @ a.to(device).float())).norm().item()
    return norms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--out", default=str(HERE / "delta_norms.json"))
    args = parser.parse_args()
    with open(args.config) as f:
        adapters = yaml.safe_load(f)["model"]["adapters"]

    result = {}
    for ad in adapters:
        cfg, state = load_adapter(ad["path"])
        norms = layer_norms(cfg, state)
        sq_by_type = defaultdict(float)
        for layer, norm in norms.items():
            sq_by_type[layer.split(".")[-1]] += norm**2
        result[ad["name"]] = {
            "path": ad["path"],
            "rank": cfg["r"],
            "alpha": cfg["lora_alpha"],
            "use_rslora": cfg.get("use_rslora", False),
            "n_adapted_layers": len(norms),
            "total_frobenius_norm": math.sqrt(sum(sq_by_type.values())),
            "mean_per_layer_norm": sum(norms.values()) / len(norms),
            "norm_by_module_type": {t: math.sqrt(v) for t, v in sq_by_type.items()},
            "per_layer_norm": norms,
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"{'adapter':<8}{'rank':>5}{'alpha':>7}{'layers':>8}{'total ||dW||_F':>17}{'mean/layer':>12}")
    for name, r in result.items():
        print(f"{name:<8}{r['rank']:>5}{r['alpha']:>7}{r['n_adapted_layers']:>8}"
              f"{r['total_frobenius_norm']:>17.4f}{r['mean_per_layer_norm']:>12.4f}")
    print(f"\nPer-module-type and per-layer norms saved to {args.out}")


if __name__ == "__main__":
    main()
