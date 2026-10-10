"""Small helpers shared across the benchmark."""

from __future__ import annotations

import gc
import logging

import torch
from transformers import set_seed


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_global_seed(seed: int) -> None:
    """Seeds python, numpy and torch."""
    set_seed(seed)


def free_gpu_memory() -> None:
    """Call after dropping the last reference to a model."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unknown dtype '{name}'. Choose from {sorted(mapping)}")
    return mapping[name]
