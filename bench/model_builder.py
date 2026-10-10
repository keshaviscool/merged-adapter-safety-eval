"""Builds the single model that will be benchmarked."""

from __future__ import annotations

import logging
from dataclasses import asdict

from transformers import AutoModelForCausalLM, AutoTokenizer

from bench.config import MergeConfig, ModelConfig
from bench.merging import apply_adapters
from bench.utils import get_torch_dtype

logger = logging.getLogger(__name__)


def build_model(model_cfg: ModelConfig, merge_cfg: MergeConfig):
    """Returns (model, tokenizer).

    No adapters -> the base model as is.
    Adapters    -> the base model with the adapter(s) merged in.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.base_model, trust_remote_code=model_cfg.trust_remote_code
    )
    tokenizer.padding_side = "left"  # required for batched generation with decoder-only models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model: %s", model_cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.base_model,
        torch_dtype=get_torch_dtype(model_cfg.dtype),
        device_map=model_cfg.device_map,
        trust_remote_code=model_cfg.trust_remote_code,
    )

    if model_cfg.adapters:
        model = apply_adapters(model, model_cfg.adapters, merge_cfg)
    else:
        logger.info("No adapters given: benchmarking the base model alone.")

    model.eval()
    return model, tokenizer


def describe_model(model_cfg: ModelConfig, merge_cfg: MergeConfig) -> dict:
    """A JSON-friendly description of what was evaluated (stored in the results)."""
    n_adapters = len(model_cfg.adapters)

    if n_adapters == 0:
        mode = "base_only"
    elif n_adapters == 1:
        mode = "single_adapter"
    else:
        mode = "merged_adapters"

    return {
        "mode": mode,
        "base_model": model_cfg.base_model,
        "dtype": model_cfg.dtype,
        "adapters": [asdict(a) for a in model_cfg.adapters],
        "merge": asdict(merge_cfg) if mode == "merged_adapters" else None,
    }
