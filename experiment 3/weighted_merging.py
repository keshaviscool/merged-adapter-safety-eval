"""Loads LoRA adapters onto a base model and merges them.

Cases handled:
  * 1 adapter   -> load it and fold it into the base weights.
  * 2+ adapters -> combine them with PEFT's `add_weighted_adapter` using the
                   strategy, weights and density from the config, then fold the
                   result into the base weights.

The returned model is a plain `transformers` model (no PEFT wrapper), so the
generation code does not need to know adapters were ever involved.
"""

from __future__ import annotations

import logging

from peft import PeftModel

from bench.config import (
    STRATEGIES_NEEDING_DENSITY,
    STRATEGIES_USING_SIGN_METHOD,
    AdapterConfig,
    MergeConfig,
)

logger = logging.getLogger(__name__)

MERGED_ADAPTER_NAME = "merged"


def _load_all_adapters(base_model, adapters: list[AdapterConfig]) -> PeftModel:
    """Attach every adapter to the base model under its own name."""
    first = adapters[0]
    peft_model = PeftModel.from_pretrained(base_model, first.path, adapter_name=first.name)
    for adapter in adapters[1:]:
        peft_model.load_adapter(adapter.path, adapter_name=adapter.name)
    return peft_model


def _build_merge_kwargs(merge_cfg: MergeConfig) -> dict:
    """Only pass the options that the chosen strategy actually uses."""
    kwargs = {"combination_type": merge_cfg.strategy}

    if merge_cfg.strategy in STRATEGIES_NEEDING_DENSITY:
        kwargs["density"] = merge_cfg.density
    if merge_cfg.strategy in STRATEGIES_USING_SIGN_METHOD:
        kwargs["majority_sign_method"] = merge_cfg.majority_sign_method
    if merge_cfg.svd_rank is not None:
        kwargs["svd_rank"] = merge_cfg.svd_rank

    return kwargs


def _merge_into_new_adapter(
    peft_model: PeftModel, adapters: list[AdapterConfig], merge_cfg: MergeConfig
) -> None:
    kwargs = _build_merge_kwargs(merge_cfg)
    logger.info(
        "Merging %d adapters: names=%s weights=%s %s",
        len(adapters),
        [a.name for a in adapters],
        [a.weight for a in adapters],
        kwargs,
    )
    peft_model.add_weighted_adapter(
        adapters=[a.name for a in adapters],
        weights=[a.weight for a in adapters],
        adapter_name=MERGED_ADAPTER_NAME,
        **kwargs,
    )


def apply_adapters(base_model, adapters: list[AdapterConfig], merge_cfg: MergeConfig):
    """Returns a plain model with the adapter(s) folded into the weights."""
    peft_model = _load_all_adapters(base_model, adapters)

    if len(adapters) == 1:
        single = adapters[0]
        logger.info("One adapter given ('%s'): applying its configured weight.", single.name)
        if single.weight != 1.0:
            logger.info("Applying single-adapter weight %.3f", single.weight)
            peft_model.add_weighted_adapter(
                adapters=[single.name], weights=[single.weight],
                adapter_name=MERGED_ADAPTER_NAME, combination_type="linear",
            )
            peft_model.set_adapter(MERGED_ADAPTER_NAME)
        else:
            peft_model.set_adapter(single.name)
    else:
        _merge_into_new_adapter(peft_model, adapters, merge_cfg)
        # Only the merged adapter is active, so merge_and_unload folds in just that one.
        peft_model.set_adapter(MERGED_ADAPTER_NAME)

    return peft_model.merge_and_unload()
