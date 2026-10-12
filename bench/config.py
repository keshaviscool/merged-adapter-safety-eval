"""Configuration dataclasses and the YAML loader.

Every knob of the benchmark lives in one YAML file. Unknown keys raise an
error on purpose, so typos in the config never pass silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Merge strategies. These are the names PEFT calls "combination types" in
# `PeftModel.add_weighted_adapter`.
# ---------------------------------------------------------------------------
MERGE_STRATEGIES = {
    "linear",
    "cat",
    "svd",
    "ties",
    "ties_svd",
    "dare_linear",
    "dare_linear_svd",
    "dare_ties",
    "dare_ties_svd",
    "magnitude_prune",
    "magnitude_prune_svd",
}

# Strategies that prune parameters and therefore need a `density` in (0, 1].
STRATEGIES_NEEDING_DENSITY = {
    "ties",
    "ties_svd",
    "dare_linear",
    "dare_linear_svd",
    "dare_ties",
    "dare_ties_svd",
    "magnitude_prune",
    "magnitude_prune_svd",
}

# Strategies that resolve sign conflicts (TIES-style) and use `majority_sign_method`.
STRATEGIES_USING_SIGN_METHOD = {"ties", "ties_svd", "dare_ties", "dare_ties_svd"}


# ---------------------------------------------------------------------------
# Model and merging
# ---------------------------------------------------------------------------
@dataclass
class AdapterConfig:
    path: str                    # local path or Hugging Face hub id of a PEFT LoRA adapter
    name: str | None = None      # label used inside PEFT (avoid "." in names)
    weight: float = 1.0          # this adapter's weight in the merge


@dataclass
class ModelConfig:
    base_model: str
    adapters: list[AdapterConfig] = field(default_factory=list)
    dtype: str = "bfloat16"
    device_map: str = "cuda"
    trust_remote_code: bool = False


@dataclass
class MergeConfig:
    """Only used when two or more adapters are given."""

    strategy: str = "linear"
    density: float | None = None            # for ties / dare_* / magnitude_prune
    majority_sign_method: str = "total"     # for ties / dare_ties ("total" or "frequency")
    svd_rank: int | None = None             # for *_svd strategies

    def __post_init__(self) -> None:
        if self.strategy not in MERGE_STRATEGIES:
            raise ValueError(
                f"Unknown merge strategy '{self.strategy}'. "
                f"Choose from: {sorted(MERGE_STRATEGIES)}"
            )
        if self.strategy in STRATEGIES_NEEDING_DENSITY:
            if self.density is None or not (0.0 < self.density <= 1.0):
                raise ValueError(
                    f"Strategy '{self.strategy}' needs `density` in (0, 1], got {self.density}"
                )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    do_sample: bool = False          # False = greedy (temperature/top_p/top_k ignored)
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0
    batch_size: int = 8
    seed: int = 42
    system_prompt: str | None = None


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
@dataclass
class HarmBenchConfig:
    enabled: bool = True
    functional_categories: list[str] = field(default_factory=lambda: ["standard", "contextual"])
    max_examples: int | None = None  # None = use all; handy for smoke tests


@dataclass
class AdvBenchConfig:
    enabled: bool = True
    max_examples: int | None = None


@dataclass
class AlpacaConfig:
    """Benign prompts (tatsu-lab/alpaca) used for the false refusal rate (FRR)."""

    enabled: bool = True
    num_samples: int | None = 500   # random sample size out of ~52k prompts; None = use all
    seed: int = 42                  # makes the sample identical on every run

    def __post_init__(self) -> None:
        if self.num_samples is not None and self.num_samples < 1:
            raise ValueError(f"alpaca.num_samples must be a positive integer or null, got {self.num_samples}")


@dataclass
class DatasetsConfig:
    cache_dir: str = "data_cache"
    harmbench: HarmBenchConfig = field(default_factory=HarmBenchConfig)
    advbench: AdvBenchConfig = field(default_factory=AdvBenchConfig)
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
@dataclass
class LlamaGuard3Config:
    enabled: bool = True
    model_id: str = "meta-llama/Llama-Guard-3-8B"
    device_map: str = "cuda"
    max_new_tokens: int = 20


@dataclass
class HarmClsConfig:
    enabled: bool = True
    model_id: str = "cais/HarmBench-Llama-2-13b-cls"
    device_map: str = "cuda"
    batch_size: int = 8
    # HarmBench clips each response to this many tokens before classifying.
    max_generation_tokens: int = 512


@dataclass
class RejectionClassifierConfig:
    """Refusal detector used for FRR on the benign (Alpaca) prompts."""

    enabled: bool = True
    model_id: str = "ProtectAI/distilroberta-base-rejection-v1"
    batch_size: int = 32
    max_length: int = 512   # responses are truncated to this many tokens


@dataclass
class KeywordRefusalConfig:
    """Keyword-based refusal detector for FRR (no model involved)."""

    enabled: bool = True
    check_first_chars: int = 200          # only the start of the response is searched
    phrases: list[str] | None = None      # None = built-in list; a list here replaces it entirely

    def __post_init__(self) -> None:
        if self.check_first_chars < 1:
            raise ValueError(f"keyword_refusal.check_first_chars must be >= 1, got {self.check_first_chars}")
        if self.phrases is not None:
            if not self.phrases or any(not phrase.strip() for phrase in self.phrases):
                raise ValueError("keyword_refusal.phrases must be null or a list of non-empty strings")


@dataclass
class DetectorsConfig:
    llama_guard3: LlamaGuard3Config = field(default_factory=LlamaGuard3Config)
    harm_cls: HarmClsConfig = field(default_factory=HarmClsConfig)
    rejection: RejectionClassifierConfig = field(default_factory=RejectionClassifierConfig)
    keyword_refusal: KeywordRefusalConfig = field(default_factory=KeywordRefusalConfig)


# ---------------------------------------------------------------------------
# Output and top-level
# ---------------------------------------------------------------------------
@dataclass
class OutputConfig:
    root: str = "results"


@dataclass
class BenchConfig:
    model: ModelConfig
    merge: MergeConfig = field(default_factory=MergeConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    datasets: DatasetsConfig = field(default_factory=DatasetsConfig)
    detectors: DetectorsConfig = field(default_factory=DetectorsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def _load_model_config(raw: dict) -> ModelConfig:
    raw = dict(raw)
    adapters_raw = raw.pop("adapters", None) or []

    adapters = []
    for index, adapter_raw in enumerate(adapters_raw):
        adapter = AdapterConfig(**adapter_raw)
        if adapter.name is None:
            adapter.name = f"adapter_{index}"
        adapters.append(adapter)

    names = [a.name for a in adapters]
    if len(set(names)) != len(names):
        raise ValueError(f"Adapter names must be unique, got {names}")

    return ModelConfig(adapters=adapters, **raw)


def load_config(path: str | Path) -> BenchConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    datasets_raw = raw.get("datasets", {})
    datasets = DatasetsConfig(
        cache_dir=datasets_raw.get("cache_dir", "data_cache"),
        harmbench=HarmBenchConfig(**datasets_raw.get("harmbench", {})),
        advbench=AdvBenchConfig(**datasets_raw.get("advbench", {})),
        alpaca=AlpacaConfig(**datasets_raw.get("alpaca", {})),
    )

    detectors_raw = raw.get("detectors", {})
    unknown_detectors = set(detectors_raw) - {"llama_guard3", "harm_cls", "rejection", "keyword_refusal"}
    if unknown_detectors:
        raise ValueError(
            f"Unknown detector(s) in config: {sorted(unknown_detectors)}. "
            "Valid keys: ['llama_guard3', 'harm_cls', 'rejection', 'keyword_refusal']"
        )
    detectors = DetectorsConfig(
        llama_guard3=LlamaGuard3Config(**detectors_raw.get("llama_guard3", {})),
        harm_cls=HarmClsConfig(**detectors_raw.get("harm_cls", {})),
        rejection=RejectionClassifierConfig(**detectors_raw.get("rejection", {})),
        keyword_refusal=KeywordRefusalConfig(**detectors_raw.get("keyword_refusal", {})),
    )

    return BenchConfig(
        model=_load_model_config(raw["model"]),
        merge=MergeConfig(**raw.get("merge", {})),
        generation=GenerationConfig(**raw.get("generation", {})),
        datasets=datasets,
        detectors=detectors,
        output=OutputConfig(**raw.get("output", {})),
    )