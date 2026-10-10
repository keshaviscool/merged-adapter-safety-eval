"""Common interface for all detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from bench.generation import Generation
from bench.utils import free_gpu_memory


@dataclass
class Verdict:
    harmful: bool          # True = the response is harmful, i.e. the attack succeeded
    raw_output: str        # what the detector model literally produced
    details: dict = field(default_factory=dict)  # detector-specific extras


class Detector(ABC):
    """Loads a judge model, labels generations, then frees the GPU.

    Use as a context manager so the model is always released:

        with LlamaGuard3Detector(cfg) as detector:
            verdicts = detector.judge(generations)
    """

    name: str  # used as the key in all result files

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def judge(self, generations: list[Generation]) -> list[Verdict]:
        """Returns one Verdict per generation, in the same order."""

    def unload(self) -> None:
        for attribute in ("model", "processor", "tokenizer"):
            if hasattr(self, attribute):
                setattr(self, attribute, None)
        free_gpu_memory()

    def __enter__(self) -> "Detector":
        self.load()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.unload()