"""Detector factory: turns the config into a list of detector objects."""

from __future__ import annotations

from bench.config import DetectorsConfig
from bench.detectors.base import Detector, Verdict
from bench.detectors.harm_cls import HarmClsDetector
from bench.detectors.llama_guard3 import LlamaGuard3Detector
from bench.detectors.rejection_classifier import RejectionDetector

__all__ = ["Detector", "Verdict", "build_detectors"]


def build_detectors(config: DetectorsConfig) -> list[Detector]:
    """Creates the enabled detectors. Models are only loaded later, one at a time."""
    detectors: list[Detector] = []

    if config.llama_guard3.enabled:
        detectors.append(LlamaGuard3Detector(config.llama_guard3))   # metric: asr
    if config.harm_cls.enabled:
        detectors.append(HarmClsDetector(config.harm_cls))           # metric: asr
    if config.rejection.enabled:
        detectors.append(RejectionDetector(config.rejection))        # metric: frr

    if not detectors:
        raise ValueError("No detectors enabled. Enable at least one in the config.")
    return detectors