"""Detector factory: turns the config into a list of detector objects."""

from __future__ import annotations

from bench.config import DetectorsConfig
from bench.detectors.base import Detector, Verdict
from bench.detectors.harm_cls import HarmClsDetector
from bench.detectors.llama_guard3 import LlamaGuard3Detector

__all__ = ["Detector", "Verdict", "build_detectors"]


def build_detectors(config: DetectorsConfig) -> list[Detector]:
    detectors: list[Detector] = []

    if config.llama_guard3.enabled:
        detectors.append(LlamaGuard3Detector(config.llama_guard3))
    if config.harm_cls.enabled:
        detectors.append(HarmClsDetector(config.harm_cls))

    if not detectors:
        raise ValueError("No detectors enabled. Enable at least one in the config.")
    return detectors