"""Compatibility facade for the public detection API."""

from .detector import GarbleDetector
from .ensemble import EnsembleDetector
from .registry import STRATEGY_MAP, Strategy

__all__ = ["GarbleDetector", "EnsembleDetector", "Strategy", "STRATEGY_MAP"]
