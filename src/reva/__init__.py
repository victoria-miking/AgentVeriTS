"""REVA public inference package."""

from __future__ import annotations

from typing import Any

__all__ = ["REVAConfig", "REVAPipeline"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name == "REVAConfig":
        from .config import REVAConfig
        return REVAConfig
    if name == "REVAPipeline":
        from .pipeline import REVAPipeline
        return REVAPipeline
    raise AttributeError(name)
