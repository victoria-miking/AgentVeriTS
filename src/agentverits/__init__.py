"""AgentVeriTS public inference package."""

from __future__ import annotations

from typing import Any

__all__ = ["AgentVeriTSConfig", "AgentVeriTSPipeline"]
__version__ = "0.2.0"


def __getattr__(name: str) -> Any:
    if name == "AgentVeriTSConfig":
        from .config import AgentVeriTSConfig
        return AgentVeriTSConfig
    if name == "AgentVeriTSPipeline":
        from .pipeline import AgentVeriTSPipeline
        return AgentVeriTSPipeline
    raise AttributeError(name)
