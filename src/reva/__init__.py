"""Compatibility exports for the former public package name.

New code should import AgentVeriTSConfig and AgentVeriTSPipeline from agentverits.
"""
from __future__ import annotations

from typing import Any
import warnings

__all__ = ["REVAConfig", "REVAPipeline"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        warnings.warn("reva is deprecated; import agentverits instead", DeprecationWarning, stacklevel=2)
        import agentverits
        return getattr(agentverits, "AgentVeriTSConfig" if name == "REVAConfig" else "AgentVeriTSPipeline")
    raise AttributeError(name)
