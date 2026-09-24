"""Visual screening and rendering components."""

from __future__ import annotations

from typing import Any

__all__ = ["VisualScreening", "SeriesRenderer"]


def __getattr__(name: str) -> Any:
    if name == "VisualScreening":
        from .screening import VisualScreening
        return VisualScreening
    if name == "SeriesRenderer":
        from .render import SeriesRenderer
        return SeriesRenderer
    raise AttributeError(name)
