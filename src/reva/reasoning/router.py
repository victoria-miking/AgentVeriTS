from __future__ import annotations

from dataclasses import dataclass

from ..types import GlobalDecision, GlobalHypothesis


@dataclass
class RoutingResult:
    direct: list[GlobalDecision]
    uncertain: list[GlobalDecision]


class UncertaintyRouter:
    """Route by confidence only. Action type is deliberately not a routing signal."""

    def __init__(self, confidence_threshold: float = 0.75) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0,1]")
        self.threshold = float(confidence_threshold)

    def route(self, hypothesis: GlobalHypothesis) -> RoutingResult:
        direct: list[GlobalDecision] = []
        uncertain: list[GlobalDecision] = []
        for decision in hypothesis.decisions:
            (uncertain if decision.confidence < self.threshold else direct).append(decision)
        return RoutingResult(direct=direct, uncertain=uncertain)
