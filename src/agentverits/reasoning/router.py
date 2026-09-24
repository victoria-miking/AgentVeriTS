from __future__ import annotations

from dataclasses import dataclass

from ..types import GlobalDecision, GlobalHypothesis, validate_confidence


@dataclass
class RoutingResult:
    direct: list[GlobalDecision]
    uncertain: list[GlobalDecision]


class UncertaintyRouter:
    """Paper Eq. (5): only q < tau enters agentic verification."""

    def __init__(self, confidence_threshold: float = 0.95) -> None:
        self.threshold = validate_confidence(confidence_threshold)
        if self.threshold == 0:
            raise ValueError("confidence_threshold must be positive")

    def route(self, hypothesis: GlobalHypothesis) -> RoutingResult:
        direct: list[GlobalDecision] = []
        uncertain: list[GlobalDecision] = []
        for decision in hypothesis.decisions:
            if decision.confidence >= self.threshold:
                direct.append(decision)
            else:
                uncertain.append(decision)
        return RoutingResult(direct=direct, uncertain=uncertain)
