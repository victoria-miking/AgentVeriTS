from __future__ import annotations

from dataclasses import dataclass

from ..types import GlobalDecision, GlobalHypothesis


@dataclass
class RoutingResult:
    direct: list[GlobalDecision]
    uncertain: list[GlobalDecision]


class UncertaintyRouter:
    """Fixed discrete routing: confidence 1/2 -> evidence agent; confidence 3 -> direct closure."""

    def route(self, hypothesis: GlobalHypothesis) -> RoutingResult:
        direct: list[GlobalDecision] = []
        uncertain: list[GlobalDecision] = []
        for decision in hypothesis.decisions:
            if int(decision.confidence) == 3:
                direct.append(decision)
            else:
                uncertain.append(decision)
        return RoutingResult(direct=direct, uncertain=uncertain)
