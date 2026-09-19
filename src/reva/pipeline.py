from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .config import REVAConfig
from .io import write_json
from .reasoning.evidence_agent import EvidenceAgent
from .reasoning.evidence_tools import EvidenceTools
from .reasoning.global_hypothesis import GlobalHypothesisBuilder
from .reasoning.provider import OpenAIReasoningClient
from .reasoning.router import UncertaintyRouter
from .types import EvidenceDecision, GlobalDecision, Interval, REVAResult
from .visual.render import SeriesRenderer
from .visual.screening import VisualScreening


def _merge_intervals(intervals: Sequence[Interval]) -> list[Interval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda x: (x.start, x.end))
    out: list[Interval] = []
    cur = ordered[0]
    for item in ordered[1:]:
        if item.start <= cur.end + 1:
            cur = Interval(cur.start, max(cur.end, item.end))
        else:
            out.append(cur)
            cur = item
    out.append(cur)
    return out


def _direct_interval(decision: GlobalDecision) -> Interval | None:
    if decision.action == "remove":
        return None
    return decision.final_interval or decision.reviewed_interval


class REVAPipeline:
    def __init__(
        self,
        config: REVAConfig | None = None,
        *,
        screening: VisualScreening | None = None,
        reasoning_client: OpenAIReasoningClient | None = None,
    ) -> None:
        self.config = config or REVAConfig()
        self.screening = screening or VisualScreening(self.config.screening)
        self.reasoning_client = reasoning_client or OpenAIReasoningClient(
            model=self.config.reasoning.model,
            reasoning_effort=self.config.reasoning.reasoning_effort,
            max_output_tokens=self.config.reasoning.max_output_tokens,
            store=self.config.reasoning.store_responses,
        )
        self.renderer = SeriesRenderer(self.config.render)

    def run(
        self,
        values: Sequence[float],
        *,
        signal_id: str = "signal",
        output_dir: str | Path = "outputs/reva",
    ) -> REVAResult:
        values = list(float(x) for x in values)
        if len(values) < max(self.config.screening.scales):
            raise ValueError(f"signal length must be >= {max(self.config.screening.scales)}")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        screening_result = self.screening.run(values)
        write_json(
            out / "screening.json",
            {
                "selected_alpha": screening_result.selected_alpha,
                "candidate_sets": {
                    alpha: [
                        {
                            "candidate_id": c.candidate_id,
                            "interval": c.interval.as_list(),
                            "alpha": c.alpha,
                            "score_peak": c.score_peak,
                        }
                        for c in rows
                    ]
                    for alpha, rows in screening_result.candidate_sets.items()
                },
                "reference_traces": {
                    str(scale): trace.as_dict()
                    for scale, trace in screening_result.reference_traces.items()
                },
            },
        )

        global_image = self.renderer.global_plot(
            values,
            screening_result.selected_candidates,
            out / "global_candidates.png",
        )
        global_hypothesis = GlobalHypothesisBuilder(self.reasoning_client).run(
            signal_id=signal_id,
            signal_length=len(values),
            candidates=screening_result.selected_candidates,
            global_image=str(global_image),
        )
        write_json(out / "global_hypothesis.json", global_hypothesis.as_dict())

        routed = UncertaintyRouter().route(global_hypothesis)
        write_json(
            out / "routing.json",
            {
                "rule": "confidence 1/2 -> evidence agent; confidence 3 -> direct closure",
                "direct": [x.decision_id for x in routed.direct],
                "uncertain": [x.decision_id for x in routed.uncertain],
            },
        )

        tools = EvidenceTools(
            values,
            out / "evidence",
            self.renderer,
            global_image=global_image,
            context_points=self.config.reasoning.context_points,
            reference_count=self.config.reasoning.reference_count,
            reference_traces=screening_result.reference_traces,
        )
        agent = EvidenceAgent(
            self.reasoning_client,
            tools,
            max_tool_calls=self.config.reasoning.max_evidence_calls,
            max_global_rescan_calls=self.config.reasoning.max_global_rescan_calls,
            signal_length=len(values),
        )

        evidence_decisions: list[EvidenceDecision] = []
        previous_response_id = global_hypothesis.response_id
        for uncertain in routed.uncertain:
            verified, previous_response_id = agent.verify(
                uncertain,
                global_hypothesis,
                previous_response_id=previous_response_id,
            )
            evidence_decisions.append(verified)
            write_json(out / "evidence" / f"{uncertain.decision_id}.json", verified.as_dict())

        provisional_intervals: list[Interval] = []
        for decision in routed.direct:
            interval = _direct_interval(decision)
            if interval is not None:
                provisional_intervals.append(interval)

        verified_by_id = {x.decision_id: x for x in evidence_decisions}
        for original in routed.uncertain:
            verified = verified_by_id[original.decision_id]
            if verified.final_action != "remove" and verified.final_interval is not None:
                provisional_intervals.append(verified.final_interval)

        provisional_intervals = _merge_intervals(provisional_intervals)

        agent_discoveries, previous_response_id = agent.global_rescan(
            global_hypothesis,
            provisional_intervals,
            previous_response_id=previous_response_id,
        )
        write_json(
            out / "evidence" / "global_rescan.json",
            {
                "discoveries": [x.as_dict() for x in agent_discoveries],
                "response_id": previous_response_id,
            },
        )

        final_intervals = _merge_intervals(
            provisional_intervals + [x.interval for x in agent_discoveries]
        )
        result = REVAResult(
            final_intervals=final_intervals,
            global_hypothesis=global_hypothesis,
            evidence_decisions=evidence_decisions,
            agent_discoveries=agent_discoveries,
            selected_alpha=screening_result.selected_alpha,
            output_dir=str(out),
        )
        write_json(out / "result.json", result.as_dict())
        return result
