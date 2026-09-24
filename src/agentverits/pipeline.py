from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .config import AgentVeriTSConfig
from .io import write_json
from .reasoning.agentic_verification import AgenticVerification
from .reasoning.evidence_tools import EvidenceTools
from .reasoning.candidate_assessment import CandidateAssessment
from .reasoning.provider import OpenAIReasoningClient
from .reasoning.router import UncertaintyRouter
from .types import EvidenceDecision, GlobalDecision, Interval, AgentVeriTSResult
from .visual.render import SeriesRenderer
from .visual.screening import VisualScreening
import numpy as np


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


class AgentVeriTSPipeline:
    def __init__(
        self,
        config: AgentVeriTSConfig | None = None,
        *,
        screening: VisualScreening | None = None,
        reasoning_client: OpenAIReasoningClient | None = None,
    ) -> None:
        self.config = config or AgentVeriTSConfig()
        self.config.validate()
        self.screening = screening or VisualScreening(self.config.screening)
        self.reasoning_client = reasoning_client or OpenAIReasoningClient(
            model=self.config.reasoning.model,
            reasoning_effort=self.config.reasoning.reasoning_effort,
            max_output_tokens=self.config.reasoning.max_output_tokens,
            store=self.config.reasoning.store_responses,
            timeout_seconds=self.config.reasoning.timeout_seconds,
            max_retries=self.config.reasoning.max_retries,
        )
        self.renderer = SeriesRenderer(self.config.render)

    def run(self, values: Sequence[float], *, signal_id: str = "signal",
            output_dir: str | Path = "outputs/agentverits") -> AgentVeriTSResult:
        log = getattr(self.reasoning_client, "call_log", [])
        start = len(log)
        try:
            return self._run(values, signal_id=signal_id, output_dir=output_dir)
        finally:
            # Persist IDs and usage even when a later response fails. No keys or image payloads.
            write_json(Path(output_dir) / "api_calls.json", log[start:])

    def _run(
        self,
        values: Sequence[float],
        *,
        signal_id: str = "signal",
        output_dir: str | Path = "outputs/agentverits",
    ) -> AgentVeriTSResult:
        self.config.validate()
        values = list(float(x) for x in values)
        if not np.isfinite(values).all():
            raise ValueError("signal contains non-finite values; clean missing data before inference")
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
        global_hypothesis = CandidateAssessment(self.reasoning_client, self.config.reasoning.confidence_threshold).run(
            signal_id=signal_id,
            signal_length=len(values),
            candidates=screening_result.selected_candidates,
            global_image=str(global_image),
        )
        write_json(out / "global_hypothesis.json", global_hypothesis.as_dict())

        routed = UncertaintyRouter(self.config.reasoning.confidence_threshold).route(global_hypothesis)
        write_json(
            out / "routing.json",
            {
                "rule": "q < confidence_threshold -> agentic verification; q >= threshold -> direct",
                "confidence_threshold": self.config.reasoning.confidence_threshold,
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
        agent = AgenticVerification(
            self.reasoning_client,
            tools,
            max_tool_calls=self.config.reasoning.max_evidence_calls,
            confidence_threshold=self.config.reasoning.confidence_threshold,
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

        # Paper Eq. (8): direct high-confidence operations plus verified low-confidence outputs.
        # ADD remains available inside verification, with no unconditional extra global pass.
        agent_discoveries = [item for decision in evidence_decisions for item in decision.additions]

        final_intervals = _merge_intervals(
            provisional_intervals + [x.interval for x in agent_discoveries]
        )
        result = AgentVeriTSResult(
            final_intervals=final_intervals,
            global_hypothesis=global_hypothesis,
            evidence_decisions=evidence_decisions,
            agent_discoveries=agent_discoveries,
            selected_alpha=screening_result.selected_alpha,
            output_dir=str(out),
        )
        write_json(out / "result.json", result.as_dict())
        return result
