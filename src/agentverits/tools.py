from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import EvidenceToolAdapter, SignalContext
from .contracts import ToolResult
from .contracts import AgentSessionState, GroupRuntimeState


@dataclass(frozen=True)
class ToolSpec:
    name: str
    evidence_family: str
    specialty: str
    when_to_use: str
    caution: str
    args_schema: dict[str, Any]
    estimated_cost: str = "low"

    def prompt_block(self) -> str:
        return (
            f"- {self.name}\n"
            f"  EVIDENCE FAMILY: {self.evidence_family}\n"
            f"  SPECIALTY: {self.specialty}\n"
            f"  WHEN: {self.when_to_use}\n"
            f"  CAUTION: {self.caution}\n"
            f"  ARGS: {self.args_schema}"
        )


class ToolRegistry:
    """Reasoning-facing catalog + repository-facing execution adapter."""

    def __init__(self, adapter: EvidenceToolAdapter) -> None:
        self.adapter = adapter
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate tool: {spec.name}")
        self._specs[spec.name] = spec

    def has(self, name: str) -> bool:
        return name in self._specs

    def spec(self, name: str) -> ToolSpec:
        if name not in self._specs:
            raise KeyError(f"unknown tool: {name}")
        return self._specs[name]

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def prompt_block(self) -> str:
        return "\n\n".join(spec.prompt_block() for spec in self.specs())

    def dispatch(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        context: SignalContext,
        session_state: AgentSessionState,
        group_state: GroupRuntimeState,
    ) -> ToolResult:
        self.spec(name)  # validate existence
        return self.adapter.execute(
            name,
            arguments,
            context=context,
            session_state=session_state,
            group_state=group_state,
        )


def build_v1_tool_specs() -> list[ToolSpec]:
    """Semantic contracts only. No implementation assumptions."""

    return [
        ToolSpec(
            name="get_references",
            evidence_family="reference",
            specialty="Retrieve highly similar historical reference candidates for the active group.",
            when_to_use="When historical recurrence or compatibility is the main unresolved question.",
            caution=(
                "Returned references are highly similar candidates, NOT certified normal examples. "
                "They may share the same anomaly as the query. Validate them with context before "
                "using them as normal evidence."
            ),
            args_schema={
                "scale": "optional str|int, default auto",
                "limit": "optional int",
                "offset": "optional int",
            },
        ),
        ToolSpec(
            name="compare_references_visually",
            evidence_family="reference_visual",
            specialty=(
                "Compare the query region and selected references with enough surrounding context "
                "to judge both similarity and possible shared-anomaly contamination."
            ),
            when_to_use=(
                "After references are available and visual/contextual validation can decide "
                "whether they are credible normal, contaminated, or inconclusive."
            ),
            caution=(
                "High query-reference similarity is not evidence of normality by itself. "
                "Inspect each reference's own temporal context."
            ),
            args_schema={
                "reference_ids": "list[str]",
                "context": "optional short|medium|long",
            },
            estimated_cost="medium",
        ),
        ToolSpec(
            name="inspect_raw_segment",
            evidence_family="raw_numeric",
            specialty="Read exact raw values for one evidence-backed local range.",
            when_to_use="When image resolution is insufficient for exact values or boundaries.",
            caution=(
                "Use short, evidence-backed ranges. Never request the whole series merely because "
                "it is available."
            ),
            args_schema={"start": "int", "end": "int"},
        ),
        ToolSpec(
            name="plot_focused_segment",
            evidence_family="visual",
            specialty="Render a higher-resolution local or wider-context time-series view.",
            when_to_use="When the full-series overview hides shape, transition, or boundary detail.",
            caution="Choose the smallest context that can resolve the current uncertainty.",
            args_schema={
                "start": "int",
                "end": "int",
            },
            estimated_cost="medium",
        ),
        ToolSpec(
            name="segment_statistics",
            evidence_family="statistical",
            specialty="Summarize local level, spread, range, quantiles, and trend against a baseline.",
            when_to_use="When magnitude, duration, spread, or trend consistency is uncertain.",
            caution="Statistics describe evidence; they do not decide anomaly labels.",
            args_schema={
                "start": "int",
                "end": "int",
            },
        ),
        ToolSpec(
            name="value_zscore",
            evidence_family="statistical",
            specialty="Measure absolute-value extremeness relative to a numerical baseline.",
            when_to_use="For spikes, dips, level excursions, or amplitude anomalies.",
            caution="A low z-score does not prove normality; contextual anomalies may remain.",
            args_schema={
                "start": "int",
                "end": "int",
            },
        ),
        ToolSpec(
            name="diff_zscore",
            evidence_family="statistical",
            specialty="Measure first-difference extremeness for abrupt changes.",
            when_to_use="For jumps, sharp recovery, spikes, dips, or transition anomalies.",
            caution="Interpret with context; legitimate transitions can also have large derivatives.",
            args_schema={
                "start": "int",
                "end": "int",
            },
        ),
        ToolSpec(
            name="get_screening_evidence",
            evidence_family="screening",
            specialty="Inspect Anomaly Screening evidence for the active group or a retrieved reference.",
            when_to_use=(
                "When the agent needs to know why Anomaly Screening considered a region suspicious, "
                "or whether multiple Anomaly Screening views/scales agree."
            ),
            caution=(
                "Anomaly Screening evidence is a second opinion, not ground truth. Do not let it override "
                "contradictory contextual or reference evidence automatically."
            ),
            args_schema={
                "target_type": "group|reference",
                "target_id": "str",
            },
        ),
    ]
