from __future__ import annotations

from typing import Any, Sequence

from ..types import AgentDiscovery, EvidenceDecision, GlobalDecision, GlobalHypothesis, Interval
from .evidence_tools import EvidenceTools
from .prompts import EVIDENCE_SYSTEM_PROMPT, evidence_tool_block
from .provider import OpenAIReasoningClient


def _interval_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "integer", "minimum": 0},
        "minItems": 2,
        "maxItems": 2,
    }


def evidence_schema() -> dict[str, Any]:
    interval = _interval_schema()
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "unresolved_question": {"type": "string"},
            "evidence_assessment": {"type": "string"},
            "action": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"type": "string", "enum": ["call_tool", "finalize"]},
                    "tool": {"type": ["string", "null"]},
                    "arguments": {"type": "object"},
                    "final_action": {"type": ["string", "null"], "enum": ["keep", "remove", "refine", None]},
                    "final_interval": {"anyOf": [{"type": "null"}, interval]},
                    "confidence": {"type": ["integer", "null"], "enum": [1, 2, 3, None]},
                    "rationale": {"type": "string"},
                },
                "required": ["type", "tool", "arguments", "final_action", "final_interval", "confidence", "rationale"],
            },
        },
        "required": ["unresolved_question", "evidence_assessment", "action"],
    }


def global_rescan_schema() -> dict[str, Any]:
    interval = _interval_schema()
    discovery = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "discovery_id": {"type": "string"},
            "interval": interval,
            "confidence": {"type": "integer", "enum": [2, 3]},
            "rationale": {"type": "string"},
        },
        "required": ["discovery_id", "interval", "confidence", "rationale"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "search_assessment": {"type": "string"},
            "action": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"type": "string", "enum": ["call_tool", "finalize"]},
                    "tool": {"type": ["string", "null"]},
                    "arguments": {"type": "object"},
                    "target_interval": {"anyOf": [{"type": "null"}, interval]},
                    "discoveries": {"type": "array", "items": discovery},
                },
                "required": ["type", "tool", "arguments", "target_interval", "discoveries"],
            },
        },
        "required": ["search_assessment", "action"],
    }


class EvidenceAgent:
    def __init__(
        self,
        client: OpenAIReasoningClient,
        tools: EvidenceTools,
        *,
        max_tool_calls: int = 3,
        max_global_rescan_calls: int = 3,
        signal_length: int,
    ) -> None:
        self.client = client
        self.tools = tools
        self.max_tool_calls = int(max_tool_calls)
        self.max_global_rescan_calls = int(max_global_rescan_calls)
        self.signal_length = int(signal_length)

    @staticmethod
    def _target_text(decision: GlobalDecision, hypothesis: GlobalHypothesis) -> str:
        return (
            f"GLOBAL NORMAL PATTERN:\n{hypothesis.normal_pattern}\n\n"
            f"GLOBAL ANOMALY PATTERN:\n{hypothesis.anomaly_pattern}\n\n"
            f"TARGET UNCERTAINTY:\n{decision.as_dict()}\n\n"
            f"{evidence_tool_block()}\n\n"
            "Resolve this confidence-1/2 decision first. Its current action is a hypothesis, not a command. "
            "The signal-level session will later perform a separate evidence-aware global rescan."
        )

    def verify(
        self,
        decision: GlobalDecision,
        hypothesis: GlobalHypothesis,
        *,
        previous_response_id: str | None,
    ) -> tuple[EvidenceDecision, str | None]:
        response_id = previous_response_id
        text = self._target_text(decision, hypothesis)
        images: list[str] = []
        evidence_log: list[dict[str, Any]] = []
        tool_calls = 0
        for _ in range(self.max_tool_calls + 4):
            reply = self.client.complete(
                instructions=EVIDENCE_SYSTEM_PROMPT,
                text=text,
                schema=evidence_schema(),
                schema_name="reva_evidence_action",
                images=images,
                previous_response_id=response_id,
            )
            response_id = reply.response_id
            row = reply.payload
            action = row["action"]
            if action["type"] == "call_tool" and tool_calls < self.max_tool_calls:
                tool = str(action["tool"] or "")
                obs = self.tools.execute(tool, decision, dict(action.get("arguments") or {}))
                tool_calls += 1
                evidence_log.append({"tool": tool, "summary": obs.summary, "data": obs.data})
                text = (
                    f"Evidence observation {tool_calls}/{self.max_tool_calls}:\n{obs.as_prompt_text()}\n\n"
                    "Update the uncertainty using this evidence. Request one more tool only if it can materially change the conclusion."
                )
                images = obs.images
                continue
            if action["type"] == "call_tool":
                text = "Evidence budget is exhausted. Finalize this target now; do not request another tool."
                images = []
                continue
            final_action = str(action["final_action"])
            final_interval_raw = action["final_interval"]
            final_interval = None if final_interval_raw is None else Interval(int(final_interval_raw[0]), int(final_interval_raw[1]))
            confidence_raw = action["confidence"]
            if type(confidence_raw) is not int or confidence_raw not in {1, 2, 3}:
                raise ValueError("evidence confidence must be an integer in {1,2,3}")
            confidence = confidence_raw
            if final_interval is not None and final_interval.end >= self.signal_length:
                raise ValueError("evidence decision exceeds signal bounds")
            if final_action == "remove" and final_interval is not None:
                raise ValueError("remove requires final_interval=null")
            if final_action in {"keep", "refine"} and final_interval is None:
                if final_action == "keep":
                    final_interval = decision.final_interval or decision.reviewed_interval
                else:
                    raise ValueError("refine requires final_interval")
            return EvidenceDecision(
                decision_id=decision.decision_id,
                final_action=final_action,  # type: ignore[arg-type]
                final_interval=final_interval,
                confidence=confidence,  # type: ignore[arg-type]
                rationale=str(action["rationale"]),
                evidence_log=evidence_log,
            ), response_id
        raise RuntimeError("evidence agent failed to finalize")

    def global_rescan(
        self,
        hypothesis: GlobalHypothesis,
        accepted_intervals: Sequence[Interval],
        *,
        previous_response_id: str | None,
    ) -> tuple[list[AgentDiscovery], str | None]:
        response_id = previous_response_id
        evidence_log: list[dict[str, Any]] = []
        text = (
            f"GLOBAL NORMAL PATTERN:\n{hypothesis.normal_pattern}\n\n"
            f"CURRENT ANOMALY PATTERN:\n{hypothesis.anomaly_pattern}\n\n"
            f"CURRENT ACCEPTED/VERIFIED INTERVALS:\n{[x.as_list() for x in accepted_intervals]}\n\n"
            f"{evidence_tool_block()}\n\n"
            "Perform an evidence-aware GLOBAL RESCAN of the entire series for anomalies still missed by the "
            "visual screening and first global hypothesis. Do not simply repeat already accepted intervals. "
            "When a new suspicious region needs evidence, call one tool with target_interval=[start,end]. "
            "At finalization, return only genuinely missed intervals supported at confidence 2 or 3."
        )
        images: list[str] = []
        tool_calls = 0
        for _ in range(self.max_global_rescan_calls + 4):
            reply = self.client.complete(
                instructions=EVIDENCE_SYSTEM_PROMPT,
                text=text,
                schema=global_rescan_schema(),
                schema_name="reva_evidence_global_rescan",
                images=images,
                previous_response_id=response_id,
            )
            response_id = reply.response_id
            action = reply.payload["action"]
            if action["type"] == "call_tool" and tool_calls < self.max_global_rescan_calls:
                raw_target = action.get("target_interval")
                target = None if raw_target is None else Interval(int(raw_target[0]), int(raw_target[1]))
                if target is not None and target.end >= self.signal_length:
                    raise ValueError("global rescan target exceeds signal bounds")
                tool = str(action["tool"] or "")
                obs = self.tools.execute_for_interval(
                    tool,
                    target,
                    dict(action.get("arguments") or {}),
                    decision_id=f"GLOBAL_RESCAN_{tool_calls + 1}",
                )
                tool_calls += 1
                evidence_log.append({
                    "tool": tool,
                    "target_interval": None if target is None else target.as_list(),
                    "summary": obs.summary,
                    "data": obs.data,
                })
                text = (
                    f"Global-rescan evidence {tool_calls}/{self.max_global_rescan_calls}:\n{obs.as_prompt_text()}\n\n"
                    "Continue the global search. Request another tool only if it can materially resolve a still-missed region."
                )
                images = obs.images
                continue
            if action["type"] == "call_tool":
                text = (
                    "Global-rescan evidence budget is exhausted. Finalize the global rescan now. "
                    "Do not output confidence-1 suspicions as discoveries."
                )
                images = []
                continue
            discoveries: list[AgentDiscovery] = []
            seen_ids: set[str] = set()
            for row in action["discoveries"]:
                discovery_id = str(row["discovery_id"])
                if discovery_id in seen_ids:
                    raise ValueError(f"duplicate discovery_id: {discovery_id}")
                seen_ids.add(discovery_id)
                interval = Interval(int(row["interval"][0]), int(row["interval"][1]))
                if interval.end >= self.signal_length:
                    raise ValueError("global rescan discovery exceeds signal bounds")
                confidence_raw = row["confidence"]
                if type(confidence_raw) is not int or confidence_raw not in {2, 3}:
                    raise ValueError("global-rescan discoveries require integer confidence 2 or 3")
                confidence = confidence_raw
                discoveries.append(AgentDiscovery(
                    discovery_id=discovery_id,
                    interval=interval,
                    confidence=confidence,  # type: ignore[arg-type]
                    rationale=str(row["rationale"]),
                    evidence_log=list(evidence_log),
                ))
            return discoveries, response_id
        raise RuntimeError("evidence-aware global rescan failed to finalize")
