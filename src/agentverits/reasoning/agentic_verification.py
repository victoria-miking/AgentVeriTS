from __future__ import annotations

import json
from typing import Any

from ..types import AgentDiscovery, EvidenceDecision, GlobalDecision, GlobalHypothesis, Interval, validate_confidence
from .evidence_tools import EvidenceTools
from .prompts import EVIDENCE_SYSTEM_PROMPT, evidence_tool_block
from .provider import OpenAIReasoningClient


def _interval_schema() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 2, "maxItems": 2}


def tool_arguments_schema() -> dict[str, Any]:
    """Closed, nullable fields satisfy the official strict JSON-schema contract."""
    fields = {
        "interval": {"anyOf": [{"type": "null"}, _interval_schema()]},
        "context_points": {"type": ["integer", "null"], "minimum": 0},
        "mode": {"type": ["string", "null"], "enum": ["local_y", "global_y", None]},
        "count": {"type": ["integer", "null"], "minimum": 1, "maximum": 16},
        "scale": {"type": ["integer", "null"], "minimum": 2},
        "shared_y": {"type": ["boolean", "null"]},
        "baseline": {"type": ["string", "null"], "enum": ["surrounding", "left", "right", "global", None]},
        "diagnostic": {"type": ["string", "null"], "enum": ["robust", "spike", None]},
        "max_points": {"type": ["integer", "null"], "minimum": 1, "maximum": 2048},
        "page_start": {"type": ["integer", "null"], "minimum": 0},
    }
    return {"type": "object", "additionalProperties": False, "properties": fields, "required": list(fields)}


def evidence_schema(*, allow_tools: bool = True) -> dict[str, Any]:
    interval = _interval_schema()
    confidence = {"type": "number", "minimum": 0, "maximum": 1}
    discovery_fields = {
        "discovery_id": {"type": "string"}, "interval": interval,
        "confidence": confidence, "rationale": {"type": "string"},
    }
    fields = {
        "type": {"type": "string", "enum": ["call_tool", "finalize"] if allow_tools else ["finalize"]},
        "tool": {"type": ["string", "null"], "enum": ["raw", "focus", "reference", "statistics", None]},
        "arguments": {"anyOf": [{"type": "null"}, tool_arguments_schema()]},
        "final_action": {"type": ["string", "null"], "enum": ["keep", "remove", "refine", "add", None]},
        "final_interval": {"anyOf": [{"type": "null"}, interval]},
        "confidence": {"anyOf": [{"type": "null"}, confidence]},
        "rationale": {"type": "string"},
        "additions": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": discovery_fields, "required": list(discovery_fields),
        }},
    }
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "unresolved_question": {"type": "string"},
            "evidence_assessment": {"type": "string"},
            "action": {"type": "object", "additionalProperties": False, "properties": fields, "required": list(fields)},
        },
        "required": ["unresolved_question", "evidence_assessment", "action"],
    }


class AgenticVerification:
    def __init__(self, client: OpenAIReasoningClient, tools: EvidenceTools, *, max_tool_calls: int = 3,
                 signal_length: int, confidence_threshold: float = 0.95) -> None:
        if type(max_tool_calls) is not int or max_tool_calls < 0:
            raise ValueError("max_tool_calls must be a nonnegative integer")
        self.client, self.tools = client, tools
        self.max_tool_calls = max_tool_calls
        self.signal_length = int(signal_length)
        self.confidence_threshold = validate_confidence(confidence_threshold)

    def _interval(self, value: Any) -> Interval:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("interval must have two integer endpoints")
        result = Interval(value[0], value[1])
        if result.end >= self.signal_length:
            raise ValueError("evidence interval exceeds signal bounds")
        return result

    def verify(self, decision: GlobalDecision, hypothesis: GlobalHypothesis, *,
               previous_response_id: str | None) -> tuple[EvidenceDecision, str]:
        if not previous_response_id:
            raise ValueError("agentic verification requires the preceding response id")
        if decision.confidence >= self.confidence_threshold:
            raise ValueError("high-confidence decisions bypass agentic verification")
        response_id = previous_response_id
        text = (
            f"Signal length: {self.signal_length}; valid indices: 0..{self.signal_length - 1}.\n"
            f"GLOBAL NORMAL PATTERN: {hypothesis.normal_pattern}\n"
            f"GLOBAL ANOMALY PATTERN: {hypothesis.anomaly_pattern}\n"
            f"TARGET LOW-CONFIDENCE DECISION:\n{json.dumps(decision.as_dict())}\n"
            f"{evidence_tool_block()}\n"
            f"You have up to {self.max_tool_calls} evidence requests. Resolve this decision using the "
            "global plot and hypothesis already in this response chain. You may inspect other suspected "
            "regions and ADD missed anomalies during verification. High-confidence decisions bypass "
            "this module and must not be edited. Return new regions in additions; do not repeat existing hypotheses."
        )
        images: list[str] = []
        evidence_log: list[dict[str, Any]] = []
        for turn in range(self.max_tool_calls + 1):
            allow_tools = turn < self.max_tool_calls
            if not allow_tools:
                text += "\nThe tool budget is exhausted. Finalize with the evidence collected so far."
            reply = self.client.complete(
                instructions=EVIDENCE_SYSTEM_PROMPT, text=text,
                schema=evidence_schema(allow_tools=allow_tools), schema_name="agentverits_verification",
                images=images, previous_response_id=response_id,
            )
            response_id = reply.response_id
            if not response_id:
                raise RuntimeError("missing response id during verification")
            action = reply.payload["action"]
            if action["type"] == "call_tool":
                if not allow_tools:
                    raise ValueError("model requested a tool after the budget was exhausted")
                tool = action["tool"]
                args = {k: v for k, v in (action.get("arguments") or {}).items() if v is not None}
                try:
                    obs = self.tools.execute(tool, decision, args)
                except (ValueError, TypeError) as exc:
                    # Bad model arguments consume a turn and can be repaired in the same chain.
                    evidence_log.append({"tool": tool, "arguments": args, "error": str(exc), "response_id": response_id})
                    text, images = f"Tool request rejected: {exc}. Correct the parameters or finalize.", []
                    continue
                evidence_log.append({"tool": tool, "arguments": args, "summary": obs.summary,
                                     "data": obs.data, "images": obs.images, "response_id": response_id})
                text = f"Evidence observation {turn + 1}/{self.max_tool_calls}:\n{obs.as_prompt_text()}\nUpdate the decision with this evidence."
                images = obs.images
                continue
            if action["type"] != "finalize":
                raise ValueError("unknown verification action")
            final_action = action["final_action"]
            if final_action not in {"keep", "remove", "refine", "add"}:
                raise ValueError("unknown final interval operation")
            if final_action == "add" and decision.source != "added":
                raise ValueError("use additions for newly discovered intervals; an existing candidate cannot become add")
            interval = None if action["final_interval"] is None else self._interval(action["final_interval"])
            if final_action == "keep" and interval != (decision.final_interval or decision.reviewed_interval):
                raise ValueError("keep must preserve the current target boundaries; use refine to change them")
            additions: list[AgentDiscovery] = []
            seen: set[str] = set()
            for row in action["additions"]:
                discovery_id = row["discovery_id"]
                if discovery_id in seen:
                    raise ValueError("duplicate discovery_id")
                seen.add(discovery_id)
                additions.append(AgentDiscovery(discovery_id, self._interval(row["interval"]),
                                                 validate_confidence(row["confidence"]), str(row["rationale"]), list(evidence_log)))
            return EvidenceDecision(
                decision_id=decision.decision_id, final_action=final_action, final_interval=interval,
                confidence=validate_confidence(action["confidence"]), rationale=str(action["rationale"]),
                evidence_log=evidence_log, additions=additions, response_id=response_id,
            ), response_id
        raise RuntimeError("verification failed to finalize")
