from __future__ import annotations

from typing import Any

from ..types import EvidenceDecision, GlobalDecision, GlobalHypothesis, Interval
from .evidence_tools import EvidenceTools
from .prompts import EVIDENCE_SYSTEM_PROMPT, evidence_tool_block
from .provider import OpenAIReasoningClient


def evidence_schema() -> dict[str, Any]:
    interval = {
        "type": "array", "items": {"type": "integer", "minimum": 0},
        "minItems": 2, "maxItems": 2,
    }
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
                    "confidence": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
                    "rationale": {"type": "string"},
                },
                "required": ["type", "tool", "arguments", "final_action", "final_interval", "confidence", "rationale"],
            },
        },
        "required": ["unresolved_question", "evidence_assessment", "action"],
    }


class EvidenceAgent:
    def __init__(
        self,
        client: OpenAIReasoningClient,
        tools: EvidenceTools,
        *,
        max_tool_calls: int = 3,
        signal_length: int,
    ) -> None:
        self.client = client
        self.tools = tools
        self.max_tool_calls = int(max_tool_calls)
        self.signal_length = int(signal_length)

    @staticmethod
    def _target_text(decision: GlobalDecision, hypothesis: GlobalHypothesis) -> str:
        return (
            f"GLOBAL NORMAL PATTERN:\n{hypothesis.normal_pattern}\n\n"
            f"GLOBAL ANOMALY PATTERN:\n{hypothesis.anomaly_pattern}\n\n"
            f"TARGET UNCERTAINTY:\n{decision.as_dict()}\n\n"
            f"{evidence_tool_block()}\n\n"
            "Resolve only this uncertain decision. The target's current action is a hypothesis, not a command."
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
        for turn in range(self.max_tool_calls + 4):
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
                text = "Evidence budget is exhausted. Finalize now; do not request another tool."
                images = []
                continue
            final_action = str(action["final_action"])
            final_interval_raw = action["final_interval"]
            final_interval = None if final_interval_raw is None else Interval(int(final_interval_raw[0]), int(final_interval_raw[1]))
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
                confidence=float(action["confidence"]),
                rationale=str(action["rationale"]),
                evidence_log=evidence_log,
            ), response_id
        raise RuntimeError("evidence agent failed to finalize")
