from __future__ import annotations

from typing import Any, Sequence

from ..types import GlobalDecision, GlobalHypothesis, Interval, VisualCandidate
from .prompts import GLOBAL_SYSTEM_PROMPT
from .provider import OpenAIReasoningClient


def global_schema() -> dict[str, Any]:
    interval = {
        "type": "array",
        "items": {"type": "integer", "minimum": 0},
        "minItems": 2,
        "maxItems": 2,
    }
    nullable_interval = {"anyOf": [{"type": "null"}, interval]}
    item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision_id": {"type": "string"},
            "source": {"type": "string", "enum": ["candidate", "added"]},
            "candidate_id": {"type": ["string", "null"]},
            "reviewed_interval": interval,
            "action": {"type": "string", "enum": ["keep", "remove", "refine", "add"]},
            "final_interval": nullable_interval,
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "string"},
        },
        "required": [
            "decision_id", "source", "candidate_id", "reviewed_interval", "action",
            "final_interval", "confidence", "rationale",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "normal_pattern": {"type": "string"},
            "anomaly_pattern": {"type": "string"},
            "missed_anomaly_scan": {"type": "string"},
            "decisions": {"type": "array", "items": item},
            "summary": {"type": "string"},
        },
        "required": ["normal_pattern", "anomaly_pattern", "missed_anomaly_scan", "decisions", "summary"],
    }


def _interval(value: Sequence[int]) -> Interval:
    if len(value) != 2:
        raise ValueError("interval must have two endpoints")
    return Interval(int(value[0]), int(value[1]))


class GlobalHypothesisBuilder:
    def __init__(self, client: OpenAIReasoningClient) -> None:
        self.client = client

    def run(
        self,
        *,
        signal_id: str,
        signal_length: int,
        candidates: Sequence[VisualCandidate],
        global_image: str,
    ) -> GlobalHypothesis:
        candidate_rows = [
            {
                "candidate_id": c.candidate_id,
                "interval": c.interval.as_list(),
                "screening_alpha": c.alpha,
            }
            for c in candidates
        ]
        text = (
            f"Signal ID: {signal_id}\nSignal length: {signal_length}\n"
            f"Orange visual candidates: {candidate_rows}\n\n"
            "Return one decision for every listed candidate and add any highly suspicious missed region. "
            "First characterize normal behavior and the likely anomaly morphology of this sequence."
        )
        reply = self.client.complete(
            instructions=GLOBAL_SYSTEM_PROMPT,
            text=text,
            schema=global_schema(),
            schema_name="reva_global_hypothesis",
            images=[global_image],
        )
        payload = reply.payload
        expected = {c.candidate_id: c for c in candidates}
        seen: set[str] = set()
        decision_ids: set[str] = set()
        parsed: list[GlobalDecision] = []
        for row in payload["decisions"]:
            decision_id = str(row["decision_id"])
            if decision_id in decision_ids:
                raise ValueError(f"duplicate decision_id: {decision_id}")
            decision_ids.add(decision_id)
            source = str(row["source"])
            action = str(row["action"])
            candidate_id = row.get("candidate_id")
            reviewed = _interval(row["reviewed_interval"])
            final = None if row["final_interval"] is None else _interval(row["final_interval"])
            confidence = float(row["confidence"])
            if reviewed.end >= signal_length or (final is not None and final.end >= signal_length):
                raise ValueError("global decision exceeds signal bounds")
            if source == "candidate":
                if candidate_id not in expected:
                    raise ValueError(f"unknown candidate_id: {candidate_id}")
                if candidate_id in seen:
                    raise ValueError(f"candidate returned more than once: {candidate_id}")
                seen.add(str(candidate_id))
                original = expected[str(candidate_id)].interval
                if reviewed != original:
                    raise ValueError(f"reviewed_interval changed for {candidate_id}; use final_interval for refinement")
                if action not in {"keep", "remove", "refine"}:
                    raise ValueError("original candidates may only use keep/remove/refine")
                if action == "remove" and final is not None:
                    raise ValueError("remove requires final_interval=null")
                if action == "keep" and final != original:
                    raise ValueError("keep requires unchanged final_interval")
                if action == "refine" and final is None:
                    raise ValueError("refine requires final_interval")
            else:
                if source != "added" or action != "add" or candidate_id is not None or final is None:
                    raise ValueError("added records require source=added, action=add, candidate_id=null and final_interval")
            parsed.append(GlobalDecision(
                decision_id=decision_id,
                source=source,  # type: ignore[arg-type]
                candidate_id=None if candidate_id is None else str(candidate_id),
                reviewed_interval=reviewed,
                action=action,  # type: ignore[arg-type]
                final_interval=final,
                confidence=confidence,
                rationale=str(row["rationale"]),
            ))
        missing = sorted(set(expected) - seen)
        if missing:
            raise ValueError(f"global model omitted candidates: {missing}")
        return GlobalHypothesis(
            normal_pattern=str(payload["normal_pattern"]),
            anomaly_pattern=str(payload["anomaly_pattern"]),
            missed_anomaly_scan=str(payload["missed_anomaly_scan"]),
            decisions=parsed,
            summary=str(payload["summary"]),
            response_id=reply.response_id,
        )
