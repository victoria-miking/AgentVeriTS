from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from numbers import Integral, Real
from typing import Any, Literal


Action = Literal["keep", "remove", "refine", "add"]
FinalAction = Action


def validate_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("confidence must be a finite number in [0,1]")
    return float(value)


def validate_record_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError("record ID must be a safe alphanumeric identifier (up to 80 characters)")
    return value


@dataclass(frozen=True)
class Interval:
    start: int
    end: int

    def __post_init__(self) -> None:
        if any(isinstance(x, bool) or not isinstance(x, Integral) for x in (self.start, self.end)):
            raise ValueError("interval endpoints must be integers")
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid interval [{self.start}, {self.end}]")

    def as_list(self) -> list[int]:
        return [int(self.start), int(self.end)]


@dataclass
class VisualCandidate:
    candidate_id: str
    interval: Interval
    alpha: float
    score_peak: float | None = None


@dataclass
class GlobalDecision:
    decision_id: str
    source: Literal["candidate", "added"]
    candidate_id: str | None
    reviewed_interval: Interval
    action: Action
    final_interval: Interval | None
    confidence: float
    rationale: str

    def __post_init__(self) -> None:
        self.confidence = validate_confidence(self.confidence)
        validate_record_id(self.decision_id)
        if self.action not in {"keep", "remove", "refine", "add"}:
            raise ValueError("unknown global action")

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["reviewed_interval"] = self.reviewed_interval.as_list()
        out["final_interval"] = self.final_interval.as_list() if self.final_interval else None
        out["confidence"] = self.confidence
        return out


@dataclass
class GlobalHypothesis:
    normal_pattern: str
    anomaly_pattern: str
    missed_anomaly_scan: str
    decisions: list[GlobalDecision]
    summary: str
    response_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "normal_pattern": self.normal_pattern,
            "anomaly_pattern": self.anomaly_pattern,
            "missed_anomaly_scan": self.missed_anomaly_scan,
            "decisions": [x.as_dict() for x in self.decisions],
            "summary": self.summary,
            "response_id": self.response_id,
        }


@dataclass
class EvidenceDecision:
    decision_id: str
    final_action: FinalAction
    final_interval: Interval | None
    confidence: float
    rationale: str
    evidence_log: list[dict[str, Any]] = field(default_factory=list)
    additions: list[AgentDiscovery] = field(default_factory=list)
    response_id: str | None = None

    def __post_init__(self) -> None:
        self.confidence = validate_confidence(self.confidence)
        validate_record_id(self.decision_id)
        if self.final_action not in {"keep", "remove", "refine", "add"}:
            raise ValueError("unknown evidence action")
        if (self.final_action == "remove") != (self.final_interval is None):
            raise ValueError("only remove may have a null final_interval")

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "final_action": self.final_action,
            "final_interval": self.final_interval.as_list() if self.final_interval else None,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "evidence_log": self.evidence_log,
            "additions": [x.as_dict() for x in self.additions],
            "response_id": self.response_id,
        }


@dataclass
class AgentDiscovery:
    discovery_id: str
    interval: Interval
    confidence: float
    rationale: str
    evidence_log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.confidence = validate_confidence(self.confidence)
        validate_record_id(self.discovery_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "discovery_id": self.discovery_id,
            "interval": self.interval.as_list(),
            "confidence": self.confidence,
            "rationale": self.rationale,
            "evidence_log": self.evidence_log,
        }


@dataclass
class ScaleReferenceTrace:
    scale: int
    window_starts: list[int]
    retained_reference_starts: list[list[int]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scale": int(self.scale),
            "window_starts": [int(x) for x in self.window_starts],
            "retained_reference_starts": [
                [int(x) for x in row] for row in self.retained_reference_starts
            ],
        }


@dataclass
class ScreeningResult:
    scores: list[float]
    candidate_sets: dict[str, list[VisualCandidate]]
    selected_alpha: float
    selected_candidates: list[VisualCandidate]
    reference_traces: dict[int, ScaleReferenceTrace] = field(default_factory=dict)


@dataclass
class AgentVeriTSResult:
    final_intervals: list[Interval]
    global_hypothesis: GlobalHypothesis
    evidence_decisions: list[EvidenceDecision]
    agent_discoveries: list[AgentDiscovery]
    selected_alpha: float
    output_dir: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "final_intervals": [x.as_list() for x in self.final_intervals],
            "selected_alpha": float(self.selected_alpha),
            "global_hypothesis": self.global_hypothesis.as_dict(),
            "evidence_decisions": [x.as_dict() for x in self.evidence_decisions],
            "agent_discoveries": [x.as_dict() for x in self.agent_discoveries],
            "output_dir": self.output_dir,
        }
