from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Action = Literal["keep", "remove", "refine", "add"]
FinalAction = Literal["keep", "remove", "refine"]
ConfidenceLevel = Literal[1, 2, 3]


@dataclass(frozen=True)
class Interval:
    start: int
    end: int

    def __post_init__(self) -> None:
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
    confidence: ConfidenceLevel
    rationale: str

    def __post_init__(self) -> None:
        if int(self.confidence) not in {1, 2, 3}:
            raise ValueError("confidence must be one of {1,2,3}")

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["reviewed_interval"] = self.reviewed_interval.as_list()
        out["final_interval"] = self.final_interval.as_list() if self.final_interval else None
        out["confidence"] = int(self.confidence)
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
    confidence: ConfidenceLevel
    rationale: str
    evidence_log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if int(self.confidence) not in {1, 2, 3}:
            raise ValueError("confidence must be one of {1,2,3}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "final_action": self.final_action,
            "final_interval": self.final_interval.as_list() if self.final_interval else None,
            "confidence": int(self.confidence),
            "rationale": self.rationale,
            "evidence_log": self.evidence_log,
        }


@dataclass
class AgentDiscovery:
    discovery_id: str
    interval: Interval
    confidence: ConfidenceLevel
    rationale: str
    evidence_log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if int(self.confidence) not in {2, 3}:
            raise ValueError("final global-rescan discoveries must have confidence 2 or 3")

    def as_dict(self) -> dict[str, Any]:
        return {
            "discovery_id": self.discovery_id,
            "interval": self.interval.as_list(),
            "confidence": int(self.confidence),
            "rationale": self.rationale,
            "evidence_log": self.evidence_log,
        }


@dataclass
class ScreeningResult:
    scores: list[float]
    candidate_sets: dict[str, list[VisualCandidate]]
    selected_alpha: float
    selected_candidates: list[VisualCandidate]


@dataclass
class REVAResult:
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
