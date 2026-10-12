"""Small data contracts shared by rendering, evidence and the official API."""

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class MediaRef:
    uri: str
    media_type: str = "image/png"
    description: str = ""
    detail: str = "high"


@dataclass(frozen=True)
class CandidateInterval:
    start: int
    end: int
    candidate_id: str = ""

    def __post_init__(self):
        if self.start < 0 or self.end < self.start:
            raise ValueError("invalid inclusive interval")

    @property
    def span(self):
        return self.end - self.start + 1

    def as_pair(self):
        return [self.start, self.end]


@dataclass
class CandidateGroup:
    group_id: str
    intervals: list[CandidateInterval]
    focus_start: int | None = None
    focus_end: int | None = None

    def __post_init__(self):
        if not self.intervals:
            raise ValueError("empty group")
        if self.focus_start is None:
            self.focus_start = min(x.start for x in self.intervals)
        if self.focus_end is None:
            self.focus_end = max(x.end for x in self.intervals)


@dataclass
class ToolResult:
    tool_name: str
    status: str
    summary: str
    evidence_family: str
    data: dict[str, Any] = field(default_factory=dict)
    media: list[MediaRef] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class ModelUsage:
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ModelStepResult:
    response_id: str
    parsed_action: dict[str, Any]
    usage: ModelUsage = field(default_factory=ModelUsage)
    raw_response: Any = None
    status: str = "completed"


def dataclass_to_jsonable(value):
    if is_dataclass(value):
        return dataclass_to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): dataclass_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [dataclass_to_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class SignalContext:
    """Only public fields may be serialized; adapter_payload stays private."""

    signal_id: str
    signal_length: int
    full_series_media: list[MediaRef]
    candidate_intervals: list[CandidateInterval]
    public_metadata: dict[str, Any]
    adapter_payload: Any = None


@dataclass
class GroupRuntimeState:
    group: CandidateGroup


@dataclass
class AgentSessionState:
    signal_id: str


class EvidenceToolAdapter(Protocol):
    def execute(
        self, tool_name: str, arguments: Mapping[str, Any], **scope
    ) -> ToolResult: ...


class AdjacentCandidateGrouping:
    def group(self, intervals: Sequence[CandidateInterval]) -> list[CandidateGroup]:
        return [CandidateGroup(f"G{i:04d}", [c]) for i, c in enumerate(intervals, 1)]
