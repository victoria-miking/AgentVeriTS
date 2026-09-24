from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import math

import yaml


@dataclass
class ScreeningConfig:
    scales: tuple[int, ...] = (224, 448, 672)
    step_ratio: float = 4.0
    image_size: int = 224
    patch_size: int = 16
    top_k: int = 16
    retained_references: int = 4
    aggregate_top_fraction: float = 0.25
    alpha: float = 0.01
    alpha_candidates: tuple[float, ...] = (0.1, 0.01, 0.001)
    smoothing: bool = True
    encoder_name: str = "ViT-B-16"
    encoder_weights: str = "openai"
    batch_size: int = 20
    device: str = "auto"

    def validate(self) -> None:
        if not self.scales or any(type(x) is not int or x < 2 for x in self.scales):
            raise ValueError("scales must contain positive integer window lengths >= 2")
        if not math.isfinite(self.step_ratio) or self.step_ratio < 1 or self.step_ratio > min(self.scales):
            raise ValueError("step_ratio must be between 1 and the smallest scale")
        if not 0 < self.aggregate_top_fraction <= 1:
            raise ValueError("aggregate_top_fraction must be in (0,1]")
        if any(not 0 < float(x) < 1 for x in (self.alpha, *self.alpha_candidates)):
            raise ValueError("all alpha values must be in (0,1)")
        if any(type(x) is not int or x < 1 for x in (self.batch_size, self.top_k, self.retained_references)):
            raise ValueError("batch size and reference counts must be positive integers")
        if self.image_size != 224 or self.patch_size != 16 or self.encoder_name != "ViT-B-16":
            raise ValueError("the published screening configuration requires ViT-B-16 at 224x224")


@dataclass
class ReasoningConfig:
    model: str = "gpt-5.6-sol"
    reasoning_effort: str = "medium"
    max_output_tokens: int = 5000
    max_evidence_calls: int = 3
    confidence_threshold: float = 0.95
    context_points: int = 256
    reference_count: int = 4
    store_responses: bool = True
    timeout_seconds: float = 120.0
    max_retries: int = 2

    def validate(self) -> None:
        if not 0 < self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in (0,1]")
        if type(self.max_evidence_calls) is not int or self.max_evidence_calls < 0:
            raise ValueError("max_evidence_calls must be a nonnegative integer")
        if not self.store_responses:
            raise ValueError("previous_response_id continuity requires store_responses=true")
        if self.context_points < 0 or self.reference_count < 1:
            raise ValueError("invalid evidence context or reference count")
        if self.max_output_tokens < 1 or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0 or self.max_retries < 0:
            raise ValueError("invalid API token, timeout or retry budget")


@dataclass
class RenderConfig:
    global_width_px: int = 4096
    global_height_px: int = 512
    dpi: int = 100
    candidate_color: str = "#d97706"
    line_color: str = "#334155"
    focus_color: str = "#2563eb"


@dataclass
class AgentVeriTSConfig:
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    def validate(self) -> None:
        self.screening.validate()
        self.reasoning.validate()
        if min(self.render.global_width_px, self.render.global_height_px, self.render.dpi) <= 0:
            raise ValueError("render dimensions and dpi must be positive")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentVeriTSConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        screening_raw = dict(raw.get("screening") or {})
        reasoning_raw = dict(raw.get("reasoning") or {})
        render_raw = dict(raw.get("render") or {})
        for key in ("scales", "alpha_candidates"):
            if key in screening_raw:
                screening_raw[key] = tuple(screening_raw[key])
        config = cls(
            screening=ScreeningConfig(**screening_raw),
            reasoning=ReasoningConfig(**reasoning_raw),
            render=RenderConfig(**render_raw),
        )
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
