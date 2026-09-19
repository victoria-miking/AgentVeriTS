from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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


@dataclass
class ReasoningConfig:
    model: str = "gpt-5.6-sol"
    reasoning_effort: str = "medium"
    max_output_tokens: int = 5000
    max_evidence_calls: int = 3
    max_global_rescan_calls: int = 3
    context_points: int = 256
    reference_count: int = 4
    store_responses: bool = True


@dataclass
class RenderConfig:
    global_width_px: int = 4096
    global_height_px: int = 512
    dpi: int = 100
    candidate_color: str = "#d97706"
    line_color: str = "#334155"
    focus_color: str = "#2563eb"


@dataclass
class REVAConfig:
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "REVAConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        screening_raw = dict(raw.get("screening") or {})
        reasoning_raw = dict(raw.get("reasoning") or {})
        render_raw = dict(raw.get("render") or {})
        for key in ("scales", "alpha_candidates"):
            if key in screening_raw:
                screening_raw[key] = tuple(screening_raw[key])
        return cls(
            screening=ScreeningConfig(**screening_raw),
            reasoning=ReasoningConfig(**reasoning_raw),
            render=RenderConfig(**render_raw),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
