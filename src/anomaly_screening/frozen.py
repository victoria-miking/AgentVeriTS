"""Frozen Raw4 policy contracts and non-trainable three-scale pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence

from .retrieval import score_retained_patch_bank
import numpy as np
from models.decision import compute_detection_intervals, robust_positive_z


@dataclass
class SimilarityOutput:
    """Patch-level anomaly evidence for one query window or batch."""

    patch_scores: Any
    confidence: Any = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionOutput:
    """Normalized scale scores and their fused continuous score."""

    normalized_scale_scores: dict[int, Any]
    weights: dict[int, Any]
    fused_score: Any
    confidence: Any = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionOutput:
    """Thresholding result, preserving the processed score vector."""

    threshold: Any
    processed_scores: Any
    anomaly_flags: Any
    intervals: list[tuple[int, int]]
    diagnostics: dict[str, Any] = field(default_factory=dict)


class SimilarityPolicy(Protocol):
    def score(
        self,
        query_tokens: Any,
        retained_reference_tokens: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> SimilarityOutput:
        """Compare query patches with the retained-reference patch bank."""


class MultiScaleFusionPolicy(Protocol):
    def fuse(
        self,
        scale_scores: Mapping[int, Any],
        *,
        weights: Mapping[int, Any] | Sequence[float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> FusionOutput:
        """Normalize and combine per-scale continuous scores."""


class DecisionPolicy(Protocol):
    def decide(
        self,
        fused_score: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> DecisionOutput:
        """Convert a continuous score vector to flags and intervals."""


@dataclass
class ScreeningState:
    dataset: str = ""
    signal_id: str = ""
    scale: Optional[int] = None

    query_window_ids: Any = None
    valid_reference_ids: Any = None
    selected_reference_ids: Any = None
    retained_reference_ids: Any = None

    query_patch_tokens: Any = None
    retained_reference_tokens: Any = None

    patch_scores: Any = None
    scale_scores: Any = None
    normalized_scale_scores: Any = None

    fusion_weights: Any = None
    fused_score: Any = None

    threshold: Any = None
    processed_scores: Any = None
    anomaly_flags: Any = None
    detection_intervals: Any = None

    diagnostics: dict[str, Any] = field(default_factory=dict)


class BaselineSimilarityPolicy:
    """Frozen unaggregated patch-bank scoring policy.

    The policy accepts one query window (``[P,D]``) and reference windows
    (``[K,P,D]``).  A batch form (``[B,P,D]`` / ``[B,K,P,D]``) is supported by
    invoking the same scorer once per batch item; the production runner keeps
    its existing batched path unchanged.
    """

    name = "BaselineSimilarityPolicy"

    def __init__(self, *, retained_count: int = 4, trim_ratio: float = 0.75) -> None:
        if int(retained_count) < 1:
            raise ValueError("retained_count must be positive")
        if not 0.0 <= float(trim_ratio) < 1.0:
            raise ValueError("trim_ratio must be in [0, 1)")
        self.retained_count = int(retained_count)
        self.trim_ratio = float(trim_ratio)

    def score(
        self,
        query_tokens: Any,
        retained_reference_tokens: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> SimilarityOutput:
        q_ndim = getattr(query_tokens, "ndim", None)
        r_ndim = getattr(retained_reference_tokens, "ndim", None)
        if q_ndim is None or r_ndim is None:
            raise TypeError(
                "query_tokens and retained_reference_tokens must be tensors"
            )

        scores = []
        retained = []
        if q_ndim == 2 and r_ndim in (3, 4):
            score, indices = score_retained_patch_bank(
                query_tokens,
                retained_reference_tokens,
                retained_count=self.retained_count,
                trim_ratio=self.trim_ratio,
            )
            scores = score
            retained = indices
        elif q_ndim == 3 and r_ndim == 4:
            if int(query_tokens.shape[0]) != int(retained_reference_tokens.shape[0]):
                raise ValueError("batched query/reference batch sizes must match")
            for batch_id in range(int(query_tokens.shape[0])):
                score, indices = score_retained_patch_bank(
                    query_tokens[batch_id],
                    retained_reference_tokens[batch_id],
                    retained_count=self.retained_count,
                    trim_ratio=self.trim_ratio,
                )
                scores.append(score)
                retained.append(indices)
            scores = _stack(scores)
        else:
            raise ValueError(
                "expected query [P,D] and refs [K,P,D], or batched [B,P,D]/[B,K,P,D]"
            )

        diagnostics: dict[str, Any] = {
            "policy": self.name,
            "method": "nearest_cosine_distance_over_unaggregated_retained_patch_bank",
            "aggregation": "none",
            "retained_count_requested": self.retained_count,
            "trim_ratio_compatibility": self.trim_ratio,
            "retained_reference_indices": retained,
        }
        if metadata:
            diagnostics.update(dict(metadata))
        return SimilarityOutput(patch_scores=scores, diagnostics=diagnostics)


def _stack(values: list[Any]) -> Any:
    if not values:
        return values
    first = values[0]
    try:
        import torch

        return torch.stack(values, dim=0)
    except Exception:
        import numpy as np

        return np.stack(values, axis=0)


class BaselineFusionPolicy:
    """Reproduce per-scale robust-positive-z followed by fixed weighted mean."""

    name = "BaselineFusionPolicy"

    def __init__(
        self, *, scales: Sequence[int] = (224, 448, 672), normalize: bool = True
    ) -> None:
        self.scales = tuple(int(scale) for scale in scales)
        if not self.scales:
            raise ValueError("at least one scale is required")
        self.normalize = bool(normalize)

    def fuse(
        self,
        scale_scores: Mapping[int, Any],
        *,
        weights: Mapping[int, Any] | Sequence[float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> FusionOutput:
        supplied = {int(scale): value for scale, value in scale_scores.items()}
        missing = [scale for scale in self.scales if scale not in supplied]
        if missing:
            raise ValueError(f"missing scale scores for {missing}")

        if weights is None:
            weight_map = {scale: 1.0 / len(self.scales) for scale in self.scales}
        elif isinstance(weights, Mapping):
            weight_map = {
                scale: float(
                    weights[scale] if scale in weights else weights[str(scale)]
                )
                for scale in self.scales
            }
        else:
            values = list(weights)
            if len(values) != len(self.scales):
                raise ValueError("weight sequence length must match scales")
            weight_map = {
                scale: float(value) for scale, value in zip(self.scales, values)
            }
        total = float(sum(weight_map.values()))
        if not np.isfinite(total) or total <= 0:
            raise ValueError("fusion weights must have a positive finite sum")
        weight_map = {scale: value / total for scale, value in weight_map.items()}

        normalized: dict[int, np.ndarray] = {}
        fused: np.ndarray | None = None
        for scale in self.scales:
            score = np.asarray(supplied[scale], dtype=float)
            norm = robust_positive_z(score) if self.normalize else score
            normalized[scale] = norm
            contribution = weight_map[scale] * norm
            fused = contribution if fused is None else fused + contribution

        diagnostics: dict[str, Any] = {
            "policy": self.name,
            "normalization": "robust_positive_z" if self.normalize else "none",
            "fusion": "weighted_mean",
            "scale_order": list(self.scales),
        }
        if metadata:
            diagnostics.update(dict(metadata))
        return FusionOutput(
            normalized_scale_scores=normalized,
            weights=weight_map,
            fused_score=np.asarray(fused, dtype=float),
            diagnostics=diagnostics,
        )

    # ``score`` is a convenient alias for callers using policy-neutral naming.
    score = fuse


class BaselineDecisionPolicy:
    """Thin adapter around ``models.decision.compute_detection_intervals``."""

    name = "BaselineDecisionPolicy"

    def __init__(
        self,
        *,
        alpha: float = 0.01,
        method: str = "mean",
        smoothing: bool = True,
        sliding: bool = False,
        anomaly_padding: int = 0,
    ) -> None:
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.method = str(method)
        self.smoothing = bool(smoothing)
        self.sliding = bool(sliding)
        self.anomaly_padding = int(anomaly_padding)

    def decide(
        self,
        fused_score: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> DecisionOutput:
        intervals, threshold, processed = compute_detection_intervals(
            score_vector=np.asarray(fused_score, dtype=float),
            alpha=self.alpha,
            method=self.method,
            smoothing=self.smoothing,
            sliding=self.sliding,
            anomaly_padding=self.anomaly_padding,
        )
        processed_array = np.asarray(processed, dtype=float)
        flags = np.zeros(processed_array.shape, dtype=bool)
        for start, end in intervals:
            if processed_array.size:
                flags[max(0, int(start)) : min(processed_array.size, int(end) + 1)] = (
                    True
                )
        diagnostics: dict[str, Any] = {
            "policy": self.name,
            "alpha": self.alpha,
            "method": self.method,
            "smoothing": self.smoothing,
            "sliding": self.sliding,
            "anomaly_padding": self.anomaly_padding,
        }
        if metadata:
            diagnostics.update(dict(metadata))
        return DecisionOutput(
            threshold=threshold,
            processed_scores=processed_array,
            anomaly_flags=flags,
            intervals=[(int(start), int(end)) for start, end in intervals],
            diagnostics=diagnostics,
        )

    # Policy-neutral callers may prefer ``score`` for the final stage.
    score = decide


@dataclass
class ModularPipelineOutput:
    """All stage outputs, with the shared state retained for diagnostics."""

    similarity_by_scale: dict[int, SimilarityOutput]
    fusion: FusionOutput
    decision: DecisionOutput
    state: ScreeningState


class FrozenScreening:
    """Run the frozen retrieval output through replaceable baseline policies.

    Reference retrieval and trim selection remain outside this object.  Each
    ``scale_inputs`` value is either ``(query_tokens, retained_reference_tokens)``
    or a mapping with ``query_tokens`` and ``retained_reference_tokens`` keys.
    This boundary lets a future agent replace one policy without re-encoding or
    re-running retrieval.
    """

    def __init__(
        self,
        *,
        similarity_policy: Any | None = None,
        fusion_policy: Any | None = None,
        decision_policy: Any | None = None,
        dataset: str = "",
        signal_id: str = "",
    ) -> None:
        self.similarity_policy = similarity_policy or BaselineSimilarityPolicy()
        self.fusion_policy = fusion_policy or BaselineFusionPolicy()
        self.decision_policy = decision_policy or BaselineDecisionPolicy()
        self.dataset = str(dataset)
        self.signal_id = str(signal_id)

    def score_scales(
        self,
        scale_inputs: Mapping[int, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[int, SimilarityOutput]:
        outputs: dict[int, SimilarityOutput] = {}
        for scale, value in scale_inputs.items():
            if isinstance(value, Mapping):
                query = value["query_tokens"]
                refs = value["retained_reference_tokens"]
                local_metadata = value.get("metadata", metadata)
            else:
                try:
                    query, refs = value
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "scale input must be (query_tokens, retained_reference_tokens)"
                    ) from exc
                local_metadata = metadata
            outputs[int(scale)] = self.similarity_policy.score(
                query,
                refs,
                metadata=local_metadata,
            )
        if not outputs:
            raise ValueError("scale_inputs must contain at least one scale")
        return outputs

    def run(
        self,
        scale_inputs: Mapping[int, Any],
        *,
        weights: Mapping[int, Any] | list[float] | tuple[float, ...] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ModularPipelineOutput:
        similarities = self.score_scales(scale_inputs, metadata=metadata)
        return self.run_from_similarity_outputs(
            similarities, weights=weights, metadata=metadata
        )

    def run_from_similarity_outputs(
        self,
        similarity_outputs: Mapping[int, SimilarityOutput],
        *,
        weights: Mapping[int, Any] | list[float] | tuple[float, ...] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ModularPipelineOutput:
        scale_scores = {
            int(scale): output.patch_scores
            for scale, output in similarity_outputs.items()
        }
        return self.run_from_scale_scores(
            scale_scores,
            weights=weights,
            metadata=metadata,
            similarity_outputs=dict(similarity_outputs),
        )

    def run_from_scale_scores(
        self,
        scale_scores: Mapping[int, Any],
        *,
        weights: Mapping[int, Any] | list[float] | tuple[float, ...] | None = None,
        metadata: Mapping[str, Any] | None = None,
        similarity_outputs: Mapping[int, SimilarityOutput] | None = None,
    ) -> ModularPipelineOutput:
        fusion = self.fusion_policy.fuse(
            scale_scores, weights=weights, metadata=metadata
        )
        decision = self.decision_policy.decide(fusion.fused_score, metadata=metadata)
        state = ScreeningState(
            dataset=self.dataset,
            signal_id=self.signal_id,
            scale=None,
            scale_scores=dict(scale_scores),
            patch_scores=dict(scale_scores),
            retained_reference_ids={
                int(scale): output.diagnostics.get("retained_reference_indices")
                for scale, output in (similarity_outputs or {}).items()
                if "retained_reference_indices" in output.diagnostics
            },
            normalized_scale_scores=fusion.normalized_scale_scores,
            fusion_weights=fusion.weights,
            fused_score=fusion.fused_score,
            threshold=decision.threshold,
            processed_scores=decision.processed_scores,
            anomaly_flags=decision.anomaly_flags,
            detection_intervals=decision.intervals,
            diagnostics={
                "similarity": {
                    int(scale): dict(output.diagnostics)
                    for scale, output in (similarity_outputs or {}).items()
                },
                "fusion": dict(fusion.diagnostics),
                "decision": dict(decision.diagnostics),
            },
        )
        return ModularPipelineOutput(
            similarity_by_scale=dict(similarity_outputs or {}),
            fusion=fusion,
            decision=decision,
            state=state,
        )
