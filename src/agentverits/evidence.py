from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import SignalContext, AdjacentCandidateGrouping
from .rendering import RendererConfig, SignalRenderer
from .windows import candidate_home_windows, plan_assessment_windows, render_profile
from .contracts import CandidateInterval, MediaRef, ToolResult
from .contracts import AgentSessionState, GroupRuntimeState
from anomaly_screening.cache import load_signal_cache, load_scale_cache


def _finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


@dataclass
class CachedSignal:
    dataset: str
    signal_id: str
    values: np.ndarray
    timestamps: np.ndarray
    scales: dict[int, dict[str, Any] | None]
    screening_intervals: list[CandidateInterval]
    screening_alpha: float
    root: Path
    evidence_root: Path


class CachedSignalContextProvider:
    """Build Agent-safe context from the completed ANOMALY_SCREENING cache.

    Evaluation-selected alpha/intervals are deliberately ignored because the
    cache chooses them with F1 against held-out annotations. The adapter uses a
    fixed, explicit Anomaly Screening alpha instead.
    """

    def __init__(
        self,
        cache_root: str | Path,
        *,
        screening_alpha: float,
        renderer: SignalRenderer | None = None,
        render_root: str | Path | None = None,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.screening_alpha = float(screening_alpha)
        if self.screening_alpha not in (0.1, 0.01, 0.001):
            raise ValueError("screening_alpha must be an explicit experiment threshold")
        self.renderer = renderer or SignalRenderer(RendererConfig())
        self.render_root = (
            Path(render_root)
            if render_root is not None
            else self.cache_root / "agent_previews"
        )
        self._signals: dict[tuple[str, str], CachedSignal] = {}

    def _load(self, dataset: str, signal_id: str) -> CachedSignal:
        key = (str(dataset), str(signal_id))
        if key in self._signals:
            return self._signals[key]
        cache = load_signal_cache(self.cache_root, *key, include_scales=False)
        manifest = cache["manifest"]
        decision = manifest.get("decision", {})
        decision_rows = decision.get("alphas", [])
        alpha = self.screening_alpha
        row = next(
            (
                r
                for r in decision_rows
                if abs(float(r.get("alpha")) - float(alpha)) < 1e-12
            ),
            None,
        )
        if row is None:
            raise KeyError(f"screening alpha {alpha} is not in cache for {key}")
        intervals: list[CandidateInterval] = []
        selected_pairs = row.get("intervals", [])
        for i, pair in enumerate(selected_pairs, 1):
            intervals.append(CandidateInterval(int(pair[0]), int(pair[1]), f"C{i:04d}"))
        dec = cache.get("decision", {})
        values = np.asarray(dec.get("values"), dtype=np.float32).reshape(-1)
        timestamps = np.asarray(dec.get("source_timestamps", np.arange(len(values))))
        scales = {int(s): None for s in manifest.get("scales", {})}
        signal = CachedSignal(
            str(dataset),
            str(signal_id),
            values,
            timestamps,
            scales,
            intervals,
            float(alpha),
            Path(cache["root"]),
            self.render_root / dataset / signal_id / "evidence",
        )
        self._signals[key] = signal
        return signal

    def build_context(self, signal_key: Any) -> SignalContext:
        if isinstance(signal_key, (tuple, list)) and len(signal_key) == 2:
            dataset, signal_id = str(signal_key[0]), str(signal_key[1])
        elif isinstance(signal_key, str) and "/" in signal_key:
            dataset, signal_id = signal_key.split("/", 1)
        else:
            raise ValueError(
                "signal_key must be (dataset, signal_id) or 'dataset/signal_id'"
            )
        signal = self._load(dataset, signal_id)
        groups = AdjacentCandidateGrouping().group(signal.screening_intervals)
        group_by_candidate = {
            interval.candidate_id: group.group_id
            for group in groups
            for interval in group.intervals
        }
        candidates = [
            {
                "candidate_id": c.candidate_id,
                "start": c.start,
                "end": c.end,
                "group_id": group_by_candidate.get(c.candidate_id, ""),
            }
            for c in signal.screening_intervals
        ]
        out_dir = self.render_root / dataset / signal_id
        profile = render_profile(len(signal.values))
        candidate_dicts = candidates
        windows = plan_assessment_windows(
            len(signal.values), [c.as_pair() for c in signal.screening_intervals]
        )
        if profile.kind == "ultra_long":
            nav_meta = self.renderer.navigation_overview(
                signal.values,
                [w.to_dict() for w in windows],
                candidate_dicts,
                out_dir / "global_navigation.png",
                title="Global Navigation Overview | macro context only",
            )
            media = [
                MediaRef(
                    uri=str(nav_meta["path"]),
                    description="Global Navigation Overview; macro context and window navigation only, not local anomaly judgment",
                    detail="high",
                )
            ]
            renderer_preview = nav_meta
        else:
            meta = self.renderer.global_signal_plot(
                signal.values,
                candidate_dicts,
                out_dir / "global_signal.png",
                active_group=(groups[0].group_id if groups else None),
                title="Candidate Assessment | Global Signal",
                width_px=profile.width_px,
                height_px=profile.height_px,
                plot_role="candidate_assessment",
            )
            media = [
                MediaRef(
                    uri=str(meta["path"]),
                    description="persistent adaptive global signal visualization with Anomaly Screening Candidate proposals",
                    detail="high",
                )
            ]
            renderer_preview = meta
        ownership = candidate_home_windows(
            [c.as_pair() for c in signal.screening_intervals], windows
        )
        public = {
            "dataset": dataset,
            "signal_length": int(len(signal.values)),
            "screening_alpha": signal.screening_alpha,
            "screening_candidate_count": len(signal.screening_intervals),
            "scales": sorted(signal.scales),
            "adaptive_render_profile": profile.to_dict(),
            "assessment_window_policy": {
                "nominal_max_points": 20_000,
                "overlap_points": 2_000,
                "boundary_policy": "nearest candidate-free gap within overlap budget",
                "ownership_policy": "candidate center in authoritative core",
            },
            "assessment_windows": [w.to_dict() for w in windows],
            "candidate_home_windows": ownership,
            "renderer_preview": renderer_preview,
        }
        return SignalContext(
            signal_id=f"{dataset}/{signal_id}",
            signal_length=len(signal.values),
            full_series_media=media,
            candidate_intervals=list(signal.screening_intervals),
            public_metadata=public,
            adapter_payload=signal,
        )


def _interval(values: Sequence[float], start: int, end: int) -> tuple[int, int]:
    n = len(values)
    start, end = max(0, int(start)), min(n - 1, int(end))
    if end < start:
        raise ValueError("empty interval")
    return start, end


class CachedEvidenceToolAdapter:
    """Read-only implementation of the eight semantic Evidence Tools."""

    def __init__(self, *, renderer: SignalRenderer | None = None) -> None:
        self.renderer = renderer or SignalRenderer(RendererConfig())

    @staticmethod
    def _signal(context: SignalContext) -> CachedSignal:
        if not isinstance(context.adapter_payload, CachedSignal):
            raise TypeError(
                "SignalContext was not built by CachedSignalContextProvider"
            )
        return context.adapter_payload

    @staticmethod
    def _scale(signal: CachedSignal, scale: Any) -> tuple[int, dict[str, Any]]:
        if scale in (None, "auto"):
            scale = 224
        scale = int(scale)
        if scale not in signal.scales:
            raise KeyError(f"scale {scale} is unavailable")
        if signal.scales[scale] is None:
            signal.scales[scale] = load_scale_cache(
                signal.root.parents[1], signal.dataset, signal.signal_id, scale
            )
        return scale, signal.scales[scale]

    @staticmethod
    def _query(
        signal: CachedSignal,
        group: GroupRuntimeState,
        scale: int,
        payload: Mapping[str, Any],
    ) -> int:
        focus = (group.group.focus_start + group.group.focus_end) / 2
        starts = np.asarray(payload["window_starts"], dtype=int)
        size = int(np.asarray(payload["window_size"]).item())
        centers = starts + (size - 1) / 2
        return int(np.argmin(np.abs(centers - focus)))

    def _query_row(
        self,
        signal: CachedSignal,
        group: GroupRuntimeState,
        scale: int,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        q = self._query(signal, group, scale, payload)
        starts = np.asarray(payload["window_starts"], dtype=int)
        size = int(np.asarray(payload["window_size"]).item())
        start = int(starts[q])
        return {
            "query_window_id": q,
            "start": start,
            "end": start + size - 1,
            "core_interval": [start, start + size - 1],
            "scale": scale,
            "candidate_group": group.group.group_id,
        }

    def _reference_rows(
        self,
        signal: CachedSignal,
        group: GroupRuntimeState,
        scale: int,
        payload: Mapping[str, Any],
        limit: int = 4,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        q = self._query(signal, group, scale, payload)
        topk_ids = np.asarray(payload["topk_indices"])[q].astype(int)
        sims = np.asarray(payload["topk_similarities"])[q]
        retained_ids = np.asarray(payload["retained_window_ids"])[q].astype(int)
        retained_local = np.asarray(payload["retained_indices"])[q].astype(int)
        starts = np.asarray(payload["window_starts"], dtype=int)
        size = int(np.asarray(payload["window_size"]).item())
        rows: list[dict[str, Any]] = []
        for retained_rank, (wid, local_index) in enumerate(
            zip(retained_ids.tolist(), retained_local.tolist()), 1
        ):
            if retained_rank <= offset or len(rows) >= min(4, int(limit)):
                continue
            wid = int(wid)
            local_index = int(local_index)
            retrieval_rank = (
                local_index + 1 if 0 <= local_index < len(topk_ids) else None
            )
            sim = sims[local_index] if retrieval_rank is not None else np.nan
            r0, r1 = int(starts[wid]), int(starts[wid] + size - 1)
            rows.append(
                {
                    "reference_id": f"s{scale}:q{q:06d}:r{wid:06d}",
                    "start": r0,
                    "end": r1,
                    "core_interval": [r0, r1],
                    "retrieval_rank": retrieval_rank,
                    "retained_rank": retained_rank,
                    "similarity": _finite(sim),
                    "scale": scale,
                    "reference_source": "trim_retained_window",
                    "normality": "unknown",
                    "description": "trim-retained high-similarity historical reference candidate; normality unknown",
                }
            )
        return rows

    @staticmethod
    def _json_data(data: Mapping[str, Any]) -> dict[str, Any]:
        def conv(x: Any) -> Any:
            if isinstance(x, np.ndarray):
                return conv(x.tolist())
            if isinstance(x, (np.integer,)):
                return int(x)
            if isinstance(x, (np.floating,)):
                return _finite(x)
            if isinstance(x, dict):
                return {str(k): conv(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [conv(v) for v in x]
            return x

        return conv(dict(data))

    def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        context: SignalContext,
        session_state: AgentSessionState,
        group_state: GroupRuntimeState,
    ) -> ToolResult:
        signal = self._signal(context)
        try:
            if tool_name == "get_references":
                scale, payload = self._scale(signal, arguments.get("scale"))
                rows = self._reference_rows(
                    signal,
                    group_state,
                    scale,
                    payload,
                    max(1, min(4, int(arguments.get("limit", 4)))),
                    max(0, int(arguments.get("offset", 0))),
                )
                query = self._query_row(signal, group_state, scale, payload)
                return ToolResult(
                    tool_name,
                    "ok",
                    f"Retrieved {len(rows)} trim-retained reference candidates for {group_state.group.group_id}; normality unknown.",
                    "reference",
                    {
                        "scale": scale,
                        "query_group": group_state.group.group_id,
                        "query": query,
                        "reference_set": "trim_retain_4",
                        "references": rows,
                    },
                    warnings=[
                        "Similarity is retrieval evidence only; reference normality is unknown."
                    ],
                )
            if tool_name == "compare_references_visually":
                refs = []
                query = None
                for rid in arguments.get("reference_ids", []):
                    m = re.match(r"s(\d+):q(\d+):r(\d+)$", str(rid))
                    if not m:
                        continue
                    scale, query_id = int(m.group(1)), int(m.group(2))
                    _, payload = self._scale(signal, scale)
                    rows = self._reference_rows(signal, group_state, scale, payload, 4)
                    refs.extend([r for r in rows if r["reference_id"] == rid])
                    starts = np.asarray(payload["window_starts"], dtype=int)
                    size = int(np.asarray(payload["window_size"]).item())
                    qstart = int(starts[query_id])
                    query = {
                        "query_window_id": query_id,
                        "start": qstart,
                        "end": qstart + size - 1,
                        "scale": scale,
                    }
                if query is None or not refs:
                    raise ValueError("no valid retained reference IDs were supplied")
                context_mode = str(arguments.get("context") or "medium")
                context_points = {
                    "short": 64,
                    "medium": self.renderer.config.context_points,
                    "long": 256,
                }.get(context_mode, self.renderer.config.context_points)
                meta = self.renderer.query_reference_comparison(
                    signal.values,
                    query,
                    refs,
                    signal.evidence_root
                    / f"{group_state.group.group_id}_references.png",
                    mode="shared_xy",
                    context_points=context_points,
                )
                return ToolResult(
                    tool_name,
                    "ok",
                    "Query and each reference are shown with their own surrounding context; reference normality unknown.",
                    "reference_visual",
                    {"references": refs, "renderer": meta},
                    [
                        MediaRef(
                            meta["path"],
                            description="query-reference comparison; normality unknown",
                            detail="high",
                        )
                    ],
                    ["High similarity does not establish normality."],
                )
            if tool_name in {
                "inspect_raw_segment",
                "plot_focused_segment",
                "segment_statistics",
                "value_zscore",
                "diff_zscore",
            }:
                default = (group_state.group.focus_start, group_state.group.focus_end)
                start, end = _interval(
                    signal.values,
                    arguments.get("start", default[0]),
                    arguments.get("end", default[1]),
                )
                y = signal.values[start : end + 1].astype(float)
                if tool_name == "inspect_raw_segment":
                    if len(y) > 512:
                        raise ValueError("inspect_raw_segment range exceeds 512 points")
                    return ToolResult(
                        tool_name,
                        "ok",
                        f"Exact canonical values for absolute range [{start}, {end}].",
                        "raw_numeric",
                        {
                            "absolute_indices": list(range(start, end + 1)),
                            "values": self._json_data({"v": y})["v"],
                        },
                    )
                if tool_name == "plot_focused_segment":
                    meta = self.renderer.focused_candidate_plot(
                        signal.values,
                        start,
                        end,
                        signal.evidence_root
                        / f"{group_state.group.group_id}_focused.png",
                    )
                    return ToolResult(
                        tool_name,
                        "ok",
                        f"Focused candidate view for absolute range [{start}, {end}].",
                        "visual",
                        {"start": start, "end": end, "renderer": meta},
                        [
                            MediaRef(
                                meta["path"],
                                description="focused Candidate plot",
                                detail="high",
                            )
                        ],
                    )
                diffs = np.diff(signal.values.astype(float))
                if tool_name == "segment_statistics":
                    global_y = signal.values.astype(float)
                    baseline = float(np.median(global_y))
                    q = np.quantile(y, [0.05, 0.25, 0.5, 0.75, 0.95])
                    return ToolResult(
                        tool_name,
                        "ok",
                        f"Descriptive statistics for canonical range [{start}, {end}].",
                        "statistical",
                        {
                            "start": start,
                            "end": end,
                            "count": len(y),
                            "mean": _finite(np.mean(y)),
                            "std": _finite(np.std(y)),
                            "min": _finite(np.min(y)),
                            "max": _finite(np.max(y)),
                            "quantiles": {
                                "q05": _finite(q[0]),
                                "q25": _finite(q[1]),
                                "q50": _finite(q[2]),
                                "q75": _finite(q[3]),
                                "q95": _finite(q[4]),
                            },
                            "global_median_baseline": baseline,
                        },
                    )
                if tool_name == "value_zscore":
                    base = signal.values.astype(float)
                    med = float(np.median(base))
                    mad = float(np.median(np.abs(base - med)))
                    denom = 1.4826 * mad if mad > 1e-12 else float(np.std(base) or 1.0)
                    z = (y - med) / denom
                    return ToolResult(
                        tool_name,
                        "ok",
                        f"Absolute-value z-score evidence for canonical range [{start}, {end}].",
                        "statistical",
                        {
                            "start": start,
                            "end": end,
                            "baseline_median": med,
                            "robust_scale": denom,
                            "z_scores": self._json_data({"z": z})["z"],
                            "max_abs_z": _finite(np.max(np.abs(z))),
                        },
                    )
                local_diff = diffs[max(0, start - 1) : min(len(diffs), end) + 1]
                med = float(np.median(diffs))
                mad = float(np.median(np.abs(diffs - med)))
                denom = 1.4826 * mad if mad > 1e-12 else float(np.std(diffs) or 1.0)
                z = (local_diff - med) / denom
                return ToolResult(
                    tool_name,
                    "ok",
                    f"First-difference z-score evidence for canonical range [{start}, {end}].",
                    "statistical",
                    {
                        "start": max(0, start - 1),
                        "end": min(len(signal.values) - 1, end),
                        "baseline_median_diff": med,
                        "robust_scale": denom,
                        "z_scores": self._json_data({"z": z})["z"],
                        "max_abs_z": _finite(np.max(np.abs(z))),
                    },
                )
            if tool_name == "get_screening_evidence":
                target = str(arguments.get("target_id") or group_state.group.group_id)
                if str(arguments.get("target_type")) == "reference":
                    m = re.match(r"s(\d+):q(\d+):r(\d+)$", target)
                    if not m:
                        raise ValueError("unknown reference id")
                    scale, q, rid = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    _, payload = self._scale(signal, scale)
                    sims = np.asarray(payload["topk_similarities"])[q]
                    row = np.asarray(payload["aligned_score"])
                    return ToolResult(
                        tool_name,
                        "ok",
                        "Anomaly Screening retrieval diagnostics for a reference candidate; no label judgment is included.",
                        "screening",
                        {
                            "target_id": target,
                            "scale": scale,
                            "retrieval_similarity": (
                                _finite(
                                    sims[
                                        np.where(
                                            np.asarray(payload["topk_indices"])[q]
                                            == rid
                                        )[0][0]
                                    ]
                                )
                                if np.any(np.asarray(payload["topk_indices"])[q] == rid)
                                else None
                            ),
                            "scale_score_summary": {
                                "mean": _finite(np.mean(row)),
                                "peak": _finite(np.max(row)),
                            },
                        },
                    )
                start, end = group_state.group.focus_start, group_state.group.focus_end
                payloads = {}
                for_scale = []
                for scale in sorted(signal.scales):
                    _, payload = self._scale(signal, scale)
                    score = np.asarray(payload["aligned_score"], dtype=float)
                    local = score[start : min(len(score), end + 1)]
                    payloads[str(scale)] = {
                        "mean": _finite(np.mean(local)),
                        "peak": _finite(np.max(local)),
                        "global_mean": _finite(np.mean(score)),
                    }
                    for_scale.append(float(np.max(local)))
                return ToolResult(
                    tool_name,
                    "ok",
                    "Anomaly Screening score and multi-scale diagnostics for the active Candidate group.",
                    "screening",
                    {
                        "target_id": target,
                        "interval": [int(start), int(end)],
                        "scales": payloads,
                        "multi_scale_peak_support": int(sum(v > 0 for v in for_scale)),
                    },
                )
            raise KeyError(f"unknown evidence tool: {tool_name}")
        except Exception as exc:
            return ToolResult(
                tool_name,
                "error",
                "Evidence unavailable for the requested scope.",
                "error",
                warnings=["Tool arguments could not be resolved."],
                error=str(exc),
            )


__all__ = ["CachedSignalContextProvider", "CachedEvidenceToolAdapter", "CachedSignal"]
