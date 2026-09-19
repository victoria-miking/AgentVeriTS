from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from ..types import GlobalDecision, Interval, ScaleReferenceTrace
if TYPE_CHECKING:
    from ..visual.render import SeriesRenderer


@dataclass
class ToolObservation:
    tool: str
    summary: str
    data: dict[str, Any]
    images: list[str]

    def as_prompt_text(self) -> str:
        return f"TOOL={self.tool}\nSUMMARY={self.summary}\nDATA={self.data}"


class EvidenceTools:
    def __init__(
        self,
        values: Sequence[float],
        output_dir: str | Path,
        renderer: SeriesRenderer,
        *,
        global_image: str | Path,
        context_points: int = 256,
        reference_count: int = 4,
        reference_traces: dict[int, ScaleReferenceTrace] | None = None,
    ) -> None:
        self.values = np.asarray(values, dtype=float).reshape(-1)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.renderer = renderer
        self.global_image = str(global_image)
        self.context_points = int(context_points)
        self.reference_count = int(reference_count)
        self.reference_traces = dict(reference_traces or {})

    @staticmethod
    def _robust_stats(x: np.ndarray) -> dict[str, float]:
        finite = x[np.isfinite(x)]
        if not len(finite):
            return {"mean": 0.0, "std": 0.0, "median": 0.0, "mad": 0.0, "min": 0.0, "max": 0.0, "slope": 0.0}
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        slope = float(np.polyfit(np.arange(len(finite)), finite, 1)[0]) if len(finite) > 1 else 0.0
        return {
            "mean": float(np.mean(finite)), "std": float(np.std(finite)), "median": median,
            "mad": mad, "min": float(np.min(finite)), "max": float(np.max(finite)), "slope": slope,
        }

    def _focus(self, decision: GlobalDecision) -> Interval:
        return decision.final_interval or decision.reviewed_interval

    def global_context(self, decision: GlobalDecision, args: dict[str, Any]) -> ToolObservation:
        return ToolObservation("global_context", "Full-series context reused.", {"decision_id": decision.decision_id}, [self.global_image])

    def local_context(self, decision: GlobalDecision, args: dict[str, Any]) -> ToolObservation:
        interval = self._focus(decision)
        context = int(args.get("context_points", self.context_points))
        path = self.output_dir / f"{decision.decision_id}_local.png"
        self.renderer.local_plot(self.values, interval, path, context_points=context, local_y=True)
        return ToolObservation("local_context", "Focused raw-series context around the target interval.", {"interval": interval.as_list(), "context_points": context}, [str(path)])

    def scale_view(self, decision: GlobalDecision, args: dict[str, Any]) -> ToolObservation:
        interval = self._focus(decision)
        mode = str(args.get("mode", "global_y"))
        local_y = mode == "local_y"
        path = self.output_dir / f"{decision.decision_id}_{mode}.png"
        self.renderer.local_plot(self.values, interval, path, context_points=int(args.get("context_points", self.context_points)), local_y=local_y)
        return ToolObservation("scale_view", f"Replot using {'local' if local_y else 'global/shared'} y-range.", {"mode": mode, "interval": interval.as_list()}, [str(path)])

    def raw_segment(self, decision: GlobalDecision, args: dict[str, Any]) -> ToolObservation:
        interval = self._focus(decision)
        context = int(args.get("context_points", min(self.context_points, 64)))
        lo = max(0, interval.start - context)
        hi = min(len(self.values) - 1, interval.end + context)
        segment = self.values[lo:hi + 1]
        diffs = np.diff(segment)
        sample_count = min(160, len(segment))
        sample_idx = np.linspace(0, max(0, len(segment) - 1), sample_count, dtype=int) if len(segment) else np.array([], dtype=int)
        data = {
            "range": [lo, hi],
            "sample_indices": (sample_idx + lo).astype(int).tolist(),
            "sample_values": segment[sample_idx].astype(float).tolist() if len(sample_idx) else [],
            "first_difference_stats": self._robust_stats(diffs),
        }
        return ToolObservation("raw_segment", "Raw values and first-difference evidence around the target.", data, [])

    def stat_features(self, decision: GlobalDecision, args: dict[str, Any]) -> ToolObservation:
        interval = self._focus(decision)
        context = int(args.get("context_points", self.context_points))
        lo = max(0, interval.start - context)
        hi = min(len(self.values) - 1, interval.end + context)
        inside = self.values[interval.start:interval.end + 1]
        left = self.values[lo:interval.start]
        right = self.values[interval.end + 1:hi + 1]
        background = np.concatenate([left, right]) if len(left) + len(right) else np.array([], dtype=float)
        inside_stats = self._robust_stats(inside)
        bg_stats = self._robust_stats(background)
        scale = 1.4826 * bg_stats["mad"] + 1e-12
        robust_location_z = (inside_stats["median"] - bg_stats["median"]) / scale
        robust_range_ratio = (inside_stats["max"] - inside_stats["min"] + 1e-12) / (bg_stats["max"] - bg_stats["min"] + 1e-12)
        data = {
            "interval": interval.as_list(), "context_range": [lo, hi],
            "inside": inside_stats, "context_excluding_interval": bg_stats,
            "robust_location_z": float(robust_location_z),
            "robust_range_ratio": float(robust_range_ratio),
        }
        return ToolObservation("stat_features", "Robust interval-vs-context statistics.", data, [])

    @staticmethod
    def _resample_z(x: np.ndarray, n: int = 128) -> np.ndarray:
        if len(x) < 2:
            return np.zeros(n, dtype=float)
        y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
        y = y - np.mean(y)
        std = np.std(y)
        return y / (std + 1e-12)

    def _reference_intervals(self, query: Interval, count: int) -> tuple[list[Interval], list[float]]:
        length = query.end - query.start + 1
        stride = max(1, length // 4)
        q = self._resample_z(self.values[query.start:query.end + 1])
        qn = q / (np.linalg.norm(q) + 1e-12)
        rows: list[tuple[float, Interval]] = []
        for start in range(0, max(1, len(self.values) - length + 1), stride):
            end = start + length - 1
            if end >= len(self.values):
                break
            if not (end < query.start or start > query.end):
                continue
            r = self._resample_z(self.values[start:end + 1])
            score = float(np.dot(qn, r / (np.linalg.norm(r) + 1e-12)))
            rows.append((score, Interval(start, end)))
        rows.sort(key=lambda item: item[0], reverse=True)
        chosen = rows[: max(0, int(count))]
        return [x[1] for x in chosen], [x[0] for x in chosen]

    def _screening_reference_intervals(
        self,
        target: Interval,
        count: int,
    ) -> tuple[Interval | None, list[Interval], int | None]:
        if not self.reference_traces:
            return None, [], None
        target_length = target.end - target.start + 1
        scale = min(self.reference_traces, key=lambda value: abs(int(value) - int(target_length)))
        trace = self.reference_traces[int(scale)]
        if not trace.window_starts:
            return None, [], int(scale)
        target_center = (target.start + target.end) / 2.0
        starts = np.asarray(trace.window_starts, dtype=np.int64)
        centers = starts.astype(float) + (int(scale) - 1) / 2.0
        query_index = int(np.argmin(np.abs(centers - target_center)))
        query_start = int(starts[query_index])
        query_end = min(len(self.values) - 1, query_start + int(scale) - 1)
        query_window = Interval(query_start, query_end)
        rows = trace.retained_reference_starts[query_index] if query_index < len(trace.retained_reference_starts) else []
        refs = [
            Interval(int(start), min(len(self.values) - 1, int(start) + int(scale) - 1))
            for start in rows[: max(0, int(count))]
        ]
        return query_window, refs, int(scale)

    def reference_context(self, decision: GlobalDecision, args: dict[str, Any]) -> ToolObservation:
        target = self._focus(decision)
        count = int(args.get("count", self.reference_count))
        query_window, refs, scale = self._screening_reference_intervals(target, count)
        context_points = int(args.get("context_points", min(self.context_points, 128)))
        shared_y = bool(args.get("shared_y", True))
        images: list[str] = []

        if query_window is not None and refs:
            target_path = self.output_dir / f"{decision.decision_id}_reference_target.png"
            comparison_path = self.output_dir / f"{decision.decision_id}_references.png"
            self.renderer.local_plot(
                self.values,
                target,
                target_path,
                context_points=context_points,
                local_y=False,
            )
            self.renderer.comparison_plot(
                self.values,
                query_window,
                refs,
                comparison_path,
                context_points=context_points,
                shared_y=shared_y,
            )
            images = [str(target_path), str(comparison_path)]
            data = {
                "target_interval": target.as_list(),
                "screening_scale": int(scale),
                "screening_query_window": query_window.as_list(),
                "references": [r.as_list() for r in refs],
                "reference_source": "stage1_retained_visual_references",
                "warning": "These references were retained by visual screening. Similarity and retention do not establish normality.",
            }
            return ToolObservation(
                "reference_context",
                "Reused the robust reference windows retained by visual screening for the nearest matching screening window.",
                data,
                images,
            )

        refs, similarities = self._reference_intervals(target, count)
        path = self.output_dir / f"{decision.decision_id}_references_fallback.png"
        self.renderer.comparison_plot(
            self.values,
            target,
            refs,
            path,
            context_points=context_points,
            shared_y=shared_y,
        )
        data = {
            "target_interval": target.as_list(),
            "references": [r.as_list() for r in refs],
            "shape_cosine_similarity": similarities,
            "reference_source": "fallback_raw_shape_retrieval",
            "warning": "Fallback retrieval was used because no retained screening references were available. Similarity does not establish normality.",
        }
        return ToolObservation(
            "reference_context",
            "No retained screening references were available; used non-overlapping raw-shape retrieval as a fallback.",
            data,
            [str(path)],
        )

    def spike_scan(self, decision: GlobalDecision, args: dict[str, Any]) -> ToolObservation:
        interval = self._focus(decision)
        context = int(args.get("context_points", min(self.context_points, 128)))
        lo = max(0, interval.start - context)
        hi = min(len(self.values) - 1, interval.end + context)
        bg = np.concatenate([self.values[lo:interval.start], self.values[interval.end + 1:hi + 1]])
        inside = self.values[interval.start:interval.end + 1]
        bg_med = float(np.median(bg)) if len(bg) else float(np.median(inside))
        bg_mad = float(np.median(np.abs(bg - bg_med))) if len(bg) else 0.0
        scale = 1.4826 * bg_mad + 1e-12
        peak_z = float((np.max(inside) - bg_med) / scale) if len(inside) else 0.0
        trough_z = float((bg_med - np.min(inside)) / scale) if len(inside) else 0.0
        data = {
            "interval": interval.as_list(), "duration": len(inside),
            "peak_robust_z": peak_z, "trough_robust_z": trough_z,
            "dominant_direction": "peak" if peak_z >= trough_z else "trough",
            "context_median": bg_med, "context_mad": bg_mad,
        }
        return ToolObservation("spike_scan", "Robust peak/trough prominence and duration evidence.", data, [])

    def execute(self, name: str, decision: GlobalDecision, args: dict[str, Any] | None = None) -> ToolObservation:
        args = dict(args or {})
        handlers = {
            "global_context": self.global_context,
            "local_context": self.local_context,
            "reference_context": self.reference_context,
            "raw_segment": self.raw_segment,
            "stat_features": self.stat_features,
            "scale_view": self.scale_view,
            "spike_scan": self.spike_scan,
        }
        if name not in handlers:
            raise ValueError(f"unknown evidence tool: {name}")
        return handlers[name](decision, args)

    def execute_for_interval(
        self,
        name: str,
        interval: Interval | None,
        args: dict[str, Any] | None = None,
        *,
        decision_id: str = "GLOBAL_RESCAN",
    ) -> ToolObservation:
        args = dict(args or {})
        if name == "global_context":
            placeholder = interval or Interval(0, 0)
        else:
            if interval is None:
                raise ValueError(f"{name} requires a target interval during global rescan")
            placeholder = interval
        pseudo = GlobalDecision(
            decision_id=decision_id,
            source="added",
            candidate_id=None,
            reviewed_interval=placeholder,
            action="add",
            final_interval=placeholder,
            confidence=1,
            rationale="temporary evidence target",
        )
        return self.execute(name, pseudo, args)
