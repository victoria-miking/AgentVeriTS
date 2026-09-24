from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import json

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
        return f"TOOL={self.tool}\nSUMMARY={self.summary}\nDATA={json.dumps(self.data, allow_nan=False)}"


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
        self._request_index = 0
        if not len(self.values) or not np.isfinite(self.values).all():
            raise ValueError("evidence tools require a nonempty finite signal")

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
        page_start = int(args.get("page_start", lo))
        if not lo <= page_start <= hi:
            raise ValueError("page_start must lie within the requested raw range")
        max_points = int(args.get("max_points", 2048))
        if not 1 <= max_points <= 2048:
            raise ValueError("max_points must be between 1 and 2048")
        page_end = min(hi, page_start + max_points - 1)
        data = {
            "requested_range": [lo, hi], "range": [page_start, page_end],
            "sample_indices": list(range(page_start, page_end + 1)),
            "sample_values": self.values[page_start:page_end + 1].tolist(),
            "next_start": page_end + 1 if page_end < hi else None,
            "first_difference_stats": self._robust_stats(diffs),
        }
        return ToolObservation("raw", "Exact consecutive raw samples; next_start indicates remaining samples.", data, [])

    def _background(self, interval: Interval, args: dict[str, Any]) -> tuple[np.ndarray, int, int]:
        context = int(args.get("context_points", self.context_points))
        lo, hi = max(0, interval.start - context), min(len(self.values) - 1, interval.end + context)
        baseline = args.get("baseline", "surrounding")
        if baseline == "global":
            lo, hi = 0, len(self.values) - 1
        left, right = self.values[lo:interval.start], self.values[interval.end + 1:hi + 1]
        if baseline == "left":
            return left, lo, interval.start - 1
        if baseline == "right":
            return right, interval.end + 1, hi
        if baseline not in {"surrounding", "global"}:
            raise ValueError("unknown statistics baseline")
        return np.concatenate([left, right]), lo, hi

    def stat_features(self, decision: GlobalDecision, args: dict[str, Any]) -> ToolObservation:
        interval = self._focus(decision)
        background, lo, hi = self._background(interval, args)
        inside_stats = self._robust_stats(self.values[interval.start:interval.end + 1])
        bg_stats = self._robust_stats(background) if len(background) else None
        data = {
            "interval": interval.as_list(), "context_range": [lo, hi],
            "baseline": args.get("baseline", "surrounding"), "baseline_count": len(background),
            "inside": inside_stats, "context_excluding_interval": bg_stats,
            "robust_location_z": None if bg_stats is None else (inside_stats["median"] - bg_stats["median"]) / (1.4826 * bg_stats["mad"] + 1e-12),
            "robust_range_ratio": None if bg_stats is None else (inside_stats["max"] - inside_stats["min"] + 1e-12) / (bg_stats["max"] - bg_stats["min"] + 1e-12),
        }
        return ToolObservation("statistics", "Robust interval-vs-baseline statistics; null deviations mean no baseline samples.", data, [])

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
        requested_scale: int | None = None,
    ) -> tuple[Interval | None, list[Interval], int | None]:
        if not self.reference_traces:
            return None, [], None
        target_length = target.end - target.start + 1
        scale = requested_scale or min(self.reference_traces, key=lambda value: abs(int(value) - int(target_length)))
        if scale not in self.reference_traces:
            return None, [], scale
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
        refs = [r for r in refs if r.end < target.start or r.start > target.end]
        return query_window, refs, int(scale)

    def reference_context(self, decision: GlobalDecision, args: dict[str, Any]) -> ToolObservation:
        target = self._focus(decision)
        count = int(args.get("count", self.reference_count))
        query_window, refs, scale = self._screening_reference_intervals(target, count, args.get("scale"))
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

        query = target
        if args.get("scale") is not None:
            length = min(len(self.values), int(args["scale"]))
            start = max(0, min(len(self.values) - length, (target.start + target.end - length + 1) // 2))
            query = Interval(start, start + length - 1)
        refs, similarities = self._reference_intervals(query, count)
        pairs = [(r, sim) for r, sim in zip(refs, similarities) if r.end < target.start or r.start > target.end]
        refs, similarities = [x[0] for x in pairs], [x[1] for x in pairs]
        path = self.output_dir / f"{decision.decision_id}_references_fallback.png"
        self.renderer.comparison_plot(
            self.values,
            query,
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
        bg, lo, hi = self._background(interval, args)
        inside = self.values[interval.start:interval.end + 1]
        bg_med = float(np.median(bg)) if len(bg) else float(np.median(inside))
        bg_mad = float(np.median(np.abs(bg - bg_med))) if len(bg) else 0.0
        scale = 1.4826 * bg_mad + 1e-12
        peak_z = float((np.max(inside) - bg_med) / scale) if len(inside) else 0.0
        trough_z = float((bg_med - np.min(inside)) / scale) if len(inside) else 0.0
        data = {
            "interval": interval.as_list(), "duration": len(inside),
            "peak_robust_z": peak_z if len(bg) else None, "trough_robust_z": trough_z if len(bg) else None,
            "baseline": args.get("baseline", "surrounding"), "baseline_count": len(bg),
            "dominant_direction": "peak" if peak_z >= trough_z else "trough",
            "context_median": bg_med, "context_mad": bg_mad,
        }
        return ToolObservation("spike_scan", "Robust peak/trough prominence and duration evidence.", data, [])

    def execute(self, name: str, decision: GlobalDecision, args: dict[str, Any] | None = None) -> ToolObservation:
        args = {k: v for k, v in (args or {}).items() if v is not None}
        allowed = {"interval", "context_points", "mode", "count", "scale", "shared_y", "baseline", "diagnostic", "max_points", "page_start"}
        if set(args) - allowed:
            raise ValueError("unknown evidence parameters")
        for key in ("context_points", "count", "scale", "max_points", "page_start"):
            if key in args and (type(args[key]) is not int or args[key] < 0):
                raise ValueError(f"{key} must be a nonnegative integer")
        if "count" in args and not 1 <= args["count"] <= 16:
            raise ValueError("count must be between 1 and 16")
        if "scale" in args and args["scale"] < 2:
            raise ValueError("scale must be >= 2")
        if "shared_y" in args and type(args["shared_y"]) is not bool:
            raise ValueError("shared_y must be boolean")
        if args.get("mode", "local_y") not in {"local_y", "global_y"}:
            raise ValueError("unknown plot mode")
        if args.get("diagnostic", "robust") not in {"robust", "spike"}:
            raise ValueError("unknown diagnostic")
        target = self._focus(decision)
        if "interval" in args:
            raw = args.pop("interval")
            if not isinstance(raw, (tuple, list)) or len(raw) != 2:
                raise ValueError("interval requires two integer endpoints")
            target = Interval(raw[0], raw[1])
        if target.end >= len(self.values):
            raise ValueError("tool target exceeds signal bounds")
        self._request_index += 1
        # Each observation has a distinct artifact path; repeat requests cannot overwrite evidence.
        proxy = replace(decision, decision_id=f"{decision.decision_id[:64]}_{self._request_index}",
                        reviewed_interval=target, final_interval=target)
        handlers = {"raw": self.raw_segment, "reference": self.reference_context,
                    "focus": self.scale_view if args.get("mode") == "global_y" else self.local_context,
                    "statistics": self.spike_scan if args.get("diagnostic") == "spike" else self.stat_features}
        if name not in handlers:
            raise ValueError(f"unknown evidence tool: {name}")
        observation = handlers[name](proxy, args)
        observation.tool = name
        return observation
