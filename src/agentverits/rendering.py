from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)


@dataclass(frozen=True)
class RendererConfig:
    """Preview renderer controls; these are not frozen experiment semantics."""

    width_px: int = 1800
    height_px: int = 900
    dpi: int = 100
    context_points: int = 128
    focus_context_points: int = 128
    max_points_per_column: int = 2
    line_color: str = "#334155"
    candidate_color: str = "#b45309"
    active_color: str = "#2563eb"
    reference_color: str = "#15803d"
    global_width_px: int = 2048
    global_height_px: int = 512
    global_label_fontsize: float = 7.0
    detail_plot_height_px: int = 320
    comparison_panel_height_px: int = 220
    detail_left_margin_px: int = 112
    detail_right_margin_px: int = 28
    detail_top_margin_px: int = 52
    detail_bottom_margin_px: int = 58
    comparison_panel_gap_px: int = 24


def _extrema_envelope(
    x: np.ndarray, y: np.ndarray, columns: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bucket to pixel columns while retaining both extrema in every bucket."""
    if len(x) <= columns:
        return x, y, np.zeros(len(x), dtype=np.int8)
    edges = np.linspace(0, len(x), columns + 1, dtype=np.int64)
    xo: list[int] = []
    yo: list[float] = []
    marker: list[int] = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        sl = y[start:end]
        finite = np.isfinite(sl)
        if not finite.any():
            continue
        ids = np.flatnonzero(finite)
        lo = int(ids[np.argmin(sl[ids])])
        hi = int(ids[np.argmax(sl[ids])])
        chosen = [lo] if lo == hi else sorted((lo, hi))
        for local in chosen:
            xo.append(int(x[start + local]))
            yo.append(float(sl[local]))
            marker.append(1)
    return (
        np.asarray(xo, dtype=np.int64),
        np.asarray(yo, dtype=float),
        np.asarray(marker, dtype=np.int8),
    )


class SignalRenderer:
    def __init__(self, config: RendererConfig | None = None) -> None:
        self.config = config or RendererConfig()

    def _figure(
        self, title: str, *, width_px: int | None = None, height_px: int | None = None
    ):
        import matplotlib.pyplot as plt

        width = int(width_px or self.config.width_px)
        height = int(height_px or self.config.height_px)
        fig, ax = plt.subplots(
            figsize=(width / self.config.dpi, height / self.config.dpi),
            dpi=self.config.dpi,
        )
        fig.suptitle(title, fontsize=13)
        ax.set_xlabel("absolute integer x index")
        ax.set_ylabel("canonical signal value")
        ax.grid(True, alpha=0.2)
        return fig, ax

    @staticmethod
    def _span(
        ax: Any, start: int, end: int, color: str, label: str, alpha: float = 0.18
    ) -> None:
        ax.axvspan(int(start), int(end), color=color, alpha=alpha, label=label)
        ax.axvline(int(start), color=color, alpha=0.75, linewidth=0.8)
        ax.axvline(int(end), color=color, alpha=0.75, linewidth=0.8)

    @staticmethod
    def _centered_range(center: float, span: int, length: int) -> tuple[int, int]:
        span = max(1, min(int(span), int(length)))
        lo = int(round(float(center) - (span - 1) / 2.0))
        lo = max(0, min(lo, int(length) - span))
        return lo, lo + span - 1

    @staticmethod
    def _shared_ylim(segments: Sequence[np.ndarray]) -> tuple[float, float]:
        finite = np.concatenate(
            [
                np.asarray(v, dtype=float)[np.isfinite(v)]
                for v in segments
                if np.isfinite(v).any()
            ]
        )
        if not len(finite):
            return -1.0, 1.0
        lo, hi = float(np.min(finite)), float(np.max(finite))
        pad = max((hi - lo) * 0.05, 1e-6)
        return lo - pad, hi + pad

    def _pixel_exact_axis(self, *, plot_width_px: int, plot_height_px: int, title: str):
        import matplotlib.pyplot as plt

        left = self.config.detail_left_margin_px
        right = self.config.detail_right_margin_px
        top = self.config.detail_top_margin_px
        bottom = self.config.detail_bottom_margin_px
        width = left + int(plot_width_px) + right
        height = top + int(plot_height_px) + bottom
        fig = plt.figure(
            figsize=(width / self.config.dpi, height / self.config.dpi),
            dpi=self.config.dpi,
        )
        ax = fig.add_axes(
            [
                left / width,
                bottom / height,
                plot_width_px / width,
                plot_height_px / height,
            ]
        )
        fig.suptitle(title, fontsize=12, y=1.0 - 14 / height)
        ax.set_xlabel("absolute integer x index")
        ax.set_ylabel("canonical signal value")
        ax.grid(True, alpha=0.2)
        return fig, ax, width, height

    def global_signal_plot(
        self,
        values: Sequence[float],
        candidates: Sequence[Mapping[str, Any]],
        out_path: str | Path,
        *,
        active_group: str | None = None,
        title: str | None = None,
        width_px: int | None = None,
        height_px: int | None = None,
        x_start: int = 0,
        signal_length: int | None = None,
        plot_role: str = "candidate_assessment",
    ) -> dict[str, Any]:
        import matplotlib.pyplot as plt

        y = np.asarray(values, dtype=float).reshape(-1)
        x = np.arange(int(x_start), int(x_start) + len(y), dtype=np.int64)
        canvas_width = int(width_px or self.config.global_width_px)
        canvas_height = int(height_px or self.config.global_height_px)
        fig, ax = self._figure(
            title or "Global Signal Plot | Anomaly Screening Candidates",
            width_px=canvas_width,
            height_px=canvas_height,
        )
        px = max(2, int(canvas_width * 0.92))
        xd, yd, _ = _extrema_envelope(x, y, px)
        ax.plot(
            xd,
            yd,
            color=self.config.line_color,
            linewidth=0.75,
            label="canonical signal",
        )
        lane_count = max(1, min(8, int(np.ceil(max(1, len(candidates)) / 3.0))))
        lane_last_right = [-1e9] * lane_count
        label_width_px = 76.0
        plot_x_span = max(
            1, (int(signal_length) - 1) if signal_length is not None else (len(y) - 1)
        )
        for index, cand in enumerate(candidates):
            start, end = int(cand["start"]), int(cand["end"])
            gid = str(cand.get("group_id", "candidate"))
            # Global Plot is shared by all groups; group focus is conveyed by
            # text/state outside this persistent image, not a color cue.
            color = self.config.candidate_color
            candidate_id = str(cand.get("candidate_id", f"C{index + 1:04d}"))
            self._span(ax, start, end, color, "_nolegend_", alpha=0.14)
            center = (start + end) / 2.0
            center_px = (center - int(x_start)) / max(1, len(y) - 1) * px
            lane = next(
                (
                    i
                    for i, right_edge in enumerate(lane_last_right)
                    if center_px - label_width_px / 2.0 > right_edge + 4.0
                ),
                None,
            )
            if lane is None:
                lane = int(np.argmin(lane_last_right))
            text_center_px = max(
                label_width_px / 2.0, min(px - label_width_px / 2.0, center_px)
            )
            lane_last_right[lane] = text_center_px + label_width_px / 2.0
            text_x = int(x_start) + text_center_px / px * max(1, len(y) - 1)
            label_y = 0.88 - lane * 0.09
            ax.annotate(
                f"{gid}/{candidate_id}\n[{start},{end}]",
                xy=(center, 0.985),
                xycoords=("data", "axes fraction"),
                xytext=(text_x, label_y),
                textcoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=self.config.global_label_fontsize,
                color="#0f172a",
                bbox={
                    "facecolor": "white",
                    "edgecolor": color,
                    "alpha": 0.82,
                    "linewidth": 0.6,
                    "pad": 1.5,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": color,
                    "linewidth": 0.55,
                    "alpha": 0.8,
                },
                annotation_clip=True,
            )
        ax.text(
            0.002,
            0.985,
            "Anomaly Screening Candidate proposals | intervals shown inside Candidate blocks",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=max(7.0, self.config.global_label_fontsize),
            color="#334155",
        )
        ax.set_xlim(int(x_start) - 0.5, int(x_start) + len(y) - 0.5)
        ax.margins(x=0.005, y=0.04)
        fig.tight_layout()
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=self.config.dpi)
        plot_bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.close(fig)
        return {
            "path": str(out),
            "width_px": canvas_width,
            "height_px": canvas_height,
            "plot_width_px": float(round(plot_bbox.width * self.config.dpi, 2)),
            "plot_height_px": float(round(plot_bbox.height * self.config.dpi, 2)),
            "signal_length": int(
                signal_length if signal_length is not None else len(y)
            ),
            "displayed_length": int(len(y)),
            "candidate_count": int(len(candidates)),
            "downsampled": bool(len(y) > px),
            "aggregation": (
                "min_max_extrema_envelope_per_pixel_column" if len(y) > px else "none"
            ),
            "extrema_strategy": "retain finite min and max per pixel bucket",
            "candidate_annotation_strategy": "direct in-block Group/Candidate ID plus [start,end]",
            "context_range": [int(x_start), int(x_start) + max(0, len(y) - 1)],
            "plot_role": plot_role,
        }

    def navigation_overview(
        self,
        values: Sequence[float],
        windows: Sequence[Mapping[str, Any]],
        candidates: Sequence[Mapping[str, Any]],
        out_path: str | Path,
        *,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Render an ultra-long navigation-only overview at fixed 4096x512."""
        nav = [
            {
                "start": int(w["start"]),
                "end": int(w["end"]),
                "group_id": str(w.get("window_id", "W")),
                "candidate_id": str(w.get("window_id", "W")),
            }
            for w in windows
        ]
        meta = self.global_signal_plot(
            values,
            nav + list(candidates),
            out_path,
            width_px=4096,
            height_px=512,
            title=title or "Global Navigation Overview | macro context only",
            plot_role="navigation",
        )
        meta.update(
            {
                "navigation_only": True,
                "navigation_semantics": [
                    "macro context",
                    "overall trend",
                    "regime",
                    "large-scale periodicity",
                    "anchor window positions",
                    "candidate density",
                ],
                "local_anomaly_judgment_allowed": False,
                "window_count": len(windows),
            }
        )
        return meta

    def focused_candidate_plot(
        self,
        values: Sequence[float],
        start: int,
        end: int,
        out_path: str | Path,
        *,
        candidates: Sequence[Mapping[str, Any]] = (),
        title: str | None = None,
    ) -> dict[str, Any]:
        import matplotlib.pyplot as plt

        y = np.asarray(values, dtype=float).reshape(-1)
        lo = max(0, int(start) - self.config.focus_context_points)
        hi = min(len(y) - 1, int(end) + self.config.focus_context_points)
        x = np.arange(lo, hi + 1, dtype=np.int64)
        displayed_steps = int(hi - lo + 1)
        fig, ax, canvas_width, canvas_height = self._pixel_exact_axis(
            plot_width_px=displayed_steps,
            plot_height_px=self.config.detail_plot_height_px,
            title=title or f"Focused Candidate Plot | [{start}, {end}]",
        )
        ax.plot(
            x,
            y[lo : hi + 1],
            color=self.config.line_color,
            linewidth=0.9,
            label="canonical signal",
        )
        ax.set_xlim(lo - 0.5, hi + 0.5)
        self._span(
            ax, int(start), int(end), self.config.active_color, "Candidate", alpha=0.22
        )
        for cand in candidates:
            cs, ce = int(cand["start"]), int(cand["end"])
            if cs == int(start) and ce == int(end):
                continue
            if ce >= lo and cs <= hi:
                self._span(
                    ax,
                    max(lo, cs),
                    min(hi, ce),
                    self.config.candidate_color,
                    "Candidate",
                    alpha=0.10,
                )
        if int(end) - int(start) + 1 <= 4:
            ax.plot(
                [int(start), int(end)],
                [y[int(start)], y[int(end)]],
                color=self.config.active_color,
                marker="o",
                markersize=5,
                linewidth=2,
                label="candidate location marker",
            )
        ax.legend(loc="upper right", fontsize=8)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=self.config.dpi)
        plot_bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.close(fig)
        plot_width = float(round(plot_bbox.width * self.config.dpi, 2))
        return {
            "path": str(out),
            "width_px": int(canvas_width),
            "height_px": int(canvas_height),
            "plot_width_px": plot_width,
            "plot_height_px": float(round(plot_bbox.height * self.config.dpi, 2)),
            "signal_length": int(len(y)),
            "candidate_count": int(len(candidates)),
            "displayed_steps": displayed_steps,
            "pixels_per_timestep": float(plot_width / displayed_steps),
            "downsampled": False,
            "aggregation": "none",
            "extrema_strategy": "all canonical samples rendered; explicit marker for short candidate",
            "context_range": [int(lo), int(hi)],
        }

    def query_reference_comparison(
        self,
        values: Sequence[float],
        query: Mapping[str, Any],
        references: Sequence[Mapping[str, Any]],
        out_path: str | Path,
        *,
        mode: str = "shared_xy",
        context_points: int | None = None,
    ) -> dict[str, Any]:
        import matplotlib.pyplot as plt

        y = np.asarray(values, dtype=float).reshape(-1)
        q0, q1 = int(query["start"]), int(query["end"])
        context = (
            self.config.context_points
            if context_points is None
            else max(0, int(context_points))
        )
        core_spans = [q1 - q0 + 1] + [
            int(ref["end"]) - int(ref["start"]) + 1 for ref in references
        ]
        displayed_steps = min(len(y), max(core_spans) + 2 * context)
        qlo, qhi = self._centered_range((q0 + q1) / 2.0, displayed_steps, len(y))
        series: list[tuple[str, int, int, int, int, str]] = [
            ("Query", qlo, qhi, q0, q1, self.config.active_color)
        ]
        for i, ref in enumerate(references):
            r0, r1 = int(ref["start"]), int(ref["end"])
            lo, hi = self._centered_range((r0 + r1) / 2.0, displayed_steps, len(y))
            series.append(
                (
                    str(ref.get("reference_id", f"Reference {i+1}")),
                    lo,
                    hi,
                    r0,
                    r1,
                    self.config.reference_color,
                )
            )

        left = self.config.detail_left_margin_px
        right = self.config.detail_right_margin_px
        top = self.config.detail_top_margin_px
        bottom = self.config.detail_bottom_margin_px
        panel_h = self.config.comparison_panel_height_px
        gap = self.config.comparison_panel_gap_px
        canvas_width = left + displayed_steps + right
        canvas_height = (
            top + bottom + len(series) * panel_h + max(0, len(series) - 1) * gap
        )
        fig = plt.figure(
            figsize=(canvas_width / self.config.dpi, canvas_height / self.config.dpi),
            dpi=self.config.dpi,
        )
        fig.suptitle(
            "Query–Reference Context Comparison | shared x/y scale",
            fontsize=12,
            y=1.0 - 14 / canvas_height,
        )
        segments = [y[lo : hi + 1] for _, lo, hi, _, _, _ in series]
        shared_ylim = self._shared_ylim(segments)
        axes: list[Any] = []
        for panel_index, (name, lo, hi, core0, core1, color) in enumerate(series):
            panel_bottom = bottom + (len(series) - 1 - panel_index) * (panel_h + gap)
            ax = fig.add_axes(
                [
                    left / canvas_width,
                    panel_bottom / canvas_height,
                    displayed_steps / canvas_width,
                    panel_h / canvas_height,
                ]
            )
            axes.append(ax)
            xx = np.arange(lo, hi + 1, dtype=np.int64)
            ax.plot(xx, y[lo : hi + 1], color=self.config.line_color, linewidth=0.85)
            ax.set_xlim(lo - 0.5, hi + 0.5)
            ax.set_ylim(*shared_ylim)
            ax.axvspan(core0, core1, color=color, alpha=0.20)
            ax.axvline(core0, color=color, linewidth=0.8)
            ax.axvline(core1, color=color, linewidth=0.8)
            ax.set_ylabel(name + "\n(normality unknown)" if name != "Query" else name)
            ax.grid(True, alpha=0.2)
            ax.set_xlabel("absolute integer x index")
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=self.config.dpi)
        bbox = axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.close(fig)
        plot_width = float(round(bbox.width * self.config.dpi, 2))
        return {
            "path": str(out),
            "width_px": int(canvas_width),
            "height_px": int(canvas_height),
            "plot_width_px": plot_width,
            "plot_height_px": float(round(bbox.height * self.config.dpi, 2)),
            "signal_length": int(len(y)),
            "candidate_count": 1,
            "reference_count": int(len(references)),
            "displayed_steps_per_panel": int(displayed_steps),
            "pixels_per_timestep": float(plot_width / displayed_steps),
            "x_scale_consistent": True,
            "y_scale_consistent": True,
            "shared_y_limits": [float(shared_ylim[0]), float(shared_ylim[1])],
            "downsampled": False,
            "aggregation": "none",
            "extrema_strategy": "all canonical samples rendered at one plot pixel per time step",
            "context_range": {
                "query": [int(qlo), int(qhi)],
                "references": {
                    name: [int(lo), int(hi)] for name, lo, hi, _, _, _ in series[1:]
                },
            },
        }


__all__ = ["RendererConfig", "SignalRenderer"]
