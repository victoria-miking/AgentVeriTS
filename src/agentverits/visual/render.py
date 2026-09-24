from __future__ import annotations

from pathlib import Path
from typing import Sequence
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor

from ..config import RenderConfig
from ..types import Interval, VisualCandidate


def preprocess_series(values: Sequence[float]) -> np.ndarray:
    from scipy.signal import detrend

    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size < 2:
        return np.zeros_like(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("screening requires finite values")
    y = detrend(x)
    lo, hi = float(np.min(y)), float(np.max(y))
    if hi - lo < 1e-12:
        return np.zeros_like(y, dtype=np.float32)
    return ((y - lo) / (hi - lo)).astype(np.float32)


def render_window_tensor(values: Sequence[float], image_size: int = 224, dpi: int = 100) -> np.ndarray:
    """Render a window as the axis-free line image consumed by the visual encoder."""
    y = np.asarray(values, dtype=float).reshape(-1)
    x = np.arange(len(y))
    fig = plt.figure(figsize=(image_size / dpi, image_size / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(x, y, linestyle="-", linewidth=1, marker="*", markersize=0.1, color="black")
    ax.set_xlim(0, max(1, len(y) - 1))
    ax.set_ylim(0, 1)
    ax.axis("off")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    return to_tensor(image).numpy().astype(np.float32)


def _extrema_envelope(values: np.ndarray, columns: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(values)
    if n <= columns:
        return np.arange(n), values
    edges = np.linspace(0, n, columns + 1, dtype=int)
    xs: list[int] = []
    ys: list[float] = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        segment = values[a:b]
        finite = np.flatnonzero(np.isfinite(segment))
        if not len(finite):
            continue
        lo = int(finite[np.argmin(segment[finite])])
        hi = int(finite[np.argmax(segment[finite])])
        for local in sorted(set((lo, hi))):
            xs.append(a + local)
            ys.append(float(segment[local]))
    return np.asarray(xs), np.asarray(ys)


class SeriesRenderer:
    def __init__(self, config: RenderConfig | None = None) -> None:
        self.config = config or RenderConfig()

    def global_plot(
        self,
        values: Sequence[float],
        candidates: Sequence[VisualCandidate],
        out_path: str | Path,
        *,
        title: str = "Global time-series view with visual candidate windows",
    ) -> Path:
        y = np.asarray(values, dtype=float).reshape(-1)
        width, height = self.config.global_width_px, self.config.global_height_px
        fig, ax = plt.subplots(figsize=(width / self.config.dpi, height / self.config.dpi), dpi=self.config.dpi)
        x_draw, y_draw = _extrema_envelope(y, max(512, int(width * 0.92)))
        ax.plot(x_draw, y_draw, color=self.config.line_color, linewidth=0.8)
        for cand in candidates:
            s, e = cand.interval.start, cand.interval.end
            ax.axvspan(s, e, color=self.config.candidate_color, alpha=0.22)
            ax.axvline(s, color=self.config.candidate_color, linewidth=0.65, alpha=0.8)
            ax.axvline(e, color=self.config.candidate_color, linewidth=0.65, alpha=0.8)
            ax.annotate(
                f"{cand.candidate_id} [{s},{e}]",
                xy=((s + e) / 2.0, 0.96),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": self.config.candidate_color, "alpha": 0.78, "pad": 1.2},
            )
        ax.set_title(title)
        ax.set_xlabel("time index")
        ax.set_ylabel("value")
        ax.grid(alpha=0.18)
        ax.margins(x=0.002)
        fig.tight_layout()
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.config.dpi)
        plt.close(fig)
        return path

    def local_plot(
        self,
        values: Sequence[float],
        interval: Interval,
        out_path: str | Path,
        *,
        context_points: int = 256,
        local_y: bool = True,
    ) -> Path:
        y = np.asarray(values, dtype=float).reshape(-1)
        lo = max(0, interval.start - int(context_points))
        hi = min(len(y) - 1, interval.end + int(context_points))
        x = np.arange(lo, hi + 1)
        fig, ax = plt.subplots(figsize=(12, 4), dpi=self.config.dpi)
        ax.plot(x, y[lo:hi + 1], color=self.config.line_color, linewidth=0.9)
        ax.axvspan(interval.start, interval.end, color=self.config.focus_color, alpha=0.22)
        if not local_y and np.isfinite(y).any():
            lo_y, hi_y = float(np.nanmin(y)), float(np.nanmax(y))
            pad = max((hi_y - lo_y) * 0.03, 1e-8)
            ax.set_ylim(lo_y - pad, hi_y + pad)
        ax.set_title(f"Evidence view [{interval.start}, {interval.end}]")
        ax.set_xlabel("time index")
        ax.set_ylabel("value")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.config.dpi)
        plt.close(fig)
        return path

    def comparison_plot(
        self,
        values: Sequence[float],
        query: Interval,
        references: Sequence[Interval],
        out_path: str | Path,
        *,
        context_points: int = 128,
        shared_y: bool = True,
    ) -> Path:
        y = np.asarray(values, dtype=float).reshape(-1)
        rows = [query, *references]
        segments: list[tuple[np.ndarray, np.ndarray, Interval]] = []
        all_values: list[np.ndarray] = []
        for item in rows:
            lo = max(0, item.start - context_points)
            hi = min(len(y) - 1, item.end + context_points)
            xx = np.arange(lo, hi + 1)
            yy = y[lo:hi + 1]
            segments.append((xx, yy, item))
            all_values.append(yy[np.isfinite(yy)])
        shared_limits = None
        finite = np.concatenate([v for v in all_values if len(v)]) if any(len(v) for v in all_values) else np.array([])
        if shared_y and len(finite):
            a, b = float(np.min(finite)), float(np.max(finite))
            pad = max((b - a) * 0.04, 1e-8)
            shared_limits = (a - pad, b + pad)
        fig, axes = plt.subplots(len(rows), 1, figsize=(12, max(3, 2.6 * len(rows))), dpi=self.config.dpi)
        if len(rows) == 1:
            axes = [axes]
        for idx, (ax, (xx, yy, item)) in enumerate(zip(axes, segments)):
            ax.plot(xx, yy, color=self.config.line_color, linewidth=0.85)
            ax.axvspan(item.start, item.end, color=self.config.focus_color if idx == 0 else "#15803d", alpha=0.20)
            if shared_limits:
                ax.set_ylim(*shared_limits)
            ax.set_ylabel("query" if idx == 0 else f"ref-{idx}")
            ax.grid(alpha=0.18)
        axes[-1].set_xlabel("time index")
        fig.tight_layout()
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.config.dpi)
        plt.close(fig)
        return path
