"""Shared CGRaw/Vit4TS time-series image rendering helpers.

This module is a thin wrapper around the runtime renderer in
``src/preprocessing/preprocess.py``.  It exists so detection code and offline
semantic-alignment tools call the same renderer instead of maintaining two
near-duplicate matplotlib implementations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple
import warnings

import numpy as np


RENDERER_NAME = "screening_visual_screening_runtime_line_renderer"
RENDERER_SOURCE_FILE = "src/preprocessing/preprocess.py"
RENDERER_FUNCTION = "draw_image"
PREPROCESS_SOURCE_FILE = "src/preprocessing/preprocess.py"
PREPROCESS_FUNCTION = "preprocess_time_series"


def make_screening_plot_params(
    standardize: bool = True,
) -> Tuple[str, int, str, float, str, Optional[Tuple[int, int]]]:
    """Return the plot style used by the current CGRaw runtime."""

    return ("-", 1, "*", 0.1, "black", (0, 1) if standardize else None)


def renderer_config(
    *,
    image_size: Tuple[int, int] = (224, 224),
    dpi: int = 100,
    standardize: bool = True,
    clip_model_name: str = "ViT-B-16",
) -> Dict[str, Any]:
    """Return reproducibility metadata for the shared runtime renderer."""

    plot_params = make_screening_plot_params(standardize)
    return {
        "renderer_name": RENDERER_NAME,
        "renderer_source_file": RENDERER_SOURCE_FILE,
        "renderer_function": RENDERER_FUNCTION,
        "preprocess_source_file": PREPROCESS_SOURCE_FILE,
        "preprocess_function": PREPROCESS_FUNCTION,
        "detrend_enabled": bool(standardize),
        "detrend_method": "scipy.signal.detrend" if standardize else "",
        "normalization_enabled": bool(standardize),
        "normalization_method": "global min-max after detrend" if standardize else "",
        "output_image_width": int(image_size[1]),
        "output_image_height": int(image_size[0]),
        "dpi": int(dpi),
        "line_color": plot_params[4],
        "background_color": "white",
        "axis_visible": False,
        "marker": plot_params[2],
        "marker_size": plot_params[3],
        "line_width": plot_params[1],
        "clip_model_name": clip_model_name,
    }


def render_line_image_tensor(
    *,
    series_id: str,
    save_path: Path,
    time_series: Sequence[float],
    time_points: Sequence[float],
    image_size: Tuple[int, int] = (224, 224),
    dpi: int = 100,
    plot_params: Optional[tuple] = None,
    save_image: bool = False,
) -> Optional[np.ndarray]:
    """Render one window using the exact runtime ``draw_image`` function."""

    from preprocessing.preprocess import draw_image

    params = (
        plot_params if plot_params is not None else make_screening_plot_params(True)
    )
    return draw_image(
        series_id=series_id,
        save_path=str(save_path),
        time_series=np.asarray(time_series, dtype=float),
        time_points=np.asarray(time_points, dtype=float),
        override=True,
        save_image=save_image,
        image_size=image_size,
        dpi=dpi,
        plot_params=params,
    )


def render_windowed_line_images(
    *,
    base_series_id: str,
    save_path: Path,
    time_series: Sequence[float],
    time_points: Sequence[float],
    window_size: int,
    step_size: int,
    image_size: Tuple[int, int] = (224, 224),
    dpi: int = 100,
    plot_params: Optional[tuple] = None,
    save_image: bool = False,
) -> bool:
    """Render sliding-window images through the runtime ``draw_windowed_images``."""

    from preprocessing.preprocess import draw_windowed_images

    params = (
        plot_params if plot_params is not None else make_screening_plot_params(True)
    )
    return bool(
        draw_windowed_images(
            base_series_id=base_series_id,
            save_path=str(save_path),
            time_series=np.asarray(time_series, dtype=float),
            time_points=np.asarray(time_points, dtype=float),
            window_size=int(window_size),
            step_size=int(step_size),
            override=True,
            save_image=save_image,
            image_size=image_size,
            dpi=int(dpi),
            plot_params=params,
        )
    )


def render_window_array_line_images(
    *,
    base_series_id: str,
    save_path: Path,
    windows: np.ndarray,
    image_size: Tuple[int, int] = (224, 224),
    dpi: int = 100,
    plot_params: Optional[tuple] = None,
) -> bool:
    """Render an explicit array of windows and save the runtime npy bundle."""

    save_path.mkdir(parents=True, exist_ok=True)
    windows = np.asarray(windows, dtype=float)
    params = (
        plot_params if plot_params is not None else make_screening_plot_params(True)
    )
    aggregated = []
    for idx, window in enumerate(windows, start=1):
        img_tensor = render_line_image_tensor(
            series_id=f"{base_series_id}_{idx}",
            save_path=save_path,
            time_series=window,
            time_points=np.arange(len(window)),
            image_size=image_size,
            dpi=dpi,
            plot_params=params,
            save_image=False,
        )
        if img_tensor is not None:
            aggregated.append(img_tensor)
    if not aggregated:
        warnings.warn("No window-array images were generated.")
        return False
    arr = np.stack(aggregated)
    np.save(save_path / f"{base_series_id}_line_img.npy", arr)
    return True
