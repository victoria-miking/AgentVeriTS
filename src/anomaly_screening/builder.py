#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建 ANOMALY_SCREENING_FROZEN 的完整、可回放离线缓存。

该 runner 只实现一件事：按 frozen 配置运行真实的窗口渲染、ViT-B/16
编码、raw Top-K=16 检索、retain-4/no-aggregation patch-bank scoring，并将
每个 query 的全过程写入磁盘。缓存按 signal/scale 原子落盘，支持中断后
``--resume``，不会因为后续阶段改变而重新渲染或重新编码。
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from preprocessing import datasets
from evaluation.metrics import evaluate_one_score
from evaluation.evaluate import evaluate_intervals
from preprocessing.data_utils import intervals_from_indices
from models.decision import compute_detection_intervals, robust_positive_z
from anomaly_screening.retrieval import (
    select_reference_indices_screening,
    compute_trimmed_patch_bank_distance,
    select_retained_reference_indices,
)
from models.clip_vision import CLIP_AD  # noqa: E402
from models.model_utils import (  # noqa: E402
    aggregate_anomaly_map,
    align_anomaly_vector,
    harmonic_aggregation,
    stitch_anomaly_maps,
)
from models.timeseries_image_renderer import render_line_image_tensor  # noqa: E402
from preprocessing.data_utils import orion_to_internal  # noqa: E402
from preprocessing.preprocess import preprocess_time_series  # noqa: E402
from anomaly_screening.retrieval import filter_reference_candidates  # noqa: E402
from anomaly_screening.audit import (  # noqa: E402
    config_fingerprint,
    environment_fingerprint,
    git_fingerprint,
    load_frozen_config,
)
from anomaly_screening.cache import CACHE_SCHEMA  # noqa: E402


DATASETS_9 = [
    "realTweets",
    "realAWSCloudwatch",
    "realAdExchange",
    "realTraffic",
    "artificialWithAnomaly",
    "SMAP",
    "MSL",
    "WSD",
    "IOPS",
]
SCALES = (224, 448, 672)
ALPHAS = (0.1, 0.01, 0.001)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=True, indent=2, allow_nan=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _atomic_npz(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.npz")
    if temp.exists():
        temp.unlink()
    np.savez_compressed(temp, **payload)
    os.replace(temp, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, torch.Tensor)):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _render_windows(
    windows: np.ndarray,
    out_dir: Path,
    *,
    image_size: tuple[int, int],
    standardize: bool,
    save_png: bool,
) -> np.ndarray:
    """Render every selected window with the production draw_image function."""

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_params = ("-", 1, "*", 0.1, "black", (0, 1) if standardize else None)
    images: list[np.ndarray] = []
    # draw_image reports every PNG path to stdout; the cache already records
    # paths in its manifest, so silence only this high-volume progress noise.
    with open(os.devnull, "w", encoding="utf-8") as null_stream:
        for index, window in enumerate(np.asarray(windows, dtype=np.float32)):
            output_context = (
                contextlib.redirect_stdout(null_stream)
                if save_png
                else contextlib.nullcontext()
            )
            with output_context:
                image = render_line_image_tensor(
                    series_id=f"window_{index:06d}",
                    save_path=out_dir,
                    time_series=window,
                    time_points=np.arange(len(window)),
                    image_size=image_size,
                    dpi=100,
                    plot_params=plot_params,
                    save_image=bool(save_png),
                )
            if image is None:
                raise RuntimeError(f"renderer returned no image for window {index}")
            images.append(np.asarray(image, dtype=np.float32))
    result = (
        np.stack(images, axis=0)
        if images
        else np.zeros((0, 3, *image_size), dtype=np.float32)
    )
    np.save(out_dir / "window_images.npy", result)
    return result


def _encode_images(
    model: Any, images: np.ndarray, device: torch.device, batch_size: int
) -> dict[str, np.ndarray]:
    outputs: dict[str, list[np.ndarray]] = {
        "large": [],
        "mid": [],
        "patch": [],
        "class": [],
    }
    large_mask = None
    mid_mask = None
    with torch.no_grad():
        for start in range(0, len(images), max(1, int(batch_size))):
            batch = torch.from_numpy(images[start : start + int(batch_size)]).to(device)
            large, mid, patch, classes, large_mask, mid_mask = model.encode_image(
                batch, 16
            )
            if patch.ndim == 4 and patch.shape[2] == 1:
                patch = patch.squeeze(2)
            outputs["large"].append(large.detach().cpu().numpy().astype(np.float32))
            outputs["mid"].append(mid.detach().cpu().numpy().astype(np.float32))
            outputs["patch"].append(patch.detach().cpu().numpy().astype(np.float32))
            outputs["class"].append(
                classes.reshape(classes.shape[0], -1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    if large_mask is None or mid_mask is None:
        raise ValueError("cannot encode an empty image bank")
    return {
        "large_tokens": np.concatenate(outputs["large"], axis=0),
        "mid_tokens": np.concatenate(outputs["mid"], axis=0),
        "patch_tokens": np.concatenate(outputs["patch"], axis=0),
        "class_tokens": np.concatenate(outputs["class"], axis=0),
        "large_mask": large_mask.detach().cpu().numpy().astype(np.int16),
        "mid_mask": mid_mask.detach().cpu().numpy().astype(np.int16),
    }


def _trim_details(
    refs: torch.Tensor, retained_count: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic local retained indices and median-center distances."""

    if refs.ndim == 4 and refs.shape[2] == 1:
        refs = refs.squeeze(2)
    if refs.shape[0] == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32)
    flat = refs.reshape(refs.shape[0], -1)
    center = torch.median(flat, dim=0).values.reshape(1, -1)
    distances = 1.0 - torch.matmul(
        F.normalize(flat, dim=-1),
        F.normalize(center, dim=-1).T,
    ).squeeze(-1)
    keep = select_retained_reference_indices(
        refs, retained_count=int(retained_count), trim_ratio=0.75
    )
    return keep.detach().cpu(), distances.detach().cpu()


def _as_tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value)).to(device)


def _score_scale(
    *,
    model: Any,
    df: pd.DataFrame,
    dataset: str,
    signal_id: str,
    scale: int,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    max_windows: int,
    save_png: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    values, source_timestamps = orion_to_internal(df)
    values_proc = preprocess_time_series(values)
    step = max(1, int(int(scale) / 4.0))
    all_starts = np.arange(0, len(values_proc) - int(scale) + 1, step, dtype=np.int64)
    if len(all_starts) < 2:
        raise ValueError(f"too few windows for scale {scale}: {len(all_starts)}")

    selected_positions = np.arange(len(all_starts), dtype=np.int64)
    if int(max_windows) > 0 and len(selected_positions) > int(max_windows):
        selected_positions = np.linspace(
            0, len(selected_positions) - 1, int(max_windows), dtype=np.int64
        )
    starts = all_starts[selected_positions]
    window_ids = selected_positions.copy()
    windows = np.stack(
        [values_proc[start : start + int(scale)] for start in starts]
    ).astype(np.float32)

    scale_root = out_dir / str(dataset) / str(signal_id) / f"scale_{int(scale)}"
    images = _render_windows(
        windows,
        scale_root / "images",
        image_size=(224, 224),
        standardize=True,
        save_png=save_png,
    )
    encoded = _encode_images(model, images, device, batch_size)
    classes = _as_tensor(encoded["class_tokens"], device)
    patch_tokens = _as_tensor(encoded["patch_tokens"], device)
    mid_tokens = _as_tensor(encoded["mid_tokens"], device)
    large_tokens = _as_tensor(encoded["large_tokens"], device)
    ids_tensor = torch.as_tensor(window_ids, dtype=torch.long, device=device)
    candidate_ids: list[np.ndarray] = []
    valid_ids: list[np.ndarray] = []
    candidate_sims: list[np.ndarray] = []
    retained_ids: list[np.ndarray] = []
    retained_local: list[np.ndarray] = []
    trim_distances: list[np.ndarray] = []
    mid_trim_distances: list[np.ndarray] = []
    large_trim_distances: list[np.ndarray] = []
    mid_retained_local: list[np.ndarray] = []
    large_retained_local: list[np.ndarray] = []
    patch_scores: list[np.ndarray] = []
    mid_scores: list[np.ndarray] = []
    large_scores: list[np.ndarray] = []
    no_reference_queries: list[int] = []

    for local_query in range(len(starts)):
        query_id = int(window_ids[local_query])
        valid_local, _ = filter_reference_candidates(
            candidate_starts=window_ids * int(step),
            query_start=query_id * int(step),
            window_length=int(scale),
            policy="non_overlap",
        )
        valid_local_tensor = torch.as_tensor(
            valid_local, dtype=torch.long, device=device
        )
        selected_local = select_reference_indices_screening(
            coarse_bank=classes,
            query_coarse=classes[local_query],
            raw_class_bank=classes,
            query_raw_class=classes[local_query],
            bank_window_ids=ids_tensor,
            query_window_id=query_id,
            top_m=max(64, len(starts)),
            top_k=16,
            retrieval_mode="coarse",
            temporal_gap=0,
            overlap_policy="non_overlap",
            window_size=int(scale),
            step_size=int(step),
        )
        selected_local = torch.as_tensor(
            selected_local, dtype=torch.long, device=device
        ).reshape(-1)
        if selected_local.numel() == 0:
            # Very short signals can have no valid non-overlap reference for a
            # boundary query. Preserve the query trace and emit a neutral,
            # finite score map instead of dropping the entire signal cache.
            no_reference_queries.append(int(local_query))
            candidate_ids.append(np.empty(0, dtype=np.int64))
            valid_ids.append(
                ids_tensor.index_select(0, valid_local_tensor)
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
            )
            candidate_sims.append(np.empty(0, dtype=np.float32))
            retained_ids.append(np.empty(0, dtype=np.int64))
            retained_local.append(np.empty(0, dtype=np.int64))
            trim_distances.append(np.empty(0, dtype=np.float32))
            mid_retained_local.append(np.empty(0, dtype=np.int64))
            large_retained_local.append(np.empty(0, dtype=np.int64))
            mid_trim_distances.append(np.empty(0, dtype=np.float32))
            large_trim_distances.append(np.empty(0, dtype=np.float32))
            patch_scores.append(np.zeros(int(patch_tokens.shape[1]), dtype=np.float32))
            mid_scores.append(np.zeros(int(mid_tokens.shape[1]), dtype=np.float32))
            large_scores.append(np.zeros(int(large_tokens.shape[1]), dtype=np.float32))
            continue
        ref_patch = patch_tokens.index_select(0, selected_local)
        keep_local_tensor, distances = _trim_details(ref_patch, retained_count=4)
        keep_local_tensor = keep_local_tensor.to(device)
        mid_keep_tensor, mid_distances = _trim_details(
            mid_tokens.index_select(0, selected_local), retained_count=4
        )
        large_keep_tensor, large_distances = _trim_details(
            large_tokens.index_select(0, selected_local), retained_count=4
        )
        selected_ids = ids_tensor.index_select(0, selected_local)
        kept_ids = selected_ids.index_select(0, keep_local_tensor)
        sims = F.cosine_similarity(
            classes.index_select(0, selected_local),
            classes[local_query].unsqueeze(0),
            dim=-1,
        )

        candidate_ids.append(selected_ids.detach().cpu().numpy().astype(np.int64))
        valid_ids.append(
            ids_tensor.index_select(0, valid_local_tensor)
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        candidate_sims.append(sims.detach().cpu().numpy().astype(np.float32))
        retained_ids.append(kept_ids.detach().cpu().numpy().astype(np.int64))
        retained_local.append(keep_local_tensor.detach().cpu().numpy().astype(np.int64))
        trim_distances.append(distances.numpy().astype(np.float32))
        mid_retained_local.append(mid_keep_tensor.numpy().astype(np.int64))
        large_retained_local.append(large_keep_tensor.numpy().astype(np.int64))
        mid_trim_distances.append(mid_distances.numpy().astype(np.float32))
        large_trim_distances.append(large_distances.numpy().astype(np.float32))

        retained_patch = ref_patch
        patch_scores.append(
            compute_trimmed_patch_bank_distance(
                patch_tokens[local_query],
                retained_patch,
                trim_ratio=0.75,
                retained_count=4,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        mid_scores.append(
            compute_trimmed_patch_bank_distance(
                mid_tokens[local_query],
                mid_tokens.index_select(0, selected_local),
                trim_ratio=0.75,
                retained_count=4,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        large_scores.append(
            compute_trimmed_patch_bank_distance(
                large_tokens[local_query],
                large_tokens.index_select(0, selected_local),
                trim_ratio=0.75,
                retained_count=4,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    patch_score_tensor = torch.from_numpy(np.stack(patch_scores)).to(device)
    mid_score_tensor = torch.from_numpy(np.stack(mid_scores)).to(device)
    large_score_tensor = torch.from_numpy(np.stack(large_scores)).to(device)
    large_mask = torch.from_numpy(encoded["large_mask"]).to(device)
    mid_mask = torch.from_numpy(encoded["mid_mask"]).to(device)
    patch_grid = int(round(float(patch_score_tensor.shape[1]) ** 0.5))
    if patch_grid * patch_grid != int(patch_score_tensor.shape[1]):
        raise ValueError(
            f"patch token count is not square: {patch_score_tensor.shape[1]}"
        )
    large_map = harmonic_aggregation(
        (len(starts), patch_grid, patch_grid),
        large_score_tensor,
        large_mask,
    )
    mid_map = harmonic_aggregation(
        (len(starts), patch_grid, patch_grid),
        mid_score_tensor,
        mid_mask,
    )
    patch_map = patch_score_tensor.reshape(len(starts), patch_grid, patch_grid)
    # Keep the legacy runner's float64 harmonic/interpolation path intact.
    map_grid = torch.nan_to_num(
        (large_map + mid_map + patch_map) / 3.0, nan=0.0, posinf=0.0, neginf=0.0
    )
    dense_maps = F.interpolate(
        map_grid.unsqueeze(1), size=(224, 224), mode="bilinear"
    ).squeeze(1)
    dense_np = dense_maps.detach().cpu().numpy().astype(np.float64)
    window_vectors = np.stack(
        [aggregate_anomaly_map(row, 0.25) for row in dense_np]
    ).astype(np.float64)
    stitched = stitch_anomaly_maps(dense_np, 4.0, 0.25).astype(np.float64)
    aligned = align_anomaly_vector(
        stitched, len(values_proc), int(scale), int(step), len(starts)
    ).astype(np.float64)

    bundle = {
        "window_ids": window_ids.astype(np.int64),
        "window_starts": starts.astype(np.int64),
        "window_values": windows.astype(np.float32),
        "source_timestamps": np.asarray(source_timestamps),
        "window_size": np.asarray(int(scale), dtype=np.int64),
        "step_size": np.asarray(int(step), dtype=np.int64),
        "query_class_tokens": encoded["class_tokens"],
        "class_tokens": encoded["class_tokens"],
        "patch_tokens": encoded["patch_tokens"],
        "mid_tokens": encoded["mid_tokens"],
        "large_tokens": encoded["large_tokens"],
        "large_mask": encoded["large_mask"],
        "mid_mask": encoded["mid_mask"],
        "valid_reference_ids": np.asarray(valid_ids, dtype=object),
        "topk_indices": np.asarray(candidate_ids, dtype=object),
        "candidate_reference_ids": np.asarray(candidate_ids, dtype=object),
        "topk_similarities": np.asarray(candidate_sims, dtype=object),
        "candidate_retrieval_similarity": np.asarray(candidate_sims, dtype=object),
        "trim_retained_indices": np.asarray(retained_local, dtype=object),
        "retained_indices": np.asarray(retained_local, dtype=object),
        "retained_window_ids": np.asarray(retained_ids, dtype=object),
        "trim_distances": np.asarray(trim_distances, dtype=object),
        "mid_trim_retained_indices": np.asarray(mid_retained_local, dtype=object),
        "large_trim_retained_indices": np.asarray(large_retained_local, dtype=object),
        "mid_trim_distances": np.asarray(mid_trim_distances, dtype=object),
        "large_trim_distances": np.asarray(large_trim_distances, dtype=object),
        "no_reference_queries": np.asarray(no_reference_queries, dtype=np.int64),
        "patch_scores": np.stack(patch_scores).astype(np.float32),
        "mid_patch_scores": np.stack(mid_scores).astype(np.float32),
        "large_patch_scores": np.stack(large_scores).astype(np.float32),
        "patch_score_map": patch_map.detach().cpu().numpy().astype(np.float32),
        "large_score_map": large_map.detach().cpu().numpy().astype(np.float32),
        "mid_score_map": mid_map.detach().cpu().numpy().astype(np.float32),
        "window_anomaly_map": dense_np,
        "window_vectors": window_vectors,
        "stitched_score": stitched,
        "aligned_score": aligned,
    }
    bundle_path = scale_root / "artifacts.npz"
    _atomic_npz(bundle_path, bundle)
    trace = pd.DataFrame(
        {
            "query_window_id": window_ids.astype(np.int64),
            "query_start": starts.astype(np.int64),
            "query_end": (starts + int(scale)).astype(np.int64),
            "reference_status": [
                (
                    "no_non_overlap_reference"
                    if index in set(no_reference_queries)
                    else "ok"
                )
                for index in range(len(starts))
            ],
            "valid_reference_ids": [json.dumps(row.tolist()) for row in valid_ids],
            "topk_indices": [json.dumps(row.tolist()) for row in candidate_ids],
            "topk_similarities": [
                json.dumps([float(value) for value in row]) for row in candidate_sims
            ],
            "trim_retained_indices": [
                json.dumps(row.tolist()) for row in retained_local
            ],
            "retained_window_ids": [json.dumps(row.tolist()) for row in retained_ids],
            "trim_distances": [
                json.dumps([float(value) for value in row]) for row in trim_distances
            ],
            "patch_score_mean": np.asarray(np.stack(patch_scores)).mean(axis=1),
            "patch_score_max": np.asarray(np.stack(patch_scores)).max(axis=1),
            "window_score_mean": window_vectors.mean(axis=1),
            "window_score_max": window_vectors.max(axis=1),
        }
    )
    trace.to_csv(scale_root / "query_trace.csv", index=False)
    entry = {
        "scale": int(scale),
        "bundle": bundle_path.relative_to(
            out_dir / str(dataset) / str(signal_id)
        ).as_posix(),
        "images": (scale_root / "images")
        .relative_to(out_dir / str(dataset) / str(signal_id))
        .as_posix(),
        "query_trace": (scale_root / "query_trace.csv")
        .relative_to(out_dir / str(dataset) / str(signal_id))
        .as_posix(),
        "num_queries": int(len(starts)),
        "original_window_count": int(len(all_starts)),
        "window_sampling": (
            "all" if len(all_starts) == len(starts) else "uniform_deterministic"
        ),
        "candidate_count_min": int(min(len(row) for row in candidate_ids)),
        "candidate_count_max": int(max(len(row) for row in candidate_ids)),
        "retained_count_min": int(min(len(row) for row in retained_ids)),
        "retained_count_max": int(max(len(row) for row in retained_ids)),
        "no_reference_query_count": int(len(no_reference_queries)),
        "no_reference_queries": [int(value) for value in no_reference_queries],
        "shapes": {
            name: list(np.asarray(value).shape)
            for name, value in bundle.items()
            if name
            in {
                "patch_tokens",
                "topk_indices",
                "retained_indices",
                "patch_scores",
                "window_anomaly_map",
                "aligned_score",
            }
        },
    }
    return bundle, entry


def _build_signal(
    *,
    model: Any,
    df: pd.DataFrame,
    dataset: str,
    signal_id: str,
    out_dir: Path,
    project_root: Path,
    device: torch.device,
    batch_size: int,
    max_windows: int,
    save_png: bool,
    alphas: tuple[float, ...],
    frozen_config: Mapping[str, Any],
    repo: Mapping[str, Any],
) -> dict[str, Any]:
    values, source_timestamps = orion_to_internal(df)
    gt_intervals, labels, label_source = datasets.get_gt_intervals_and_labels(
        df, signal_id, datasets.load_anomalies(project_root / "anomalies.csv")
    )
    scale_scores: dict[int, np.ndarray] = {}
    scale_entries: dict[str, Any] = {}
    for scale in SCALES:
        payload, entry = _score_scale(
            model=model,
            df=df,
            dataset=dataset,
            signal_id=signal_id,
            scale=scale,
            out_dir=out_dir,
            device=device,
            batch_size=batch_size,
            max_windows=max_windows,
            save_png=save_png,
        )
        scale_scores[int(scale)] = np.asarray(payload["aligned_score"], dtype=float)
        # The bundle has already been atomically written by _score_scale.  Do
        # not keep the token/map arrays for all three scales in RAM while the
        # signal-level fusion and decision artifacts are produced.
        del payload
        scale_entries[str(int(scale))] = entry

    weights = np.ones(len(SCALES), dtype=float) / len(SCALES)
    normalized = {
        scale: robust_positive_z(values_) for scale, values_ in scale_scores.items()
    }
    fused = sum(
        float(weight) * normalized[scale] for scale, weight in zip(SCALES, weights)
    )
    decision_rows: list[dict[str, Any]] = []
    processed_rows: list[np.ndarray] = []
    for alpha in alphas:
        intervals, threshold, processed = repo["compute_detection_intervals"](
            score_vector=np.asarray(fused, dtype=float),
            alpha=float(alpha),
            method="mean",
            smoothing=True,
            sliding=False,
            anomaly_padding=0,
        )
        decision_rows.append(
            {
                "alpha": float(alpha),
                "threshold": float(threshold),
                "intervals": [[int(s), int(e)] for s, e in intervals],
            }
        )
        processed_rows.append(np.asarray(processed, dtype=np.float32))
    eval_row, per_alpha = evaluate_one_score(
        model_name="ANOMALY_SCREENING_FROZEN",
        scores=fused,
        timestamps=np.asarray(source_timestamps),
        labels=labels,
        gt_intervals_idx=gt_intervals,
        alphas=alphas,
        repo=dict(repo),
    )
    best_alpha = eval_row.get("Best_F1_alpha")
    if best_alpha is None or not np.isfinite(float(best_alpha)):
        best_alpha = float(alphas[0])
    selected = next(
        row
        for row in decision_rows
        if abs(float(row["alpha"]) - float(best_alpha)) < 1e-12
    )
    signal_root = out_dir / str(dataset) / str(signal_id)
    _atomic_npz(
        signal_root / "signal_artifacts.npz",
        {
            "values": np.asarray(values, dtype=np.float32),
            "source_timestamps": np.asarray(source_timestamps),
            "labels": np.asarray(labels, dtype=np.int8),
            "fused_score": np.asarray(fused, dtype=np.float32),
            "scale_scores": np.stack([scale_scores[scale] for scale in SCALES]).astype(
                np.float32
            ),
            "normalized_scale_scores": np.stack(
                [normalized[scale] for scale in SCALES]
            ).astype(np.float32),
            "scale_weights": np.asarray(weights, dtype=np.float32),
            "processed_scores": np.stack(processed_rows).astype(np.float32),
            "alphas": np.asarray(alphas, dtype=np.float64),
        },
    )
    _atomic_json(
        signal_root / "decision.json",
        {
            "alphas": decision_rows,
            "selected_alpha": float(best_alpha),
            "selected_intervals": selected["intervals"],
        },
    )
    pd.DataFrame(per_alpha).to_csv(signal_root / "per_alpha_metrics.csv", index=False)
    _atomic_json(signal_root / "metrics.json", _jsonable(eval_row))
    manifest = {
        "cache_schema": CACHE_SCHEMA,
        "cache_complete": True,
        "dataset": dataset,
        "signal_id": signal_id,
        "config": _jsonable(frozen_config),
        "config_fingerprint": config_fingerprint(frozen_config),
        "label_source": label_source,
        "n_points": int(len(values)),
        "n_anomaly_points": int(np.asarray(labels).sum()),
        "gt_intervals": _jsonable(gt_intervals),
        "scales": scale_entries,
        "signal_artifacts": "signal_artifacts.npz",
        "decision": {
            "bundle": "signal_artifacts.npz",
            "alphas": decision_rows,
            "selected_alpha": float(best_alpha),
            "selected_intervals": selected["intervals"],
        },
        "metrics": _jsonable(eval_row),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    _atomic_json(signal_root / "signal_manifest.json", manifest)
    return manifest


def build_cache(args: argparse.Namespace) -> Path:
    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir or project_root / "results/anomaly_screening").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen = load_frozen_config(ROOT / "configs" / "anomaly_screening.json")
    if (
        config_fingerprint(frozen)
        != "d6001e818c8c83d98641544c135b847670c4dbfb2c32c19d65c73bd3f5a73816"
    ):
        raise RuntimeError(
            "frozen config fingerprint changed; refusing to build an incompatible cache"
        )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    repo = {
        "compute_detection_intervals": compute_detection_intervals,
        "intervals_from_indices": intervals_from_indices,
        "evaluate_intervals": evaluate_intervals,
        "evaluate_affiliation_f1": None,
        "evaluate_vus_pr": None,
    }
    model = CLIP_AD(
        model_name="ViT-B-16",
        device=device,
        image_size=(224, 224),
        preprocess_mode="legacy",
    )
    model.eval()
    data_root = (
        Path(args.data_root) if hasattr(args, "data_root") else project_root / "data"
    )
    root_manifest: dict[str, Any] = {
        "cache_schema": CACHE_SCHEMA,
        "cache_name": "ANOMALY_SCREENING_CACHE",
        "datasets": list(args.datasets),
        "scales": list(SCALES),
        "config": frozen,
        "config_fingerprint": config_fingerprint(frozen),
        "git": git_fingerprint(project_root),
        "environment": environment_fingerprint(),
        "platform": platform.platform(),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "max_windows": int(args.max_windows),
        "save_png": bool(args.save_png),
        "entries": [],
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    manifest_path = out_dir / "manifest.json"
    if args.resume and manifest_path.is_file():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                old.get("cache_schema") == CACHE_SCHEMA
                and old.get("config_fingerprint") == root_manifest["config_fingerprint"]
            ):
                root_manifest["entries"] = list(old.get("entries", []))
        except Exception:
            pass
    done = {
        (row.get("dataset"), row.get("signal_id"))
        for row in root_manifest["entries"]
        if row.get("cache_complete")
    }
    for dataset in args.datasets:
        files = datasets.select_files(
            data_root / dataset, cases=None, max_files=args.max_signals
        )
        for path in files:
            signal_id = path.stem
            if (
                hasattr(args, "signal_roster")
                and (dataset, signal_id) not in args.signal_roster
            ):
                continue
            signal_manifest_path = (
                out_dir / str(dataset) / str(signal_id) / "signal_manifest.json"
            )
            if (
                args.resume
                and (dataset, signal_id) in done
                and signal_manifest_path.is_file()
            ):
                print(f"[SKIP] {dataset}/{signal_id}", flush=True)
                continue
            started = time.perf_counter()
            try:
                df = datasets.normalize_dataframe_columns(pd.read_csv(path))
                manifest = _build_signal(
                    model=model,
                    df=df,
                    dataset=dataset,
                    signal_id=signal_id,
                    out_dir=out_dir,
                    project_root=data_root,
                    device=device,
                    batch_size=args.batch_size,
                    max_windows=args.max_windows,
                    save_png=args.save_png,
                    alphas=tuple(float(v) for v in args.alphas),
                    frozen_config=frozen,
                    repo=repo,
                )
                manifest["runtime_sec"] = float(time.perf_counter() - started)
                _atomic_json(signal_manifest_path, manifest)
                root_manifest["entries"] = [
                    row
                    for row in root_manifest["entries"]
                    if (row.get("dataset"), row.get("signal_id"))
                    != (dataset, signal_id)
                ]
                root_manifest["entries"].append(manifest)
                _atomic_json(manifest_path, root_manifest)
                print(
                    f"[OK] {dataset}/{signal_id} scales={list(SCALES)} time={manifest['runtime_sec']:.1f}s",
                    flush=True,
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                failed = {
                    "dataset": dataset,
                    "signal_id": signal_id,
                    "cache_complete": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                root_manifest["entries"] = [
                    row
                    for row in root_manifest["entries"]
                    if (row.get("dataset"), row.get("signal_id"))
                    != (dataset, signal_id)
                ]
                root_manifest["entries"].append(failed)
                _atomic_json(manifest_path, root_manifest)
                print(f"[FAIL] {dataset}/{signal_id}: {failed['error']}", flush=True)
                if args.stop_on_error:
                    raise
    root_manifest["completed_signals"] = int(
        sum(bool(row.get("cache_complete")) for row in root_manifest["entries"])
    )
    root_manifest["failed_signals"] = int(
        sum(not bool(row.get("cache_complete")) for row in root_manifest["entries"])
    )
    root_manifest["cache_complete"] = (
        root_manifest["failed_signals"] == 0 and root_manifest["completed_signals"] > 0
    )
    _atomic_json(manifest_path, root_manifest)
    return out_dir
