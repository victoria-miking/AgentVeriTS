"""Dataset loading and annotation conversion, separate from inference."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with timestamp, value, and optional label columns.

    Supports common dataset variants:
      value / Value / data / Data
      timestamp / Timestamp / time / Time
      label / Label / anomaly / is_anomaly
    """
    if df.empty:
        raise ValueError("empty csv")

    cols = list(df.columns)
    lower = {str(c).strip().lower(): c for c in cols}

    ts_col = None
    for name in ["timestamp", "time", "date", "datetime"]:
        if name in lower:
            ts_col = lower[name]
            break

    value_col = None
    for name in ["value", "data", "kpi", "metric", "y"]:
        if name in lower:
            value_col = lower[name]
            break

    label_col = None
    for name in ["label", "labels", "anomaly", "is_anomaly", "is_anomalous"]:
        if name in lower:
            label_col = lower[name]
            break

    if value_col is None:
        numeric_cols = []
        for c in cols:
            try:
                pd.to_numeric(df[c], errors="raise")
                numeric_cols.append(c)
            except Exception:
                continue
        label_like = set()
        if label_col is not None:
            label_like.add(label_col)
        numeric_non_label = [c for c in numeric_cols if c not in label_like]
        if ts_col is not None:
            numeric_non_label = [c for c in numeric_non_label if c != ts_col]
        if not numeric_non_label:
            raise ValueError(f"Could not infer value column. columns={cols}")
        value_col = numeric_non_label[0]

    out = pd.DataFrame()
    if ts_col is None:
        out["timestamp"] = np.arange(len(df), dtype=np.int64)
    else:
        out["timestamp"] = df[ts_col].values

    out["value"] = (
        pd.to_numeric(df[value_col], errors="coerce")
        .interpolate()
        .bfill()
        .ffill()
        .astype(float)
        .values
    )

    if label_col is not None:
        out["label"] = (
            pd.to_numeric(df[label_col], errors="coerce")
            .fillna(0)
            .astype(int)
            .clip(0, 1)
            .values
        )

    return out


def load_anomalies(anomalies_csv: Path) -> Dict[str, List[List[Any]]]:
    if not anomalies_csv.exists():
        return {}
    try:
        df = pd.read_csv(anomalies_csv)
        if "signal" not in df.columns:
            df = pd.read_csv(anomalies_csv, header=None, names=["signal", "anomalies"])
    except Exception:
        df = pd.read_csv(anomalies_csv, header=None, names=["signal", "anomalies"])

    out: Dict[str, List[List[Any]]] = {}
    for _, row in df.iterrows():
        signal = str(row.get("signal", "")).strip()
        raw = row.get("anomalies", "[]")
        try:
            intervals = ast.literal_eval(raw) if isinstance(raw, str) else []
        except Exception:
            intervals = []
        out[signal] = intervals
    return out


def intervals_to_index_space(
    intervals: Optional[List[List[Any]]], timestamps: np.ndarray
) -> List[List[int]]:
    if intervals is None:
        return []
    ts = np.asarray(timestamps)
    out: List[List[int]] = []
    for pair in intervals:
        if pair is None or len(pair) != 2:
            continue
        s_raw, e_raw = pair
        try:
            s_idx = int(np.searchsorted(ts, s_raw, side="left"))
            e_idx = int(np.searchsorted(ts, e_raw, side="right") - 1)
        except Exception:
            try:
                s_idx = int(s_raw)
                e_idx = int(e_raw)
            except Exception:
                continue
        if len(ts) == 0:
            continue
        s_idx = max(0, min(len(ts) - 1, s_idx))
        e_idx = max(0, min(len(ts) - 1, e_idx))
        if e_idx >= s_idx:
            out.append([s_idx, e_idx])
    return out


def binary_to_intervals(labels: np.ndarray) -> List[List[int]]:
    labels = np.asarray(labels).astype(bool)
    intervals: List[List[int]] = []
    in_seg = False
    s = 0
    for i, flag in enumerate(labels):
        if flag and not in_seg:
            s = i
            in_seg = True
        elif (not flag) and in_seg:
            intervals.append([s, i - 1])
            in_seg = False
    if in_seg:
        intervals.append([s, len(labels) - 1])
    return intervals


def intervals_to_binary(intervals: Sequence[Sequence[int]], length: int) -> np.ndarray:
    y = np.zeros(length, dtype=int)
    for s, e in intervals:
        s_i = max(0, min(length - 1, int(s)))
        e_i = max(0, min(length - 1, int(e)))
        if e_i >= s_i:
            y[s_i : e_i + 1] = 1
    return y


def get_gt_intervals_and_labels(
    df_norm: pd.DataFrame,
    signal: str,
    anomalies: Dict[str, List[List[Any]]],
) -> Tuple[List[List[int]], np.ndarray, str]:
    timestamps = df_norm["timestamp"].values
    n = len(df_norm)

    if "label" in df_norm.columns:
        labels = df_norm["label"].values.astype(int).clip(0, 1)
        return binary_to_intervals(labels), labels, "label_column"

    intervals_raw = anomalies.get(signal, None)
    if intervals_raw is None:
        labels = np.zeros(n, dtype=int)
        return [], labels, "none"

    intervals_idx = intervals_to_index_space(intervals_raw, timestamps)
    labels = intervals_to_binary(intervals_idx, n)
    return intervals_idx, labels, "anomalies_csv"


def select_files(
    dataset_dir: Path,
    cases: Optional[List[str]] = None,
    max_files: Optional[int] = None,
) -> List[Path]:
    files = sorted(dataset_dir.glob("*.csv"))
    if cases:
        selected: List[Path] = []
        for case in cases:
            case_s = str(case)
            matches = [
                p
                for p in files
                if p.stem == case_s
                or p.name.startswith(case_s + "_")
                or case_s in p.stem
            ]
            selected.extend(matches)
        seen = set()
        files = []
        for p in selected:
            if p not in seen:
                files.append(p)
                seen.add(p)
    if max_files is not None and max_files > 0:
        files = files[:max_files]
    return files
