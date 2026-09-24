from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_signal_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("input CSV is empty")
    lower = {str(c).strip().lower(): c for c in df.columns}
    value_col = next((lower[k] for k in ("value", "data", "kpi", "metric", "y") if k in lower), None)
    time_col = next((lower[k] for k in ("timestamp", "time", "date", "datetime") if k in lower), None)
    if value_col is None:
        excluded = {time_col} if time_col else set()
        excluded.update(lower[k] for k in ("label", "labels", "anomaly", "is_anomaly", "anomaly_label", "anomaly_labels", "ground_truth") if k in lower)
        numeric = [c for c in df.columns if c not in excluded and pd.to_numeric(df[c], errors="coerce").notna().any()]
        if not numeric:
            raise ValueError("could not infer a numeric signal column")
        if len(numeric) > 1:
            raise ValueError("multiple numeric signal columns; name the univariate signal column 'value'")
        value_col = numeric[0]
    values = pd.to_numeric(df[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan).interpolate().bfill().ffill().to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("signal column contains no finite numeric data")
    timestamps = np.arange(len(values), dtype=np.int64) if time_col is None else df[time_col].to_numpy()
    return values, timestamps


def write_json(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=p.parent, delete=False) as stream:
            temporary = stream.name
            stream.write(content)
        os.replace(temporary, p)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return p
