from __future__ import annotations

import json
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
        numeric = [c for c in df.columns if c not in excluded and pd.to_numeric(df[c], errors="coerce").notna().any()]
        if not numeric:
            raise ValueError("could not infer a numeric signal column")
        value_col = numeric[0]
    values = pd.to_numeric(df[value_col], errors="coerce").interpolate().bfill().ffill().to_numpy(dtype=float)
    timestamps = np.arange(len(values), dtype=np.int64) if time_col is None else df[time_col].to_numpy()
    return values, timestamps


def write_json(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
