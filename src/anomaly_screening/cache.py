"""Readers and lightweight replay helpers for the Screening Raw4 full cache.

The cache is deliberately file based and dependency light.  Building the cache
requires PyTorch/CLIP, but reading a completed cache only requires NumPy and
the Python standard library.  The builder writes one immutable signal
manifest and one compressed bundle per scale; a later stage can therefore
consume exactly the artifacts it needs without re-rendering or re-encoding a
signal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


CACHE_SCHEMA = "ANOMALY_SCREENING_CACHE_V1"
SUPPORTED_CACHE_SCHEMAS = {CACHE_SCHEMA, "CGRF_RAW4_FULL_CACHE_V1"}
SCALE_KEYS = ("large", "mid", "patch")


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def list_cached_signals(root: str | Path) -> list[tuple[str, str]]:
    """Return complete ``(dataset, signal_id)`` entries in deterministic order."""

    root_path = Path(root)
    entries: list[tuple[str, str]] = []
    if not root_path.is_dir():
        return entries
    for dataset_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        for signal_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            manifest_path = signal_dir / "signal_manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = _json_load(manifest_path)
            except Exception:
                continue
            if manifest.get("cache_schema") in SUPPORTED_CACHE_SCHEMAS and manifest.get(
                "cache_complete"
            ):
                entries.append((dataset_dir.name, signal_dir.name))
    return entries


def load_scale_cache(
    root: str | Path,
    dataset: str,
    signal_id: str,
    scale: int,
    *,
    allow_pickle: bool = True,
) -> dict[str, Any]:
    """Load one scale bundle and its manifest.

    Object arrays are used for ragged candidate/trimmed reference lists.  The
    builder emits only numeric arrays and integer indices into those arrays;
    callers should keep ``allow_pickle=True`` for those ragged lists.
    """

    root_path = Path(root)
    signal_root = root_path / str(dataset) / str(signal_id)
    manifest_path = signal_root / "signal_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"signal cache manifest not found: {manifest_path}")
    signal_manifest = _json_load(manifest_path)
    scales = signal_manifest.get("scales", {})
    scale_key = str(int(scale))
    entry = scales.get(scale_key)
    if not isinstance(entry, Mapping):
        raise KeyError(f"scale {scale_key} is not present in {manifest_path}")
    bundle_path = _resolve(signal_root, str(entry["bundle"]))
    with np.load(bundle_path, allow_pickle=allow_pickle) as bundle:
        payload = {name: bundle[name] for name in bundle.files}
    payload["_manifest"] = entry
    payload["_signal_manifest"] = signal_manifest
    payload["_bundle_path"] = str(bundle_path)
    return payload


def load_signal_cache(
    root: str | Path, dataset: str, signal_id: str, *, include_scales: bool = True
) -> dict[str, Any]:
    """Load all signal-level and scale-level artifacts for one cached signal."""

    root_path = Path(root)
    signal_root = root_path / str(dataset) / str(signal_id)
    signal_manifest_path = signal_root / "signal_manifest.json"
    if not signal_manifest_path.is_file():
        raise FileNotFoundError(
            f"signal cache manifest not found: {signal_manifest_path}"
        )
    manifest = _json_load(signal_manifest_path)
    if manifest.get("cache_schema") not in SUPPORTED_CACHE_SCHEMAS:
        raise ValueError(f"unsupported cache schema: {manifest.get('cache_schema')!r}")
    if not manifest.get("cache_complete"):
        raise ValueError(f"cache is incomplete: {signal_manifest_path}")

    result: dict[str, Any] = {
        "manifest": manifest,
        "root": str(signal_root),
        "scales": {},
    }
    if include_scales:
        for scale in sorted((int(value) for value in manifest.get("scales", {}))):
            result["scales"][scale] = load_scale_cache(
                root_path, dataset, signal_id, scale
            )

    decision_entry = manifest.get("decision", {})
    decision_path_value = (
        decision_entry.get("bundle") if isinstance(decision_entry, Mapping) else None
    )
    if decision_path_value:
        decision_path = _resolve(signal_root, str(decision_path_value))
        if decision_path.is_file():
            with np.load(decision_path, allow_pickle=False) as bundle:
                result["decision"] = {name: bundle[name] for name in bundle.files}
    return result


def replay_intervals(
    cache: Mapping[str, Any],
    alpha: float | None = None,
) -> dict[str, Any]:
    """Return cached decision output for ``alpha`` without model execution.

    The builder stores every configured alpha.  This helper only reads the
    JSON decision table and the already computed processed score vector.
    """

    manifest = cache.get("manifest", {})
    decision = cache.get("decision", {})
    rows = manifest.get("decision", {}).get("alphas", [])
    requested = (
        float(alpha)
        if alpha is not None
        else manifest.get("decision", {}).get("selected_alpha")
    )
    if requested is None:
        raise KeyError("no selected alpha is available in the cache")
    match = None
    for row in rows:
        try:
            if abs(float(row["alpha"]) - float(requested)) < 1e-12:
                match = row
                break
        except (KeyError, TypeError, ValueError):
            continue
    if match is None:
        raise KeyError(f"alpha {requested} is not present in the cache")
    processed = decision.get("processed_scores")
    if processed is not None:
        processed = np.asarray(processed, dtype=float)
        if processed.ndim == 2:
            cached_alphas = np.asarray(decision.get("alphas", []), dtype=float).reshape(
                -1
            )
            matches = np.where(
                np.isclose(cached_alphas, float(match["alpha"]), rtol=0.0, atol=1e-12)
            )[0]
            processed = processed[int(matches[0])] if len(matches) else processed[0]
    return {
        "alpha": float(match["alpha"]),
        "threshold": float(match["threshold"]),
        "intervals": [
            tuple(int(v) for v in interval) for interval in match.get("intervals", [])
        ],
        "processed_scores": (
            None if processed is None else np.asarray(processed, dtype=float)
        ),
        "fused_score": (
            None
            if decision.get("fused_score") is None
            else np.asarray(decision["fused_score"], dtype=float)
        ),
    }


def stage_summary(cache: Mapping[str, Any]) -> dict[str, Any]:
    """Build a compact machine-readable summary for notebooks and scripts."""

    manifest = cache.get("manifest", {})
    summary: dict[str, Any] = {
        "dataset": manifest.get("dataset"),
        "signal_id": manifest.get("signal_id"),
        "cache_schema": manifest.get("cache_schema"),
        "cache_complete": bool(manifest.get("cache_complete")),
        "scales": sorted(int(key) for key in manifest.get("scales", {})),
        "selected_alpha": manifest.get("decision", {}).get("selected_alpha"),
        "selected_intervals": manifest.get("decision", {}).get(
            "selected_intervals", []
        ),
    }
    for scale, payload in sorted(cache.get("scales", {}).items()):
        summary.setdefault("scale_shapes", {})[str(scale)] = {
            key: list(np.asarray(payload[key]).shape)
            for key in (
                "patch_tokens",
                "topk_indices",
                "retained_indices",
                "patch_scores",
                "aligned_score",
            )
            if key in payload
        }
    return summary


__all__ = [
    "CACHE_SCHEMA",
    "SCALE_KEYS",
    "list_cached_signals",
    "load_scale_cache",
    "load_signal_cache",
    "replay_intervals",
    "stage_summary",
]
