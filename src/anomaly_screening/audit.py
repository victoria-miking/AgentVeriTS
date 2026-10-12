"""Frozen configuration and reproducibility fingerprints."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

FROZEN_CONFIG_NAME = "ANOMALY_SCREENING_FROZEN"


def _torch_module() -> Any:
    try:
        import torch
    except Exception:
        return None
    return torch


def load_frozen_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the repository's explicit frozen baseline config."""

    config_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[2] / "configs" / "anomaly_screening.json"
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("name") != FROZEN_CONFIG_NAME:
        raise ValueError(f"Expected {FROZEN_CONFIG_NAME}, got {payload.get('name')!r}")
    if [int(s) for s in payload.get("scales", [])] != [224, 448, 672]:
        raise ValueError("ANOMALY_SCREENING_FROZEN must use scales [224, 448, 672]")
    trim = payload.get("reference_trim", {})
    if int(trim.get("retained_count", -1)) != 4 or trim.get("aggregation") != "none":
        raise ValueError("frozen baseline must retain four windows without aggregation")
    return payload


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Stable SHA-256 fingerprint of a JSON-compatible config mapping."""

    encoded = json.dumps(
        config, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_fingerprint(project_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root or Path(__file__).resolve().parents[2])

    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return ""

    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def environment_fingerprint() -> dict[str, Any]:
    """Best-effort offline runtime fingerprint, including failed torch loads."""

    torch = _torch_module()
    result: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": None,
        "cuda_version": None,
        "gpu": None,
    }
    try:
        result["torch_version"] = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        pass
    if torch is not None:
        result["torch_version"] = getattr(torch, "__version__", result["torch_version"])
        result["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        try:
            if torch.cuda.is_available():
                result["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            result["gpu"] = None
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "open_clip_torch"):
        try:
            result[f"{package}_version"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[f"{package}_version"] = None
    return result
