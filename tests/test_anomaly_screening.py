"""Offline regression cases for the public Anomaly Screening and cache contracts."""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import numpy as np
import pytest

from anomaly_screening.cache import CACHE_SCHEMA, load_signal_cache, replay_intervals
from anomaly_screening.frozen import BaselineSimilarityPolicy, FrozenScreening
from anomaly_screening.retrieval import filter_reference_candidates
from models.decision import compute_detection_intervals
from preprocessing.preprocess import compress_temporal_window


def test_complementary_references_remain_an_unaggregated_bank():
    references = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[-1.0, 0.0], [0.0, -1.0]],
            [[0.0, -1.0], [-1.0, 0.0]],
        ]
    )
    result = BaselineSimilarityPolicy(retained_count=4).score(references[0], references)
    torch.testing.assert_close(result.patch_scores, torch.zeros(2), rtol=0, atol=0)
    assert result.diagnostics["aggregation"] == "none"


@pytest.mark.parametrize("reference_count,expected", [(1, 1), (2, 2), (7, 4)])
def test_retain_four_never_pads_a_short_reference_bank(reference_count, expected):
    generator = torch.Generator().manual_seed(27)
    references = torch.randn(reference_count, 4, 3, generator=generator)
    query = torch.randn(4, 3, generator=generator)
    result = BaselineSimilarityPolicy().score(query, references)
    assert result.diagnostics["retained_reference_indices"].numel() == expected
    assert result.patch_scores.shape == (4,)


def test_tied_retention_is_deterministic():
    references = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]], [[-1.0, 0.0]]])
    result = BaselineSimilarityPolicy(retained_count=2).score(references[0], references)
    assert result.diagnostics["retained_reference_indices"].tolist() == [0, 1]


@pytest.mark.parametrize(
    "policy,expected",
    [
        ("self_only", [0, 1, 3, 4]),
        ("non_overlap", [0, 4]),
        ("past_only_non_overlap", [0]),
    ],
)
def test_reference_overlap_uses_half_open_windows(policy, expected):
    indices, audit = filter_reference_candidates(
        [0, 56, 112, 168, 224], 112, 112, policy
    )
    assert indices.tolist() == expected
    assert audit["valid_candidate_count"] == len(expected)


def test_fused_pipeline_preserves_thresholds_and_original_interval_contract():
    scores = {
        224: np.array([0.0, 1.0, 2.0, 0.0]),
        448: np.array([0.0, 2.0, 1.0, 0.0]),
        672: np.array([0.0, 0.5, 1.5, 0.0]),
    }
    result = FrozenScreening().run_from_scale_scores(scores)
    assert set(result.fusion.normalized_scale_scores) == {224, 448, 672}
    assert result.state.fused_score is result.fusion.fused_score
    intervals, threshold, processed = compute_detection_intervals(
        result.fusion.fused_score, alpha=0.01
    )
    assert result.decision.intervals == intervals
    assert result.decision.threshold == threshold
    np.testing.assert_array_equal(result.decision.processed_scores, processed)


def test_area_compression_preserves_constants_and_mean():
    np.testing.assert_array_equal(
        compress_temporal_window(np.full(448, 7.0)), np.full(224, 7.0)
    )
    compressed = compress_temporal_window(np.arange(448, dtype=float))
    np.testing.assert_array_equal(compressed, np.arange(0.5, 448, 2))
    original = np.arange(448, dtype=np.float32)
    unchanged = compress_temporal_window(original, mode="implicit_legacy")
    np.testing.assert_array_equal(unchanged, original)
    assert unchanged is not original


def test_cache_replay_does_not_load_the_encoder_runtime(tmp_path):
    signal = tmp_path / "MSL" / "C-1"
    signal.mkdir(parents=True)
    np.savez_compressed(
        signal / "signal_artifacts.npz",
        values=np.zeros(4),
        fused_score=np.arange(4, dtype=float),
        processed_scores=np.array([[0.0, 1.0, 2.0, 3.0]]),
        alphas=np.array([0.1]),
    )
    manifest = {
        "cache_schema": CACHE_SCHEMA,
        "cache_complete": True,
        "dataset": "MSL",
        "signal_id": "C-1",
        "scales": {},
        "decision": {
            "bundle": "signal_artifacts.npz",
            "selected_alpha": 0.1,
            "selected_intervals": [[3, 3]],
            "alphas": [{"alpha": 0.1, "threshold": 2.0, "intervals": [[3, 3]]}],
        },
    }
    (signal / "signal_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cache = load_signal_cache(tmp_path, "MSL", "C-1")
    assert replay_intervals(cache, 0.1)["intervals"] == [(3, 3)]
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; import anomaly_screening.cache; assert 'torch' not in sys.modules",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
