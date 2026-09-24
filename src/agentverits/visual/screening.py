from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import norm

from ..config import ScreeningConfig
from ..types import Interval, ScaleReferenceTrace, ScreeningResult, VisualCandidate
from .encoder import VisualEncoder
from .render import preprocess_series, render_window_tensor


def robust_positive_z(scores: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-12:
        std = np.std(x)
        if std < 1e-12:
            return np.zeros_like(x)
        z = (x - med) / (std + 1e-12)
    else:
        z = (x - med) / (1.4826 * mad + 1e-12)
    return np.maximum(z, 0.0)


def _stable_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    if scores.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    k = min(int(k), int(scores.numel()))
    order = torch.argsort(scores, descending=True, stable=True)
    return order[:k]


def _non_overlap(starts: np.ndarray, query_index: int, window_size: int) -> np.ndarray:
    q0 = int(starts[query_index])
    q1 = q0 + int(window_size) - 1
    ends = starts + int(window_size) - 1
    return np.flatnonzero((ends < q0) | (starts > q1)).astype(np.int64)


def _retained_indices(refs: torch.Tensor, retained_count: int) -> torch.Tensor:
    if refs.ndim != 3 or refs.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=refs.device)
    flat = refs.reshape(refs.shape[0], -1)
    center = torch.median(flat, dim=0).values.reshape(1, -1)
    dist = 1.0 - torch.matmul(F.normalize(flat, dim=-1), F.normalize(center, dim=-1).T).squeeze(-1)
    return torch.argsort(dist, stable=True)[: min(retained_count, refs.shape[0])]


def _patch_bank_distance(query: torch.Tensor, refs: torch.Tensor, retained_count: int) -> torch.Tensor:
    if refs.shape[0] == 0:
        return torch.zeros(query.shape[0], device=query.device, dtype=query.dtype)
    keep = _retained_indices(refs, retained_count)
    bank = F.normalize(refs.index_select(0, keep).reshape(-1, refs.shape[-1]), dim=-1)
    q = F.normalize(query, dim=-1)
    sim = torch.matmul(q, bank.T)
    return 0.5 * (1.0 - torch.max(sim, dim=1).values)


def _harmonic_aggregation(score_size: tuple[int, int, int], similarity: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    b, h, w = score_size
    similarity = similarity.double().clamp_min(1e-12)
    # Encoder masks are zero based. Build patch membership once, without per-patch GPU syncs.
    ids = torch.arange(h * w, device=similarity.device)
    membership = (mask.to(similarity.device).T[:, :, None] == ids[None, None, :]).any(dim=1).double()
    counts = membership.sum(dim=0)
    denominator = (1.0 / similarity) @ membership
    score = torch.where(counts > 0, counts / denominator.clamp_min(1e-12), 0.0)
    return score.reshape(b, h, w)


def _aggregate_map(anomaly_map: np.ndarray, top_fraction: float) -> np.ndarray:
    h, w = anomaly_map.shape
    k = max(1, int(np.ceil(h * float(top_fraction))))
    return np.sort(anomaly_map, axis=0)[-k:, :].mean(axis=0)


def _aligned_scores(maps: torch.Tensor, starts: np.ndarray, full_length: int, window_size: int,
                    image_size: int, top_fraction: float, batch_size: int) -> np.ndarray:
    """Paper Eq. (2): align/average overlapping maps before top-rho reduction."""
    total = np.zeros((image_size, full_length), dtype=np.float32)
    count = np.zeros(full_length, dtype=np.int32)
    for lo in range(0, len(starts), batch_size):
        dense = F.interpolate(maps[lo:lo + batch_size].unsqueeze(1).float(),
                              size=(image_size, window_size), mode="bilinear", align_corners=False).squeeze(1)
        for start, window in zip(starts[lo:lo + batch_size], dense.detach().cpu().numpy()):
            total[:, start:start + window_size] += window
            count[start:start + window_size] += 1
    if np.any(count == 0):
        raise ValueError("window grid leaves uncovered time points")
    total /= count[None, :]
    return _aggregate_map(total, top_fraction)


def detection_intervals(scores: np.ndarray, alpha: float, smoothing: bool = True) -> tuple[list[Interval], float, np.ndarray]:
    x = np.asarray(scores, dtype=float)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("scores must be a nonempty finite one-dimensional array")
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must be in (0,1)")
    if smoothing:
        span = max(1, int(len(x) * 0.01))
        x = pd.Series(x).ewm(span=span).mean().values
    threshold = float(x.mean() + norm.ppf(1.0 - float(alpha)) * x.std())
    mask = x > threshold
    intervals: list[Interval] = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            intervals.append(Interval(start, i - 1))
            start = None
    if start is not None:
        intervals.append(Interval(start, len(mask) - 1))
    return intervals, threshold, x


@dataclass
class ScaleResult:
    scale: int
    scores: np.ndarray
    window_starts: np.ndarray
    retained_reference_starts: list[list[int]]


class VisualScreening:
    def __init__(self, config: ScreeningConfig | None = None, encoder: VisualEncoder | None = None) -> None:
        self.config = config or ScreeningConfig()
        self.config.validate()
        self.encoder = encoder or VisualEncoder(
            model_name=self.config.encoder_name,
            pretrained=self.config.encoder_weights,
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            device=self.config.device,
        )

    def _encode_windows(self, windows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chunks = {"large": [], "mid": [], "patch": [], "class": []}
        large_mask = mid_mask = None
        for start in range(0, len(windows), self.config.batch_size):
            tensors = np.stack([render_window_tensor(w, self.config.image_size) for w in windows[start:start + self.config.batch_size]])
            batch = torch.from_numpy(tensors)
            out = self.encoder.encode(batch)
            chunks["large"].append(out.large_tokens.detach().cpu())
            chunks["mid"].append(out.mid_tokens.detach().cpu())
            chunks["patch"].append(out.patch_tokens.detach().cpu())
            chunks["class"].append(out.class_tokens.detach().cpu())
            large_mask = out.large_mask.detach().cpu()
            mid_mask = out.mid_mask.detach().cpu()
        assert large_mask is not None and mid_mask is not None
        return (
            torch.cat(chunks["large"]), torch.cat(chunks["mid"]),
            torch.cat(chunks["patch"]), torch.cat(chunks["class"]),
            large_mask, mid_mask,
        )

    def _score_scale(self, values: np.ndarray, scale: int) -> ScaleResult:
        step = max(1, int(scale / self.config.step_ratio))
        starts = np.arange(0, len(values) - scale + 1, step, dtype=np.int64)
        if not len(starts):
            raise ValueError(f"signal is too short for scale {scale}")
        if starts[-1] != len(values) - scale:
            starts = np.append(starts, len(values) - scale)
        windows = np.stack([values[s:s + scale] for s in starts]).astype(np.float32)
        large, mid, patch, class_tokens, large_mask, mid_mask = self._encode_windows(windows)
        device = self.encoder.device
        large, mid, patch, class_tokens = [x.to(device) for x in (large, mid, patch, class_tokens)]
        patch_scores: list[np.ndarray] = []
        mid_scores: list[np.ndarray] = []
        large_scores: list[np.ndarray] = []
        retained_reference_starts: list[list[int]] = []
        norm_classes = F.normalize(class_tokens, dim=-1)
        for qi in range(len(starts)):
            valid_np = _non_overlap(starts, qi, scale)
            valid = torch.as_tensor(valid_np, dtype=torch.long, device=device)
            if valid.numel() == 0:
                patch_scores.append(np.zeros(patch.shape[1], dtype=np.float32))
                mid_scores.append(np.zeros(mid.shape[1], dtype=np.float32))
                large_scores.append(np.zeros(large.shape[1], dtype=np.float32))
                retained_reference_starts.append([])
                continue
            sims = torch.matmul(norm_classes.index_select(0, valid), norm_classes[qi])
            refs = valid.index_select(0, _stable_topk(sims, self.config.top_k))
            patch_ref_bank = patch.index_select(0, refs)
            patch_keep = _retained_indices(patch_ref_bank, self.config.retained_references)
            retained_ids = refs.index_select(0, patch_keep)
            retained_reference_starts.append([int(starts[int(idx)]) for idx in retained_ids.detach().cpu().tolist()])
            patch_scores.append(_patch_bank_distance(patch[qi], patch_ref_bank, self.config.retained_references).detach().cpu().numpy())
            mid_scores.append(_patch_bank_distance(mid[qi], mid.index_select(0, refs), self.config.retained_references).detach().cpu().numpy())
            large_scores.append(_patch_bank_distance(large[qi], large.index_select(0, refs), self.config.retained_references).detach().cpu().numpy())
        p = torch.from_numpy(np.stack(patch_scores)).to(device)
        m = torch.from_numpy(np.stack(mid_scores)).to(device)
        l = torch.from_numpy(np.stack(large_scores)).to(device)
        side = int(round(p.shape[1] ** 0.5))
        patch_map = p.reshape(len(starts), side, side)
        mid_map = _harmonic_aggregation((len(starts), side, side), m, mid_mask.to(device))
        large_map = _harmonic_aggregation((len(starts), side, side), l, large_mask.to(device))
        fused_map = torch.nan_to_num((patch_map.double() + mid_map + large_map) / 3.0)
        aligned = _aligned_scores(fused_map, starts, len(values), scale, self.config.image_size,
                                  self.config.aggregate_top_fraction, self.config.batch_size)
        return ScaleResult(
            scale=scale,
            scores=aligned,
            window_starts=starts.copy(),
            retained_reference_starts=retained_reference_starts,
        )

    def run(self, raw_values: Sequence[float]) -> ScreeningResult:
        self.config.validate()
        values = preprocess_series(raw_values)
        per_scale = [self._score_scale(values, int(scale)) for scale in self.config.scales]
        normalized = [robust_positive_z(x.scores) for x in per_scale]
        fused = np.mean(np.stack(normalized), axis=0)
        reference_traces = {
            int(item.scale): ScaleReferenceTrace(
                scale=int(item.scale),
                window_starts=[int(x) for x in item.window_starts.tolist()],
                retained_reference_starts=[
                    [int(x) for x in row] for row in item.retained_reference_starts
                ],
            )
            for item in per_scale
        }
        candidate_sets: dict[str, list[VisualCandidate]] = {}
        selected: list[VisualCandidate] | None = None
        for alpha in self.config.alpha_candidates:
            intervals, _, processed = detection_intervals(fused, alpha, self.config.smoothing)
            rows = [
                VisualCandidate(
                    candidate_id=f"V{i + 1:04d}",
                    interval=interval,
                    alpha=float(alpha),
                    score_peak=float(np.max(processed[interval.start:interval.end + 1])) if interval.end >= interval.start else None,
                )
                for i, interval in enumerate(intervals)
            ]
            candidate_sets[f"{alpha:g}"] = rows
            if abs(float(alpha) - float(self.config.alpha)) < 1e-12:
                selected = rows
        if selected is None:
            intervals, _, processed = detection_intervals(fused, self.config.alpha, self.config.smoothing)
            selected = [
                VisualCandidate(f"V{i + 1:04d}", interval, float(self.config.alpha), float(np.max(processed[interval.start:interval.end + 1])))
                for i, interval in enumerate(intervals)
            ]
            candidate_sets[f"{self.config.alpha:g}"] = selected
        return ScreeningResult(
            scores=fused.astype(float).tolist(),
            candidate_sets=candidate_sets,
            selected_alpha=float(self.config.alpha),
            selected_candidates=selected,
            reference_traces=reference_traces,
        )
