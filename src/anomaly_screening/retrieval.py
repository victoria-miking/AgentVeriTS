"""Frozen Raw4 reference filtering, Top-K, retention and patch-bank scoring."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch
import torch.nn.functional as F
import numpy as np

POLICIES = {"self_only", "non_overlap", "past_only_non_overlap"}


def filter_reference_candidates(
    candidate_starts: Iterable[int],
    query_start: int,
    window_length: int,
    policy: str = "non_overlap",
) -> tuple[np.ndarray, dict[str, Any]]:
    """按左闭右开区间过滤 reference，并返回候选统计。"""
    policy = str(policy).lower()
    if policy not in POLICIES:
        raise ValueError(f"未知 overlap policy: {policy}")
    starts = np.asarray(list(candidate_starts), dtype=np.int64)
    query_start = int(query_start)
    window_length = int(window_length)
    if window_length <= 0:
        raise ValueError("window_length 必须为正整数")
    query_end = query_start + window_length
    ends = starts + window_length
    self_mask = starts == query_start
    overlap_mask = (starts < query_end) & (ends > query_start)
    future_mask = ends > query_start
    if policy == "self_only":
        valid_mask = ~self_mask
    elif policy == "non_overlap":
        valid_mask = ~overlap_mask
    else:
        valid_mask = ends <= query_start
    audit = {
        "raw_candidate_count": int(starts.size),
        "self_excluded_count": int(self_mask.sum()),
        "overlap_excluded_count": (
            int((overlap_mask & ~self_mask).sum()) if policy != "self_only" else 0
        ),
        "past_only_excluded_count": (
            int((~overlap_mask & future_mask).sum())
            if policy == "past_only_non_overlap"
            else 0
        ),
        "valid_candidate_count": int(valid_mask.sum()),
    }
    return np.flatnonzero(valid_mask), audit


def audit_selected_references(
    selected_starts: Iterable[int],
    query_start: int,
    window_length: int,
    topk_requested: int,
    step_size: int = 1,
) -> dict[str, Any]:
    """统计最终 Top-k 的重叠率、共享比例和相邻窗口数。"""
    starts = np.asarray(list(selected_starts), dtype=np.int64)
    query_start = int(query_start)
    query_end = query_start + int(window_length)
    ends = starts + int(window_length)
    shared = np.maximum(
        0, np.minimum(ends, query_end) - np.maximum(starts, query_start)
    )
    offsets = (starts - query_start) / max(1, int(step_size))
    return {
        "topk_size": int(starts.size),
        "overlap_ref_count": int((shared > 0).sum()),
        "overlap_ref_rate": float((shared > 0).mean()) if starts.size else 0.0,
        "mean_shared_ratio": (
            float((shared / float(window_length)).mean()) if starts.size else 0.0
        ),
        "adjacent_q1_count": int((np.abs(offsets) == 1).sum()),
        "adjacent_q2_count": int((np.abs(offsets) == 2).sum()),
        "adjacent_q3_count": int((np.abs(offsets) == 3).sum()),
        "reference_shortage": bool(starts.size < int(topk_requested)),
    }


def select_retained_reference_indices(
    retained_ref_tokens: torch.Tensor,
    trim_ratio: float = 0.0,
    retained_count: Optional[int] = None,
) -> torch.Tensor:
    """Select references using the frozen median-center distance and input order."""
    refs = retained_ref_tokens
    if refs.ndim == 4 and refs.shape[2] == 1:
        refs = refs.squeeze(2)
    if refs.ndim != 3:
        raise ValueError(
            f"retained_ref_tokens should be [K,P,D], got {tuple(refs.shape)}"
        )
    k = int(refs.shape[0])
    if k == 0:
        return torch.empty(0, dtype=torch.long, device=refs.device)
    if k == 1:
        return torch.zeros(1, dtype=torch.long, device=refs.device)
    flat = refs.reshape(k, -1)
    # Preserve the flattened torch.median center and reference selection order.
    center = torch.median(flat, dim=0).values.reshape(1, -1)
    distances = 1.0 - torch.matmul(
        F.normalize(flat, dim=-1), F.normalize(center, dim=-1).T
    ).squeeze(-1)
    if retained_count is None:
        keep = max(1, int(round(k * (1.0 - float(trim_ratio)))))
    else:
        keep = max(1, int(retained_count))
    keep = min(keep, k)
    return torch.argsort(distances)[:keep]


def _topk_indices(sim: torch.Tensor, k: int) -> torch.Tensor:
    if sim.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=sim.device)
    k = min(max(1, int(k)), sim.numel())
    return torch.topk(sim, k=k, largest=True).indices


def _cosine_similarity_flat(bank: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    bank = bank.reshape(bank.shape[0], -1)
    query = query.reshape(-1)
    bank_n = F.normalize(bank, dim=-1)
    query_n = F.normalize(query.unsqueeze(0), dim=-1)
    return torch.matmul(query_n, bank_n.T).squeeze(0)


def select_reference_indices_screening(
    coarse_bank: torch.Tensor,
    query_coarse: torch.Tensor,
    raw_class_bank: torch.Tensor,
    query_raw_class: torch.Tensor,
    bank_window_ids: torch.Tensor,
    query_window_id: int,
    top_m: int,
    top_k: int,
    retrieval_mode: str,
    temporal_gap: int = 0,
    overlap_policy: str = "self_only",
    window_size: int = 224,
    step_size: int = 56,
    return_audit: bool = False,
    geometry_mode: str = "none",
    geometry_candidate_r: int = 64,
    geometry_bank: Optional[Any] = None,
):
    candidate_starts = bank_window_ids.detach().cpu().numpy() * int(step_size)
    valid_local, audit = filter_reference_candidates(
        candidate_starts=candidate_starts,
        query_start=int(query_window_id) * int(step_size),
        window_length=int(window_size),
        policy=overlap_policy,
    )
    valid_idx = torch.as_tensor(
        valid_local, dtype=torch.long, device=bank_window_ids.device
    )
    if temporal_gap > 0:
        gap_mask = torch.abs(bank_window_ids[valid_idx] - int(query_window_id)) > int(
            temporal_gap
        )
        valid_idx = valid_idx[gap_mask]
        audit["valid_candidate_count"] = int(valid_idx.numel())
    if valid_idx.numel() == 0:
        selected_audit = audit_selected_references(
            [],
            int(query_window_id) * int(step_size),
            int(window_size),
            top_k,
            step_size=int(step_size),
        )
        return (valid_idx, {**audit, **selected_audit}) if return_audit else valid_idx

    coarse_sim = _cosine_similarity_flat(coarse_bank, query_coarse)
    if str(retrieval_mode).lower() != "coarse" or str(geometry_mode).lower() != "none":
        raise ValueError(
            "The frozen public pipeline supports raw class-token retrieval only"
        )
    top_local = _topk_indices(coarse_sim[valid_idx], top_k)
    selected = valid_idx[top_local]
    if not return_audit:
        return selected
    selected_starts = bank_window_ids[selected].detach().cpu().numpy() * int(step_size)
    selected_audit = audit_selected_references(
        selected_starts,
        int(query_window_id) * int(step_size),
        int(window_size),
        top_k,
        step_size=int(step_size),
    )
    return selected, {**audit, **selected_audit}


def compute_trimmed_patch_bank_distance(
    token_norm: torch.Tensor,
    ref_mem: torch.Tensor,
    trim_ratio: float,
    retained_count: Optional[int] = None,
) -> torch.Tensor:
    """Compare query patches against all patches in trimmed reference windows.

    Trimming is performed at the reference-window level using the same
    flattened median-center cosine distance used by the frozen implementation.
    Unlike prototype aggregation, retained windows are not averaged: their
    ``K_retained * N_patch`` tokens form one reference bank and each query
    patch uses its nearest bank token.
    """
    if ref_mem.ndim == 4 and ref_mem.shape[2] == 1:
        ref_mem = ref_mem.squeeze(2)
    if ref_mem.ndim != 3:
        raise ValueError(f"ref_mem should be [K,N,D], got {tuple(ref_mem.shape)}")
    k = int(ref_mem.shape[0])
    if k == 0:
        return torch.zeros(
            token_norm.shape[0], device=token_norm.device, dtype=token_norm.dtype
        )
    if k > 1:
        keep_idx = select_retained_reference_indices(
            ref_mem, trim_ratio, retained_count
        )
        ref_mem = ref_mem[keep_idx]
    ref_bank = F.normalize(ref_mem.reshape(-1, ref_mem.shape[-1]), dim=-1)
    query = F.normalize(token_norm, dim=-1)
    sim = torch.matmul(query, ref_bank.T)
    return 0.5 * (1.0 - torch.max(sim, dim=1).values)


def score_retained_patch_bank(
    query_tokens, retained_reference_tokens, *, retained_count=4, trim_ratio=0.75
):
    refs = retained_reference_tokens
    if refs.ndim == 4 and refs.shape[2] == 1:
        refs = refs.squeeze(2)
    indices = select_retained_reference_indices(refs, trim_ratio, retained_count)
    scores = compute_trimmed_patch_bank_distance(
        query_tokens, refs, trim_ratio, retained_count
    )
    return scores, indices
