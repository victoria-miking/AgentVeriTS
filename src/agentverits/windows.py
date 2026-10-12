"""Length-aware global rendering and ownership-safe anchor window planning."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence


@dataclass(frozen=True)
class AdaptiveRenderProfile:
    signal_length: int
    kind: str
    width_px: int
    height_px: int = 512
    navigation_width_px: int | None = None
    navigation_height_px: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AssessmentWindow:
    window_id: str
    start: int
    end: int
    core_start: int
    core_end: int
    candidate_ids: tuple[str, ...] = ()

    @property
    def span(self) -> int:
        return self.end - self.start + 1

    @property
    def core_span(self) -> int:
        return self.core_end - self.core_start + 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["candidate_ids"] = list(self.candidate_ids)
        d["span"] = self.span
        d["core_span"] = self.core_span
        return d


def render_profile(length: int) -> AdaptiveRenderProfile:
    n = int(length)
    if n < 4_000:
        return AdaptiveRenderProfile(n, "short", 3072)
    if n < 10_000:
        return AdaptiveRenderProfile(n, "medium", 4096)
    if n < 15_000:
        return AdaptiveRenderProfile(n, "long_medium", 5120)
    if n <= 20_000:
        return AdaptiveRenderProfile(n, "long", 6000)
    return AdaptiveRenderProfile(n, "ultra_long", 4096, 512, 4096, 512)


def _safe_cut(
    boundary: int, candidates: Sequence[tuple[int, int]], lo: int, hi: int
) -> int:
    """Move a nominal cut to the nearest candidate-free gap."""
    b = max(lo, min(int(boundary), hi))
    if not any(a <= b <= z for a, z in candidates):
        return b
    gaps: list[tuple[int, int]] = []
    cursor = lo
    for a, z in sorted(candidates):
        if a > cursor:
            gaps.append((cursor, a - 1))
        cursor = max(cursor, z + 1)
    if cursor <= hi:
        gaps.append((cursor, hi))
    if not gaps:
        return b
    # Prefer a gap at or before the nominal boundary so the normal case stays
    # within the 20k budget.  Move forward only when no preceding safe gap is
    # available within this window (for example a candidate starts at index 0).
    before = [z for a, z in gaps if z <= b]
    if before:
        return max(before)
    return min((max(a, min(z, b)) for a, z in gaps), key=lambda x: abs(x - b))


def plan_assessment_windows(
    length: int,
    candidates: Iterable[Sequence[int]] = (),
    *,
    window_size: int = 20_000,
    overlap: int = 2_000,
) -> list[AssessmentWindow]:
    """Plan overlapping windows with authoritative non-overlap cores.

    Candidate-containing nominal cuts are shifted to the nearest safe gap. A
    candidate belongs to exactly one home window based on its center in the
    authoritative core; overlap remains context only.
    """
    n = int(length)
    if n <= window_size:
        rows = [tuple(int(v) for v in c[:2]) for c in candidates if len(c) >= 2]
        return [AssessmentWindow("W1", 0, max(0, n - 1), 0, max(0, n - 1), tuple())]
    if overlap <= 0 or overlap >= window_size:
        raise ValueError("overlap must be positive and smaller than window_size")
    cand = sorted(
        (max(0, int(c[0])), min(n - 1, int(c[1]))) for c in candidates if len(c) >= 2
    )
    step = window_size - overlap
    windows: list[tuple[int, int]] = []
    start = 0
    while start < n:
        nominal_end = min(n - 1, start + window_size - 1)
        if nominal_end < n - 1:
            cut = _safe_cut(nominal_end, cand, start, min(n - 1, nominal_end + overlap))
            # Do not let a safe cut create an over-budget window unless no
            # candidate-free gap exists inside the overlap allowance.
            end = max(nominal_end, cut)
        else:
            end = nominal_end
        windows.append((start, end))
        if end >= n - 1:
            break
        start = end - overlap + 1
    # Authoritative cores are midpoint partitions between adjacent windows.
    core_bounds = [0]
    for i in range(len(windows) - 1):
        nominal = (windows[i][1] + windows[i + 1][0]) // 2
        core_bounds.append(_safe_cut(nominal, cand, windows[i + 1][0], windows[i][1]))
    core_bounds.append(n - 1)
    result: list[AssessmentWindow] = []
    for i, (start, end) in enumerate(windows):
        core_start = core_bounds[i]
        core_end = core_bounds[i + 1]
        ids = tuple(
            f"C{j + 1:04d}"
            for j, (a, z) in enumerate(cand)
            if core_start <= (a + z) // 2 <= core_end
        )
        result.append(
            AssessmentWindow(f"W{i + 1}", start, end, core_start, core_end, ids)
        )
    return result


def candidate_home_windows(
    candidates: Iterable[Sequence[int]], windows: Sequence[AssessmentWindow]
) -> dict[str, str]:
    """Assign each candidate center to one authoritative home window."""
    result = {}
    for i, c in enumerate(candidates, 1):
        a, z = int(c[0]), int(c[1])
        center = (a + z) // 2
        owner = next(
            (w for w in windows if w.core_start <= center <= w.core_end), windows[-1]
        )
        result[f"C{i:04d}"] = owner.window_id
    return result


__all__ = [
    "AdaptiveRenderProfile",
    "AssessmentWindow",
    "render_profile",
    "plan_assessment_windows",
    "candidate_home_windows",
]
