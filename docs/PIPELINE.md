# REVA pipeline contract

## Global hypothesis

The global module receives one complete signal image with orange visual candidate spans. It must return every input candidate once, preserving its candidate ID and reviewed interval. Original candidates can only receive `keep`, `remove`, or `refine`. A missed region can only enter through a separate `add` record.

The output confidence measures confidence in the selected action and its boundaries. It is not an anomaly score.

## Verification routing

Let `c_i` be the global confidence of decision `i` and `tau` the configured verification threshold. The router is intentionally simple:

```text
verify(i) = c_i < tau
```

No action-specific rule is added. Thus a low-confidence `keep`, `remove`, `refine`, or `add` is verified, while a high-confidence decision of any of those types bypasses the agent.

## Evidence verification

The evidence agent receives the global normal-pattern hypothesis, the global anomaly-pattern hypothesis, and one uncertain decision. It remains in the same response chain as the global hypothesis so the full-series context is retained. It can adaptively request one tool at a time and then returns `keep`, `remove`, or `refine`.

For a global `add` item, verified `keep` means the addition is accepted, `remove` rejects it, and `refine` changes the proposed boundaries.

## Final closure

High-confidence global decisions are applied directly. Low-confidence global decisions are replaced by their verified agent decisions. Removed intervals are discarded; retained intervals are merged only when they overlap or are immediately adjacent.
