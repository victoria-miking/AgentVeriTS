# REVA pipeline contract

## Global hypothesis

The global module receives one complete signal image with orange visual candidate spans. It must return every input candidate once, preserving its candidate ID and reviewed interval. Original candidates can only receive `keep`, `remove`, or `refine`. A missed region can enter through a separate `add` record.

The confidence value is discrete and measures confidence in the selected action and its boundaries:

- `1` — low confidence: ambiguous or very subtle deviation, roughly 50%–70%.
- `2` — medium confidence: local abnormality is fairly clear but global interpretation remains uncertain, roughly 70%–95%.
- `3` — high confidence: strong statistical or contextual evidence, above roughly 95%.

The percentages are calibration guides, not exact probabilities.

## Verification routing

Routing is fixed and action-independent:

```text
confidence = 1 or 2  -> evidence verification
confidence = 3       -> direct closure
```

A `keep`, `remove`, `refine`, or `add` action is never routed merely because of its action type, edit magnitude, or interval length.

## Evidence verification

The evidence agent receives the global normal-pattern hypothesis, the global anomaly-pattern hypothesis, and one confidence-1/2 decision at a time. It remains in the same response chain as the global hypothesis so the full-series context is retained. It can adaptively request one evidence tool at a time and then returns `keep`, `remove`, or `refine`.

For a global `add` item, verified `keep` accepts the addition, `remove` rejects it, and `refine` changes its boundaries.

## Evidence-aware global rescan

After all confidence-1/2 decisions have been verified, the same evidence agent performs one additional global search over the full sequence.

This rescan is allowed to discover anomalies still missed by both the visual screening stage and the first global hypothesis. The agent can revisit the global plot and call local, statistical, raw-value, reference, scale, or spike tools around newly suspected regions.

Only evidence-supported discoveries with confidence `2` or `3` are emitted as new intervals. A remaining confidence-1 suspicion is not added to the final result.

## Final closure

Confidence-3 global decisions are applied directly. Confidence-1/2 global decisions are replaced by their verified agent decisions. Evidence-aware global-rescan discoveries are then added, and overlapping or immediately adjacent intervals are merged deterministically.
