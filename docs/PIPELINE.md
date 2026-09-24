# AgentVeriTS implementation contract

## Anomaly Screening — Eqs. (1)–(3)

A univariate signal is detrended and normalized using the existing rendering convention. Window lengths are 224, 448 and 672, with quarter-window strides. A final window is anchored at the actual signal end when the regular stride grid leaves a tail.

Frozen CLIP ViT-B/16 features retrieve top-16 windows that do not overlap the query. The existing robust retention selects four references. Cosine distance to the closest reference patch gives patch discrepancies. Existing mid/large neighborhood features and harmonic fusion are retained. All mask indices are zero based.

Window maps are resized to their actual temporal windows and overlap-averaged into the aligned anomaly map. The top 0.25 fraction along the visual axis is then averaged, in the order specified by Eq. (2). Robust positive normalization, equal scale fusion and configured smoothing are retained. A Gaussian-quantile threshold extracts contiguous intervals. Label data is never used to select the runtime alpha.

## Candidate Assessment — Eqs. (4)–(5)

One global plot highlights all selected candidates. The VLM receives it with candidate intervals and task instructions. Its parsed global anomaly hypothesis contains one record for every original candidate, plus optional added regions. Original candidates keep their identifiers and reviewed endpoints.

Each record contains:

- `reviewed_interval` and `final_interval`: zero-based inclusive endpoints; the latter is null for REMOVE.
- `action`: KEEP, REMOVE, REFINE or ADD (lowercase in JSON).
- `confidence`: finite numeric q in [0,1], measuring confidence in the operation and its boundaries.
- `rationale`: a brief explanation.

Only q < tau_G enters verification. The default tau_G is 0.95. Equality belongs to the direct branch; action type does not override routing.

## Agentic Verification — Eqs. (6)–(7)

The agent processes the low-confidence set using the global image, full hypothesis and accumulated evidence in a continuous response chain. Processing targets sequentially is an implementation detail; it does not reset context or reopen candidate-wise conversations.

The model can request one of four evidence tools or finalize. Each tool accepts a target interval and typed parameters. Unused parameters are null in the strict response schema and omitted before dispatch. Raw evidence is returned as exact consecutive pages; focus supports local/shared axes; reference supports retrieval scale/count; statistics supports surrounding, left, right or global baselines, excluding the target.

The original maximum of three tool requests per uncertain target is retained. Invalid tool parameters consume a turn and return an error observation in the same chain. On budget exhaustion the schema permits only finalization. If existing global evidence is sufficient, the agent may finalize without a tool call. It reports uncertainty honestly; final confidence is not forced above 0.95.

KEEP preserves the current proposed target boundaries. REMOVE discards the target. REFINE changes its boundaries. ADD accepts a proposed added target or records newly supported missed intervals in `additions`. The agent retains the ability to inspect other suspected regions using global context during verification. High-confidence records themselves are never sent back for verification. The legacy unconditional global rescan is removed because the paper restricts verification to the uncertain set.

## Final intervals — Eq. (8)

Apply high-confidence operations directly (excluding REMOVE). Replace each uncertain operation with its verified result, include agent additions, and merge overlapping or immediately adjacent accepted intervals. A high-confidence REMOVE is therefore never accidentally retained merely because its confidence is high.

Each signal has a separate response chain. The complete global plot is sent once; new evidence images and observations are added incrementally. Response IDs are recorded for auditing. The client does not reconstruct past messages or silently restart a failed chain.
