# AgentVeriTS paper-alignment audit — 2026-09-24

Base public commit: `ee4fe2a236c6cd1df1f80c4d3a2e09fa5a5e2916`.
Source of methodological authority: *AgentVeriTS: Confidence-Guided Agentic Verification for Time-Series Anomaly Detection*, the supplied manuscript. This is an implementation migration, not a reproduction of the paper's reported measurements.

## Paper-to-code mapping

| Paper requirement | Previous public code | Current implementation |
| --- | --- | --- |
| AgentVeriTS title and repository identity | REVA package, CLI, config and documentation | `agentverits` package and CLI; `AgentVeriTSConfig` and `AgentVeriTSPipeline`; updated examples, metadata and links |
| Three named modules | Visual screening, global hypothesis and evidence module | Anomaly Screening, Candidate Assessment and Agentic Verification; the global hypothesis remains the output of Candidate Assessment |
| Eq. (1), cosine patch-bank discrepancy | Existing closest-patch comparison | Preserved |
| Eq. (2), aligned map followed by top-rho reduction | Reduced each window before overlap fusion | Overlap-average on actual temporal positions, then top-rho reduction |
| Eq. (3), Gaussian-quantile screening | Existing implementation | Preserved, with finite-data and alpha validation |
| Eqs. (4)–(5), q in [0,1], tau_G=0.95 | Discrete confidence 1/2/3 | Numeric confidence and configurable threshold, default 0.95; only strict q < tau_G enters verification |
| Eq. (6), four evidence tools | Seven exposed tool names | `raw`, `focus`, `reference`, `statistics`; plot-scale and spike options are internal modes |
| Eq. (7), agent verifies the uncertain set using global context | Sequential verification plus unconditional global rescan | Continuous response-chain verification; ADD remains available inside verification; no mandatory extra pass when there is no uncertain decision |
| Eq. (8), direct hypotheses union verified outputs | Extra rescan discoveries always appended | Apply high-confidence operations, exclude REMOVE, and merge verified outputs plus verification-time additions |

## Official API migration

The base public repository already used `previous_response_id`; the experiment-time relay/replay client was not present in this public checkout. This migration hardens the public stored-response integration:

- Explicit official OpenAI endpoint; no `OPENAI_BASE_URL` relay override.
- `store=True`, parent ID propagation across all tool rounds and verification targets, a fresh chain for each signal, per-request instructions and incremental inputs.
- GPT-5.6 `reasoning.context=all_turns` for compatible available reasoning state.
- Closed structured-output schemas instead of the previous unrestricted `arguments` object, which was incompatible with strict schemas.
- Explicit checks for completion, refusal, JSON shape and response IDs; bounded SDK retries preserve the parent.
- Response ID, status and token-usage audit records, including when a later request fails.

The paper's structured action/observation protocol is retained. Local tools are not reimplemented as native function-call events. See [OPENAI_API.md](OPENAI_API.md).

## Correctness fixes

- Harmonic patch projection used `patch_index + 1` against a zero-based encoder mask, shifting assignments and leaving one edge inconsistent. Projection now uses matching zero-based indices.
- The end of a signal could be filled by linear score extrapolation. A real final window now covers the tail, and scores are mapped to actual window starts.
- Raw evidence silently subsampled long intervals to at most 160 points. It now returns exact consecutive pages with explicit remaining-page information.
- A KEEP result could silently change its interval; it must now preserve the current boundaries. Unknown operations, non-finite/out-of-range confidence and noninteger/out-of-bounds endpoints are rejected.
- Model-generated record IDs could be used as arbitrary output paths. Record identifiers are validated before file use.
- Repeated plot requests overwrote earlier evidence images. Each observation now has a distinct path.
- Numeric CSV inference could select a label column. Known time/label columns are excluded; ambiguous numeric inputs and entirely missing signals are rejected.
- Empty statistical baselines no longer manufacture a zero-valued comparison and large deviation; missing comparisons are explicitly null.
- OpenCLIP's model configuration is preserved while enabling patch-token output explicitly.
- JSON records are written atomically and reject non-finite JSON numbers.

## Optimizations and retained details

Rendering is batched before encoder calls instead of stacking all rendered windows in memory. Token pooling and harmonic projection use vectorized tensor operations. Empty selected candidate sets no longer cause redundant threshold computation. Image payloads use a bounded encoding cache.

The existing defaults and mechanisms that the paper does not expand remain: quarter-window strides, top-16 retrieval, robust selection of four references, mid/large visual neighborhoods, harmonic neighborhood fusion, input image normalization convention, detrending, robust positive cross-scale normalization, equal scale fusion, smoothing, evidence budgets and rendering sizes. Reference contamination cautions remain in prompts. No labels, new training phase, benchmark-specific rules or new detection model were introduced.

## Validation

- 33 regression tests: passed. Covers confidence boundaries, direct REMOVE, full output merging, new anomalies, response chaining across targets and signals, exact raw pages, reference reuse, interval validation, strict schemas and SDK error/retry behavior.
- Editable installation, wheel build, Python 3.10 syntax compatibility, CLI entry points, YAML/dataclass default agreement and relative documentation links: passed.
- Full numerical screening smoke: passed with a deterministic test encoder on 1,401 points, all three temporal scales, real rendering/retrieval/scoring and verified tail coverage.
- OpenCLIP ViT-B/16 initialization and forward smoke with random weights: patch/mid/large shapes `(1,196,768)`, `(1,169,768)`, `(1,144,768)`.
- Local CPU microbenchmark, one thread, 20 windows and 144 pooled neighborhoods: corrected reference loop 13.557 ms/call; vectorized projection 0.743 ms/call. Numerical agreement was checked. These measurements concern only this kernel, not total inference time or RTX 5090 performance.

The local regression runtime used Python 3.12, torch 2.9.1+cpu, torchvision 0.24.1+cpu, NumPy 2.3.5, pandas 2.2.3, matplotlib 3.10.8, open_clip_torch 3.2.0, openai 2.11.0, SciPy 1.17.0, Pillow 12.3.0 and PyYAML 6.0.3. CI installs the repository's pinned dependencies on Python 3.10.

No official API key or benchmark datasets were available during this migration. Authenticated model inference, pretrained-weight benchmark evaluation and the paper's F1/runtime tables have **not** been rerun. Fixing numerical alignment and changing confidence/routing can change predictions; the existing paper results must not be represented as measurements of this revision without rerunning the evaluation.

## Compatibility

The old `reva` command and top-level `REVAConfig`/`REVAPipeline` imports are deprecated aliases. Legacy internal module imports, discrete-confidence records, old config paths and the removed `max_global_rescan_calls` option must be migrated. The GitHub repository rename is a separate hosting setting; source-code branding alone does not change its URL.
