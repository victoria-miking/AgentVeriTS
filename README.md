# REVA

**REVA: Reference-Guided Screening and Evidence-Driven Agentic Verification for Time Series Anomaly Detection**

This repository contains a clean public implementation of the complete REVA inference pipeline for univariate time-series anomaly detection.

## Method flow

REVA separates broad visual screening from evidence-driven agentic verification:

1. **Reference-guided visual screening** renders multi-scale windows, retrieves non-overlapping visually similar reference windows, retains robust references, and produces dense anomaly scores and visual candidate intervals.
2. **Global Anomaly Hypothesis** reviews the full-series plot. Orange spans are coarse visual candidate windows. Every candidate receives `keep`, `remove`, or `refine`; highly suspicious missed regions receive `add`.
3. **Discrete confidence routing** assigns every global decision confidence `1`, `2`, or `3`. Confidence `1/2` enters the evidence agent; confidence `3` closes directly. Action type does not independently trigger verification.
4. **Evidence-driven verification** lets the agent request targeted global/local plots, raw samples, robust statistics, non-overlapping historical references, scale views, or spike diagnostics to resolve confidence-1/2 decisions.
5. **Evidence-aware global rescan** keeps the agent's global anomaly-search responsibility: after uncertain decisions are verified, the same continuous session searches the complete series again for still-missed anomalies and may gather evidence around new suspected regions.
6. **Deterministic closure** combines confidence-3 global decisions, verified confidence-1/2 decisions, and evidence-supported global-rescan discoveries.

The orange candidate windows are visual relative anomalies from a coarse-grained, purely visual detector. They may contain false positives—local shapes that look unusual but are globally normal—and may miss statistically or contextually abnormal regions whose visual deviation is subtle. The global reasoning stages therefore infer the sequence's likely anomaly morphology rather than treating orange spans as ground truth.

## Confidence scale

All global and evidence decisions use the same discrete scale:

- `1` — **low confidence**: ambiguity remains or the deviation is very subtle; roughly 50%–70%.
- `2` — **medium confidence**: local abnormality is fairly clear, but the global interpretation remains uncertain; roughly 70%–95%.
- `3` — **high confidence**: strong statistical or contextual evidence supports the decision; above roughly 95%.

These percentages are calibration guides rather than exact probabilities.

## Important inference rule

The public inference path **never uses ground-truth labels to choose a screening threshold**. `screening.alpha` is a runtime configuration (default `0.01`). The implementation can emit candidate sets for `0.1`, `0.01`, and `0.001`, but the selected runtime candidate set is determined only by configuration.

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
pip install -e .
```

Set the model API key:

```bash
export OPENAI_API_KEY="..."
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="..."
```

## Input format

The CLI accepts a CSV with a `value` column and optional `timestamp` column. Common alternatives such as `data`, `kpi`, `metric`, or `y` are also recognized. Label columns are not required and are not consumed by inference.

```text
timestamp,value
0,0.31
1,0.29
2,0.33
...
```

## Run

```bash
reva \
  --input data/example.csv \
  --config configs/reva_default.yaml \
  --output-dir outputs/example
```

Useful overrides:

```bash
reva --input data/example.csv --alpha 0.01 --model gpt-5.6-sol
```

## Output

Each run writes:

```text
outputs/example/
├── screening.json
├── global_candidates.png
├── global_hypothesis.json
├── routing.json
├── evidence/
│   ├── <decision_id>.json
│   ├── global_rescan.json
│   └── evidence images ...
└── result.json
```

`global_hypothesis.json` is the auditable boundary between full-series visual reasoning and evidence verification. Example:

```json
{
  "decision_id": "V0001",
  "source": "candidate",
  "candidate_id": "V0001",
  "reviewed_interval": [1200, 1285],
  "action": "refine",
  "final_interval": [1214, 1271],
  "confidence": 2,
  "rationale": "..."
}
```

Added regions use `source: "added"`, `candidate_id: null`, and `action: "add"`.

## Evidence tools

The evidence agent exposes seven read-only tools: `global_context`, `local_context`, `reference_context`, `raw_segment`, `stat_features`, `scale_view`, and `spike_scan`. Tool calls are adaptive rather than a fixed sequence.

Historical similarity is retrieval evidence only. A similar historical window is not automatically normal.

## Reproducibility defaults

The visual screening stage uses three temporal scales (`224`, `448`, `672`), quarter-window stride, top-16 non-overlapping visual references, robust retention of four references, multi-scale patch discrepancy, robust positive normalization across scales, and equal scale fusion. See `configs/reva_default.yaml` for the public runtime configuration.

## Repository layout

```text
src/reva/
├── visual/              # rendering, visual encoder, reference-guided screening
├── reasoning/           # prompts, global hypothesis, routing, evidence tools, agent
├── config.py
├── io.py
├── pipeline.py
└── cli.py
```

The repository intentionally excludes one-off development experiments, private paths, cached outputs, API credentials, and prompt text tied to external comparison methods.
