# REVA

**REVA: Reference-Guided Screening and Evidence-Driven Agentic Verification for Time Series Anomaly Detection**

This repository contains a clean public implementation of the complete REVA inference pipeline for univariate time-series anomaly detection.

## Method flow

REVA separates broad visual screening from expensive evidence verification:

1. **Reference-guided visual screening** renders multi-scale windows, retrieves non-overlapping visually similar reference windows, retains robust references, and produces a dense anomaly score.
2. **Global Anomaly Hypothesis** reviews the full-series plot. Orange spans are coarse visual candidate windows. Every candidate receives `keep`, `remove`, or `refine`; highly suspicious missed regions receive `add`. Every decision also receives a confidence in `[0,1]`.
3. **Confidence-only routing** sends only decisions below the configured confidence threshold to the evidence agent. Action type, edit magnitude, and interval length do not independently trigger verification.
4. **Evidence-driven verification** lets the agent request targeted global/local plots, raw samples, robust statistics, non-overlapping historical references, scale views, or spike diagnostics. It then finalizes the uncertain item as `keep`, `remove`, or `refine`.
5. **Deterministic closure** combines high-confidence global decisions with verified uncertain decisions and writes the final anomaly intervals.

The global hypothesis is intentionally broader than candidate verification. The orange candidates are relative visual abnormalities from a coarse visual detector: they may include globally normal fluctuations and may miss subtle contextual or statistical anomalies. The model therefore learns the sequence-level anomaly morphology from the complete plot and may add a strongly indicated missed interval.

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
reva --input data/example.csv --alpha 0.01 --confidence-threshold 0.75 --model gpt-5.6-sol
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
│   └── evidence images ...
└── result.json
```

`global_hypothesis.json` is the auditable boundary between global visual reasoning and agentic evidence verification. Its decisions have the following contract:

```json
{
  "decision_id": "V0001",
  "source": "candidate",
  "candidate_id": "V0001",
  "reviewed_interval": [1200, 1285],
  "action": "refine",
  "final_interval": [1214, 1271],
  "confidence": 0.61,
  "rationale": "..."
}
```

Added regions use `source: "added"`, `candidate_id: null`, and `action: "add"`.

## Evidence tools

The evidence verifier exposes seven read-only tools: `global_context`, `local_context`, `reference_context`, `raw_segment`, `stat_features`, `scale_view`, and `spike_scan`. Tool calls are adaptive rather than a fixed sequence, and the configured evidence budget defaults to three calls per uncertain decision.

Historical similarity is treated only as retrieval evidence. A similar reference is not automatically considered normal.

## Reproducibility defaults

The visual screening stage uses three temporal scales (`224`, `448`, `672`), quarter-window stride, top-16 non-overlapping visual references, robust retention of four references, multi-scale patch discrepancy, robust positive normalization across scales, and equal scale fusion. See `configs/reva_default.yaml` for the public runtime configuration.

## Repository layout

```text
src/reva/
├── visual/              # rendering, visual encoder, reference-guided screening
├── reasoning/           # prompts, structured global reasoning, routing, tools, agent
├── config.py
├── io.py
├── pipeline.py
└── cli.py
```

The repository intentionally excludes one-off development experiments, old comparison runners, private paths, cached outputs, API credentials, and prompt text tied to external comparison methods.
