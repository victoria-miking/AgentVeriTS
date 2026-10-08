# Usage guide

[Back to the project overview](../README.md)

## Installation

Python 3.10+ is supported. Install a PyTorch/torchvision build suitable for your device before installing the package if needed.

```bash
pip install -r requirements.txt
pip install -e .
```

The public dependency pins are in `pyproject.toml` and `requirements.txt`. The first screening run downloads the pretrained OpenCLIP weights. No labeled training data is consumed.

## Official OpenAI API

Use an **official OpenAI API key**:

```bash
export OPENAI_API_KEY="your-api-key"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

The client uses the official SDK and `https://api.openai.com/v1/responses`. `OPENAI_BASE_URL` does not override that endpoint. The default model is the paper's `gpt-5.6-sol`; access depends on your API account.

- Candidate assessment starts a stored response with `store=True`.
- Each verification request sends `previous_response_id` plus only the new target or evidence observation and new images.
- The latest response ID is carried across tool rounds and uncertain targets. Each signal starts its own chain.
- Instructions are supplied on every request. GPT-5.6-family requests also use `reasoning.context=all_turns` for available compatible reasoning state.
- The SDK handles bounded transient retries. Refused, incomplete or invalid responses fail explicitly. A missing/expired parent response never silently falls back to a fresh conversation or manual history reconstruction.

Tool selection follows the paper's **structured action/observation protocol**: the model returns a closed JSON action, the local system executes the selected evidence tool, and the next response receives its observation. This uses official Responses structured outputs; it does not claim native function-call events. See [API details](OPENAI_API.md).

## Input and execution

For the seven datasets used in the paper, see the [dataset preparation guide](DATASETS.md): sources, downloads, suggested local directories, conversion, and ground-truth alignment. Datasets are downloaded separately and are not bundled with this repository.

Input is a CSV with a `value` column and an optional `timestamp` column. The aliases `data`, `kpi`, `metric`, and `y` are recognized. A single unnamed numeric signal column is also accepted after excluding time and label columns; ambiguous input is rejected. CSV gaps are interpolated, with endpoint filling. Direct Python input must be finite. Intervals use **zero-based, inclusive endpoints**.

CSV files must include a header. Rows are consumed in file order; the loader does not sort or resample timestamps. Labels are not passed to inference. Use one signal per file and keep its evaluation labels aligned with the same rows.

```bash
agentverits --input data/example.csv \
  --config configs/agentverits_default.yaml \
  --output-dir outputs/example

agentverits --input data/example.csv --alpha 0.01 \
  --model gpt-5.6-sol --confidence-threshold 0.95
```

```python
from agentverits import AgentVeriTSConfig, AgentVeriTSPipeline

config = AgentVeriTSConfig.from_yaml("configs/agentverits_default.yaml")
result = AgentVeriTSPipeline(config).run(values, signal_id="example", output_dir="outputs/example")
print([interval.as_list() for interval in result.final_intervals])
```

The runtime screening threshold is chosen by configuration, never from ground-truth labels. Candidate sets for `alpha` in `{0.1, 0.01, 0.001}` are emitted; the selected set is `screening.alpha` (default `0.01`). To evaluate multiple thresholds, run the complete pipeline separately for each. This release does not recreate the paper's benchmark tables or repeated-run measurements.

## Evidence tools

| Paper tool | Public tool name | Inputs and result |
| --- | --- | --- |
| Raw values | `raw` | Target interval, context and paging controls; exact consecutive numeric samples, never silent subsampling |
| Focused plot | `focus` | Interval, context and local/shared y-range; high-resolution local plot |
| References | `reference` | Interval, retrieval scale/count and plotting context; retained screening references, with shape retrieval as a fallback |
| Statistics | `statistics` | Interval, predefined baseline and diagnostic; robust deviations or peak/trough diagnostics |

Plot-scale options and spike diagnostics are modes within these four tools. The global image remains available through the response chain. Similar references can contain anomalies; similarity is not proof of normality. Long raw segments are returned in exact pages with `next_start` so omitted samples are explicit.

## Outputs

Each run writes `screening.json`, `global_candidates.png`, `global_hypothesis.json`, `routing.json`, `api_calls.json`, per-target evidence records/images under `evidence/`, and `result.json`. API logs contain response IDs, parent IDs, status and usage; no keys or encoded image payloads. They are also written when inference fails. Use a fresh output directory for independent experiments.

The public defaults retain the existing retrieval and rendering details: scales `{224,448,672}`, quarter-window stride, top-16 references, four robustly retained references, patch/mid/large visual neighborhoods, robust positive score normalization, equal scale fusion, smoothing, and 4096x512 global plots. See [the implementation contract](PIPELINE.md) and [migration audit](MIGRATION_20260924.md).

## Tests

```bash
pip install -e '.[test]'
python -m unittest discover -s tests -v
```

Tests cover routing at the threshold, interval operations, four-tool dispatch, exact raw evidence, reference reuse, numerical patch/time alignment, end-to-end closure, and real OpenAI SDK serialization/retries using a mocked HTTP transport. Tests do not call a paid model API or require model weights.
