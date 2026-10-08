# Dataset preparation

[Back to the project overview](../README.md) · [Usage guide](USAGE.md)

AgentVeriTS operates on **one univariate time series per CSV**. Download datasets from their upstream providers and keep them locally. This repository includes documentation and code, but does not redistribute the datasets.

## Sources and dataset names

The paper evaluates seven datasets. **IOPS and WSD come from TSB-AD.** For **Art, AWS, Tweets, MSL, and SMAP**, the names and download procedure below follow the original [VLM4TS repository](https://github.com/ZLHe0/VLM4TS) and its [download script](https://github.com/ZLHe0/VLM4TS/blob/main/src/preprocessing/download_data.py), which retrieves prepared CSVs from the Sintel / Orion data bucket.

| Paper name | Original data / benchmark release | Selection to download |
| :--- | :--- | :--- |
| IOPS | Operational KPI series, distributed by [TSB-AD](https://github.com/thedatumorg/TSB-AD) | Files with `_IOPS_` in `TSB-AD-U` |
| WSD | Web-service series, distributed by [TSB-AD](https://github.com/thedatumorg/TSB-AD) | Files with `_WSD_` in `TSB-AD-U` |
| Art | [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB): synthetic series with anomalies | `artificialWithAnomaly` |
| AWS | [NAB](https://github.com/numenta/NAB): AWS CloudWatch metrics | `realAWSCloudwatch` |
| Tweets | [NAB](https://github.com/numenta/NAB): tweet volumes mentioning companies | `realTweets` |
| MSL | [NASA / Telemanom](https://github.com/khundman/telemanom): Mars Science Laboratory rover telemetry | `MSL` |
| SMAP | [NASA / Telemanom](https://github.com/khundman/telemanom): Soil Moisture Active Passive satellite telemetry | `SMAP` |

See the [NAB data descriptions](https://github.com/numenta/NAB/blob/master/data/README.md) and [Telemanom data documentation](https://github.com/khundman/telemanom#data) for the original datasets. Retain upstream attribution and follow each provider's terms; TSB-AD links dataset-specific license information from its [dataset page](https://thedatumorg.github.io/TSB-AD/).

## Where to put the data

Run the commands below from the **AgentVeriTS repository root**. The suggested layout is:

| Local path | Contents |
| :--- | :--- |
| `data/raw/TSB-AD-U.zip` | Downloaded TSB-AD univariate archive |
| `data/raw/TSB-AD-U/` | Extracted TSB-AD CSVs; retain original filenames |
| `data/raw/VLM4TS/` | A local checkout of the upstream download helper |
| `data/raw/VLM4TS/data/<dataset>/` | CSVs downloaded by that helper for the five dataset names above |
| `data/raw/VLM4TS/data/datasets.csv` | Upstream dataset-to-signal manifest |
| `data/raw/VLM4TS/data/anomalies.csv` | Upstream anomaly intervals for external evaluation |
| `data/processed/IOPS/`, `data/processed/WSD/` | TSB-AD signals converted to explicit `timestamp,value,label` columns |
| `data/processed/<dataset>/` | Any other locally converted signals, if needed |

The entire `data/` directory is ignored by Git. These paths are conventions, not hard-coded requirements: `--input` accepts any local CSV path. The CLI processes one file per invocation; it does not automatically download or discover datasets.

## Download IOPS and WSD from TSB-AD

1. Open the official [TSB-AD download instructions](https://github.com/thedatumorg/TSB-AD/blob/main/Datasets/README.md).
2. Download **[TSB-AD-U.zip](https://www.thedatum.org/datasets/TSB-AD-U.zip)**, the univariate release. The GitHub repository contains examples; cloning it alone does not provide the full benchmark.
3. Extract locally, then select the CSVs whose names contain `_IOPS_` or `_WSD_`.

On Bash:

```bash
mkdir -p data/raw
curl --fail --location https://www.thedatum.org/datasets/TSB-AD-U.zip \
  --output data/raw/TSB-AD-U.zip
python -m zipfile -e data/raw/TSB-AD-U.zip data/raw/
```

You can also download and extract the archive through your browser or file manager. Locate the extracted `TSB-AD-U` directory; if the archive has an extra enclosing directory, adjust `source_root` in the conversion example below.

TSB-AD filenames encode the source dataset, signal identifier, training boundary, and first anomaly position. Preserve them when converting. The release uses a signal column and a `Label` column for univariate files. The following example creates an explicit input schema without dropping or reordering rows; save it locally as a Python script and run it from the repository root:

```python
from pathlib import Path
import numpy as np
import pandas as pd

source_root = Path("data/raw/TSB-AD-U")
for dataset in ("IOPS", "WSD"):
    files = sorted(source_root.rglob(f"*_{dataset}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No {dataset} CSVs under {source_root}")
    destination = Path("data/processed") / dataset
    destination.mkdir(parents=True, exist_ok=True)
    for source in files:
        frame = pd.read_csv(source)
        label_columns = [c for c in frame if str(c).strip().lower() == "label"]
        if len(label_columns) != 1:
            raise ValueError(f"Expected one Label column: {source}")
        label_column = label_columns[0]
        signal_columns = [c for c in frame if c != label_column]
        if len(signal_columns) != 1:
            raise ValueError(f"Expected a univariate TSB-AD file: {source}")
        signal = pd.to_numeric(frame[signal_columns[0]], errors="raise")
        labels = pd.to_numeric(frame[label_column], errors="raise")
        if frame.empty or not np.isfinite(signal).all() or not labels.isin([0, 1]).all():
            raise ValueError(f"Inspect missing values or invalid labels: {source}")
        output = pd.DataFrame({
            "timestamp": np.arange(len(frame)),
            "value": signal.to_numpy(),
            "label": labels.astype(int).to_numpy(),
        })
        output.to_csv(destination / source.name, index=False)
```

This conversion is deliberately strict so data-quality issues are visible before evaluation. The runtime loader can also read a headered TSB-AD file directly when exactly one numeric signal column remains after excluding `Label`. Labels never enter inference.

## Download Art, AWS, Tweets, MSL, and SMAP as in VLM4TS

Use the upstream **download helper only**. It needs `requests` and `tqdm`; no VLM4TS model execution or API key is needed to download data.

```bash
git clone https://github.com/ZLHe0/VLM4TS.git data/raw/VLM4TS
python -m pip install requests tqdm
python data/raw/VLM4TS/src/preprocessing/download_data.py \
  artificialWithAnomaly realAWSCloudwatch realTweets MSL SMAP
```

If you already have that checkout, reuse it and adjust the script path. The helper writes into **its own repository's `data/` directory**, regardless of your working directory. With the command above, files appear under `data/raw/VLM4TS/data/`. It also downloads `datasets.csv` and `anomalies.csv`. Keep these metadata files with the CSVs and check the helper's output for failed downloads.

The prepared files have `timestamp,value` columns and can be passed directly to AgentVeriTS. For example, after installing AgentVeriTS and setting `OPENAI_API_KEY` as described in the [usage guide](USAGE.md):

```bash
agentverits \
  --input data/raw/VLM4TS/data/artificialWithAnomaly/art_daily_jumpsup.csv \
  --config configs/agentverits_default.yaml \
  --output-dir outputs/art_daily_jumpsup
```

For manual download, the same bucket provides the [dataset manifest](https://sintel-orion.s3.us-east-2.amazonaws.com/datasets.csv), [anomaly metadata](https://sintel-orion.s3.us-east-2.amazonaws.com/anomalies.csv), and signal files at `https://sintel-orion.s3.us-east-2.amazonaws.com/<signal>.csv`. Select the dataset row in the manifest and use its signal names as filenames.

### Original-source alternatives

- **NAB:** download or clone [numenta/NAB](https://github.com/numenta/NAB). The original CSVs are in `data/artificialWithAnomaly/`, `data/realAWSCloudwatch/`, and `data/realTweets/`. Anomaly windows are in [`labels/combined_windows.json`](https://github.com/numenta/NAB/blob/master/labels/combined_windows.json). The CSVs already have `timestamp,value` columns.
- **MSL / SMAP:** follow the current download instructions in [Telemanom](https://github.com/khundman/telemanom). Its raw release contains per-channel `train/` and `test/` NumPy arrays and [`labeled_anomalies.csv`](https://github.com/khundman/telemanom/blob/master/labeled_anomalies.csv). In each two-dimensional array, the **first feature is the telemetry signal**; additional features encode commands. Export the first column to a CSV `value` column, with row indices as `timestamp`. Do not flatten the array or treat command features as extra time steps.

For the VLM4TS preparation route, prefer the prepared Orion CSVs above. Original-source files may use different timestamps, splits, or annotation conventions; keep each version paired with its own labels.

## Required input format

The recommended schema is a UTF-8, comma-separated CSV **with a header**:

```csv
timestamp,value,label
0,0.12,0
1,0.15,0
2,0.91,1
3,0.14,0
```

This is a small synthetic schema example, not a benchmark sample.

| Field | Requirement | Runtime behavior |
| :--- | :--- | :--- |
| `value` | One numeric signal; recommended column name | Read as floating-point values |
| `timestamp` | Optional; numeric or textual time values | Preserved in file order; absent timestamps become `0, 1, …, N−1` |
| `label` | Optional; conventionally `0` for normal and `1` for anomalous | Ignored by inference; retain for external evaluation |

- The value aliases `data`, `kpi`, `metric`, and `y` are recognized, as are time aliases `time`, `date`, and `datetime` (case-insensitive). Prefer the explicit names above.
- Without a recognized value name, the loader accepts exactly one numeric signal column after excluding recognized time and label columns. Multiple candidates are rejected. It does not split multivariate input into separate runs.
- Rows are processed **in file order**. AgentVeriTS does not sort, resample, or parse timestamps. Prepare chronological rows and align labels before running.
- Non-numeric values and infinities in the signal become missing values. The CSV loader interpolates gaps and fills endpoints without dropping rows; an empty file or a signal with no finite numeric values is rejected. For benchmark evaluation, resolve missing-data handling explicitly and record any preprocessing.
- The CLI accepts CSVs, not ZIP archives, `.npy` arrays, folders, or annotation JSON files.

## Labels and reproducible evaluation

AgentVeriTS returns anomaly intervals as **zero-based, inclusive row indices** into the exact input file. These are not timestamp values. Ground truth is not read by the pipeline, and the CLI does not compute benchmark F1 automatically.

| Source | Ground truth | Alignment with predictions |
| :--- | :--- | :--- |
| TSB-AD | Per-row `Label` column | Preserve row count/order when converting or slicing |
| VLM4TS / Orion | `anomalies.csv`: `signal` and timestamp-based `events` | Match the signal name, then map events to the matching CSV's timestamp rows using the evaluation protocol |
| Original NAB | `combined_windows.json`: timestamp windows keyed by relative file path | Use the original CSV timestamps and the chosen NAB evaluation convention |
| Original Telemanom | `labeled_anomalies.csv`: `channel_id`, `spacecraft`, and `anomaly_sequences` | Sequence indices refer to the channel's **test** array; offset them if concatenating train and test |

Keep a record of the source version, selected filenames/channels, train/test selection, preprocessing, and metric/interval conventions. Downloading an upstream collection does not by itself define the paper's exact experimental subset. The current public release does not bundle an exact per-file experiment manifest or a complete benchmark evaluation harness; the README tables are manuscript results. See the [usage guide](USAGE.md) and [implementation audit](MIGRATION_20260924.md) for validation scope.
