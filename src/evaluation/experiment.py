"""Original interval F1max: max across three alphas per signal, then macro mean."""

import csv
from collections import defaultdict
from pathlib import Path
from .evaluate import evaluate_intervals
from agentverits.workflow import atomic_json, read_json

ALPHAS = (0.1, 0.01, 0.001)


def alpha_name(alpha):
    return f"alpha_{alpha:g}"


def summarize(roster, cache_root, output_root):
    groups = defaultdict(list)
    per_alpha = []
    complete_signals = []
    for entry in roster:
        dataset, sid = entry["dataset"], entry["signal_id"]
        groups[dataset]  # include datasets with zero completed results
        signal_root = Path(cache_root) / dataset / sid
        manifest_path = signal_root / "signal_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        gt = manifest["gt_intervals"]  # Evaluation only, never a model input.
        metrics = []
        for alpha in ALPHAS:
            result_path = (
                Path(output_root) / dataset / sid / alpha_name(alpha) / "result.json"
            )
            status_path = result_path.with_name("status.json")
            if (
                not result_path.is_file()
                or not status_path.is_file()
                or read_json(status_path).get("status") != "ok"
            ):
                continue
            result = read_json(result_path)
            screening = next(
                r["intervals"]
                for r in manifest["decision"]["alphas"]
                if abs(r["alpha"] - alpha) < 1e-12
            )
            assessed = read_json(result_path.with_name("assessment.json"))["decisions"]
            from agentverits.decisions import final_intervals

            row = {
                "dataset": dataset,
                "signal_id": sid,
                "alpha": alpha,
                "Anomaly_Screening": evaluate_intervals(gt, screening),
                "Candidate_Assessment": evaluate_intervals(
                    gt, final_intervals(assessed)
                ),
                "Agentic_Verification": evaluate_intervals(
                    gt, result["final_intervals"]
                ),
                "usage": result["usage"],
            }
            metrics.append(row)
            per_alpha.append(row)
        if len(metrics) != 3:
            continue
        best = {
            stage + "_F1max": max(r[stage]["F1"] for r in metrics)
            for stage in (
                "Anomaly_Screening",
                "Candidate_Assessment",
                "Agentic_Verification",
            )
        }
        signal_row = {
            "dataset": dataset,
            "signal_id": sid,
            **best,
            "Agent_Best_alpha": max(
                metrics, key=lambda r: r["Agentic_Verification"]["F1"]
            )["alpha"],
        }
        groups[dataset].append(signal_row)
        complete_signals.append(signal_row)
    expected = {d: sum(r["dataset"] == d for r in roster) for d in groups}
    dataset_rows = []
    for dataset, rows in sorted(groups.items()):
        count = len(rows)
        row = {
            "dataset": dataset,
            "completed_signals": count,
            "expected_signals": expected[dataset],
            "complete": count == expected[dataset],
        }
        for stage in (
            "Anomaly_Screening",
            "Candidate_Assessment",
            "Agentic_Verification",
        ):
            row[stage + "_F1max"] = (
                sum(r[stage + "_F1max"] for r in rows) / count if count else None
            )
        row["net_F1max_gain"] = (
            row["Agentic_Verification_F1max"] - row["Anomaly_Screening_F1max"]
            if count
            else None
        )
        dataset_rows.append(row)
    summary = {
        "scoring": "interval overlap F1; per signal max over alpha=0.1/0.01/0.001; dataset macro average",
        "complete": len(complete_signals) == len(roster),
        "completed_signals": len(complete_signals),
        "expected_signals": len(roster),
        "completed_alpha_runs": len(per_alpha),
        "datasets": dataset_rows,
        "signals": complete_signals,
        "per_alpha": per_alpha,
    }
    atomic_json(Path(output_root) / "metrics_summary.json", summary)
    fields = list(dataset_rows[0]) if dataset_rows else ["dataset"]
    with (Path(output_root) / "dataset_f1max.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dataset_rows)
    with (Path(output_root) / "signal_f1max.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        fields = (
            list(complete_signals[0]) if complete_signals else ["dataset", "signal_id"]
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(complete_signals)
    return summary
