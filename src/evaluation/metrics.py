"""Original CandidateAssessment F1max evaluation; point F1max is reported separately."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from preprocessing.datasets import intervals_to_index_space


def safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if average_precision_score is None:
        return float("nan")
    if labels.sum() == 0:
        return float("nan")
    return float(average_precision_score(labels, scores))


def point_f1max(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if precision_recall_curve is None or labels.sum() == 0:
        return {
            "Point_F1max": np.nan,
            "Point_Precision_at_F1max": np.nan,
            "Point_Recall_at_F1max": np.nan,
            "Point_Threshold_at_F1max": np.nan,
        }
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    idx = int(np.nanargmax(f1))
    thr = thresholds[idx] if idx < len(thresholds) else np.nan
    return {
        "Point_F1max": float(f1[idx]),
        "Point_Precision_at_F1max": float(precision[idx]),
        "Point_Recall_at_F1max": float(recall[idx]),
        "Point_Threshold_at_F1max": float(thr),
    }


def safe_eval_intervals(
    evaluate_intervals_func,
    gt_intervals: List[List[int]],
    pred_intervals: List[List[int]],
) -> Dict[str, float]:
    if not gt_intervals:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan}
    try:
        res = evaluate_intervals_func(gt_intervals, pred_intervals)
        return {
            "precision": float(res.get("precision", np.nan)),
            "recall": float(res.get("recall", np.nan)),
            "f1": float(res.get("F1", res.get("f1", np.nan))),
        }
    except Exception:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan}


def intervals_from_scores_by_alpha(
    scores: np.ndarray,
    timestamps: np.ndarray,
    alpha: float,
    compute_detection_intervals_func,
    intervals_from_indices_func,
) -> Tuple[List[List[Any]], List[List[int]]]:
    interval_indices, _, _ = compute_detection_intervals_func(
        score_vector=scores, alpha=alpha
    )
    pred_df = intervals_from_indices_func(
        interval_indices=interval_indices, timestamps=timestamps, scores=scores
    )
    pred_intervals_ts = (
        pred_df[["start", "end"]].values.tolist() if len(pred_df) else []
    )
    pred_intervals_idx = intervals_to_index_space(pred_intervals_ts, timestamps)
    return pred_intervals_ts, pred_intervals_idx


def evaluate_one_score(
    model_name: str,
    scores: np.ndarray,
    timestamps: np.ndarray,
    labels: np.ndarray,
    gt_intervals_idx: List[List[int]],
    alphas: Sequence[float],
    repo: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    scores = np.asarray(scores, dtype=float)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    per_alpha_rows: List[Dict[str, Any]] = []
    best_f1 = -1.0
    best_f1_alpha = None
    best_prec = np.nan
    best_rec = np.nan
    best_aff = -1.0
    best_aff_alpha = None

    vus_res = {"VUS_PR": np.nan, "VUS_ROC": np.nan, "AUC_PR": np.nan, "AUC_ROC": np.nan}
    if repo["evaluate_vus_pr"] is not None and gt_intervals_idx:
        try:
            vus_res = repo["evaluate_vus_pr"](scores, gt_intervals_idx)
        except Exception:
            pass

    for alpha in alphas:
        _, pred_intervals_idx = intervals_from_scores_by_alpha(
            scores=scores,
            timestamps=timestamps,
            alpha=float(alpha),
            compute_detection_intervals_func=repo["compute_detection_intervals"],
            intervals_from_indices_func=repo["intervals_from_indices"],
        )
        f1_res = safe_eval_intervals(
            repo["evaluate_intervals"], gt_intervals_idx, pred_intervals_idx
        )

        aff_res = {
            "Affiliation_Precision": np.nan,
            "Affiliation_Recall": np.nan,
            "Affiliation_F1": np.nan,
        }
        if repo["evaluate_affiliation_f1"] is not None and gt_intervals_idx:
            try:
                aff_res = repo["evaluate_affiliation_f1"](
                    gt_intervals=gt_intervals_idx,
                    pred_intervals=pred_intervals_idx,
                    series_length=len(scores),
                )
            except Exception:
                pass

        per_alpha_rows.append(
            {
                "model": model_name,
                "alpha": float(alpha),
                "n_detected": len(pred_intervals_idx),
                "n_gt": len(gt_intervals_idx),
                "precision": f1_res["precision"],
                "recall": f1_res["recall"],
                "f1": f1_res["f1"],
                "Affiliation_Precision": aff_res.get("Affiliation_Precision", np.nan),
                "Affiliation_Recall": aff_res.get("Affiliation_Recall", np.nan),
                "Affiliation_F1": aff_res.get("Affiliation_F1", np.nan),
                "VUS_PR": vus_res.get("VUS_PR", np.nan),
                "VUS_ROC": vus_res.get("VUS_ROC", np.nan),
                "AUC_PR": vus_res.get("AUC_PR", np.nan),
                "AUC_ROC": vus_res.get("AUC_ROC", np.nan),
            }
        )

        if not np.isnan(f1_res["f1"]) and f1_res["f1"] > best_f1:
            best_f1 = f1_res["f1"]
            best_f1_alpha = float(alpha)
            best_prec = f1_res["precision"]
            best_rec = f1_res["recall"]

        aff_f1 = aff_res.get("Affiliation_F1", np.nan)
        if not np.isnan(aff_f1) and aff_f1 > best_aff:
            best_aff = float(aff_f1)
            best_aff_alpha = float(alpha)

    point_res = point_f1max(labels, scores)
    auprc = safe_auprc(labels, scores)

    row = {
        "model": model_name,
        "F1max": np.nan if best_f1 < 0 else best_f1,
        "Best_F1_alpha": best_f1_alpha,
        "Precision_at_Best_F1_alpha": best_prec,
        "Recall_at_Best_F1_alpha": best_rec,
        "AUPRC": auprc,
        **point_res,
        "Affiliation_F1max": np.nan if best_aff < 0 else best_aff,
        "Best_Affiliation_alpha": best_aff_alpha,
        "VUS_PR": vus_res.get("VUS_PR", np.nan),
        "VUS_ROC": vus_res.get("VUS_ROC", np.nan),
        "AUC_PR": vus_res.get("AUC_PR", np.nan),
        "AUC_ROC": vus_res.get("AUC_ROC", np.nan),
    }
    return row, per_alpha_rows
