"""Strict all-action protocol and confidence-level-only verification routing."""

from __future__ import annotations
import math
from typing import Any


def object_schema(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def nullable(schema):
    return {"anyOf": [schema, {"type": "null"}]}


INTERVAL_SCHEMA = {
    "type": "array",
    "items": {"type": "integer"},
    "minItems": 2,
    "maxItems": 2,
}
DECISION_SCHEMA = object_schema(
    {
        "candidate_id": nullable({"type": "string"}),
        "reviewed_interval": INTERVAL_SCHEMA,
        "action": {"type": "string", "enum": ["keep", "remove", "refine", "add"]},
        "final_interval": {"anyOf": [INTERVAL_SCHEMA, {"type": "null"}]},
        "confidence_level": {"type": "integer", "enum": [1, 2, 3]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
        "evidence_sufficient": {"type": "boolean"},
    }
)
ASSESSMENT_SCHEMA = object_schema(
    {
        "normal_model": {"type": "string"},
        "decisions": {"type": "array", "items": DECISION_SCHEMA},
    }
)
ARGUMENT_SCHEMA = object_schema(
    {
        **{
            k: nullable({"type": "integer"})
            for k in ("start", "end", "scale", "limit", "offset")
        },
        **{
            k: nullable({"type": "string"})
            for k in ("context", "target_type", "target_id")
        },
        "reference_ids": {
            "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]
        },
    }
)
TOOLS = [
    "get_references",
    "compare_references_visually",
    "inspect_raw_segment",
    "plot_focused_segment",
    "segment_statistics",
    "value_zscore",
    "diff_zscore",
    "get_screening_evidence",
]
VERIFICATION_SCHEMA = object_schema(
    {
        "type": {"type": "string", "enum": ["call_tool", "finalize"]},
        "focus_interval": {"anyOf": [INTERVAL_SCHEMA, {"type": "null"}]},
        "tool": nullable({"type": "string", "enum": TOOLS}),
        "arguments": {"anyOf": [ARGUMENT_SCHEMA, {"type": "null"}]},
        "decisions": {"type": "array", "items": DECISION_SCHEMA},
        "additions": {"type": "array", "items": DECISION_SCHEMA},
        "rationale": {"type": "string"},
    }
)


def interval(pair, n):
    if (
        not isinstance(pair, list)
        or len(pair) != 2
        or any(type(x) is not int for x in pair)
    ):
        raise ValueError("interval must contain exactly two integers")
    if not 0 <= pair[0] <= pair[1] < n:
        raise ValueError("interval lies outside the signal")
    return pair


def validate_record(row, n):
    if not isinstance(row, dict) or set(row) != set(DECISION_SCHEMA["properties"]):
        raise ValueError("decision fields do not match the protocol")
    interval(row["reviewed_interval"], n)
    action = row["action"]
    if action not in {"keep", "remove", "refine", "add"}:
        raise ValueError("invalid action")
    if action == "remove":
        if row["final_interval"] is not None:
            raise ValueError("REMOVE final_interval must be null")
    else:
        interval(row["final_interval"], n)
        if (
            action in {"keep", "add"}
            and row["reviewed_interval"] != row["final_interval"]
        ):
            raise ValueError("KEEP/ADD must preserve reviewed boundaries")
    grade = row["confidence_level"]
    q = row["confidence"]
    if type(grade) is not int or grade not in (1, 2, 3):
        raise ValueError("invalid confidence level")
    if type(q) not in (float, int) or not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("confidence must be finite and in [0,1]")
    if not (
        grade == 1
        and q <= 0.70
        or grade == 2
        and 0.70 < q <= 0.95
        or grade == 3
        and q > 0.95
    ):
        raise ValueError("confidence and confidence_level are inconsistent")
    if type(row["evidence_sufficient"]) is not bool or not isinstance(
        row["rationale"], str
    ):
        raise ValueError("invalid decision explanation/evidence flag")
    if row["candidate_id"] is not None and not isinstance(row["candidate_id"], str):
        raise ValueError("invalid candidate id")
    return row


def overlaps(a, b):
    return max(a[0], b[0]) <= min(a[1], b[1])


def key(row):
    return (
        row["candidate_id"]
        if row["candidate_id"] is not None
        else "ADD:" + ":".join(map(str, row["reviewed_interval"]))
    )


def validate_assessment(payload, candidates, n):
    if set(payload) != {"normal_model", "decisions"} or not isinstance(
        payload["normal_model"], str
    ):
        raise ValueError("invalid assessment object")
    seen = set()
    adds = []
    rows = payload["decisions"]
    for row in rows:
        validate_record(row, n)
        cid = row["candidate_id"]
        if cid is None:
            if row["action"] != "add":
                raise ValueError("new assessed events must use ADD")
            adds.append(row)
        else:
            if (
                cid not in candidates
                or cid in seen
                or row["reviewed_interval"] != candidates[cid]
            ):
                raise ValueError(
                    "candidate missing, duplicated, unknown, or reviewed boundaries changed"
                )
            if row["action"] == "add":
                raise ValueError("original candidate cannot use ADD")
            seen.add(cid)
    if seen != set(candidates):
        raise ValueError("not every candidate was assessed")
    validate_additions(
        adds,
        list(candidates.values())
        + [
            r["final_interval"]
            for r in rows
            if r["candidate_id"] is not None and r["final_interval"]
        ],
        n,
    )
    return rows


def validate_additions(additions, occupied, n):
    occupied = list(occupied)
    for row in additions:
        validate_record(row, n)
        if row["candidate_id"] is not None or row["action"] != "add":
            raise ValueError("invalid ADD record")
        region = row["final_interval"]
        if any(overlaps(region, old) for old in occupied):
            raise ValueError("ADD duplicates an existing event")
        occupied.append(region)


def verification_targets(rows):
    return [r for r in rows if r["confidence_level"] in (1, 2)]


def arbitrate(original, proposed):
    if (
        key(original) != key(proposed)
        or original["reviewed_interval"] != proposed["reviewed_interval"]
    ):
        raise ValueError("verification cannot change record identity")
    changed = (original["action"], original["final_interval"]) != (
        proposed["action"],
        proposed["final_interval"],
    )
    if changed and (
        not proposed["evidence_sufficient"] or proposed["confidence_level"] == 1
    ):
        return dict(original), "assessment_preserved_insufficient_evidence"
    return dict(proposed), "verification_accepted"


def final_intervals(rows):
    merged = []
    for a, b in sorted(r["final_interval"] for r in rows if r["action"] != "remove"):
        if not merged or a > merged[-1][1] + 1:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    return merged
