"""Durable, window-batched assessment and verification in one response chain."""

from __future__ import annotations
import hashlib, json, os, tempfile
from dataclasses import asdict
from pathlib import Path
from .decisions import (
    ASSESSMENT_SCHEMA,
    VERIFICATION_SCHEMA,
    TOOLS,
    ARGUMENT_SCHEMA,
    arbitrate,
    final_intervals,
    interval,
    key,
    overlaps,
    validate_record,
    validate_assessment,
    validate_additions,
    verification_targets,
)
from .prompts import CANDIDATE_ASSESSMENT_PROMPT, AGENTIC_VERIFICATION_PROMPT
from .contracts import (
    CandidateGroup,
    CandidateInterval,
    MediaRef,
    dataclass_to_jsonable,
)
from .contracts import AgentSessionState, GroupRuntimeState
from .tools import ToolRegistry, build_v1_tool_specs
from .windows import plan_assessment_windows, candidate_home_windows, render_profile


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def message(payload, media=()):
    return {
        "role": "user",
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
        + [{"type": "media", **asdict(m)} for m in media],
    }


def public_tool_payload(value):
    """Keep numeric/rendering evidence, excluding local paths and filenames."""
    if isinstance(value, dict):
        return {
            k: public_tool_payload(v)
            for k, v in value.items()
            if k not in {"path", "uri", "root", "dataset", "signal_id", "error"}
        }
    if isinstance(value, list):
        return [public_tool_payload(v) for v in value]
    return value


class Session:
    """Commit each completed response before any tool execution or transition.

    Committed responses and tool results are reused on resume. A crash during
    an uncommitted network request may cost a repeated request, but never silently
    resets previous_response_id or replays all local messages.
    """

    def __init__(self, root, client):
        self.root = Path(root)
        self.client = client
        self.previous = None

    def step(self, name, instructions, input_message, schema):
        request = {
            "instructions": instructions,
            "input_message": input_message,
            "previous_response_id": self.previous,
            "schema": schema,
            "model": self.client.model_name,
        }
        digest = hashlib.sha256(
            json.dumps(request, sort_keys=True).encode()
        ).hexdigest()
        path = self.root / "responses" / f"{name}.json"
        if path.is_file():
            commit = read_json(path)
            if commit["request_hash"] != digest:
                raise ValueError(
                    "checkpoint request changed; use a new output directory"
                )
        else:
            atomic_json(
                path.with_suffix(".request.json"), {"request_hash": digest, **request}
            )
            before_attempts = getattr(self.client, "attempt_count", 0)
            attempt_path = path.with_suffix(".attempts.json")
            attempts = read_json(attempt_path) if attempt_path.is_file() else []

            def observe(event):
                attempts.append(event)
                atomic_json(attempt_path, attempts)

            if hasattr(self.client, "attempt_observer"):
                self.client.attempt_observer = observe
            try:
                result = self.client.step(
                    instructions=instructions,
                    input_message=input_message,
                    previous_response_id=self.previous,
                    response_schema=schema,
                    prompt_cache_key="agentverits-" + name.split("_")[0],
                    context_compact_threshold=None,
                    metadata=None,
                )
            finally:
                if hasattr(self.client, "attempt_observer"):
                    self.client.attempt_observer = None
            if result.status != "completed" or not result.response_id:
                raise ValueError("cannot commit an incomplete response")
            commit = {
                "request_hash": digest,
                "response_id": result.response_id,
                "parsed": result.parsed_action,
                "usage": asdict(result.usage),
                "raw": result.raw_response,
                "request_attempts": max(
                    1, getattr(self.client, "attempt_count", 0) - before_attempts
                ),
            }
            atomic_json(path, commit)
        self.previous = commit["response_id"]
        return commit["parsed"]

    def checked_step(self, name, instructions, input_message, schema, validator):
        # Invalid completed JSON is recorded and corrected in the same chain.
        for correction in range(3):
            payload = self.step(
                name if correction == 0 else f"{name}_repair{correction}",
                instructions,
                input_message,
                schema,
            )
            try:
                validator(payload)
                return payload
            except (ValueError, TypeError, KeyError) as exc:
                if correction == 2:
                    raise ValueError(
                        f"decision protocol remained invalid: {exc}"
                    ) from exc
                input_message = message(
                    {
                        "protocol_error": str(exc),
                        "instruction": "Correct your immediately preceding output, preserving all required candidate IDs and reviewed intervals. Return the entire corrected object.",
                    }
                )

    def totals(self):
        commits = [
            read_json(p)
            for p in (self.root / "responses").glob("*.json")
            if not p.name.endswith((".request.json", ".attempts.json"))
        ]
        attempts = [
            e
            for p in (self.root / "responses").glob("*.attempts.json")
            for e in read_json(p)
        ]
        tools = [read_json(p) for p in (self.root / "tools").glob("*.json")]
        fields = (
            "input_tokens",
            "cached_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
        )
        return {
            "completed_api_calls": len(commits),
            "tool_calls": len(tools),
            "recorded_request_attempts": sum(
                e["status"] == "started" for e in attempts
            ),
            "failed_request_attempts": sum(e["status"] == "failed" for e in attempts),
            "request_attempts_for_completed_calls": sum(
                c.get("request_attempts", 1) for c in commits
            ),
            **{f: sum(int(c["usage"].get(f, 0)) for c in commits) for f in fields},
        }


def owner(pair, windows):
    center = sum(pair) // 2
    return next(
        (w.window_id for w in windows if w.core_start <= center <= w.core_end),
        windows[-1].window_id,
    )


def evidence_support(row, evidence):
    regions = [row["reviewed_interval"]]
    if row["final_interval"]:
        regions.append(row["final_interval"])
    supporting = [
        e
        for e in evidence
        if e["result"]["status"] == "ok"
        and any(overlaps(e["focus_interval"], p) for p in regions)
    ]
    families = {e["result"]["evidence_family"] for e in supporting}
    if row["action"] == "refine" and row["final_interval"]:
        a, b = row["final_interval"]
        covered = all(
            any(
                e["focus_interval"][0] <= bound <= e["focus_interval"][1]
                for e in supporting
            )
            for bound in (a, b)
        )
        if not covered:
            return False
    return len(families) >= 2 and bool(
        families & {"raw_numeric", "statistical", "screening"}
    )


def run_session(
    context, client, adapter, renderer, out_dir, *, max_tool_calls=5, max_turns=8
):
    out_dir = Path(out_dir)
    session = Session(out_dir, client)
    n = context.signal_length
    candidates = {c.candidate_id: c.as_pair() for c in context.candidate_intervals}
    windows = plan_assessment_windows(n, candidates.values())
    owners = candidate_home_windows(candidates.values(), windows)
    assessment = []
    normal_models = []
    window_media = {}
    occupied = list(candidates.values())
    signal = context.adapter_payload
    for w in windows:
        visible = [
            {"candidate_id": cid, "start": p[0], "end": p[1], "owner": owners[cid]}
            for cid, p in candidates.items()
            if overlaps(p, [w.start, w.end])
        ]
        if len(windows) == 1:
            plot = context.full_series_media[0]
        else:
            profile = render_profile(min(w.span, 20000))
            meta = renderer.global_signal_plot(
                signal.values[w.start : w.end + 1],
                visible,
                out_dir / "plots" / f"{w.window_id}.png",
                width_px=profile.width_px,
                height_px=profile.height_px,
                x_start=w.start,
                signal_length=n,
                title=f"Candidate Assessment | {w.window_id}",
                plot_role="assessment_window",
            )
            plot = MediaRef(str(meta["path"]))
        window_media[w.window_id] = plot
        owned = {cid: p for cid, p in candidates.items() if owners[cid] == w.window_id}

        def check_assessment(payload):
            rows = validate_assessment(payload, owned, n)
            additions = [r for r in rows if r["candidate_id"] is None]
            validate_additions(additions, occupied, n)
            if any(
                owner(r["reviewed_interval"], windows) != w.window_id for r in additions
            ):
                raise ValueError("ADD belongs to another window")

        payload = session.checked_step(
            f"assessment_{w.window_id}",
            CANDIDATE_ASSESSMENT_PROMPT,
            message(
                {
                    "stage": "Candidate Assessment",
                    "signal_length": n,
                    "alpha": signal.screening_alpha,
                    "window": w.to_dict(),
                    "owned_candidates": owned,
                    "context_candidates": visible,
                    "previously_assessed_events": assessment,
                    "evidence_sufficient": "Set false at the assessment stage; verification updates this flag.",
                },
                (
                    [*context.full_series_media, plot]
                    if len(windows) > 1 and w == windows[0]
                    else [plot]
                ),
            ),
            ASSESSMENT_SCHEMA,
            check_assessment,
        )
        for r in payload["decisions"]:
            assessment.append(r)
            if r["final_interval"]:
                occupied.append(r["final_interval"])
        normal_models.append(
            {"window_id": w.window_id, "normal_model": payload["normal_model"]}
        )
    # Every screened candidate has one record, even if its final_interval=null.
    atomic_json(
        out_dir / "assessment.json",
        {
            "stage": "Candidate Assessment",
            "response_id": session.previous,
            "decisions": assessment,
            "normal_models": normal_models,
            "verification_target_ids": [
                key(r) for r in verification_targets(assessment)
            ],
        },
    )

    registry = ToolRegistry(adapter)
    for spec in build_v1_tool_specs():
        registry.register(spec)
    finals = {key(r): dict(r) for r in assessment}
    audit = []
    accepted_adds = []
    for w in windows:
        targets = [
            r
            for r in verification_targets(assessment)
            if owner(r["reviewed_interval"], windows) == w.window_id
        ]
        if not targets:
            continue
        target_by_key = {key(r): r for r in targets}
        evidence = []
        finished = False
        input_message = message(
            {
                "stage": "Agentic Verification",
                "window": w.to_dict(),
                "uncertain_decisions": targets,
                "frozen_decisions": [
                    r for r in assessment if r["confidence_level"] == 3
                ],
                "tools": registry.prompt_block(),
                "max_tool_calls": max_tool_calls,
                "max_model_turns": max_turns,
            },
            [window_media[w.window_id]],
        )
        for turn in range(max_turns):
            force = len(evidence) >= max_tool_calls or turn == max_turns - 1
            if force:
                input_message = message(
                    {
                        "observation": input_message["content"][0]["text"],
                        "budget": "No tools remain. Finalize all uncertain records now. Preserve assessment on insufficient evidence.",
                    },
                    [
                        MediaRef(p["uri"])
                        for p in input_message["content"]
                        if p["type"] == "media"
                    ],
                )

            def check_verification(payload):
                if payload["type"] == "call_tool":
                    if force:
                        raise ValueError("tool budget exhausted; finalize")
                    if payload["tool"] not in TOOLS or not isinstance(
                        payload["arguments"], dict
                    ):
                        raise ValueError("invalid tool request")
                    interval(payload["focus_interval"], n)
                    if payload["decisions"] or payload["additions"]:
                        raise ValueError("tool request cannot finalize records")
                    if set(payload["arguments"]) != set(ARGUMENT_SCHEMA["properties"]):
                        raise ValueError("invalid argument keys")
                elif payload["type"] == "finalize":
                    if any(
                        payload[f] is not None
                        for f in ("tool", "arguments", "focus_interval")
                    ):
                        raise ValueError("finalize tool fields must be null")
                    seen = set()
                    for row in payload["decisions"]:
                        validate_record(row, n)
                        k = key(row)
                        if k not in target_by_key or k in seen:
                            raise ValueError("unknown, frozen, or duplicate decision")
                        if row["action"] == "add":
                            raise ValueError(
                                "existing uncertain ADD is confirmed with KEEP/REFINE or rejected with REMOVE"
                            )
                        arbitrate(target_by_key[k], row)
                        seen.add(k)
                    if seen != set(target_by_key):
                        raise ValueError("every uncertain action must be returned")
                    validate_additions(
                        payload["additions"],
                        occupied
                        + [
                            r["final_interval"]
                            for r in payload["decisions"]
                            if r["final_interval"]
                        ],
                        n,
                    )
                    if any(
                        owner(r["reviewed_interval"], windows) != w.window_id
                        for r in payload["additions"]
                    ):
                        raise ValueError("new ADD belongs to another window")
                    validate_additions(
                        payload["additions"],
                        [
                            r["final_interval"]
                            for r in finals.values()
                            if r["final_interval"]
                        ]
                        + [r["final_interval"] for r in accepted_adds],
                        n,
                    )
                else:
                    raise ValueError("unknown verification step")

            payload = session.checked_step(
                f"verification_{w.window_id}_{turn:02d}",
                AGENTIC_VERIFICATION_PROMPT,
                input_message,
                VERIFICATION_SCHEMA,
                check_verification,
            )
            if payload["type"] == "finalize":
                for proposed in payload["decisions"]:
                    original = target_by_key[key(proposed)]
                    guarded = dict(proposed)
                    if not evidence_support(guarded, evidence):
                        guarded["evidence_sufficient"] = False
                    chosen, reason = arbitrate(original, guarded)
                    finals[key(original)] = chosen
                    audit.append(
                        {
                            "decision_id": key(original),
                            "assessment": original,
                            "verification": proposed,
                            "accepted": chosen,
                            "arbitration": reason,
                        }
                    )
                for row in payload["additions"]:
                    accept = (
                        row["evidence_sufficient"]
                        and row["confidence_level"] >= 2
                        and evidence_support(row, evidence)
                    )
                    audit.append(
                        {
                            "decision_id": key(row),
                            "verification": row,
                            "arbitration": (
                                "new_add_accepted"
                                if accept
                                else "new_add_rejected_insufficient_evidence"
                            ),
                        }
                    )
                    if accept:
                        accepted_adds.append(row)
                        occupied.append(row["final_interval"])
                finished = True
                break
            tool_path = out_dir / "tools" / f"{w.window_id}_{turn:02d}.json"
            if tool_path.is_file():
                tool_commit = read_json(tool_path)
            else:
                focus = payload["focus_interval"]
                group = CandidateGroup(
                    f"{w.window_id}_T{turn:02d}", [CandidateInterval(*focus)]
                )
                args = {k: v for k, v in payload["arguments"].items() if v is not None}
                if payload["tool"] in {
                    "inspect_raw_segment",
                    "plot_focused_segment",
                    "segment_statistics",
                    "value_zscore",
                    "diff_zscore",
                }:
                    focus = interval(
                        [args.get("start", focus[0]), args.get("end", focus[1])], n
                    )
                result = registry.dispatch(
                    payload["tool"],
                    args,
                    context=context,
                    session_state=AgentSessionState(context.signal_id),
                    group_state=GroupRuntimeState(group),
                )
                tool_commit = {
                    "response_id": session.previous,
                    "focus_interval": focus,
                    "result": dataclass_to_jsonable(result),
                }
                atomic_json(tool_path, tool_commit)
            if tool_commit["response_id"] != session.previous:
                raise ValueError("tool checkpoint response chain mismatch")
            evidence.append(tool_commit)
            result = tool_commit["result"]
            input_message = message(
                {
                    "tool_result": public_tool_payload(
                        {k: v for k, v in result.items() if k != "media"}
                    ),
                    "remaining_tool_calls": max_tool_calls - len(evidence),
                },
                [MediaRef(**m) for m in result["media"]],
            )
        if not finished:
            raise RuntimeError("verification did not finalize within budget")
    records = list(finals.values()) + accepted_adds
    result = {
        "stage": "Agentic Verification",
        "alpha": signal.screening_alpha,
        "decisions": records,
        "final_intervals": final_intervals(records),
        "arbitration": audit,
        "last_response_id": session.previous,
        "usage": session.totals(),
    }
    atomic_json(out_dir / "result.json", result)
    return result
