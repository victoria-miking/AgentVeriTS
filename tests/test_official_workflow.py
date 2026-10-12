"""Offline checks for routing, server continuity, ADD and durable evidence."""

import copy, json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from agentverits.decisions import (
    ARGUMENT_SCHEMA,
    ASSESSMENT_SCHEMA,
    validate_assessment,
    validate_record,
    verification_targets,
    final_intervals,
)
from agentverits.evidence import CachedSignalContextProvider, CachedEvidenceToolAdapter
from agentverits.providers.openai_official import (
    OfficialResponsesClient,
    OfficialResponsesConfig,
)
from agentverits.rendering import SignalRenderer
from agentverits.contracts import ModelStepResult, ToolResult
from agentverits.workflow import (
    Session,
    atomic_json,
    message,
    read_json,
    run_session,
    public_tool_payload,
)
from evaluation.experiment import alpha_name, summarize
from run_experiment import load_roster


def record(cid, region, action="keep", grade=3, q=0.98, final="same", sufficient=False):
    return {
        "candidate_id": cid,
        "reviewed_interval": region,
        "action": action,
        "final_interval": (
            (None if action == "remove" else region) if final == "same" else final
        ),
        "confidence_level": grade,
        "confidence": q,
        "rationale": "contextual evidence",
        "evidence_sufficient": sufficient,
    }


def tool(name, region, **args):
    arguments = {k: None for k in ARGUMENT_SCHEMA["properties"]}
    arguments.update(args)
    return {
        "type": "call_tool",
        "focus_interval": region,
        "tool": name,
        "arguments": arguments,
        "decisions": [],
        "additions": [],
        "rationale": "resolve uncertainty",
    }


def final(decisions, additions=None):
    return {
        "type": "finalize",
        "focus_interval": None,
        "tool": None,
        "arguments": None,
        "decisions": decisions,
        "additions": additions or [],
        "rationale": "complete",
    }


class QueueModel:
    model_name = "gpt-5.6-sol"

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def step(self, **kwargs):
        self.calls.append(kwargs)
        return ModelStepResult(
            f"resp_{len(self.calls)}",
            self.payloads.pop(0),
            raw_response={"status": "completed"},
        )


def cache(tmp_path, pairs, *, length=200):
    signal = tmp_path / "cache" / "MSL" / "synthetic"
    signal.mkdir(parents=True)
    np.savez_compressed(
        signal / "signal_artifacts.npz",
        values=np.arange(length, dtype=np.float32),
        source_timestamps=np.arange(length),
        labels=np.ones(length, dtype=np.int8),
    )
    atomic_json(
        signal / "signal_manifest.json",
        {
            "cache_schema": "ANOMALY_SCREENING_CACHE_V1",
            "cache_complete": True,
            "dataset": "MSL",
            "signal_id": "synthetic",
            "scales": {},
            "gt_intervals": [[10, 20]],
            "decision": {
                "bundle": "signal_artifacts.npz",
                "selected_alpha": 0.001,
                "selected_intervals": [[190, 199]],
                "alphas": [
                    {"alpha": a, "threshold": 0.0, "intervals": pairs}
                    for a in (0.1, 0.01, 0.001)
                ],
            },
        },
    )
    return signal.parents[1]


def context(tmp_path, pairs, alpha=0.1, length=200):
    root = cache(tmp_path, pairs, length=length)
    renderer = SignalRenderer()
    ctx = CachedSignalContextProvider(
        root, screening_alpha=alpha, renderer=renderer, render_root=tmp_path / "plots"
    ).build_context(("MSL", "synthetic"))
    return root, renderer, ctx


def test_all_actions_coverage_removal_and_grade_only_routing():
    rows = [
        record("C0001", [1, 5]),
        record("C0002", [10, 20], "remove", 2, 0.85),
        record("C0003", [30, 40], "refine", 1, 0.6, final=[29, 43]),
        record(None, [60, 65], "add", 3, 0.97),
    ]
    validate_assessment(
        {"normal_model": "periodic", "decisions": rows},
        {"C0001": [1, 5], "C0002": [10, 20], "C0003": [30, 40]},
        100,
    )
    assert [r["candidate_id"] for r in verification_targets(rows)] == ["C0002", "C0003"]
    assert (
        rows[1]["reviewed_interval"] == [10, 20] and rows[1]["final_interval"] is None
    )
    assert final_intervals(rows) == [[1, 5], [29, 43], [60, 65]]
    assert verification_targets([record("C0001", [1, 5], "remove")]) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["decisions"].clear(),
        lambda p: p["decisions"].append(copy.deepcopy(p["decisions"][0])),
        lambda p: p["decisions"][0].update(candidate_id="unknown"),
        lambda p: p["decisions"][0].update(confidence=0.9),
        lambda p: p["decisions"][0].update(confidence=float("nan")),
        lambda p: p["decisions"][0].update(action="remove", final_interval=[1, 5]),
        lambda p: p["decisions"].append(record(None, [3, 6], "add")),
    ],
)
def test_invalid_or_missing_decisions_are_rejected(mutate):
    payload = {"normal_model": "stable", "decisions": [record("C0001", [1, 5])]}
    mutate(payload)
    with pytest.raises(ValueError):
        validate_assessment(payload, {"C0001": [1, 5]}, 100)


def test_low_remove_verified_after_assessment_in_same_chain_and_resumed(tmp_path):
    root, renderer, ctx = context(tmp_path, [[10, 20], [40, 50]])
    assessed = [record("C0001", [10, 20]), record("C0002", [40, 50], "remove", 2, 0.85)]
    model = QueueModel(
        [
            {"normal_model": "trend", "decisions": assessed},
            tool("plot_focused_segment", [40, 50]),
            tool("segment_statistics", [40, 50]),
            final([record("C0002", [40, 50], "keep", 2, 0.90, sufficient=True)]),
        ]
    )
    out = tmp_path / "out"
    adapter = CachedEvidenceToolAdapter(renderer=renderer)
    result = run_session(ctx, model, adapter, renderer, out)
    assert result["final_intervals"] == [[10, 20], [40, 50]]
    assert result["decisions"][0] == assessed[0]
    assert [c["previous_response_id"] for c in model.calls] == [
        None,
        "resp_1",
        "resp_2",
        "resp_3",
    ]
    assert result["usage"]["tool_calls"] == 2
    # Resume reconstructs local control state by reading commits, not sending history.
    retry_model = QueueModel([])
    again = run_session(ctx, retry_model, adapter, renderer, out)
    assert again == result and retry_model.calls == []
    sent = json.dumps(model.calls)
    assert "gt_intervals" not in sent and "selected_alpha" not in sent
    assert "label" not in json.loads(
        model.calls[0]["input_message"]["content"][0]["text"]
    )
    assert not (root / "MSL/synthetic/agent_evidence").exists()


def test_insufficient_evidence_preserves_uncertain_assessment(tmp_path):
    _, renderer, ctx = context(tmp_path, [[10, 20]])
    original = record("C0001", [10, 20], "remove", 2, 0.8)
    model = QueueModel(
        [
            {"normal_model": "normal", "decisions": [original]},
            final([record("C0001", [10, 20], "keep", 3, 0.97, sufficient=True)]),
        ]
    )
    result = run_session(
        ctx,
        model,
        CachedEvidenceToolAdapter(renderer=renderer),
        renderer,
        tmp_path / "out",
    )
    assert result["decisions"] == [original]
    assert result["final_intervals"] == []
    assert (
        result["arbitration"][0]["arbitration"]
        == "assessment_preserved_insufficient_evidence"
    )


def test_assessed_add_is_verified_and_its_remove_keeps_reviewed_region(tmp_path):
    _, renderer, ctx = context(tmp_path, [])
    added = record(None, [60, 70], "add", 1, 0.65)
    model = QueueModel(
        [
            {"normal_model": "normal", "decisions": [added]},
            tool("plot_focused_segment", [60, 70]),
            tool("segment_statistics", [60, 70]),
            final([record(None, [60, 70], "remove", 3, 0.97, sufficient=True)]),
        ]
    )
    result = run_session(
        ctx,
        model,
        CachedEvidenceToolAdapter(renderer=renderer),
        renderer,
        tmp_path / "out",
    )
    assert result["decisions"][0]["action"] == "remove"
    assert result["decisions"][0]["reviewed_interval"] == [60, 70]
    assert result["final_intervals"] == []


def test_verification_can_discover_a_global_missed_event(tmp_path):
    _, renderer, ctx = context(tmp_path, [[10, 20]])
    original = record("C0001", [10, 20], "keep", 2, 0.9)
    added = record(None, [130, 140], "add", 2, 0.9, sufficient=True)
    model = QueueModel(
        [
            {"normal_model": "trend", "decisions": [original]},
            tool("plot_focused_segment", [130, 140]),
            tool("segment_statistics", [130, 140]),
            final([original], [added]),
        ]
    )
    result = run_session(
        ctx,
        model,
        CachedEvidenceToolAdapter(renderer=renderer),
        renderer,
        tmp_path / "out",
    )
    assert result["final_intervals"] == [[10, 20], [130, 140]]


def test_three_alpha_chains_are_independent_and_ignore_selected_alpha(tmp_path):
    root = cache(tmp_path, [[10, 20]])
    model = QueueModel(
        [
            {"normal_model": "normal", "decisions": [record("C0001", [10, 20])]}
            for _ in range(3)
        ]
    )
    for alpha in (0.1, 0.01, 0.001):
        renderer = SignalRenderer()
        ctx = CachedSignalContextProvider(
            root,
            screening_alpha=alpha,
            renderer=renderer,
            render_root=tmp_path / alpha_name(alpha),
        ).build_context(("MSL", "synthetic"))
        assert ctx.candidate_intervals[0].as_pair() == [10, 20]
        run_session(
            ctx,
            model,
            CachedEvidenceToolAdapter(renderer=renderer),
            renderer,
            tmp_path / "out" / alpha_name(alpha),
        )
    assert [c["previous_response_id"] for c in model.calls] == [None, None, None]


def test_long_signal_all_candidates_are_owned_once(tmp_path):
    _, renderer, ctx = context(tmp_path, [[100, 200], [22000, 22100]], length=25000)
    model = QueueModel(
        [
            {"normal_model": "normal", "decisions": [record("C0001", [100, 200])]},
            {"normal_model": "normal", "decisions": [record("C0002", [22000, 22100])]},
        ]
    )
    result = run_session(
        ctx,
        model,
        CachedEvidenceToolAdapter(renderer=renderer),
        renderer,
        tmp_path / "out",
    )
    assert result["final_intervals"] == [[100, 200], [22000, 22100]]
    assert model.calls[1]["previous_response_id"] == "resp_1"


def test_original_f1max_is_per_signal_max_before_dataset_macro(tmp_path):
    roster = [
        {"dataset": "MSL", "signal_id": "a"},
        {"dataset": "MSL", "signal_id": "b"},
    ]
    patterns = {"a": [[[10, 20]], [], []], "b": [[], [[10, 20]], []]}
    for row in roster:
        sid = row["signal_id"]
        atomic_json(
            tmp_path / "cache/MSL" / sid / "signal_manifest.json",
            {
                "gt_intervals": [[10, 20]],
                "decision": {
                    "alphas": [
                        {"alpha": a, "intervals": p}
                        for a, p in zip((0.1, 0.01, 0.001), patterns[sid])
                    ]
                },
            },
        )
        for a, p in zip((0.1, 0.01, 0.001), patterns[sid]):
            path = tmp_path / "out/MSL" / sid / alpha_name(a)
            atomic_json(path / "result.json", {"final_intervals": p, "usage": {}})
            atomic_json(path / "status.json", {"status": "ok"})
            atomic_json(
                path / "assessment.json",
                {"decisions": [record("C0001", p[0])] if p else []},
            )
    summary = summarize(roster, tmp_path / "cache", tmp_path / "out")
    assert (
        summary["complete"]
        and summary["datasets"][0]["Agentic_Verification_F1max"] == 1
    )
    # Two alpha successes are not a completed three-alpha signal.
    atomic_json(tmp_path / "out/MSL/b/alpha_0.001/status.json", {"status": "error"})
    partial = summarize(roster, tmp_path / "cache", tmp_path / "out")
    assert not partial["complete"] and partial["completed_signals"] == 1


def test_official_endpoint_pinned_and_server_chain(monkeypatch):
    calls = []
    response_calls = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.responses = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            response_calls.append(kwargs)
            return {
                "id": "resp_ok",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": '{"x":1}'}],
                    }
                ],
            }

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://relay.invalid")
    client = OfficialResponsesClient(
        config=OfficialResponsesConfig(api_key="test-only", stream_responses=False)
    )
    result = client.step(
        instructions="resent every turn",
        input_message=message({"new": "observation"}),
        previous_response_id="resp_previous",
        response_schema=ASSESSMENT_SCHEMA,
        metadata={"stage": "verification"},
    )
    assert (
        calls[0]["base_url"] == "https://api.openai.com/v1"
        and calls[0]["max_retries"] == 0
    )
    request = response_calls[0]
    assert (
        request["previous_response_id"] == "resp_previous" and request["store"] is True
    )
    assert len(request["input"]) == 1 and request["instructions"] == "resent every turn"
    assert "default_headers" not in calls[0] and "prompt_cache_options" not in request
    assert request["metadata"] == {
        "stage": "verification"
    } and result.parsed_action == {"x": 1}


def test_interrupted_sse_does_not_commit_partial_and_closes_stream(monkeypatch):
    responses = iter(
        [
            [{"type": "response.output_text.delta", "delta": "{broken"}],
            [
                {"type": "response.output_text.delta", "delta": '{"ok":true}'},
                {
                    "type": "response.completed",
                    "response": {"id": "resp_complete", "status": "completed"},
                },
            ],
        ]
    )
    closed = []

    class Stream:
        def __init__(self, events):
            self.events = events

        def __iter__(self):
            return iter(self.events)

        def close(self):
            closed.append(True)

    fake = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **kw: Stream(next(responses)))
    )
    monkeypatch.setattr(
        "agentverits.providers.openai_official.time.sleep", lambda seconds: None
    )
    client = OfficialResponsesClient(
        config=OfficialResponsesConfig(request_retries=1), client=fake
    )
    result = client.step(
        instructions="test",
        input_message=message({}),
        previous_response_id=None,
        response_schema={},
    )
    assert result.response_id == "resp_complete" and result.parsed_action == {
        "ok": True
    }
    assert client.attempt_count == 2 and len(closed) == 2


@pytest.mark.parametrize("status", ["failed", "incomplete", "in_progress", None])
def test_noncompleted_api_result_never_committed(tmp_path, status):
    fake = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kw: {
                "id": "resp_bad",
                "status": status,
                "output_text": "{}",
            }
        )
    )
    client = OfficialResponsesClient(
        config=OfficialResponsesConfig(stream_responses=False, request_retries=0),
        client=fake,
    )
    with pytest.raises(RuntimeError):
        Session(tmp_path, client).step("test", "instructions", message({}), {})
    assert not (tmp_path / "responses/test.json").exists()


def test_refine_evidence_must_cover_both_new_boundaries():
    from agentverits.workflow import evidence_support

    proposal = record(
        "C1", [100, 200], "refine", 2, 0.85, final=[80, 220], sufficient=True
    )
    short = [
        {"focus_interval": [100, 200], "result": {"status": "ok", "evidence_family": f}}
        for f in ("visual", "statistical")
    ]
    assert not evidence_support(proposal, short)
    wide = [
        {"focus_interval": [50, 250], "result": {"status": "ok", "evidence_family": f}}
        for f in ("visual", "statistical")
    ]
    assert evidence_support(proposal, wide)


def test_roster_and_public_evidence_do_not_contain_private_paths():
    root = Path(__file__).resolve().parents[1]
    roster = load_roster(root / "configs/signals_252.json")
    assert len(roster) == 252 and len(roster) * 3 == 756
    assert public_tool_payload(
        {"renderer": {"path": "private/file", "width_px": 6000}, "error": "private"}
    ) == {"renderer": {"width_px": 6000}}


def test_window_batches_multiple_uncertain_actions_in_one_verification(tmp_path):
    _, renderer, ctx = context(tmp_path, [[10, 20], [40, 50]])
    rows = [
        record("C0001", [10, 20], grade=1, q=0.65),
        record("C0002", [40, 50], "remove", 2, 0.90),
    ]
    model = QueueModel([{"normal_model": "trend", "decisions": rows}, final(rows)])
    run_session(
        ctx,
        model,
        CachedEvidenceToolAdapter(renderer=renderer),
        renderer,
        tmp_path / "out",
    )
    assert len(model.calls) == 2
    targets = json.loads(model.calls[1]["input_message"]["content"][0]["text"])[
        "uncertain_decisions"
    ]
    assert len(targets) == 2


def test_output_lock_blocks_an_existing_process_and_recovers_stale_lock(
    tmp_path, monkeypatch
):
    from agentverits.locking import output_lock

    with output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="owned by running PID"):
            with output_lock(tmp_path):
                pass
    assert not (tmp_path / ".experiment.lock").exists()
    atomic_json(tmp_path / ".experiment.lock", {"pid": 9999999})
    monkeypatch.setattr("agentverits.locking.process_alive", lambda pid: False)
    with output_lock(tmp_path):
        assert (tmp_path / ".experiment.lock").is_file()
    assert not (tmp_path / ".experiment.lock").exists()


def test_fresh_screening_build_executes_render_scoring_and_all_alphas(tmp_path):
    import torch
    import pandas as pd
    from anomaly_screening.builder import _build_signal
    from anomaly_screening.audit import load_frozen_config
    from models.clip_vision import multi_scale_aggregation
    from models.model_utils import patch_scale
    from models.decision import compute_detection_intervals
    from preprocessing.data_utils import intervals_from_indices
    from evaluation.evaluate import evaluate_intervals
    from anomaly_screening.cache import load_signal_cache

    class OfflineEncoder:
        def encode_image(self, images, patch_size):
            # Deterministic stand-in embeddings; rendering/scoring/fusion are real.
            pool = (
                torch.nn.functional.adaptive_avg_pool2d(images, (14, 14))
                .mean(dim=1)
                .reshape(len(images), 196, 1)
            )
            weights = torch.linspace(0.1, 1.0, 8).reshape(1, 1, 8)
            tokens = pool * weights + torch.arange(196).reshape(1, 196, 1) / 196.0
            large, mid, lm, mm = multi_scale_aggregation(
                tokens, patch_size, patch_scale((224, 224))
            )
            return (
                large,
                mid,
                tokens.unsqueeze(2),
                tokens.mean(dim=1, keepdim=True),
                lm,
                mm,
            )

    values = np.sin(np.arange(850) * 0.04)
    values[400:430] += 3
    frame = pd.DataFrame(
        {
            "timestamp": np.arange(850),
            "value": values,
            "label": ((np.arange(850) >= 400) & (np.arange(850) < 430)).astype(int),
        }
    )
    repo = {
        "compute_detection_intervals": compute_detection_intervals,
        "intervals_from_indices": intervals_from_indices,
        "evaluate_intervals": evaluate_intervals,
        "evaluate_affiliation_f1": None,
        "evaluate_vus_pr": None,
    }
    root = Path(__file__).resolve().parents[1]
    manifest = _build_signal(
        model=OfflineEncoder(),
        df=frame,
        dataset="MSL",
        signal_id="offline",
        out_dir=tmp_path,
        project_root=tmp_path,
        device=torch.device("cpu"),
        batch_size=20,
        max_windows=0,
        save_png=False,
        alphas=(0.1, 0.01, 0.001),
        frozen_config=load_frozen_config(root / "configs/anomaly_screening.json"),
        repo=repo,
    )
    assert manifest["cache_complete"] and manifest["gt_intervals"] == [[400, 429]]
    result = load_signal_cache(tmp_path, "MSL", "offline")
    assert set(result["scales"]) == {224, 448, 672}
    assert result["decision"]["fused_score"].shape == (850,)
    assert [r["alpha"] for r in manifest["decision"]["alphas"]] == [0.1, 0.01, 0.001]
    from agentverits.contracts import (
        CandidateGroup,
        CandidateInterval,
        GroupRuntimeState,
        AgentSessionState,
    )
    from agentverits.decisions import TOOLS

    renderer = SignalRenderer()
    context = CachedSignalContextProvider(
        tmp_path,
        screening_alpha=0.1,
        renderer=renderer,
        render_root=tmp_path / "previews",
    ).build_context(("MSL", "offline"))
    assert all(payload is None for payload in context.adapter_payload.scales.values())
    adapter = CachedEvidenceToolAdapter(renderer=renderer)
    scope = {
        "context": context,
        "session_state": AgentSessionState("MSL/offline"),
        "group_state": GroupRuntimeState(
            CandidateGroup("G1", [CandidateInterval(400, 429)])
        ),
    }
    refs = adapter.execute("get_references", {"scale": 224}, **scope)
    assert refs.status == "ok" and refs.data["references"]
    assert all(r["normality"] == "unknown" for r in refs.data["references"])
    reference_ids = [r["reference_id"] for r in refs.data["references"]]
    for name in TOOLS[1:]:
        args = {"start": 400, "end": 429, "reference_ids": reference_ids}
        result = adapter.execute(name, args, **scope)
        assert result.status == "ok", (name, result.error)
        assert "gt_intervals" not in json.dumps(asdict(result))
    assert not (tmp_path / "MSL/offline/evidence").exists()
