#!/usr/bin/env python3
"""Run 252 signals: screening -> three independent assessments -> verification."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Some Windows binary wheels share DLLs; load torch before NumPy/OpenAI.
import torch
from agentverits.workflow import atomic_json, read_json, run_session
from agentverits.evidence import CachedSignalContextProvider, CachedEvidenceToolAdapter
from agentverits.rendering import SignalRenderer
from agentverits.locking import output_lock
from agentverits.providers.openai_official import (
    OfficialResponsesClient,
    OfficialResponsesConfig,
)
from evaluation.experiment import ALPHAS, alpha_name, summarize

DATASETS = {
    "realTweets",
    "realAWSCloudwatch",
    "realAdExchange",
    "realTraffic",
    "artificialWithAnomaly",
    "SMAP",
    "MSL",
    "WSD",
    "IOPS",
}


def load_roster(path):
    roster = read_json(path)["signals"]
    identities = [(r["dataset"], r["signal_id"]) for r in roster]
    if (
        len(roster) != 252
        or len(set(identities)) != 252
        or {d for d, _ in identities} != DATASETS
    ):
        raise ValueError(
            "the full experiment requires exactly 252 unique signals from the nine datasets"
        )
    for dataset, sid in identities:
        if any(Path(v).name != v or v in (".", "..") for v in (dataset, sid)):
            raise ValueError("roster contains an unsafe path component")
    return roster


def ensure_screening(args, roster):
    missing = []
    for r in roster:
        path = args.cache_root / r["dataset"] / r["signal_id"] / "signal_manifest.json"
        if not path.is_file() or not read_json(path).get("cache_complete"):
            missing.append(r)
        else:
            manifest = read_json(path)
            if {float(x["alpha"]) for x in manifest["decision"]["alphas"]} != set(
                ALPHAS
            ):
                raise ValueError(
                    f"screening cache does not have all three alphas: {path}"
                )
            for scale in manifest.get("scales", {}).values():
                if not (path.parent / scale["bundle"]).is_file():
                    raise FileNotFoundError("screening evidence bundle is missing")
            if not (path.parent / manifest["decision"]["bundle"]).is_file():
                raise FileNotFoundError("screening decision bundle is missing")
    if args.dry_run:
        return missing
    if missing:
        for r in missing:
            source = args.data_root / r["dataset"] / (r["signal_id"] + ".csv")
            if not source.is_file():
                raise FileNotFoundError(f"missing source signal: {source}")
        from anomaly_screening.builder import build_cache

        build_cache(
            argparse.Namespace(
                project_root=str(ROOT),
                data_root=str(args.data_root),
                out_dir=str(args.cache_root),
                device=args.device,
                batch_size=args.batch_size,
                max_windows=0,
                save_png=True,
                datasets=sorted({r["dataset"] for r in missing}),
                max_signals=None,
                alphas=ALPHAS,
                resume=True,
                stop_on_error=True,
                signal_roster={(r["dataset"], r["signal_id"]) for r in missing},
            )
        )
        # Verify all 252 manifest commits after screening, not just the root summary.
        for r in missing:
            if not read_json(
                args.cache_root / r["dataset"] / r["signal_id"] / "signal_manifest.json"
            ).get("cache_complete"):
                raise RuntimeError(
                    "Anomaly Screening did not complete the requested roster"
                )
    return missing


def run_one_signal(args, entry):
    dataset, sid = entry["dataset"], entry["signal_id"]
    config = OfficialResponsesConfig(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        stream_responses=not args.no_stream,
        request_timeout_seconds=args.request_timeout,
        request_retries=args.request_retries,
        max_output_tokens=args.max_output_tokens,
    )
    client = OfficialResponsesClient(config=config)
    outcomes = []
    for alpha in ALPHAS:
        out = args.output_root / dataset / sid / alpha_name(alpha)
        status = out / "status.json"
        if (
            status.is_file()
            and read_json(status).get("status") == "ok"
            and (out / "result.json").is_file()
        ):
            outcomes.append("ok")
            continue
        atomic_json(status, {"status": "running", "alpha": alpha})
        try:
            renderer = SignalRenderer()
            provider = CachedSignalContextProvider(
                args.cache_root,
                screening_alpha=alpha,
                renderer=renderer,
                render_root=out / "plots",
            )
            context = provider.build_context((dataset, sid))
            run_session(
                context,
                client,
                CachedEvidenceToolAdapter(renderer=renderer),
                renderer,
                out,
                max_tool_calls=args.max_tool_calls,
                max_turns=args.max_turns,
            )
            atomic_json(status, {"status": "ok", "alpha": alpha})
            outcomes.append("ok")
        except Exception as exc:
            # API keys and client configuration are never serialized.
            atomic_json(
                status,
                {
                    "status": "error",
                    "alpha": alpha,
                    "error_type": type(exc).__name__,
                    "http_status": getattr(exc, "status_code", None),
                    "error": (
                        str(exc)
                        if isinstance(
                            exc, (ValueError, RuntimeError, FileNotFoundError)
                        )
                        else "Request failed; inspect recorded HTTP status and attempt logs. Completed responses are retained for resume."
                    ),
                },
            )
            print(
                f"[ERROR] {dataset}/{sid} alpha={alpha:g}: {type(exc).__name__}",
                flush=True,
            )
            if getattr(exc, "status_code", None) in {400, 401, 403, 404}:
                raise RuntimeError(
                    "Official API rejected the model, credentials or request configuration; check status.json before resuming"
                ) from exc
            outcomes.append("error")
    print(
        f"[{ 'OK' if all(s == 'ok' for s in outcomes) else 'PARTIAL' }] {dataset}/{sid}: {outcomes}",
        flush=True,
    )
    return outcomes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roster", type=Path, default=ROOT / "configs" / "signals_252.json"
    )
    # Standalone publication defaults; this local source snapshot can reuse
    # its parent workspace data/cache without copying private experiment files.
    data_default = ROOT / "data"
    if not data_default.is_dir() and (ROOT.parent / "data").is_dir():
        data_default = ROOT.parent / "data"
    cache_default = ROOT / "results" / "anomaly_screening"
    parent_caches = sorted(ROOT.parent.glob("results*_full_cache_9datasets"))
    if not cache_default.is_dir() and len(parent_caches) == 1:
        cache_default = parent_caches[0]
    parser.add_argument("--data-root", type=Path, default=data_default)
    parser.add_argument("--cache-root", type=Path, default=cache_default)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "results" / "official_experiment"
    )
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument(
        "--reasoning-effort", default="high", choices=["low", "medium", "high", "xhigh"]
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel signals; one sequential API chain per alpha",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-tool-calls", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=240)
    parser.add_argument("--request-retries", type=int, default=6)
    parser.add_argument("--max-output-tokens", type=int, default=24000)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate roster/cache without model/API requests",
    )
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.workers < 1
        or args.max_turns < 1
        or args.max_tool_calls < 0
        or args.max_output_tokens < 1
    ):
        parser.error("invalid worker/tool/turn budget")
    return args


def main(argv=None):
    args = parse_args(argv)
    roster = load_roster(args.roster)
    if args.report_only:
        summary = summarize(roster, args.cache_root, args.output_root)
        print(json.dumps(summary["datasets"], indent=2))
        return 0 if summary["complete"] else 2
    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "Set OPENAI_API_KEY for your official OpenAI account before running. No relay credential file is loaded."
        )
    if args.dry_run:
        missing = ensure_screening(args, roster)
        print(
            json.dumps(
                {
                    "signals": 252,
                    "alpha_sessions": 756,
                    "datasets": sorted(DATASETS),
                    "cached_signals": 252 - len(missing),
                    "screening_signals_to_build": len(missing),
                    "api_endpoint": "https://api.openai.com/v1",
                    "model": args.model,
                    "reasoning_effort": args.reasoning_effort,
                    "stream": not args.no_stream,
                },
                indent=2,
            )
        )
        return 0
    with output_lock(args.output_root):
        return run_locked(args, roster)


def run_locked(args, roster):
    ensure_screening(args, roster)
    # Prevent accidental mixing of different prompts/models/configurations.
    from agentverits.prompts import (
        CANDIDATE_ASSESSMENT_PROMPT,
        AGENTIC_VERIFICATION_PROMPT,
    )

    identity = {
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "alphas": list(ALPHAS),
        "max_tool_calls": args.max_tool_calls,
        "max_turns": args.max_turns,
        "max_output_tokens": args.max_output_tokens,
        "roster": roster,
        "prompt_hash": hashlib.sha256(
            (CANDIDATE_ASSESSMENT_PROMPT + AGENTIC_VERIFICATION_PROMPT).encode()
        ).hexdigest(),
    }
    identity["source_hash"] = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted((ROOT / "src").rglob("*.py")))
    ).hexdigest()
    identity["screening_decision_hash"] = hashlib.sha256(
        b"".join(
            json.dumps(
                read_json(
                    args.cache_root
                    / r["dataset"]
                    / r["signal_id"]
                    / "signal_manifest.json"
                )["decision"],
                sort_keys=True,
            ).encode()
            for r in roster
        )
    ).hexdigest()
    path = args.output_root / "experiment.json"
    if path.is_file() and read_json(path) != identity:
        raise ValueError("output directory belongs to a different experiment")
    atomic_json(path, identity)
    if args.workers == 1:
        for entry in roster:
            run_one_signal(args, entry)
    else:
        # Separate processes keep Matplotlib state and lazy evidence arrays isolated.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one_signal, args, r) for r in roster]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                for pending in futures:
                    pending.cancel()
                raise
    summary = summarize(roster, args.cache_root, args.output_root)
    print(json.dumps(summary["datasets"], indent=2))
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        detail = (
            str(exc)
            if isinstance(exc, (ValueError, RuntimeError, FileNotFoundError))
            else "Inspect local status and transport attempt logs."
        )
        print(f"[FAILED] {type(exc).__name__}: {detail}", file=sys.stderr)
        raise SystemExit(2)
