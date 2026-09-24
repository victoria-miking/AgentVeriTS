from __future__ import annotations

import argparse
from pathlib import Path

from .config import AgentVeriTSConfig
from .io import load_signal_csv
from .pipeline import AgentVeriTSPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the complete AgentVeriTS inference pipeline.")
    parser.add_argument("--input", required=True, help="CSV containing timestamp/value or a numeric signal column")
    parser.add_argument("--signal-id", default=None)
    parser.add_argument("--config", default=None, help="YAML config; defaults to package settings")
    parser.add_argument("--output-dir", default="outputs/agentverits")
    parser.add_argument("--alpha", type=float, default=None, help="override visual-screening alpha without labels")
    parser.add_argument("--model", default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None, help="verification threshold (paper: 0.95)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = AgentVeriTSConfig.from_yaml(args.config) if args.config else AgentVeriTSConfig()
    if args.alpha is not None:
        config.screening.alpha = float(args.alpha)
    if args.model is not None:
        config.reasoning.model = str(args.model)
    if args.confidence_threshold is not None:
        config.reasoning.confidence_threshold = args.confidence_threshold
    config.validate()
    values, _ = load_signal_csv(args.input)
    signal_id = args.signal_id or Path(args.input).stem
    result = AgentVeriTSPipeline(config).run(values, signal_id=signal_id, output_dir=args.output_dir)
    print(Path(result.output_dir) / "result.json")


if __name__ == "__main__":
    main()
