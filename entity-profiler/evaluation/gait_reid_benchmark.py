"""Skeleton for a gait re-identification benchmark.

This script is not fully implemented yet. It documents the intended interface
for evaluating gait-based re-identification on a gait corpus such as OU-MVLP.

Planned responsibilities:
- Load a JSONL index of sequences (subject_id, view_id, frames_dir).
- Extract deterministic FusedFeatures for each sequence using the existing
  gait/soft-biometrics pipeline.
- Optionally apply a learned embedding from models/gait_embedder.onnx.
- Compute standard re-id metrics (Rank-k, mAP) across views.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - stub
    parser = argparse.ArgumentParser(description="Gait re-identification benchmark (stub)")
    parser.add_argument("--index", required=True, help="Path to gait sequence index JSONL")
    parser.add_argument("--output", required=True, help="Path to JSON results file")
    parser.parse_args(argv)
    raise SystemExit(
        "Gait re-identification benchmark logic is not yet implemented. "
        "See docs/evaluation_and_benchmarks.md for the planned evaluation.")


if __name__ == "__main__":  # pragma: no cover
    main()
