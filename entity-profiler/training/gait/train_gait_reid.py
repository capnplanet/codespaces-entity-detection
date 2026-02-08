"""Stub script for training a gait / re-identification embedding.

This will eventually:
- Consume fused features extracted from a gait corpus (e.g., OU-MVLP) via
  the existing deterministic feature pipeline.
- Train a small metric-learning head to map `FusedFeatures.as_array()` to a
  lower-dimensional embedding for re-identification.
- Export the trained head to `models/gait_embedder.onnx`.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - training stub
    parser = argparse.ArgumentParser(description="Train gait re-id embedding (stub)")
    parser.add_argument("--config", help="Path to training config JSON/YAML", required=False)
    parser.parse_args(argv)
    raise SystemExit(
        "Gait re-identification training is not yet implemented. "
        "Adapters and evaluation hooks are in place; see docs/evaluation_and_benchmarks.md."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
