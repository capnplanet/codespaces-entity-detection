"""Stub script for training a pose estimator.

Intended to train or fine-tune a single-person COCO-17 keypoint model and
export it to `models/pose_estimator.onnx` for use by the pose pipeline.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - training stub
    parser = argparse.ArgumentParser(description="Train a pose estimator (stub)")
    parser.add_argument("--config", help="Path to training config JSON/YAML", required=False)
    parser.parse_args(argv)
    raise SystemExit(
        "Pose training is not yet implemented in this repository. "
        "See docs/evaluation_and_benchmarks.md for the planned training pipeline."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
