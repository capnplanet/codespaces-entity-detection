"""Stub script for training a COCO-based person detector.

This script is a placeholder for a future training pipeline that will use
COCO person data to train or fine-tune a detector compatible with
`entity_profiler.vision.detector_onnx.OnnxPersonDetector`.

The expected outcome is an ONNX model written to `models/detector.onnx` with
output `(N, 6)` rows of `[x1, y1, x2, y2, score, class]`.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - training stub
    parser = argparse.ArgumentParser(description="Train a COCO-based detector (stub)")
    parser.add_argument("--config", help="Path to training config JSON/YAML", required=False)
    parser.parse_args(argv)
    raise SystemExit(
        "Detector training is not yet implemented in this repository. "
        "See docs/evaluation_and_benchmarks.md for the planned training pipeline."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
