"""Stub script for offline calibration of health, fall, safety, and tracking thresholds.

The intended workflow is:
- Sample parameter grids or use Bayesian optimization over HealthConfig,
  SafetyConfig, and ProfilingConfig fields.
- For each candidate, run existing evaluation scripts (fall_benchmark,
  health_safety_scenarios, gait benchmarks) and measure metrics.
- Select parameter sets that meet or approximate 95th-percentile targets.
- Write updated configs to data/health_config.json and data/safety_config.json
  plus small calibration reports under evaluation/results/.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - training stub
    parser = argparse.ArgumentParser(description="Calibrate health/safety thresholds (stub)")
    parser.add_argument("--config", help="Path to calibration config JSON/YAML", required=False)
    parser.parse_args(argv)
    raise SystemExit(
        "Rule calibration is not yet automated. Use the evaluation scripts "
        "and docs/evaluation_and_benchmarks.md as a guide for manual tuning."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
