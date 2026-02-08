"""Lightweight smoke tests for evaluation scripts.

These tests do not exercise real datasets, but they ensure that the top-level
CLI entry points for the evaluation scripts can be imported and invoked with
minimal in-memory configuration.
"""

from __future__ import annotations

import json
import pathlib
from tempfile import TemporaryDirectory

import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "evaluation"
SRC_ROOT = ROOT / "src"

for p in (EVAL_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


detection_benchmark = importlib.import_module("detection_benchmark")
fall_benchmark = importlib.import_module("fall_benchmark")
health_safety_scenarios = importlib.import_module("health_safety_scenarios")


def _write_temp_config(tmpdir: pathlib.Path, name: str, data: dict) -> pathlib.Path:
    path = tmpdir / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_detection_benchmark_empty_index_smoke() -> None:
    with TemporaryDirectory() as d:
        tmpdir = pathlib.Path(d)
        index = tmpdir / "index.jsonl"
        index.write_text("", encoding="utf-8")
        cfg_path = _write_temp_config(
            tmpdir,
            "det_cfg.json",
            {"dataset_index": str(index)},
        )
        cfg = detection_benchmark.load_config(cfg_path)
        result = detection_benchmark.evaluate(cfg)
        assert result["num_images"] == 0


def test_fall_benchmark_empty_inputs_smoke() -> None:
    with TemporaryDirectory() as d:
        tmpdir = pathlib.Path(d)
        obs = tmpdir / "obs.jsonl"
        obs.write_text("", encoding="utf-8")
        labels = tmpdir / "labels.jsonl"
        labels.write_text("", encoding="utf-8")
        health_cfg = tmpdir / "health.json"
        # Minimal config; individual values are not important for an empty store.
        health_cfg.write_text(json.dumps({"severity": "info"}), encoding="utf-8")

        cfg_path = _write_temp_config(
            tmpdir,
            "fall_cfg.json",
            {
                "observations_path": str(obs),
                "labels_path": str(labels),
                "health_config_path": str(health_cfg),
            },
        )
        cfg = fall_benchmark.load_config(cfg_path)
        result = fall_benchmark.evaluate(cfg)
        assert result["num_labels"] == 0


def test_health_safety_scenarios_empty_inputs_smoke() -> None:
    with TemporaryDirectory() as d:
        tmpdir = pathlib.Path(d)
        obs = tmpdir / "obs.jsonl"
        obs.write_text("", encoding="utf-8")
        labels = tmpdir / "labels.jsonl"
        labels.write_text("", encoding="utf-8")
        health_cfg = tmpdir / "health.json"
        health_cfg.write_text(json.dumps({"severity": "info"}), encoding="utf-8")
        safety_cfg = tmpdir / "safety.json"
        safety_cfg.write_text(json.dumps({"severity": "info"}), encoding="utf-8")

        cfg_path = _write_temp_config(
            tmpdir,
            "scen_cfg.json",
            {
                "observations_path": str(obs),
                "labels_path": str(labels),
                "health_config_path": str(health_cfg),
                "safety_config_path": str(safety_cfg),
            },
        )
        cfg = health_safety_scenarios.load_config(cfg_path)
        result = health_safety_scenarios.evaluate(cfg)
        assert result["num_labels"] == 0
