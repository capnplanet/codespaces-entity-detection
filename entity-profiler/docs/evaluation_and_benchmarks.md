# Evaluation and Benchmarks (Plan)

This document outlines how Entity Profiler will be evaluated against baseline methods and, where possible, compared to industry-standard analytics behaviors.

## 1. Goals

- Provide concrete, reproducible metrics for key capabilities:
  - Person detection performance (precision/recall, latency).
  - Fall detection sensitivity/specificity.
  - Health and safety rule accuracy on curated scenarios.
  - End-to-end latency and throughput for `/ingest_frame`.
- Define simple baselines (e.g., motion-only, wearable-only) to quantify added value from entity profiling and multimodal fusion.

## 2. Evaluation Suite Structure

A new `evaluation/` directory will hold scripts and configuration for running benchmarks. Planned components include:

- `evaluation/detection_benchmark.py` — runs person detection on standard or user-provided datasets using HOG and optional ONNX detectors.
- `evaluation/fall_benchmark.py` — evaluates fall heuristics and fall model over labeled sequences.
- `evaluation/health_safety_scenarios.py` — runs health and safety rules on synthetic or recorded timelines to measure true/false positives.
- `evaluation/latency_throughput.py` — measures end-to-end ingest latency and sustainable frame rate under configurable load.

Each script will:
- Accept a configuration file and random seed.
- Emit structured JSON or CSV results.
- Be documented so external reviewers can reproduce results.

## 3. Baseline Methods

To contextualize performance, we will implement simple baseline methods:

- **Motion-only baseline**: frame-differencing or basic motion zones without entity tracking.
- **Wearable-only baseline**: HR/SpO2 thresholds without visual context.
- **Naive activity count baseline**: simple counts of observations per time window without entity clustering or pattern-of-life.

Entity Profiler’s rules (e.g., quiet-hours motion, linger, burst, low mobility) will be evaluated side-by-side with these baselines.

## 4. Reporting

Once scripts are in place, this document will be updated with:
- Summary tables of detection and fall metrics.
- Confusion matrices for key rules.
- Latency and throughput plots for representative hardware.

Those summaries can then be referenced directly in proposals or external reports to answer “how does this system perform compared to simpler baselines?”
