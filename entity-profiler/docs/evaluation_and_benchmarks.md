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

A new `evaluation/` directory holds scripts and configuration for running benchmarks. Implemented components include:

- `evaluation/detection_benchmark.py` — runs person detection on standard or user-provided datasets using either the built-in HOG `PersonDetector` or the optional ONNX detector, driven by a simple JSONL index of frames and ground-truth boxes.
- `evaluation/fall_benchmark.py` — replays observation and wearable timelines through the health rule engine to evaluate fall heuristics and the fall model against labeled fall events.
- `evaluation/health_safety_scenarios.py` — runs health and safety rules on synthetic or recorded timelines (encoded as NDJSON) to measure true/false positives per event type.
- `evaluation/latency_throughput.py` — drives the `/ingest_frame` API with frames listed in a JSONL index and measures end-to-end latency and an approximate throughput.

Planned / partially implemented gait-specific components include:

- `evaluation/datasets/ou_mvlp_index.py` — builds a JSONL index from an unpacked OU-MVLP-style gait corpus on disk (subject/view/sequence → frames_dir).
- `evaluation/datasets/physionet_multi_gait_posture_index.py` — builds a JSONL index from the PhysioNet multi-gait-posture processed data (participant / condition / corridor → skeleton CSVs).
- `evaluation/physionet_gait_summary.py` — computes simple gait sequence statistics (frames per sequence and pelvis trajectory length) from that index to validate and summarize integration with the PhysioNet corpus.
- `evaluation/physionet_gait_advanced.py` — derives richer kinematic metrics (pelvis speed, foot motion ranges, asymmetry, and per-speed aggregates) from the same PhysioNet index.
- `evaluation/gait_reid_benchmark.py` — will evaluate deterministic and learned gait embeddings on standard re-identification metrics using gait-style indexes.

Each script currently:
- Accepts a JSON configuration file describing input indexes and paths.
- Emits structured JSON results that can be aggregated or plotted.
- Is designed to be extended with dataset-specific adapters without modifying core library code.

### Example: COCO 2017 person detection benchmark

To make the plan concrete, the repository includes a small adapter for the COCO 2017 dataset (a widely used academic benchmark with permissive research licensing):

- `evaluation/datasets/coco_person_index.py` — reads COCO 2017 images and `instances_*.json` annotations and writes a JSONL detection index compatible with `evaluation/detection_benchmark.py`.
- `evaluation/configs/coco_val_detection_example.json` — example config pointing at a COCO person index and choosing the default HOG detector.

Workflow (once you have downloaded COCO 2017 val images and annotations to `data/raw/coco2017`):

1. Build a person-only index for the validation split:

   ```bash
   python -m evaluation.datasets.coco_person_index \
     --images-dir data/raw/coco2017/val2017 \
     --annotations data/raw/coco2017/annotations/instances_val2017.json \
     --output evaluation/indexes/coco_val2017_person.jsonl
   ```

2. Run the detection benchmark over that index:

   ```bash
   python -m evaluation.detection_benchmark \
     --config evaluation/configs/coco_val_detection_example.json \
     --output evaluation/results/coco_val_detection_example.json
   ```

This yields precision/recall and latency metrics for person detection on a real, widely used academic corpus without pulling any dataset files into the repository itself.

### Gait corpora: OU-MVLP (planned) and PhysioNet multi-gait-posture (implemented)

For gait and re-identification, the intended primary silhouette-style corpus remains the
OU-ISIR Gait Database, Large Population Dataset (OU-MVLP), which requires registration
and is therefore not downloaded automatically. The workflow for OU-MVLP is:

1. Obtain access to OU-MVLP under its academic terms and unpack silhouettes under a path such as `data/raw/ou_mvlp`.
2. Build an index:

  ```bash
  python -m evaluation.datasets.ou_mvlp_index \
    --root data/raw/ou_mvlp/silhouettes \
    --output evaluation/indexes/ou_mvlp_sequences.jsonl
  ```

3. Use that index with future gait benchmarks (for example `evaluation/gait_reid_benchmark.py`) and training scripts under `training/gait/` to learn and evaluate gait-based embeddings.

In addition, this repository now integrates the PhysioNet
“A multi-camera and multimodal dataset for posture and gait analysis” corpus
(`multi-gait-posture`, Creative Commons Attribution 4.0 license). The expected
workflow, after following PhysioNet's instructions to download the data, is:

1. Ensure the processed data are extracted under a path like:

  ```
  data/raw/physionet.org/files/multi-gait-posture/1.0.0/processed_data
  ```

  with per-participant folders containing `aligned_skeleton_2d_gait.csv` and related
  files as described in PhysioNet's `processed_data_description.txt`.

2. Build a gait sequence index over the processed data:

  ```bash
  python -m evaluation.datasets.physionet_multi_gait_posture_index \
    --root data/raw/physionet.org/files/multi-gait-posture/1.0.0/processed_data \
    --output evaluation/indexes/physionet_multi_gait_posture_sequences.jsonl
  ```

3. Run a simple gait summary evaluation over that index:

  ```bash
  python -m evaluation.physionet_gait_summary \
    --index evaluation/indexes/physionet_multi_gait_posture_sequences.jsonl \
    --output evaluation/results/physionet_multi_gait_posture_summary.json
  ```

This produces dataset-level statistics (number of participants and sequences,
frames per sequence, and pelvis trajectory length distributions) while relying
only on the already-downloaded, extracted CSV files rather than the original
ZIP archives.

4. Optionally, run a more detailed gait evaluation that looks at pelvis speed
   and left/right foot motion, aggregated globally and by target walking
   speed encoded in the condition name:

   ```bash
   python -m evaluation.physionet_gait_advanced \
     --index evaluation/indexes/physionet_multi_gait_posture_sequences.jsonl \
     --output evaluation/results/physionet_multi_gait_posture_advanced.json
   ```

The advanced evaluation report includes dataset-level distributions for frame
counts, pelvis path length, approximate per-frame speed, vertical foot motion,
and a simple asymmetry indicator, plus summaries grouped by nominal walking
speed (0.3 / 0.5 / 0.7).

## 3. Baseline Methods

To contextualize performance, we will implement simple baseline methods:

- **Motion-only baseline**: frame-differencing or basic motion zones without entity tracking.
- **Wearable-only baseline**: HR/SpO2 thresholds without visual context.
- **Naive activity count baseline**: simple counts of observations per time window without entity clustering or pattern-of-life.

Entity Profiler’s rules (e.g., quiet-hours motion, linger, burst, low mobility) will be evaluated side-by-side with these baselines.

## 4. Gait-Specific Evaluation (Planned)

Given the richer, multi-frame gait features, we plan a dedicated gait evaluation track under `evaluation/` with the following focus areas:

- **Mobility proxy calibration**
  - Validate that the gait speed proxy (and related variability measures) correlates with annotated walking pace and qualitative mobility labels (e.g., "slow", "normal", "fast").
  - Derive recommended ranges or camera-specific adjustments for the low-mobility threshold used in health rules.

- **Robustness across views and conditions**
  - Measure how stable the 18-D gait descriptor is across different camera placements (frontal vs side view), moderate occlusion, and varying clothing.
  - Report per-condition statistics (means/variances) rather than a single global score, to keep results interpretable.

- **Re-identification signal strength (within pseudonymous scope)**
  - On curated, privacy-appropriate datasets, measure how well the deterministic gait descriptor helps link pseudonymous entities across time and cameras when combined with clothing and soft biometrics.
  - Use standard re-id style metrics (e.g., Rank-k retrieval accuracy, simple top-N match rates) while keeping all IDs pseudonymous.

- **Ablation studies**
  - Compare performance using only speed-based features vs. the enriched 18-D descriptor (angles, asymmetry, ankle motion).
  - Quantify incremental benefits for both mobility detection and short-horizon re-identification tasks.

Planned gait evaluation scripts will follow the same pattern as the other evaluation tools (config-driven, seeded, and JSON/CSV outputs) and will be added alongside the existing detection, fall, and rule-evaluation plans once suitable datasets are available.

## 5. Reporting

Once scripts are in place, this document will be updated with:
- Summary tables of detection and fall metrics.
- Confusion matrices for key rules.
- Latency and throughput plots for representative hardware.

Those summaries can then be referenced directly in proposals or external reports to answer “how does this system perform compared to simpler baselines?”
