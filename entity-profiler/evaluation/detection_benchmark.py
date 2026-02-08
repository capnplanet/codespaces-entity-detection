"""Person detection benchmark for Entity Profiler.

This script is intentionally minimal. It provides a config-driven entry point
for benchmarking the HOG- and ONNX-based detectors against datasets that can
be exposed as frames + ground-truth person bounding boxes.

It is designed to be extended with concrete dataset adapters rather than
hard-coding any particular corpus.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time
from typing import Iterable, List, Tuple

import cv2
import numpy as np

from entity_profiler.vision.detection import PersonDetector
from entity_profiler.vision.detector_onnx import OnnxPersonDetector


@dataclasses.dataclass
class DetectionSample:
    """Single annotated image for detection evaluation.

    This is a small, repo-local abstraction that dataset adapters can emit.

    Attributes
    ----------
    image_path: Path to an RGB image readable by OpenCV.
    boxes: List of ground-truth person boxes in (x, y, w, h) format.
    """

    image_path: pathlib.Path
    boxes: List[Tuple[float, float, float, float]]


@dataclasses.dataclass
class DetectionBenchmarkConfig:
    """Configuration for detection benchmark.

    This is intentionally narrow and JSON-serializable so that future
    dataset-specific configs can be translated into this structure.
    """

    dataset_index: pathlib.Path
    onnx_detector_path: pathlib.Path | None = None
    detector_type: str = "hog"  # "hog" or "onnx"
    iou_threshold: float = 0.5


def load_config(path: str | pathlib.Path) -> DetectionBenchmarkConfig:
    data = json.loads(pathlib.Path(path).read_text())
    return DetectionBenchmarkConfig(
        dataset_index=pathlib.Path(data["dataset_index"]),
        onnx_detector_path=pathlib.Path(data["onnx_detector_path"]) if data.get("onnx_detector_path") else None,
        detector_type=str(data.get("detector_type", "hog")),
        iou_threshold=float(data.get("iou_threshold", 0.5)),
    )


def iter_samples(index_path: pathlib.Path) -> Iterable[DetectionSample]:
    """Yield DetectionSample objects from a JSONL/NDJSON index.

    The index format is deliberately simple and documented in
    docs/evaluation_and_benchmarks.md. Each line is a JSON object:

        {"image_path": "/abs/or/relative/path.jpg",
         "boxes": [[x, y, w, h], ...]}
    """

    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            image_path = pathlib.Path(obj["image_path"]).expanduser()
            boxes = [tuple(map(float, b)) for b in obj.get("boxes", [])]
            yield DetectionSample(image_path=image_path, boxes=boxes)


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two (x, y, w, h) boxes."""

    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def evaluate(config: DetectionBenchmarkConfig) -> dict:
    """Run detection over all samples and compute simple metrics.

    Metrics are intentionally basic (per-sample precision/recall aggregated
    over the dataset) and can be extended later.
    """

    if config.detector_type == "onnx":
        detector = OnnxPersonDetector(model_path=config.onnx_detector_path)
    else:
        detector = PersonDetector()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    num_images = 0
    latencies: List[float] = []

    for sample in iter_samples(config.dataset_index):
        frame = cv2.imread(str(sample.image_path))
        if frame is None:
            # Skip unreadable frames but keep track of them.
            continue

        num_images += 1

        start = time.perf_counter()
        preds = detector.detect(frame, frame_index=0)
        latencies.append(time.perf_counter() - start)

        pred_boxes = np.array([p.bbox for p in preds], dtype=float) if preds else np.zeros((0, 4), dtype=float)
        gt_boxes = np.array(sample.boxes, dtype=float) if sample.boxes else np.zeros((0, 4), dtype=float)

        matched_gt = np.zeros(len(gt_boxes), dtype=bool)

        for pb in pred_boxes:
            best_iou = 0.0
            best_idx = -1
            for idx, gb in enumerate(gt_boxes):
                if matched_gt[idx]:
                    continue
                val = iou(pb, gb)
                if val > best_iou:
                    best_iou = val
                    best_idx = idx
            if best_iou >= config.iou_threshold and best_idx >= 0:
                total_tp += 1
                matched_gt[best_idx] = True
            else:
                total_fp += 1

        total_fn += int((~matched_gt).sum())

    precision = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    recall = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    avg_latency_ms = float(np.mean(latencies) * 1000.0) if latencies else 0.0

    return {
        "num_images": num_images,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": precision,
        "recall": recall,
        "avg_latency_ms": avg_latency_ms,
        "iou_threshold": config.iou_threshold,
    }


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Entity Profiler person detection benchmark")
    parser.add_argument("--config", required=True, help="Path to JSON config for detection benchmark")
    parser.add_argument("--output", required=True, help="Path to JSON file for results")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    results = evaluate(cfg)

    output_path = pathlib.Path(args.output)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
