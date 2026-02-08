"""Utilities for building detection indexes from MOTChallenge-style datasets.

This module assumes a standard MOT17-style layout:

    MOT17/
      train/
        MOT17-02-FRCNN/
          img1/000001.jpg
          gt/gt.txt
        ...

The gt.txt file is expected to contain CSV rows with at least the first six
fields: frame, id, x, y, w, h, ... as described by the MOTChallenge format.

We ignore track IDs and keep only per-frame person boxes, writing out a JSONL
index suitable for evaluation/detection_benchmark.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Dict, List, Tuple


def _collect_frame_boxes(gt_path: pathlib.Path) -> Dict[int, List[Tuple[float, float, float, float]]]:
    frame_to_boxes: Dict[int, List[Tuple[float, float, float, float]]] = {}
    with gt_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            try:
                frame = int(float(row[0]))
                x = float(row[2])
                y = float(row[3])
                w = float(row[4])
                h = float(row[5])
            except (ValueError, IndexError):
                continue
            frame_to_boxes.setdefault(frame, []).append((x, y, w, h))
    return frame_to_boxes


def build_detection_index(mot_root: pathlib.Path, split: str, output_path: pathlib.Path) -> None:
    """Build a detection index JSONL from a MOT-style dataset.

    Parameters
    ----------
    mot_root: Path to the MOT17 root directory.
    split: Subdirectory under mot_root to use (e.g. "train" or "test").
    output_path: JSONL file to write.
    """

    split_dir = mot_root / split
    sequences = sorted(p for p in split_dir.iterdir() if p.is_dir())

    with output_path.open("w", encoding="utf-8") as out_f:
        for seq in sequences:
            img_dir = seq / "img1"
            gt_path = seq / "gt" / "gt.txt"
            if not img_dir.is_dir() or not gt_path.is_file():
                continue

            frame_to_boxes = _collect_frame_boxes(gt_path)
            if not frame_to_boxes:
                continue

            for frame_idx, boxes in sorted(frame_to_boxes.items()):
                # MOT17 images are usually named as 6-digit frame index.
                image_name = f"{frame_idx:06d}.jpg"
                image_path = img_dir / image_name
                obj = {"image_path": str(image_path), "boxes": boxes}
                out_f.write(json.dumps(obj) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build MOT17 detection index for Entity Profiler benchmarks")
    parser.add_argument("--mot-root", required=True, help="Path to MOT17 root directory")
    parser.add_argument("--split", default="train", help='Dataset split under MOT17 (default: "train")')
    parser.add_argument("--output", required=True, help="Path to JSONL index to write")
    args = parser.parse_args(argv)

    mot_root = pathlib.Path(args.mot_root)
    output_path = pathlib.Path(args.output)
    build_detection_index(mot_root=mot_root, split=args.split, output_path=output_path)


if __name__ == "__main__":  # pragma: no cover
    main()
