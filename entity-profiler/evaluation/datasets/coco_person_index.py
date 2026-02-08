"""Build a detection index from COCO 2017 person annotations.

This adapter expects the official COCO 2017 layout, for example:

    data/raw/coco2017/
      val2017/
        000000000139.jpg
        ...
      annotations/
        instances_val2017.json

It writes a JSONL index compatible with evaluation/detection_benchmark.py,
containing image paths and person bounding boxes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Tuple


def build_person_index(images_dir: pathlib.Path, annotations_path: pathlib.Path, output_path: pathlib.Path) -> None:
    images_dir = images_dir.expanduser().resolve()
    annotations_path = annotations_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(annotations_path.read_text())

    # COCO structure: images (id, file_name, ...), annotations (image_id, category_id, bbox, iscrowd, ...)
    id_to_filename: Dict[int, str] = {int(img["id"]): img["file_name"] for img in data.get("images", [])}

    person_category_id = 1  # standard COCO person class id
    image_to_boxes: Dict[int, List[Tuple[float, float, float, float]]] = {}

    for ann in data.get("annotations", []):
        if int(ann.get("category_id", -1)) != person_category_id:
            continue
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        image_id = int(ann["image_id"])
        x, y, w, h = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
        image_to_boxes.setdefault(image_id, []).append((x, y, w, h))

    with output_path.open("w", encoding="utf-8") as f:
        for image_id, boxes in sorted(image_to_boxes.items()):
            filename = id_to_filename.get(image_id)
            if not filename:
                continue
            image_path = images_dir / filename
            obj = {"image_path": str(image_path), "boxes": boxes}
            f.write(json.dumps(obj) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build COCO person detection index for Entity Profiler benchmarks")
    parser.add_argument("--images-dir", required=True, help="Path to COCO images directory (e.g. data/raw/coco2017/val2017)")
    parser.add_argument("--annotations", required=True, help="Path to COCO instances JSON (e.g. instances_val2017.json)")
    parser.add_argument("--output", required=True, help="Path to JSONL index to write")
    args = parser.parse_args(argv)

    build_person_index(
        images_dir=pathlib.Path(args.images_dir),
        annotations_path=pathlib.Path(args.annotations),
        output_path=pathlib.Path(args.output),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
