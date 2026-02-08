"""Simple gait summary evaluation for the PhysioNet multi-gait-posture corpus.

This script consumes a JSONL index produced by
`evaluation.datasets.physionet_multi_gait_posture_index` and computes a few
lightweight summary statistics directly from the aligned 2D gait skeletons.

It is intentionally conservative: it does **not** attempt full re-
identification, but instead reports per-sequence length and a crude motion
metric based on pelvis displacement, aggregated across the dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class SequenceMetrics:
    participant_id: str
    condition: str
    corridor: str
    num_frames: int
    pelvis_path_length: float


def _load_pelvis_trajectory(csv_path: Path) -> np.ndarray:
    """Load pelvis (x, y) over time from an aligned_skeleton_2d_gait CSV.

    The PhysioNet CSV has a leading index column, followed by pairs of
    `<joint>_x`, `<joint>_y` columns. We only need `pelvis_x` / `pelvis_y`.
    """

    xs: list[float] = []
    ys: list[float] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row["pelvis_x"])
                y = float(row["pelvis_y"])
            except (KeyError, ValueError):
                continue
            xs.append(x)
            ys.append(y)

    if not xs:
        return np.zeros((0, 2), dtype=np.float32)

    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    return coords


def iter_sequence_metrics(index_path: Path) -> Iterable[SequenceMetrics]:
    index_path = index_path.expanduser().resolve()
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            gait_csv = rec.get("aligned_skeleton_2d_gait")
            if not gait_csv:
                continue

            traj = _load_pelvis_trajectory(Path(gait_csv))
            if traj.shape[0] < 2:
                num_frames = int(traj.shape[0])
                path_len = 0.0
            else:
                diffs = np.diff(traj, axis=0)
                step = np.linalg.norm(diffs, axis=1)
                path_len = float(step.sum())
                num_frames = int(traj.shape[0])

            yield SequenceMetrics(
                participant_id=str(rec.get("participant_id", "")),
                condition=str(rec.get("condition", "")),
                corridor=str(rec.get("corridor", "")),
                num_frames=num_frames,
                pelvis_path_length=path_len,
            )


def summarize(index_path: Path) -> dict:
    per_seq = list(iter_sequence_metrics(index_path))
    if not per_seq:
        return {"num_sequences": 0}

    num_frames = np.array([s.num_frames for s in per_seq], dtype=np.int32)
    path_lengths = np.array([s.pelvis_path_length for s in per_seq], dtype=np.float32)

    return {
        "num_sequences": int(len(per_seq)),
        "num_participants": int(len({s.participant_id for s in per_seq})),
        "frames": {
            "mean": float(num_frames.mean()),
            "min": int(num_frames.min()),
            "max": int(num_frames.max()),
        },
        "pelvis_path_length": {
            "mean": float(path_lengths.mean()),
            "min": float(path_lengths.min()),
            "max": float(path_lengths.max()),
        },
        "sequences": [asdict(s) for s in per_seq],
    }


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description="PhysioNet multi-gait-posture gait summary")
    parser.add_argument("--index", required=True, help="Path to PhysioNet gait index JSONL")
    parser.add_argument("--output", required=True, help="Path to JSON results file")
    args = parser.parse_args(argv)

    result = summarize(Path(args.index))
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":  # pragma: no cover
    main()
