"""Advanced gait evaluation for the PhysioNet multi-gait-posture corpus.

This script builds on the JSONL index produced by
`evaluation.datasets.physionet_multi_gait_posture_index` and computes
sequence-level and aggregate kinematic metrics from the aligned 2D gait
skeletons.

It is still intentionally lightweight (no learned models, no full re-id),
but more expressive than the simple pelvis path-length summary:

- Per-sequence metrics:
  - number of frames
  - pelvis path length and mean per-frame speed
  - vertical range of left/right feet (step amplitude proxy)
  - asymmetry between left and right foot motion

- Aggregated metrics:
  - distributions over all sequences
  - per-target-speed aggregates (0.3 / 0.5 / 0.7), inferred from the
    condition string in the index (e.g. "participant01_straight_0.5").
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Dict, Any

import numpy as np


@dataclass
class AdvancedSequenceMetrics:
    participant_id: str
    condition: str
    corridor: str
    target_speed: float | None
    num_frames: int
    pelvis_path_length: float
    pelvis_speed_mean: float
    left_foot_vert_range: float
    right_foot_vert_range: float
    foot_vert_asymmetry: float


def _load_skeleton_2d(csv_path: Path) -> Dict[str, np.ndarray]:
    """Load time series for selected joints from an aligned_skeleton_2d_gait CSV.

    Returns a mapping from joint name ("pelvis", "left_foot", "right_foot")
    to an array of shape (T, 2) with (x, y) in image pixels.
    """

    xs: dict[str, list[float]] = {
        "pelvis_x": [],
        "pelvis_y": [],
        "right_foot_x": [],
        "right_foot_y": [],
        "left_foot_x": [],
        "left_foot_y": [],
    }

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                px = float(row["pelvis_x"])
                py = float(row["pelvis_y"])
                rfx = float(row["right_foot_x"])
                rfy = float(row["right_foot_y"])
                lfx = float(row["left_foot_x"])
                lfy = float(row["left_foot_y"])
            except (KeyError, ValueError):
                # Skip malformed frames rather than failing the whole sequence.
                continue
            xs["pelvis_x"].append(px)
            xs["pelvis_y"].append(py)
            xs["right_foot_x"].append(rfx)
            xs["right_foot_y"].append(rfy)
            xs["left_foot_x"].append(lfx)
            xs["left_foot_y"].append(lfy)

    if not xs["pelvis_x"]:
        # Empty sequence.
        zeros = np.zeros((0, 2), dtype=np.float32)
        return {"pelvis": zeros, "right_foot": zeros, "left_foot": zeros}

    def _stack(name_x: str, name_y: str) -> np.ndarray:
        arr = np.stack([xs[name_x], xs[name_y]], axis=1).astype(np.float32)
        return arr

    return {
        "pelvis": _stack("pelvis_x", "pelvis_y"),
        "right_foot": _stack("right_foot_x", "right_foot_y"),
        "left_foot": _stack("left_foot_x", "left_foot_y"),
    }


def _parse_target_speed(condition: str) -> float | None:
    """Extract target speed (0.3/0.5/0.7, etc.) from a condition string.

    Conditions look like "participant01_left_0.3" or "participant01_straight_0.7".
    We parse the last underscore-delimited token as a float when possible.
    """

    if not condition:
        return None
    parts = condition.split("_")
    if not parts:
        return None
    try:
        return float(parts[-1])
    except ValueError:
        return None


def iter_advanced_metrics(index_path: Path) -> Iterable[AdvancedSequenceMetrics]:
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

            series = _load_skeleton_2d(Path(gait_csv))
            pelvis = series["pelvis"]
            right_foot = series["right_foot"]
            left_foot = series["left_foot"]

            num_frames = int(pelvis.shape[0])
            if num_frames < 2:
                path_len = 0.0
                speed_mean = 0.0
            else:
                diffs = np.diff(pelvis, axis=0)
                step = np.linalg.norm(diffs, axis=1)
                path_len = float(step.sum())
                speed_mean = float(step.mean())

            def _vert_range(traj: np.ndarray) -> float:
                if traj.shape[0] == 0:
                    return 0.0
                y = traj[:, 1]
                return float(y.max() - y.min())

            lf_range = _vert_range(left_foot)
            rf_range = _vert_range(right_foot)
            asym = float(abs(lf_range - rf_range))

            yield AdvancedSequenceMetrics(
                participant_id=str(rec.get("participant_id", "")),
                condition=str(rec.get("condition", "")),
                corridor=str(rec.get("corridor", "")),
                target_speed=_parse_target_speed(str(rec.get("condition", ""))),
                num_frames=num_frames,
                pelvis_path_length=path_len,
                pelvis_speed_mean=speed_mean,
                left_foot_vert_range=lf_range,
                right_foot_vert_range=rf_range,
                foot_vert_asymmetry=asym,
            )


def _agg_stats(values: np.ndarray) -> Dict[str, Any]:
    if values.size == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize_advanced(index_path: Path) -> dict:
    per_seq = list(iter_advanced_metrics(index_path))
    if not per_seq:
        return {"num_sequences": 0}

    # Global aggregates.
    num_frames = np.array([s.num_frames for s in per_seq], dtype=np.int32)
    path_lengths = np.array([s.pelvis_path_length for s in per_seq], dtype=np.float32)
    speeds = np.array([s.pelvis_speed_mean for s in per_seq], dtype=np.float32)
    lf_ranges = np.array([s.left_foot_vert_range for s in per_seq], dtype=np.float32)
    rf_ranges = np.array([s.right_foot_vert_range for s in per_seq], dtype=np.float32)
    asym = np.array([s.foot_vert_asymmetry for s in per_seq], dtype=np.float32)

    # Group by nominal target speed when available.
    by_speed: dict[float, list[AdvancedSequenceMetrics]] = {}
    for s in per_seq:
        if s.target_speed is None:
            continue
        by_speed.setdefault(s.target_speed, []).append(s)

    grouped: dict[str, Any] = {}
    for spd, seqs in sorted(by_speed.items(), key=lambda kv: kv[0]):
        arr_speed = np.array([s.pelvis_speed_mean for s in seqs], dtype=np.float32)
        arr_path = np.array([s.pelvis_path_length for s in seqs], dtype=np.float32)
        grouped[str(spd)] = {
            "num_sequences": int(len(seqs)),
            "pelvis_speed_mean": _agg_stats(arr_speed),
            "pelvis_path_length": _agg_stats(arr_path),
        }

    return {
        "num_sequences": int(len(per_seq)),
        "num_participants": int(len({s.participant_id for s in per_seq})),
        "frames": _agg_stats(num_frames.astype(np.float32)),
        "pelvis_path_length": _agg_stats(path_lengths),
        "pelvis_speed_mean": _agg_stats(speeds),
        "left_foot_vert_range": _agg_stats(lf_ranges),
        "right_foot_vert_range": _agg_stats(rf_ranges),
        "foot_vert_asymmetry": _agg_stats(asym),
        "by_target_speed": grouped,
        "sequences": [asdict(s) for s in per_seq],
    }


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description="Advanced PhysioNet gait evaluation")
    parser.add_argument("--index", required=True, help="Path to PhysioNet gait index JSONL")
    parser.add_argument("--output", required=True, help="Path to JSON results file")
    args = parser.parse_args(argv)

    result = summarize_advanced(Path(args.index))
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":  # pragma: no cover
    main()
