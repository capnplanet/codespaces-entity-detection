"""Index builder for the PhysioNet multi-gait-posture dataset.

This adapter walks the **processed** data tree from PhysioNet's
"A multi-camera and multimodal dataset for posture and gait analysis"
corpus (multi-gait-posture) and emits a JSONL index of sequences.

The expected layout (after extracting the participant ZIP files in
`processed_data/`) is, per the dataset documentation:

    processed_data/
      participant00/
        participant00/
          participant00_straight_0.7/
            participant00_straight_0.7_corridor3/
              aligned_skeleton_2d_gait.csv
              aligned_skeleton_2d_posture.csv
              aligned_skeleton_3d.csv
              normalized_skeleton_3d.csv
              synchronized_data_idx.csv

We treat each leaf "corridor" directory as a separate gait sequence and
record the paths to its skeleton CSVs.

This script does **not** download or unzip anything; it only indexes
already-extracted files.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Iterable


def iter_sequences(root: pathlib.Path) -> Iterable[dict]:
    """Yield simple sequence records from the PhysioNet processed layout.

    Parameters
    ----------
    root:
        Path to the `processed_data` directory, e.g.::

            data/raw/physionet.org/files/multi-gait-posture/1.0.0/processed_data
    """

    root = root.expanduser().resolve()

    # First level: participantXX directories created when unzipping
    # participantXX.zip into processed_data/participantXX/.
    for participant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        # Inside each, the zips expand to another participantXX/ folder.
        inner_dirs = [d for d in participant_dir.iterdir() if d.is_dir()]
        if not inner_dirs:
            continue
        # In practice there is one inner directory with the same name,
        # but we iterate to be robust.
        for inner in sorted(inner_dirs):
            participant_id = inner.name
            # Next level: condition directories such as
            # `participant01_straight_0.7`.
            for condition_dir in sorted(d for d in inner.iterdir() if d.is_dir()):
                condition_name = condition_dir.name
                # Final level: corridor folders such as
                # `participant01_straight_0.7_corridor3`.
                for corridor_dir in sorted(d for d in condition_dir.iterdir() if d.is_dir()):
                    corridor_name = corridor_dir.name

                    seq_rec: dict[str, object] = {
                        "participant_id": participant_id,
                        "condition": condition_name,
                        "corridor": corridor_name,
                    }

                    # Attach any skeleton CSVs that are present.
                    for fname, key in [
                        ("aligned_skeleton_2d_gait.csv", "aligned_skeleton_2d_gait"),
                        ("aligned_skeleton_2d_posture.csv", "aligned_skeleton_2d_posture"),
                        ("aligned_skeleton_3d.csv", "aligned_skeleton_3d"),
                        ("normalized_skeleton_3d.csv", "normalized_skeleton_3d"),
                        ("synchronized_data_idx.csv", "synchronized_data_idx"),
                    ]:
                        path = corridor_dir / fname
                        if path.is_file():
                            seq_rec[key] = str(path)

                    # Only yield sequences that actually have at
                    # least one skeleton file.
                    if any(k in seq_rec for k in ("aligned_skeleton_2d_gait", "aligned_skeleton_3d", "normalized_skeleton_3d")):
                        yield seq_rec


def build_index(root: pathlib.Path, output: pathlib.Path) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for rec in iter_sequences(root):
            f.write(json.dumps(rec) + "\n")


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - adapter CLI
    parser = argparse.ArgumentParser(
        description=(
            "Build gait sequence index for the PhysioNet multi-gait-posture "
            "processed dataset"
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help=(
            "Path to processed_data root, e.g. "
            "data/raw/physionet.org/files/multi-gait-posture/1.0.0/processed_data"
        ),
    )
    parser.add_argument("--output", required=True, help="Path to JSONL index to write")
    args = parser.parse_args(argv)

    build_index(pathlib.Path(args.root), pathlib.Path(args.output))


if __name__ == "__main__":  # pragma: no cover
    main()
