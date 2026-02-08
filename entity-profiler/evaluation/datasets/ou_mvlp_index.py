"""Index builder for OU-MVLP-style gait datasets.

This is a placeholder adapter for the OU-ISIR Gait Database, Large Population
Dataset (OU-MVLP). It assumes that silhouettes or frames are unpacked under a
root like `data/raw/ou_mvlp` with a consistent directory structure.

Because OU-MVLP requires registration and approval, this script does not
attempt to download it; it only turns an existing local copy into a JSONL
index for training and evaluation.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Iterable


def iter_sequences(root: pathlib.Path) -> Iterable[dict]:
    """Yield simple sequence records from an OU-MVLP-style layout.

    The exact layout can vary depending on how the archives are unpacked. A
    common pattern is something like:

        root/
          subject_id/
            view_id/
              seq_id/
                frame_0001.png
                ...

    This function walks three directory levels below `root` and treats each
    leaf directory as one sequence.
    """

    root = root.expanduser().resolve()
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        subject_id = subject_dir.name
        for view_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            view_id = view_dir.name
            for seq_dir in sorted(p for p in view_dir.iterdir() if p.is_dir()):
                seq_id = seq_dir.name
                yield {
                    "subject_id": subject_id,
                    "view_id": view_id,
                    "sequence_id": seq_id,
                    "frames_dir": str(seq_dir),
                }


def build_index(root: pathlib.Path, output: pathlib.Path) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for rec in iter_sequences(root):
            f.write(json.dumps(rec) + "\n")


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - adapter CLI
    parser = argparse.ArgumentParser(description="Build OU-MVLP gait index (placeholder)")
    parser.add_argument("--root", required=True, help="Path to OU-MVLP root (e.g. data/raw/ou_mvlp/silhouettes)")
    parser.add_argument("--output", required=True, help="Path to JSONL index to write")
    args = parser.parse_args(argv)

    build_index(pathlib.Path(args.root), pathlib.Path(args.output))


if __name__ == "__main__":  # pragma: no cover
    main()
