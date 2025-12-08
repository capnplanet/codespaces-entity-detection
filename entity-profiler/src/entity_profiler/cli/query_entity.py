import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from ..profiling.entity_store import EntityStore
from ..profiling.pattern_of_life import summarize_all_entities


def _load_summaries_from_store(store_path: Path) -> List[Dict[str, Any]]:
    store = EntityStore.load_json(store_path)
    return summarize_all_entities(store)


def _load_summaries_from_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Profile summary file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Profile summary file must contain a list of entities")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Query entity summaries produced by build_tracks or the API."
    )
    parser.add_argument(
        "summary_file",
        type=Path,
        nargs="?",
        default=Path("entity_profiles.json"),
        help="Path to JSON summaries (default: entity_profiles.json)",
    )
    parser.add_argument(
        "--store-file",
        type=Path,
        default=Path("entity_store.json"),
        help="Optional path to a persisted entity store JSON; if present, takes precedence",
    )
    parser.add_argument(
        "--entity-id",
        type=str,
        help="Optional entity_id to filter; if omitted, prints all entities",
    )
    args = parser.parse_args()

    if args.store_file.exists():
        summaries = _load_summaries_from_store(args.store_file)
    else:
        summaries = _load_summaries_from_file(args.summary_file)

    if args.entity_id:
        summaries = [s for s in summaries if s.get("entity_id") == args.entity_id]
        if not summaries:
            print(f"No entity found with id {args.entity_id}")
            return

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
