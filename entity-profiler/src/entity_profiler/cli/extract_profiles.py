import argparse
import json

from ..profiling.entity_store import EntityStore
from ..profiling.pattern_of_life import summarize_all_entities


def main():
    parser = argparse.ArgumentParser(description="Dump entity profiles as JSON.")
    parser.add_argument("--output", type=str, default="entity_profiles.json")
    args = parser.parse_args()

    store = EntityStore()  # In a real app, load from persistence
    summaries = summarize_all_entities(store)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote {len(summaries)} entity profiles to {args.output}")


if __name__ == "__main__":
    main()
