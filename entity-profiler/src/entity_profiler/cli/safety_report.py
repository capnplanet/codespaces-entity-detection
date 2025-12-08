import argparse
import json
from pathlib import Path
import time

from ..profiling.entity_store import EntityStore
from ..config import load_safety_config
from ..safety.rules import evaluate_safety_events


def main():
    parser = argparse.ArgumentParser(description="Generate safety events report from a persisted entity store.")
    parser.add_argument("store_file", type=Path, nargs="?", default=Path("entity_store.json"))
    parser.add_argument("--output", type=Path, default=Path("safety_events.json"))
    parser.add_argument("--now", type=float, default=None, help="Override current timestamp for evaluation")
    args = parser.parse_args()

    store = EntityStore.load_json(args.store_file)
    safety_cfg = load_safety_config()
    events = evaluate_safety_events(store, safety_cfg, now_ts=args.now or time.time())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump([ev.__dict__ for ev in events], f, indent=2)
    print(f"Wrote {len(events)} safety events to {args.output}")


if __name__ == "__main__":
    main()
