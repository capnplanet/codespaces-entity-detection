"""Latency and throughput benchmark for the /ingest_frame API.

This script drives a running FastAPI instance with frames sourced from a
predefined list of images and measures request latency and effective
throughput. It is intentionally simple and can be extended with more complex
load models later.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time
from typing import Iterable, List

import requests


@dataclasses.dataclass
class LatencyConfig:
    """Config for latency/throughput benchmark.

    Attributes
    ----------
    api_url: Base URL for the FastAPI app (e.g. http://localhost:8000).
    frames_index: JSONL/NDJSON with one image path per line.
    camera_id: Camera identifier to send with requests.
    concurrency: Number of concurrent workers to use.
    total_requests: Total number of requests to send.
    """

    api_url: str
    frames_index: pathlib.Path
    camera_id: str
    concurrency: int = 1
    total_requests: int = 100


def load_config(path: str | pathlib.Path) -> LatencyConfig:
    data = json.loads(pathlib.Path(path).read_text())
    return LatencyConfig(
        api_url=data["api_url"],
        frames_index=pathlib.Path(data["frames_index"]),
        camera_id=data.get("camera_id", "bench_cam"),
        concurrency=int(data.get("concurrency", 1)),
        total_requests=int(data.get("total_requests", 100)),
    )


def iter_frame_paths(index_path: pathlib.Path) -> Iterable[pathlib.Path]:
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line) if line.startswith("{") else {"image_path": line}
            yield pathlib.Path(obj["image_path"]).expanduser()


def run_benchmark(cfg: LatencyConfig) -> dict:
    frame_paths = list(iter_frame_paths(cfg.frames_index))
    if not frame_paths:
        return {"num_requests": 0, "latencies_ms": []}

    url = cfg.api_url.rstrip("/") + "/ingest_frame"

    latencies_ms: List[float] = []
    num_sent = 0

    for i in range(cfg.total_requests):
        path = frame_paths[i % len(frame_paths)]
        with path.open("rb") as f:
            files = {"frame": (path.name, f, "image/jpeg")}
            data = {"camera_id": cfg.camera_id, "timestamp": str(time.time())}
            start = time.perf_counter()
            resp = requests.post(url, data=data, files=files, timeout=30)
            elapsed = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed)
            num_sent += 1
            resp.raise_for_status()

    latencies_ms.sort()
    p50 = latencies_ms[int(0.5 * len(latencies_ms))] if latencies_ms else 0.0
    p95 = latencies_ms[int(0.95 * len(latencies_ms))] if latencies_ms else 0.0
    p99 = latencies_ms[int(0.99 * len(latencies_ms))] if latencies_ms else 0.0

    total_time_s = (latencies_ms[-1] - latencies_ms[0]) / 1000.0 if len(latencies_ms) > 1 else 0.0
    throughput = float(num_sent / total_time_s) if total_time_s > 0 else 0.0

    return {
        "num_requests": num_sent,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "throughput_estimate_fps": throughput,
    }


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Entity Profiler /ingest_frame latency benchmark")
    parser.add_argument("--config", required=True, help="Path to JSON config for latency benchmark")
    parser.add_argument("--output", required=True, help="Path to JSON file for results")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    results = run_benchmark(cfg)

    output_path = pathlib.Path(args.output)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
