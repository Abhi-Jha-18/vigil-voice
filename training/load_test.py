"""
VigilVoice — Concurrency & Load Testing (Phase 14)
===================================================
Simulates concurrent API detection and health requests using FastAPI TestClient
to evaluate server throughput, concurrency limiting, and response latencies.

Usage:
    python training/load_test.py --concurrency 10 --requests 30
"""

import os
import sys
import io
import time
import argparse
import concurrent.futures
import numpy as np
from scipy.io import wavfile
from fastapi.testclient import TestClient

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.main import app


def _create_in_memory_wav(duration_sec: float = 1.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    samples = (0.4 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, samples)
    return buf.getvalue()


def run_concurrency_test(num_requests: int = 20, max_workers: int = 5) -> dict:
    print("=" * 70)
    print(f"  VigilVoice — Concurrency Load Test ({num_requests} requests across {max_workers} workers)")
    print("=" * 70)

    client = TestClient(app)
    wav_bytes = _create_in_memory_wav(duration_sec=1.5)

    def _single_request(req_id: int):
        t0 = time.perf_counter()
        try:
            resp = client.post(
                "/api/detect",
                files={"file": (f"test_load_{req_id}.wav", wav_bytes, "audio/wav")},
                data={"phase": "phase1"}
            )
            lat_ms = (time.perf_counter() - t0) * 1000
            return {
                "id": req_id,
                "status_code": resp.status_code,
                "latency_ms": lat_ms,
                "success": resp.status_code == 200,
                "request_id": resp.headers.get("X-Request-ID")
            }
        except Exception as exc:
            lat_ms = (time.perf_counter() - t0) * 1000
            return {
                "id": req_id,
                "status_code": 0,
                "latency_ms": lat_ms,
                "success": False,
                "error": str(exc)
            }

    results = []
    t_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_single_request, i) for i in range(num_requests)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    total_wall_sec = time.perf_counter() - t_start

    latencies = [r["latency_ms"] for r in results]
    successes = sum(1 for r in results if r["success"])
    failures = len(results) - successes
    status_codes = {}
    for r in results:
        code = r["status_code"]
        status_codes[code] = status_codes.get(code, 0) + 1

    summary = {
        "total_requests": num_requests,
        "concurrency_workers": max_workers,
        "total_wall_time_sec": total_wall_sec,
        "requests_per_second": num_requests / total_wall_sec if total_wall_sec > 0 else 0,
        "successful_requests": successes,
        "failed_requests": failures,
        "status_code_distribution": status_codes,
        "latency_metrics": {
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies))
        }
    }

    print("\n--- Concurrency Test Results ---")
    print(f"Total Requests:     {num_requests}")
    print(f"Successful:         {successes} ({successes/num_requests*100:.1f}%)")
    print(f"Failed:             {failures}")
    print(f"Throughput:         {summary['requests_per_second']:.2f} req/sec")
    print(f"Mean Latency:       {summary['latency_metrics']['mean_ms']:.2f} ms")
    print(f"Median Latency:     {summary['latency_metrics']['median_ms']:.2f} ms")
    print(f"P95 Latency:        {summary['latency_metrics']['p95_ms']:.2f} ms")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VigilVoice Concurrency Test")
    parser.add_argument("--requests", type=int, default=20, help="Total requests to dispatch")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent worker threads")
    args = parser.parse_args()

    run_concurrency_test(num_requests=args.requests, max_workers=args.concurrency)
