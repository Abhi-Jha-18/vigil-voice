"""
VigilVoice — Inference Performance & Resource Benchmarking (Phase 14)
======================================================================
Measures real performance metrics across model loading, preprocessing,
feature extraction, inference latency, and memory footprint.

Usage:
    python training/benchmark.py --iterations 25 --output reports/performance_benchmark.json
"""

import os
import sys
import json
import time
import argparse
import platform
import tracemalloc
from pathlib import Path

import numpy as np
import torch

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.config import settings
from backend.audio.config import default_config
from backend.detection.detector import SimpleCNNDetector, get_model_status
from backend.audio.pipeline import process_canonical
from backend.audio.features import extract_acoustic_features
from training.preprocessing import preprocess_for_training


def _generate_synthetic_wav(path: Path, duration_sec: float = 3.0, sr: int = 16000) -> None:
    import wave
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(signal.tobytes())


def run_benchmark(iterations: int = 25, output_file: str | Path = "reports/performance_benchmark.json") -> dict:
    print("=" * 70)
    print(f"  VigilVoice — Performance & Resource Benchmark ({iterations} iterations)")
    print("=" * 70)

    import tempfile
    tracemalloc.start()
    baseline_mem_kb = tracemalloc.get_traced_memory()[0] / 1024.0

    # 1. Model Loading Latency
    load_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNNDetector().to(device)
    if settings.model_path.is_file():
        model.load_state_dict(torch.load(settings.model_path, map_location=device, weights_only=True))
    model.eval()
    model_load_time_ms = (time.perf_counter() - load_start) * 1000

    # 2. Prepare Benchmark Audio Fixture
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_path = Path(temp_wav.name)
    _generate_synthetic_wav(temp_path, duration_sec=3.0, sr=default_config.sample_rate)

    prep_times = []
    feat_times = []
    infer_times = []
    total_times = []

    print(f"Executing {iterations} warm-up and measured inference trials...")

    for i in range(iterations):
        t_total_start = time.perf_counter()

        # Preprocessing
        t_prep_start = time.perf_counter()
        prep_res = process_canonical(str(temp_path), config=default_config)
        t_prep_end = time.perf_counter()
        prep_times.append((t_prep_end - t_prep_start) * 1000)

        # Feature Extraction
        t_feat_start = time.perf_counter()
        features = extract_acoustic_features(prep_res.audio_data, sr=default_config.sample_rate)
        mfcc_seq = features.get("mfcc_sequence", [])
        mfcc = np.array(mfcc_seq, dtype=np.float32)
        if mfcc.shape[1] < 64:
            mfcc = np.pad(mfcc, ((0, 0), (0, 64 - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :64]
        inp_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        t_feat_end = time.perf_counter()
        feat_times.append((t_feat_end - t_feat_start) * 1000)

        # Inference
        t_infer_start = time.perf_counter()
        with torch.no_grad():
            _ = model(inp_tensor).cpu().numpy()
        t_infer_end = time.perf_counter()
        infer_times.append((t_infer_end - t_infer_start) * 1000)

        t_total_end = time.perf_counter()
        total_times.append((t_total_end - t_total_start) * 1000)

    # Clean up temp file
    if temp_path.exists():
        os.remove(temp_path)

    current_mem_kb, peak_mem_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    def _stats(arr: list[float]) -> dict:
        np_arr = np.array(arr)
        return {
            "mean_ms": float(np.mean(np_arr)),
            "median_ms": float(np.median(np_arr)),
            "p95_ms": float(np.percentile(np_arr, 95)),
            "min_ms": float(np.min(np_arr)),
            "max_ms": float(np.max(np_arr)),
            "std_ms": float(np.std(np_arr)),
        }

    report = {
        "status": "COMPLETED",
        "timestamp": str(np.datetime64('now')),
        "environment": {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "device": str(device),
            "model_status": get_model_status(),
            "sample_rate_hz": default_config.sample_rate,
        },
        "model_loading": {
            "load_time_ms": float(model_load_time_ms)
        },
        "benchmarks": {
            "iterations": iterations,
            "preprocessing": _stats(prep_times),
            "feature_extraction": _stats(feat_times),
            "model_inference": _stats(infer_times),
            "total_pipeline": _stats(total_times),
        },
        "memory_profile_kb": {
            "baseline_kb": float(baseline_mem_kb),
            "peak_inference_kb": float(peak_mem_kb / 1024.0),
            "net_allocated_kb": float((current_mem_kb - baseline_mem_kb) / 1024.0),
        }
    }

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n--- Benchmark Summary ---")
    print(f"Model Load Time:          {model_load_time_ms:.2f} ms")
    print(f"Preprocessing (mean):     {report['benchmarks']['preprocessing']['mean_ms']:.2f} ms")
    print(f"Feature Extraction (mean):{report['benchmarks']['feature_extraction']['mean_ms']:.2f} ms")
    print(f"Model Inference (mean):   {report['benchmarks']['model_inference']['mean_ms']:.2f} ms")
    print(f"Total Pipeline (mean):    {report['benchmarks']['total_pipeline']['mean_ms']:.2f} ms")
    print(f"Total Pipeline (p95):     {report['benchmarks']['total_pipeline']['p95_ms']:.2f} ms")
    print(f"Peak Memory Footprint:    {report['memory_profile_kb']['peak_inference_kb']:.2f} KB")
    print(f"\nSaved benchmark report to: {out_path}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VigilVoice Performance Benchmark")
    parser.add_argument("--iterations", type=int, default=25, help="Number of benchmark iterations")
    parser.add_argument("--output", type=Path, default=Path("reports/performance_benchmark.json"), help="Output JSON path")
    args = parser.parse_args()

    run_benchmark(iterations=args.iterations, output_file=args.output)
