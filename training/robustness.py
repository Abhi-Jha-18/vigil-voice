"""
VigilVoice — Robustness, Generalization & Adversarial Audio Testing (Phase 13)
================================================================================
Evaluates a trained VigilVoice model against realistic audio transformations:
- Background Noise (20dB, 10dB, 5dB, 0dB)
- Sample Rate Transcoding (8kHz, 22.05kHz, 44.1kHz)
- Volume Gain Adjustments (-12dB, -6dB, +6dB)
- Truncation / Short Audio (1.0s, 3.0s)
- Simulated Reverberation (Room impulse response)
- Combined Conditions (Noise + Volume, Resampling + Noise)

Usage:
    python training/robustness.py \
        --manifest data/splits/manifest.csv \
        --checkpoint models/production/vigilvoice_cnn_best.pth \
        --reports-dir reports
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.detection.detector import SimpleCNNDetector, get_model_status
from backend.audio.features import extract_acoustic_features
from backend.audio.config import default_config
from training.dataset import AntiSpoofDataset, resolve_audio_path
from training.preprocessing import preprocess_for_training
from training.metrics import calculate_metrics, calculate_eer


# ── Audio Transformation Functions ─────────────────────────────────────────────

def add_noise(y: np.ndarray, sr: int = 16000, snr_db: float = 10.0, seed: int = 42) -> np.ndarray:
    """Adds white Gaussian noise at a target Signal-to-Noise Ratio (SNR in dB)."""
    if len(y) == 0:
        return y
    signal_power = np.mean(y ** 2)
    if signal_power < 1e-10:
        return y
    
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    rng = np.random.RandomState(seed)
    noise = rng.normal(0.0, np.sqrt(noise_power), size=len(y)).astype(np.float32)
    
    noisy_y = y + noise
    # Peak normalize to prevent clipping distortion
    max_val = np.max(np.abs(noisy_y))
    if max_val > 1.0:
        noisy_y = noisy_y / max_val
    return noisy_y.astype(np.float32)


def resample_audio(y: np.ndarray, orig_sr: int = 16000, target_sr: int = 8000) -> np.ndarray:
    """Downsamples audio to target_sr and resamples back to orig_sr."""
    if len(y) == 0 or orig_sr == target_sr:
        return y
    
    # Resample to bottleneck rate
    downsampled = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
    # Resample back to model standard input rate
    resampled_back = librosa.resample(downsampled, orig_sr=target_sr, target_sr=orig_sr)
    return resampled_back.astype(np.float32)


def change_volume(y: np.ndarray, gain_db: float = 0.0) -> np.ndarray:
    """Adjusts volume by gain_db and clips to [-1.0, 1.0]."""
    if len(y) == 0 or gain_db == 0.0:
        return y
    factor = 10.0 ** (gain_db / 20.0)
    scaled = y * factor
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)


def truncate_audio(y: np.ndarray, sr: int = 16000, duration_seconds: float = 3.0) -> np.ndarray:
    """Crops audio to duration_seconds or zero-pads if shorter."""
    target_samples = int(duration_seconds * sr)
    if len(y) >= target_samples:
        return y[:target_samples].astype(np.float32)
    else:
        pad_len = target_samples - len(y)
        return np.pad(y, (0, pad_len), mode='constant').astype(np.float32)


def simulate_reverberation(y: np.ndarray, sr: int = 16000, decay: float = 5.0) -> np.ndarray:
    """Simulates mild room impulse decay (reverberation)."""
    if len(y) == 0:
        return y
    ir_len = int(sr * 0.1)  # 100ms impulse response
    t = np.linspace(0, 0.1, ir_len, endpoint=False)
    ir = np.exp(-decay * t) * np.random.RandomState(42).normal(0, 0.1, size=ir_len)
    ir[0] = 1.0  # Direct path
    
    reverb_y = np.convolve(y, ir, mode='full')[:len(y)]
    max_val = np.max(np.abs(reverb_y))
    if max_val > 0:
        reverb_y = reverb_y / max_val
    return reverb_y.astype(np.float32)


def apply_transformation(y: np.ndarray, sr: int, condition_name: str, seed: int = 42) -> np.ndarray:
    """Applies a specified transformation condition to audio array."""
    if condition_name == "clean":
        return y
    elif condition_name == "noise_20db":
        return add_noise(y, sr, snr_db=20.0, seed=seed)
    elif condition_name == "noise_10db":
        return add_noise(y, sr, snr_db=10.0, seed=seed)
    elif condition_name == "noise_5db":
        return add_noise(y, sr, snr_db=5.0, seed=seed)
    elif condition_name == "noise_0db":
        return add_noise(y, sr, snr_db=0.0, seed=seed)
    elif condition_name == "resample_8khz":
        return resample_audio(y, orig_sr=sr, target_sr=8000)
    elif condition_name == "resample_22khz":
        return resample_audio(y, orig_sr=sr, target_sr=22050)
    elif condition_name == "resample_44khz":
        return resample_audio(y, orig_sr=sr, target_sr=44100)
    elif condition_name == "volume_-12db":
        return change_volume(y, gain_db=-12.0)
    elif condition_name == "volume_-6db":
        return change_volume(y, gain_db=-6.0)
    elif condition_name == "volume_+6db":
        return change_volume(y, gain_db=6.0)
    elif condition_name == "short_audio_1s":
        return truncate_audio(y, sr, duration_seconds=1.0)
    elif condition_name == "short_audio_3s":
        return truncate_audio(y, sr, duration_seconds=3.0)
    elif condition_name == "reverberation_mild":
        return simulate_reverberation(y, sr, decay=5.0)
    elif condition_name == "combined_noise_volume":
        y_noisy = add_noise(y, sr, snr_db=10.0, seed=seed)
        return change_volume(y_noisy, gain_db=-6.0)
    elif condition_name == "combined_resample_noise":
        y_resampled = resample_audio(y, orig_sr=sr, target_sr=8000)
        return add_noise(y_resampled, sr, snr_db=10.0, seed=seed)
    else:
        raise ValueError(f"Unknown transformation condition: {condition_name}")


# ── Robustness Evaluation Engine ────────────────────────────────────────────────

def run_robustness_evaluation(args):
    print("=" * 70)
    print("  VigilVoice — Robustness & Generalization Evaluation Engine")
    print("=" * 70)

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    checkpoint_path = Path(args.checkpoint)

    # 1. Safety Guard Check: Model & Dataset Availability
    if not manifest_path.exists() or not checkpoint_path.exists():
        reason = []
        if not manifest_path.exists():
            reason.append(f"Manifest missing ({manifest_path})")
        if not checkpoint_path.exists():
            reason.append(f"Checkpoint missing ({checkpoint_path})")

        block_msg = "ROBUSTNESS EVALUATION BLOCKED — " + ", ".join(reason)
        print(f"\n[INFO] {block_msg}")
        print("Real dataset acquisition or model training required before robustness testing.")

        scorecard = {
            "status": "BLOCKED",
            "model_status": get_model_status(),
            "reason": ", ".join(reason),
            "baseline": None,
            "conditions": {}
        }
        with open(reports_dir / "robustness_report.json", "w") as f:
            json.dump(scorecard, f, indent=2)
        return scorecard

    # 2. Load Dataset (strictly TEST split)
    print(f"Loading test split from {manifest_path}...")
    try:
        test_dataset = AntiSpoofDataset(manifest_path, split="test", root_dir=project_root)
    except Exception as e:
        print(f"\n[INFO] ROBUSTNESS EVALUATION BLOCKED — {e}")
        scorecard = {
            "status": "BLOCKED",
            "model_status": get_model_status(),
            "reason": str(e),
            "baseline": None,
            "conditions": {}
        }
        with open(reports_dir / "robustness_report.json", "w") as f:
            json.dump(scorecard, f, indent=2)
        return scorecard

    print(f"Test samples: {len(test_dataset)}")

    # 3. Load Model
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = SimpleCNNDetector().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    conditions_to_test = [
        "clean",
        "noise_20db",
        "noise_10db",
        "noise_5db",
        "noise_0db",
        "resample_8khz",
        "resample_22khz",
        "resample_44khz",
        "volume_-12db",
        "volume_-6db",
        "volume_+6db",
        "short_audio_1s",
        "short_audio_3s",
        "reverberation_mild",
        "combined_noise_volume",
        "combined_resample_noise"
    ]

    results_by_condition = {}
    predictions_store = {}
    failure_cases = []

    print("\nBeginning Transformation Suite Evaluation...")

    for cond_idx, cond in enumerate(conditions_to_test, 1):
        all_scores = []
        all_labels = []

        for idx in range(len(test_dataset)):
            record = test_dataset.records[idx]
            audio_path = resolve_audio_path(record, project_root)
            
            try:
                y = preprocess_for_training(str(audio_path))
            except Exception:
                continue

            # Apply transformation
            y_trans = apply_transformation(y, default_config.sample_rate, cond, seed=args.seed + idx)

            # Feature extraction
            feat_dict = extract_acoustic_features(y_trans, sr=default_config.sample_rate)
            mfcc_seq = feat_dict.get("mfcc_sequence")
            if mfcc_seq is None:
                continue

            mfcc = np.array(mfcc_seq, dtype=np.float32)
            if mfcc.shape[1] < 64:
                mfcc = np.pad(mfcc, ((0, 0), (0, 64 - mfcc.shape[1])), mode='constant')
            else:
                mfcc = mfcc[:, :64]

            inp_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                out_prob = float(model(inp_tensor).squeeze().cpu().numpy())

            all_scores.append(out_prob)
            all_labels.append(record.label)

            # Failure Analysis Tracking (Clean vs Transformed)
            if cond != "clean" and "clean" in predictions_store:
                clean_prob = predictions_store["clean"]["scores"][idx]
                clean_pred = 1 if clean_prob >= 0.5 else 0
                trans_pred = 1 if out_prob >= 0.5 else 0
                true_label = record.label

                if clean_pred == true_label and trans_pred != true_label:
                    failure_cases.append({
                        "sample_idx": idx,
                        "filepath": record.filepath,
                        "speaker_id": record.speaker_id,
                        "attack_type": record.attack_type,
                        "true_label": "BONAFIDE" if true_label == 1 else "SPOOF",
                        "condition": cond,
                        "clean_score": clean_prob,
                        "transformed_score": out_prob
                    })

        labels_arr = np.array(all_labels)
        scores_arr = np.array(all_scores)

        predictions_store[cond] = {
            "scores": scores_arr,
            "labels": labels_arr
        }

        metrics = calculate_metrics(labels_arr, scores_arr, threshold=0.5)
        eer, eer_thresh = calculate_eer(labels_arr, scores_arr)

        cond_result = {
            "samples": len(labels_arr),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "far": metrics["far"],
            "frr": metrics["frr"],
            "roc_auc": metrics["roc_auc"],
            "eer": eer,
            "eer_threshold": eer_thresh
        }

        results_by_condition[cond] = cond_result
        print(f"[{cond_idx:2d}/{len(conditions_to_test)}] {cond:24s} | Acc: {metrics['accuracy']*100:5.1f}% | EER: {eer*100:5.1f}% | AUC: {metrics['roc_auc']:.3f}")

    # 4. Compute Degradation Relative to Baseline
    baseline = results_by_condition["clean"]
    degradation = {}

    for cond, res in results_by_condition.items():
        degradation[cond] = {
            "accuracy_delta": res["accuracy"] - baseline["accuracy"],
            "eer_delta": res["eer"] - baseline["eer"],
            "roc_auc_delta": res["roc_auc"] - baseline["roc_auc"],
            "far_delta": res["far"] - baseline["far"],
            "frr_delta": res["frr"] - baseline["frr"]
        }

    # 5. Threshold Analysis
    threshold_analysis = []
    clean_scores = predictions_store["clean"]["scores"]
    clean_labels = predictions_store["clean"]["labels"]

    for th in np.arange(0.1, 1.0, 0.1):
        th = float(np.round(th, 2))
        m = calculate_metrics(clean_labels, clean_scores, threshold=th)
        threshold_analysis.append({
            "threshold": th,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "far": m["far"],
            "frr": m["frr"]
        })

    # 6. Save Artifacts & Reports
    import hashlib
    sha256_hash = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    model_sha256 = sha256_hash.hexdigest()

    with open(reports_dir / "robustness_baseline.json", "w") as f:
        json.dump(baseline, f, indent=2)

    with open(reports_dir / "failure_analysis.json", "w") as f:
        json.dump(failure_cases[:50], f, indent=2)  # Cap at top 50 failures

    scorecard = {
        "status": "COMPLETED",
        "provenance": {
            "model_status": get_model_status(),
            "model_checkpoint": str(checkpoint_path),
            "model_sha256": model_sha256,
            "dataset": "ASVspoof_or_similar",
            "dataset_type": "REAL",
            "manifest": str(manifest_path),
            "split": "test",
            "sample_rate": default_config.sample_rate,
            "seed": args.seed,
            "timestamp": str(np.datetime64('now'))
        },
        "model_status": get_model_status(),
        "model_sha256": model_sha256,
        "dataset": "ASVspoof_or_similar",
        "split": "test",
        "test_samples": len(clean_labels),
        "baseline": baseline,
        "conditions": results_by_condition,
        "degradation": degradation,
        "threshold_analysis": threshold_analysis,
        "eer_recommended_threshold": baseline["eer_threshold"]
    }

    with open(reports_dir / "robustness_report.json", "w") as f:
        json.dump(scorecard, f, indent=2)

    # 7. Visualizations Generation
    generate_robustness_plots(results_by_condition, predictions_store, reports_dir)

    print("\n" + "=" * 70)
    print(f"Robustness Evaluation Complete. Reports saved to '{reports_dir}'")
    print("=" * 70)
    return scorecard


# ── Plot Generation ────────────────────────────────────────────────────────────

def generate_robustness_plots(results: dict, predictions_store: dict, reports_dir: Path):
    cond_names = list(results.keys())
    accuracies = [results[c]["accuracy"] * 100 for c in cond_names]
    eers = [results[c]["eer"] * 100 for c in cond_names]
    aucs = [results[c]["roc_auc"] for c in cond_names]

    plt.style.use('dark_background')

    # 1. Robustness Accuracy Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(cond_names, accuracies, color='#06b6d4')
    bars[0].set_color('#10b981')  # Clean baseline highlighted in emerald
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('VigilVoice Model Accuracy Under Audio Transformations')
    ax.set_xlim(0, 105)
    plt.tight_layout()
    plt.savefig(reports_dir / "robustness_accuracy.png", dpi=150)
    plt.close()

    # 2. Robustness EER Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(cond_names, eers, color='#e11d48')
    bars[0].set_color('#10b981')
    ax.set_xlabel('Equal Error Rate (EER %)')
    ax.set_title('VigilVoice Equal Error Rate (EER) Under Audio Transformations')
    plt.tight_layout()
    plt.savefig(reports_dir / "robustness_eer.png", dpi=150)
    plt.close()

    # 3. Robustness ROC-AUC Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(cond_names, aucs, color='#8b5cf6')
    bars[0].set_color('#10b981')
    ax.set_xlabel('ROC-AUC')
    ax.set_title('VigilVoice ROC-AUC Under Audio Transformations')
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(reports_dir / "robustness_auc.png", dpi=150)
    plt.close()

    # 4. Confidence Distribution (Clean vs Noise 10dB)
    if "clean" in predictions_store and "noise_10db" in predictions_store:
        fig, ax = plt.subplots(figsize=(8, 5))

        clean_labels = predictions_store["clean"]["labels"]
        clean_scores = predictions_store["clean"]["scores"]
        noise_scores = predictions_store["noise_10db"]["scores"]

        bonafide_clean = clean_scores[clean_labels == 1]
        spoof_clean = clean_scores[clean_labels == 0]
        bonafide_noise = noise_scores[clean_labels == 1]
        spoof_noise = noise_scores[clean_labels == 0]

        if len(bonafide_clean) > 0:
            ax.hist(bonafide_clean, bins=15, alpha=0.5, label='Bonafide (Clean)', color='#10b981')
        if len(spoof_clean) > 0:
            ax.hist(spoof_clean, bins=15, alpha=0.5, label='Spoof (Clean)', color='#e11d48')
        if len(bonafide_noise) > 0:
            ax.hist(bonafide_noise, bins=15, alpha=0.3, label='Bonafide (Noise 10dB)', color='#06b6d4', linestyle='--')
        if len(spoof_noise) > 0:
            ax.hist(spoof_noise, bins=15, alpha=0.3, label='Spoof (Noise 10dB)', color='#f59e0b', linestyle='--')

        ax.set_xlabel('Predicted Probability (REAL)')
        ax.set_ylabel('Sample Count')
        ax.set_title('Prediction Score Distributions (Clean vs Noise 10dB)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(reports_dir / "confidence_distribution.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VigilVoice Robustness Evaluation")
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/manifest.csv"), help="Path to test split manifest")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/production/vigilvoice_cnn_best.pth"), help="Path to trained .pth model")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"), help="Output directory for robustness reports & plots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible transformations")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cuda/cpu)")

    args = parser.parse_args()
    run_robustness_evaluation(args)
