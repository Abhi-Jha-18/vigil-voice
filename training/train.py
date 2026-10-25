"""
VigilVoice — Production Training Pipeline (Phase 11)
===================================================
Trains a SimpleCNNDetector on MFCC features extracted from a real anti-spoofing corpus.

Usage:
    python training/train.py \
        --manifest data/splits/manifest.csv \
        --output models/production \
        --epochs 20 \
        --batch-size 16 \
        --learning-rate 0.001 \
        --seed 42
"""

import os
import sys
import json
import random
import argparse
import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Make sure imports work from project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.detection.detector import SimpleCNNDetector
from backend.audio.config import default_config
from training.dataset import AntiSpoofDataset
from training.metrics import calculate_metrics

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def fit_temperature_scaling(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Fits a single temperature scalar T > 0 on validation set to minimize NLL.
    """
    z = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    
    log_temp = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = optim.LBFGS([log_temp], lr=0.05, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(z / torch.exp(log_temp), y)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        T = float(torch.exp(log_temp).item())
        return float(np.clip(T, 0.1, 10.0))
    except Exception:
        return 1.0


def choose_thresholds_from_roc(y_true: np.ndarray, calibrated_probs: np.ndarray, target_far: float = 0.05) -> tuple[float, float]:
    """
    Chooses REAL and FAKE decision thresholds from validation ROC.
    - threshold_real: threshold at target_far (FAR <= target_far).
    - threshold_fake: threshold at target_frr (FRR <= target_far).
    Between threshold_fake and threshold_real is the UNCERTAIN zone.
    """
    from sklearn.metrics import roc_curve
    try:
        fpr, tpr, thresholds = roc_curve(y_true, calibrated_probs, pos_label=1)
        
        valid_far_idx = np.where(fpr <= target_far)[0]
        if len(valid_far_idx) > 0:
            best_far_idx = valid_far_idx[-1]
            threshold_real = float(thresholds[best_far_idx])
        else:
            threshold_real = 0.75

        fnr = 1.0 - tpr
        valid_frr_idx = np.where(fnr <= target_far)[0]
        if len(valid_frr_idx) > 0:
            best_frr_idx = valid_frr_idx[0]
            threshold_fake = float(thresholds[best_frr_idx])
        else:
            threshold_fake = 0.40

        threshold_real = float(np.clip(threshold_real, 0.50, 0.95))
        threshold_fake = float(np.clip(threshold_fake, 0.05, 0.50))
        if threshold_fake >= threshold_real:
            threshold_fake = max(0.05, threshold_real - 0.20)

        return threshold_real, threshold_fake
    except Exception:
        return 0.75, 0.40


def train_model(args):
    print("=" * 60)
    print("  VigilVoice — Production CNN Training Pipeline")
    print("=" * 60)

    # 1. Setup & Reproducibility
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Device: {device} | Seed: {args.seed}")

    # 2. Artifact Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(project_root) / "training" / "runs" / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Output Checkpoint Path
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path(project_root) / output_dir
    best_model_path = output_dir / "vigilvoice_cnn_best.pth"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "timestamp": timestamp,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "manifest_path": str(args.manifest),
        "preprocessing_config": default_config.to_dict(),
        "model_architecture": "SimpleCNNDetector",
        "dataset_identifier": "unknown"
    }

    # 4. Load Datasets
    print(f"Loading training data from {args.manifest}...")
    try:
        train_dataset = AntiSpoofDataset(
            args.manifest,
            split="train",
            root_dir=Path(project_root),
            allow_synthetic=args.allow_synthetic,
            cache_features=True,
            augment=True,
        )
        val_dataset = AntiSpoofDataset(
            args.manifest,
            split="validation",
            root_dir=Path(project_root),
            allow_synthetic=args.allow_synthetic,
            cache_features=True,
            augment=False,
        )
    except Exception as e:
        print(f"\n[ERROR] Dataset initialization failed: {e}")
        print("REAL DATASET REQUIRED — TRAINING NOT YET EXECUTED.")
        sys.exit(1)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("\n[ERROR] Dataset splits are empty or manifest not found.")
        print("REAL DATASET REQUIRED — TRAINING NOT YET EXECUTED.")
        sys.exit(1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_real = sum(1 for r in train_dataset.records if r.label == 1)
    num_fake = sum(1 for r in train_dataset.records if r.label == 0)
    
    if num_real == 0 or num_fake == 0:
        print("\nINSUFFICIENT REAL DATASET\nTraining aborted.")
        sys.exit(1)

    dataset_sources = sorted(list({r.dataset_source for r in train_dataset.records if r.dataset_source}))
    metadata["dataset_sources"] = dataset_sources
    metadata["dataset_identifier"] = "+".join(dataset_sources) if dataset_sources else "GenericDataset"

    # Identify synthetic or demo/fixture data
    is_demo_fixture = any(
        any(k in (r.dataset_source or "").lower() or k in (r.filepath or "").lower() for k in ("demo", "fixture", "mock", "synthetic"))
        for r in train_dataset.records
    )
    is_synthetic = bool(args.allow_synthetic or is_demo_fixture)
    metadata["is_synthetic"] = is_synthetic
    metadata["model_status"] = "DEMO_MODEL" if is_synthetic else "REAL_MODEL"
    if is_synthetic:
        metadata["performance_disclaimer"] = "Metrics computed on synthetic/demo fixtures; do NOT report as real performance."

    print(f"Train samples: {len(train_dataset)} (REAL: {num_real}, FAKE: {num_fake}) | Val samples: {len(val_dataset)}")
    print(f"Dataset sources: {dataset_sources} | is_synthetic: {is_synthetic}")

    # 5. Model & Optimizer
    model = SimpleCNNDetector().to(device)
    pos_weight = torch.tensor([num_fake / max(1, num_real)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    unweighted_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # 6. Training Loop (select best checkpoint & early-stop on validation EER)
    history = {"train_loss": [], "val_loss": [], "val_unweighted_loss": [], "val_acc": [], "val_eer": [], "val_f1": []}
    best_val_eer = float('inf')
    early_stop_patience = 5
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_dataset)

        # Validation (using same criterion with pos_weight, plus tracking unweighted loss)
        model.eval()
        val_loss = 0.0
        val_unweighted_loss = 0.0
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze(1)
                
                loss_batch = criterion(outputs, labels)
                val_loss += loss_batch.item() * inputs.size(0)
                unweighted_batch = unweighted_criterion(outputs, labels)
                val_unweighted_loss += unweighted_batch.item() * inputs.size(0)

                all_logits.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss /= len(val_dataset)
        val_unweighted_loss /= len(val_dataset)

        all_logits_arr = np.array(all_logits)
        all_labels_arr = np.array(all_labels)
        all_probs_arr = 1.0 / (1.0 + np.exp(-all_logits_arr))

        # Compute EER/AUC on raw logits; clip probabilities only for display
        metrics = calculate_metrics(all_labels_arr, all_probs_arr, raw_logits=all_logits_arr)
        val_eer = metrics["eer"]
        
        # Step scheduler on validation EER
        scheduler.step(val_eer)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_unweighted_loss"].append(val_unweighted_loss)
        history["val_acc"].append(metrics['accuracy'])
        history["val_eer"].append(val_eer)
        history["val_f1"].append(metrics['f1'])

        status = ""
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            epochs_no_improve = 0
            status = f" [New Best EER: {val_eer*100:.2f}%]"

            # Fit temperature scaling and ROC thresholds on validation set
            temperature = fit_temperature_scaling(all_logits_arr, all_labels_arr)
            calibrated_logits = all_logits_arr / max(0.01, temperature)
            calibrated_probs = 1.0 / (1.0 + np.exp(-calibrated_logits))
            threshold_real, threshold_fake = choose_thresholds_from_roc(all_labels_arr, calibrated_probs, target_far=0.05)

            # Save checkpoint
            torch.save(model.state_dict(), best_model_path)
            metadata["best_epoch"] = epoch + 1
            metadata["best_metrics"] = metrics
            metadata["temperature"] = float(round(temperature, 4))
            metadata["threshold_real"] = float(round(threshold_real, 4))
            metadata["threshold_fake"] = float(round(threshold_fake, 4))
            metadata["calibration"] = {
                "temperature": float(round(temperature, 4)),
                "target_far": 0.05,
                "threshold_real": float(round(threshold_real, 4)),
                "threshold_fake": float(round(threshold_fake, 4)),
            }
            metadata["label_mapping"] = {"bonafide": 1, "spoof": 0}

            import hashlib
            sha = hashlib.sha256()
            with open(best_model_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha.update(chunk)
            metadata["model_sha256"] = sha.hexdigest()
            meta_path = best_model_path.with_name(best_model_path.stem + "_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch+1:02d}/{args.epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} (unweighted: {val_unweighted_loss:.4f}) | "
            f"Val EER: {val_eer*100:.2f}% | Val Acc: {metrics['accuracy']*100:.1f}%{status}"
        )

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs (no EER improvement for {early_stop_patience} epochs).")
            break

    # 7. Save Run Artifacts
    with open(run_dir / "config.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # 8. Deployment: Only deploy when --deploy flag is provided AND evaluate passes --max-eer threshold
    if getattr(args, "deploy", False):
        max_eer = getattr(args, "max_eer", 0.05)
        if is_synthetic:
            print("\n[Deploy Blocked] Checkpoint is from synthetic/demo data; refusing deployment to production.")
        else:
            try:
                from training.evaluate import evaluate_model
                eval_args = argparse.Namespace(
                    manifest=args.manifest,
                    checkpoint=best_model_path,
                    output=run_dir / "deployment_eval_report.json",
                    batch_size=args.batch_size,
                    device=args.device,
                    allow_synthetic=args.allow_synthetic,
                    deploy=True,
                    max_eer=max_eer,
                )
                print("\nRunning test evaluation for deployment qualification...")
                evaluate_model(eval_args)
            except Exception as exc:
                print(f"[Deploy Error] Evaluation during deploy failed: {exc}")

    print("=" * 60)
    if is_synthetic:
        print("  [DEMO / SYNTHETIC] TRAINING COMPLETED")
        print("  NOTE: Metrics computed on synthetic/demo fixtures do NOT reflect real-world performance.")
    else:
        print("  REAL TRAINING COMPLETED")
    print(f"  Best Validation EER: {best_val_eer*100:.2f}%")
    print(f"  Best Validation Metrics: {metadata.get('best_metrics', {})}")
    print(f"  Calibrated Temperature: {metadata.get('temperature', 1.0)}")
    print(f"  Calibrated Thresholds: REAL >= {metadata.get('threshold_real')}, FAKE <= {metadata.get('threshold_fake')}")
    print(f"  Model Checkpoint: {best_model_path}")
    print(f"  Run Artifacts: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VigilVoice Production CNN Training")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to split manifest CSV")
    parser.add_argument("--output", type=Path, default=Path("models/production"), help="Output directory for best model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow training on DEMO/synthetic fixtures when a licensed corpus is unavailable.",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy best model to models/cnn_weight.pth only if evaluation passes max-eer threshold.",
    )
    parser.add_argument(
        "--max-eer",
        type=float,
        default=0.05,
        help="Maximum validation EER allowed for deployment (default: 0.05).",
    )

    args = parser.parse_args()
    train_model(args)
