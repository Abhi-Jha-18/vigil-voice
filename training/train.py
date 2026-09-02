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
        train_dataset = AntiSpoofDataset(args.manifest, split="train", root_dir=Path(project_root))
        val_dataset = AntiSpoofDataset(args.manifest, split="validation", root_dir=Path(project_root))
    except Exception as e:
        print(f"\n[ERROR] Dataset initialization failed: {e}")
        print("REAL DATASET REQUIRED — TRAINING NOT YET EXECUTED.")
        sys.exit(1)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("\n[ERROR] Dataset splits are empty or manifest not found.")
        print("REAL DATASET REQUIRED — TRAINING NOT YET EXECUTED.")
        sys.exit(1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    num_real = sum(1 for r in train_dataset.records if r.label == 1)
    num_fake = sum(1 for r in train_dataset.records if r.label == 0)
    
    if num_real == 0 or num_fake == 0:
        print("\nINSUFFICIENT REAL DATASET\nTraining aborted.")
        sys.exit(1)
        
    total_samples = num_real + num_fake
    real_weight = total_samples / (2.0 * max(1, num_real))
    fake_weight = total_samples / (2.0 * max(1, num_fake))

    dataset_sources = list({r.dataset_source for r in train_dataset.records if r.dataset_source})
    metadata["dataset_identifier"] = dataset_sources[0] if dataset_sources else "GenericDataset"

    print(f"Train samples: {len(train_dataset)} (REAL: {num_real}, FAKE: {num_fake}) | Val samples: {len(val_dataset)}")
    print(f"Class Weights - REAL: {real_weight:.2f}, FAKE: {fake_weight:.2f}")

    # 5. Model & Optimizer
    model = SimpleCNNDetector().to(device)
    pos_weight = torch.tensor([num_fake / max(1, num_real)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # 6. Training Loop
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_loss = float('inf')
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
            if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze(1)
                if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
                
                # Use BCEWithLogitsLoss for validation loss tracking
                val_loss_batch = nn.BCEWithLogitsLoss()(outputs, labels)
                val_loss += val_loss_batch.item() * inputs.size(0)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())
                
        val_loss /= len(val_dataset)
        metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
        
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(metrics['accuracy'])
        history["val_f1"].append(metrics['f1'])

        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            status = " [New Best]"
            # Save checkpoint
            torch.save(model.state_dict(), best_model_path)
            metadata["best_epoch"] = epoch + 1
            metadata["best_metrics"] = metrics
            metadata["model_status"] = "REAL_MODEL"
            metadata["label_mapping"] = {"bonafide": 1, "spoof": 0}
            import hashlib
            sha = hashlib.sha256()
            with open(best_model_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha.update(chunk)
            metadata["model_sha256"] = sha.hexdigest()
            with open(best_model_path.with_name("vigilvoice_cnn_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch+1:02d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {metrics['accuracy']*100:.1f}%{status}")

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # 7. Save Artifacts
    with open(run_dir / "config.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
        
    print("=" * 60)
    print("  REAL TRAINING COMPLETED")
    print(f"  Best Validation Metrics: {metadata.get('best_metrics', {})}")
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
    
    args = parser.parse_args()
    train_model(args)
