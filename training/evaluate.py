"""
VigilVoice — Production Evaluation Pipeline (Phase 11)
======================================================
Evaluates a trained model strictly on the independent 'test' split.

Usage:
    python training/evaluate.py \
        --manifest data/splits/manifest.csv \
        --checkpoint models/production/vigilvoice_cnn_best.pth \
        --output training/runs/evaluation_report.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.detection.detector import SimpleCNNDetector
from training.dataset import AntiSpoofDataset
from training.metrics import calculate_metrics, calculate_eer

def evaluate_model(args):
    print("=" * 60)
    print("  VigilVoice — Production CNN Evaluation Pipeline")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # 1. Load Dataset (strictly TEST split)
    print(f"Loading test split from {args.manifest}...")
    try:
        test_dataset = AntiSpoofDataset(args.manifest, split="test", root_dir=Path(project_root))
    except Exception as e:
        print(f"\n[ERROR] Dataset initialization failed: {e}")
        print("REAL DATASET REQUIRED — EVALUATION NOT YET EXECUTED.")
        sys.exit(1)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")

    # 2. Load Model
    model = SimpleCNNDetector().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # 3. Evaluate
    all_preds = []
    all_labels = []
    all_attack_types = []
    
    with torch.no_grad():
        for inputs, labels, metas in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze(1)
            if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
            outputs = torch.sigmoid(outputs)
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Extract attack types from metadata dictionary
            if "attack_type" in metas:
                all_attack_types.extend(metas["attack_type"])
            else:
                all_attack_types.extend(["unknown"] * inputs.size(0))
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_attack_types = np.array(all_attack_types)

    # 4. Metrics
    metrics = calculate_metrics(all_labels, all_preds)
    
    # EER expects positive class scores.
    # In VigilVoice, 1 = REAL, 0 = FAKE.
    eer, eer_threshold = calculate_eer(all_labels, all_preds)
    
    report = {
        "dataset": "ASVspoof_or_similar",
        "dataset_type": "REAL",
        "model_status": "REAL_MODEL",
        "split": "test",
        "samples": len(test_dataset),
        "accuracy": metrics['accuracy'],
        "precision": metrics['precision'],
        "recall": metrics['recall'],
        "f1": metrics['f1'],
        "roc_auc": metrics['roc_auc'],
        "eer": eer,
        "far": metrics['far'],
        "frr": metrics['frr'],
        "eer_threshold": eer_threshold,
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint)
    }

    print("\n--- Evaluation Results ---")
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"EER:       {eer*100:.2f}% (at threshold {eer_threshold:.4f})")
    print(f"FAR:       {metrics['far']*100:.2f}%")
    print(f"FRR:       {metrics['frr']*100:.2f}%")
    
    # 5. Save Report & Artifacts
    if args.output:
        out_path = Path(args.output)
        reports_dir = out_path.parent
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {out_path}")
        
        # 6. Generate Confusion Matrix
        y_pred_classes = (all_preds >= 0.5).astype(int)
        cm = confusion_matrix(all_labels, y_pred_classes)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['SPOOF (0)', 'BONAFIDE (1)'])
        plt.yticks(tick_marks, ['SPOOF (0)', 'BONAFIDE (1)'])
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), horizontalalignment="center",
                         color="white" if cm[i, j] > cm.max() / 2. else "black")
        plt.tight_layout()
        cm_path = reports_dir / "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion matrix saved to: {cm_path}")
        
        # 7. Generate ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(all_labels, all_preds, pos_label=1)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        roc_path = reports_dir / "roc_curve.png"
        plt.savefig(roc_path)
        plt.close()
        print(f"ROC curve saved to: {roc_path}")
        
        # 8. Attack-Type Analysis
        attack_results = {}
        unique_attacks = np.unique(all_attack_types)
        if len(unique_attacks) > 0 and "unknown" not in unique_attacks:
            print("\n--- Attack-Type Analysis ---")
            for attack in unique_attacks:
                mask = (all_attack_types == attack)
                y_true_a = all_labels[mask]
                y_pred_a = all_preds[mask]
                
                # Accuracy
                y_pred_classes_a = (y_pred_a >= 0.5).astype(int)
                acc = float(np.mean(y_true_a == y_pred_classes_a))
                
                # EER (Need positive and negative classes for ROC, if attack is only SPOOF, EER against all BONAFIDE)
                # To be robust, if it's a SPOOF attack, we combine it with ALL BONAFIDE for EER calculation
                if np.all(y_true_a == 0):
                    bonafide_mask = (all_labels == 1)
                    y_true_combined = np.concatenate([y_true_a, all_labels[bonafide_mask]])
                    y_pred_combined = np.concatenate([y_pred_a, all_preds[bonafide_mask]])
                else:
                    y_true_combined = y_true_a
                    y_pred_combined = y_pred_a
                    
                try:
                    att_eer, _ = calculate_eer(y_true_combined, y_pred_combined)
                except Exception:
                    att_eer = -1.0
                    
                attack_results[attack] = {
                    "samples": int(np.sum(mask)),
                    "accuracy": acc,
                    "eer": att_eer
                }
                print(f"  {attack:12s} | Samples: {np.sum(mask):4d} | Acc: {acc*100:5.1f}% | EER: {att_eer*100:5.1f}%")
            
            attack_out = reports_dir / "attack_type_results.json"
            with open(attack_out, "w") as f:
                json.dump(attack_results, f, indent=2)
            print(f"\nAttack-type analysis saved to: {attack_out}")
        else:
            print("\nAttack-type evaluation unavailable for this dataset.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VigilVoice Evaluation")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to split manifest CSV")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained .pth checkpoint")
    parser.add_argument("--output", type=Path, help="Output JSON report path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    evaluate_model(args)
