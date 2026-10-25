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
from backend.detection.detector import (
    blend_cnn_and_heuristic,
    checkpoint_temperature,
    validate_checkpoint_feature_version,
)
from backend.audio.config import default_config
from training.dataset import AntiSpoofDataset, load_metadata
from training.metrics import calculate_metrics, calculate_eer

def evaluate_model(args):
    print("=" * 60)
    print("  VigilVoice — Production CNN Evaluation Pipeline")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    feature_version = getattr(args, "feature_version", default_config.feature_version)
    validate_checkpoint_feature_version(args.checkpoint, feature_version)
    temperature = checkpoint_temperature(args.checkpoint)

    # 1. Load Dataset (TEST and unseen attack splits)
    print(f"Loading evaluation split from {args.manifest}...")
    try:
        all_manifest_records = load_metadata(args.manifest)
    except Exception:
        all_manifest_records = []

    train_generators = {
        r.generator_id for r in all_manifest_records
        if r.split.lower() == "train" and r.label == 0 and r.generator_id and r.generator_id.lower() not in {"none", "unknown", ""}
    }

    manifest_splits = {r.split.lower() for r in all_manifest_records}
    eval_split = getattr(args, "split", "test")
    if eval_split == "test":
        unseen_matches = [s for s in ("test_unseen", "unseen", "unseen_attack") if s in manifest_splits]
        if unseen_matches:
            eval_split = ["test"] + unseen_matches

    try:
        test_dataset = AntiSpoofDataset(
            args.manifest,
            split=eval_split,
            root_dir=Path(project_root),
            allow_synthetic=getattr(args, "allow_synthetic", False),
            cache_features=False,
            feature_version=feature_version,
        )
    except Exception as e:
        print(f"\n[ERROR] Dataset initialization failed: {e}")
        print("REAL DATASET REQUIRED — EVALUATION NOT YET EXECUTED.")
        sys.exit(1)

    is_synthetic = bool(getattr(args, "allow_synthetic", False)) or any(
        any(marker in (value or "").lower() for marker in ("demo", "fixture", "mock", "synthetic"))
        for record in test_dataset.records
        for value in (record.dataset_source, record.filepath)
    )
    if is_synthetic:
        report = {
            "status": "BLOCKED",
            "dataset_type": "SYNTHETIC",
            "is_synthetic": True,
            "reason": "Evaluation metrics are not reported for demo or synthetic data.",
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
        }
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("[BLOCKED] Evaluation metrics are not reported for demo or synthetic data.")
        return report

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")

    # 2. Load Model
    model = SimpleCNNDetector(feature_version=feature_version).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    # 3. Evaluate
    all_logits = []
    all_labels = []
    all_attack_types = []
    all_generator_ids = []
    all_splits = []
    all_heuristic_scores = []
    
    with torch.no_grad():
        for inputs, labels, metas in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze(1)
            
            all_logits.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            heuristic_batch = metas.get("heuristic_score")
            if heuristic_batch is None:
                all_heuristic_scores.extend([0.5] * inputs.size(0))
            elif torch.is_tensor(heuristic_batch):
                all_heuristic_scores.extend(heuristic_batch.cpu().numpy())
            else:
                all_heuristic_scores.extend(heuristic_batch)
            
            # Extract attack types from metadata dictionary
            if "attack_type" in metas:
                all_attack_types.extend(metas["attack_type"])
            else:
                all_attack_types.extend(["unknown"] * inputs.size(0))

            if "generator_id" in metas:
                all_generator_ids.extend(metas["generator_id"])
            else:
                all_generator_ids.extend(["unknown"] * inputs.size(0))

            if "split" in metas:
                all_splits.extend(metas["split"])
            else:
                all_splits.extend(["test"] * inputs.size(0))
            
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    # Probabilities clipped only for display
    calibrated_logits = all_logits / temperature
    all_preds = 1.0 / (1.0 + np.exp(-calibrated_logits))
    blended_preds = blend_cnn_and_heuristic(
        all_preds, np.asarray(all_heuristic_scores, dtype=np.float64)
    )
    all_attack_types = np.array(all_attack_types)
    all_generator_ids = np.array(all_generator_ids)
    all_splits = np.array(all_splits)

    # 4. Metrics (computed on raw logits; clip only for display)
    metrics = calculate_metrics(all_labels, all_preds, raw_logits=calibrated_logits)
    blended_metrics = calculate_metrics(all_labels, blended_preds)
    eer = metrics["eer"]
    eer_threshold = metrics["eer_threshold"]

    # Seen vs Unseen Attack Analysis
    bonafide_mask = (all_labels == 1)
    seen_spoof_mask = np.zeros(len(all_labels), dtype=bool)
    unseen_spoof_mask = np.zeros(len(all_labels), dtype=bool)

    for i in range(len(all_labels)):
        if all_labels[i] == 0:
            gen = all_generator_ids[i]
            sp = str(all_splits[i]).lower()
            if sp in ("test_unseen", "unseen", "unseen_attack") or (train_generators and gen not in train_generators):
                unseen_spoof_mask[i] = True
            else:
                seen_spoof_mask[i] = True

    def _calc_subset_metrics(spoof_mask):
        count = int(np.sum(spoof_mask))
        if count == 0:
            return None
        y_true_s = all_labels[spoof_mask]
        y_pred_s = all_preds[spoof_mask]
        acc = float(np.mean((y_pred_s >= 0.5).astype(int) == y_true_s))
        if np.sum(bonafide_mask) > 0:
            y_comb = np.concatenate([y_true_s, all_labels[bonafide_mask]])
            p_comb = np.concatenate([y_pred_s, all_preds[bonafide_mask]])
            try:
                sub_eer, _ = calculate_eer(y_comb, p_comb)
            except Exception:
                sub_eer = -1.0
        else:
            sub_eer = -1.0
        return {"samples": count, "accuracy": acc, "eer": sub_eer}

    seen_attack_metrics = _calc_subset_metrics(seen_spoof_mask)
    unseen_attack_metrics = _calc_subset_metrics(unseen_spoof_mask)

    # Check if dataset contains demo/fixture records
    is_synthetic = False
    
    report = {
        "dataset": "synthetic_local" if is_synthetic else "ASVspoof_or_similar",
        "dataset_type": "SYNTHETIC" if is_synthetic else "REAL",
        "is_synthetic": is_synthetic,
        "model_status": "DEMO_MODEL" if is_synthetic else "REAL_MODEL",
        "split": str(eval_split),
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
        "checkpoint": str(args.checkpoint),
        "feature_version": feature_version,
        "temperature": temperature,
        "model_comparison": {
            "cnn_only": metrics,
            "blended": blended_metrics,
        },
        "cross_dataset": {
            "status": "NOT_AVAILABLE",
            "reason": "No second real dataset manifest was supplied.",
        },
    }
    if seen_attack_metrics:
        report["seen_attacks"] = seen_attack_metrics
    if unseen_attack_metrics:
        report["unseen_attacks"] = unseen_attack_metrics

    if is_synthetic:
        report["performance_disclaimer"] = (
            "Metrics computed on synthetic/demo fixtures; do NOT report as real performance."
        )

    print("\n--- Evaluation Results ---")
    if is_synthetic:
        print("[NOTICE] DEMO / SYNTHETIC DATA: Metrics below reflect synthetic demo validation only, NOT real-world performance.")
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"EER:       {eer*100:.2f}% (at threshold {eer_threshold:.4f})")
    print(f"FAR:       {metrics['far']*100:.2f}%")
    print(f"FRR:       {metrics['frr']*100:.2f}%")
    print("\n--- CNN-only vs Blended ---")
    print(
        f"CNN-only | Accuracy: {metrics['accuracy']*100:.2f}% | "
        f"EER: {metrics['eer']*100:.2f}% | AUC: {metrics['roc_auc']:.4f}"
    )
    print(
        f"Blended  | Accuracy: {blended_metrics['accuracy']*100:.2f}% | "
        f"EER: {blended_metrics['eer']*100:.2f}% | AUC: {blended_metrics['roc_auc']:.4f}"
    )

    print("\n--- Seen vs Unseen Attack Analysis ---")
    if seen_attack_metrics:
        print(f"  seen_attacks   | Samples: {seen_attack_metrics['samples']:4d} | Acc: {seen_attack_metrics['accuracy']*100:5.1f}% | EER: {seen_attack_metrics['eer']*100:5.1f}%")
    else:
        print("  seen_attacks   | None present")
    if unseen_attack_metrics:
        print(f"  unseen_attacks | Samples: {unseen_attack_metrics['samples']:4d} | Acc: {unseen_attack_metrics['accuracy']*100:5.1f}% | EER: {unseen_attack_metrics['eer']*100:5.1f}%")
    else:
        print("  unseen_attacks | None present")

    # Optional deployment if passing --max-eer threshold
    if getattr(args, "deploy", False):
        max_eer = getattr(args, "max_eer", 0.05)
        if is_synthetic:
            print("\n[Deploy Blocked] Checkpoint is marked synthetic/demo; refusing deployment to production weights.")
        elif eer <= max_eer:
            import shutil
            deployed_path = Path(project_root) / "models" / "cnn_weight.pth"
            shutil.copy2(args.checkpoint, deployed_path)
            meta_src = Path(args.checkpoint).with_name(Path(args.checkpoint).stem + "_metadata.json")
            if meta_src.is_file():
                shutil.copy2(meta_src, deployed_path.with_name("cnn_weight_metadata.json"))
            print(f"\n[Deploy Success] EER {eer:.4f} <= threshold {max_eer:.4f}. Model deployed to {deployed_path}.")
        else:
            print(f"\n[Deploy Aborted] EER {eer:.4f} exceeds max-eer threshold {max_eer:.4f}. Model not deployed.")
    
    # 5. Save Report & Artifacts
    if args.output:
        out_path = Path(args.output)
        reports_dir = out_path.parent
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {out_path}")

        if seen_attack_metrics or unseen_attack_metrics:
            seen_unseen_path = reports_dir / "seen_vs_unseen_results.json"
            with open(seen_unseen_path, "w") as f:
                json.dump({
                    "seen_attacks": seen_attack_metrics,
                    "unseen_attacks": unseen_attack_metrics,
                }, f, indent=2)
            print(f"Seen vs unseen attack results saved to: {seen_unseen_path}")
        
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
                    bonafide_mask_att = (all_labels == 1)
                    y_true_combined = np.concatenate([y_true_a, all_labels[bonafide_mask_att]])
                    y_pred_combined = np.concatenate([y_pred_a, all_preds[bonafide_mask_att]])
                elif np.all(y_true_a == 1):
                    y_true_combined = np.array([], dtype=all_labels.dtype)
                    y_pred_combined = np.array([], dtype=all_preds.dtype)
                else:
                    y_true_combined = y_true_a
                    y_pred_combined = y_pred_a
                    
                if len(y_true_combined) == 0:
                    att_eer = None
                else:
                    try:
                        att_eer, _ = calculate_eer(y_true_combined, y_pred_combined)
                    except ValueError:
                        att_eer = None
                    
                attack_results[attack] = {
                    "samples": int(np.sum(mask)),
                    "accuracy": acc,
                    "eer": att_eer
                }
                eer_text = f"{att_eer*100:5.1f}%" if att_eer is not None else "  n/a "
                print(f"  {attack:12s} | Samples: {np.sum(mask):4d} | Acc: {acc*100:5.1f}% | EER: {eer_text}")
            
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
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate (default: 'test')")
    parser.add_argument(
        "--feature-version",
        choices=("mfcc_v1", "logmel_delta_v1"),
        default=default_config.feature_version,
        help="Must match the feature_version recorded by the checkpoint.",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Load demo fixtures for diagnostics; performance metrics remain blocked.",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy checkpoint to models/cnn_weight.pth if evaluation passes max-eer threshold.",
    )
    parser.add_argument(
        "--max-eer",
        type=float,
        default=0.05,
        help="Maximum allowed EER to qualify for deployment (default: 0.05).",
    )

    args = parser.parse_args()
    evaluate_model(args)
