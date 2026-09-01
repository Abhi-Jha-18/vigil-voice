"""
Command-line pipeline for dataset validation and splitting.
Phase 12: Advanced Reporting and Validation.
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Fix python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from training.dataset import validate_dataset, dataset_statistics
from training.splits import make_speaker_disjoint_splits, split_integrity_issues, write_metadata

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset metadata by validating and creating train/val/test splits.")
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata.csv"), help="Path to input metadata CSV")
    parser.add_argument("--root", type=Path, default=Path("data"), help="Root directory for dataset files")
    parser.add_argument("--output", type=Path, default=Path("data/splits/manifest.csv"), help="Path to output split manifest")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"), help="Directory to save JSON reports")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    
    args = parser.parse_args()
    
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Validating dataset from {args.metadata}...")
    try:
        report = validate_dataset(args.metadata, args.root)
    except FileNotFoundError:
        print(f"\n[ERROR] REAL DATASET REQUIRED: The metadata file '{args.metadata}' is missing.")
        print("Please manually acquire a valid anti-spoofing corpus (e.g., ASVspoof) and configure it.")
        sys.exit(1)
    
    # Save audio validation report
    audio_val_path = args.reports_dir / "audio_validation.json"
    val_report_dict = report.to_dict()
    with open(audio_val_path, "w") as f:
        json.dump(val_report_dict, f, indent=2)
    print(f"Audio validation report saved to {audio_val_path}")
        
    if not report.valid:
        print("\nDataset validation failed. Please fix the following errors:", file=sys.stderr)
        for error in report.errors:
            print(f"  [{error.code}] {error.filepath}: {error.message}", file=sys.stderr)
        sys.exit(1)
        
    print("Dataset validation successful.")
    
    if report.issues:
        print("Warnings:")
        for warning in report.issues:
            print(f"  [{warning.code}] {warning.filepath}: {warning.message}")
            
    # Proceed to split
    print(f"\nGenerating speaker-disjoint splits...")
    try:
        assigned_records = make_speaker_disjoint_splits(report.records, seed=args.seed)
    except ValueError as exc:
        print(f"Failed to generate splits: {exc}", file=sys.stderr)
        sys.exit(1)
        
    print("Validating split integrity...")
    integrity_issues = split_integrity_issues(assigned_records, report.file_hashes)
    if integrity_issues:
        print("Split integrity validation failed (leakage detected):", file=sys.stderr)
        for issue in integrity_issues:
            print(f"  {issue}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Writing output manifest to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_metadata(assigned_records, args.output)
    
    print(f"\nDataset Preparation Complete.")
    
    stats = dataset_statistics(assigned_records)
    
    stats_path = args.reports_dir / "dataset_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Dataset statistics saved to {stats_path}")
    
    print("\nDataset Statistics Summary:")
    print(f"  Total samples: {stats['samples']}")
    print(f"  Bonafide: {stats['labels'].get(1, 0)}")
    print(f"  Spoof: {stats['labels'].get(0, 0)}")
    print(f"  Train: {stats['splits'].get('train', 0)}")
    print(f"  Validation: {stats['splits'].get('validation', 0)}")
    print(f"  Test: {stats['splits'].get('test', 0)}")

if __name__ == "__main__":
    main()
