"""Command-line pipeline for dataset validation and splitting."""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is on the path when run as a script
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from training.dataset import validate_dataset, dataset_statistics
from training.splits import make_speaker_disjoint_splits, split_integrity_issues, write_metadata

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset metadata by validating and creating train/val/test splits.")
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata.csv"), help="Path to input metadata CSV")
    parser.add_argument("--root", type=Path, default=Path("data"), help="Root directory for dataset files")
    parser.add_argument("--output", type=Path, default=Path("data/splits/manifest.csv"), help="Path to output split manifest")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    
    args = parser.parse_args()
    
    print(f"Validating dataset from {args.metadata}...")
    report = validate_dataset(args.metadata, args.root)
    
    if not report.valid:
        print("Dataset validation failed. Please fix the following errors:", file=sys.stderr)
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
        
    print("Writing output manifest...")
    write_metadata(assigned_records, args.output)
    
    print(f"\nDataset Preparation Complete. Manifest saved to {args.output}")
    print("\nDataset Statistics:")
    stats = dataset_statistics(assigned_records)
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
