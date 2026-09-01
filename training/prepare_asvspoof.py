import argparse
import csv
from pathlib import Path
import sys

def parse_asvspoof_protocol(protocol_file: Path, audio_dir: Path, split: str) -> list:
    records = []
    if not protocol_file.exists():
        print(f"Warning: Protocol file {protocol_file} not found. Skipping {split} split.")
        return records

    with open(protocol_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            speaker_id = parts[0]
            audio_file = parts[1]
            system_id = parts[3]
            key = parts[4].lower()
            
            # ASVspoof 2019 uses FLAC
            audio_path = audio_dir / f"{audio_file}.flac"
            
            label = 1 if key == 'bonafide' else 0
            attack_type = key if key == 'bonafide' else f"asvspoof_{system_id}"
            
            records.append({
                "filepath": str(audio_path.as_posix()),
                "label": label,
                "speaker_id": speaker_id,
                "dataset_source": "ASVspoof_2019_LA",
                "attack_type": attack_type,
                "generator_id": system_id if system_id != '-' else 'none',
                "language": "en",
                "channel": "unknown",
                "split": split,
            })
            
    return records

def main():
    parser = argparse.ArgumentParser(description="Prepare ASVspoof 2019 dataset for VigilVoice")
    parser.add_argument("--asvspoof-root", type=Path, required=True, help="Path to ASVspoof 2019 LA root directory")
    parser.add_argument("--output", type=Path, default=Path("data/metadata.csv"), help="Output metadata CSV")
    
    args = parser.parse_args()
    
    # Check if ASVspoof structure exists
    asvspoof_root = args.asvspoof_root
    
    protocols_dir = asvspoof_root / "ASVspoof2019_LA_cm_protocols"
    
    train_protocol = protocols_dir / "ASVspoof2019.LA.cm.train.trn.txt"
    dev_protocol = protocols_dir / "ASVspoof2019.LA.cm.dev.trl.txt"
    eval_protocol = protocols_dir / "ASVspoof2019.LA.cm.eval.trl.txt"
    
    train_audio = asvspoof_root / "ASVspoof2019_LA_train" / "flac"
    dev_audio = asvspoof_root / "ASVspoof2019_LA_dev" / "flac"
    eval_audio = asvspoof_root / "ASVspoof2019_LA_eval" / "flac"
    
    all_records = []
    
    print("Parsing ASVspoof 2019 LA protocols...")
    all_records.extend(parse_asvspoof_protocol(train_protocol, train_audio, "train"))
    all_records.extend(parse_asvspoof_protocol(dev_protocol, dev_audio, "validation"))
    all_records.extend(parse_asvspoof_protocol(eval_protocol, eval_audio, "test"))
    
    if not all_records:
        print("[ERROR] No records found. Make sure the ASVspoof directory is correct.")
        sys.exit(1)
        
    print(f"Writing {len(all_records)} records to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "filepath", "label", "speaker_id", "dataset_source", "attack_type", 
        "generator_id", "language", "channel", "split", "source_id", "augmentation_of"
    ]
    
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
        
    print("Done. You can now run dataset validation.")

if __name__ == "__main__":
    main()
