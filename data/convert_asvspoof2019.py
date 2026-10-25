import argparse, csv, random
from pathlib import Path

# folder, protocol file, your split name
SPLITS = [
    ("ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt", "train"),
    ("ASVspoof2019_LA_dev",   "ASVspoof2019.LA.cm.dev.trl.txt",   "validation"),
    ("ASVspoof2019_LA_eval",  "ASVspoof2019.LA.cm.eval.trl.txt",  "test"),
]
FIELDS = ["filepath", "label", "speaker_id", "dataset_source", "attack_type",
          "generator_id", "language", "channel", "split"]

def main(root: Path, out: Path, per_split: int, seed: int):
    rng = random.Random(seed)
    proto_dir = root / "ASVspoof2019_LA_cm_protocols"
    rows = []
    for folder, proto, split in SPLITS:
        lines = [l.split() for l in (proto_dir / proto).read_text().splitlines() if l.strip()]
        # line: SPEAKER FILE - SYSTEM KEY   (SYSTEM is "-" for bonafide)
        bona = [l for l in lines if l[4] == "bonafide"]
        spoof = [l for l in lines if l[4] == "spoof"]
        k = per_split // 2
        picked = rng.sample(bona, min(k, len(bona))) + rng.sample(spoof, min(k, len(spoof)))
        for spk, fname, _, system, key in picked:
            is_real = key == "bonafide"
            rel = f"{folder}/flac/{fname}.flac"
            if not (root / rel).is_file():
                raise FileNotFoundError(rel)
            rows.append({
                "filepath": str((root / rel).as_posix()),
                "label": 1 if is_real else 0,
                "speaker_id": spk,
                "dataset_source": "ASVspoof2019_LA",
                "attack_type": "bona_fide" if is_real else system,
                "generator_id": "none" if is_real else system,
                "language": "en",
                "channel": "microphone",
                "split": split,
            })
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("data/metadata.csv"))
    p.add_argument("--per-split", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    main(**vars(p.parse_args()))