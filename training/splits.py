"""Speaker-disjoint split creation and leakage-integrity checks."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from training.dataset import AudioRecord


VALID_SPLITS = {"train", "validation", "test"}


def make_speaker_disjoint_splits(
    records: list[AudioRecord],
    *,
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> list[AudioRecord]:
    """Assign whole speakers to train/validation/test deterministically.

    Speakers are the atomic unit. A corpus without stable speaker IDs must not
    use this splitter because it cannot make a defensible leakage claim.
    """
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must be positive and sum to less than one")
    if any(not record.speaker_id for record in records):
        raise ValueError("speaker_id is required for every record before splitting")

    speaker_records: dict[str, list[AudioRecord]] = defaultdict(list)
    for record in records:
        speaker_records[record.speaker_id].append(record)
    speakers = list(speaker_records)
    if len(speakers) < 3:
        raise ValueError("at least three distinct speakers are required for train/validation/test splitting")

    # A declared source/augmentation can connect more than one speaker (for
    # example, a replay recording). Keep the entire connected component atomic.
    parent = {speaker: speaker for speaker in speakers}

    def find(speaker: str) -> str:
        while parent[speaker] != speaker:
            parent[speaker] = parent[parent[speaker]]
            speaker = parent[speaker]
        return speaker

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    source_speakers: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.source_group:
            source_speakers[record.source_group].append(record.speaker_id)
    for linked_speakers in source_speakers.values():
        for speaker in linked_speakers[1:]:
            union(linked_speakers[0], speaker)

    components: dict[str, list[str]] = defaultdict(list)
    for speaker in speakers:
        components[find(speaker)].append(speaker)
    units = list(components.values())
    random.Random(seed).shuffle(units)
    targets = {
        "train": len(records) * train_fraction,
        "validation": len(records) * validation_fraction,
        "test": len(records) * (1 - train_fraction - validation_fraction),
    }
    assigned_counts = {name: 0 for name in VALID_SPLITS}
    speaker_split: dict[str, str] = {}
    for unit in units:
        size = sum(len(speaker_records[speaker]) for speaker in unit)
        split = min(VALID_SPLITS, key=lambda name: (assigned_counts[name] / max(targets[name], 1), name))
        for speaker in unit:
            speaker_split[speaker] = split
        assigned_counts[split] += size

    return [AudioRecord(**{**asdict(record), "split": speaker_split[record.speaker_id]}) for record in records]


def split_integrity_issues(records: list[AudioRecord], file_hashes: dict[str, str] | None = None) -> list[str]:
    """Find cross-split speaker, duplicate, augmentation, and source leakage."""
    issues: list[str] = []
    by_speaker: dict[str, set[str]] = defaultdict(set)
    by_source: dict[str, set[str]] = defaultdict(set)
    by_hash: dict[str, set[str]] = defaultdict(set)

    for record in records:
        if record.split not in VALID_SPLITS:
            issues.append(f"{record.filepath}: split must be one of {sorted(VALID_SPLITS)}")
            continue
        by_speaker[record.speaker_id].add(record.split)
        if record.source_group:
            by_source[record.source_group].add(record.split)
        if file_hashes and record.filepath in file_hashes:
            by_hash[file_hashes[record.filepath]].add(record.split)

    issues.extend(f"speaker overlap across splits: {speaker}" for speaker, splits in by_speaker.items() if len(splits) > 1)
    issues.extend(f"source/augmentation overlap across splits: {source}" for source, splits in by_source.items() if len(splits) > 1)
    issues.extend(f"duplicate audio overlap across splits: {digest}" for digest, splits in by_hash.items() if len(splits) > 1)
    return issues


def write_metadata(records: list[AudioRecord], output_path: str | Path) -> None:
    """Write split-assigned metadata without touching source audio."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()) if records else ["filepath", "label", "speaker_id", "dataset_source", "attack_type", "generator_id", "language", "channel", "split", "source_id", "augmentation_of"])
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
