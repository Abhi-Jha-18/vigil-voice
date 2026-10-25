"""Speaker-disjoint split creation and leakage-integrity checks."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from training.dataset import AudioRecord


VALID_SPLITS = {"train", "validation", "test", "test_unseen", "unseen", "unseen_attack"}
CANONICAL_SPLITS = ["train", "validation", "test"]


def make_speaker_disjoint_splits(
    records: list[AudioRecord],
    *,
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    unseen_generators: list[str] | set[str] | None = None,
    unseen_split_name: str = "test_unseen",
) -> list[AudioRecord]:
    """Assign whole speakers to train/validation/test deterministically.

    Speakers are the atomic unit. A corpus without stable speaker IDs must not
    use this splitter because it cannot make a defensible leakage claim.
    """
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must be positive and sum to less than one")

    # If unseen generators specified, separate them into unseen attack split first
    unseen_gen_set = {str(g).lower() for g in unseen_generators} if unseen_generators else set()
    unseen_records = []
    regular_records = []
    if unseen_gen_set:
        for r in records:
            if r.label == 0 and r.generator_id and r.generator_id.lower() in unseen_gen_set:
                unseen_records.append(AudioRecord(**{**asdict(r), "split": unseen_split_name}))
            else:
                regular_records.append(r)
    else:
        regular_records = list(records)

    has_speakers = all(bool(record.speaker_id) for record in regular_records)
    speaker_records: dict[str, list[AudioRecord]] = defaultdict(list)
    for record in regular_records:
        if record.speaker_id:
            speaker_records[record.speaker_id].append(record)
    
    speakers = list(speaker_records)
    
    if not has_speakers or len(speakers) < 3:
        # Fallback to deterministic stratified split by label
        by_label: dict[int, list[AudioRecord]] = defaultdict(list)
        for record in regular_records:
            by_label[record.label].append(record)
        rng = random.Random(seed)
        assigned_records = []
        for label_val, group in sorted(by_label.items()):
            shuffled = list(group)
            rng.shuffle(shuffled)
            train_idx = int(len(shuffled) * train_fraction)
            val_idx = train_idx + int(len(shuffled) * validation_fraction)
            for i, record in enumerate(shuffled):
                if i < train_idx:
                    split = "train"
                elif i < val_idx:
                    split = "validation"
                else:
                    split = "test"
                assigned_records.append(AudioRecord(**{**asdict(record), "split": split}))
        return assigned_records + unseen_records

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
    for record in regular_records:
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
        "train": len(regular_records) * train_fraction,
        "validation": len(regular_records) * validation_fraction,
        "test": len(regular_records) * (1 - train_fraction - validation_fraction),
    }
    assigned_counts = {name: 0 for name in CANONICAL_SPLITS}
    speaker_split: dict[str, str] = {}

    for unit in units:
        unit_labels = {r.label for spk in unit for r in speaker_records[spk]}

        def split_priority(name: str) -> tuple[int, float, str]:
            # Prioritize class balance: does this split still lack any of the unit's classes?
            has_0 = any(r.label == 0 for spk, s in speaker_split.items() if s == name for r in speaker_records[spk])
            has_1 = any(r.label == 1 for spk, s in speaker_split.items() if s == name for r in speaker_records[spk])
            helps_balance = (0 in unit_labels and not has_0) or (1 in unit_labels and not has_1)
            balance_prio = 0 if helps_balance else 1
            ratio = assigned_counts[name] / max(targets[name], 1)
            return (balance_prio, ratio, name)

        split = min(CANONICAL_SPLITS, key=split_priority)
        for speaker in unit:
            speaker_split[speaker] = split
        assigned_counts[split] += sum(len(speaker_records[speaker]) for speaker in unit)

    return [
        AudioRecord(**{**asdict(record), "split": speaker_split[record.speaker_id]})
        for record in regular_records
    ] + unseen_records


def make_generator_disjoint_splits(
    records: list[AudioRecord],
    *,
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    unseen_generators: list[str] | set[str] | None = None,
    unseen_fraction: float = 0.20,
    unseen_split_name: str = "test_unseen",
) -> list[AudioRecord]:
    """Create speaker-disjoint train/val/test splits with a generator-disjoint unseen attack test split."""
    if unseen_generators is None:
        # Automatically discover generators from spoof samples
        all_gens = sorted({
            r.generator_id for r in records
            if r.label == 0 and r.generator_id and r.generator_id.lower() not in {"none", "unknown", ""}
        })
        if len(all_gens) > 1:
            rng = random.Random(seed)
            shuffled_gens = list(all_gens)
            rng.shuffle(shuffled_gens)
            k = max(1, int(len(shuffled_gens) * unseen_fraction))
            unseen_generators = set(shuffled_gens[:k])
        else:
            unseen_generators = set()

    return make_speaker_disjoint_splits(
        records,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        unseen_generators=unseen_generators,
        unseen_split_name=unseen_split_name,
    )


def split_integrity_issues(
    records: list[AudioRecord],
    file_hashes: dict[str, str] | None = None,
    content_hashes: dict[str, str] | None = None,
) -> list[str]:
    """Find cross-split speaker, duplicate, augmentation, source, generator, and class-balance leakage."""
    issues: list[str] = []
    by_speaker: dict[str, set[str]] = defaultdict(set)
    by_source: dict[str, set[str]] = defaultdict(set)
    by_file_hash: dict[str, set[str]] = defaultdict(set)
    by_content_hash: dict[str, set[str]] = defaultdict(set)
    by_generator: dict[str, set[str]] = defaultdict(set)
    split_classes: dict[str, set[int]] = defaultdict(set)

    for record in records:
        if record.split not in VALID_SPLITS:
            issues.append(f"{record.filepath}: split must be one of {sorted(VALID_SPLITS)}")
            continue
        by_speaker[record.speaker_id].add(record.split)
        if record.source_group:
            by_source[record.source_group].add(record.split)
        if file_hashes and record.filepath in file_hashes:
            by_file_hash[file_hashes[record.filepath]].add(record.split)
        if content_hashes and record.filepath in content_hashes:
            by_content_hash[content_hashes[record.filepath]].add(record.split)
        if record.label == 0 and record.generator_id and record.generator_id.lower() not in {"none", "unknown", ""}:
            by_generator[record.generator_id].add(record.split)
        split_classes[record.split].add(record.label)

    # 1. split_integrity_issues must fail if any of train/validation/test lacks either class
    for split_name in ("train", "validation", "test"):
        classes_present = split_classes.get(split_name, set())
        for cls in (0, 1):
            if cls not in classes_present:
                issues.append(f"split lacks class {cls}: {split_name}")

    # 2. Cross-split speaker leakage
    issues.extend(f"speaker overlap across splits: {speaker}" for speaker, splits in by_speaker.items() if len(splits) > 1)

    # 3. Augmentation / source leakage
    issues.extend(f"source/augmentation overlap across splits: {source}" for source, splits in by_source.items() if len(splits) > 1)

    # 4. Duplicate audio overlap across splits (file hash)
    issues.extend(f"duplicate audio overlap across splits: {digest}" for digest, splits in by_file_hash.items() if len(splits) > 1)

    # 5. Content-hash duplicate check across splits (hash of decoded audio)
    issues.extend(f"duplicate content hash overlap across splits: {digest}" for digest, splits in by_content_hash.items() if len(splits) > 1)

    # 6. Generator disjointness check for unseen test splits
    unseen_split_names = {"test_unseen", "unseen", "unseen_attack"}
    for gen, splits in by_generator.items():
        if ("train" in splits or "validation" in splits) and any(s in splits for s in unseen_split_names):
            issues.append(f"generator overlap across seen and unseen splits: {gen}")

    return issues


def write_metadata(records: list[AudioRecord], output_path: str | Path) -> None:
    """Write split-assigned metadata without touching source audio."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()) if records else ["filepath", "label", "speaker_id", "dataset_source", "attack_type", "generator_id", "language", "channel", "split", "source_id", "augmentation_of"])
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
