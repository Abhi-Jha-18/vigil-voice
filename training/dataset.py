"""Metadata-first dataset ingestion and validation for local anti-spoof corpora.

This module never downloads data. Dataset owners provide audio files and a CSV
manifest using the columns documented in ``data/README.md``.
"""

from __future__ import annotations

import csv
import hashlib
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


REQUIRED_COLUMNS = {
    "filepath", "label", "speaker_id", "dataset_source", "attack_type",
    "generator_id", "language", "channel", "split",
}
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".webm", ".mp4"}
LABEL_ALIASES = {
    "1": 1, "real": 1, "genuine": 1, "bonafide": 1, "bona_fide": 1,
    "0": 0, "fake": 0, "spoof": 0, "synthetic": 0,
}


@dataclass(frozen=True)
class AudioRecord:
    filepath: str
    label: int
    speaker_id: str
    dataset_source: str
    attack_type: str
    generator_id: str
    language: str
    channel: str
    split: str = ""
    source_id: str = ""
    augmentation_of: str = ""

    @property
    def source_group(self) -> str:
        """Stable grouping key for originals and their declared augmentations."""
        parent = self.augmentation_of or self.source_id
        return f"{self.dataset_source}:{parent}" if parent else ""


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    filepath: str
    message: str


@dataclass
class ValidationReport:
    records: list[AudioRecord] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    file_hashes: dict[str, str] = field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "record_count": len(self.records),
            "error_count": len(self.errors),
            "warning_count": len(self.issues) - len(self.errors),
            "issues": [asdict(issue) for issue in self.issues],
        }


def normalize_label(value: str | int) -> int:
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in LABEL_ALIASES:
        raise ValueError("label must identify genuine/bona_fide (1) or spoof (0)")
    return LABEL_ALIASES[key]


def load_metadata(metadata_path: str | Path) -> list[AudioRecord]:
    """Load a metadata manifest and normalize the binary label representation."""
    path = Path(metadata_path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Metadata is missing required columns: {', '.join(sorted(missing))}")

        records = []
        for row_number, row in enumerate(reader, start=2):
            try:
                records.append(AudioRecord(
                    filepath=(row.get("filepath") or "").strip(),
                    label=normalize_label(row.get("label") or ""),
                    speaker_id=(row.get("speaker_id") or "").strip(),
                    dataset_source=(row.get("dataset_source") or "").strip(),
                    attack_type=(row.get("attack_type") or "").strip(),
                    generator_id=(row.get("generator_id") or "").strip(),
                    language=(row.get("language") or "").strip(),
                    channel=(row.get("channel") or "").strip(),
                    split=(row.get("split") or "").strip().lower(),
                    source_id=(row.get("source_id") or "").strip(),
                    augmentation_of=(row.get("augmentation_of") or "").strip(),
                ))
            except ValueError as exc:
                raise ValueError(f"Invalid metadata at row {row_number}: {exc}") from exc
    return records


def resolve_audio_path(record: AudioRecord, dataset_root: str | Path) -> Path:
    path = Path(record.filepath)
    return path if path.is_absolute() else (Path(dataset_root) / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audio_is_silent(path: Path, threshold: float) -> bool:
    max_amplitude = 0.0
    with sf.SoundFile(path) as audio:
        for block in audio.blocks(blocksize=65_536, dtype="float32", always_2d=False):
            if block.size:
                max_amplitude = max(max_amplitude, float(np.max(np.abs(block))))
    return max_amplitude < threshold


def validate_dataset(
    metadata_path: str | Path,
    dataset_root: str | Path,
    *,
    minimum_duration_seconds: float = 0.25,
    silence_threshold: float = 1e-4,
) -> ValidationReport:
    """Validate metadata and locally stored audio without changing any files."""
    try:
        records = load_metadata(metadata_path)
    except ValueError as exc:
        return ValidationReport(issues=[ValidationIssue("error", "invalid_metadata", "", str(exc))])
    report = ValidationReport(records=records)
    seen_paths: set[Path] = set()
    hashes: dict[str, str] = {}

    for record in records:
        path = resolve_audio_path(record, dataset_root)
        path_key = str(path).casefold() if os.name == "nt" else str(path)
        if not record.filepath:
            report.issues.append(ValidationIssue("error", "missing_filepath", record.filepath, "filepath is empty"))
            continue
        if not record.speaker_id:
            report.issues.append(ValidationIssue("warning", "missing_speaker", record.filepath, "speaker_id missing, will use random stratified split"))
        if not record.dataset_source:
            report.issues.append(ValidationIssue("warning", "missing_dataset_source", record.filepath, "dataset_source missing, using 'unknown'"))
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            report.issues.append(ValidationIssue("error", "unsupported_format", record.filepath, f"unsupported extension: {path.suffix}"))
            continue
        if not path.is_file():
            report.issues.append(ValidationIssue("error", "missing_file", record.filepath, "audio file does not exist"))
            continue
        if path in seen_paths:
            report.issues.append(ValidationIssue("error", "duplicate_filepath", record.filepath, "same file appears more than once in metadata"))
            continue
        seen_paths.add(path)

        try:
            info = sf.info(path)
            if info.samplerate <= 0 or info.frames <= 0:
                raise RuntimeError("no decodable audio frames")
            if info.duration < minimum_duration_seconds:
                report.issues.append(ValidationIssue("error", "sample_too_short", record.filepath, f"duration {info.duration:.3f}s is below {minimum_duration_seconds:.3f}s"))
            if _audio_is_silent(path, silence_threshold):
                report.issues.append(ValidationIssue("error", "silence_only", record.filepath, "audio has no measurable signal"))
        except Exception as exc:
            report.issues.append(ValidationIssue("error", "corrupt_audio", record.filepath, f"cannot decode audio: {exc}"))
            continue

        digest = _sha256(path)
        report.file_hashes[record.filepath] = digest
        if digest in hashes:
            report.issues.append(ValidationIssue("error", "duplicate_audio", record.filepath, f"same audio content as {hashes[digest]}"))
        else:
            hashes[digest] = record.filepath

    return report


def dataset_statistics(records: Iterable[AudioRecord]) -> dict[str, object]:
    """Return JSON-serializable corpus counts, including attack/generator slices."""
    records = list(records)
    count = lambda values: dict(sorted(Counter(values).items()))
    return {
        "samples": len(records),
        "labels": count("genuine" if record.label else "spoof" for record in records),
        "speakers": len({record.speaker_id for record in records if record.speaker_id}),
        "dataset_sources": count(record.dataset_source or "unknown" for record in records),
        "attack_types": count(record.attack_type or "unknown" for record in records),
        "generator_ids": count(record.generator_id or "unknown" for record in records),
        "languages": count(record.language or "unknown" for record in records),
        "channels": count(record.channel or "unknown" for record in records),
        "splits": count(record.split or "unassigned" for record in records),
    }


class AntiSpoofDataset(Dataset):
    """
    PyTorch Dataset for VigilVoice using real anti-spoofing data.
    Uses canonical preprocessing to match inference exactly.
    NEVER silently falls back to synthetic tensors.
    """
    def __init__(
        self,
        manifest_path: str | Path,
        split: str = None,
        root_dir: str | Path = None,
        allow_synthetic: bool = False,
        cache_features: bool = True,
        augment: bool = False,
    ):
        self.root_dir = Path(root_dir) if root_dir else Path(manifest_path).parent
        self.augment = bool(augment and (split or "").lower() == "train")
        try:
            all_records = load_metadata(manifest_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset manifest: {e}")

        if split:
            self.records = [r for r in all_records if r.split.lower() == split.lower()]
        else:
            self.records = all_records

        if not self.records:
            raise ValueError(f"No records found for split '{split}'.")

        # Reject if these are the demo tone generation files (fail-safe)
        demo_count = sum(1 for r in self.records if "DEMO_DATA" in r.dataset_source.upper())
        if demo_count == len(self.records) and demo_count > 0 and not allow_synthetic:
            raise ValueError("REAL DATASET REQUIRED. Manifest contains only synthetic DEMO fixtures. "
                             "Please supply a legitimate speech anti-spoofing dataset.")

        self._cache: list[tuple[torch.Tensor, torch.Tensor, dict]] | None = None
        if cache_features:
            self._cache = [self._load_item(i) for i in range(len(self.records))]

    def __len__(self) -> int:
        return len(self.records)

    def _load_item(self, idx: int):
        record = self.records[idx]
        filepath = resolve_audio_path(record, self.root_dir)

        if not filepath.exists():
            raise FileNotFoundError(f"Missing audio file: {filepath}")

        # Lazy imports — avoids pulling matplotlib into validation-only code paths
        from backend.audio.features import extract_mfcc_for_model
        from training.preprocessing import preprocess_for_training

        # Use canonical preprocessing (to ensure training == inference)
        try:
            audio_array = preprocess_for_training(str(filepath))
        except ValueError as e:
            # Re-raise explicit error instead of faking data
            raise RuntimeError(f"Preprocessing failed for {filepath}: {e}")

        mfcc = extract_mfcc_for_model(audio_array)
        if mfcc is None or mfcc.shape != (13, 64):
            raise ValueError(f"Feature extraction failed for {filepath}")

        tensor = torch.from_numpy(np.ascontiguousarray(mfcc, dtype=np.float32)).unsqueeze(0)
        label = torch.tensor(record.label, dtype=torch.float32)

        metadata = {
            "speaker_id": record.speaker_id,
            "attack_type": record.attack_type,
            "filepath": str(filepath)
        }
        return tensor, label, metadata

    def __getitem__(self, idx: int):
        if self._cache is not None:
            tensor, label, metadata = self._cache[idx]
            tensor = tensor.clone()
        else:
            tensor, label, metadata = self._load_item(idx)

        if self.augment:
            shift = int(torch.randint(-6, 7, (1,)).item())
            if shift:
                tensor = torch.roll(tensor, shifts=shift, dims=-1)
            tensor = tensor + torch.randn_like(tensor) * 0.35

        return tensor, label, metadata
