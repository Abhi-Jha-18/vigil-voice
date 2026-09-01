# Dataset Configuration

VigilVoice uses a metadata-first approach for managing local anti-spoofing datasets. 
**Dataset acquisition is manual.** You must place your audio files locally and create a CSV manifest.

> [!CAUTION]
> Demo synthetic data must NOT be used to report anti-spoofing performance.
> The production pipeline (`training/train.py`) will explicitly **FAIL** and refuse to train if a real dataset is not supplied. It will not silently fall back to synthetic data.

## Adding a Dataset

1. Manually download your corpus (e.g., ASVspoof 2019 LA). Ensure you respect licensing and usage terms.
2. Place audio files in a local folder (e.g., `data/raw/`). Do NOT commit large audio datasets to Git.
3. Update `data/metadata.csv` conforming to the schema below.
4. Run `python data/prepare.py` to perform leakage-safe splitting and generate `data/splits/manifest.csv`.

## CSV Schema Requirements

```text
data/
├── raw/          # Dataset audio placed locally; ignored by Git
├── processed/    # Future derived artifacts; ignored by Git
├── splits/       # Split manifests generated locally
└── metadata.csv  # Metadata manifest, never a benchmark claim by itself
```

## Metadata schema

`metadata.csv` must include these columns:

| Column | Meaning |
| --- | --- |
| `filepath` | Path relative to the repository root (or absolute local path). |
| `label` | `1`/`genuine`/`bona_fide` for genuine speech; `0`/`spoof` for attacks. |
| `speaker_id` | Stable corpus speaker ID; required for speaker-disjoint splits. |
| `dataset_source` | Dataset/corpus version, e.g. `ASVspoof_2019_LA`. |
| `attack_type` | `bona_fide`, `tts`, `voice_conversion`, `replay`, or a corpus-native label. |
| `generator_id` | TTS/VC/replay system identifier where provided; use `unknown` only when unavailable. |
| `language` | BCP-47 code or corpus value, e.g. `en`. |
| `channel` | Recording/channel condition, e.g. `telephony`, `microphone`, `unknown`. |
| `split` | `train`, `validation`, `test`, or blank before splitting. |

Optional `source_id` and `augmentation_of` fields identify common originals. They prevent original/augmentation or source overlap across evaluation splits.

## Workflow

1. Put licensed audio below `data/raw/` without committing it.
2. Create a manifest following the schema above. Preserve upstream license, provenance, protocol, and attack metadata separately with the dataset.
3. Validate the manifest and audio before training:

   ```powershell
   python -c "from training.dataset import validate_dataset; r = validate_dataset('data/metadata.csv', '.'); print(r.to_dict()); raise SystemExit(not r.valid)"
   ```

4. Create speaker-disjoint train/validation/test assignments using `training.splits.make_speaker_disjoint_splits`, then validate with `split_integrity_issues` before model work.
5. Report evaluation by `attack_type` and `generator_id` whenever those fields are available.

## Bundled audio warning

`data/generate_synthetic_data.py` and its generated files are **DEMO DATA — NOT VALID FOR MODEL PERFORMANCE CLAIMS**. They exist only for UI/integration testing.
