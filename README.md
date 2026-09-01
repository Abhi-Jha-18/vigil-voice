# VigilVoice - AI Audio Deepfake & Spoofing Detection Dashboard

VigilVoice is an advanced, real-time audio analysis and security engine designed to detect synthetic voice clones, AI-generated speech, and audio spoofing attacks. Developed as a prototype for college Smart India Hackathon (SIH) presentations.

---

## 🚀 Key Features

* **Dual Capture & Live Call Terminal**: Upload audio files, record snippets, or initiate real-time browser-based live call monitoring via WebSocket/WebRTC audio capture.
* **Real-Time Sliding Window Spoof Detection**: Evaluates incoming live voice in 2.5s sliding windows (1.0s hop) with rolling multi-window voting and single-spike false alarm protection.
* **Fraud Incident Response & Evidence Export**: Automatically tracks suspicious call episodes, aggregates technical metadata (duration, peak spoof score, model SHA-256), and exports structured evidence JSON (`reports/incidents/{incident_id}.json`).
* **Take Action / Official Reporting Center**: Direct navigation assistance to official Government of India cybercrime portals (CyberCrime Portal, Chakshu, NCRP Suspect Report, National Helpline 1930).
* **Voice Activity Detection (VAD)**: Preprocessing pipeline filters silence out of signals to isolate voice syllables using energy-based voice activity clipping.
* **Spectrogram Feature Mapping**: Generates a high-resolution, colorized Mel-frequency power spectrogram on the fly and visualizes it inside the dashboard.
* **Acoustic Analytics**: Extracts statistical MFCCs, Spectral Centroid, Spectral Roll-off, and Zero Crossing Rates.
* **Presentation Injection (SIH Demo Mode)**: Enables presenters to seamlessly inject custom verdicts (`REAL`, `FAKE`, `UNCERTAIN`) for live presentations and slide demonstrations.

---

## 📂 Project Structure

```text
VigilVoice/
│
├── backend/
│   ├── main.py                # FastAPI app endpoints & static asset mounts
│   │
│   ├── audio/
│   │   ├── processor.py       # Audio validation, loading, resampling & normalization
│   │   ├── vad.py             # Voice Activity Detection (silence clipper)
│   │   └── features.py        # Mel spectrogram base64 plotter & MFCC extractor
│   │
│   ├── detection/
│   │   ├── detector.py        # AI classification logic (includes stub & PyTorch model definitions)
│   │   └── decision.py        # Decision maps and risk assignment
│   │
│   └── utils/
│       └── audio_utils.py     # Upload and cache file helper utilities
│
├── training/
│   ├── prepare_dataset.py     # Dataset validation, statistics generation, and split creation
│   ├── train.py               # Production CNN training loop with class imbalance weighting
│   ├── evaluate.py            # Test set evaluation: EER, ROC-AUC, FAR/FRR, Confusion Matrix, and Attack-Type analysis
│   ├── robustness.py          # Robustness & adversarial transformation suite (Noise, Resampling, Volume, Reverberation)
│   └── metrics.py             # EER, ROC-AUC, Precision/Recall, FAR/FRR metrics
│
├── models/
│   └── README.md              # Checkpoint repository details
│
├── data/
│   ├── raw/                   # Unprocessed audio samples
│   ├── processed/             # Featurized dataset arrays
│   └── metadata.csv           # File labels indexing spreadsheet
│
├── tests/
│   └── test_pipeline.py       # Automated integration test suite
│
├── static/
│   ├── index.html             # Dashboard structure & responsive layout
│   ├── css/
│   │   └── style.css          # Glassmorphic UI styling & neon animations
│   └── js/
│       └── app.js             # Web Audio canvas visualizer & API interface
│
├── requirements.txt           # PIP packages list
├── README.md                  # Project documentation manual
├── .gitignore                 # VCS filters
└── .env.example               # Environmental variables skeleton
```

---

## 🛠️ Installation & Setup

1. **Prerequisites**: Install **Python 3.11.x**. This is the supported development version; do not use the current Python 3.13 environment for this project.
2. **Clone the directory** and navigate to the project directory:
   ```bash
VigilVoice requires Python 3.11+.

> [!WARNING]
> **Demo synthetic data must not be used to report anti-spoofing performance.**
> The demo data inside `data/raw` is for architectural demonstrations only. 
> To train the actual model, you MUST manually supply a legitimate dataset (such as ASVspoof).

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

`requirements.txt` pins the CPU-only PyTorch wheel from the official PyTorch CPU index. It is intentionally a CPU development environment; CUDA is not installed or assumed.

### Windows audio decoding

WAV is the most portable input format. MP3, M4A, AAC, and WebM decoding may require FFmpeg to be installed and available on `PATH`, depending on the codecs in the uploaded file and the local audio backend. Install an FFmpeg build for Windows, add its `bin` directory to `PATH`, then open a new PowerShell window and verify:

```powershell
ffmpeg -version
```

For a custom local configuration, copy `.env.example` to `.env` and set the desired values. Environment variables are read at startup. Relative `MODEL_PATH` values are always resolved from the repository root.

---

## 🖥️ Running the Application

### Development Run
```bash
python -m uvicorn backend.main:app --reload --port 8000
```

### Production Hardened Run
```bash
ENVIRONMENT=production python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Container Deployment
```bash
# Build and run using Docker Compose
docker-compose up -d --build

# Or build standalone image
docker build -t vigilvoice-api .
docker run -p 8000:8000 vigilvoice-api
```

Once started, open your web browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### API Health & Readiness Probes
- `GET /api/health`: Liveness probe for containers and load-balancers.
- `GET /api/ready`: Readiness probe verifying runtime capability and model integrity state.

---

## ⚡ Performance Benchmarking & Load Testing

Run the performance profiling script to measure model loading latency, preprocessing time, feature extraction, and neural inference percentiles:

```bash
python training/benchmark.py --iterations 25
```

Run the concurrency load testing tool:

```bash
python training/load_test.py --requests 20 --concurrency 5
```

---

## 🧪 Testing the Audio Pipeline

Ensure all audio analysis, security validations, and ML pipelines function properly by executing the automated test suite:

```bash
pytest tests/ -v
```

This runs all 60 integration and security unit tests covering Phases 11 through 14.

## Model readiness

Phase 1 is an explainable acoustic heuristic suitable for the dashboard and
demo workflow. Phase 2 is a CNN that uses a 13 x 64 MFCC time-frequency input.
The included sample checkpoint was trained on the small synthetic audio set in
`data/raw`; it is useful for integration testing but must be retrained and
evaluated on a representative, independently held-out spoofing dataset before
using the application for security decisions. Phases 3 and 4 remain future
model options and transparently fall back to Phase 2/Phase 1 when selected.

## Canonical Preprocessing Pipeline

VigilVoice uses a unified, canonical preprocessing pipeline (`backend/audio/pipeline.py`) across all dataset generation, model training, offline validation, and real-time backend inference workflows.

1. **Decoding & Resampling**: Input files are decoded, converted to mono, and resampled to the configured sample rate (`16000` Hz by default).
2. **Quality Checks**: Evaluates audio for being too short, entirely silent, or suffering from severe clipping. Statuses are classified as `GOOD`, `DEGRADED`, or `INSUFFICIENT`. **Note:** Audio quality affects reliability but is not itself evidence of synthetic speech.
3. **Peak Normalization**: Normalizes amplitude to a maximum value of 1.0.
4. **Voice Activity Detection (VAD)**: Trims non-speech silence intervals.
5. **Segmentation**: Splits audio into configurable segments (e.g., 3s duration, 1s overlap).

If an uploaded file or dataset sample is classified as `INSUFFICIENT`, the pipeline fails safely with a structured message instead of generating misleading predictions.

Supported extensions include: `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, `.webm`, `.aac`.

## Dataset ingestion

The bundled audio fixtures are **DEMO DATA — NOT VALID FOR MODEL PERFORMANCE CLAIMS**. Real model development must use locally placed, properly licensed/public anti-spoof corpora with genuine, TTS, voice-conversion, and replay/spoof samples where available. Nothing is downloaded automatically.

To prepare a dataset, first place the raw audio in `data/raw/` and create `data/metadata.csv` following the schema in [data/README.md](data/README.md).

Then, run the preparation script to validate the dataset, generate speaker-independent train/val/test splits, and view dataset statistics:

```bash
python data/prepare.py --metadata data/metadata.csv --root data --output data/splits/manifest.csv
```

The script will halt and report any errors if there are missing/corrupt files, unsupported formats, or data leakage issues across splits.

## Configuration

The centralized configuration is in `backend/config.py`; `.env.example` documents all environment variables. Defaults are intentionally conservative for local development:

* `SAMPLE_RATE=16000`
* `MAX_UPLOAD_MB=50` (the dashboard reads this exact value from `/api/status`)
* `MAX_AUDIO_DURATION_SECONDS=120`
* `SEGMENT_DURATION_SECONDS=4` and `SEGMENT_OVERLAP_SECONDS=1` (reserved for the future segmented model pipeline)
* `MODEL_PATH=models/cnn_weight.pth` and `MODEL_VERSION=unversioned-demo-checkpoint` (replace with a traceable version after real model training)
* `CORS_ORIGINS` is an explicit comma-separated allowlist.
* `DEMO_MODE=true` enables presentation verdict overrides. Set it to `false` outside the demo workflow.

---

## 💡 SIH Presentation Mode

For live presentations to reviewers where you need to showcase both authentic and deepfake examples reliably:
1. Open the UI.
2. Under **Demo Simulation Injection (SIH Presentation Mode)** on the left panel, select:
   * **Force REAL**: Guarantees a high-confidence, green-colored `REAL` verdict.
   * **Force FAKE**: Guarantees a high-risk, red-colored `FAKE` warning verdict.
3. Upload or record your audio and click **Run AI Detection Engine**. The dashboard will step through all stages and display the selected target verdict with correct waveforms.
