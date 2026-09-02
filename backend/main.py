"""
VigilVoice — FastAPI Application (Production Hardened — Phase 14)
==================================================================
Main entrypoint for the VigilVoice AI Audio Spoofing Detection API.

Run with:
    python -m uvicorn backend.main:app --reload --port 8000
"""

import asyncio
import importlib.metadata
import logging
import platform
import sys
import time
import uuid
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.utils.audio_utils import save_temp_file, remove_temp_file, sanitize_filename, get_safe_extension
from backend.audio.processor import validate_format
from backend.audio.vad import remove_silence
from backend.audio.features import extract_acoustic_features, get_mel_spectrogram_b64
from backend.detection.detector import run_ai_detection, is_cnn_model_available, get_model_status
from backend.detection.decision import make_decision
from backend.config import PROJECT_ROOT, settings
from backend.api.stream import router as stream_router
from backend.api.live import router as live_router
from backend.api.incidents import router as incidents_router

# ── Logging Configuration ──────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] [req:%(name)s] %(message)s",
)
logger = logging.getLogger("vigilvoice")

# Static files directory (resolved from project root)
STATIC_DIR = PROJECT_ROOT / "frontend" / "dist"

# App startup time (for uptime tracking)
_start_time = time.time()

# Concurrency Throttling Semaphore
_inference_semaphore = asyncio.Semaphore(settings.max_concurrent_inferences)

# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="VigilVoice API",
    description="AI-powered Audio Spoofing & Deepfake Detection Engine",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

app.include_router(stream_router)
app.include_router(live_router)
app.include_router(incidents_router)

VALID_PHASES = {"phase1", "phase2", "phase3", "phase4"}
VALID_DEMO_VERDICTS = {"real", "fake", "uncertain"}


# ── Middleware: Request ID & Security Headers ──────────────────────────────────

@app.middleware("http")
async def security_and_tracing_middleware(request: Request, call_next):
    # 1. Generate or extract unique Request ID
    req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = req_id

    # 2. Process request
    start_time = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000

    # 3. Add Tracing & Security Headers
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self' ws: wss:;"
    )

    # 4. Log non-static access
    if not request.url.path.startswith("/css") and not request.url.path.startswith("/js"):
        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.1f}ms) [id:{req_id}]"
        )

    return response


# ── CORS Middleware ────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── Standardized Error Handlers ────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    req_id = getattr(request.state, "request_id", "unknown")
    
    # Map status code to standard code string if detail is a simple string
    code_map = {
        400: "INVALID_REQUEST",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        413: "FILE_TOO_LARGE",
        415: "UNSUPPORTED_FORMAT",
        422: "UNPROCESSABLE_ENTITY",
        429: "CONCURRENCY_LIMIT_EXCEEDED",
        500: "SERVER_ERROR",
        503: "SERVICE_UNAVAILABLE",
        504: "INFERENCE_TIMEOUT"
    }
    
    error_code = code_map.get(exc.status_code, "ERROR")
    error_message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    
    # Check if specific custom code was provided in detail
    if "too large" in error_message.lower():
        error_code = "FILE_TOO_LARGE"
    elif "empty" in error_message.lower():
        error_code = "EMPTY_FILE"
    elif "format" in error_message.lower() or "extension" in error_message.lower():
        error_code = "UNSUPPORTED_FORMAT"
    elif "too long" in error_message.lower():
        error_code = "AUDIO_TOO_LONG"
    elif "quality" in error_message.lower():
        error_code = "INVALID_AUDIO"

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "detail": error_message,
            "error": {
                "code": error_code,
                "message": error_message,
                "request_id": req_id
            }
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    req_id = getattr(request.state, "request_id", "unknown")
    first_error = exc.errors()[0] if exc.errors() else {}
    msg = first_error.get("msg", "Invalid request parameters.")
    loc = " -> ".join(str(l) for l in first_error.get("loc", []))
    error_message = f"Validation error at {loc}: {msg}" if loc else msg
    
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "detail": error_message,
            "error": {
                "code": "INVALID_REQUEST",
                "message": error_message,
                "request_id": req_id
            }
        }
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", "unknown")
    logger.exception(f"Unhandled server error [req:{req_id}]: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "detail": "Internal server error during audio processing.",
            "error": {
                "code": "SERVER_ERROR",
                "message": "Internal server error occurred during audio processing.",
                "request_id": req_id
            }
        }
    )


# ── Health, Readiness & Status Endpoints ────────────────────────────────────────

def _runtime_info() -> dict[str, object]:
    packages = ("fastapi", "librosa", "numpy", "scipy", "soundfile", "torch", "scikit-learn")
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": {
            package: _distribution_version(package) for package in packages
        },
    }


def _distribution_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


@app.get("/api/health", tags=["System"])
async def health_check():
    """
    Lightweight health check endpoint.
    Returns HTTP 200 when the server process is alive.
    Used by container orchestration and load balancers.
    """
    return {
        "status": "ok",
        "service": "VigilVoice",
        "timestamp": time.time(),
        "model_status": get_model_status(),
        "model": {
            "phase2_available": is_cnn_model_available(),
            "version": settings.model_version,
        },
        "configuration": settings.public_dict(),
        "runtime": _runtime_info(),
    }


@app.get("/api/ready", tags=["System"])
async def readiness_check():
    """
    Readiness probe verifying the backend is ready to accept detection requests.
    """
    status = get_model_status()
    is_ready = status != "MODEL_INTEGRITY_FAILURE"
    
    response_payload = {
        "ready": is_ready,
        "service": "VigilVoice",
        "model_status": status,
        "environment": settings.environment
    }
    
    if not is_ready:
        return JSONResponse(status_code=503, content=response_payload)
    return response_payload


@app.get("/api/status", tags=["System"])
async def get_status():
    """
    Returns server status, configuration, and model capabilities.
    """
    cnn_ready = is_cnn_model_available()
    uptime_sec = int(time.time() - _start_time)

    return {
        "server": "VigilVoice API",
        "version": "1.0.0",
        "uptime_seconds": uptime_sec,
        "debug_mode": settings.debug,
        "configuration": settings.public_dict(),
        "runtime": _runtime_info(),
        "models": {
            "phase1_heuristic": {
                "available": True,
                "description": "Acoustic heuristic (MFCC variance + spectral centroid)"
            },
            "phase2_cnn": {
                "available": cnn_ready,
                "checkpoint": str(settings.model_path),
                "version": settings.model_version,
                "description": "SimpleCNNDetector trained on MFCC spectrograms"
            },
            "phase3_cnn_lstm": {
                "available": False,
                "description": "CNN-LSTM (requires training data)"
            },
            "phase4_wav2vec": {
                "available": False,
                "description": "Wav2Vec (requires pre-trained transformer weights)"
            }
        }
    }


@app.get("/api/model/info", tags=["System"])
async def get_model_info():
    """
    Returns verified evaluation metrics, metadata, and robustness data.
    """
    import json
    metadata_path = settings.model_path.with_name(settings.model_path.stem + "_metadata.json")
    evaluation_path = PROJECT_ROOT / "training" / "runs" / "evaluation_report.json"
    
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except Exception:
            pass
            
    evaluation = {}
    if evaluation_path.exists():
        try:
            with open(evaluation_path, "r") as f:
                evaluation = json.load(f)
        except Exception:
            pass

    robustness_path = PROJECT_ROOT / "reports" / "robustness_report.json"
    robustness = {"available": False}
    if robustness_path.exists():
        try:
            with open(robustness_path, "r") as f:
                rob_data = json.load(f)
                robustness = {
                    "available": rob_data.get("status") == "COMPLETED",
                    "status": rob_data.get("status"),
                    "data": rob_data
                }
        except Exception:
            pass
            
    return {
        "status": "success",
        "model_status": get_model_status(),
        "metadata": metadata,
        "evaluation": evaluation,
        "robustness": robustness
    }


# ── Detection Endpoint ─────────────────────────────────────────────────────────

@app.post("/api/detect", tags=["Detection"])
async def detect_audio(
    file: UploadFile = File(...),
    phase: str = Form("phase1"),
    force_verdict: str = Form(None)
):
    """
    Hardened main detection endpoint with strict input validation,
    concurrency throttling, inference timeout, and guaranteed resource cleanup.
    """
    # 1. Validate Phase and Force Verdict
    phase = phase.lower().strip()
    if phase not in VALID_PHASES:
        raise HTTPException(status_code=422, detail="Invalid detection phase.")
        
    if force_verdict:
        if not settings.demo_mode:
            raise HTTPException(status_code=403, detail="Demo verdict overrides are disabled in production.")
        force_verdict = force_verdict.lower().strip()
        if force_verdict not in VALID_DEMO_VERDICTS:
            raise HTTPException(status_code=422, detail="Invalid demo verdict override.")

    # 2. Validate Filename & Extension (Path Traversal Protection)
    safe_name = sanitize_filename(file.filename)
    safe_ext = get_safe_extension(safe_name)
    
    if not validate_format(safe_name) or safe_ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Allowed: WAV, MP3, M4A, MP4, OGG, FLAC, AAC, WEBM"
        )

    # 3. Validate MIME type if provided
    if file.content_type and file.content_type.lower() not in settings.allowed_mime_types:
        # Some browsers send general octet-stream for audio, and mp4 might come as video/mp4
        if not file.content_type.startswith("audio/") and file.content_type.lower() != "video/mp4":
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported MIME type '{file.content_type}'. Please upload a valid audio or MP4 file."
            )

    # 4. Check File Size (Read up to max + 1 byte)
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty (0 bytes).")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum upload size is {settings.max_upload_mb}MB."
        )
    await file.seek(0)

    # 5. Acquire Concurrency Semaphore
    try:
        acquired = _inference_semaphore.locked()
        if acquired and _inference_semaphore._value == 0:
            raise HTTPException(
                status_code=429,
                detail="Server busy: maximum concurrent audio analyses reached. Please retry."
            )
    except AttributeError:
        pass

    async with _inference_semaphore:
        start_time = time.perf_counter()
        temp_path = None

        try:
            # 6. Save to Unpredictable Temp File
            temp_path = save_temp_file(file)

            # 7. Execute Analysis with Timeout Protection
            result_payload = await asyncio.wait_for(
                _run_analysis_pipeline(temp_path, phase, force_verdict, safe_name, start_time),
                timeout=settings.inference_timeout_seconds
            )
            return result_payload

        except asyncio.TimeoutError:
            logger.error(f"Inference timeout after {settings.inference_timeout_seconds}s for {safe_name}")
            raise HTTPException(
                status_code=504,
                detail=f"Audio analysis exceeded timeout limit of {settings.inference_timeout_seconds}s."
            )
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.exception(f"Internal error during audio analysis of {safe_name}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error during audio analysis.")
        finally:
            if temp_path:
                remove_temp_file(temp_path)


async def _run_analysis_pipeline(temp_path: str, phase: str, force_verdict: Optional[str], filename: str, start_time: float) -> dict:
    """Executes the canonical audio preprocessing and inference pipeline in a thread worker."""
    def _sync_pipeline():
        from backend.audio.pipeline import process_canonical
        from backend.audio.config import default_config

        preprocessing_result = process_canonical(temp_path, config=default_config)

        if preprocessing_result.quality.status == "INSUFFICIENT":
            reason = preprocessing_result.quality.reason or "Unknown reason"
            raise HTTPException(
                status_code=400,
                detail=f"Audio quality insufficient for analysis: {reason}"
            )

        y_clean = preprocessing_result.audio_data
        sr = default_config.sample_rate

        if len(y_clean) == 0:
            raise HTTPException(
                status_code=400,
                detail="No readable speech found after voice activity detection."
            )

        # Feature Extraction
        features = extract_acoustic_features(y_clean, sr=sr)
        spectrogram_b64 = get_mel_spectrogram_b64(y_clean, sr=sr)

        # Segment Detection & Aggregation
        from backend.detection.detector import run_segment_detection
        from backend.detection.aggregator import aggregate_predictions
        from backend.detection.risk import generate_risk_report
        from backend.detection.explainability import generate_evidence
        from backend.detection.recommendations import get_recommendation
        
        segment_predictions = run_segment_detection(preprocessing_result.segments, phase=phase, force_verdict=force_verdict)
        aggregation = aggregate_predictions(segment_predictions, strategy="max_risk")
        real_score = aggregation.overall_probability

        risk_report = generate_risk_report(
            max_fake_prob=aggregation.maximum_fake_probability,
            avg_fake_prob=aggregation.average_fake_probability,
            temporal_variance=aggregation.temporal_consistency.get("variance", 0.0),
            suspicious_segments=aggregation.suspicious_segments,
            total_segments=len(segment_predictions),
            quality_status=preprocessing_result.quality.status
        )
        
        evidence = generate_evidence(
            features=features,
            temporal_variance=aggregation.temporal_consistency.get("variance", 0.0),
            suspicious_segments_count=len(aggregation.suspicious_segments),
            total_segments=len(segment_predictions)
        )
        risk_report["signals"] = evidence

        security_recommendation = get_recommendation(
            risk_level=risk_report["risk_level"],
            suspicious_segments_count=len(aggregation.suspicious_segments),
            evidence_signals=evidence
        )

        decision = make_decision(real_score)
        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        return {
            "success": True,
            "filename": filename,
            "phase_used": phase,
            "cnn_model_active": (phase != "phase1") and is_cnn_model_available(),
            "model_status": get_model_status(),
            "processing_time_ms": processing_time_ms,
            "verdict": decision["verdict"],
            "confidence": decision["confidence"],
            "risk_level": decision["risk_level"],
            "raw_score": decision["raw_score"],
            "audio_duration_seconds": features["audio_duration_seconds"],
            "features": {
                "spectral_centroid": features["spectral_centroid_mean"],
                "spectral_rolloff":  features["spectral_rolloff_mean"],
                "zero_crossing_rate": features["zero_crossing_rate_mean"],
                "mfcc_mean": features["mfcc_mean"]
            },
            "spectrogram": f"data:image/png;base64,{spectrogram_b64}" if spectrogram_b64 else None,
            "segments": [s.to_dict() for s in segment_predictions],
            "aggregation": aggregation.to_dict(),
            "risk_report": risk_report,
            "security_recommendation": security_recommendation
        }

    return await asyncio.to_thread(_sync_pipeline)


# ── Static File Serving & SPA Fallback ─────────────────────────────────────────
# The React build (Vite) is emitted into STATIC_DIR. Hashed assets under /assets
# are served by StaticFiles; any other non-API GET path returns index.html so the
# React Router can handle client-side routes (e.g. /live, /incidents/:id).

if STATIC_DIR.exists():
    # Vite emits content-hashed, long-lived cacheable bundles under /assets.
    _assets_dir = STATIC_DIR / "assets"
    if _assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        # Never shadow API routes (they are registered before this catch-all, but
        # guard explicitly for clarity).
        if full_path.startswith("api/"):
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": {"code": "NOT_FOUND", "message": "Unknown API endpoint."}},
            )

        # Serve a real static file if it exists (favicon, robots.txt, etc.).
        if full_path:
            candidate = (STATIC_DIR / full_path).resolve()
            try:
                candidate.relative_to(STATIC_DIR.resolve())
            except ValueError:
                return JSONResponse(status_code=404, content={"detail": "Not found."})
            if candidate.is_file():
                return FileResponse(candidate)

        # Otherwise hand control to the React single-page app.
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": {"code": "NOT_FOUND", "message": "Frontend build (static/index.html) was not found."}},
        )
else:
    logger.warning(f"Static files directory not found at {STATIC_DIR}")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def no_frontend_fallback(full_path: str):
        if full_path.startswith("api/"):
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": {"code": "NOT_FOUND", "message": "Unknown API endpoint."}},
            )
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": {"code": "NOT_FOUND", "message": "Frontend build not deployed."}},
        )
