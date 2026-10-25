"""
VigilVoice — FastAPI Application (Production Hardened — Phase 16)
==================================================================
Main entrypoint for the VigilVoice AI Audio Spoofing Detection API.

Deployment Architecture Note:
-----------------------------
Active real-time live calling sessions and concurrency throttling are coordinated
per-process. For production deployments with multiple workers, either:
1. Run with a single worker: `python -m uvicorn backend.main:app --port 8000 --workers 1`
2. Configure sticky sessions on the reverse proxy (e.g. Nginx/HAProxy), or
3. Back session state and incident tracking with a shared SQLite/PostgreSQL/Redis cluster.
Per-worker CPU contention is prevented by setting `torch.set_num_threads(1)`.
"""

import asyncio
import os
import importlib.metadata
import logging
import platform
import sys
import time
import uuid
import contextvars
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

# Enable direct script execution: python backend/main.py
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
# Bound torch worker threads to 1 per worker to prevent CPU oversubscription
try:
    torch.set_num_threads(1)
except Exception:
    pass

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.utils.audio_utils import (
    save_temp_file,
    remove_temp_file,
    sanitize_filename,
    get_safe_extension,
    validate_magic_bytes,
    get_audio_header_duration,
)
from backend.audio.processor import validate_format
from backend.audio.vad import remove_silence
from backend.audio.features import extract_acoustic_features, get_mel_spectrogram_b64
from backend.detection.detector import (
    run_ai_detection,
    is_cnn_model_available,
    get_model_status,
    warmup_cnn_model,
)
from backend.detection.decision import make_decision
from backend.config import PROJECT_ROOT, settings
from backend.api.auth import verify_api_key
from backend.api.stream import router as stream_router
from backend.api.live import router as live_router
from backend.api.incidents import router as incidents_router

# ── Logging Configuration with Request ID Filter ───────────────────────────────
current_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("current_request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Injects current_request_id contextvar into log records for structured tracing."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = current_request_id.get()
        return True


_log_handler = logging.StreamHandler()
_log_handler.addFilter(RequestIdFilter())
_log_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] [req:%(request_id)s] %(message)s")
)
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    handlers=[_log_handler],
)
logger = logging.getLogger("vigilvoice")

# Static files directory (resolved from project root)
STATIC_DIR = PROJECT_ROOT / "static"

# App startup time (for uptime tracking)
_start_time = time.time()

# Concurrency Throttling Semaphore
_inference_semaphore = asyncio.Semaphore(settings.max_concurrent_inferences)


# ── Per-Client Rate Limiter ────────────────────────────────────────────────────
class ClientRateLimiter:
    """In-memory sliding-window per-client rate limiter."""
    def __init__(self, max_requests: int = 120, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._history: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, client_id: str) -> tuple[bool, int, int]:
        async with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds
            timestamps = [t for t in self._history.get(client_id, []) if t > cutoff]
            if len(timestamps) >= self.max_requests:
                self._history[client_id] = timestamps
                retry_after = max(1, int(timestamps[0] + self.window_seconds - now) + 1)
                return False, 0, retry_after
            timestamps.append(now)
            self._history[client_id] = timestamps
            remaining = self.max_requests - len(timestamps)
            return True, remaining, 0


_rate_limiter = ClientRateLimiter(
    max_requests=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds
)


# ── Domain Exceptions ─────────────────────────────────────────────────────────
class AppError(Exception):
    """Domain exception with explicit machine-readable code and HTTP status code."""
    def __init__(self, code: str, status_code: int, message: str):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.message = message


# ── Lifespan Context Manager ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan managing worker initialization and model warmup."""
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    try:
        status = warmup_cnn_model()
        logger.info(f"Detection engine ready (model_status={status})")
    except Exception as exc:
        logger.warning(f"Model warmup skipped: {exc}")

    yield


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="VigilVoice API",
    description="AI-powered Audio Spoofing & Deepfake Detection Engine",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
)

app.include_router(stream_router)
app.include_router(live_router)
app.include_router(incidents_router)

VALID_PHASES = {"phase1", "phase2", "phase3", "phase4"}
VALID_DEMO_VERDICTS = {"real", "fake", "uncertain"}


# ── Middleware: Request ID, Rate Limiting, Content-Length & Security Headers ──
@app.middleware("http")
async def security_and_tracing_middleware(request: Request, call_next):
    # 1. Generate or extract unique Request ID and bind to ContextVar
    req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = req_id
    token = current_request_id.set(req_id)

    try:
        # 2. Content-Length early validation for file upload size
        content_length_hdr = request.headers.get("content-length")
        if content_length_hdr:
            try:
                if int(content_length_hdr) > settings.max_upload_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "success": False,
                            "detail": f"File too large. Maximum upload size is {settings.max_upload_mb}MB.",
                            "error": {
                                "code": "FILE_TOO_LARGE",
                                "message": f"File too large. Maximum upload size is {settings.max_upload_mb}MB.",
                                "request_id": req_id
                            }
                        }
                    )
            except ValueError:
                pass

        # 3. Per-client Rate Limiting for API routes
        if request.url.path.startswith("/api/"):
            client_ip = request.client.host if request.client else "unknown"
            allowed, remaining, retry_after = await _rate_limiter.check(client_ip)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                    content={
                        "success": False,
                        "detail": "Rate limit exceeded. Please retry later.",
                        "error": {
                            "code": "RATE_LIMIT_EXCEEDED",
                            "message": "Rate limit exceeded. Please retry later.",
                            "request_id": req_id
                        }
                    }
                )

        # 4. Process request
        start_time = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        # 5. Add Tracing & Security Headers (Hardened CSP without unsafe-inline for scripts)
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self' ws: wss:;"
        )

        # 6. Log non-static access
        if not request.url.path.startswith("/css") and not request.url.path.startswith("/js"):
            logger.info(
                f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.1f}ms) [id:{req_id}]"
            )

        return response
    finally:
        current_request_id.reset(token)


# ── CORS Middleware ────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── Standardized Error Handlers ────────────────────────────────────────────────
@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    req_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "detail": exc.message,
            "error": {
                "code": exc.code,
                "message": exc.message,
                "request_id": req_id
            }
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    req_id = getattr(request.state, "request_id", "unknown")

    code_map = {
        400: "INVALID_REQUEST",
        401: "UNAUTHORIZED",
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
    Minimized, lightweight health check endpoint.
    Exposes no platform internals, dependency versions, or full configuration.
    Detailed diagnostics are moved to the authenticated /api/system/details endpoint.
    """
    return {
        "status": "ok",
        "service": "VigilVoice",
        "timestamp": time.time(),
        "model_status": get_model_status(),
        "model": {
            "version": settings.model_version,
        },
        "configuration": {
            "environment": settings.environment,
        },
        "runtime": {
            "python_version": sys.version.split()[0],
        },
    }


@app.get("/api/ready", tags=["System"])
async def readiness_check():
    """Readiness probe verifying the backend is ready to accept detection requests."""
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
    """Returns server status, configuration, and model capabilities."""
    cnn_ready = is_cnn_model_available()
    uptime_sec = int(time.time() - _start_time)

    return {
        "server": "VigilVoice API",
        "version": "1.0.0",
        "uptime_seconds": uptime_sec,
        "debug_mode": settings.debug,
        "configuration": settings.public_dict(),
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


@app.get("/api/system/details", tags=["System"], dependencies=[Depends(verify_api_key)])
async def get_system_details():
    """
    Authenticated system details endpoint exposing configuration and runtime diagnostics.
    Requires valid API key when API_KEY is configured.
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


@app.get("/api/model/info", tags=["System"], dependencies=[Depends(verify_api_key)])
async def get_model_info():
    """Returns verified evaluation metrics, metadata, and robustness data (authenticated)."""
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
        raise AppError("INVALID_REQUEST", 422, "Invalid detection phase.")

    if force_verdict:
        if not settings.demo_mode:
            force_verdict = None
        else:
            force_verdict = force_verdict.lower().strip()
            if force_verdict not in VALID_DEMO_VERDICTS:
                raise AppError("INVALID_REQUEST", 422, "Invalid demo verdict override.")

    # 2. Validate Filename & Extension (Path Traversal Protection)
    safe_name = sanitize_filename(file.filename)
    safe_ext = get_safe_extension(safe_name)

    if not validate_format(safe_name) or safe_ext not in settings.allowed_extensions:
        raise AppError(
            "UNSUPPORTED_FORMAT",
            400,
            "Unsupported file format. Allowed: WAV, MP3, M4A, MP4, OGG, FLAC, AAC, WEBM"
        )

    # 3. Validate MIME type if provided
    if file.content_type and file.content_type.lower() not in settings.allowed_mime_types:
        if not file.content_type.startswith("audio/") and file.content_type.lower() != "video/mp4":
            raise AppError(
                "UNSUPPORTED_FORMAT",
                415,
                f"Unsupported MIME type '{file.content_type}'. Please upload a valid audio or MP4 file."
            )

    # 4. Check File Size (Read once up to max + 1 byte)
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) == 0:
        raise AppError("EMPTY_FILE", 400, "Uploaded audio file is empty (0 bytes).")
    if len(content) > settings.max_upload_bytes:
        raise AppError(
            "FILE_TOO_LARGE",
            413,
            f"File too large. Maximum upload size is {settings.max_upload_mb}MB."
        )

    # 5. Validate Magic Bytes
    if not validate_magic_bytes(content):
        raise AppError(
            "UNSUPPORTED_FORMAT",
            415,
            "Unsupported file format. Audio signature does not match allowed types."
        )

    # 6. Non-blocking try-acquire for Concurrency Semaphore
    if _inference_semaphore.locked():
        raise AppError(
            "CONCURRENCY_LIMIT_EXCEEDED",
            429,
            "Server busy: maximum concurrent audio analyses reached. Please retry."
        )
    await _inference_semaphore.acquire()

    temp_path = None
    worker_task = None
    start_time = time.perf_counter()

    try:
        # 7. Save to Unpredictable Temp File with already-read bytes (no double read)
        temp_path = save_temp_file(file, content=content)

        # 8. Check duration from metadata header before decode
        header_duration = get_audio_header_duration(temp_path)
        if header_duration is not None and header_duration > settings.max_audio_duration_seconds:
            raise AppError(
                "AUDIO_TOO_LONG",
                400,
                f"Audio too long (total duration: {header_duration:.2f}s > {settings.max_audio_duration_seconds}s)"
            )

        loop = asyncio.get_running_loop()

        # 9. Execute worker task in thread; keep semaphore and temp file until thread finishes
        worker_task = asyncio.create_task(
            _run_analysis_pipeline(temp_path, phase, force_verdict, safe_name, start_time, loop)
        )

        # Await worker shielded so timeout leaves worker running to finish cleanup
        result_payload = await asyncio.wait_for(
            asyncio.shield(worker_task),
            timeout=settings.inference_timeout_seconds
        )
        return result_payload

    except asyncio.TimeoutError:
        logger.error(f"Inference timeout after {settings.inference_timeout_seconds}s for {safe_name}")
        raise AppError(
            "INFERENCE_TIMEOUT",
            504,
            f"Audio analysis exceeded timeout limit of {settings.inference_timeout_seconds}s."
        )
    except AppError:
        if worker_task is None:
            if temp_path:
                remove_temp_file(temp_path)
            _inference_semaphore.release()
        raise
    except HTTPException:
        if worker_task is None:
            if temp_path:
                remove_temp_file(temp_path)
            _inference_semaphore.release()
        raise
    except ValueError as e:
        if worker_task is None:
            if temp_path:
                remove_temp_file(temp_path)
            _inference_semaphore.release()
        raise AppError("INVALID_AUDIO", 400, str(e)) from e
    except Exception as e:
        if worker_task is None:
            if temp_path:
                remove_temp_file(temp_path)
            _inference_semaphore.release()
        logger.exception(f"Internal error during audio analysis of {safe_name}: {e}")
        raise AppError("SERVER_ERROR", 500, "Internal server error during audio analysis.")


async def _run_analysis_pipeline(
    temp_path: str,
    phase: str,
    force_verdict: Optional[str],
    filename: str,
    start_time: float,
    loop: asyncio.AbstractEventLoop,
) -> dict:
    """Executes the canonical audio preprocessing and inference pipeline in a thread worker."""
    def _sync_pipeline():
        try:
            from backend.audio.pipeline import process_canonical
            from backend.audio.config import default_config

            preprocessing_result = process_canonical(temp_path, config=default_config)

            if preprocessing_result.quality.status == "INSUFFICIENT":
                reason = preprocessing_result.quality.reason or "Unknown reason"
                raise AppError(
                    "INVALID_AUDIO",
                    400,
                    f"Audio quality insufficient for analysis: {reason}"
                )

            y_clean = preprocessing_result.audio_data
            sr = default_config.sample_rate

            if len(y_clean) == 0:
                raise AppError(
                    "INVALID_AUDIO",
                    400,
                    "No readable speech found after voice activity detection."
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

            effective_phase = phase.lower().strip() if phase else "phase1"
            if effective_phase in ("phase3", "phase4"):
                effective_phase = "phase2"
            if effective_phase == "phase2" and not is_cnn_model_available():
                effective_phase = "phase1"

            segment_predictions = run_segment_detection(
                preprocessing_result.segments,
                phase=effective_phase,
                force_verdict=force_verdict
            )
            agg_strategy = settings.aggregation_strategy
            aggregation = aggregate_predictions(segment_predictions, strategy=agg_strategy)
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

            from backend.detection.detector import get_model_thresholds
            t_real, t_fake = get_model_thresholds()
            decision = make_decision(real_score, threshold_real=t_real, threshold_fake=t_fake)
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)
            segments_payload = [segment.to_dict() for segment in segment_predictions]

            return {
                "success": True,
                "filename": filename,
                "phase_used": effective_phase,
                "cnn_model_active": (effective_phase != "phase1") and is_cnn_model_available(),
                "model_status": get_model_status(),
                "demo_override": bool(force_verdict and settings.demo_mode),
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
                "segments": segments_payload,
                "score_timeline": [
                    {
                        "start_time": segment["start_time"],
                        "end_time": segment["end_time"],
                        "fake_probability": segment["fake_probability"],
                        "real_probability": segment["real_probability"],
                    }
                    for segment in segments_payload
                ],
                "aggregation": aggregation.to_dict(),
                "risk_report": risk_report,
                "security_recommendation": security_recommendation
            }
        finally:
            remove_temp_file(temp_path)
            try:
                loop.call_soon_threadsafe(_inference_semaphore.release)
            except RuntimeError:
                # If event loop was closed before thread finished
                pass

    return await asyncio.to_thread(_sync_pipeline)


# ── Static File Serving & SPA Fallback ─────────────────────────────────────────
_assets_dir = STATIC_DIR / "assets"
if _assets_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    # Never shadow API routes
    if full_path.startswith("api/"):
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": {"code": "NOT_FOUND", "message": "Unknown API endpoint."}},
        )

    if STATIC_DIR.exists():
        if full_path:
            candidate = (STATIC_DIR / full_path).resolve()
            try:
                candidate.relative_to(STATIC_DIR.resolve())
                if candidate.is_file():
                    return FileResponse(candidate)
            except ValueError:
                return JSONResponse(status_code=404, content={"detail": "Not found."})

        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)

    return JSONResponse(
        status_code=404,
        content={"success": False, "error": {"code": "NOT_FOUND", "message": "Frontend build not deployed."}},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
