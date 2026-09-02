"""Centralized, environment-driven configuration for VigilVoice."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_origins(value: str | None, environment: str = "development") -> tuple[str, ...]:
    if value:
        return tuple(origin.strip() for origin in value.split(",") if origin.strip())
    if environment.lower() == "production":
        return ("https://vigilvoice.internal",)
    return ("http://127.0.0.1:8000", "http://localhost:8000", "http://localhost:3000")


def _resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


ALLOWED_AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".flac", ".webm", ".aac"})
ALLOWED_AUDIO_MIME_TYPES = frozenset({
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mpeg", "audio/mp3", "audio/mp4", "audio/x-m4a",
    "audio/ogg", "audio/flac", "audio/x-flac",
    "audio/webm", "audio/aac", "video/mp4", "application/octet-stream"
})


@dataclass(frozen=True)
class Settings:
    environment: str
    host: str
    port: int
    debug: bool
    log_level: str
    sample_rate: int
    max_upload_mb: int
    max_audio_duration_seconds: float
    inference_timeout_seconds: float
    max_concurrent_inferences: int
    segment_duration_seconds: float
    segment_overlap_seconds: float
    model_path: Path
    model_version: str
    cors_origins: tuple[str, ...]
    demo_mode: bool
    vad_top_db: int
    # Phase 15: Live Calling & Incident Response
    live_sample_rate: int
    live_window_seconds: float
    live_hop_seconds: float
    live_high_risk_threshold: float
    live_suspicious_threshold: float
    live_min_suspicious_windows: int
    live_max_session_seconds: int
    live_max_buffer_mb: float

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def allowed_extensions(self) -> frozenset[str]:
        return ALLOWED_AUDIO_EXTENSIONS

    @property
    def allowed_mime_types(self) -> frozenset[str]:
        return ALLOWED_AUDIO_MIME_TYPES

    def public_dict(self) -> dict[str, object]:
        """Return non-secret settings safe to expose through the status endpoint."""
        return {
            "environment": self.environment,
            "sample_rate_hz": self.sample_rate,
            "max_upload_mb": self.max_upload_mb,
            "max_audio_duration_seconds": self.max_audio_duration_seconds,
            "inference_timeout_seconds": self.inference_timeout_seconds,
            "max_concurrent_inferences": self.max_concurrent_inferences,
            "segment_duration_seconds": self.segment_duration_seconds,
            "segment_overlap_seconds": self.segment_overlap_seconds,
            "model_path": str(self.model_path),
            "model_version": self.model_version,
            "cors_origins": list(self.cors_origins),
            "demo_mode": self.demo_mode,
            "live_window_seconds": self.live_window_seconds,
            "live_hop_seconds": self.live_hop_seconds,
            "live_high_risk_threshold": self.live_high_risk_threshold,
            "live_suspicious_threshold": self.live_suspicious_threshold,
        }


def _resolve_model_path() -> Path:
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return _resolve_project_path(env_path)
    primary = PROJECT_ROOT / "models" / "production" / "vigilvoice_cnn_best.pth"
    if primary.is_file():
        return primary
    fallback = PROJECT_ROOT / "models" / "cnn_weight.pth"
    if fallback.is_file():
        return fallback
    return primary


def load_settings() -> Settings:
    env = os.getenv("ENVIRONMENT", "development").strip().lower()
    settings = Settings(
        environment=env,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        debug=_as_bool(os.getenv("DEBUG"), env != "production"),
        log_level=os.getenv("LOG_LEVEL", "info"),
        sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "50")),
        max_audio_duration_seconds=float(os.getenv("MAX_AUDIO_DURATION_SECONDS", "120")),
        inference_timeout_seconds=float(os.getenv("INFERENCE_TIMEOUT", "60.0")),
        max_concurrent_inferences=int(os.getenv("MAX_CONCURRENT_INFERENCES", "10")),
        segment_duration_seconds=float(os.getenv("SEGMENT_DURATION_SECONDS", "4")),
        segment_overlap_seconds=float(os.getenv("SEGMENT_OVERLAP_SECONDS", "1")),
        model_path=_resolve_model_path(),
        model_version=os.getenv("MODEL_VERSION", "v1.0-production"),
        cors_origins=_as_origins(os.getenv("CORS_ORIGINS"), env),
        demo_mode=_as_bool(os.getenv("DEMO_MODE"), env != "production"),
        vad_top_db=int(os.getenv("VAD_TOP_DB", "30")),
        # Phase 15 settings
        live_sample_rate=int(os.getenv("LIVE_AUDIO_SAMPLE_RATE", "16000")),
        live_window_seconds=float(os.getenv("LIVE_WINDOW_SECONDS", "2.5")),
        live_hop_seconds=float(os.getenv("LIVE_HOP_SECONDS", "1.0")),
        live_high_risk_threshold=float(os.getenv("LIVE_HIGH_RISK_THRESHOLD", "0.80")),
        live_suspicious_threshold=float(os.getenv("LIVE_SUSPICIOUS_THRESHOLD", "0.60")),
        live_min_suspicious_windows=int(os.getenv("LIVE_MIN_SUSPICIOUS_WINDOWS", "2")),
        live_max_session_seconds=int(os.getenv("LIVE_MAX_SESSION_SECONDS", "3600")),
        live_max_buffer_mb=float(os.getenv("LIVE_MAX_BUFFER_MB", "10.0")),
    )
    if settings.sample_rate <= 0 or settings.max_upload_mb <= 0:
        raise ValueError("SAMPLE_RATE and MAX_UPLOAD_MB must be positive.")
    if settings.max_audio_duration_seconds <= 0 or settings.segment_duration_seconds <= 0:
        raise ValueError("Audio and segment durations must be positive.")
    if not 0 <= settings.segment_overlap_seconds < settings.segment_duration_seconds:
        raise ValueError("SEGMENT_OVERLAP_SECONDS must be >= 0 and below segment duration.")
    if settings.inference_timeout_seconds <= 0 or settings.max_concurrent_inferences <= 0:
        raise ValueError("INFERENCE_TIMEOUT and MAX_CONCURRENT_INFERENCES must be positive.")
    return settings


settings = load_settings()
