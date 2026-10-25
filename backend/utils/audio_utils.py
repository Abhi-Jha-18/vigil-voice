"""Audio utility functions for secure temporary file handling and path sanitization."""

import gc
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from fastapi import UploadFile

from backend.config import settings, ALLOWED_AUDIO_EXTENSIONS


def sanitize_filename(filename: str | None) -> str:
    """
    Sanitizes an untrusted filename to prevent path traversal and shell injection.
    Strips directory separators, relative traversal sequences, and control characters.
    """
    if not filename:
        return "audio.wav"
    
    # Strip directory components (both POSIX and Windows style)
    base = os.path.basename(filename.replace("\\", "/"))
    
    # Remove traversal patterns and dangerous characters
    clean = re.sub(r'[\r\n\t\0]', '', base)
    clean = re.sub(r'\.{2,}', '.', clean)  # collapse multiple dots
    
    # If the filename becomes empty or dot-only, default safely
    if not clean or clean.startswith("."):
        clean = "upload" + (os.path.splitext(base)[1] or ".wav")
    
    return clean


def get_safe_extension(filename: str | None) -> str:
    """Extracts and validates file extension against allowed audio extensions."""
    if not filename:
        return ".wav"
    ext = os.path.splitext(filename)[1].lower()
    return ext if ext in ALLOWED_AUDIO_EXTENSIONS else ".wav"


def validate_magic_bytes(content: bytes) -> bool:
    """
    Validates file magic numbers against supported audio formats.
    Supported: WAV, MP3, FLAC, OGG, MP4/M4A, WEBM, AAC.
    """
    if not content or len(content) < 4:
        return False

    # WAV: RIFF....WAVE
    if content.startswith(b"RIFF") and len(content) >= 12 and content[8:12] == b"WAVE":
        return True

    # FLAC: fLaC
    if content.startswith(b"fLaC"):
        return True

    # OGG: OggS
    if content.startswith(b"OggS"):
        return True

    # MP4 / M4A: ftyp atom at offset 4
    if len(content) >= 8 and content[4:8] == b"ftyp":
        return True

    # WEBM / Matroska: \x1a\x45\xdf\xa3
    if content.startswith(b"\x1a\x45\xdf\xa3"):
        return True

    # MP3: ID3 header
    if content.startswith(b"ID3"):
        return True

    # MP3 / AAC ADTS: Frame sync (0xFF followed by 3 bits set)
    if len(content) >= 2 and content[0] == 0xFF and (content[1] & 0xE0) == 0xE0:
        return True

    return False


def get_audio_header_duration(filepath: str | Path) -> float | None:
    """
    Reads audio duration directly from container/file metadata headers
    without decoding uncompressed sample data into memory.
    """
    path_str = str(filepath)
    if not os.path.exists(path_str) or os.path.getsize(path_str) == 0:
        return None

    # 1. Try soundfile.info (fast header read for WAV, FLAC, OGG, etc.)
    try:
        import soundfile as sf
        info = sf.info(path_str)
        if info.duration is not None and info.duration > 0:
            return float(info.duration)
    except Exception:
        pass

    # 2. Try PyAV container duration (for MP4, M4A, WEBM, MP3 containers)
    try:
        import av
        container = av.open(path_str)
        duration = None
        if container.duration is not None:
            duration = float(container.duration / av.time_base)
        else:
            for s in container.streams:
                if s.type == "audio" and s.duration and s.time_base:
                    duration = float(s.duration * s.time_base)
                    break
        container.close()
        if duration is not None and duration > 0:
            return float(duration)
    except Exception:
        pass

    return None


def save_temp_file(
    upload_file: UploadFile | None = None,
    content: bytes | None = None,
    filename: str | None = None,
) -> str:
    """
    Securely saves uploaded audio bytes or stream to an unpredictable temporary file.
    Guarantees storage within system temp directory with no user-controlled path execution.
    If content is provided, writes directly without reading the upload stream twice.
    """
    target_name = filename or (upload_file.filename if upload_file else None)
    safe_ext = get_safe_extension(target_name)
    unique_name = f"vigilvoice_{uuid.uuid4().hex}{safe_ext}"
    temp_dir = tempfile.gettempdir()
    target_path = os.path.join(temp_dir, unique_name)

    if content is not None:
        with open(target_path, "wb") as buffer:
            buffer.write(content)
    elif upload_file is not None:
        with open(target_path, "wb") as buffer:
            while chunk := upload_file.file.read(65536):
                buffer.write(chunk)
        upload_file.file.seek(0)
    else:
        raise ValueError("Either content or upload_file must be provided.")

    return target_path


def remove_temp_file(filepath: str | Path | None, retries: int = 5, delay: float = 0.2) -> bool:
    """
    Safely removes a temporary file with retry logic to handle OS file-locking (especially Windows).
    """
    if not filepath:
        return True
        
    path_str = str(filepath)
    if not os.path.exists(path_str):
        return True

    for attempt in range(retries):
        try:
            gc.collect()  # Force garbage collection to release file handles
            os.remove(path_str)
            return True
        except PermissionError:
            if attempt < retries - 1:
                time.sleep(delay)
        except Exception:
            break
            
    return not os.path.exists(path_str)
