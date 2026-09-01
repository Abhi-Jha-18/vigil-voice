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


def save_temp_file(upload_file: UploadFile) -> str:
    """
    Securely saves an uploaded audio stream to an unpredictable temporary file.
    Guarantees storage within system temp directory with no user-controlled path execution.
    """
    safe_ext = get_safe_extension(upload_file.filename)
    unique_name = f"vigilvoice_{uuid.uuid4().hex}{safe_ext}"
    temp_dir = tempfile.gettempdir()
    target_path = os.path.join(temp_dir, unique_name)

    with open(target_path, "wb") as buffer:
        while chunk := upload_file.file.read(65536):
            buffer.write(chunk)
            
    # Reset file pointer for any downstream consumption
    upload_file.file.seek(0)
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
