"""
Authentication and security validation dependencies for VigilVoice API.
"""

from typing import Optional
from fastapi import Request, HTTPException, Security, WebSocket
from fastapi.security.api_key import APIKeyHeader

from backend.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(
    request: Request,
    api_key_header_val: Optional[str] = Security(api_key_header),
) -> bool:
    """
    Validates API key from X-API-Key header, Authorization Bearer header, or query param.
    When settings.api_key is configured, requests without a valid key are rejected with 401.
    """
    expected_key = settings.api_key
    if not expected_key:
        return True

    key = api_key_header_val or request.query_params.get("api_key")
    if not key:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            key = auth_header[7:].strip()

    if key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key."
        )

    return True


def validate_websocket_origin(websocket: WebSocket) -> bool:
    """
    Validates WebSocket Origin header against configured CORS origins.
    Protects against Cross-Site WebSocket Hijacking (CSWSH).
    """
    origin = websocket.headers.get("origin")
    if not origin:
        return True

    if "*" in settings.cors_origins:
        return True

    origin_normalized = origin.rstrip("/")
    allowed_origins = {o.rstrip("/") for o in settings.cors_origins}
    return origin_normalized in allowed_origins
