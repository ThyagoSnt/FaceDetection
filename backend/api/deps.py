from __future__ import annotations

import os
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


# Load AUTH_TOKEN from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


AUTH_TOKEN = os.getenv("AUTH_TOKEN")
if not AUTH_TOKEN:
    raise RuntimeError("AUTH_TOKEN is not set. Please define it in your .env file.")


# Bearer token auth (Authorization: Bearer <AUTH_TOKEN>)
_bearer = HTTPBearer(auto_error=False)


def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> None:
    """
    Validates 'Authorization: Bearer <token>' against AUTH_TOKEN.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header (expected: Bearer <token>).",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid auth scheme (expected: Bearer).",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.credentials != AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid token.",
        )
