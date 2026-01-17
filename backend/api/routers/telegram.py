from __future__ import annotations

from typing import Any, Dict

import requests
from fastapi import APIRouter, HTTPException

from backend.api.schemas import (
    TelegramBotTestRequest,
    TelegramChatTestRequest,
    TelegramTestResponse,
)

router = APIRouter(prefix="/telegram", tags=["telegram"])


def _tg_url(token: str, method: str) -> str:
    token = (token or "").strip()
    return f"https://api.telegram.org/bot{token}/{method}"


def _safe_strip(s: str) -> str:
    return (s or "").strip()


@router.post("/test-bot", response_model=TelegramTestResponse)
def test_bot(req: TelegramBotTestRequest) -> TelegramTestResponse:
    token = _safe_strip(req.telegram_api_token)
    if not token:
        raise HTTPException(status_code=422, detail="telegram_api_token is required")

    try:
        resp = requests.get(_tg_url(token, "getMe"), timeout=req.timeout_seconds)
        resp.raise_for_status()
        payload: Dict[str, Any] = resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Telegram API request failed: {e}")

    return TelegramTestResponse(
        ok=bool(payload.get("ok", False)),
        method="getMe",
        telegram=payload,
        details={"bot_username": payload.get("result", {}).get("username")},
    )


@router.post("/test-chat", response_model=TelegramTestResponse)
def test_chat(req: TelegramChatTestRequest) -> TelegramTestResponse:
    token = _safe_strip(req.telegram_api_token)
    chat_id = _safe_strip(req.telegram_chat_id)

    if not token:
        raise HTTPException(status_code=422, detail="telegram_api_token is required")
    if not chat_id:
        raise HTTPException(status_code=422, detail="telegram_chat_id is required")

    method = "getChat" if req.mode == "getChat" else "sendMessage"

    try:
        if method == "getChat":
            resp = requests.post(
                _tg_url(token, "getChat"),
                data={"chat_id": chat_id},
                timeout=req.timeout_seconds,
            )
        else:
            resp = requests.post(
                _tg_url(token, "sendMessage"),
                data={"chat_id": chat_id, "text": req.text or "ping"},
                timeout=req.timeout_seconds,
            )

        resp.raise_for_status()
        payload: Dict[str, Any] = resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Telegram API request failed: {e}")

    result = payload.get("result") or {}
    chat = result.get("chat") if isinstance(result, dict) else {}

    details: Dict[str, Any] = {"mode": req.mode}
    if isinstance(chat, dict):
        details.update(
            {
                "chat_title": chat.get("title"),
                "chat_type": chat.get("type"),
                "chat_id": chat.get("id"),
            }
        )

    return TelegramTestResponse(
        ok=bool(payload.get("ok", False)),
        method=method,
        telegram=payload,
        details=details,
    )
