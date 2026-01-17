from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class CameraRequisition(BaseModel):
    rtsp_link: str = Field(..., description="RTSP stream URL.")
    camera_nickname: str = Field(..., description="Camera nickname.")
    telegram_api_token: str = Field(..., description="Telegram bot token.")
    telegram_chat_id: str = Field(..., description="Telegram chat id.")
    threshold: float = Field(0.5, description="Match threshold.")
    cooldown_seconds: int = Field(3600, description="Cooldown (seconds).")


class TelegramBotTestRequest(BaseModel):
    telegram_api_token: str = Field(..., description="Telegram bot token to validate.")
    timeout_seconds: int = Field(10, ge=1, le=60, description="HTTP timeout for Telegram API calls.")


class TelegramChatTestRequest(BaseModel):
    telegram_api_token: str = Field(..., description="Telegram bot token.")
    telegram_chat_id: str = Field(..., description="Chat ID to validate (group/private/channel).")
    mode: Literal["getChat", "sendMessage"] = Field(
        "getChat",
        description="Validation mode: getChat validates visibility; sendMessage validates ability to send.",
    )
    text: str = Field("ping", description="Message text used only when mode=sendMessage.")
    timeout_seconds: int = Field(10, ge=1, le=60, description="HTTP timeout for Telegram API calls.")


class TelegramTestResponse(BaseModel):
    ok: bool = Field(..., description="True if Telegram returned ok=true.")
    method: str = Field(..., description="Telegram method used (getMe/getChat/sendMessage).")
    telegram: Dict[str, Any] = Field(..., description="Raw Telegram API response JSON.")
    details: Optional[Dict[str, Any]] = Field(None, description="Extra normalized info.")
