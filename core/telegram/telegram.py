import requests
import os

class TelegramClient:    
    @staticmethod
    def send_telegram_image(
        telegram_api_token: str,
        telegram_chat_id: str,
        image_path: str,
        caption: str = "",
        timeout_sec: float = 15.0,
    ) -> None:
        if not telegram_api_token or not telegram_chat_id:
            raise RuntimeError("Telegram token/chat_id are required to send an image.")
        if not os.path.isfile(image_path):
            raise RuntimeError(f"Image path does not exist: {image_path}")

        url = f"https://api.telegram.org/bot{telegram_api_token}/sendPhoto"

        with open(image_path, "rb") as f:
            files = {"photo": f}
            data = {
                "chat_id": telegram_chat_id,
                "caption": caption or "",
                "parse_mode": "Markdown",
            }
            resp = requests.post(url, data=data, files=files, timeout=timeout_sec)

        if resp.status_code != 200:
            raise RuntimeError(f"Telegram HTTP {resp.status_code}: {resp.text}")

        payload = resp.json()
        if not payload.get("ok", False):
            raise RuntimeError(f"Telegram API error: {payload}")