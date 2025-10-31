import os
import json
from dataclasses import dataclass, field
from typing import List, Union, Any, Dict, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv, find_dotenv


# =========================
# Telegram helper (getUpdates -> map de chats)
# =========================

def list_telegram_chats(bot_token: str, timeout: int = 20) -> Dict[str, int]:
    """
    Chama Telegram Bot API getUpdates e extrai um mapeamento
    { 'nome-do-chat (tipo)': chat_id } cobrindo private/group/supergroup/channel.
    """
    if not bot_token:
        raise ValueError("Telegram bot token é obrigatório.")

    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    if not payload.get("ok", False):
        raise RuntimeError(f"Erro na API do Telegram: {payload}")

    # Dedup por chat_id
    id_to_label: Dict[int, str] = {}

    def _label_for_chat(chat: Dict[str, Any]) -> Tuple[str, int]:
        chat_id = chat.get("id")
        chat_type = chat.get("type")
        title = chat.get("title")

        if title:
            name = title
        else:
            first = (chat.get("first_name") or "").strip()
            last = (chat.get("last_name") or "").strip()
            username = chat.get("username")
            if first or last:
                name = (first + " " + last).strip()
            elif username:
                name = f"@{username}"
            else:
                name = str(chat_id)  # fallback

        label = f"{name} ({chat_type})" if chat_type else name
        return label, chat_id

    def _try_add(chat_obj: Any) -> None:
        if not isinstance(chat_obj, dict):
            return
        label, cid = _label_for_chat(chat_obj)
        # Guarda o primeiro label visto para cada id
        if cid not in id_to_label:
            id_to_label[cid] = label

    # Varre campos comuns que podem conter chat
    for upd in payload.get("result", []):
        if "message" in upd and isinstance(upd["message"], dict):
            _try_add(upd["message"].get("chat"))

        if "edited_message" in upd and isinstance(upd["edited_message"], dict):
            _try_add(upd["edited_message"].get("chat"))

        if "channel_post" in upd and isinstance(upd["channel_post"], dict):
            _try_add(upd["channel_post"].get("chat"))

        if "edited_channel_post" in upd and isinstance(upd["edited_channel_post"], dict):
            _try_add(upd["edited_channel_post"].get("chat"))

        if "my_chat_member" in upd and isinstance(upd["my_chat_member"], dict):
            _try_add(upd["my_chat_member"].get("chat"))

        if "chat_member" in upd and isinstance(upd["chat_member"], dict):
            _try_add(upd["chat_member"].get("chat"))

        if "callback_query" in upd and isinstance(upd["callback_query"], dict):
            msg = upd["callback_query"].get("message") or {}
            _try_add(msg.get("chat"))

    # Converte para {label: id}, ordenado por label
    label_to_id: Dict[str, int] = {}
    for cid, label in sorted(id_to_label.items(), key=lambda kv: kv[1].lower()):
        label_to_id[label] = cid

    return label_to_id


# =========================
# Configuration
# =========================

class AppConfig:
    """Loads application configuration from environment variables (.env)."""

    def __init__(self, utilities_url: str, camera_scheduler_url: str, auth_token: str, timeout: int = 60):
        self.utilities_url = utilities_url.rstrip("/") if utilities_url else ""
        self.camera_scheduler_url = camera_scheduler_url.rstrip("/") if camera_scheduler_url else ""
        self.auth_token = auth_token
        self.timeout = timeout

    @classmethod
    def from_env(cls, timeout: int = 60) -> "AppConfig":
        """Factory method that loads values from .env/environment."""
        load_dotenv(find_dotenv(".env"), override=True)
        utilities_url = os.getenv("UTILITIES_URL", "")
        camera_scheduler_url = os.getenv("CAMERA_SCHEDULER_URL", "")
        auth_token = os.getenv("AUTH_TOKEN", "")
        return cls(utilities_url, camera_scheduler_url, auth_token, timeout)

    def validate_or_raise(self) -> None:
        """Validates mandatory fields; raises if missing."""
        missing = [k for k, v in {
            "UTILITIES_URL": self.utilities_url,
            "CAMERA_SCHEDULER_URL": self.camera_scheduler_url,
            "AUTH_TOKEN": self.auth_token
        }.items() if not v]
        if missing:
            raise RuntimeError(
                "Missing required configuration in .env: " + ", ".join(missing)
            )


# =========================
# Domain Models
# =========================

@dataclass
class CameraJob:
    """Represents a camera analysis job payload."""
    rtsp_link: str
    camera_nickname: str
    telegram_api_token: str
    telegram_chat_id: Union[str, int]
    threshold: float = 0.5
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """Converts to API payload dict."""
        base = {
            "rtsp_link": self.rtsp_link,
            "camera_nickname": self.camera_nickname,
            "telegram_api_token": self.telegram_api_token,
            "telegram_chat_id": str(self.telegram_chat_id),
            "threshold": float(self.threshold),
        }
        base.update(self.extra or {})
        return base

    def validate_or_raise(self) -> None:
        """Simple validation for mandatory fields."""
        if not self.rtsp_link:
            raise ValueError("rtsp_link is required")
        if not self.camera_nickname:
            raise ValueError("camera_nickname is required")
        if not self.telegram_api_token:
            raise ValueError("telegram_api_token is required")
        if self.telegram_chat_id in (None, ""):
            raise ValueError("telegram_chat_id is required")
        if not (0.0 <= float(self.threshold) <= 1.0):
            raise ValueError("threshold must be between 0.0 and 1.0")


# =========================
# Service Client (HTTP)
# =========================

class FaceUtilitiesClient:
    """HTTP client that talks to Utilities and Camera Scheduler services."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.headers = {"Authorization": f"Bearer {self.config.auth_token}"} if self.config.auth_token else {}

    # -------- Utilities (disk paths, legacy) --------
    def add_face(self, person_id: str, image_paths: Union[str, List[str]]) -> dict:
        """Adds face embeddings using file paths (kept for backward compatibility)."""
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        files = []
        file_handlers = []
        try:
            for p in image_paths:
                fh = open(p, "rb")
                file_handlers.append(fh)
                files.append(("image", (os.path.basename(p), fh, "image/jpeg")))

            data = {"action": "add", "person_id": person_id}
            resp = requests.post(
                self.config.utilities_url,
                headers=self.headers,
                files=files,
                data=data,
                timeout=self.config.timeout
            )
            resp.raise_for_status()
            return resp.json()
        finally:
            for fh in file_handlers:
                try:
                    fh.close()
                except Exception:
                    pass

    def retrieve_face(self, image_path: str) -> dict:
        """Recognizes a face using a file path (kept for backward compatibility)."""
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
            data = {"action": "retrieve"}
            resp = requests.post(
                self.config.utilities_url,
                headers=self.headers,
                files=files,
                data=data,
                timeout=self.config.timeout
            )
        resp.raise_for_status()
        return resp.json()

    # (ADDED) -------- Utilities: remove face --------
    def remove_face(self, person_id: str) -> dict:
        """Removes a face/person by person_id (server must support action=remove)."""
        data = {"action": "remove", "person_id": person_id}
        resp = requests.post(
            self.config.utilities_url,
            headers=self.headers,
            data=data,
            timeout=self.config.timeout
        )
        resp.raise_for_status()
        return resp.json()

    # -------- Utilities (in-memory, preferred) --------
    def add_face_from_memory(self, person_id: str, images: List[Tuple[str, bytes, str]]) -> dict:
        """
        Adds face embeddings using in-memory images.
        `images` must be a list of tuples: (filename, bytes_data, mime_type).
        """
        if not images:
            raise ValueError("No images provided")

        files = [("image", (name, data, mime or "application/octet-stream"))
                 for (name, data, mime) in images]

        data = {"action": "add", "person_id": person_id}
        resp = requests.post(
            self.config.utilities_url,
            headers=self.headers,
            files=files,
            data=data,
            timeout=self.config.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def retrieve_face_from_memory(self, image: Tuple[str, bytes, str]) -> dict:
        """
        Recognizes a face using an in-memory image.
        `image` is a tuple: (filename, bytes_data, mime_type).
        """
        name, data, mime = image
        files = {"image": (name, data, mime or "application/octet-stream")}
        payload = {"action": "retrieve"}
        resp = requests.post(
            self.config.utilities_url,
            headers=self.headers,
            files=files,
            data=payload,
            timeout=self.config.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def camera_status(self, camera_id: str) -> dict:
        """Queries camera status by camera_id."""
        data = {"action": "camera_status", "camera_id": camera_id}
        resp = requests.post(
            self.config.utilities_url,
            headers=self.headers,
            data=data,
            timeout=self.config.timeout
        )
        resp.raise_for_status()
        return resp.json()

    # -------- Camera Scheduler --------
    def start_camera_single(self, job: CameraJob) -> dict:
        """Starts a single camera analysis job."""
        job.validate_or_raise()
        payload = job.to_payload()
        resp = requests.post(
            self.config.camera_scheduler_url,
            json=payload,
            headers=self.headers,
            timeout=self.config.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def start_camera_batch(self, jobs: List[CameraJob]) -> dict:
        """Starts a batch of camera analysis jobs."""
        payload_list = []
        for job in jobs:
            job.validate_or_raise()
            payload_list.append(job.to_payload())

        resp = requests.post(
            self.config.camera_scheduler_url,
            json=payload_list,
            headers=self.headers,
            timeout=self.config.timeout
        )
        resp.raise_for_status()
        return resp.json()


# =========================
# Streamlit UI (Portuguese)
# =========================

class StreamlitApp:
    """Streamlit front-end in Portuguese, backed by the English service code."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.client = FaceUtilitiesClient(config)

    def run(self) -> None:
        """Entry point to render the app."""
        st.set_page_config(page_title="👤 Detecção Facial", layout="wide")
        st.title("Utilitários")

        # Validate configuration
        try:
            self.config.validate_or_raise()
        except Exception as e:
            st.error(str(e))
            st.stop()

        # Sidebar menu
        choice = st.sidebar.radio(
            "Escolha uma ação:",
            [
                "Adicionar Rosto",
                "Consultar Rosto",
                "Remover Rosto",
                "Câmera - Status",
                "Câmera - Iniciar",
                "Telegram - Chat IDs",
            ],
        )

        # Dispatch
        if choice == "Adicionar Rosto":
            self.page_add_face()
        elif choice == "Consultar Rosto":
            self.page_retrieve_face()
        elif choice == "Remover Rosto":
            self.page_remove_face()
        elif choice == "Câmera - Status":
            self.page_camera_status()
        elif choice == "Câmera - Iniciar":
            self.page_camera_start()
        elif choice == "Telegram - Chat IDs":
            self.page_telegram_list_chat_ids()

    # -------- Pages --------

    def page_add_face(self) -> None:
        st.header("Adicionar Rosto (1 ou várias imagens)")
        person_id = st.text_input("Person ID (opcional)", "web_person")
        uploads = st.file_uploader(
            "Envie uma ou mais imagens",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if uploads and st.button("Adicionar"):
            try:
                # Convert Streamlit uploads into (filename, bytes, mime) tuples
                images_mem: List[Tuple[str, bytes, str]] = []
                for uf in uploads:
                    # IMPORTANT: getvalue() returns the full bytes (no temp files)
                    data = uf.getvalue()
                    if not data:
                        # In case it was read elsewhere, ensure we read the buffer
                        data = uf.read()
                    mime = uf.type or "application/octet-stream"
                    name = uf.name or "upload.bin"
                    images_mem.append((name, data, mime))

                result = self.client.add_face_from_memory(person_id or "web_person", images_mem)
                st.json(result)
            except requests.HTTPError as e:
                st.error(f"Erro HTTP: {e.response.text if e.response is not None else str(e)}")
            except Exception as e:
                st.error(f"Erro: {e}")

    def page_retrieve_face(self) -> None:
        st.header("Consultar / Reconhecer Rosto (única imagem)")
        upload = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])

        if upload and st.button("Consultar"):
            try:
                data = upload.getvalue() or upload.read()
                mime = upload.type or "application/octet-stream"
                name = upload.name or "upload.bin"
                result = self.client.retrieve_face_from_memory((name, data, mime))
                st.json(result)
            except requests.HTTPError as e:
                st.error(f"Erro HTTP: {e.response.text if e.response is not None else str(e)}")
            except Exception as e:
                st.error(f"Erro: {e}")

    def page_remove_face(self) -> None:
        st.header("Remover Rosto")
        person_id = st.text_input("Person ID")
        if person_id and st.button("Remover"):
            try:
                result = self.client.remove_face(person_id)
                st.json(result)
            except requests.HTTPError as e:
                st.error(f"Erro HTTP: {e.response.text if e.response is not None else str(e)}")
            except Exception as e:
                st.error(f"Erro: {e}")

    def page_camera_status(self) -> None:
        st.header("Status da Câmera")
        camera_id = st.text_input("Camera ID")
        if camera_id and st.button("Ver Status"):
            try:
                result = self.client.camera_status(camera_id)
                st.json(result)
            except requests.HTTPError as e:
                st.error(f"Erro HTTP: {e.response.text if e.response is not None else str(e)}")
            except Exception as e:
                st.error(f"Erro: {e}")

    def page_camera_start(self) -> None:
        st.header("Iniciar Análise da Câmera (Única ou Lote)")

        mode = st.radio("Modo:", ["Única", "Lote (JSON)"], horizontal=True)

        if mode == "Única":
            rtsp = st.text_input("Link RTSP")
            nickname = st.text_input("Apelido da Câmera")
            telegram_token = st.text_input("Telegram Bot Token", type="password")
            telegram_chat_id = st.text_input("Telegram Chat ID")
            threshold = st.number_input("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

            if all([rtsp, nickname, telegram_token, telegram_chat_id]) and st.button("Iniciar Única"):
                try:
                    job = CameraJob(
                        rtsp_link=rtsp,
                        camera_nickname=nickname,
                        telegram_api_token=telegram_token,
                        telegram_chat_id=telegram_chat_id,
                        threshold=float(threshold),
                    )
                    result = self.client.start_camera_single(job)
                    st.json(result)
                except requests.HTTPError as e:
                    st.error(f"Erro HTTP: {e.response.text if e.response is not None else str(e)}")
                except Exception as e:
                    st.error(f"Erro: {e}")

        else:
            st.markdown("Cole um **array JSON** de jobs (cada item deve conter os campos esperados pelo scheduler).")
            example = [
                {
                    "rtsp_link": "rtsp://user:pass@ip:port/stream1",
                    "camera_nickname": "Cam01",
                    "telegram_api_token": "<token>",
                    "telegram_chat_id": "<chat_id>",
                    "threshold": 0.55
                },
                {
                    "rtsp_link": "rtsp://user:pass@ip:port/stream2",
                    "camera_nickname": "Cam02",
                    "telegram_api_token": "<token>",
                    "telegram_chat_id": "<chat_id>",
                    "threshold": 0.60
                }
            ]
            json_text = st.text_area("Payload JSON (lote)", value=json.dumps(example, indent=2), height=260)

            if st.button("Iniciar Lote"):
                try:
                    raw = json.loads(json_text)
                    if not isinstance(raw, list):
                        st.error("O payload deve ser um **array JSON** (lista).")
                        return

                    jobs: List[CameraJob] = []
                    for item in raw:
                        jobs.append(CameraJob(
                            rtsp_link=item.get("rtsp_link", ""),
                            camera_nickname=item.get("camera_nickname", ""),
                            telegram_api_token=item.get("telegram_api_token", ""),
                            telegram_chat_id=item.get("telegram_chat_id", ""),
                            threshold=float(item.get("threshold", 0.5)),
                            extra={k: v for k, v in item.items()
                                   if k not in {"rtsp_link", "camera_nickname", "telegram_api_token", "telegram_chat_id", "threshold"}}
                        ))

                    result = self.client.start_camera_batch(jobs)
                    st.json(result)
                except requests.HTTPError as e:
                    st.error(f"Erro HTTP: {e.response.text if e.response is not None else str(e)}")
                except Exception as e:
                    st.error(f"JSON inválido ou erro na requisição: {e}")

    def page_telegram_list_chat_ids(self) -> None:
        st.header("Telegram - Chat IDs (via getUpdates)")
        st.markdown(
            "Cole o **token do bot** (do @BotFather). Eu busco os chats que "
            "enviaram mensagens recentes ao bot e listo como `nome-do-chat : id`."
        )

        token = st.text_input("Telegram Bot Token", type="password")
        if st.button("Listar Chats"):
            if not token:
                st.error("Informe o token do bot.")
                return

            try:
                mapping = list_telegram_chats(token, timeout=self.config.timeout or 20)
                if not mapping:
                    st.info(
                        "Nenhum chat encontrado nos updates.\n\n"
                        "- Envie uma **mensagem nova** para o bot (privado ou no grupo)\n"
                        "- Em grupos, **desative o Group Privacy** no @BotFather e envie algo no grupo\n"
                        "- Se usa webhook, chame `deleteWebhook` e tente novamente"
                    )
                    return

                lines = [f"{name} : {chat_id}" for name, chat_id in mapping.items()]
                st.subheader("Possíveis IDs de conversa")
                st.code("\n".join(lines))

            except requests.HTTPError as e:
                st.error(f"Erro HTTP: {e.response.text if e.response is not None else str(e)}")
            except Exception as e:
                st.error(f"Erro: {e}")


# =========================
# Main
# =========================

def main() -> None:
    app = StreamlitApp(AppConfig.from_env())
    app.run()


if __name__ == "__main__":
    main()
