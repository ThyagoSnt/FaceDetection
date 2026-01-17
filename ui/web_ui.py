# web_ui.py
import os
import json
from dataclasses import dataclass, field
from typing import List, Union, Any, Dict, Tuple, Optional

import requests
import streamlit as st
from dotenv import load_dotenv, find_dotenv


def list_telegram_chats(bot_token: str, timeout: int = 20) -> Dict[str, int]:
    """
    Calls Telegram Bot API getUpdates and extracts a mapping:
    { 'chat-name (type)': chat_id } covering private/group/supergroup/channel.
    """
    if not bot_token:
        raise ValueError("Telegram bot token is required.")

    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    if not payload.get("ok", False):
        raise RuntimeError(f"Telegram API error: {payload}")

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
                name = str(chat_id)

        label = f"{name} ({chat_type})" if chat_type else name
        return label, chat_id

    def _try_add(chat_obj: Any) -> None:
        if not isinstance(chat_obj, dict):
            return
        label, cid = _label_for_chat(chat_obj)
        if cid not in id_to_label:
            id_to_label[cid] = label

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

    label_to_id: Dict[str, int] = {}
    for cid, label in sorted(id_to_label.items(), key=lambda kv: kv[1].lower()):
        label_to_id[label] = cid

    return label_to_id


class AppConfig:
    """Loads application configuration from environment variables (.env)."""

    def __init__(self, utilities_url: str, camera_scheduler_url: str, auth_token: str, timeout: int = 60):
        self.utilities_url = utilities_url.rstrip("/") if utilities_url else ""
        self.camera_scheduler_url = camera_scheduler_url.rstrip("/") if camera_scheduler_url else ""
        self.auth_token = auth_token or ""
        self.timeout = timeout

    @classmethod
    def from_env(cls, timeout: int = 60) -> "AppConfig":
        load_dotenv(find_dotenv(".env"), override=True)
        utilities_url = os.getenv("UTILITIES_URL", "")
        camera_scheduler_url = os.getenv("CAMERA_SCHEDULER_URL", "")
        auth_token = os.getenv("AUTH_TOKEN", "")
        return cls(utilities_url, camera_scheduler_url, auth_token, timeout)

    def validate_or_raise(self) -> None:
        missing = [
            k
            for k, v in {
                "UTILITIES_URL": self.utilities_url,
                "CAMERA_SCHEDULER_URL": self.camera_scheduler_url,
            }.items()
            if not v
        ]
        if missing:
            raise RuntimeError("Missing required configuration in .env: " + ", ".join(missing))


@dataclass
class CameraJob:
    rtsp_link: str
    camera_nickname: str
    telegram_api_token: str
    telegram_chat_id: Union[str, int]
    threshold: float = 0.5
    cooldown_seconds: int = 3600
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        base = {
            "rtsp_link": self.rtsp_link,
            "camera_nickname": self.camera_nickname,
            "telegram_api_token": self.telegram_api_token,
            "telegram_chat_id": str(self.telegram_chat_id),
            "threshold": float(self.threshold),
            "cooldown_seconds": int(self.cooldown_seconds),
        }
        base.update(self.extra or {})
        return base

    def validate_or_raise(self) -> None:
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
        if int(self.cooldown_seconds) <= 0:
            raise ValueError("cooldown_seconds must be a positive integer")


class FaceUtilitiesClient:
    """HTTP client that talks to Utilities and Camera Scheduler services."""

    def __init__(self, config: AppConfig):
        self.config = config
        # UI code in this file calls `self.client.*`; keep it working with minimal changes.
        self.client = self

    def _headers(self) -> Dict[str, str]:
        token = (self.config.auth_token or "").strip()
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _utilities_base(self) -> str:
        """
        Some legacy configs set UTILITIES_URL ending with /person.
        This helper normalizes it to the API root (no trailing /person).
        """
        base = self.config.utilities_url.rstrip("/")
        if base.endswith("/person"):
            base = base[: -len("/person")]
        return base

    def _camera_base(self) -> str:
        """
        Normalizes CAMERA_SCHEDULER_URL to the API root.

        It supports:
          - .../cameras/schedule
          - .../cameras
          - ... (already root)
        """
        base = (self.config.camera_scheduler_url or "").rstrip("/")
        if base.endswith("/cameras/schedule"):
            base = base[: -len("/cameras/schedule")]
        elif base.endswith("/cameras"):
            base = base[: -len("/cameras")]
        return base.rstrip("/")

    def _camera_schedule_url(self) -> str:
        return self._camera_base() + "/cameras/schedule"

    def _camera_list_url(self) -> str:
        return self._camera_base() + "/cameras"

    def _camera_status_url(self, camera_id: str) -> str:
        return self._camera_base() + f"/cameras/{camera_id}/status"

    def _camera_delete_url(self, camera_id: str) -> str:
        return self._camera_base() + f"/cameras/{camera_id}"

    def add_face(self, person_id: str, image_paths: Union[str, List[str]]) -> dict:
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
                headers=self._headers(),
                files=files,
                data=data,
                timeout=self.config.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        finally:
            for fh in file_handlers:
                try:
                    fh.close()
                except Exception:
                    pass

    def list_persons(self) -> dict:
        """
        Lists all persons and how many images/embeddings each one has.
        Works whether UTILITIES_URL is .../person or API root.
        """
        url = self._utilities_base() + "/persons"
        resp = requests.get(url, headers=self._headers(), timeout=self.config.timeout)
        resp.raise_for_status()
        return resp.json()

    def retrieve_face(self, image_path: str) -> dict:
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
            data = {"action": "retrieve"}
            resp = requests.post(
                self.config.utilities_url,
                headers=self._headers(),
                files=files,
                data=data,
                timeout=self.config.timeout,
            )
        resp.raise_for_status()
        return resp.json()

    def remove_face(self, person_id: str) -> dict:
        data = {"action": "remove", "person_id": person_id}
        resp = requests.post(
            self.config.utilities_url,
            headers=self._headers(),
            data=data,
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def add_face_from_memory(self, person_id: str, images: List[Tuple[str, bytes, str]]) -> dict:
        if not images:
            raise ValueError("No images provided")

        files = [("image", (name, data, mime or "application/octet-stream")) for (name, data, mime) in images]
        data = {"action": "add", "person_id": person_id}
        resp = requests.post(
            self.config.utilities_url,
            headers=self._headers(),
            files=files,
            data=data,
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def retrieve_face_from_memory(self, image: Tuple[str, bytes, str]) -> dict:
        name, data, mime = image
        files = {"image": (name, data, mime or "application/octet-stream")}
        payload = {"action": "retrieve"}
        resp = requests.post(
            self.config.utilities_url,
            headers=self._headers(),
            files=files,
            data=payload,
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def camera_status(self, camera_id: str) -> dict:
        data = {"action": "camera_status", "camera_id": camera_id}
        resp = requests.post(
            self.config.utilities_url,
            headers=self._headers(),
            data=data,
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def list_cameras(self) -> dict:
        resp = requests.get(self._camera_list_url(), headers=self._headers(), timeout=self.config.timeout)
        resp.raise_for_status()
        return resp.json()

    def camera_job_status(self, camera_id: str) -> dict:
        resp = requests.get(self._camera_status_url(camera_id), headers=self._headers(), timeout=self.config.timeout)
        resp.raise_for_status()
        return resp.json()

    def delete_camera(self, camera_id: str) -> dict:
        resp = requests.delete(self._camera_delete_url(camera_id), headers=self._headers(), timeout=self.config.timeout)
        resp.raise_for_status()
        return resp.json()

    def start_camera_single(self, job: CameraJob) -> dict:
        job.validate_or_raise()
        payload = job.to_payload()
        resp = requests.post(
            self._camera_schedule_url(),
            json=payload,
            headers=self._headers(),
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def start_camera_batch(self, jobs: List[CameraJob]) -> dict:
        payload_list = []
        for job in jobs:
            job.validate_or_raise()
            payload_list.append(job.to_payload())

        resp = requests.post(
            self._camera_schedule_url(),
            json=payload_list,
            headers=self._headers(),
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # -----------------------------
    # Telegram test routes (FastAPI)
    # -----------------------------
    def _telegram_test_bot_url(self) -> str:
        # Uses the same API root as /persons (utilities_base)
        return self._utilities_base() + "/telegram/test-bot"

    def _telegram_test_chat_url(self) -> str:
        return self._utilities_base() + "/telegram/test-chat"

    def telegram_test_bot_route(self, telegram_api_token: str, timeout_seconds: int = 10) -> dict:
        payload = {
            "telegram_api_token": (telegram_api_token or "").strip(),
            "timeout_seconds": int(timeout_seconds),
        }
        resp = requests.post(
            self._telegram_test_bot_url(),
            json=payload,
            headers=self._headers(),
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def telegram_test_chat_route(
        self,
        telegram_api_token: str,
        telegram_chat_id: Union[str, int],
        mode: str = "getChat",
        text: str = "ping",
        timeout_seconds: int = 10,
    ) -> dict:
        payload = {
            "telegram_api_token": (telegram_api_token or "").strip(),
            "telegram_chat_id": str(telegram_chat_id),
            "mode": mode,
            "text": text,
            "timeout_seconds": int(timeout_seconds),
        }
        resp = requests.post(
            self._telegram_test_chat_url(),
            json=payload,
            headers=self._headers(),
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _auth_sidebar(self) -> None:
        st.sidebar.markdown("### 🔐 Autenticação")

        if "auth_token" not in st.session_state:
            st.session_state["auth_token"] = (self.config.auth_token or "").strip()

        autenticado = bool(st.session_state["auth_token"])

        if autenticado:
            st.sidebar.success("Autenticado")

            cols = st.sidebar.columns(2)
            with cols[0]:
                if st.button("Trocar token", use_container_width=True):
                    st.session_state["show_token_input"] = True
                    st.rerun()

            with cols[1]:
                if st.button("Limpar", use_container_width=True):
                    st.session_state["auth_token"] = ""
                    self.config.auth_token = ""
                    st.session_state["show_token_input"] = True
                    st.rerun()

        if not autenticado or st.session_state.get("show_token_input", False):
            token = st.sidebar.text_input(
                "AUTH_TOKEN (Bearer)",
                type="password",
                help="Este token será enviado em todas as requisições como: Authorization: Bearer <token>.",
            )

            cols = st.sidebar.columns(2)
            with cols[0]:
                if st.button("Salvar", use_container_width=True):
                    st.session_state["auth_token"] = (token or "").strip()
                    self.config.auth_token = st.session_state["auth_token"]
                    st.session_state["show_token_input"] = False
                    st.rerun()

            with cols[1]:
                if st.button("Usar token do .env", use_container_width=True):
                    st.session_state["auth_token"] = (AppConfig.from_env().auth_token or "").strip()
                    self.config.auth_token = st.session_state["auth_token"]
                    st.session_state["show_token_input"] = False
                    st.rerun()

        self.config.auth_token = st.session_state["auth_token"]

        if not self.config.auth_token:
            st.sidebar.warning("Sem AUTH_TOKEN. As rotas protegidas vão retornar 401/403.")

    def _ensure_auth_or_warn(self) -> bool:
        if not (self.config.auth_token or "").strip():
            st.error("AUTH_TOKEN não definido. Configure na barra lateral (🔐 Autenticação).")
            return False
        return True

    def run(self) -> None:
        st.set_page_config(page_title="👤 Detecção Facial", layout="wide")
        st.title("Utilitários")

        try:
            self.config.validate_or_raise()
        except Exception as e:
            st.error(str(e))
            st.stop()

        self._auth_sidebar()

        choice = st.sidebar.radio(
            "Escolha uma ação:",
            [
                "Adicionar Rosto",
                "Consultar Rosto",
                "Remover Rosto",
                "Listar Pessoas",
                "Câmeras - Listar",
                "Câmeras - Status (Job)",
                "Câmeras - Remover (Job)",
                "Câmera - Iniciar",
                "Telegram - Chat IDs",
                "Telegram - Test Bot (API)",
                "Telegram - Test Chat (API)",
            ],
        )

        if choice == "Adicionar Rosto":
            self.page_add_face()
        elif choice == "Consultar Rosto":
            self.page_retrieve_face()
        elif choice == "Remover Rosto":
            self.page_remove_face()
        elif choice == "Listar Pessoas":
            self.page_list_persons()
        elif choice == "Câmeras - Listar":
            self.page_cameras_list()
        elif choice == "Câmeras - Status (Job)":
            self.page_cameras_job_status()
        elif choice == "Câmeras - Remover (Job)":
            self.page_cameras_delete()
        elif choice == "Câmera - Iniciar":
            self.page_camera_start()
        elif choice == "Telegram - Chat IDs":
            self.page_telegram_list_chat_ids()
        elif choice == "Telegram - Test Bot (API)":
            self.page_telegram_test_bot_api()
        elif choice == "Telegram - Test Chat (API)":
            self.page_telegram_test_chat_api()

    def page_add_face(self) -> None:
        st.header("Adicionar Rosto (1 ou várias imagens)")
        person_id = st.text_input("Person ID (opcional)", "web_person")
        uploads = st.file_uploader("Envie uma ou mais imagens", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploads and st.button("Adicionar"):
            if not self._ensure_auth_or_warn():
                return
            try:
                images_mem: List[Tuple[str, bytes, str]] = []
                for uf in uploads:
                    data = uf.getvalue() or uf.read()
                    mime = uf.type or "application/octet-stream"
                    name = uf.name or "upload.bin"
                    images_mem.append((name, data, mime))

                result = self.client.add_face_from_memory(person_id or "web_person", images_mem)
                st.json(result)
            except requests.HTTPError as e:
                st.error(f"Erro HTTP: {e.response.text if e.response is not None else str(e)}")
            except Exception as e:
                st.error(f"Erro: {e}")

    def page_list_persons(self) -> None:
        st.header("Listar Pessoas Cadastradas")
        st.markdown("Mostra todas as pessoas registradas e a quantidade de imagens/embeddings por pessoa.")

        if st.button("Atualizar lista"):
            if not self._ensure_auth_or_warn():
                return
            try:
                result = self.client.list_persons()
                st.json(result)

                persons = result.get("persons", [])
                if persons:
                    st.subheader("Tabela")
                    st.dataframe(persons, use_container_width=True)
                else:
                    st.info("Nenhuma pessoa cadastrada ainda.")
            except requests.HTTPError as e:
                st.error(f"Erro HTTP: {e.response.text if e.response is not None else str(e)}")
            except Exception as e:
                st.error(f"Erro: {e}")

    def page_retrieve_face(self) -> None:
        st.header("Consultar / Reconhecer Rosto (única imagem)")
        upload = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])

        if upload and st.button("Consultar"):
            if not self._ensure_auth_or_warn():
                return
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
            if not self._ensure_auth_or_warn():
                return
            try:
                result = self.client.remove_face(person_id)
                st.json(result)
            except requests.HTTPError as e:
                st.error(f"Erro HTTP: {e.response.text if e.response is not None else str(e)}")
            except Exception as e:
                st.error(f"Erro: {e}")

    def page_cameras_list(self) -> None:
        st.header("Listar Câmeras (Jobs Persistidos)")
        st.markdown("Mostra todas as câmeras persistidas no scheduler (sem expor telegram_api_token).")

        if st.button("Atualizar lista"):
            if not self._ensure_auth_or_warn():
                return
            try:
                result = self.client.list_cameras()
                st.json(result)

                cams = result.get("cameras", [])
                if cams:
                    st.subheader("Tabela")
                    st.dataframe(cams, use_container_width=True)
                else:
                    st.info("Nenhuma câmera cadastrada ainda.")
            except requests.HTTPError as e:
                st.error(f"Erro HTTP: {e.response.text if e.response is not None else str(e)}")
            except Exception as e:
                st.error(f"Erro: {e}")

    def page_cameras_job_status(self) -> None:
        st.header("Status da Câmera (Job)")

        if not self._ensure_auth_or_warn():
            st.stop()

        camera_id_from_list = ""
        try:
            result = self.client.list_cameras()
            cams = result.get("cameras", []) if isinstance(result, dict) else []
            options = [""] + [str(c.get("camera_id", "")) for c in cams if c.get("camera_id")]
            camera_id_from_list = st.selectbox("Selecionar Camera ID (opcional)", options=options)
        except Exception:
            camera_id_from_list = ""

        camera_id_manual = st.text_input("Camera ID (manual)", value="")
        camera_id = (camera_id_manual or "").strip() or (camera_id_from_list or "").strip()

        if camera_id and st.button("Ver Status"):
            try:
                result = self.client.camera_job_status(camera_id)
                st.json(result)
            except requests.HTTPError as e:
                st.error(f"Erro HTTP: {e.response.text if e.response is not None else str(e)}")
            except Exception as e:
                st.error(f"Erro: {e}")

    def page_cameras_delete(self) -> None:
        st.header("Remover Câmera (Job)")

        if not self._ensure_auth_or_warn():
            st.stop()

        camera_id_from_list = ""
        try:
            result = self.client.list_cameras()
            cams = result.get("cameras", []) if isinstance(result, dict) else []
            options = [""] + [str(c.get("camera_id", "")) for c in cams if c.get("camera_id")]
            camera_id_from_list = st.selectbox("Selecionar Camera ID (opcional)", options=options)
        except Exception:
            camera_id_from_list = ""

        camera_id_manual = st.text_input("Camera ID (manual)", value="")
        camera_id = (camera_id_manual or "").strip() or (camera_id_from_list or "").strip()

        confirm = st.checkbox("Confirmar remoção", value=False)

        if camera_id and confirm and st.button("Remover"):
            try:
                result = self.client.delete_camera(camera_id)
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
            cooldown_seconds = st.number_input("Cooldown (segundos)", min_value=1, max_value=7 * 24 * 3600, value=3600, step=60)

            if all([rtsp, nickname, telegram_token, telegram_chat_id]) and st.button("Iniciar Única"):
                if not self._ensure_auth_or_warn():
                    return
                try:
                    job = CameraJob(
                        rtsp_link=rtsp,
                        camera_nickname=nickname,
                        telegram_api_token=telegram_token,
                        telegram_chat_id=telegram_chat_id,
                        threshold=float(threshold),
                        cooldown_seconds=int(cooldown_seconds),
                    )
                    result = self.client.start_camera_single(job)
                    st.json(result)
                except requests.HTTPError as e:
                    st.error(f"Erro HTTP: {e.response.text if e.response is not None else str(e)}")
                except Exception as e:
                    st.error(f"Erro: {e}")
        else:
            st.markdown("Cole um array JSON de jobs (cada item deve conter os campos esperados pelo scheduler).")
            example = [
                {
                    "rtsp_link": "rtsp://user:pass@ip:port/stream1",
                    "camera_nickname": "Cam01",
                    "telegram_api_token": "<token>",
                    "telegram_chat_id": "<chat_id>",
                    "threshold": 0.55,
                    "cooldown_seconds": 3600,
                }
            ]
            json_text = st.text_area("Payload JSON (lote)", value=json.dumps(example, indent=2), height=260)

            if st.button("Iniciar Lote"):
                if not self._ensure_auth_or_warn():
                    return
                try:
                    raw = json.loads(json_text)
                    if not isinstance(raw, list):
                        st.error("O payload deve ser um array JSON (lista).")
                        return

                    jobs: List[CameraJob] = []
                    for item in raw:
                        jobs.append(
                            CameraJob(
                                rtsp_link=item.get("rtsp_link", ""),
                                camera_nickname=item.get("camera_nickname", ""),
                                telegram_api_token=item.get("telegram_api_token", ""),
                                telegram_chat_id=item.get("telegram_chat_id", ""),
                                threshold=float(item.get("threshold", 0.5)),
                                cooldown_seconds=int(item.get("cooldown_seconds", 3600)),
                                extra={
                                    k: v
                                    for k, v in item.items()
                                    if k
                                    not in {
                                        "rtsp_link",
                                        "camera_nickname",
                                        "telegram_api_token",
                                        "telegram_chat_id",
                                        "threshold",
                                        "cooldown_seconds",
                                    }
                                },
                            )
                        )

                    result = self.client.start_camera_batch(jobs)
                    st.json(result)
                except requests.HTTPError as e:
                    st.error(f"Erro HTTP: {e.response.text if e.response is not None else str(e)}")
                except Exception as e:
                    st.error(f"JSON inválido ou erro na requisição: {e}")

    def page_telegram_list_chat_ids(self) -> None:
        st.header("Telegram - Chat IDs (via getUpdates)")
        st.markdown(
            "Cole o token do bot (do @BotFather). Eu busco os chats que "
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
                        "- Envie uma mensagem nova para o bot (privado ou no grupo)\n"
                        "- Em grupos, desative o Group Privacy no @BotFather e envie algo no grupo\n"
                        "- Se usa webhook, chame deleteWebhook e tente novamente"
                    )
                    return

                lines = [f"{name} : {chat_id}" for name, chat_id in mapping.items()]
                st.subheader("Possíveis IDs de conversa")
                st.code("\n".join(lines))
            except requests.HTTPError as e:
                st.error(f"Erro HTTP: {e.response.text if e.response is not None else str(e)}")
            except Exception as e:
                st.error(f"Erro: {e}")

    def page_telegram_test_bot_api(self) -> None:
        st.header("Telegram - Test Bot (via FastAPI route)")
        st.markdown("Calls backend route `/telegram/test-bot` which calls Telegram `getMe`.")

        if not (self.config.auth_token or "").strip():
            st.warning("AUTH_TOKEN is not set. If the backend protects /telegram routes, you may get 401/403.")

        token = st.text_input("Telegram Bot Token", type="password")
        timeout_seconds = st.number_input("Telegram timeout (seconds)", min_value=1, max_value=60, value=10, step=1)

        if st.button("Test bot token", use_container_width=True):
            if not token:
                st.error("Telegram Bot Token is required.")
                return
            try:
                result = self.client.telegram_test_bot_route(token, timeout_seconds=int(timeout_seconds))
                st.json(result)
                if result.get("ok") is True:
                    st.success("Bot token is valid (ok=true).")
                else:
                    st.error("Bot token test failed (ok=false). Check the returned payload.")
            except requests.HTTPError as e:
                st.error(f"HTTP error: {e.response.text if e.response is not None else str(e)}")
            except Exception as e:
                st.error(f"Error: {e}")

    def page_telegram_test_chat_api(self) -> None:
        st.header("Telegram - Test Chat (via FastAPI route)")
        st.markdown("Calls backend route `/telegram/test-chat` (getChat or sendMessage).")

        if not (self.config.auth_token or "").strip():
            st.warning("AUTH_TOKEN is not set. If the backend protects /telegram routes, you may get 401/403.")

        token = st.text_input("Telegram Bot Token", type="password")
        chat_id = st.text_input("Telegram Chat ID (e.g. -100..., -476..., or user id)")
        mode = st.selectbox("Mode", options=["getChat", "sendMessage"], index=0)

        text = "ping"
        if mode == "sendMessage":
            text = st.text_input("Message text", value="ping")

        timeout_seconds = st.number_input("Telegram timeout (seconds)", min_value=1, max_value=60, value=10, step=1)

        if st.button("Test chat", use_container_width=True):
            if not token:
                st.error("Telegram Bot Token is required.")
                return
            if not chat_id:
                st.error("Telegram Chat ID is required.")
                return
            try:
                result = self.client.telegram_test_chat_route(
                    telegram_api_token=token,
                    telegram_chat_id=chat_id,
                    mode=mode,
                    text=text,
                    timeout_seconds=int(timeout_seconds),
                )
                st.json(result)
                if result.get("ok") is True:
                    st.success("Chat test succeeded (ok=true).")
                else:
                    st.error("Chat test failed (ok=false). Check the returned payload.")
            except requests.HTTPError as e:
                st.error(f"HTTP error: {e.response.text if e.response is not None else str(e)}")
            except Exception as e:
                st.error(f"Error: {e}")


class StreamlitApp(FaceUtilitiesClient):
    """Streamlit UI wrapper (kept as a subclass for compatibility with main())."""
    pass


def main() -> None:
    app = StreamlitApp(AppConfig.from_env())
    app.run()


if __name__ == "__main__":
    main()
