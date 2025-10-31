import modal

from pydantic import BaseModel
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi import Depends, HTTPException, status
from typing import List, Union

auth_scheme = HTTPBearer()

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "fastapi[standard]==0.117.1",
        "opencv-python-headless==4.12.0.88",
        "numpy==2.2.6",
        "onnxruntime-gpu==1.23.0",
        "pandas==2.2.3",
        "insightface==0.7.3",
        "scikit-learn==1.7.1",
        "requests==2.32.5",
        "python-telegram-bot==22.5",
        "pytz"
    )
    .add_local_dir("./models", remote_path="/root/models")
)

app = modal.App(name="modal_camera_feed_analisys", image=image)
persons = modal.Dict.from_name("person_storage", create_if_missing=True)

import os
import time
import numpy as np


# Face Embedding Utilities
class FaceEmbedder:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FaceEmbedder, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_root: str = "./models"):
        import insightface
        if not hasattr(self, "model"):
            self.model = insightface.app.FaceAnalysis(
                root=model_root,
                name="buffalo_l",
                providers=['CUDAExecutionProvider']
            )
            self.model.prepare(ctx_id=0)
            self.threshold = 0.5

    def detect_and_embed(self, image: np.ndarray):
        faces = self.model.get(image)
        results = []
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            face_crop = image[y1:y2, x1:x2]
            results.append(
                {
                    "face_crop": face_crop,
                    "embedding": f.embedding,
                    "bbox": (x1, y1, x2, y2),
                }
            )
        return results

    def capture_from_rtsp(self, rtsp_url: str) -> np.ndarray:
        import cv2
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open RTSP stream: {rtsp_url}")
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError("Failed to capture frame from RTSP")
        return frame

    def send_telegram_image(self, token, chat_id, image_path, caption):
        import requests
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        with open(image_path, "rb") as image:
            files = {"photo": image}
            data = {"chat_id": chat_id, "caption": caption}
            response = requests.post(url, data=data, files=files)
        if response.status_code != 200:
            print("Error sending image:", response.text)
        else:
            print("Image sent")

    def draw_bbox_and_save(self, frame: np.ndarray, bbox, out_path: str):
        import cv2
        x1, y1, x2, y2 = bbox
        img = frame.copy()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(out_path, img)
        return out_path

    def query_embedding(self, image: np.ndarray):
        from sklearn.metrics.pairwise import cosine_similarity
        results = []
        detections = self.detect_and_embed(image)
        for d in detections:
            emb = d["embedding"]
            if not list(persons.keys()):
                results.append((None, 0.0))
                continue
            best_id, best_score = None, -1
            for pid, embs in persons.items():
                sims = [cosine_similarity([emb], [np.array(e)])[0, 0] for e in embs]
                score = max(sims) if sims else -1
                if score > best_score:
                    best_id, best_score = pid, score
            if best_score >= self.threshold:
                results.append((best_id, best_score))
            else:
                results.append((None, best_score))
        return results


@app.function(
    image=image,
    timeout=60 * 60 * 24,
)
async def camera_feed_analysis(
    rtsp_link: str,
    threshold: float,
    telegram_api_token: str,
    telegram_chat_id: str,
    camera_nickname: str,
    cooldown_seconds: int = 3600,
):
    from datetime import datetime
    import pytz
    import asyncio
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    embedder = FaceEmbedder()
    embedder.threshold = threshold

    last_notified_ts: dict[str, float] = {}

    while True:
        frame = embedder.capture_from_rtsp(rtsp_link)
        if frame is None:
            await asyncio.sleep(0.01)
            continue

        if not persons:
            await asyncio.sleep(0)
            continue

        detections = embedder.detect_and_embed(frame) or []

        for d in detections:
            emb = d["embedding"]
            bbox = d["bbox"]

            # matching por similaridade
            best_id, best_score = None, -1.0
            for pid, embs in persons.items():
                if not embs:
                    continue
                sims = [float(cosine_similarity([emb], [np.asarray(e)])[0, 0]) for e in embs]
                score = max(sims) if sims else -1.0
                if score > best_score:
                    best_id, best_score = pid, score

            # verificação de limiar + cooldown
            if best_id is not None and best_score >= threshold:
                now_ts = time.time()
                last_ts = last_notified_ts.get(best_id)

                # se já notificado recentemente, pula
                if last_ts is not None and (now_ts - last_ts) < cooldown_seconds:
                    # (opcional) log de debug:
                    # print(f"[{camera_nickname}] ID {best_id} em cooldown, faltam {int(cooldown_seconds - (now_ts - last_ts))}s")
                    continue

                # registra o envio antes para evitar flood em detecções muito próximas
                last_notified_ts[best_id] = now_ts

                brasilia_tz = pytz.timezone("America/Sao_Paulo")
                ts_brasilia = datetime.now(brasilia_tz)
                dia_semana = ts_brasilia.strftime("%A").capitalize()
                data_formatada = ts_brasilia.strftime("%d/%m/%Y")
                hora_formatada = ts_brasilia.strftime("%H:%M:%S")

                out_path = f"/tmp/match_{best_id}_{int(now_ts)}.jpg"
                embedder.draw_bbox_and_save(frame, bbox, out_path)

                caption = (
                    "📸 Indivíduo detectado!\n"
                    f"⌛ Dia da Semana: {dia_semana}\n"
                    f"🗓️ Data: {data_formatada}\n"
                    f"⏰ Horário: {hora_formatada}\n"
                    f"🆔 Identificador: `{best_id}`\n"
                    f"📷 Câmera: {camera_nickname}\n"
                )

                if telegram_api_token and telegram_chat_id:
                    embedder.send_telegram_image(
                        telegram_api_token,
                        telegram_chat_id,
                        out_path,
                        caption
                    )



# Request/Response IO
class CameraRequisition(BaseModel):
    rtsp_link: str
    camera_nickname: str
    telegram_api_token: str
    telegram_chat_id: str
    threshold: float | None = 0.5


def _spawn_camera_job(req: CameraRequisition) -> dict:
    """
    Spawn a camera_feed_analysis job for a single request and return metadata.
    """
    execution = camera_feed_analysis.spawn(
        req.rtsp_link,
        req.threshold if req.threshold is not None else 0.5,
        req.telegram_api_token,
        req.telegram_chat_id,
        req.camera_nickname
    )
    return {
        "camera_nickname": req.camera_nickname,
        "object_id": execution.object_id,
    }


# FastAPI: Batch or Single
@app.function(secrets=[modal.Secret.from_name("primai-api-key")])
@modal.fastapi_endpoint(method="POST")
def modal_camera_feed_analysis_scheduler(
    args: Union[CameraRequisition, List[CameraRequisition]],
    token: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    """
    Accepts either a single CameraRequisition object or a list of them.
    Spawns one Modal job per camera and returns the list of object IDs.
    """
    if token.credentials != os.environ["AUTH_TOKEN"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Normalize to list for uniform processing
    if isinstance(args, list):
        requests_list: List[CameraRequisition] = args
    else:
        requests_list = [args]

    results = []
    for req in requests_list:
        results.append(_spawn_camera_job(req))

    return {
        "count": len(results),
        "jobs": results
    }
