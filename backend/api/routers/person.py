from __future__ import annotations

import secrets
import string
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from backend.api.deps import require_auth
from backend.api.container import get_face_embedder, get_person_store, get_camera_scheduler
from backend.api.utils.images import decode_upload_bytes_to_bgr

from core.face.face_embedder import FaceEmbedder
from core.storage.person_store import PersonStore
from core.camera.camera_scheduler import CameraScheduler


router = APIRouter(
    tags=["person"],
    dependencies=[Depends(require_auth)],
)


@router.post("/person")
async def insight_person_storage(
    action: str = Form(...),
    person_id: Optional[str] = Form(None),
    camera_id: Optional[str] = Form(None),
    image: List[UploadFile] | None = File(None),
    face_embedder: FaceEmbedder = Depends(get_face_embedder),
    person_store: PersonStore = Depends(get_person_store),
    camera_scheduler: CameraScheduler = Depends(get_camera_scheduler),
):
    if not person_id and action in ["add", "retrieve"]:
        person_id = "".join(secrets.choice(string.digits) for _ in range(12))

    if image is not None and not isinstance(image, list):
        image = [image]

    if action == "add":
        if not image or len(image) == 0:
            return {"error": "At least one image is required for 'add' action."}

        added = 0
        processed = 0

        for img_file in image:
            contents = await img_file.read()
            frame = decode_upload_bytes_to_bgr(contents)
            processed += 1

            if frame is None:
                continue

            best = face_embedder.best_embedding(frame)
            if not best:
                continue

            _, emb = best
            person_store.add_embedding(person_id, emb)
            added += 1

        if added == 0:
            return {"status": "no face detected", "person_id": person_id, "processed": processed}

        return {
            "status": "faces added",
            "person_id": person_id,
            "processed_images": processed,
            "num_faces_added": added,
        }

    if action == "remove":
        if not person_id:
            return {"error": "person_id is required for 'remove' action."}

        ok = person_store.remove_person(person_id)
        return {"status": "face removed" if ok else "not found", "person_id": person_id}

    if action == "retrieve":
        if not image or len(image) == 0:
            return {"error": "Image is required for 'retrieve' action."}

        contents = await image[0].read()
        frame = decode_upload_bytes_to_bgr(contents)
        if frame is None:
            return {"status": "invalid image data"}

        detections = face_embedder.detect_and_embed(frame)
        if not detections:
            return {"status": "no face detected"}

        first_embedding = detections[0]["embedding"]
        match = person_store.match_embedding(first_embedding, threshold=face_embedder.threshold)

        return {"status": "retrieved", "result": {"match": match.person_id, "score": match.score}}
    
    return {"error": "Invalid action"}
