import modal
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi import Depends, HTTPException, status, UploadFile, File, Form
import os
import numpy as np
from typing import List, Tuple


# ---------- Modal Setup ----------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]",
        "insightface==0.7.3",
        "numpy==2.1.3",
        "onnxruntime-gpu==1.23.0",
        "scikit-learn",
        "opencv-python-headless"
    )
    .add_local_dir("./models", remote_path="/root/models")
)
app = modal.App(name="person_utilities", image=image)

persons = modal.Dict.from_name("person_storage", create_if_missing=True)


# ---------- Face Embedder ----------
class FaceEmbedder:
    """
    Singleton wrapper around InsightFace FaceAnalysis to ensure the model is
    instantiated only once per container.
    """
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
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            # NOTE: ctx_id=0 assumes GPU is available; InsightFace will fall back internally if not.
            self.model.prepare(ctx_id=0)
            print("[INFO] FaceEmbedder initialized once.")
            self.threshold = 0.5

    def detect_and_embed(self, image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect faces and return a list of (face_crop, embedding) tuples.
        """
        faces = self.model.get(image)
        results: List[Tuple[np.ndarray, np.ndarray]] = []
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            face_crop = image[y1:y2, x1:x2]
            results.append((face_crop, f.embedding))
        return results

    def best_embedding(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray] | None:
        """
        Return the (face_crop, embedding) for the best (highest det_score) detected face.
        This helps avoid adding wrong identities when multiple faces appear in a photo.
        """
        faces = self.model.get(image)
        if not faces:
            return None
        # Choose the highest-confidence detection if available; fall back to the first face.
        try:
            best = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
        except ValueError:
            best = faces[0]
        x1, y1, x2, y2 = map(int, best.bbox)
        face_crop = image[y1:y2, x1:x2]
        return face_crop, best.embedding

    def query_embedding(self, image: np.ndarray):
        """
        Query the in-memory dictionary of embeddings and return (person_id, score)
        for each face detected in the image, or (None, score) if below threshold.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        results = []
        detections = self.detect_and_embed(image)
        for (_, emb) in detections:
            if not list(persons.keys()):
                results.append((None, 0.0))
                continue

            best_id, best_score = None, -1.0
            for pid, embs in persons.items():
                sims = [cosine_similarity([emb], [np.array(e)])[0, 0] for e in embs]
                score = max(sims) if sims else -1.0
                if score > best_score:
                    best_id, best_score = pid, score

            if best_score >= self.threshold:
                results.append((best_id, best_score))
            else:
                results.append((None, best_score))

        return results


# ---------- Utility Functions ----------
@app.function()
def add_face(person_id: str, embedding: list):
    """Add a single embedding to a given person_id in the dict."""
    current = persons.get(person_id, [])
    current.append(embedding)  # embedding must be JSON-serializable (list)
    persons[person_id] = current
    return True


@app.function()
def remove_face(person_id: str):
    """Remove person_id entirely from the dict."""
    if person_id in persons:
        del persons[person_id]
        return True
    return False


@app.function()
def get_face_id(embedding: list, threshold: float = 0.5):
    """Match an embedding against stored ones and return the best match (if above threshold)."""
    from sklearn.metrics.pairwise import cosine_similarity

    best_id, best_score = None, -1.0
    for pid, embs in persons.items():
        sims = [cosine_similarity([embedding], [np.array(e)])[0, 0] for e in embs]
        score = max(sims) if sims else -1.0
        if score > best_score:
            best_id, best_score = pid, score

    if best_id is not None and best_score >= threshold:
        return {"match": best_id, "score": best_score}
    return {"match": None, "score": best_score}


@app.function()
def get_camera_status(camera_id: str):
    """Check whether a Modal function call (camera job) is ON/OFF/NOT_FOUND."""
    if not camera_id:
        return {"result": "NOT_FOUND", "detail": "camera_id not provided"}

    try:
        function_call = modal.FunctionCall.from_id(camera_id)
        function_call.get(timeout=0.5)
        return {"result": "OFF"}
    except TimeoutError:
        return {"result": "ON"}
    except modal.exception.NotFoundError:
        return {"result": "NOT_FOUND"}
    except Exception:
        return {"result": "OFF"}


# ---------- FastAPI Endpoint ----------
auth_scheme = HTTPBearer()


def _decode_uploadfile_to_bgr(contents: bytes):
    """Decode raw bytes into an OpenCV BGR image."""
    import cv2
    arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


@app.function(secrets=[modal.Secret.from_name("primai-api-key")])
@modal.fastapi_endpoint(method="POST")
async def primai_person_storage(
    action: str = Form(...),
    person_id: str = Form(None),
    camera_id: str = Form(None),
    image: List[UploadFile] = File(None),  # <-- now supports multiple images
    token: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    import cv2  # local import to keep image smaller if function isn't invoked

    # ---- Auth ----
    if token.credentials != os.environ["AUTH_TOKEN"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    embedder = FaceEmbedder()

    # If no ID is provided for add/retrieve, generate a numeric ID
    if not person_id and action in ["add", "retrieve"]:
        import secrets, string
        person_id = "".join(secrets.choice(string.digits) for _ in range(12))

    # Normalize image to a list for consistent processing
    if image is not None and not isinstance(image, list):
        image = [image]

    # ---- Actions ----
    if action == "add":
        """
        Add one or more example images for the same person_id.
        For each provided image:
          - choose the best face (highest det_score) to minimize identity contamination;
          - store its embedding under the given person_id.
        """
        if not image or len(image) == 0:
            return {"error": "At least one image is required for add"}

        added = 0
        processed = 0
        for img_file in image:
            contents = await img_file.read()
            frame = _decode_uploadfile_to_bgr(contents)
            processed += 1
            if frame is None:
                continue

            best = embedder.best_embedding(frame)
            if not best:
                continue

            _, emb = best
            add_face.remote(person_id, emb.tolist())
            added += 1

        if added == 0:
            return {"status": "no face detected", "person_id": person_id, "processed": processed}

        return {
            "status": "faces added",
            "person_id": person_id,
            "processed_images": processed,
            "num_faces_added": added,
        }

    elif action == "remove":
        ok = remove_face.remote(person_id)
        return {"status": "face removed" if ok else "not found", "person_id": person_id}

    elif action == "retrieve":
        """
        Retrieve the best match for a single query image (kept as single to keep API stable).
        """
        if not image or len(image) == 0:
            return {"error": "Image is required for retrieve"}

        # Use only the first provided image for retrieval
        contents = await image[0].read()
        frame = _decode_uploadfile_to_bgr(contents)
        if frame is None:
            return {"status": "invalid image data"}

        results = embedder.detect_and_embed(frame)
        if not results:
            return {"status": "no face detected"}

        _, emb = results[0]
        res = get_face_id.remote(emb.tolist())
        return {"status": "retrieved", "result": res}

    elif action == "camera_status":
        if not camera_id:
            return {"error": "camera_id is required for status"}
        res = get_camera_status.remote(camera_id)
        return {"status": "job_status", "camera_id": camera_id, "result": res}

    else:
        return {"error": "Invalid action"}
