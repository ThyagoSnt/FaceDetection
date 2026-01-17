# src/face_embedder.py
from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import onnxruntime as ort
import os
import time
import numpy as np
import cv2
from insightface.app import FaceAnalysis


class FaceEmbedder:
    def __init__(
        self,
        model_root: str = "./models",
        det_size: tuple[int, int] = (640, 640),
        threshold: float = 0.5,
    ) -> None:
        self.threshold = float(threshold)

        # List available ONNX Runtime providers
        self.avaliable_hardware_providers = ort.get_available_providers()
        print(f"ONNX Runtime available providers: {self.avaliable_hardware_providers}")

        providers = []
        device = "cpu"

        if "CUDAExecutionProvider" in self.avaliable_hardware_providers:
            device = "cuda"
            providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.app = FaceAnalysis(
            name="buffalo_l",
            root=model_root,
            providers=providers,
        )
        # ctx_id: 0 for CUDA, -1 for CPU
        self.app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=det_size)

    # Core detection
    def detect_and_embed(self, frame_bgr: np.ndarray) -> List[Dict]:
        faces = self.app.get(frame_bgr)
        results: List[Dict] = []

        for face in faces:
            if getattr(face, "embedding", None) is None:
                continue

            results.append(
                {
                    "bbox": face.bbox.astype(int).tolist(),
                    "score": float(face.det_score),
                    "embedding": face.embedding.astype(np.float32),
                }
            )

        return results
    
    # Best face helper
    def best_embedding(self, frame_bgr: np.ndarray) -> Optional[Tuple[float, np.ndarray]]:
        detections = self.detect_and_embed(frame_bgr)
        if not detections:
            return None

        best = max(detections, key=lambda d: d["score"])
        return float(best["score"]), best["embedding"]
    
    # Similarity
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float32)
        b = b.astype(np.float32)

        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0.0 or b_norm == 0.0:
            return 0.0

        a = a / a_norm
        b = b / b_norm
        return float(np.dot(a, b))
    
    # RTSP capture
    def capture_from_rtsp(
        self,
        rtsp_link: str,
        timeout_sec: float = 8.0,
        warmup_reads: int = 5,
    ) -> np.ndarray:
        cap = cv2.VideoCapture(rtsp_link)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open RTSP stream: {rtsp_link}")

        start = time.time()

        # Warmup reads to reduce buffer latency.
        for _ in range(max(0, warmup_reads)):
            _, _ = cap.read()
            if time.time() - start > timeout_sec:
                break

        frame = None
        while (time.time() - start) <= timeout_sec:
            ok, frame = cap.read()
            if ok and frame is not None and getattr(frame, "size", 0) > 0:
                cap.release()
                return frame
            time.sleep(0.05)

        cap.release()
        raise RuntimeError(f"Timeout reading frame from RTSP: {rtsp_link}")
    
    # Draw + save bbox
    def draw_bbox_and_save(
        self,
        frame_bgr: np.ndarray,
        bbox: List[int],
        output_path: str,
        thickness: int = 3,
    ) -> str:
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            raise RuntimeError("Empty frame provided to draw_bbox_and_save().")

        x1, y1, x2, y2 = [int(v) for v in bbox]

        img = frame_bgr.copy()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), int(thickness))

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        ok = cv2.imwrite(output_path, img)
        if not ok:
            raise RuntimeError(f"Failed to save image to: {output_path}")

        return output_path


