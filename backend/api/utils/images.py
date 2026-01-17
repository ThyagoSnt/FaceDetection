from __future__ import annotations

import numpy as np
import cv2

def decode_upload_bytes_to_bgr(contents: bytes) -> np.ndarray | None:
    arr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
