import numpy as np
import cv2
from insightface.app import FaceAnalysis

class FaceEmbedder:
    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding(self, bgr_img: np.ndarray) -> np.ndarray:
        faces = self.app.get(bgr_img)

        if len(faces) == 0:
            raise ValueError("No face detected")

        # Pick largest face
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        emb = face.embedding.astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        return emb