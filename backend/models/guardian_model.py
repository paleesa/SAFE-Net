import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "guardian_lr.pkl")
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "guardian_vectorizer.pkl")


class GuardianModel:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)

        # class order matters
        self.classes = self.model.classes_

    def predict_text(self, text: str):
        vec = self.vectorizer.transform([text])
        probs = self.model.predict_proba(vec)[0]

        # classes aligned with probs
        result = dict(zip(self.classes, probs))

        predicted_label = max(result, key=result.get)
        confidence = float(result[predicted_label])

        return {
            "predicted_label": predicted_label,
            "confidence": round(confidence, 4),
            "probabilities": result
        }
        
