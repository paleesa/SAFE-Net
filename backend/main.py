from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from PIL import Image
import os
import numpy as np
import cv2
import json

from backend.models.age_model import AgePredictor
from backend.models.face_model import FaceEmbedder
from backend.db.supabase_client import supabase
from backend.services.trust_service import (
    insert_signal,
    update_scores,
    update_access_status,
    get_age_signal,
    SEC_SIGNAL_LOGIN_MATCH,
    SEC_SIGNAL_LOGIN_FAIL,
)

app = FastAPI(title="FYP Trust Platform API")

# Path to your saved .pt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../backend
PROJECT_ROOT = os.path.dirname(BASE_DIR)                       # .../fyp-backend
CKPT_PATH = os.path.join(PROJECT_ROOT, "age_resnet50_utkface.pt")

predictor = AgePredictor(CKPT_PATH)

embedder = FaceEmbedder()
DB_PATH = "embeddings.json"

def load_db():
    try:
        with open(DB_PATH, "r") as f:
            return json.load(f)
    except:
        return {}

def save_db(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict-age")
async def predict_age(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    age = predictor.predict_age(img)
    return {"predicted_age": round(age, 1)}

@app.post("/register-face")
async def register_face(user_id: str = Form(...), file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    emb = embedder.get_embedding(bgr)

    db = load_db()
    db[user_id] = emb.tolist()
    save_db(db)

    return {"status": "registered", "embedding_dim": len(emb)}

# def cosine_similarity(a, b):
#     return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# @app.post("/verify-login")
# async def verify_login(user_id: str = Form(...), file: UploadFile = File(...)):
#     db = load_db()

#     if user_id not in db:
#         return {"ok": False, "reason": "User not registered"}

#     stored = np.array(db[user_id], dtype=np.float32)

#     img = Image.open(file.file).convert("RGB")
#     bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#     live = embedder.get_embedding(bgr)

#     sim = cosine_similarity(stored, live)

#     THRESHOLD = 0.35
#     return {
#         "ok": sim >= THRESHOLD,
#         "similarity": round(sim, 3),
#         "threshold": THRESHOLD
#     }

# ########################## FACE + AGE COMBINED ENDPOINT ##########################
def to_pgvector_str(vec_list):
    # vec_list: list[float] length 512
    return "[" + ",".join(f"{float(x):.8f}" for x in vec_list) + "]"

def from_pgvector(val):
    # Supabase returns vector as string: "[0.1,0.2,...]"
    if val is None:
        return None
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        if not s:
            return []
        return [float(x) for x in s.split(",")]
    return list(val)

@app.post("/register-identity")
async def register_identity(user_id: str = Form(...), file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB")
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 1️⃣ Face embedding
        emb = embedder.get_embedding(bgr)
        emb_list = emb.tolist()
        emb_vec = to_pgvector_str(emb_list)

        # 2️⃣ Age prediction
        predicted_age = float(predictor.predict_age(img))

        # 3️⃣ Upsert identity (NO manual access_status here)
        supabase.table("user_identity").upsert({
            "user_id": user_id,
            "predicted_age": predicted_age,
            "face_embedding": emb_vec,
            "last_verification": "now()",
        }).execute()

        # 4️⃣ Insert verification attempt log
        va_res = supabase.table("verification_attempt").insert({
            "user_id": user_id,
            "method": "register_identity",
            "result_status": "success",
            "result_predicted_age": predicted_age,
            "result_confidence": None,
            "result_trust_score": None,
        }).execute()

        attempt_id = va_res.data[0]["attempt_id"]
        
        signal_val, label = get_age_signal(predicted_age)

        insert_signal(
            user_id=user_id,
            source_type="verification",
            source_id=attempt_id,
            signal_type="AGE_PREDICT",
            signal_value=signal_val,
            ai_label=label
        )

        # 6️⃣ Recalculate scores
        age_score, security_score = update_scores(user_id,reason=f"AGE_PREDICT:{label}")

        # 7️⃣ Let trust engine decide status
        status = update_access_status(user_id, predicted_age, confidence=0.85)

        return {
            "ok": True,
            "user_id": user_id,
            "predicted_age": round(predicted_age, 1),
            "access_status": status,
            "trust_score": age_score,
            "security_score": security_score,
            "face_registered": True
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
    
# ########################## LOGIN ENDPOINT (CHECK SIMILARITY) ##########################

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

@app.post("/verify-login")
async def verify_login(user_id: str = Form(...), file: UploadFile = File(...)):
    try:
        # 1️⃣ Fetch stored embedding + access_status
        res = (
            supabase.table("user_identity")
            .select("face_embedding, access_status, security_score")
            .eq("user_id", user_id)
            .single()
            .execute()
        )

        if not res.data or res.data.get("face_embedding") is None:
            raise HTTPException(status_code=404, detail="No registered face embedding for this user.")

        stored_list = from_pgvector(res.data["face_embedding"])
        stored = np.array(stored_list, dtype=np.float32)
        access_status = res.data.get("access_status", "pending")

        # 2️⃣ Live selfie embedding
        img = Image.open(file.file).convert("RGB")
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        live = embedder.get_embedding(bgr)

        # 3️⃣ Similarity
        sim = cosine_similarity(stored, live)
        FACE_MATCH_THRESHOLD= 0.35
        is_match = sim >= FACE_MATCH_THRESHOLD

        # 4️⃣ Insert login attempt
        la_res = supabase.table("login_attempt").insert({
            "user_id": user_id,
            "is_face_match_success": is_match,
            "face_match_score": round(sim, 4),
        }).execute()

        attempt_id = la_res.data[0]["attempt_id"]

        # 5️⃣ Insert SECURITY signal
        if is_match:
            signal_val = SEC_SIGNAL_LOGIN_MATCH
        else:
            signal_val = SEC_SIGNAL_LOGIN_FAIL

        insert_signal(
            user_id=user_id,
            source_type="login",
            source_id=attempt_id,
            signal_type="FACE_MATCH",
            signal_value=signal_val,
            ai_label="match" if is_match else "mismatch"
        )

        # 6️⃣ Recalculate scores
        age_score, security_score = update_scores(user_id,reason="LOGIN_FACE_MATCH" if is_match else "LOGIN_FACE_MISMATCH")

        # 7️⃣ SECURITY enforcement (identity layer)
        if not is_match:
            return {
                "ok": False,
                "login_allowed": False,
                "reason": "Face does not match.",
                "similarity": round(sim, 4),
                "security_score": security_score
            }

        if security_score < 30:
            return {
                "ok": False,
                "login_allowed": False,
                "reason": "Security risk detected. Please reverify identity.",
                "security_score": security_score
            }

        # 8️⃣ AGE enforcement (access layer)
        if access_status == "restricted":
            return {
                "ok": True,
                "login_allowed": True,
                "access_status": "restricted",
                "similarity": round(sim, 4),
                "security_score": security_score
            }

        # 9️⃣ Normal verified / under_review
        return {
            "ok": True,
            "login_allowed": True,
            "access_status": access_status,
            "similarity": round(sim, 4),
            "security_score": security_score
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")