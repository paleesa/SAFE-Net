import uuid
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Body
from typing import List, Optional
from PIL import Image
import os
import numpy as np
import cv2
import json
import re

from backend.models.age_model import AgePredictor
from backend.models.face_model import FaceEmbedder
from backend.models.guardian_model import GuardianModel
from backend.db.supabase_client import supabase
from backend.services.trust_service import (
    insert_signal,
    update_scores,
    update_access_status,
    get_age_signal,
    SEC_SIGNAL_LOGIN_MATCH,
    SEC_SIGNAL_LOGIN_FAIL,
)
from datetime import datetime, date, timedelta, timezone
from backend.schemas.guardian import AnalyzeTextRequest
from backend.schemas.post import CreatePostRequest
from backend.schemas.comment import CreateCommentRequest
from backend.schemas.bio import UpdateBioRequest
from backend.schemas.message import SendMessageRequest
from backend.schemas.ekyc import SubmitEKYCRequest

app = FastAPI(title="FYP Trust Platform API")

# Path to your saved .pt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../backend
PROJECT_ROOT = os.path.dirname(BASE_DIR)                       # .../fyp-backend
CKPT_PATH = os.path.join(PROJECT_ROOT, "age_resnet50_utkface.pt")

predictor = AgePredictor(CKPT_PATH)
embedder = FaceEmbedder()
guardian = GuardianModel()

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

from fastapi import Form, File, UploadFile, HTTPException
from datetime import datetime, timedelta, timezone
import numpy as np
import cv2
from PIL import Image

@app.post("/verify-login")
async def verify_login(user_id: str = Form(...), file: UploadFile = File(...)):
    try:
        # ==========================================
        # 1️⃣ FETCH DATA (No more security score needed!)
        # ==========================================
        res = (
            supabase.table("user_identity")
            .select("face_embedding, access_status, failed_login_attempts, lockout_until")
            .eq("user_id", user_id)
            .single()
            .execute()
        )

        if not res.data or res.data.get("face_embedding") is None:
            raise HTTPException(status_code=404, detail="No registered face embedding for this user.")

        # ==========================================
        # 2️⃣ CHECK LOCKOUT STATUS FIRST
        # ==========================================
        lockout_until = res.data.get("lockout_until")
        if lockout_until:
            lockout_time = datetime.fromisoformat(lockout_until.replace('Z', '+00:00'))
            now_utc = datetime.now(timezone.utc)
            
            if now_utc < lockout_time:
                remaining_seconds = int((lockout_time - now_utc).total_seconds())
                return {
                    "ok": False,
                    "login_allowed": False,
                    "reason": f"Too many failed attempts. Please wait {remaining_seconds} seconds."
                }

        # ==========================================
        # 3️⃣ RUN THE AI FACE MATH
        # ==========================================
        stored_list = from_pgvector(res.data["face_embedding"])
        stored = np.array(stored_list, dtype=np.float32)
        access_status = res.data.get("access_status", "pending")
        failed_attempts = res.data.get("failed_login_attempts", 0)

        img = Image.open(file.file).convert("RGB")
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        live = embedder.get_embedding(bgr)

        sim = cosine_similarity(stored, live)
        FACE_MATCH_THRESHOLD = 0.35
        is_match = sim >= FACE_MATCH_THRESHOLD

        # Optional: Keep logging the attempt for your database records
        supabase.table("login_attempt").insert({
            "user_id": user_id, "is_face_match_success": is_match, "face_match_score": round(sim, 4)
        }).execute()

        # ==========================================
        # 4️⃣ SUCCESS FLOW
        # ==========================================
        if is_match:
            # Face matched! Reset the fail counter back to 0.
            supabase.table("user_identity").update({
                "failed_login_attempts": 0,
                "lockout_until": None
            }).eq("user_id", user_id).execute()

            return {
                "ok": True, "login_allowed": True, 
                "access_status": access_status,
                "similarity": round(sim, 4)
            }

        # ==========================================
        # 5️⃣ FAILURE FLOW
        # ==========================================
        else:
            failed_attempts += 1
            is_locked_out = failed_attempts >= 3
            
            update_data = {"failed_login_attempts": failed_attempts}
            
            # 3rd Strike: Set the 1-minute timeout! (Change minutes=1 to minutes=2 if you prefer)
            if is_locked_out:
                lockout_time = datetime.now(timezone.utc) + timedelta(minutes=1)
                update_data["lockout_until"] = lockout_time.isoformat()
                
            supabase.table("user_identity").update(update_data).eq("user_id", user_id).execute()

            if is_locked_out:
                return {
                    "ok": False, "login_allowed": False, 
                    "reason": "Face does not match. You are locked out for 60 seconds.", 
                    "similarity": round(sim, 4)
                }
            else:
                return {
                    "ok": False, "login_allowed": False, 
                    "reason": f"Face does not match. {3 - failed_attempts} attempts remaining.", 
                    "similarity": round(sim, 4)
                }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
    
# ########################## GUARDIAN MODEL - POST,CAPTION,BIO ##########################
def guardian_signal(predicted_label: str, confidence: float):
    if confidence >= 0.85:
        delta = 8
    elif confidence >= 0.70:
        delta = 4
    elif confidence >= 0.60:
        delta = 2
    else:
        delta = 0

    if predicted_label == "minor":
        return -delta, "NLP_MINOR"
    return +delta, "NLP_ADULT"

@app.post("/analyze-text")
async def analyze_text(req: AnalyzeTextRequest):
    try:
        # 1) run guardian
        result = guardian.predict_text(req.text)
        label = result["predicted_label"]
        conf = float(result["confidence"])

        # 2) convert to signal
        signal_val, signal_type = guardian_signal(label, conf)

        # 3) insert signal (for now source_id can be None)
        insert_signal(
            user_id=req.user_id,
            source_type=req.source_type,
            source_id=req.source_id,   # may be None
            signal_type=signal_type,
            signal_value=float(signal_val),
            ai_label=label
        )

        # 4) recalc scores
        age_score, security_score = update_scores(
            req.user_id,
            reason=f"{signal_type}:{label}:{conf:.2f}"
        )

        # 5) fetch predicted_age for access decision
        ui_res = (
            supabase.table("user_identity")
            .select("predicted_age")
            .eq("user_id", req.user_id)
            .single()
            .execute()
        )
        if not ui_res.data:
            raise HTTPException(status_code=404, detail="user_identity not found for this user")

        predicted_age = ui_res.data.get("predicted_age")
        # if predicted_age is None:
        #     raise HTTPException(status_code=400, detail="predicted_age is missing in user_identity")

        # if user not registered via face yet
        if predicted_age is None:
            return {
                "ok": True,
                "note": "No predicted_age in user_identity yet; signal recorded and scores updated.",
                "predicted_label": label,
                "confidence": conf,
                "signal_type": signal_type,
                "signal_value": signal_val,
                "trust_score": age_score,
                "security_score": security_score
            }

        # 6) update access status using your existing logic
        status = update_access_status(req.user_id, predicted_age=float(predicted_age), confidence=conf)

        return {
            "ok": True,
            "user_id": req.user_id,
            "source_type": req.source_type,
            "predicted_label": label,
            "confidence": conf,
            "signal_type": signal_type,
            "signal_value": signal_val,
            "trust_score": round(age_score, 2),
            "security_score": round(security_score, 2),
            "access_status": status
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
    
@app.post("/create-post")
async def create_post(
    user_id: str = Form(...),
    caption_text: str = Form(...),
    images: Optional[List[UploadFile]] = File(None) 
):
    try:
        media_urls = []

        # 1️⃣ Handle File Upload to Supabase Storage
        if images:
            for file in images:
                # Create a unique filename to prevent overwriting
                file_extension = file.filename.split(".")[-1]
                file_name = f"{uuid.uuid4()}.{file_extension}"
                file_path = f"posts/{user_id}/{file_name}"

                # Read file content
                file_content = await file.read()

                # Upload to Supabase bucket
                storage_res = supabase.storage.from_("post-media").upload(
                    path=file_path,
                    file=file_content,
                    file_options={"content-type": file.content_type}
                )

                # Get Public URL
                url_res = supabase.storage.from_("post-media").get_public_url(file_path)
                media_urls.append(url_res)

        # 2️⃣ Save the post first to generate the post_id
        # Note: We removed 'media_url' from this insert since it moved to the new table
        post_res = supabase.table("post").insert({
            "user_id": user_id,
            "caption_text": caption_text
        }).execute()

        if not post_res.data:
            raise HTTPException(status_code=500, detail="Failed to save post")

        post_id = post_res.data[0]["post_id"]

        # 3️⃣ NEW: Insert all URLs into the post_media table
        if media_urls:
            media_insert_data = []
            for index, url in enumerate(media_urls):
                media_insert_data.append({
                    "post_id": post_id,
                    "media_url": url,
                    "sort_order": index # Assigns 0, 1, 2, etc. to keep them in order
                })
            
            # Perform a bulk insert to save database trips
            supabase.table("post_media").insert(media_insert_data).execute()


        # 4️⃣ Run Guardian on caption text
        result = guardian.predict_text(caption_text)
        label = result["predicted_label"]
        conf = float(result["confidence"])

        # 5️⃣ Convert Guardian output into trust signal
        signal_val, signal_type = guardian_signal(label, conf)

        # 6️⃣ Insert signal into trust_score_signals
        insert_signal(
            user_id=user_id,
            source_type="post",
            source_id=post_id,
            signal_type=signal_type,
            signal_value=signal_val,
            ai_label=label
        )

        # 7️⃣ Update age/security scores
        age_score, security_score = update_scores(
            user_id,
            reason=f"{signal_type}:{label}:{conf:.2f}"
        )

        # 8️⃣ Fetch predicted_age for access status update
        ui_res = (
            supabase.table("user_identity")
            .select("predicted_age")
            .eq("user_id", user_id)
            .single()
            .execute()
        )

        if not ui_res.data:
            raise HTTPException(status_code=404, detail="user_identity not found")

        predicted_age = ui_res.data.get("predicted_age")
        if predicted_age is None:
            raise HTTPException(status_code=400, detail="predicted_age missing in user_identity")

        # 9️⃣ Update access status
        status = update_access_status(
            user_id,
            predicted_age=float(predicted_age),
            confidence=conf
        )

        return {
            "ok": True,
            "post_id": post_id,
            "predicted_label": label,
            "confidence": conf,
            "signal_type": signal_type,
            "signal_value": signal_val,
            "trust_score": round(age_score, 2),
            "security_score": round(security_score, 2),
            "access_status": status
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
@app.post("/create-comment")
async def create_comment(req: CreateCommentRequest):
    try:
        # 1️⃣ Save comment first
        comment_res = supabase.table("comment").insert({
            "user_id": req.user_id,
            "post_id": req.post_id,
            "comment_text": req.comment_text
        }).execute()

        if not comment_res.data:
            raise HTTPException(status_code=500, detail="Failed to save comment")

        comment_id = comment_res.data[0]["comment_id"]

        # 2️⃣ Run Guardian on comment text
        result = guardian.predict_text(req.comment_text)
        label = result["predicted_label"]
        conf = float(result["confidence"])

        # 3️⃣ Convert Guardian output into trust signal
        signal_val, signal_type = guardian_signal(label, conf)

        # 4️⃣ Insert trust signal
        insert_signal(
            user_id=req.user_id,
            source_type="comment",
            source_id=comment_id,
            signal_type=signal_type,
            signal_value=signal_val,
            ai_label=label
        )

        # 5️⃣ Recalculate scores
        age_score, security_score = update_scores(
            req.user_id,
            reason=f"{signal_type}:{label}:{conf:.2f}"
        )

        # 6️⃣ Fetch predicted_age for access status update
        ui_res = (
            supabase.table("user_identity")
            .select("predicted_age")
            .eq("user_id", req.user_id)
            .single()
            .execute()
        )

        if not ui_res.data:
            raise HTTPException(status_code=404, detail="user_identity not found")

        predicted_age = ui_res.data.get("predicted_age")
        if predicted_age is None:
            raise HTTPException(status_code=400, detail="predicted_age missing in user_identity")

        # 7️⃣ Update access status
        status = update_access_status(
            req.user_id,
            predicted_age=float(predicted_age),
            confidence=conf
        )

        return {
            "ok": True,
            "comment_id": comment_id,
            "predicted_label": label,
            "confidence": conf,
            "signal_type": signal_type,
            "signal_value": signal_val,
            "trust_score": round(age_score, 2),
            "security_score": round(security_score, 2),
            "access_status": status
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
    
@app.put("/update-bio")
async def update_bio(req: UpdateBioRequest):
    try:
        # 1️⃣ Check if bio already exists
        existing = (
            supabase.table("bio")
            .select("bio_id")
            .eq("user_id", req.user_id)
            .execute()
        )

        if existing.data:
            # UPDATE
            bio_id = existing.data[0]["bio_id"]

            supabase.table("bio").update({
                "bio_text": req.bio_text
            }).eq("bio_id", bio_id).execute()

        else:
            # INSERT
            bio_res = supabase.table("bio").insert({
                "user_id": req.user_id,
                "bio_text": req.bio_text
            }).execute()

            bio_id = bio_res.data[0]["bio_id"]

        # 2️⃣ Run Guardian
        result = guardian.predict_text(req.bio_text)
        label = result["predicted_label"]
        conf = float(result["confidence"])

        # 3️⃣ Convert → signal
        signal_val, signal_type = guardian_signal(label, conf)

        # 4️⃣ Insert trust signal
        insert_signal(
            user_id=req.user_id,
            source_type="bio",
            source_id=bio_id,
            signal_type=signal_type,
            signal_value=signal_val,
            ai_label=label
        )

        # 5️⃣ Update scores
        age_score, security_score = update_scores(
            req.user_id,
            reason=f"{signal_type}:{label}:{conf:.2f}"
        )

        # 6️⃣ Get predicted_age
        ui_res = (
            supabase.table("user_identity")
            .select("predicted_age")
            .eq("user_id", req.user_id)
            .single()
            .execute()
        )

        if not ui_res.data:
            raise HTTPException(status_code=404, detail="user_identity not found")

        predicted_age = ui_res.data.get("predicted_age")

        # 7️⃣ Update access status
        status = update_access_status(
            req.user_id,
            predicted_age=float(predicted_age),
            confidence=conf
        )

        return {
            "ok": True,
            "bio_id": bio_id,
            "predicted_label": label,
            "confidence": conf,
            "signal_type": signal_type,
            "signal_value": signal_val,
            "trust_score": round(age_score, 2),
            "security_score": round(security_score, 2),
            "access_status": status
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
    
# ########################## ENFORCEMENT - ACCESS STATUS ##########################################

def get_permissions_from_status(access_status: str):
    if access_status == "verified":
        return {
            "can_post": True,
            "can_comment": True,
            "can_dm": True,
            "can_join_group_chat": True,
            "can_follow_only_verified": False,
            "content_filter_enabled": False
        }

    elif access_status == "under_review":
        return {
            "can_post": True,
            "can_comment": True,
            "can_dm": True,
            "can_join_group_chat": True,
            "can_follow_only_verified": False,
            "content_filter_enabled": True
        }

    elif access_status == "restricted":
        return {
            "can_post": True,
            "can_comment": True,
            "can_dm": False,
            "can_join_group_chat": False,
            "can_follow_only_verified": True,
            "content_filter_enabled": True
        }

    else:
        return {
            "can_post": False,
            "can_comment": False,
            "can_dm": False,
            "can_join_group_chat": False,
            "can_follow_only_verified": True,
            "content_filter_enabled": True
        }


def get_user_permissions(user_id: str):
    res = (
        supabase.table("user_identity")
        .select("access_status, trust_score, security_score")
        .eq("user_id", user_id)
        .single()
        .execute()
    )

    if not res.data:
        raise HTTPException(status_code=404, detail="user_identity not found")

    access_status = res.data.get("access_status", "under_review")
    permissions = get_permissions_from_status(access_status)

    return {
        "access_status": access_status,
        "trust_score": res.data.get("trust_score"),
        "security_score": res.data.get("security_score"),
        "permissions": permissions
    }
@app.get("/user-access/{user_id}")
async def user_access(user_id: str):
    try:
        result = get_user_permissions(user_id)

        return {
            "ok": True,
            "user_id": user_id,
            "access_status": result["access_status"],
            "trust_score": result["trust_score"],
            "security_score": result["security_score"],
            "permissions": result["permissions"]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
    
##################### SEND MESSAGE (ENFORCE DM RESTRICTION) #####################
@app.post("/send-message")
async def send_message(req: SendMessageRequest):
    try:
        sender_info = get_user_permissions(req.sender_id)

        if not sender_info["permissions"]["can_dm"]:
            raise HTTPException(
                status_code=403,
                detail="Direct messaging is not allowed for this account."
            )

        # For now we only simulate success
        # Later you can save into a real message table
        return {
            "ok": True,
            "message": "Message allowed and would be sent.",
            "sender_id": req.sender_id,
            "receiver_id": req.receiver_id,
            "access_status": sender_info["access_status"]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
    
@app.post("/join-group-chat/{user_id}")
async def join_group_chat(user_id: str):
    try:
        user_info = get_user_permissions(user_id)

        if not user_info["permissions"]["can_join_group_chat"]:
            raise HTTPException(
                status_code=403,
                detail="Group chat is not allowed for this account."
            )

        return {
            "ok": True,
            "message": "User is allowed to join group chat.",
            "user_id": user_id,
            "access_status": user_info["access_status"]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
    
############## EKYC #################

def extract_dob(text: str):
    """
    Try to extract DOB from OCR text.
    Supports common formats like:
    14/08/2008
    14-08-2008
    2008-08-14
    """
    patterns = [
        r'(\d{2}[/-]\d{2}[/-]\d{4})',
        r'(\d{4}[/-]\d{2}[/-]\d{2})'
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            dob_str = match.group(1)

            for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%Y-%m-%d"):
                try:
                    return datetime.strptime(dob_str, fmt).date()
                except ValueError:
                    continue

    return None

# Update status based on EKYC result
def calculate_age(dob: date) -> int:
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

def apply_ekyc_decision(user_id: str, real_age: int):
    # 1) Get old scores first
    res = (
        supabase.table("user_identity")
        .select("trust_score, security_score")
        .eq("user_id", user_id)
        .single()
        .execute()
    )

    old_age_score = float(res.data.get("trust_score", 50.0))
    old_security_score = float(res.data.get("security_score", 50.0))

    # 2) Decide new values
    if real_age >= 16:
        status = "verified"
        new_age_score = 100.0
    else:
        status = "restricted"
        new_age_score = 0.0

    # keep security score unchanged
    new_security_score = old_security_score

    # 3) Update user_identity
    supabase.table("user_identity").update({
        "access_status": status,
        "trust_score": new_age_score,
        "security_score": new_security_score
    }).eq("user_id", user_id).execute()

    return {
        "status": status,
        "old_age_score": old_age_score,
        "new_age_score": new_age_score,
        "old_security_score": old_security_score,
        "new_security_score": new_security_score
    }

@app.post("/submit-ekyc")
async def submit_ekyc(req: SubmitEKYCRequest):
    try:
        # 1️⃣ Extract DOB
        dob = extract_dob(req.extracted_text)

        if dob is None:
            raise HTTPException(status_code=400, detail="DOB could not be extracted")

        real_age = calculate_age(dob)

        # 2️⃣ Create verification_attempt FIRST
        va_res = supabase.table("verification_attempt").insert({
            "user_id": req.user_id,
            "method": "ekyc",
            "result_status": "success",
            "result_predicted_age": float(real_age),
            "result_confidence": 1.00,
            "result_trust_score": None
        }).execute()

        attempt_id = va_res.data[0]["attempt_id"]

        # 3️⃣ Insert into ekyc_data
        ekyc_res = supabase.table("ekyc_data").insert({
            "verification_attempt_id": attempt_id,
            "id_card_front_image_url": req.front_image_url,
            "id_card_back_image_url": req.back_image_url,
            "extracted_dob": str(dob),
            "is_success": True
        }).execute()

        ekyc_id = ekyc_res.data[0]["ekyc_id"]

        # 4️⃣ Insert strong signal
        if real_age >= 16:
            signal_value = 100.0
            label = "ekyc_adult"
        else:
            signal_value = -100.0
            label = "ekyc_minor"

        insert_signal(
            user_id=req.user_id,
            source_type="ekyc",
            source_id=ekyc_id,
            signal_type="EKYC_RESULT",
            signal_value=signal_value,
            ai_label=label
        )

        # 5️⃣ Apply hard override
        decision = apply_ekyc_decision(req.user_id, real_age)

        # 6️⃣ Insert trust history
        supabase.table("trust_score_history").insert({
            "user_id": req.user_id,
            "old_score": round(decision["old_age_score"], 2),
            "new_score": round(decision["new_age_score"], 2),
            "score_type": "AGE",
            "reason": "EKYC_VERIFIED_ADULT" if real_age >= 16 else "EKYC_VERIFIED_MINOR"
        }).execute()

        return {
            "ok": True,
            "ekyc_id": ekyc_id,
            "attempt_id": attempt_id,
            "dob_extracted": str(dob),
            "real_age": real_age,
            "access_status": decision["status"],
            "trust_score": decision["new_age_score"]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

########################### USER PROFILE STATUS - ALL DATA (FOR DEBUGGING) ##########################
    
@app.get("/user-profile-status/{user_id}")
async def user_profile_status(user_id: str):
    try:
        # 1) user identity
        ui_res = (
            supabase.table("user_identity")
            .select("user_id, access_status, predicted_age, trust_score, security_score, confidence_score, last_verification")
            .eq("user_id", user_id)
            .single()
            .execute()
        )

        if not ui_res.data:
            raise HTTPException(status_code=404, detail="user_identity not found")

        identity = ui_res.data
        access_status = identity.get("access_status", "under_review")
        permissions = get_permissions_from_status(access_status)

        # 2) latest verification attempt
        va_res = (
            supabase.table("verification_attempt")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        latest_verification = va_res.data[0] if va_res.data else None

        # 3) latest eKYC (through latest verification attempt if method=ekyc)
        latest_ekyc = None
        if latest_verification and latest_verification.get("method") == "ekyc":
            attempt_id = latest_verification["attempt_id"]
            ekyc_res = (
                supabase.table("ekyc_data")
                .select("*")
                .eq("verification_attempt_id", attempt_id)
                .limit(1)
                .execute()
            )
            latest_ekyc = ekyc_res.data[0] if ekyc_res.data else None

        # 4) recent trust history
        history_res = (
            supabase.table("trust_score_history")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(10)
            .execute()
        )

        # 5) recent trust signals
        signal_res = (
            supabase.table("trust_score_signals")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(10)
            .execute()
        )

        return {
            "ok": True,
            "user_id": user_id,
            "identity": identity,
            "permissions": permissions,
            "latest_verification": latest_verification,
            "latest_ekyc": latest_ekyc,
            "recent_trust_history": history_res.data or [],
            "recent_trust_signals": signal_res.data or []
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")