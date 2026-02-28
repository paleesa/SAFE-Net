from backend.db.supabase_client import supabase

# ----------------------------
# AGE SCORE RULES
# ----------------------------
AGE_BASE = 50.0
AGE_MIN = 0.0
AGE_MAX = 100.0

# Initial verification
AGE_SIGNAL_ADULT = +5.0
AGE_SIGNAL_MINOR = -25.0

# Behavior signals (future NLP)
AGE_SIGNAL_ADULT_BEHAVIOR = +2.0
AGE_SIGNAL_MINOR_BEHAVIOR = -5.0


# ----------------------------
# SECURITY SCORE RULES
# ----------------------------
SEC_BASE = 50.0
SEC_MIN = 0.0
SEC_MAX = 100.0

SEC_SIGNAL_LOGIN_MATCH = +2.0
SEC_SIGNAL_LOGIN_FAIL = -15.0


# ----------------------------
# INSERT SIGNAL
# ----------------------------
def insert_signal(user_id, source_type, source_id, signal_type, signal_value, ai_label):
    return supabase.table("trust_score_signals").insert({
        "user_id": user_id,
        "source_type": source_type,
        "source_id": source_id,
        "signal_type": signal_type,
        "signal_value": round(float(signal_value), 2),
        "ai_label": ai_label,
    }).execute()
    
# ----------------------------
# AGE SIGNAL 
# ----------------------------
def get_age_signal(predicted_age: float):
    """
    Returns (signal_value, label) based on age regression result.
    """

    if predicted_age >= 20:
        return +35.0, "adult_high_conf"

    elif 16 <= predicted_age < 20:
        return +15.0, "adult_borderline"

    elif 13 <= predicted_age < 16:
        return -15.0, "minor_borderline"

    else:
        return -35.0, "minor_high_conf"
    
# ----------------------------
# AGE SCORE CALCULATION
# ----------------------------
def recalc_age_score(user_id: str) -> float:
    res = supabase.table("trust_score_signals") \
        .select("signal_value, source_type") \
        .eq("user_id", user_id) \
        .in_("source_type", ["verification", "bio", "post", "comment", "ekyc"]) \
        .execute()

    total = AGE_BASE
    for row in (res.data or []):
        total += float(row["signal_value"])

    return max(AGE_MIN, min(AGE_MAX, total))


# ----------------------------
# SECURITY SCORE CALCULATION
# ----------------------------
def recalc_security_score(user_id: str) -> float:
    res = supabase.table("trust_score_signals") \
        .select("signal_value") \
        .eq("user_id", user_id) \
        .eq("source_type", "login") \
        .execute()

    total = SEC_BASE
    for row in (res.data or []):
        total += float(row["signal_value"])

    return max(SEC_MIN, min(SEC_MAX, total))


# ----------------------------
# UPDATE BOTH SCORES
# ----------------------------
def update_scores(user_id: str, reason: str = None):
    # 1️⃣ Get current stored scores
    res = supabase.table("user_identity") \
        .select("trust_score, security_score") \
        .eq("user_id", user_id) \
        .single() \
        .execute()

    old_age_score = float(res.data.get("trust_score", AGE_BASE))
    old_security_score = float(res.data.get("security_score", SEC_BASE))

    # 2️⃣ Recalculate
    new_age_score = recalc_age_score(user_id)
    new_security_score = recalc_security_score(user_id)

    # 3️⃣ Update user_identity
    supabase.table("user_identity").update({
        "trust_score": round(new_age_score, 2),
        "security_score": round(new_security_score, 2)
    }).eq("user_id", user_id).execute()

    # 4️⃣ Insert AGE score history (if changed)
    if round(old_age_score, 2) != round(new_age_score, 2):
        supabase.table("trust_score_history").insert({
            "user_id": user_id,
            "old_score": round(old_age_score, 2),
            "new_score": round(new_age_score, 2),
            "score_type": "AGE",
            "reason": reason or "AGE_SCORE_UPDATE"
        }).execute()

    # 5️⃣ Insert SECURITY score history (if changed)
    if round(old_security_score, 2) != round(new_security_score, 2):
        supabase.table("trust_score_history").insert({
            "user_id": user_id,
            "old_score": round(old_security_score, 2),
            "new_score": round(new_security_score, 2),
            "score_type": "SECURITY",
            "reason": reason or "SECURITY_SCORE_UPDATE"
        }).execute()

    return new_age_score, new_security_score

# ----------------------------
# UPDATE ACCESS STATUS (AGE ONLY)
# ----------------------------
def update_access_status(user_id: str, predicted_age: float, confidence: float):
    res = supabase.table("user_identity") \
        .select("trust_score") \
        .eq("user_id", user_id) \
        .single() \
        .execute()

    age_score = float(res.data["trust_score"])

    if predicted_age >= 16 and confidence >= 0.8 and age_score >= 80:
        status = "verified"
    elif predicted_age < 16 and confidence >= 0.8 and age_score < 50:
        status = "restricted"
    else:
        status = "under_review"

    supabase.table("user_identity").update({
        "access_status": status
    }).eq("user_id", user_id).execute()

    return status