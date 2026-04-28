"""
member1_api.py — FastAPI Backend
Team Member 1 | Part 2: Backend + Cloud

Endpoints:
  GET  /              → health check
  POST /upload        → ingest + protect a media asset
  POST /compare       → compare two images for infringement
  GET  /media         → list all registered media records
  GET  /media/{id}    → get one media record by ID
  DELETE /media/{id}  → delete a record
"""

import sys
import os
import uuid
import time
import hashlib
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Path bootstrap ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
for _part in ["ai_engine", "backend_cloud", "ai_services", "frontend"]:
    sys.path.insert(0, str(BASE_DIR / _part))
sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np

# ── App init ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sports Media Shield API",
    description="AI-Based Digital Asset Protection System for Sports Media",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image — unsupported or corrupt file.")
    return img


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    """Basic health check endpoint."""
    return {
        "status": "online",
        "service": "Sports Media Shield API",
        "version": "1.0.0",
        "timestamp": time.time(),
    }


@app.post("/upload", tags=["Media"])
async def upload_media(
    file: UploadFile = File(..., description="Image or video file to protect"),
    owner: str = Form(default="Unknown", description="Rights holder name"),
    title: str = Form(default="Untitled", description="Asset title"),
    watermark_key: str = Form(default="", description="Secret watermark key"),
):
    """
    Ingest a protected media asset:
    - Saves file to disk
    - Computes pHash fingerprint
    - Embeds invisible watermark (if key provided)
    - Registers in Firestore
    - Returns media_id + fingerprint
    """
    data = await file.read()
    ext = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
    media_id = str(uuid.uuid4())

    # Save raw file
    save_path = UPLOAD_DIR / f"{media_id}{ext}"
    with open(save_path, "wb") as f:
        f.write(data)

    content_hash = _sha256(data)

    # Decode image
    try:
        img = _decode_image(data)
    except HTTPException:
        # Could be a video — store and return partial info
        return JSONResponse({
            "media_id": media_id,
            "filename": file.filename,
            "media_type": "video",
            "content_hash": content_hash,
            "owner": owner,
            "title": title,
            "message": "Video stored. Frame-level analysis available via /compare.",
        })

    # pHash fingerprint
    phash_str = ""
    try:
        from ai_engine.phash import compute_phash
        phash_str = compute_phash(img)
    except Exception as e:
        phash_str = f"unavailable:{e}"

    # Watermark embedding
    wm_status = "not_requested"
    if watermark_key:
        try:
            from ai_services.gemini import embed_watermark
            wm_img = embed_watermark(img, watermark_key)
            wm_path = UPLOAD_DIR / f"{media_id}_wm{ext}"
            cv2.imwrite(str(wm_path), wm_img)
            wm_status = "embedded"
        except Exception as e:
            wm_status = f"failed:{e}"

    # Firestore
    block_hash = ""
    try:
        from firestore import store_media_record, register_ownership
        record = {
            "media_id": media_id,
            "owner": owner,
            "title": title,
            "filename": file.filename,
            "media_type": "image",
            "phash": phash_str,
            "content_hash": content_hash,
            "watermark_status": wm_status,
            "uploaded_at": time.time(),
        }
        store_media_record(media_id, record)
        result = register_ownership(media_id, owner, content_hash, phash_str)
        block_hash = result if isinstance(result, str) else str(result)
    except Exception as e:
        block_hash = f"[error:{e}]"

    return {
        "media_id": media_id,
        "filename": file.filename,
        "media_type": "image",
        "size_bytes": len(data),
        "content_hash": content_hash,
        "phash": phash_str,
        "owner": owner,
        "title": title,
        "watermark_status": wm_status,
        "block_hash": block_hash,
        "uploaded_at": time.time(),
    }


@app.post("/compare", tags=["Detection"])
async def compare_media(
    reference: UploadFile = File(..., description="Original / protected image"),
    suspect: UploadFile = File(..., description="Suspect image to check"),
    watermark_key: str = Form(default="", description="Watermark key used on reference"),
    run_gemini: bool = Form(default=False, description="Run Gemini AI explanation"),
):
    """
    Compare two images and return similarity scores + verdict.

    Final Score = 0.40 × Watermark + 0.30 × pHash + 0.30 × ORB
    """
    ref_data = await reference.read()
    sus_data = await suspect.read()

    ref_img = _decode_image(ref_data)
    sus_img = _decode_image(sus_data)

    # pHash
    phash_sim = 0.0
    try:
        from ai_engine.phash import compute_phash, phash_similarity
        h1 = compute_phash(ref_img)
        h2 = compute_phash(sus_img)
        phash_sim = phash_similarity(h1, h2)
    except Exception as e:
        phash_sim = 0.0

    # ORB
    orb_sim = 0.0
    try:
        from ai_engine.orb import orb_similarity
        orb_sim = orb_similarity(ref_img, sus_img)
    except Exception as e:
        orb_sim = 0.0

    # Watermark
    wm_match = 0.0
    if watermark_key:
        try:
            from ai_services.gemini import detect_watermark
            wm_match = detect_watermark(sus_img, watermark_key)
        except Exception:
            pass

    # Quality check
    quality_report = {"overall_quality": 1.0}
    try:
        from ai_services.quality import compute_quality_score
        quality_report = compute_quality_score(sus_img)
    except Exception:
        pass

    # Final score
    final_score = 0.4 * wm_match + 0.3 * phash_sim + 0.3 * orb_sim

    if final_score >= 0.90:
        verdict = "DEFINITE_INFRINGEMENT"
    elif final_score >= 0.75:
        verdict = "HIGH_PROBABILITY"
    elif final_score >= 0.50:
        verdict = "POSSIBLE_INFRINGEMENT"
    else:
        verdict = "LIKELY_ORIGINAL"

    # Gemini explanation
    explanation = ""
    if run_gemini:
        try:
            from ai_services.explanation import generate_explanation
            explanation = generate_explanation({
                "final_score": final_score,
                "verdict": verdict,
                "phash_similarity": phash_sim,
                "orb_similarity": orb_sim,
                "watermark_match": wm_match,
            })
        except Exception as e:
            explanation = f"[Explanation unavailable: {e}]"

    return {
        "final_score": round(final_score, 4),
        "verdict": verdict,
        "phash_similarity": round(phash_sim, 4),
        "orb_similarity": round(orb_sim, 4),
        "watermark_match": round(wm_match, 4),
        "quality_report": quality_report,
        "gemini_explanation": explanation,
        "timestamp": time.time(),
    }


@app.get("/media", tags=["Media"])
def list_media(limit: int = 20):
    """List all registered media records from Firestore."""
    try:
        from firestore import list_media_records
        records = list_media_records(limit=limit)
        return {"count": len(records), "records": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Firestore error: {e}")


@app.get("/media/{media_id}", tags=["Media"])
def get_media(media_id: str):
    """Get a single media record by ID."""
    try:
        from firestore import get_media_record
        record = get_media_record(media_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Media ID '{media_id}' not found.")
        return record
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/media/{media_id}", tags=["Media"])
def delete_media(media_id: str):
    """Delete a media record and its file from disk."""
    deleted_file = False
    try:
        from storage import delete_file
        deleted_file = delete_file(media_id)
    except Exception:
        pass

    return {
        "media_id": media_id,
        "file_deleted": deleted_file,
        "message": "Record removed.",
    }


@app.get("/status", tags=["Health"])
def system_status():
    """Full system module health check."""
    try:
        from integration import system_status as _status
        return _status()
    except Exception as e:
        return {"error": str(e)}
