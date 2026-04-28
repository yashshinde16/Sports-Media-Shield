"""
member4_integration.py — System Integration Orchestrator
Team Member 4 | Part 2: Backend + Cloud

Provides the high-level pipeline that ties together:
  - Part 1: AI Engine (pHash, ORB, video)
  - Part 2: Storage + Firestore
  - Part 3: AI Services (Gemini, Quality, Explanation)
  - Part 4: Watermarking

Use this module as the single entry point for programmatic access
(e.g., from Streamlit or external scripts without hitting FastAPI).
"""

import sys
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
for part in ["ai_engine", "backend_cloud", "ai_services", "frontend"]:
    sys.path.insert(0, str(BASE_DIR / part))
sys.path.insert(0, str(BASE_DIR))

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# ── Ingest Pipeline ───────────────────────────────────────────────────────────

def ingest_media(
    img_or_path,
    owner: str = "Unknown",
    title: str = "Untitled",
    watermark_key: str = "",
    media_id: Optional[str] = None,
) -> dict:
    """
    Full ingest pipeline for a new protected asset:
      1. Load / save image
      2. Generate pHash fingerprint
      3. Embed watermark (if key provided)
      4. Store in Firestore
      5. Register ownership chain
      6. Return summary

    Args:
        img_or_path: numpy array OR file path string.
        owner: Rights-holder name.
        title: Asset title.
        watermark_key: Key string for invisible watermark.
        media_id: Override UUID if provided.

    Returns:
        dict with media_id, phash, content_hash, watermark_status, block_hash
    """
    if media_id is None:
        media_id = str(uuid.uuid4())

    # ── Load image ────────────────────────────────────────────────────────────
    if isinstance(img_or_path, (str, Path)):
        img = cv2.imread(str(img_or_path))
        if img is None:
            raise ValueError(f"Cannot load: {img_or_path}")
        original_filename = Path(img_or_path).name
    else:
        img = img_or_path
        original_filename = f"{media_id}.jpg"

    # ── Save to disk ──────────────────────────────────────────────────────────
    save_path = UPLOAD_DIR / f"{media_id}.jpg"
    cv2.imwrite(str(save_path), img)

    # ── pHash fingerprint ─────────────────────────────────────────────────────
    from ai_engine.phash import compute_phash, content_hash as img_content_hash
    phash = compute_phash(img)
    content_hash = img_content_hash(img)

    # ── Watermark ─────────────────────────────────────────────────────────────
    wm_status = "not_requested"
    if watermark_key:
        try:
            from ai_services.gemini import embed_watermark
            wm_img = embed_watermark(img, watermark_key)
            wm_path = UPLOAD_DIR / f"{media_id}_wm.jpg"
            cv2.imwrite(str(wm_path), wm_img)
            wm_status = "embedded"
        except Exception as ex:
            wm_status = f"failed:{ex}"

    # ── Firestore record ──────────────────────────────────────────────────────
    try:
        from backend_cloud.firestore import store_media_record, register_ownership
        record = {
            "media_id": media_id,
            "owner": owner,
            "title": title,
            "filename": original_filename,
            "media_type": "image",
            "phash": phash,
            "content_hash": content_hash,
            "watermark_key": watermark_key or None,
            "watermark_status": wm_status,
            "uploaded_at": time.time(),
        }
        store_media_record(media_id, record)
        block_hash = register_ownership(media_id, owner, content_hash, phash)
    except Exception as ex:
        block_hash = f"[error:{ex}]"
        print(f"[Integration] Firestore warning: {ex}")

    return {
        "media_id": media_id,
        "phash": phash,
        "content_hash": content_hash,
        "watermark_status": wm_status,
        "block_hash": block_hash,
        "owner": owner,
        "title": title,
    }


# ── Detection Pipeline ────────────────────────────────────────────────────────

def detect_unauthorized(
    suspect_img_or_path,
    reference_media_id: Optional[str] = None,
    reference_img: Optional[np.ndarray] = None,
    watermark_key: str = "",
    run_gemini: bool = True,
) -> dict:
    """
    Full detection pipeline for a suspect media asset.

    Args:
        suspect_img_or_path: numpy array OR file path of the suspect image.
        reference_media_id: ID of the reference (protected) asset in Firestore.
        reference_img: Optional reference image array (used if media_id not provided).
        watermark_key: Key used during watermarking the reference.
        run_gemini: Whether to invoke Gemini for natural language explanation.

    Returns:
        Full detection result dict.
    """
    # ── Load suspect ──────────────────────────────────────────────────────────
    if isinstance(suspect_img_or_path, (str, Path)):
        suspect = cv2.imread(str(suspect_img_or_path))
        if suspect is None:
            raise ValueError(f"Cannot load suspect: {suspect_img_or_path}")
    else:
        suspect = suspect_img_or_path

    # ── Load reference ────────────────────────────────────────────────────────
    ref_img = reference_img
    if ref_img is None and reference_media_id:
        ref_path_matches = list(UPLOAD_DIR.glob(f"{reference_media_id}.*"))
        # Prefer non-watermarked version
        candidates = [p for p in ref_path_matches if "_wm" not in p.stem]
        if not candidates:
            candidates = ref_path_matches
        if candidates:
            ref_img = cv2.imread(str(candidates[0]))

    if ref_img is None:
        raise ValueError("No reference image available for comparison")

    # ── pHash ─────────────────────────────────────────────────────────────────
    from ai_engine.phash import compute_phash, phash_similarity
    h_ref = compute_phash(ref_img)
    h_sus = compute_phash(suspect)
    p_sim = phash_similarity(h_ref, h_sus)

    # ── ORB ───────────────────────────────────────────────────────────────────
    from ai_engine.orb import orb_similarity
    o_sim = orb_similarity(ref_img, suspect)

    # ── Watermark detection ───────────────────────────────────────────────────
    wm_match = 0.0
    if watermark_key:
        try:
            from ai_services.gemini import detect_watermark
            wm_match = detect_watermark(suspect, watermark_key)
        except Exception as ex:
            print(f"[Integration] Watermark detection error: {ex}")

    # ── Quality check ─────────────────────────────────────────────────────────
    quality_report = {"overall_quality": 1.0}
    try:
        from ai_services.quality import compute_quality_score
        quality_report = compute_quality_score(suspect)
    except Exception as ex:
        print(f"[Integration] Quality check error: {ex}")

    # ── Final score ───────────────────────────────────────────────────────────
    from ai_engine.similarity import ComponentScores, analyse
    scores = ComponentScores(
        phash_similarity=p_sim,
        orb_similarity=o_sim,
        watermark_match=wm_match,
        quality_score=quality_report.get("overall_quality", 1.0),
    )
    result = analyse(scores)

    # ── Gemini explanation ────────────────────────────────────────────────────
    gemini_text = ""
    if run_gemini:
        try:
            from ai_services.explanation import generate_explanation
            gemini_text = generate_explanation(result)
        except Exception as ex:
            gemini_text = f"[Explanation unavailable: {ex}]"

    # ── Log to Firestore ──────────────────────────────────────────────────────
    if reference_media_id:
        try:
            from backend_cloud.firestore import log_comparison
            log_comparison(reference_media_id, "suspect", result.to_dict())
        except Exception:
            pass

    return {
        **result.to_dict(),
        "phash_reference": h_ref,
        "phash_suspect": h_sus,
        "quality_report": quality_report,
        "gemini_explanation": gemini_text,
    }


# ── Batch Scan Pipeline ───────────────────────────────────────────────────────

def batch_scan(
    reference_media_id: str,
    suspect_paths: list[str],
    watermark_key: str = "",
) -> list[dict]:
    """
    Scan multiple suspect images against one reference.
    Returns sorted list of results (highest similarity first).
    """
    results = []
    for path in suspect_paths:
        try:
            r = detect_unauthorized(
                path,
                reference_media_id=reference_media_id,
                watermark_key=watermark_key,
                run_gemini=False,
            )
            r["suspect_path"] = path
            results.append(r)
        except Exception as ex:
            results.append({"suspect_path": path, "error": str(ex), "final_score": 0.0})

    results.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    # Log batch scan
    try:
        from backend_cloud.firestore import log_scan
        log_scan(reference_media_id, results)
    except Exception:
        pass

    return results


# ── Status Check ─────────────────────────────────────────────────────────────

def system_status() -> dict:
    """Return status of all system modules."""
    status = {
        "storage": "ok",
        "firestore": "unknown",
        "phash": "unknown",
        "orb": "unknown",
        "watermark": "unknown",
        "quality": "unknown",
        "gemini": "unknown",
    }

    # Check each module
    try:
        from ai_engine.phash import compute_phash
        import numpy as np
        compute_phash(np.zeros((100,100,3), dtype=np.uint8))
        status["phash"] = "ok"
    except Exception as e:
        status["phash"] = f"error: {e}"

    try:
        from ai_engine.orb import orb_similarity
        import numpy as np
        orb_similarity(np.zeros((100,100,3), dtype=np.uint8), np.zeros((100,100,3), dtype=np.uint8))
        status["orb"] = "ok"
    except Exception as e:
        status["orb"] = f"error: {e}"

    try:
        from backend_cloud.firestore import list_media_records
        list_media_records(limit=1)
        import backend_cloud.firestore as _fs_mod
        status["firestore"] = "ok (Firebase)" if _fs_mod._USE_FIREBASE else "ok (fallback)"
    except Exception as e:
        status["firestore"] = f"error: {e}"

    try:
        from ai_services.gemini import embed_watermark
        import numpy as np
        embed_watermark(np.zeros((100,100,3), dtype=np.uint8), "test")
        status["watermark"] = "ok"
    except Exception as e:
        status["watermark"] = f"error: {e}"

    try:
        from ai_services.quality import compute_quality_score
        import numpy as np
        compute_quality_score(np.zeros((100,100,3), dtype=np.uint8))
        status["quality"] = "ok"
    except Exception as e:
        status["quality"] = f"error: {e}"

    try:
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        status["gemini"] = "configured" if gemini_key else "no_key (using mock)"
    except Exception as e:
        status["gemini"] = f"error: {e}"

    return status


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Integration Self-Test ===\n")

    status = system_status()
    for k, v in status.items():
        print(f"  {k:15s}: {v}")

    print("\nAll integration checks complete ✓")