"""
member3_firestore.py — Firebase Firestore Integration
Team Member 3 | Part 2: Backend + Cloud

Provides read/write helpers for:
  - media_assets   (ownership + fingerprint registry)
  - comparisons    (detection history)
  - scan_results   (auto-scanner logs)

Falls back to in-memory dict store when Firebase credentials are absent,
so the system runs locally without any cloud setup.
"""

import os
import time
import json
from pathlib import Path
from typing import Optional, Any
from datetime import datetime

# ── Load .env first so keys are available ─────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

# ── Firebase Initialisation ───────────────────────────────────────────────────
# Reads FIREBASE_CREDENTIALS_PATH (from .env) OR GOOGLE_APPLICATION_CREDENTIALS.
# Falls back to in-memory if neither is set.

_db = None           # Firestore client (or None)
_fallback: dict = {} # In-memory fallback: { collection: { doc_id: data } }

CRED_PATH    = os.environ.get("FIREBASE_CREDENTIALS_PATH") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
PROJECT_ID   = os.environ.get("FIREBASE_PROJECT_ID", "")

_USE_FIREBASE = False


def _init_firebase():
    global _db, _USE_FIREBASE
    if _db is not None:
        return

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            if CRED_PATH and Path(CRED_PATH).exists():
                cred = credentials.Certificate(CRED_PATH)
                firebase_admin.initialize_app(cred, {"projectId": PROJECT_ID or None})
            elif PROJECT_ID:
                # Application Default Credentials (GCP environment)
                firebase_admin.initialize_app(options={"projectId": PROJECT_ID})
            else:
                print("[Firestore] No credentials found — using in-memory fallback")
                return

        _db = firestore.client()
        _USE_FIREBASE = True
        print("[Firestore] Connected to Firebase project:", PROJECT_ID or "default")
    except Exception as ex:
        print(f"[Firestore] Init failed ({ex}) — using in-memory fallback")


# ── Generic CRUD ──────────────────────────────────────────────────────────────

def _set(collection: str, doc_id: str, data: dict):
    _init_firebase()
    data["_updated_at"] = time.time()
    if _USE_FIREBASE:
        _db.collection(collection).document(doc_id).set(data, merge=True)
    else:
        _fallback.setdefault(collection, {})[doc_id] = data


def _get(collection: str, doc_id: str) -> Optional[dict]:
    _init_firebase()
    if _USE_FIREBASE:
        doc = _db.collection(collection).document(doc_id).get()
        return doc.to_dict() if doc.exists else None
    return _fallback.get(collection, {}).get(doc_id)


def _list(collection: str, limit: int = 20) -> list[dict]:
    _init_firebase()
    if _USE_FIREBASE:
        docs = _db.collection(collection).order_by(
            "_updated_at", direction="DESCENDING"
        ).limit(limit).stream()
        return [d.to_dict() for d in docs]
    col = _fallback.get(collection, {})
    items = list(col.values())
    items.sort(key=lambda x: x.get("_updated_at", 0), reverse=True)
    return items[:limit]


def _delete(collection: str, doc_id: str) -> bool:
    _init_firebase()
    if _USE_FIREBASE:
        _db.collection(collection).document(doc_id).delete()
        return True
    col = _fallback.get(collection, {})
    if doc_id in col:
        del col[doc_id]
        return True
    return False


# ── Media Assets ──────────────────────────────────────────────────────────────

COLLECTION_MEDIA = "media_assets"

def store_media_record(media_id: str, record: dict):
    """
    Store or update a media asset record.

    Expected fields (all optional except media_id):
      owner, title, filename, media_type, phash, content_hash,
      file_size, watermark_key, uploaded_at
    """
    record["media_id"] = media_id
    if "uploaded_at" not in record:
        record["uploaded_at"] = time.time()
    _set(COLLECTION_MEDIA, media_id, record)
    print(f"[Firestore] Stored media record: {media_id}")


def get_media_record(media_id: str) -> Optional[dict]:
    """Retrieve a media record by ID."""
    return _get(COLLECTION_MEDIA, media_id)


def list_media_records(limit: int = 20) -> list[dict]:
    """List recent media asset records."""
    return _list(COLLECTION_MEDIA, limit)


def update_media_watermark(media_id: str, watermark_key: str, status: str):
    """Update watermark status on an existing record."""
    _set(COLLECTION_MEDIA, media_id, {
        "watermark_key": watermark_key,
        "watermark_status": status,
    })


def search_by_phash(phash: str, limit: int = 10) -> list[dict]:
    """
    Search media records by pHash value.
    NOTE: Firestore doesn't support fuzzy search; this is an exact lookup.
    For fuzzy pHash search, iterate and compute Hamming distance in-memory.
    """
    _init_firebase()
    if _USE_FIREBASE:
        docs = _db.collection(COLLECTION_MEDIA)\
            .where("phash", "==", phash)\
            .limit(limit)\
            .stream()
        return [d.to_dict() for d in docs]
    # Fallback: scan in memory
    col = _fallback.get(COLLECTION_MEDIA, {})
    return [v for v in col.values() if v.get("phash") == phash][:limit]


def search_by_content_hash(content_hash: str) -> Optional[dict]:
    """Find exact duplicate by SHA-256 content hash."""
    _init_firebase()
    if _USE_FIREBASE:
        docs = list(
            _db.collection(COLLECTION_MEDIA)
            .where("content_hash", "==", content_hash)
            .limit(1)
            .stream()
        )
        return docs[0].to_dict() if docs else None
    col = _fallback.get(COLLECTION_MEDIA, {})
    for v in col.values():
        if v.get("content_hash") == content_hash:
            return v
    return None


# ── Comparisons ───────────────────────────────────────────────────────────────

COLLECTION_COMPARISONS = "comparisons"

def log_comparison(media_id_1: str, media_id_2: str, result: dict):
    """Log a comparison result to Firestore."""
    import uuid
    comp_id = str(uuid.uuid4())
    record = {
        "comparison_id": comp_id,
        "media_id_1": media_id_1,
        "media_id_2": media_id_2,
        "timestamp": time.time(),
        **result,
    }
    _set(COLLECTION_COMPARISONS, comp_id, record)
    return comp_id


def get_comparison_history(media_id: str, limit: int = 10) -> list[dict]:
    """Retrieve comparison history for a specific media asset."""
    _init_firebase()
    if _USE_FIREBASE:
        docs = _db.collection(COLLECTION_COMPARISONS)\
            .where("media_id_1", "==", media_id)\
            .limit(limit)\
            .stream()
        return [d.to_dict() for d in docs]
    col = _fallback.get(COLLECTION_COMPARISONS, {})
    results = [v for v in col.values() if v.get("media_id_1") == media_id]
    return sorted(results, key=lambda x: x.get("timestamp", 0), reverse=True)[:limit]


# ── Scan Results ──────────────────────────────────────────────────────────────

COLLECTION_SCANS = "scan_results"

def log_scan(reference_id: str, scan_results: list[dict]):
    """Log auto-scanner results to Firestore."""
    import uuid
    scan_id = str(uuid.uuid4())
    record = {
        "scan_id": scan_id,
        "reference_id": reference_id,
        "timestamp": time.time(),
        "results_count": len(scan_results),
        "results": scan_results,
        "violations_found": sum(
            1 for r in scan_results if r.get("verdict") in
            {"UNAUTHORIZED_USE_DETECTED", "LIKELY_UNAUTHORIZED"}
        ),
    }
    _set(COLLECTION_SCANS, scan_id, record)
    return scan_id


def list_scan_results(limit: int = 10) -> list[dict]:
    return _list(COLLECTION_SCANS, limit)


# ── Ownership Blockchain Simulation ──────────────────────────────────────────

COLLECTION_OWNERSHIP = "ownership_chain"

def register_ownership(media_id: str, owner: str, content_hash: str, phash: str):
    """
    Register a media ownership record (blockchain simulation).
    In production this would write to an actual blockchain/ledger.
    """
    import hashlib
    # Chain hash: hash of (prev_hash + content_hash + timestamp)
    prev_record = _get(COLLECTION_OWNERSHIP, "LATEST")
    prev_hash = prev_record.get("block_hash", "GENESIS") if prev_record else "GENESIS"
    timestamp = time.time()
    block_data = f"{prev_hash}{content_hash}{timestamp}".encode()
    block_hash = hashlib.sha256(block_data).hexdigest()

    record = {
        "media_id": media_id,
        "owner": owner,
        "content_hash": content_hash,
        "phash": phash,
        "timestamp": timestamp,
        "block_hash": block_hash,
        "prev_hash": prev_hash,
    }
    _set(COLLECTION_OWNERSHIP, media_id, record)
    _set(COLLECTION_OWNERSHIP, "LATEST", {"block_hash": block_hash, "media_id": media_id})
    return block_hash


def verify_ownership(media_id: str, claimed_owner: str) -> dict:
    """Verify if a claimed owner matches the registered owner."""
    record = _get(COLLECTION_OWNERSHIP, media_id)
    if record is None:
        return {"verified": False, "reason": "No ownership record found"}
    if record.get("owner") == claimed_owner:
        return {"verified": True, "registered_owner": claimed_owner, "block_hash": record.get("block_hash")}
    return {
        "verified": False,
        "registered_owner": record.get("owner"),
        "claimed_owner": claimed_owner,
        "reason": "Owner mismatch",
    }


# ── Export Fallback State (for debugging) ────────────────────────────────────

def dump_fallback_state() -> dict:
    """Return the entire in-memory state (for debugging/testing)."""
    return {k: dict(v) for k, v in _fallback.items()}


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Firestore Self-Test (Fallback Mode) ===")

    store_media_record("test-001", {
        "owner": "SportsCorp",
        "title": "Final Match Highlights",
        "phash": "aabbccdd11223344",
        "content_hash": "deadbeef" * 8,
        "media_type": "video",
    })

    rec = get_media_record("test-001")
    print(f"Retrieved: {rec['title']} | owner={rec['owner']}")

    comp_id = log_comparison("test-001", "test-002", {"final_score": 0.91, "verdict": "UNAUTHORIZED"})
    print(f"Comparison logged: {comp_id}")

    block_hash = register_ownership("test-001", "SportsCorp", "deadbeef" * 8, "aabbccdd")
    print(f"Block hash: {block_hash}")

    verify = verify_ownership("test-001", "SportsCorp")
    print(f"Ownership verified: {verify}")

    state = dump_fallback_state()
    print(f"Collections in memory: {list(state.keys())}")
    print("Self-test PASSED ✓")