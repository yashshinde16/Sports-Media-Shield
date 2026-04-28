"""
member2_storage.py — Local File Storage Manager
Team Member 2 | Part 2: Backend + Cloud
Handles local /uploads directory: save, retrieve, list, delete media files.
Also provides SHA-256 content fingerprinting for ownership records.
"""

import os
import uuid
import shutil
import hashlib
import mimetypes
from pathlib import Path
from typing import Optional, BinaryIO
from datetime import datetime

import cv2
import numpy as np


# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
THUMB_DIR  = BASE_DIR / "uploads" / "_thumbnails"

UPLOAD_DIR.mkdir(exist_ok=True)
THUMB_DIR.mkdir(exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
MAX_FILE_MB = 200


# ── Disk Storage ──────────────────────────────────────────────────────────────

def save_file(
    file_obj: BinaryIO,
    original_filename: str,
    media_id: Optional[str] = None,
) -> dict:
    """
    Save a file-like object to /uploads with a unique media_id.

    Args:
        file_obj: Readable binary stream.
        original_filename: Original file name (used to preserve extension).
        media_id: Override auto-generated UUID if provided.

    Returns:
        dict with media_id, path, size_bytes, ext, media_type.
    """
    if media_id is None:
        media_id = str(uuid.uuid4())

    ext = Path(original_filename).suffix.lower()
    if not ext:
        ext = ".bin"

    dest = UPLOAD_DIR / f"{media_id}{ext}"

    with open(dest, "wb") as out:
        shutil.copyfileobj(file_obj, out)

    size = dest.stat().st_size

    if size > MAX_FILE_MB * 1024 * 1024:
        dest.unlink()
        raise ValueError(f"File too large: {size / 1e6:.1f}MB > {MAX_FILE_MB}MB limit")

    media_type = _detect_media_type(ext)

    # Generate thumbnail
    thumb_path = _generate_thumbnail(dest, media_id, media_type)

    return {
        "media_id": media_id,
        "path": str(dest),
        "filename": original_filename,
        "ext": ext,
        "size_bytes": size,
        "media_type": media_type,
        "thumbnail_path": str(thumb_path) if thumb_path else None,
        "saved_at": datetime.utcnow().isoformat(),
    }


def save_bytes(data: bytes, ext: str, media_id: Optional[str] = None) -> dict:
    """Save raw bytes directly."""
    import io
    return save_file(io.BytesIO(data), f"file{ext}", media_id)


def get_file_path(media_id: str) -> Optional[Path]:
    """Resolve the file path for a given media_id (any extension)."""
    patterns = list(UPLOAD_DIR.glob(f"{media_id}.*"))
    # Exclude thumbnails
    patterns = [p for p in patterns if "_wm" not in p.stem or p.stem == media_id]
    return patterns[0] if patterns else None


def get_thumbnail_path(media_id: str) -> Optional[Path]:
    """Get thumbnail path for media_id."""
    matches = list(THUMB_DIR.glob(f"{media_id}*"))
    return matches[0] if matches else None


def delete_file(media_id: str) -> bool:
    """Delete an uploaded file and its thumbnail."""
    path = get_file_path(media_id)
    thumb = get_thumbnail_path(media_id)
    deleted = False
    if path and path.exists():
        path.unlink()
        deleted = True
    if thumb and thumb.exists():
        thumb.unlink()
    return deleted


def list_files(limit: int = 50) -> list[dict]:
    """List uploaded media files sorted by modification time (newest first)."""
    files = []
    for p in sorted(UPLOAD_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.is_file() and not p.name.startswith("_") and p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS:
            media_id = p.stem.split("_wm")[0]  # strip _wm suffix
            files.append({
                "media_id": media_id,
                "path": str(p),
                "filename": p.name,
                "ext": p.suffix,
                "size_bytes": p.stat().st_size,
                "media_type": _detect_media_type(p.suffix.lower()),
                "modified_at": datetime.utcfromtimestamp(p.stat().st_mtime).isoformat(),
            })
        if len(files) >= limit:
            break
    return files


# ── Content Hashing ───────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of file contents."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def sha256_image(img: np.ndarray) -> str:
    """Compute SHA-256 of raw pixel bytes."""
    return hashlib.sha256(img.tobytes()).hexdigest()


def md5_file(path: Path) -> str:
    """Compute MD5 hash for quick dedup checks."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            md5.update(chunk)
    return md5.hexdigest()


# ── Duplicate Detection ───────────────────────────────────────────────────────

def find_duplicate_by_hash(content_hash: str) -> Optional[str]:
    """
    Scan uploads directory for a file with matching SHA-256 hash.
    Returns media_id if found, else None.
    (For production, this lookup should be done via Firestore index.)
    """
    for p in UPLOAD_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS:
            if sha256_file(p) == content_hash:
                return p.stem
    return None


# ── Thumbnail Generation ──────────────────────────────────────────────────────

def _generate_thumbnail(src_path: Path, media_id: str, media_type: str) -> Optional[Path]:
    """Generate a 320×180 JPEG thumbnail for display in the UI."""
    try:
        thumb_path = THUMB_DIR / f"{media_id}.jpg"

        if media_type == "image":
            img = cv2.imread(str(src_path))
        else:
            cap = cv2.VideoCapture(str(src_path))
            ret, img = cap.read()
            cap.release()
            if not ret:
                return None

        if img is None:
            return None

        thumb = cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(thumb_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return thumb_path
    except Exception as ex:
        print(f"[Storage] Thumbnail generation failed: {ex}")
        return None


# ── Media Type Detection ──────────────────────────────────────────────────────

def _detect_media_type(ext: str) -> str:
    if ext in IMAGE_EXTS:
        return "image"
    elif ext in VIDEO_EXTS:
        return "video"
    else:
        return "unknown"


def get_file_info(media_id: str) -> Optional[dict]:
    """Return detailed file info for a media_id."""
    path = get_file_path(media_id)
    if path is None:
        return None
    ext = path.suffix.lower()
    stat = path.stat()
    return {
        "media_id": media_id,
        "path": str(path),
        "filename": path.name,
        "ext": ext,
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / 1e6, 2),
        "media_type": _detect_media_type(ext),
        "content_hash": sha256_file(path),
        "modified_at": datetime.utcfromtimestamp(stat.st_mtime).isoformat(),
        "thumbnail": str(get_thumbnail_path(media_id) or ""),
    }


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import io
    print("=== Storage Self-Test ===")

    # Create a fake PNG
    dummy = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(dummy, (20, 20), (180, 180), (0, 255, 0), -1)
    _, encoded = cv2.imencode(".png", dummy)
    buf = io.BytesIO(encoded.tobytes())

    result = save_file(buf, "test_image.png")
    print(f"Saved: {result}")

    info = get_file_info(result["media_id"])
    print(f"Info: {info}")

    listing = list_files(limit=5)
    print(f"Files in uploads: {len(listing)}")

    deleted = delete_file(result["media_id"])
    print(f"Deleted: {deleted}")
    print("Self-test PASSED ✓")