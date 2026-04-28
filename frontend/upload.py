"""
member2_upload.py — Upload Helper & Pre-processing Module
Team Member 2 | Part 4: Frontend + Automation

Handles all upload-side logic:
  - File validation (type, size, corruption)
  - Image pre-processing (resize, normalise, denoise)
  - Batch upload support
  - URL-based image fetching
  - Upload progress reporting
"""

import io
import sys
import uuid
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Union, BinaryIO

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_IMAGE_MB   = 50
MAX_VIDEO_MB   = 200
MAX_DIM        = 4096          # max width or height before auto-resize
TARGET_DIM     = 1280          # resize target for very large images
SUPPORTED_IMG  = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
SUPPORTED_VID  = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SUPPORTED_ALL  = SUPPORTED_IMG | SUPPORTED_VID

URL_TIMEOUT    = 10            # seconds


# ── File Validation ───────────────────────────────────────────────────────────

def validate_file(
    filename: str,
    file_bytes: bytes,
    allow_video: bool = True,
) -> dict:
    """
    Validate a file before processing.

    Returns:
        dict: valid (bool), ext, media_type, size_mb, errors (list)
    """
    errors = []
    ext = Path(filename).suffix.lower()

    # Extension check
    allowed = SUPPORTED_ALL if allow_video else SUPPORTED_IMG
    if ext not in allowed:
        errors.append(f"Unsupported format '{ext}'. Allowed: {sorted(allowed)}")

    # Size check
    size_mb = len(file_bytes) / 1_000_000
    max_mb = MAX_VIDEO_MB if ext in SUPPORTED_VID else MAX_IMAGE_MB
    if size_mb > max_mb:
        errors.append(f"File too large: {size_mb:.1f}MB (max {max_mb}MB)")

    # Integrity check — try to decode
    media_type = "video" if ext in SUPPORTED_VID else "image"
    if media_type == "image" and not errors:
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            errors.append("Cannot decode image — file may be corrupt or unsupported")

    return {
        "valid": len(errors) == 0,
        "ext": ext,
        "media_type": media_type,
        "size_mb": round(size_mb, 2),
        "errors": errors,
    }


def validate_url(url: str) -> dict:
    """
    Lightweight URL validation (no fetch).
    Returns dict: valid, reason
    """
    url = url.strip()
    if not url:
        return {"valid": False, "reason": "Empty URL"}
    if not (url.startswith("http://") or url.startswith("https://")):
        return {"valid": False, "reason": "URL must start with http:// or https://"}
    if len(url) > 2048:
        return {"valid": False, "reason": "URL too long"}
    return {"valid": True, "reason": "ok"}


# ── Image Loading ─────────────────────────────────────────────────────────────

def load_image_from_bytes(data: bytes) -> Optional[np.ndarray]:
    """Decode image bytes to BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def load_image_from_file(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Load image from disk path."""
    return cv2.imread(str(path))


def load_image_from_url(url: str, timeout: int = URL_TIMEOUT) -> tuple[Optional[np.ndarray], str]:
    """
    Download and decode an image from a URL.

    Returns:
        (image_array, error_message) — image is None on failure.
    """
    v = validate_url(url)
    if not v["valid"]:
        return None, v["reason"]

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SportsmediaShield/1.0 (+https://github.com/shield)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()

        img = load_image_from_bytes(data)
        if img is None:
            return None, "Downloaded data is not a valid image"

        return img, ""

    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return None, f"URL error: {e.reason}"
    except Exception as e:
        return None, f"Fetch failed: {e}"


def load_image_from_streamlit(uploaded_file) -> Optional[np.ndarray]:
    """
    Load image from a Streamlit UploadedFile object.
    Resets seek position after reading so file can be re-used.
    """
    data = uploaded_file.read()
    uploaded_file.seek(0)
    return load_image_from_bytes(data)


# ── Image Pre-processing ──────────────────────────────────────────────────────

def preprocess_image(
    image: np.ndarray,
    max_dim: int = TARGET_DIM,
    denoise: bool = False,
    normalise_brightness: bool = False,
) -> np.ndarray:
    """
    Pre-process an image for AI analysis:
      1. Resize if larger than max_dim
      2. Optional denoising (mild Gaussian)
      3. Optional brightness normalisation (CLAHE)

    Args:
        image: BGR numpy array.
        max_dim: Maximum dimension (width or height).
        denoise: Apply mild denoising.
        normalise_brightness: Apply CLAHE brightness normalisation.

    Returns:
        Pre-processed BGR numpy array.
    """
    h, w = image.shape[:2]

    # Resize if oversized
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Optional denoising
    if denoise:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    # Optional brightness normalisation
    if normalise_brightness:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return image


def resize_to_match(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize img2 to match img1's dimensions.
    Used for SSIM comparison which requires same size.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if (h1, w1) != (h2, w2):
        img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_AREA)
    return img1, img2


# ── Save Helpers ──────────────────────────────────────────────────────────────

def save_image(image: np.ndarray, filename: str = "", media_id: str = "") -> dict:
    """
    Save a numpy image array to the uploads directory.

    Args:
        image: BGR numpy array.
        filename: Original filename (used for extension).
        media_id: Override auto UUID.

    Returns:
        dict with media_id, path, size_bytes, sha256.
    """
    if not media_id:
        media_id = str(uuid.uuid4())

    ext = Path(filename).suffix.lower() if filename else ".jpg"
    if ext not in SUPPORTED_IMG:
        ext = ".jpg"

    dest = UPLOAD_DIR / f"{media_id}{ext}"

    # Encode and write
    quality_params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in {".jpg", ".jpeg"} else []
    success = cv2.imwrite(str(dest), image, quality_params)

    if not success:
        raise IOError(f"Failed to write image to {dest}")

    size = dest.stat().st_size
    sha = hashlib.sha256(image.tobytes()).hexdigest()

    return {
        "media_id": media_id,
        "path": str(dest),
        "filename": dest.name,
        "size_bytes": size,
        "sha256": sha,
    }


def save_bytes_to_uploads(data: bytes, original_filename: str) -> dict:
    """Save raw file bytes to uploads directory."""
    media_id = str(uuid.uuid4())
    ext = Path(original_filename).suffix.lower() or ".bin"
    dest = UPLOAD_DIR / f"{media_id}{ext}"

    with open(dest, "wb") as f:
        f.write(data)

    return {
        "media_id": media_id,
        "path": str(dest),
        "filename": dest.name,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


# ── Batch URL Fetcher ─────────────────────────────────────────────────────────

def fetch_images_from_urls(
    urls: list[str],
    max_count: int = 20,
    progress_callback=None,
) -> list[dict]:
    """
    Download multiple images from URLs.

    Args:
        urls: List of image URLs.
        max_count: Maximum URLs to process.
        progress_callback: Optional callable(i, total, url) for progress reporting.

    Returns:
        List of dicts: url, image (ndarray or None), error, status
    """
    results = []
    urls = urls[:max_count]

    for i, url in enumerate(urls):
        if progress_callback:
            progress_callback(i, len(urls), url)

        img, err = load_image_from_url(url)
        results.append({
            "url": url,
            "image": img,
            "error": err,
            "status": "ok" if img is not None else "failed",
            "index": i,
        })

    return results


# ── Image Info ────────────────────────────────────────────────────────────────

def image_info(image: np.ndarray) -> dict:
    """Return basic metadata about an image array."""
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    return {
        "width": w,
        "height": h,
        "channels": channels,
        "megapixels": round(w * h / 1_000_000, 2),
        "aspect_ratio": round(w / h, 3) if h > 0 else 0,
        "dtype": str(image.dtype),
    }


def encode_image_to_base64(image: np.ndarray, ext: str = ".jpg") -> str:
    """Encode image to base64 string (for API/HTML embedding)."""
    import base64
    _, buffer = cv2.imencode(ext, image)
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def decode_image_from_base64(b64_str: str) -> Optional[np.ndarray]:
    """Decode base64 string back to image array."""
    import base64
    data = base64.b64decode(b64_str)
    return load_image_from_bytes(data)


# ── Streamlit-specific helpers ────────────────────────────────────────────────

def streamlit_image_preview(
    image: np.ndarray,
    caption: str = "",
    max_width: int = 600,
) -> np.ndarray:
    """
    Prepare an image for Streamlit display:
    - Convert BGR→RGB
    - Resize to max_width if larger

    Returns the display-ready RGB array.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if w > max_width:
        scale = max_width / w
        rgb = cv2.resize(rgb, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return rgb


def image_diff_overlay(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Generate a visual difference overlay between two images.
    Highlights regions that differ (red overlay on black background).
    Both images resized to same dimensions first.
    """
    img1, img2 = resize_to_match(img1, img2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    overlay = img1.copy()
    overlay[thresh > 0] = [0, 0, 200]  # Red highlight on differences
    blended = cv2.addWeighted(img1, 0.6, overlay, 0.4, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Upload Module Self-Test ===\n")

    # Create a test image
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (750, 550), (0, 100, 200), -1)
    cv2.putText(img, "SPORTS MEDIA SHIELD", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)

    print(f"Image info: {image_info(img)}")

    # Test save
    saved = save_image(img, "test.jpg")
    print(f"Saved: {saved['path']} ({saved['size_bytes']} bytes)")

    # Test preprocess
    small = preprocess_image(img, max_dim=400)
    print(f"Resized to: {small.shape[1]}×{small.shape[0]}")

    # Test validation
    _, buf = cv2.imencode(".jpg", img)
    val = validate_file("test.jpg", buf.tobytes())
    print(f"Validation: {val}")

    # Test URL validation
    print(f"URL valid: {validate_url('https://example.com/image.jpg')}")
    print(f"URL invalid: {validate_url('not-a-url')}")

    # Test diff overlay
    img2 = img.copy()
    img2[100:200, 100:400] = 128
    diff = image_diff_overlay(img, img2)
    print(f"Diff overlay shape: {diff.shape}")

    # Cleanup
    import os
    os.unlink(saved["path"])

    print("\nSelf-test PASSED ✓")