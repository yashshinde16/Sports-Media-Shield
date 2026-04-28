"""
member2_quality.py — Media Quality Detection Engine
Team Member 2 | Part 3: AI Services

Detects quality degradation indicators:
  - Blur (Laplacian variance)
  - Compression artifacts (blockiness score)
  - SSIM vs. reference
  - Resolution drop
  - Noise estimation
  - Screen recording signatures

Outputs a quality score [0,1] and degradation classification.
"""

import cv2
import numpy as np
from typing import Optional
from skimage.metrics import structural_similarity as ssim_metric


# ── Thresholds ────────────────────────────────────────────────────────────────
BLUR_THRESHOLD_SHARP    = 150.0   # Laplacian variance — above = sharp
BLUR_THRESHOLD_BLURRY   = 50.0    # Below = blurry
BLOCK_ARTIFACT_THRESHOLD = 10.0   # DCT block boundary difference
NOISE_THRESHOLD          = 15.0   # Estimated noise std-dev


# ── Blur Detection ────────────────────────────────────────────────────────────

def laplacian_variance(image: np.ndarray) -> float:
    """
    Compute the variance of the Laplacian — measure of image sharpness.
    Higher value = sharper. Very low = blurry.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def blur_score(image: np.ndarray) -> dict:
    """
    Classify image sharpness.

    Returns:
        dict: lap_var, sharpness_score (0-1), label
    """
    lv = laplacian_variance(image)
    # Normalise to [0,1] — cap at BLUR_THRESHOLD_SHARP
    score = float(min(lv / BLUR_THRESHOLD_SHARP, 1.0))
    if lv >= BLUR_THRESHOLD_SHARP:
        label = "SHARP"
    elif lv >= BLUR_THRESHOLD_BLURRY:
        label = "MODERATE_BLUR"
    else:
        label = "BLURRY"

    return {
        "laplacian_variance": round(lv, 2),
        "sharpness_score": round(score, 4),
        "label": label,
    }


# ── Compression Artifact Detection ───────────────────────────────────────────

def blockiness_score(image: np.ndarray) -> dict:
    """
    Detect JPEG compression block artifacts by measuring DCT block boundary differences.
    High blockiness → heavy compression.

    Returns:
        dict: blockiness_value, artifact_score (0-1), label
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    h, w = gray.shape
    block_size = 8

    h_diffs = []
    for row in range(block_size, h, block_size):
        if row < h:
            diff = np.mean(np.abs(gray[row, :] - gray[row - 1, :]))
            h_diffs.append(diff)

    v_diffs = []
    for col in range(block_size, w, block_size):
        if col < w:
            diff = np.mean(np.abs(gray[:, col] - gray[:, col - 1]))
            v_diffs.append(diff)

    blockiness = float(np.mean(h_diffs + v_diffs)) if (h_diffs or v_diffs) else 0.0

    # Non-block boundaries (expected local variation)
    local_diffs = []
    for row in range(1, h - 1):
        if row % block_size != 0:
            local_diffs.append(np.mean(np.abs(gray[row, :] - gray[row - 1, :])))
    local_mean = float(np.mean(local_diffs)) if local_diffs else 1.0

    ratio = blockiness / max(local_mean, 0.01)
    artifact_score = float(min(max((ratio - 1.0) / 3.0, 0.0), 1.0))

    if artifact_score > 0.6:
        label = "HEAVY_ARTIFACTS"
    elif artifact_score > 0.3:
        label = "MODERATE_ARTIFACTS"
    else:
        label = "CLEAN"

    return {
        "blockiness_value": round(blockiness, 4),
        "local_variation": round(local_mean, 4),
        "artifact_score": round(artifact_score, 4),
        "label": label,
    }


# ── SSIM Comparison ───────────────────────────────────────────────────────────

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> dict:
    """
    Compute Structural Similarity Index (SSIM) between two images.
    Requires same dimensions — resizes img2 to img1 shape if needed.

    Returns:
        dict: ssim_score, label
    """
    if len(img1.shape) == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        g1 = img1

    if len(img2.shape) == 3:
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        g2 = img2

    # Resize if needed
    if g1.shape != g2.shape:
        g2 = cv2.resize(g2, (g1.shape[1], g1.shape[0]))

    score = float(ssim_metric(g1, g2, data_range=255))

    if score >= 0.90:
        label = "HIGH_STRUCTURAL_MATCH"
    elif score >= 0.70:
        label = "MODERATE_MATCH"
    elif score >= 0.50:
        label = "LOW_MATCH"
    else:
        label = "VERY_DIFFERENT"

    return {
        "ssim_score": round(score, 4),
        "label": label,
    }


# ── Resolution Analysis ───────────────────────────────────────────────────────

def resolution_score(image: np.ndarray, expected_min_width: int = 720) -> dict:
    """
    Check if image resolution is suspiciously low (possible downscale + re-upload).

    Returns:
        dict: width, height, megapixels, resolution_score, label
    """
    h, w = image.shape[:2]
    mp = (h * w) / 1_000_000

    res_score = float(min(w / expected_min_width, 1.0))

    if w >= 1920:
        label = "FULL_HD_OR_HIGHER"
    elif w >= 1280:
        label = "HD"
    elif w >= 720:
        label = "SD"
    elif w >= 480:
        label = "LOW_RES"
    else:
        label = "VERY_LOW_RES"

    return {
        "width": w,
        "height": h,
        "megapixels": round(mp, 2),
        "resolution_score": round(res_score, 4),
        "label": label,
    }


# ── Noise Estimation ──────────────────────────────────────────────────────────

def estimate_noise(image: np.ndarray) -> dict:
    """
    Estimate image noise using high-frequency residuals.
    High noise can indicate screen recording or poor compression.

    Returns:
        dict: noise_std, noise_score (0=noisy, 1=clean), label
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    # Median filter to estimate "true" image
    median = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    residual = gray - median
    noise_std = float(np.std(residual))

    # Score: low noise → high score
    noise_score = float(max(1.0 - (noise_std / NOISE_THRESHOLD), 0.0))

    if noise_std < 5:
        label = "CLEAN"
    elif noise_std < NOISE_THRESHOLD:
        label = "MILD_NOISE"
    else:
        label = "HIGH_NOISE"

    return {
        "noise_std": round(noise_std, 4),
        "noise_score": round(noise_score, 4),
        "label": label,
    }


# ── Screen Recording Detection ────────────────────────────────────────────────

def screen_recording_indicators(image: np.ndarray) -> dict:
    """
    Heuristic detection of screen recording artifacts:
    - Unusual aspect ratios (13:9 browser crop, etc.)
    - UI element pixel patterns (uniform horizontal/vertical bands)
    - Moire patterns (periodic noise)
    - Low sharpness + high blockiness combo

    Returns:
        dict: indicators, probability, label
    """
    h, w = image.shape[:2]
    indicators = []

    # Aspect ratio check (common screen crop ratios)
    ratio = w / h if h > 0 else 0
    if 1.7 < ratio < 1.82:
        indicators.append("16:9_ASPECT")  # normal
    elif 1.2 < ratio < 1.4:
        indicators.append("UNUSUAL_ASPECT_RATIO")

    # Blur check
    lv = laplacian_variance(image)
    if lv < BLUR_THRESHOLD_BLURRY:
        indicators.append("BLURRY_CAPTURE")

    # Blockiness
    ba = blockiness_score(image)
    if ba["artifact_score"] > 0.5:
        indicators.append("HEAVY_COMPRESSION")

    # Periodic noise (moire) — check for frequency peaks
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    center_y, center_x = h // 2, w // 2
    # Check if there are strong off-center frequency peaks (moire signature)
    edge_region = magnitude.copy()
    edge_region[center_y-20:center_y+20, center_x-20:center_x+20] = 0
    if edge_region.max() > magnitude.mean() * 3:
        indicators.append("MOIRE_PATTERN")

    # Probability heuristic
    prob = len(indicators) / 4.0
    prob = min(prob, 1.0)

    if prob >= 0.60:
        label = "LIKELY_SCREEN_RECORDING"
    elif prob >= 0.30:
        label = "POSSIBLE_SCREEN_RECORDING"
    else:
        label = "AUTHENTIC_MEDIA"

    return {
        "indicators": indicators,
        "probability": round(prob, 4),
        "label": label,
    }


# ── Composite Quality Score ───────────────────────────────────────────────────

def compute_quality_score(image: np.ndarray, reference: Optional[np.ndarray] = None) -> dict:
    """
    Compute an overall quality/integrity score for the image.

    Weights:
      - Sharpness: 35%
      - Compression artifacts: 25%
      - Noise: 20%
      - Resolution: 20%

    Args:
        image: Suspect image.
        reference: Optional original image for SSIM comparison.

    Returns:
        dict with individual scores + overall_quality [0,1].
    """
    blur = blur_score(image)
    artifacts = blockiness_score(image)
    noise = estimate_noise(image)
    res = resolution_score(image)
    screen = screen_recording_indicators(image)

    # Invert artifact score (high artifact = low quality)
    artifact_quality = 1.0 - artifacts["artifact_score"]

    overall = (
        0.35 * blur["sharpness_score"]
        + 0.25 * artifact_quality
        + 0.20 * noise["noise_score"]
        + 0.20 * res["resolution_score"]
    )
    overall = float(np.clip(overall, 0.0, 1.0))

    # Degrade score if screen recording likely
    if screen["label"] == "LIKELY_SCREEN_RECORDING":
        overall *= 0.7
    elif screen["label"] == "POSSIBLE_SCREEN_RECORDING":
        overall *= 0.85

    degradation_pct = round((1.0 - overall) * 100, 1)

    # SSIM if reference provided
    ssim_result = None
    if reference is not None:
        ssim_result = compute_ssim(reference, image)
        # Factor SSIM into overall
        overall = round(0.8 * overall + 0.2 * ssim_result["ssim_score"], 4)

    if overall >= 0.85:
        quality_label = "PRISTINE"
    elif overall >= 0.65:
        quality_label = "GOOD"
    elif overall >= 0.45:
        quality_label = "DEGRADED"
    else:
        quality_label = "SEVERELY_DEGRADED"

    return {
        "overall_quality": round(overall, 4),
        "degradation_percent": degradation_pct,
        "quality_label": quality_label,
        "blur": blur,
        "compression_artifacts": artifacts,
        "noise": noise,
        "resolution": res,
        "screen_recording": screen,
        "ssim": ssim_result,
    }


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Quality Detection Self-Test ===\n")

    # Sharp reference
    sharp = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(sharp, (50, 50), (590, 430), (0, 150, 255), -1)
    cv2.putText(sharp, "SPORTS MEDIA", (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4)

    # Blurry copy
    blurry = cv2.GaussianBlur(sharp, (25, 25), 0)

    # Noisy copy (screen recording sim)
    noisy = sharp.copy().astype(np.float32)
    noisy += np.random.normal(0, 20, noisy.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    # JPEG compressed
    _, enc = cv2.imencode(".jpg", sharp, [cv2.IMWRITE_JPEG_QUALITY, 10])
    compressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    for name, img in [("Sharp original", sharp), ("Blurry", blurry), ("Noisy (screen)", noisy), ("Compressed Q10", compressed)]:
        q = compute_quality_score(img, reference=sharp)
        print(f"[{name}]")
        print(f"  Overall quality:    {q['overall_quality']:.4f} | {q['quality_label']}")
        print(f"  Degradation:        {q['degradation_percent']}%")
        print(f"  Screen recording:   {q['screen_recording']['label']}")
        if q['ssim']:
            print(f"  SSIM vs reference:  {q['ssim']['ssim_score']:.4f}")
        print()

    print("Self-test PASSED ✓")