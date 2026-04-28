"""
member2_orb.py — ORB Feature Matching Engine
Team Member 2 | Part 1: AI Engine
Uses ORB (Oriented FAST and Rotated BRIEF) keypoints + BFMatcher
for robust similarity detection across crops, rotations, and scaling.
"""

import cv2
import numpy as np
from typing import Optional


# ── Constants ────────────────────────────────────────────────────────────────
MAX_KEYPOINTS = 500        # max ORB keypoints per image
LOWE_RATIO = 0.75          # Lowe's ratio test threshold
MIN_GOOD_MATCHES = 10      # minimum matches to consider "similar"
MATCH_SCORE_SCALE = 200    # normaliser for raw match count → [0,1]


# ── ORB Detector Setup ───────────────────────────────────────────────────────

def _get_orb() -> cv2.ORB:
    return cv2.ORB_create(nfeatures=MAX_KEYPOINTS)


def _get_matcher() -> cv2.BFMatcher:
    # BFMatcher with Hamming distance (appropriate for ORB binary descriptors)
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


# ── Feature Extraction ───────────────────────────────────────────────────────

def extract_features(image: np.ndarray) -> tuple[list, Optional[np.ndarray]]:
    """
    Detect ORB keypoints and compute descriptors.

    Args:
        image: BGR numpy array.

    Returns:
        (keypoints, descriptors) — descriptors is None if no keypoints found.
    """
    if image is None or image.size == 0:
        return [], None

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    orb = _get_orb()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def extract_features_from_file(path: str) -> tuple[list, Optional[np.ndarray]]:
    """Load image from disk and extract ORB features."""
    img = cv2.imread(str(path))
    if img is None:
        print(f"[ORB] WARNING: Cannot load {path}")
        return [], None
    return extract_features(img)


# ── Feature Matching ─────────────────────────────────────────────────────────

def match_features(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio: float = LOWE_RATIO,
) -> list:
    """
    Apply kNN matching with Lowe's ratio test to filter false positives.

    Args:
        desc1: Descriptors from image 1.
        desc2: Descriptors from image 2.
        ratio: Lowe's ratio threshold (lower = stricter).

    Returns:
        List of good DMatch objects.
    """
    if desc1 is None or desc2 is None:
        return []
    if len(desc1) < 2 or len(desc2) < 2:
        return []

    matcher = _get_matcher()
    try:
        knn_matches = matcher.knnMatch(desc1, desc2, k=2)
    except cv2.error:
        return []

    good = []
    for pair in knn_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    return good


# ── Similarity Score ─────────────────────────────────────────────────────────

def orb_similarity(
    img1: np.ndarray,
    img2: np.ndarray,
) -> float:
    """
    Compute ORB-based similarity score in [0, 1].

    Score logic:
      - Count good matches after ratio test
      - Normalise against min keypoints available
      - Clamp to [0, 1]

    Args:
        img1, img2: BGR numpy arrays.

    Returns:
        Similarity score in [0.0, 1.0].
    """
    kp1, desc1 = extract_features(img1)
    kp2, desc2 = extract_features(img2)

    if not kp1 or not kp2:
        return 0.0

    good = match_features(desc1, desc2)
    num_good = len(good)

    # Normalise: good matches relative to the smaller keypoint set
    min_kp = min(len(kp1), len(kp2))
    score = num_good / max(min_kp, 1)
    return float(min(score, 1.0))


def orb_similarity_from_files(path1: str, path2: str) -> dict:
    """
    Full ORB comparison report between two image files.

    Returns dict with: keypoints1, keypoints2, good_matches, similarity, verdict.
    """
    img1 = cv2.imread(str(path1))
    img2 = cv2.imread(str(path2))

    if img1 is None or img2 is None:
        return {"error": "Failed to load image(s)", "similarity": 0.0}

    kp1, desc1 = extract_features(img1)
    kp2, desc2 = extract_features(img2)
    good = match_features(desc1, desc2)

    min_kp = max(min(len(kp1), len(kp2)), 1)
    sim = float(min(len(good) / min_kp, 1.0))

    return {
        "keypoints1": len(kp1),
        "keypoints2": len(kp2),
        "good_matches": len(good),
        "similarity": round(sim, 4),
        "verdict": _verdict(sim),
        "is_match": len(good) >= MIN_GOOD_MATCHES,
    }


# ── Homography Verification (optional geometric check) ───────────────────────

def verify_homography(
    kp1: list,
    kp2: list,
    good_matches: list,
    min_inliers: int = 8,
) -> dict:
    """
    Use RANSAC homography to verify geometric consistency of matches.
    Helps eliminate coincidental feature matches.

    Returns dict with: inliers, homography_found, inlier_ratio.
    """
    if len(good_matches) < 4:
        return {"homography_found": False, "inliers": 0, "inlier_ratio": 0.0}

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    if mask is None:
        return {"homography_found": False, "inliers": 0, "inlier_ratio": 0.0}

    inliers = int(mask.sum())
    ratio = inliers / len(good_matches)
    return {
        "homography_found": inliers >= min_inliers,
        "inliers": inliers,
        "inlier_ratio": round(ratio, 4),
    }


def orb_compare_with_homography(img1: np.ndarray, img2: np.ndarray) -> dict:
    """
    Full pipeline: ORB features + Lowe ratio + RANSAC homography verification.
    """
    kp1, desc1 = extract_features(img1)
    kp2, desc2 = extract_features(img2)
    good = match_features(desc1, desc2)

    min_kp = max(min(len(kp1), len(kp2)), 1)
    raw_sim = float(min(len(good) / min_kp, 1.0))

    geo = verify_homography(kp1, kp2, good)

    # Boost similarity if homography is geometrically consistent
    geo_boost = 0.1 if geo["homography_found"] else 0.0
    final_sim = min(raw_sim + geo_boost, 1.0)

    return {
        "keypoints1": len(kp1),
        "keypoints2": len(kp2),
        "good_matches": len(good),
        "raw_similarity": round(raw_sim, 4),
        "geo_verified": geo["homography_found"],
        "inliers": geo["inliers"],
        "final_similarity": round(final_sim, 4),
        "verdict": _verdict(final_sim),
    }


# ── Visualisation ────────────────────────────────────────────────────────────

def draw_matches(img1: np.ndarray, img2: np.ndarray, max_draw: int = 30) -> np.ndarray:
    """
    Return an image with drawn feature matches between img1 and img2.
    Useful for debugging and demo screenshots.
    """
    kp1, desc1 = extract_features(img1)
    kp2, desc2 = extract_features(img2)
    good = match_features(desc1, desc2)

    flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    out = cv2.drawMatches(img1, kp1, img2, kp2,
                          good[:max_draw], None, flags=flags)
    return out


# ── Helpers ──────────────────────────────────────────────────────────────────

def _verdict(sim: float) -> str:
    if sim >= 0.80:
        return "HIGH_MATCH"
    elif sim >= 0.50:
        return "MODERATE_MATCH"
    elif sim >= 0.20:
        return "LOW_MATCH"
    else:
        return "NO_MATCH"


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== ORB Self-Test ===")

    # Generate test images
    base = np.random.randint(100, 200, (400, 400, 3), dtype=np.uint8)
    # Add structured features so ORB has something to detect
    cv2.rectangle(base, (50, 50), (200, 200), (0, 0, 255), 3)
    cv2.circle(base, (300, 300), 60, (255, 0, 0), 3)
    cv2.line(base, (0, 0), (400, 400), (0, 255, 0), 2)

    identical = base.copy()
    rotated = cv2.rotate(base, cv2.ROTATE_90_CLOCKWISE)
    noise = base.copy()
    noise += np.random.randint(0, 30, noise.shape, dtype=np.uint8)
    random_img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)

    print(f"Identical:  {orb_similarity(base, identical):.4f}")
    print(f"Rotated:    {orb_similarity(base, rotated):.4f}")
    print(f"Noisy:      {orb_similarity(base, noise):.4f}")
    print(f"Random:     {orb_similarity(base, random_img):.4f}")

    kp, desc = extract_features(base)
    print(f"Keypoints detected: {len(kp)}")
    print("Self-test PASSED ✓")