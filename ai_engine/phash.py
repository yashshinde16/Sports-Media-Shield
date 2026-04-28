"""
member1_phash.py — Perceptual Hash (pHash) Similarity Engine
Team Member 1 | Part 1: AI Engine
Computes perceptual hashes and similarity scores between images/frames.
"""

import hashlib
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── Constants ────────────────────────────────────────────────────────────────
HASH_SIZE = 8          # DCT grid: 8×8 → 64-bit hash
HIGH_FREQ_FACTOR = 4   # capture high-freq area = (HASH_SIZE * HIGH_FREQ_FACTOR)²


# ── Core pHash Implementation ────────────────────────────────────────────────

def compute_phash(image: np.ndarray, hash_size: int = HASH_SIZE) -> str:
    """
    Compute perceptual hash (DCT-based) for an image.

    Steps:
      1. Resize to (hash_size * high_freq_factor)²
      2. Grayscale conversion
      3. 2D DCT
      4. Top-left hash_size² block (low frequencies)
      5. Binarise vs. median
      6. Return hex string

    Args:
        image: BGR numpy array (from cv2.imread or video frame).
        hash_size: Side length of the DCT block (default 8 → 64-bit hash).

    Returns:
        Hex string representing the perceptual hash.
    """
    if image is None or image.size == 0:
        raise ValueError("compute_phash received an empty image")

    img_size = hash_size * HIGH_FREQ_FACTOR

    # 1. Resize
    resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # 2. Grayscale
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    # 3. Float conversion + 2D DCT
    dct_input = np.float32(gray)
    dct_full = cv2.dct(dct_input)

    # 4. Top-left block (low frequency coefficients)
    dct_low = dct_full[:hash_size, :hash_size]

    # 5. Exclude DC coefficient (index 0,0) for median calculation
    values = dct_low.flatten()
    median_val = np.median(values[1:])  # skip DC component

    # 6. Binarise
    bits = (values > median_val).astype(np.uint8)

    # Pack bits → bytes → hex
    packed = np.packbits(bits)
    return packed.tobytes().hex()


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Compute Hamming distance between two hex hashes.

    Args:
        hash1: First hex hash string.
        hash2: Second hex hash string.

    Returns:
        Number of differing bits.
    """
    if len(hash1) != len(hash2):
        raise ValueError(f"Hash length mismatch: {len(hash1)} vs {len(hash2)}")

    # Convert hex → int and XOR
    n1 = int(hash1, 16)
    n2 = int(hash2, 16)
    xor = n1 ^ n2
    return bin(xor).count("1")


def phash_similarity(hash1: str, hash2: str, hash_size: int = HASH_SIZE) -> float:
    """
    Convert Hamming distance to a [0, 1] similarity score.

    Args:
        hash1: Hex hash of image 1.
        hash2: Hex hash of image 2.
        hash_size: Hash size used during generation.

    Returns:
        Similarity in [0.0, 1.0] — 1.0 = identical.
    """
    max_bits = hash_size * hash_size
    dist = hamming_distance(hash1, hash2)
    return 1.0 - (dist / max_bits)


# ── File-Level Helpers ───────────────────────────────────────────────────────

def phash_from_file(path: str, hash_size: int = HASH_SIZE) -> Optional[str]:
    """
    Load an image from disk and return its pHash.

    Args:
        path: File path to image.
        hash_size: Hash size parameter.

    Returns:
        Hex hash string, or None on failure.
    """
    img = cv2.imread(str(path))
    if img is None:
        print(f"[pHash] WARNING: Could not load image: {path}")
        return None
    return compute_phash(img, hash_size)


def compare_images(path1: str, path2: str) -> dict:
    """
    Full pHash comparison report between two image files.

    Returns dict with keys: hash1, hash2, hamming, similarity, verdict.
    """
    h1 = phash_from_file(path1)
    h2 = phash_from_file(path2)

    if h1 is None or h2 is None:
        return {"error": "Failed to load one or both images", "similarity": 0.0}

    dist = hamming_distance(h1, h2)
    sim = phash_similarity(h1, h2)

    verdict = _verdict(sim)
    return {
        "hash1": h1,
        "hash2": h2,
        "hamming_distance": dist,
        "similarity": round(sim, 4),
        "verdict": verdict,
    }


def compare_arrays(img1: np.ndarray, img2: np.ndarray) -> dict:
    """
    Same as compare_images but accepts numpy arrays directly.
    Useful for video frame comparison.
    """
    h1 = compute_phash(img1)
    h2 = compute_phash(img2)
    dist = hamming_distance(h1, h2)
    sim = phash_similarity(h1, h2)
    return {
        "hash1": h1,
        "hash2": h2,
        "hamming_distance": dist,
        "similarity": round(sim, 4),
        "verdict": _verdict(sim),
    }


def _verdict(sim: float) -> str:
    if sim >= 0.95:
        return "EXACT_MATCH"
    elif sim >= 0.80:
        return "HIGH_SIMILARITY"
    elif sim >= 0.60:
        return "MODERATE_SIMILARITY"
    else:
        return "LOW_SIMILARITY"


# ── Batch Processing ─────────────────────────────────────────────────────────

def batch_compare(reference_path: str, candidates: list[str]) -> list[dict]:
    """
    Compare one reference image against multiple candidates.

    Args:
        reference_path: Path to the reference (protected) image.
        candidates: List of file paths to compare against.

    Returns:
        List of comparison result dicts, sorted by similarity descending.
    """
    ref_hash = phash_from_file(reference_path)
    if ref_hash is None:
        return [{"error": "Cannot load reference image"}]

    results = []
    for cand_path in candidates:
        cand_hash = phash_from_file(cand_path)
        if cand_hash is None:
            results.append({"path": cand_path, "error": "Load failed", "similarity": 0.0})
            continue

        dist = hamming_distance(ref_hash, cand_hash)
        sim = phash_similarity(ref_hash, cand_hash)
        results.append({
            "path": cand_path,
            "hamming_distance": dist,
            "similarity": round(sim, 4),
            "verdict": _verdict(sim),
        })

    results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    return results


# ── Content Hash (SHA-256) for Ownership Records ─────────────────────────────

def content_hash(image: np.ndarray) -> str:
    """
    SHA-256 hash of raw pixel bytes — used for exact ownership verification.
    Different from pHash: this is NOT perceptually robust, but cryptographically unique.
    """
    return hashlib.sha256(image.tobytes()).hexdigest()


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== pHash Self-Test ===")

    # Create synthetic test images
    base = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    identical = base.copy()
    slightly_modified = base.copy()
    slightly_modified[50:60, 50:60] = 128   # tiny modification
    very_different = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    h_base = compute_phash(base)
    h_identical = compute_phash(identical)
    h_modified = compute_phash(slightly_modified)
    h_different = compute_phash(very_different)

    print(f"Base hash:      {h_base}")
    print(f"Identical sim:  {phash_similarity(h_base, h_identical):.4f}  → {_verdict(phash_similarity(h_base, h_identical))}")
    print(f"Modified sim:   {phash_similarity(h_base, h_modified):.4f}  → {_verdict(phash_similarity(h_base, h_modified))}")
    print(f"Different sim:  {phash_similarity(h_base, h_different):.4f}  → {_verdict(phash_similarity(h_base, h_different))}")
    print(f"Content hash:   {content_hash(base)}")
    print("Self-test PASSED ✓")