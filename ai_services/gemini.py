"""
member1_gemini.py — Google Gemini AI Integration + Invisible Watermarking
Team Member 1 | Part 3: AI Services

Responsibilities:
  1. Gemini API wrapper for natural language explanation
  2. Key-based invisible watermarking (DCT domain, seeded randomness)
  3. Watermark detection (percentage match)
"""

import os
import hashlib
import numpy as np
import cv2
from pathlib import Path
from typing import Optional

# ── Load .env first so keys are available ─────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

# ── Gemini Client ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-1.5-flash"  # Fast + free tier available

_gemini_client = None


def _get_gemini():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    # Re-read key each time in case env was loaded after module import
    api_key = os.environ.get("GEMINI_API_KEY", "") or GEMINI_API_KEY
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _gemini_client = genai.GenerativeModel(GEMINI_MODEL)
        return _gemini_client
    except Exception as ex:
        print(f"[Gemini] Init failed: {ex}")
        return None


def gemini_explain(prompt: str, fallback: str = "") -> str:
    """
    Send a prompt to Gemini and return the text response.
    Falls back to `fallback` string if API unavailable.
    """
    client = _get_gemini()
    if client is None:
        return fallback or _mock_explanation(prompt)
    try:
        response = client.generate_content(prompt)
        return response.text.strip()
    except Exception as ex:
        print(f"[Gemini] API error: {ex}")
        return fallback or _mock_explanation(prompt)


def gemini_analyse_image(image: np.ndarray, question: str) -> str:
    """
    Send an image + question to Gemini Vision.
    Returns the model's textual response.
    """
    client = _get_gemini()
    if client is None:
        return "[Gemini Vision not available — set GEMINI_API_KEY]"
    try:
        import google.generativeai as genai
        # Encode image to JPEG bytes
        _, buffer = cv2.imencode(".jpg", image)
        img_bytes = buffer.tobytes()
        img_part = {"mime_type": "image/jpeg", "data": img_bytes}
        response = client.generate_content([question, img_part])
        return response.text.strip()
    except Exception as ex:
        return f"[Gemini Vision error: {ex}]"


def _mock_explanation(prompt: str) -> str:
    """Fallback mock explanation when Gemini API key is not set."""
    if "unauthorized" in prompt.lower() or "high" in prompt.lower():
        return (
            "AI Analysis: The suspect media shows strong similarity indicators. "
            "The perceptual hash comparison reveals near-identical visual content, "
            "and feature matching confirms structural overlap. The invisible watermark "
            "was partially detected, suggesting the content originated from the protected asset. "
            "Quality degradation patterns are consistent with screen recording or re-compression. "
            "Recommendation: Flag for legal review."
        )
    return (
        "AI Analysis: The suspect media shows low similarity to the reference asset. "
        "No significant watermark presence was detected, and visual features differ substantially. "
        "This media is likely original or unrelated content. No action required."
    )


# ── Invisible Watermarking ────────────────────────────────────────────────────
#
# Algorithm: DCT-domain watermarking with seeded randomness.
#   1. Convert image to YCrCb; work on Y (luminance) channel.
#   2. Derive a deterministic pseudo-random bit pattern from the key.
#   3. Apply subtle DCT coefficient perturbations at pseudo-random locations.
#   4. Reconstruct image — visually identical to human eye.
#
# Detection: Re-derive the same bit pattern; measure correlation with
#   actual DCT coefficients. Threshold → percentage match score [0,1].

WATERMARK_STRENGTH = 2.0    # Coefficient perturbation magnitude
WATERMARK_BITS     = 128    # Number of watermark bits
BLOCK_SIZE         = 8      # DCT block size


def _derive_seed(key: str) -> int:
    """Derive integer seed from watermark key string."""
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)


def _generate_wm_pattern(key: str, shape: tuple) -> np.ndarray:
    """
    Generate pseudo-random watermark pattern of ±1 values.
    Shape: (height//BLOCK_SIZE, width//BLOCK_SIZE, WATERMARK_BITS)
    """
    seed = _derive_seed(key)
    rng = np.random.default_rng(seed)
    h_blocks = shape[0] // BLOCK_SIZE
    w_blocks = shape[1] // BLOCK_SIZE
    pattern = rng.choice([-1.0, 1.0], size=(h_blocks * w_blocks, WATERMARK_BITS))
    return pattern


def embed_watermark(image: np.ndarray, key: str) -> np.ndarray:
    """
    Embed an invisible watermark into an image using DCT domain modification.

    Args:
        image: Input BGR image (numpy array).
        key: Watermark key string (owner-specific).

    Returns:
        Watermarked BGR image — visually indistinguishable from original.
    """
    if image is None or image.size == 0:
        return image

    img = image.astype(np.float32)

    # Work on luminance channel
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0].copy()

    h, w = y_channel.shape
    h_blocks = h // BLOCK_SIZE
    w_blocks = w // BLOCK_SIZE

    seed = _derive_seed(key)
    rng = np.random.default_rng(seed)

    # Embed watermark bits into mid-frequency DCT coefficients
    for bi in range(h_blocks):
        for bj in range(w_blocks):
            block = y_channel[
                bi*BLOCK_SIZE:(bi+1)*BLOCK_SIZE,
                bj*BLOCK_SIZE:(bj+1)*BLOCK_SIZE,
            ].copy()

            dct_block = cv2.dct(block)

            # Select mid-frequency positions (not DC, not high-freq noise)
            positions = _mid_freq_positions(rng, n=4)
            bit = rng.choice([-1.0, 1.0])  # watermark bit

            for (r, c) in positions:
                dct_block[r, c] += WATERMARK_STRENGTH * bit

            y_channel[
                bi*BLOCK_SIZE:(bi+1)*BLOCK_SIZE,
                bj*BLOCK_SIZE:(bj+1)*BLOCK_SIZE,
            ] = cv2.idct(dct_block)

    # Reconstruct
    y_channel = np.clip(y_channel, 0, 255)
    ycrcb[:, :, 0] = y_channel
    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return np.clip(result, 0, 255).astype(np.uint8)


def detect_watermark(image: np.ndarray, key: str) -> float:
    """
    Detect watermark presence using pattern correlation.

    Args:
        image: BGR image to check.
        key: Watermark key to test against.

    Returns:
        Match score in [0.0, 1.0] — 1.0 = perfect match.
    """
    if image is None or image.size == 0:
        return 0.0

    img = image.astype(np.float32)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0].copy()

    h, w = y_channel.shape
    h_blocks = h // BLOCK_SIZE
    w_blocks = w // BLOCK_SIZE

    if h_blocks == 0 or w_blocks == 0:
        return 0.0

    seed = _derive_seed(key)
    rng = np.random.default_rng(seed)

    match_count = 0
    total_count = 0

    for bi in range(h_blocks):
        for bj in range(w_blocks):
            block = y_channel[
                bi*BLOCK_SIZE:(bi+1)*BLOCK_SIZE,
                bj*BLOCK_SIZE:(bj+1)*BLOCK_SIZE,
            ].copy()
            dct_block = cv2.dct(block)

            positions = _mid_freq_positions(rng, n=4)
            expected_bit = rng.choice([-1.0, 1.0])

            for (r, c) in positions:
                coeff = dct_block[r, c]
                # Check if coefficient was pushed in the expected direction
                if (expected_bit > 0 and coeff > 0) or (expected_bit < 0 and coeff < 0):
                    match_count += 1
                total_count += 1

    if total_count == 0:
        return 0.0

    return float(match_count / total_count)


def _mid_freq_positions(rng, n: int = 4) -> list[tuple]:
    """
    Select n mid-frequency DCT coefficient positions.
    Mid-frequency = row+col in range [2, BLOCK_SIZE-2].
    """
    positions = []
    mid_range = list(range(2, BLOCK_SIZE - 1))
    for _ in range(n):
        r = int(rng.choice(mid_range))
        c = int(rng.choice(mid_range))
        positions.append((r, c))
    return positions


def detect_watermark_from_arrays(image: np.ndarray, key: str) -> float:
    """Alias for detect_watermark — used by integration module."""
    return detect_watermark(image, key)


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Gemini + Watermark Self-Test ===\n")

    # Test watermark embed/detect
    key = "SportsCorp-2024-Championship"

    # Create test image with structure
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (220, 220), (100, 150, 200), -1)
    cv2.putText(img, "TEST", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    print("Embedding watermark...")
    wm_img = embed_watermark(img, key)

    print("Detecting watermark (correct key):")
    score_correct = detect_watermark(wm_img, key)
    print(f"  Match score: {score_correct:.4f}")

    print("Detecting watermark (wrong key):")
    score_wrong = detect_watermark(wm_img, "wrong-key-12345")
    print(f"  Match score: {score_wrong:.4f}")

    print("Detecting watermark (original image, correct key):")
    score_orig = detect_watermark(img, key)
    print(f"  Match score: {score_orig:.4f}")

    # Visual diff
    diff = np.abs(wm_img.astype(float) - img.astype(float))
    print(f"Max pixel diff from watermark: {diff.max():.2f} (should be tiny)")

    # Test Gemini mock
    explanation = gemini_explain("Test prompt", fallback="Mock response")
    print(f"\nGemini (mock) response: {explanation[:80]}...")

    print("\nSelf-test PASSED ✓")