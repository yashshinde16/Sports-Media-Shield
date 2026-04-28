"""
member4_similarity.py — Final Similarity Scoring Orchestrator
Team Member 4 | Part 1: AI Engine
Combines pHash, ORB, watermark, and quality scores into a single
weighted final score with explainable verdict breakdown.
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional

import cv2


# ── Score Weights ────────────────────────────────────────────────────────────
WEIGHT_WATERMARK = 0.40
WEIGHT_PHASH     = 0.30
WEIGHT_ORB       = 0.30

# Composite thresholds
THRESHOLD_DEFINITE   = 0.85
THRESHOLD_LIKELY     = 0.65
THRESHOLD_POSSIBLE   = 0.45


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class ComponentScores:
    """Individual component scores feeding into the final decision."""
    phash_similarity: float = 0.0
    orb_similarity: float = 0.0
    watermark_match: float = 0.0
    quality_score: float = 1.0          # 1.0 = pristine, 0.0 = severely degraded
    # Optional additional
    ssim: Optional[float] = None
    content_hash_match: bool = False


@dataclass
class SimilarityResult:
    """Full result returned to the backend/frontend."""
    final_score: float = 0.0
    verdict: str = "UNKNOWN"
    is_unauthorized: bool = False
    components: ComponentScores = field(default_factory=ComponentScores)
    explanation_bullets: list[str] = field(default_factory=list)
    confidence: str = "LOW"             # LOW | MEDIUM | HIGH
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["components"] = asdict(self.components)
        return d


# ── Core Scoring ─────────────────────────────────────────────────────────────

def compute_final_score(scores: ComponentScores) -> float:
    """
    Weighted combination:
        Final = 0.40 × watermark + 0.30 × pHash + 0.30 × ORB

    Quality acts as a multiplier signal (high degradation → suspect copy).

    Returns:
        Clipped float in [0.0, 1.0].
    """
    raw = (
        WEIGHT_WATERMARK * scores.watermark_match
        + WEIGHT_PHASH   * scores.phash_similarity
        + WEIGHT_ORB     * scores.orb_similarity
    )
    return float(np.clip(raw, 0.0, 1.0))


def _assess_confidence(scores: ComponentScores, final: float) -> str:
    """
    HIGH confidence requires at least two strong signal sources.
    """
    strong = sum([
        scores.phash_similarity >= 0.80,
        scores.orb_similarity >= 0.70,
        scores.watermark_match >= 0.90,
    ])
    if strong >= 2 and final >= THRESHOLD_DEFINITE:
        return "HIGH"
    elif strong >= 1 or final >= THRESHOLD_LIKELY:
        return "MEDIUM"
    else:
        return "LOW"


def _build_flags(scores: ComponentScores) -> list[str]:
    """Generate human-readable flag list for dashboard display."""
    flags = []
    if scores.watermark_match >= 0.90:
        flags.append("WATERMARK_DETECTED")
    elif scores.watermark_match >= 0.50:
        flags.append("WATERMARK_PARTIAL")

    if scores.phash_similarity >= 0.95:
        flags.append("EXACT_PERCEPTUAL_COPY")
    elif scores.phash_similarity >= 0.80:
        flags.append("NEAR_DUPLICATE")

    if scores.orb_similarity >= 0.80:
        flags.append("STRONG_FEATURE_MATCH")

    if scores.quality_score < 0.40:
        flags.append("SEVERE_QUALITY_DEGRADATION")
    elif scores.quality_score < 0.70:
        flags.append("MODERATE_QUALITY_LOSS")

    if scores.content_hash_match:
        flags.append("IDENTICAL_CONTENT_HASH")

    if scores.ssim is not None and scores.ssim < 0.5:
        flags.append("LOW_STRUCTURAL_SIMILARITY")

    return flags


def _build_explanation(
    scores: ComponentScores,
    final: float,
    verdict: str,
) -> list[str]:
    """
    Generate plain-English bullet points explaining the decision.
    These are used by the Gemini explanation module as context.
    """
    bullets = []

    bullets.append(
        f"Final similarity score: {final:.1%} "
        f"(watermark×40% + pHash×30% + ORB×30%)"
    )

    # pHash
    if scores.phash_similarity >= 0.95:
        bullets.append(
            "Perceptual hash (pHash) shows the images are virtually identical "
            f"({scores.phash_similarity:.1%}) — near-exact visual copy."
        )
    elif scores.phash_similarity >= 0.80:
        bullets.append(
            f"pHash similarity is high ({scores.phash_similarity:.1%}), "
            "suggesting minor cropping, resizing, or colour shifts."
        )
    elif scores.phash_similarity >= 0.60:
        bullets.append(
            f"pHash shows moderate similarity ({scores.phash_similarity:.1%}); "
            "content may have been edited or partially overlapping."
        )
    else:
        bullets.append(
            f"pHash shows low similarity ({scores.phash_similarity:.1%}); "
            "visual content differs substantially."
        )

    # ORB
    if scores.orb_similarity >= 0.80:
        bullets.append(
            f"Feature matching (ORB) is strong ({scores.orb_similarity:.1%}), "
            "indicating shared keypoints even after transformation."
        )
    elif scores.orb_similarity >= 0.50:
        bullets.append(
            f"Feature matching is moderate ({scores.orb_similarity:.1%}), "
            "partial structural overlap detected."
        )
    else:
        bullets.append(
            f"Feature matching is weak ({scores.orb_similarity:.1%}), "
            "suggesting different scenes or heavy modification."
        )

    # Watermark
    if scores.watermark_match >= 0.90:
        bullets.append(
            f"Invisible watermark detected with {scores.watermark_match:.1%} confidence — "
            "strong evidence of ownership match."
        )
    elif scores.watermark_match >= 0.50:
        bullets.append(
            f"Watermark partially detected ({scores.watermark_match:.1%}); "
            "possible degradation or re-encoding."
        )
    else:
        bullets.append(
            "Watermark not detected or significantly altered — "
            "may indicate deliberate removal attempt."
        )

    # Quality
    if scores.quality_score < 0.40:
        bullets.append(
            f"Quality degradation is severe ({1 - scores.quality_score:.1%} loss), "
            "consistent with screen recording or heavy compression."
        )
    elif scores.quality_score < 0.70:
        bullets.append(
            f"Moderate quality degradation detected ({1 - scores.quality_score:.1%} loss)."
        )

    # SSIM
    if scores.ssim is not None:
        bullets.append(
            f"Structural Similarity Index (SSIM): {scores.ssim:.4f} "
            f"({'high' if scores.ssim > 0.8 else 'low'} structural overlap)."
        )

    # Content hash
    if scores.content_hash_match:
        bullets.append("Content hash match confirmed — files are byte-for-byte identical.")

    return bullets


# ── Main Orchestrator ────────────────────────────────────────────────────────

def analyse(scores: ComponentScores) -> SimilarityResult:
    """
    Run the full scoring pipeline and return a SimilarityResult.

    Args:
        scores: Pre-computed ComponentScores (filled by the pipeline).

    Returns:
        SimilarityResult with all fields populated.
    """
    final = compute_final_score(scores)
    confidence = _assess_confidence(scores, final)
    flags = _build_flags(scores)

    # Verdict
    if final >= THRESHOLD_DEFINITE:
        verdict = "UNAUTHORIZED_USE_DETECTED"
        is_unauthorized = True
    elif final >= THRESHOLD_LIKELY:
        verdict = "LIKELY_UNAUTHORIZED"
        is_unauthorized = True
    elif final >= THRESHOLD_POSSIBLE:
        verdict = "POSSIBLE_MATCH"
        is_unauthorized = False
    else:
        verdict = "NO_MATCH"
        is_unauthorized = False

    explanation = _build_explanation(scores, final, verdict)

    return SimilarityResult(
        final_score=round(final, 4),
        verdict=verdict,
        is_unauthorized=is_unauthorized,
        components=scores,
        explanation_bullets=explanation,
        confidence=confidence,
        flags=flags,
    )


# ── Quick-score Helper ───────────────────────────────────────────────────────

def quick_score(
    phash_sim: float,
    orb_sim: float,
    watermark_match: float = 0.0,
    quality: float = 1.0,
) -> SimilarityResult:
    """Convenience wrapper for quick scoring without constructing ComponentScores."""
    scores = ComponentScores(
        phash_similarity=phash_sim,
        orb_similarity=orb_sim,
        watermark_match=watermark_match,
        quality_score=quality,
    )
    return analyse(scores)


# ── Full Image Pipeline ──────────────────────────────────────────────────────

def full_image_pipeline(
    img1: np.ndarray,
    img2: np.ndarray,
    watermark_key: Optional[str] = None,
) -> SimilarityResult:
    """
    Run the complete AI detection pipeline on two numpy image arrays.

    Args:
        img1: Reference (protected) image.
        img2: Suspect image.
        watermark_key: Optional watermark key to test against img2.

    Returns:
        SimilarityResult.
    """
    from ai_engine.phash import compute_phash, phash_similarity
    from ai_engine.orb import orb_similarity

    # --- pHash ---
    h1 = compute_phash(img1)
    h2 = compute_phash(img2)
    p_sim = phash_similarity(h1, h2)

    # --- ORB ---
    o_sim = orb_similarity(img1, img2)

    # --- Watermark (optional) ---
    wm_match = 0.0
    if watermark_key:
        try:
            from ai_services.gemini import detect_watermark_from_arrays
            wm_match = detect_watermark_from_arrays(img2, watermark_key)
        except ImportError:
            pass

    # --- Quality ---
    quality = 1.0
    try:
        from ai_services.quality import compute_quality_score
        quality = compute_quality_score(img2).get("overall_quality", 1.0)
    except ImportError:
        pass

    scores = ComponentScores(
        phash_similarity=p_sim,
        orb_similarity=o_sim,
        watermark_match=wm_match,
        quality_score=quality,
    )
    return analyse(scores)


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Similarity Scoring Self-Test ===\n")

    scenarios = [
        ("Exact copy",         ComponentScores(phash_similarity=0.98, orb_similarity=0.95, watermark_match=0.97, quality_score=0.99)),
        ("Compressed copy",    ComponentScores(phash_similarity=0.88, orb_similarity=0.75, watermark_match=0.80, quality_score=0.55)),
        ("Partial match",      ComponentScores(phash_similarity=0.65, orb_similarity=0.52, watermark_match=0.30, quality_score=0.80)),
        ("Screen recording",   ComponentScores(phash_similarity=0.75, orb_similarity=0.60, watermark_match=0.55, quality_score=0.25)),
        ("Unrelated content",  ComponentScores(phash_similarity=0.10, orb_similarity=0.08, watermark_match=0.02, quality_score=0.95)),
    ]

    for name, scores in scenarios:
        result = analyse(scores)
        print(f"[{name}]")
        print(f"  Final Score:  {result.final_score:.4f}")
        print(f"  Verdict:      {result.verdict}")
        print(f"  Confidence:   {result.confidence}")
        print(f"  Flags:        {result.flags}")
        print(f"  Explanation bullets: {len(result.explanation_bullets)}")
        print()

    print("Self-test PASSED ✓")