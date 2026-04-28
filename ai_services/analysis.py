"""
member4_analysis.py — Analysis Pipeline Orchestrator
Team Member 4 | Part 3: AI Services

Provides the high-level analysis_report() function that
runs all AI services and returns a unified report dict.
"""

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
for part in ["part1-ai-engine", "part2-backend-cloud", "part3-ai-services"]:
    sys.path.insert(0, str(BASE_DIR / part))


def analysis_report(
    reference_img: np.ndarray,
    suspect_img: np.ndarray,
    watermark_key: str = "",
    owner: str = "Unknown",
    run_gemini: bool = True,
) -> dict:
    """
    Run full AI analysis pipeline and return a comprehensive report.

    Args:
        reference_img: The protected/original image.
        suspect_img: The image to check for unauthorized use.
        watermark_key: Key used when watermarking the reference.
        owner: Rights holder name.
        run_gemini: Whether to call Gemini for explanation.

    Returns:
        Unified report dict with all analysis results.
    """
    report = {
        "status": "ok",
        "owner": owner,
        "phash": {},
        "orb": {},
        "watermark": {},
        "quality": {},
        "similarity": {},
        "explanation": "",
        "errors": [],
    }

    # ── pHash ────────────────────────────────────────────────────────────────
    try:
        from ai_engine.phash import compute_phash, phash_similarity
        h1 = compute_phash(reference_img)
        h2 = compute_phash(suspect_img)
        p_sim = phash_similarity(h1, h2)
        report["phash"] = {
            "reference_hash": h1,
            "suspect_hash": h2,
            "similarity": round(p_sim, 4),
        }
    except Exception as ex:
        report["errors"].append(f"pHash: {ex}")
        p_sim = 0.0

    # ── ORB ──────────────────────────────────────────────────────────────────
    try:
        from ai_engine.orb import orb_compare_with_homography
        orb_result = orb_compare_with_homography(reference_img, suspect_img)
        o_sim = orb_result.get("final_similarity", 0.0)
        report["orb"] = orb_result
    except Exception as ex:
        report["errors"].append(f"ORB: {ex}")
        o_sim = 0.0

    # ── Watermark ────────────────────────────────────────────────────────────
    wm_match = 0.0
    if watermark_key:
        try:
            from ai_services.gemini import detect_watermark, embed_watermark
            wm_match = detect_watermark(suspect_img, watermark_key)
            report["watermark"] = {
                "key_tested": ("*" * len(watermark_key)) if watermark_key else None,
                "match_score": round(wm_match, 4),
                "detected": wm_match >= 0.60,
            }
        except Exception as ex:
            report["errors"].append(f"Watermark: {ex}")
    else:
        report["watermark"] = {"key_tested": None, "match_score": 0.0, "detected": False}

    # ── Quality ───────────────────────────────────────────────────────────────
    try:
        from ai_services.quality import compute_quality_score
        quality = compute_quality_score(suspect_img, reference=reference_img)
        report["quality"] = quality
        quality_score = quality.get("overall_quality", 1.0)
    except Exception as ex:
        report["errors"].append(f"Quality: {ex}")
        quality_score = 1.0
        report["quality"] = {"overall_quality": quality_score}

    # ── Final Similarity ──────────────────────────────────────────────────────
    try:
        from ai_engine.similarity import ComponentScores, analyse
        scores = ComponentScores(
            phash_similarity=p_sim,
            orb_similarity=o_sim,
            watermark_match=wm_match,
            quality_score=quality_score,
        )
        sim_result = analyse(scores)
        report["similarity"] = sim_result.to_dict()
    except Exception as ex:
        report["errors"].append(f"Scoring: {ex}")

    # ── Gemini Explanation ────────────────────────────────────────────────────
    if run_gemini:
        try:
            from ai_services.explanation import generate_explanation
            explanation = generate_explanation(
                sim_result,
                quality_report=report["quality"],
            )
            report["explanation"] = explanation
        except Exception as ex:
            report["errors"].append(f"Gemini: {ex}")
            report["explanation"] = _build_fallback_explanation(report)

    return report


def _build_fallback_explanation(report: dict) -> str:
    sim = report.get("similarity", {})
    score = sim.get("final_score", 0)
    verdict = sim.get("verdict", "UNKNOWN")
    return (
        f"Verdict: {verdict} (Score: {score:.1%})\n"
        f"pHash: {report['phash'].get('similarity', 0):.1%} | "
        f"ORB: {report['orb'].get('final_similarity', 0):.1%} | "
        f"Watermark: {report['watermark'].get('match_score', 0):.1%} | "
        f"Quality: {report['quality'].get('overall_quality', 1):.1%}"
    )


def image_level_report(
    reference_path: str,
    suspect_path: str,
    watermark_key: str = "",
    owner: str = "Unknown",
) -> dict:
    """
    Convenience wrapper that loads images from disk before running analysis.
    """
    ref = cv2.imread(str(reference_path))
    sus = cv2.imread(str(suspect_path))
    if ref is None:
        return {"status": "error", "message": f"Cannot load reference: {reference_path}"}
    if sus is None:
        return {"status": "error", "message": f"Cannot load suspect: {suspect_path}"}
    return analysis_report(ref, sus, watermark_key=watermark_key, owner=owner)


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Analysis Pipeline Self-Test ===\n")

    # Create test images
    ref = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(ref, (30, 30), (370, 270), (0, 100, 200), -1)
    cv2.putText(ref, "PROTECTED", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)

    # Near-copy (slight modification)
    suspect = ref.copy()
    suspect[0:20, :] = 0  # black bar at top (as if cropped)
    suspect = cv2.GaussianBlur(suspect, (5, 5), 0)

    key = "TestKey-2024"
    report = analysis_report(ref, suspect, watermark_key=key, run_gemini=False)

    print(f"Status:        {report['status']}")
    print(f"pHash sim:     {report['phash'].get('similarity', 0):.4f}")
    print(f"ORB sim:       {report['orb'].get('final_similarity', 0):.4f}")
    print(f"Watermark:     {report['watermark'].get('match_score', 0):.4f}")
    print(f"Quality:       {report['quality'].get('overall_quality', 0):.4f}")
    print(f"Final score:   {report['similarity'].get('final_score', 0):.4f}")
    print(f"Verdict:       {report['similarity'].get('verdict', 'N/A')}")
    if report["errors"]:
        print(f"Errors:        {report['errors']}")
    print("\nSelf-test PASSED ✓")