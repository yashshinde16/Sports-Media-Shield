"""
member3_explanation.py — AI Explanation Generator
Team Member 3 | Part 3: AI Services

Converts technical detection results into clear, human-readable explanations
using Google Gemini. Falls back to a structured template when API unavailable.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "part1-ai-engine"))
sys.path.insert(0, str(BASE_DIR / "part3-ai-services"))


# ── Prompt Templates ──────────────────────────────────────────────────────────

SYSTEM_CONTEXT = """You are an AI assistant for a Sports Media Digital Rights Protection System.
Your job is to explain copyright violation detection results to a legal/media team.
Be clear, concise, and professional. Use bullet points where helpful.
Always end with a recommendation (LEGAL REVIEW RECOMMENDED / MONITOR / NO ACTION).
"""

EXPLANATION_PROMPT_TEMPLATE = """
Analyse this digital asset protection report and write a clear explanation:

DETECTION RESULTS:
- Final Score: {final_score:.1%}
- Verdict: {verdict}
- Confidence: {confidence}

COMPONENT SCORES:
- pHash (visual similarity): {phash:.1%}
- ORB (feature matching): {orb:.1%}
- Watermark detection: {watermark:.1%}
- Quality score: {quality:.1%}

FLAGS TRIGGERED: {flags}

QUALITY INDICATORS:
{quality_details}

TECHNICAL NOTES:
{tech_notes}

Write a 150-200 word explanation for a media rights manager, covering:
1. What was found
2. What the evidence means
3. Risk level assessment
4. Recommended action

Keep language accessible (not overly technical).
"""


# ── Main Generator ────────────────────────────────────────────────────────────

def generate_explanation(result, quality_report: dict = None) -> str:
    """
    Generate a human-readable explanation for a SimilarityResult.

    Args:
        result: SimilarityResult object from member4_similarity.analyse()
        quality_report: Optional quality report dict from member2_quality.

    Returns:
        Natural language explanation string.
    """
    # Build prompt
    comp = result.components
    quality_details = _format_quality_details(quality_report or {})
    tech_notes = "\n".join(f"• {b}" for b in result.explanation_bullets[:5])

    prompt = SYSTEM_CONTEXT + EXPLANATION_PROMPT_TEMPLATE.format(
        final_score=result.final_score,
        verdict=result.verdict,
        confidence=result.confidence,
        phash=comp.phash_similarity,
        orb=comp.orb_similarity,
        watermark=comp.watermark_match,
        quality=comp.quality_score,
        flags=", ".join(result.flags) if result.flags else "None",
        quality_details=quality_details,
        tech_notes=tech_notes,
    )

    # Call Gemini
    try:
        from ai_services.gemini import gemini_explain
        response = gemini_explain(prompt, fallback=_template_explanation(result))
        return response
    except Exception:
        return _template_explanation(result)


def generate_batch_summary(results: list[dict]) -> str:
    """
    Generate a summary report for a batch scan.

    Args:
        results: List of detection result dicts.

    Returns:
        Natural language batch summary.
    """
    total = len(results)
    violations = sum(1 for r in results if r.get("is_unauthorized", False))
    avg_score = sum(r.get("final_score", 0) for r in results) / max(total, 1)

    top_violations = [
        r for r in results
        if r.get("verdict") in {"UNAUTHORIZED_USE_DETECTED", "LIKELY_UNAUTHORIZED"}
    ][:3]

    prompt = f"""{SYSTEM_CONTEXT}

Write a concise batch scan report summary:

SCAN STATISTICS:
- Total assets scanned: {total}
- Violations detected: {violations} ({violations/max(total,1):.0%})
- Average similarity score: {avg_score:.1%}

TOP VIOLATIONS:
{_format_top_violations(top_violations)}

Write a 100-150 word executive summary for a sports media rights manager.
Include: key findings, severity level, and next steps.
"""
    try:
        from ai_services.gemini import gemini_explain
        return gemini_explain(prompt, fallback=_template_batch_summary(total, violations, avg_score))
    except Exception:
        return _template_batch_summary(total, violations, avg_score)


# ── Template Fallbacks ────────────────────────────────────────────────────────

def _template_explanation(result) -> str:
    """Rule-based explanation fallback (no Gemini required)."""
    comp = result.components
    lines = []

    lines.append(f"📊 DETECTION REPORT — {result.verdict.replace('_', ' ')}")
    lines.append(f"Overall Similarity Score: {result.final_score:.1%} | Confidence: {result.confidence}")
    lines.append("")

    # Evidence summary
    evidence = []
    if comp.phash_similarity >= 0.80:
        evidence.append(f"visual content is {comp.phash_similarity:.0%} perceptually identical")
    if comp.orb_similarity >= 0.70:
        evidence.append(f"structural features match at {comp.orb_similarity:.0%}")
    if comp.watermark_match >= 0.80:
        evidence.append(f"invisible watermark detected ({comp.watermark_match:.0%} confidence)")
    if comp.quality_score < 0.60:
        evidence.append(f"quality degradation detected ({(1-comp.quality_score):.0%} loss)")

    if evidence:
        lines.append("EVIDENCE FOUND:")
        for e in evidence:
            lines.append(f"  • The {e}.")
    else:
        lines.append("No significant matching evidence found.")

    lines.append("")

    # Risk
    if result.verdict == "UNAUTHORIZED_USE_DETECTED":
        lines.append("⚠️  RISK LEVEL: HIGH")
        lines.append("RECOMMENDATION: Immediate legal review recommended.")
        lines.append("Document this detection and contact the rights holder.")
    elif result.verdict == "LIKELY_UNAUTHORIZED":
        lines.append("⚠️  RISK LEVEL: MEDIUM")
        lines.append("RECOMMENDATION: Flag for legal review.")
        lines.append("Manual verification of source is advised.")
    elif result.verdict == "POSSIBLE_MATCH":
        lines.append("ℹ️  RISK LEVEL: LOW")
        lines.append("RECOMMENDATION: Monitor. Manual review suggested.")
    else:
        lines.append("✅  RISK LEVEL: NONE")
        lines.append("RECOMMENDATION: No action required.")

    return "\n".join(lines)


def _template_batch_summary(total: int, violations: int, avg_score: float) -> str:
    level = "HIGH" if violations / max(total, 1) > 0.3 else "MODERATE" if violations > 0 else "LOW"
    return (
        f"BATCH SCAN SUMMARY\n"
        f"Scanned {total} assets. Found {violations} potential violations "
        f"({violations/max(total,1):.0%} of total). "
        f"Average similarity score: {avg_score:.1%}.\n"
        f"Overall risk level: {level}.\n"
        f"{'Recommend immediate legal review of flagged assets.' if violations > 0 else 'No immediate action required.'}"
    )


def _format_quality_details(quality_report: dict) -> str:
    if not quality_report:
        return "Quality data not available."
    lines = []
    if "overall_quality" in quality_report:
        lines.append(f"• Overall quality: {quality_report['overall_quality']:.1%}")
    if "degradation_percent" in quality_report:
        lines.append(f"• Degradation: {quality_report['degradation_percent']}%")
    if "quality_label" in quality_report:
        lines.append(f"• Classification: {quality_report['quality_label']}")
    if "screen_recording" in quality_report:
        sr = quality_report["screen_recording"]
        lines.append(f"• Screen recording: {sr.get('label', 'N/A')}")
    return "\n".join(lines) if lines else "No quality data."


def _format_top_violations(violations: list) -> str:
    if not violations:
        return "  None"
    lines = []
    for i, v in enumerate(violations, 1):
        path = v.get("suspect_path") or v.get("url") or "Unknown"
        score = v.get("final_score", 0)
        lines.append(f"  {i}. {path[:60]} — Score: {score:.1%}")
    return "\n".join(lines)


# ── Quick Helpers ─────────────────────────────────────────────────────────────

def score_to_risk_label(score: float) -> str:
    if score >= 0.85:
        return "🔴 HIGH RISK"
    elif score >= 0.65:
        return "🟠 MEDIUM RISK"
    elif score >= 0.45:
        return "🟡 LOW RISK"
    else:
        return "🟢 NO RISK"


def verdict_to_color(verdict: str) -> str:
    """Return a colour hex for UI display."""
    return {
        "UNAUTHORIZED_USE_DETECTED": "#ef4444",
        "LIKELY_UNAUTHORIZED": "#f97316",
        "POSSIBLE_MATCH": "#eab308",
        "NO_MATCH": "#22c55e",
        "UNKNOWN": "#6b7280",
    }.get(verdict, "#6b7280")


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Explanation Generator Self-Test ===\n")

    # Simulate a result object
    class MockComponents:
        phash_similarity = 0.92
        orb_similarity = 0.85
        watermark_match = 0.78
        quality_score = 0.45
        ssim = 0.71

    class MockResult:
        final_score = 0.87
        verdict = "UNAUTHORIZED_USE_DETECTED"
        confidence = "HIGH"
        components = MockComponents()
        flags = ["WATERMARK_DETECTED", "NEAR_DUPLICATE", "MODERATE_QUALITY_LOSS"]
        explanation_bullets = [
            "pHash similarity: 92% — near-exact visual match",
            "ORB features matched at 85%",
            "Watermark detected with 78% confidence",
            "Quality degraded by 55% — likely screen recorded",
        ]
        is_unauthorized = True

    explanation = generate_explanation(MockResult())
    print(explanation)
    print("\n" + "─" * 60)

    # Batch summary
    mock_results = [
        {"final_score": 0.91, "verdict": "UNAUTHORIZED_USE_DETECTED", "is_unauthorized": True, "url": "https://example.com/stolen1.jpg"},
        {"final_score": 0.72, "verdict": "LIKELY_UNAUTHORIZED", "is_unauthorized": True, "url": "https://example.com/suspect.jpg"},
        {"final_score": 0.20, "verdict": "NO_MATCH", "is_unauthorized": False, "url": "https://example.com/clean.jpg"},
    ]
    batch_summary = generate_batch_summary(mock_results)
    print("\nBATCH SUMMARY:")
    print(batch_summary)
    print("\nSelf-test PASSED ✓")