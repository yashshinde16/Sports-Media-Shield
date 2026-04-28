"""
member1_ui.py — Streamlit Frontend Dashboard
Team Member 1 | Part 4: Frontend + Automation

Full-featured sports media asset protection UI with:
  - Manual upload & compare mode
  - Auto-scanner mode
  - Blockchain ownership registry
  - Result visualization
"""

import sys
import os
from pathlib import Path

import streamlit as st
import numpy as np
import cv2

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
for part in ["ai_engine", "backend_cloud", "ai_services", "frontend"]:
    sys.path.insert(0, str(BASE_DIR / part))
sys.path.insert(0, str(BASE_DIR))

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ Sports Media Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d1b2a 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid #2d2d6b;
    box-shadow: 0 0 40px rgba(99,102,241,0.15);
}
.main-header h1 { color: #e0e7ff; font-size: 2.4rem; font-weight: 800; margin: 0; }
.main-header p { color: #818cf8; font-family: 'DM Mono', monospace; font-size: 0.9rem; margin: 0.5rem 0 0; }

.score-card {
    background: #0f0f23;
    border: 1px solid #2d2d6b;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.score-value { font-size: 2.5rem; font-weight: 800; }
.score-label { color: #818cf8; font-size: 0.8rem; font-family: 'DM Mono', monospace; }

.verdict-UNAUTHORIZED_USE_DETECTED { color: #ef4444; }
.verdict-LIKELY_UNAUTHORIZED { color: #f97316; }
.verdict-POSSIBLE_MATCH { color: #eab308; }
.verdict-NO_MATCH { color: #22c55e; }

.flag-chip {
    display: inline-block;
    background: #1e1e4a;
    border: 1px solid #4f46e5;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.75rem;
    color: #a5b4fc;
    font-family: 'DM Mono', monospace;
    margin: 0.2rem;
}
.explanation-box {
    background: #0a0a1a;
    border-left: 4px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 1.5rem;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #c7d2fe;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛡️ Sports Media Shield</h1>
    <p>AI-Based Digital Asset Protection System · Powered by Gemini + Firebase</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    mode = st.radio("Mode", ["🔍 Manual Compare", "🤖 Auto Scanner", "⛓️ Ownership Registry", "📊 System Status"])
    st.divider()
    st.markdown("**👤 Rights Owner**")
    owner_name = st.text_input("Owner Name", value="Sports Media Corp", label_visibility="collapsed")
    st.divider()
    st.markdown("**🔑 Override Gemini API Key**")
    st.caption("Leave blank to use key from .env file")
    gemini_key_input = st.text_input("Gemini API Key", type="password", placeholder="AIza... (optional)", label_visibility="collapsed")
    if gemini_key_input:
        os.environ["GEMINI_API_KEY"] = gemini_key_input
        st.success("✅ Custom key active")
    else:
        env_key = os.environ.get("GEMINI_API_KEY", "")
        if env_key:
            st.success("✅ Using key from .env")
        else:
            st.warning("⚠️ No Gemini key set")

# Defaults — watermark key and run_gemini are set per-mode below
watermark_key = ""
run_gemini = True

# ── Helper: load image from uploaded file ─────────────────────────────────────
def load_uploaded_image(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    uploaded_file.seek(0)
    return img


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def render_score_gauge(score: float, label: str, color: str = "#6366f1"):
    pct = int(score * 100)
    st.markdown(f"""
    <div class="score-card">
        <div class="score-value" style="color:{color}">{pct}%</div>
        <div class="score-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def verdict_color(verdict: str) -> str:
    return {
        "UNAUTHORIZED_USE_DETECTED": "#ef4444",
        "LIKELY_UNAUTHORIZED": "#f97316",
        "POSSIBLE_MATCH": "#eab308",
        "NO_MATCH": "#22c55e",
    }.get(verdict, "#6b7280")


# ════════════════════════════════════════════════════════════════════════════
# MODE 1: MANUAL COMPARE
# ════════════════════════════════════════════════════════════════════════════

if "Manual" in mode:
    st.markdown("## 🔍 Manual Asset Comparison")

    # ── Detection Settings (inside this mode only) ────────────────────────
    with st.expander("⚙️ Detection Settings", expanded=True):
        det_col1, det_col2 = st.columns(2)
        with det_col1:
            watermark_key = st.text_input(
                "🔑 Watermark Key",
                value="",
                type="password",
                placeholder="Enter key used when registering...",
                help="Must match the key used when the original was registered"
            )
        with det_col2:
            run_gemini = st.toggle("🤖 Gemini AI Explanation", value=True,
                help="Uses Gemini API to explain results in plain English")
            embed_wm = st.checkbox("Embed watermark before comparing", value=True,
                help="Embeds watermark into reference image before running detection")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📁 Reference Asset (Protected)")
        ref_file = st.file_uploader("Upload reference image", type=["jpg","jpeg","png","bmp","webp"], key="ref")

    with col2:
        st.markdown("### 🔎 Suspect Asset")
        sus_file = st.file_uploader("Upload suspect image", type=["jpg","jpeg","png","bmp","webp"], key="sus")

    if ref_file and sus_file:
        ref_img = load_uploaded_image(ref_file)
        sus_img = load_uploaded_image(sus_file)

        # Display images
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(bgr_to_rgb(ref_img), caption="Reference", width='stretch')
        with img_col2:
            st.image(bgr_to_rgb(sus_img), caption="Suspect", width='stretch')

        if st.button("🚀 Run AI Detection", type="primary"):
            with st.spinner("Running AI analysis pipeline..."):
                try:
                    # Embed watermark if requested
                    if embed_wm and watermark_key:
                        from ai_services.gemini import embed_watermark
                        ref_wm = embed_watermark(ref_img, watermark_key)
                    else:
                        ref_wm = ref_img

                    from ai_services.analysis import analysis_report
                    report = analysis_report(
                        ref_wm, sus_img,
                        watermark_key=watermark_key if embed_wm else "",
                        owner=owner_name,
                        run_gemini=run_gemini,
                    )

                    # ── Results ───────────────────────────────────────────────
                    st.divider()
                    sim = report.get("similarity", {})
                    verdict = sim.get("verdict", "UNKNOWN")
                    score = sim.get("final_score", 0)
                    color = verdict_color(verdict)

                    st.markdown(f"""
                    <h2 style="color:{color}; text-align:center; font-size:1.8rem;">
                        {verdict.replace('_', ' ')}
                    </h2>
                    """, unsafe_allow_html=True)

                    # Score cards
                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1: render_score_gauge(score, "FINAL SCORE", color)
                    with c2: render_score_gauge(report["phash"].get("similarity",0), "pHASH")
                    with c3: render_score_gauge(report["orb"].get("final_similarity",0), "ORB")
                    with c4: render_score_gauge(report["watermark"].get("match_score",0), "WATERMARK")
                    with c5: render_score_gauge(report["quality"].get("overall_quality",1), "QUALITY")

                    # Flags
                    flags = sim.get("flags", [])
                    if flags:
                        st.markdown("**🚩 Flags Detected:**")
                        flag_html = " ".join(f'<span class="flag-chip">{f}</span>' for f in flags)
                        st.markdown(f'<div>{flag_html}</div>', unsafe_allow_html=True)

                    st.divider()

                    # Detail tabs
                    tab1, tab2, tab3 = st.tabs(["🤖 AI Explanation", "📊 Detailed Scores", "🔧 Technical"])

                    with tab1:
                        explanation = report.get("explanation", "")
                        if explanation:
                            st.markdown(f'<div class="explanation-box">{explanation.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
                        else:
                            st.info("Enable Gemini in sidebar for AI explanation")

                    with tab2:
                        q = report.get("quality", {})
                        if q:
                            qa, qb = st.columns(2)
                            with qa:
                                st.metric("Sharpness", f"{q.get('blur',{}).get('sharpness_score',0):.1%}")
                                st.metric("Compression Artifacts", f"{q.get('compression_artifacts',{}).get('artifact_score',0):.1%}")
                                st.metric("Noise Level", q.get('noise',{}).get('label','N/A'))
                            with qb:
                                st.metric("Resolution", q.get('resolution',{}).get('label','N/A'))
                                st.metric("Screen Recording", q.get('screen_recording',{}).get('label','N/A'))
                                if q.get('ssim'):
                                    st.metric("SSIM", f"{q['ssim'].get('ssim_score',0):.4f}")

                    with tab3:
                        st.json({
                            "phash": report.get("phash", {}),
                            "orb": report.get("orb", {}),
                            "watermark": report.get("watermark", {}),
                            "errors": report.get("errors", []),
                        })

                    # Store to Firestore
                    try:
                        from backend_cloud.firestore import log_comparison
                        log_comparison("manual_ref", "manual_suspect", sim)
                    except Exception:
                        pass

                except Exception as e:
                    st.error(f"Analysis error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.info("👆 Upload both reference and suspect images to begin analysis")


# ════════════════════════════════════════════════════════════════════════════
# MODE 2: AUTO SCANNER
# ════════════════════════════════════════════════════════════════════════════

elif "Auto" in mode:
    from scanner import scan_url_list, get_demo_targets as _get_demo
    run_auto_scan = scan_url_list
    DEMO_SCAN_URLS = [t.source if hasattr(t, "source") else str(t) for t in _get_demo(8)]
    st.markdown("## 🤖 Automated Asset Scanner")
    st.markdown("Simulate scanning a batch of URLs for unauthorized use of a protected asset.")

    with st.expander("⚙️ Scanner Settings", expanded=True):
        watermark_key = st.text_input(
            "🔑 Watermark Key",
            value="",
            type="password",
            placeholder="Enter key used when registering the asset...",
            help="Must match the key used during original registration",
            key="scanner_wm_key"
        )

    ref_file = st.file_uploader("Upload reference (protected) image", type=["jpg","jpeg","png"], key="auto_ref")

    st.markdown("**Candidate URLs to scan** (one per line):")
    default_urls = "\n".join(DEMO_SCAN_URLS)
    url_text = st.text_area("URLs", value=default_urls, height=150)
    candidate_urls = [u.strip() for u in url_text.splitlines() if u.strip()]

    if ref_file and st.button("🔍 Start Scan", type="primary"):
        ref_img = load_uploaded_image(ref_file)

        if ref_img is None or not isinstance(ref_img, np.ndarray) or ref_img.size == 0:
            st.error("❌ Could not load image. Please upload a valid JPG or PNG.")
            st.stop()

        if not candidate_urls:
            st.warning("⚠️ Please enter at least one URL to scan.")
            st.stop()

        st.image(bgr_to_rgb(ref_img), caption="Reference Asset", width=300)

        with st.spinner(f"Scanning {len(candidate_urls)} URLs..."):
            results = run_auto_scan(ref_img, candidate_urls, watermark_key=watermark_key)

        if not results or not results.records:
            st.warning("No results returned from scanner.")
            st.stop()

        st.success(f"Scan complete! {results.total} URLs analysed.")

        # Summary metrics — results is a ScanBatchReport; use .records for iteration
        violations = sum(1 for r in results.records if r.is_unauthorized)
        avg_score = sum(r.final_score for r in results.records) / max(results.total, 1)
        m1, m2, m3 = st.columns(3)
        m1.metric("URLs Scanned", results.total)
        m2.metric("⚠️ Violations", violations)
        m3.metric("Avg Similarity", f"{avg_score:.1%}")

        st.divider()
        st.markdown("### Results")
        for r in sorted(results.records, key=lambda x: x.final_score, reverse=True):
            score = r.final_score
            verdict = r.verdict
            url = r.target
            color = verdict_color(verdict)

            with st.expander(f"{'🔴' if score > 0.65 else '🟡' if score > 0.45 else '🟢'} {url[:70]} — {score:.1%}"):
                if r.error:
                    st.error(r.error)
                else:
                    a, b, c = st.columns(3)
                    a.metric("Final Score", f"{score:.1%}")
                    b.metric("pHash", f"{r.phash_score:.1%}")
                    c.metric("Verdict", verdict.replace("_", " "))
                    st.markdown(f"**ORB:** {r.orb_score:.1%}")
                    if r.flags:
                        flag_html = " ".join(f'<span class="flag-chip">{f}</span>' for f in r.flags)
                        st.markdown(f'<div>{flag_html}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MODE 3: OWNERSHIP REGISTRY
# ════════════════════════════════════════════════════════════════════════════

elif "Ownership" in mode:
    from blockchain import register_asset, verify_asset, list_registry
    st.markdown("## ⛓️ Blockchain-Style Ownership Registry")

    tab_reg, tab_ver, tab_list = st.tabs(["📝 Register Asset", "✅ Verify Ownership", "📋 View Registry"])

    with tab_reg:
        st.markdown("### Register a New Protected Asset")
        col1, col2 = st.columns([1, 1])
        with col1:
            reg_file = st.file_uploader("Asset Image", type=["jpg","jpeg","png"], key="reg_img")
            reg_title = st.text_input("Asset Title", value="Championship Highlights 2024")
            reg_owner = st.text_input("Owner Name", value=owner_name)
            reg_key = st.text_input("🔑 Watermark Key", value="", type="password", placeholder="Enter secret key to embed watermark...", help="Remember this key — you need it to verify ownership later")
        with col2:
            if reg_file:
                img = load_uploaded_image(reg_file)
                st.image(bgr_to_rgb(img), caption="Asset Preview", width='stretch')

        if reg_file and st.button("⛓️ Register Asset", type="primary"):
            img = load_uploaded_image(reg_file)
            with st.spinner("Registering on blockchain..."):
                result = register_asset(img, reg_owner, reg_title, reg_key)

            # Save files to uploads/
            try:
                import cv2
                from pathlib import Path as _Path
                _UPLOAD_DIR = _Path(__file__).resolve().parent.parent / "uploads"
                _UPLOAD_DIR.mkdir(exist_ok=True)
                _media_id = result["media_id"]
                _ext = _Path(reg_file.name).suffix.lower() or ".jpg"

                # Save original
                _orig_path = _UPLOAD_DIR / f"{_media_id}_original{_ext}"
                cv2.imwrite(str(_orig_path), img)

                # Save watermarked
                if reg_key:
                    try:
                        from ai_services.gemini import embed_watermark
                        _wm_img = embed_watermark(img, reg_key)
                        _wm_path = _UPLOAD_DIR / f"{_media_id}_watermarked{_ext}"
                        cv2.imwrite(str(_wm_path), _wm_img)
                        st.success(f"✅ Asset registered + watermarked!")
                        st.info(f"📁 uploads/{_media_id}_original{_ext}")
                        st.info(f"📁 uploads/{_media_id}_watermarked{_ext}")
                    except Exception as _e:
                        st.success("✅ Asset registered!")
                        st.warning(f"Watermark embed failed: {_e}")
                else:
                    st.success("✅ Asset registered!")
                    st.info(f"📁 uploads/{_media_id}_original{_ext}")

            except Exception as _save_err:
                st.success("✅ Asset registered!")
                st.warning(f"Could not save to uploads/: {_save_err}")

            st.json(result)

    with tab_ver:
        st.markdown("### Verify Asset Ownership")
        ver_id = st.text_input("Media ID")
        ver_owner = st.text_input("Claimed Owner")
        if st.button("🔍 Verify") and ver_id:
            result = verify_asset(ver_id, ver_owner)
            if result.get("verified"):
                st.success(f"✅ Ownership verified for: {ver_owner}")
            else:
                st.error(f"❌ Verification failed: {result.get('reason')}")
            st.json(result)

    with tab_list:
        st.markdown("### Registered Assets")
        registry = list_registry()
        if registry:
            for entry in registry:
                with st.expander(f"📦 {entry.get('title','Unknown')} — {entry.get('owner','?')}"):
                    st.json(entry)
        else:
            st.info("No assets registered yet.")


# ════════════════════════════════════════════════════════════════════════════
# MODE 4: SYSTEM STATUS
# ════════════════════════════════════════════════════════════════════════════

elif "Status" in mode:
    st.markdown("## 📊 System Status")

    if st.button("🔄 Refresh Status"):
        pass

    try:
        from backend_cloud.integration import system_status
        status = system_status()

        for module, state in status.items():
            ok = "ok" in str(state).lower() or "configured" in str(state).lower()
            icon = "✅" if ok else "❌"
            color = "#22c55e" if ok else "#ef4444"
            st.markdown(f"{icon} **{module.upper()}**: `{state}`")
    except Exception as e:
        st.error(f"Status check failed: {e}")

    st.divider()
    st.markdown("### 📁 Recent Uploads")
    try:
        from backend_cloud.storage import list_files
        files = list_files(limit=10)
        if files:
            for f in files:
                st.markdown(f"- `{f['filename']}` ({f['size_bytes']//1024}KB) — `{f['media_id'][:8]}…`")
        else:
            st.info("No files uploaded yet.")
    except Exception as e:
        st.warning(f"Cannot list files: {e}")

    st.divider()
    st.markdown("### ☁️ Firestore Records")
    try:
        from backend_cloud.firestore import list_media_records
        records = list_media_records(limit=5)
        if records:
            st.json(records[:3])
        else:
            st.info("No Firestore records yet (using in-memory fallback).")
    except Exception as e:
        st.warning(f"Firestore unavailable: {e}")