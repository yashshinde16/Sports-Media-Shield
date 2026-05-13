import sys
import os
from pathlib import Path
import json, tempfile
import streamlit as st
import numpy as np
import cv2

if "firebase" in st.secrets and "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    creds = dict(st.secrets["firebase"])
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(creds, tmp)
    tmp.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
for part in ["ai_engine", "backend_cloud", "ai_services", "frontend"]:
    sys.path.insert(0, str(BASE_DIR / part))
sys.path.insert(0, str(BASE_DIR))

# NOTE: UPLOAD_DIR is kept as a temp scratch space for intermediate processing
# only — files here are NOT persisted. All permanent storage goes to Firebase.
UPLOAD_DIR = Path(tempfile.gettempdir()) / "sports_media_shield_scratch"
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

# ── Firebase Storage helpers ──────────────────────────────────────────────────

def _get_firebase_bucket():
    """Return a firebase_admin storage bucket, initialising the app if needed."""
    import firebase_admin
    from firebase_admin import credentials as fb_creds, storage as fb_storage

    if not firebase_admin._apps:
        cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        project_id = os.environ.get("FIREBASE_PROJECT_ID", "")
        bucket_name = os.environ.get("FIREBASE_STORAGE_BUCKET", f"{project_id}.appspot.com")

        if not cred_path:
            raise RuntimeError(
                "GOOGLE_APPLICATION_CREDENTIALS env var not set. "
                "Add firebase credentials to st.secrets['firebase']."
            )

        cred = fb_creds.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})

    return fb_storage.bucket()


def firebase_upload_image(img: np.ndarray, blob_path: str) -> str:
    """
    Encode `img` (BGR numpy array) as JPEG and upload to Firebase Storage.

    Args:
        img:       BGR numpy image.
        blob_path: Destination path inside the bucket, e.g.
                   "uploads/<media_id>_original.jpg"

    Returns:
        Public download URL (with long-lived signed token).
    """
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    bucket = _get_firebase_bucket()
    blob = bucket.blob(blob_path)
    blob.upload_from_string(buf.tobytes(), content_type="image/jpeg")
    blob.make_public()
    return blob.public_url


def firebase_upload_bytes(data: bytes, blob_path: str, content_type: str = "application/octet-stream") -> str:
    """Upload raw bytes to Firebase Storage and return a public URL."""
    bucket = _get_firebase_bucket()
    blob = bucket.blob(blob_path)
    blob.upload_from_string(data, content_type=content_type)
    blob.make_public()
    return blob.public_url


def firebase_download_json(blob_path: str) -> list | dict | None:
    """Download and parse a JSON blob from Firebase Storage. Returns None if missing."""
    try:
        bucket = _get_firebase_bucket()
        blob = bucket.blob(blob_path)
        if not blob.exists():
            return None
        raw = blob.download_as_bytes()
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def firebase_upload_json(data, blob_path: str) -> None:
    """Serialise `data` to JSON and upload to Firebase Storage."""
    raw = json.dumps(data, indent=2).encode("utf-8")
    firebase_upload_bytes(raw, blob_path, content_type="application/json")


def firebase_list_blobs(prefix: str) -> list[dict]:
    """
    List blobs under `prefix` in Firebase Storage.

    Returns a list of dicts with keys: name, size, updated, public_url.
    """
    bucket = _get_firebase_bucket()
    blobs = list(bucket.list_blobs(prefix=prefix))
    result = []
    for b in blobs:
        b.make_public()
        result.append({
            "name":       b.name,
            "size":       b.size or 0,
            "updated":    b.updated.isoformat() if b.updated else "",
            "public_url": b.public_url,
        })
    return result


# ── Blockchain chain persistence via Firebase Storage ─────────────────────────

CHAIN_BLOB_PATH = "blockchain/ownership_chain.json"


def load_chain_from_firebase() -> list:
    """Load the ownership chain JSON from Firebase Storage."""
    data = firebase_download_json(CHAIN_BLOB_PATH)
    return data if isinstance(data, list) else []


def save_chain_to_firebase(chain: list) -> None:
    """Persist the ownership chain JSON to Firebase Storage."""
    firebase_upload_json(chain, CHAIN_BLOB_PATH)


# ── Patched BlockchainRegistry that uses Firebase instead of local disk ───────

def _get_firebase_registry():
    """
    Return a BlockchainRegistry whose _load / _save methods are wired
    to Firebase Storage so the chain survives Streamlit Cloud restarts.
    """
    from blockchain import BlockchainRegistry, Block

    class FirebaseBlockchainRegistry(BlockchainRegistry):
        """Subclass that persists the chain to Firebase Storage."""

        def __init__(self):
            # Skip parent __init__ to avoid touching the local filesystem
            self._chain: list[Block] = []
            self._load()

        def _load(self):
            """Load chain from Firebase Storage → Firestore → genesis."""
            raw = load_chain_from_firebase()
            if raw:
                try:
                    self._chain = [Block.from_dict(b) for b in raw]
                    return
                except Exception as exc:
                    st.warning(f"[Blockchain] Firebase chain parse error: {exc}")

            # Fallback: Firestore
            try:
                from backend_cloud.firestore import _list
                records = _list("blockchain_blocks", limit=1000)
                if records:
                    records = [r for r in records if r.get("media_id") != "genesis" or r.get("index") == 0]
                    records.sort(key=lambda x: x.get("index", 0))
                    self._chain = [Block.from_dict(b) for b in records]
                    self._save()
                    return
            except Exception:
                pass

            # Fresh genesis
            genesis = self._create_genesis()
            self._chain = [genesis]
            self._save()

        def _save(self):
            """Persist chain to Firebase Storage (primary) and Firestore (sync)."""
            try:
                save_chain_to_firebase([b.to_dict() for b in self._chain])
            except Exception as exc:
                st.warning(f"[Blockchain] Could not save chain to Firebase: {exc}")

            # Also mirror to Firestore for query capability
            try:
                from backend_cloud.firestore import _set
                for b in self._chain:
                    _set("blockchain_blocks", str(b.index), b.to_dict())
            except Exception:
                pass

    return FirebaseBlockchainRegistry()


# Session-scoped registry cache
if "firebase_registry" not in st.session_state:
    st.session_state["firebase_registry"] = None


def _get_registry():
    if st.session_state["firebase_registry"] is None:
        st.session_state["firebase_registry"] = _get_firebase_registry()
    return st.session_state["firebase_registry"]


def register_asset_firebase(img, owner, title, watermark_key="", media_id=None):
    """Register an asset using the Firebase-backed registry."""
    from blockchain import register_asset as _orig_register
    import uuid as _uuid

    reg = _get_registry()
    block = reg.register(
        img, owner, title, watermark_key,
        media_id=media_id or str(_uuid.uuid4())
    )
    from datetime import datetime, timezone
    return {
        "media_id":          block.media_id,
        "title":             block.title,
        "owner":             block.owner,
        "block_index":       block.index,
        "block_hash":        block.block_hash,
        "content_hash":      block.content_hash,
        "phash":             block.phash,
        "watermark_secured": block.watermark_key_hash != "0" * 64,
        "registered_at":     datetime.fromtimestamp(block.timestamp, tz=timezone.utc).isoformat(),
        "chain_length":      len(reg._chain),
        "merkle_root":       reg.merkle_root(),
    }


def verify_asset_firebase(media_id, claimed_owner=None, img=None):
    reg = _get_registry()
    result = reg.verify(media_id, claimed_owner, img)
    return result.to_dict()


def list_registry_firebase(skip_genesis=True):
    reg = _get_registry()
    return reg.list_assets(skip_genesis=skip_genesis)


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
        "LIKELY_UNAUTHORIZED":       "#f97316",
        "POSSIBLE_MATCH":            "#eab308",
        "NO_MATCH":                  "#22c55e",
    }.get(verdict, "#6b7280")


# ════════════════════════════════════════════════════════════════════════════
# MODE 1: MANUAL COMPARE
# ════════════════════════════════════════════════════════════════════════════

if "Manual" in mode:
    st.markdown("## 🔍 Manual Asset Comparison")

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

        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(bgr_to_rgb(ref_img), caption="Reference", use_column_width=True)
        with img_col2:
            st.image(bgr_to_rgb(sus_img), caption="Suspect", use_column_width=True)

        if st.button("🚀 Run AI Detection", type="primary"):
            with st.spinner("Running AI analysis pipeline..."):
                try:
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

                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1: render_score_gauge(score, "FINAL SCORE", color)
                    with c2: render_score_gauge(report["phash"].get("similarity",0), "pHASH")
                    with c3: render_score_gauge(report["orb"].get("final_similarity",0), "ORB")
                    with c4: render_score_gauge(report["watermark"].get("match_score",0), "WATERMARK")
                    with c5: render_score_gauge(report["quality"].get("overall_quality",1), "QUALITY")

                    flags = sim.get("flags", [])
                    if flags:
                        st.markdown("**🚩 Flags Detected:**")
                        flag_html = " ".join(f'<span class="flag-chip">{f}</span>' for f in flags)
                        st.markdown(f'<div>{flag_html}</div>', unsafe_allow_html=True)

                    st.divider()

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
                            "phash":     report.get("phash", {}),
                            "orb":       report.get("orb", {}),
                            "watermark": report.get("watermark", {}),
                            "errors":    report.get("errors", []),
                        })

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
# MODE 3: OWNERSHIP REGISTRY  (now backed by Firebase Storage)
# ════════════════════════════════════════════════════════════════════════════

elif "Ownership" in mode:
    st.markdown("## ⛓️ Blockchain-Style Ownership Registry")
    st.caption("🔥 Chain persisted to **Firebase Storage** — survives restarts & redeployments.")

    tab_reg, tab_ver, tab_list = st.tabs(["📝 Register Asset", "✅ Verify Ownership", "📋 View Registry"])

    with tab_reg:
        st.markdown("### Register a New Protected Asset")
        col1, col2 = st.columns([1, 1])
        with col1:
            reg_file = st.file_uploader("Asset Image", type=["jpg","jpeg","png"], key="reg_img")
            reg_title = st.text_input("Asset Title", value="Championship Highlights 2024", key="reg_title")
            reg_owner = st.text_input("Owner Name", value=owner_name, key="reg_owner")
            reg_key = st.text_input(
                "🔑 Watermark Key", value="", type="password",
                placeholder="Enter secret key to embed watermark...",
                help="Remember this key — you need it to verify ownership later",
                key="reg_wm_key"
            )
        with col2:
            if reg_file:
                img = load_uploaded_image(reg_file)
                st.image(bgr_to_rgb(img), caption="Asset Preview", use_column_width=True)

        if reg_file and st.button("⛓️ Register Asset", type="primary"):
            img = load_uploaded_image(reg_file)
            with st.spinner("Registering on blockchain + uploading to Firebase Storage..."):
                # 1. Register on the Firebase-backed blockchain
                result = register_asset_firebase(img, reg_owner, reg_title, reg_key)
                _media_id = result["media_id"]
                _ext = ".jpg"  # always store as JPEG in Firebase

                # 2. Upload original image to Firebase Storage
                try:
                    orig_url = firebase_upload_image(
                        img,
                        f"uploads/{_media_id}_original{_ext}"
                    )
                    result["original_url"] = orig_url
                    st.success("✅ Asset registered + original uploaded to Firebase!")
                    st.markdown(f"🔗 [View original]({orig_url})")
                except Exception as _e:
                    st.success("✅ Asset registered!")
                    st.warning(f"Could not upload original to Firebase Storage: {_e}")

                # 3. Upload watermarked image (if key provided)
                if reg_key:
                    try:
                        from ai_services.gemini import embed_watermark
                        _wm_img = embed_watermark(img, reg_key)
                        wm_url = firebase_upload_image(
                            _wm_img,
                            f"uploads/{_media_id}_watermarked{_ext}"
                        )
                        result["watermarked_url"] = wm_url
                        st.markdown(f"🔗 [View watermarked]({wm_url})")
                    except Exception as _e:
                        st.warning(f"Watermark embed/upload failed: {_e}")

            st.json(result)

    with tab_ver:
        st.markdown("### Verify Asset Ownership")
        ver_id = st.text_input("Media ID")
        ver_owner = st.text_input("Claimed Owner")
        if st.button("🔍 Verify") and ver_id:
            result = verify_asset_firebase(ver_id, ver_owner)
            if result.get("verified"):
                st.success(f"✅ Ownership verified for: {ver_owner}")
            else:
                st.error(f"❌ Verification failed: {result.get('reason')}")
            st.json(result)

    with tab_list:
        st.markdown("### Registered Assets")
        if st.button("🔄 Refresh Registry"):
            # Force reload from Firebase
            st.session_state["firebase_registry"] = None
        registry = list_registry_firebase()
        if registry:
            for entry in registry:
                with st.expander(f"📦 {entry.get('title','Unknown')} — {entry.get('owner','?')}"):
                    # Show Firebase Storage links if available
                    mid = entry.get("media_id", "")
                    orig_blob = f"uploads/{mid}_original.jpg"
                    wm_blob   = f"uploads/{mid}_watermarked.jpg"
                    try:
                        bucket = _get_firebase_bucket()
                        if bucket.blob(orig_blob).exists():
                            b = bucket.blob(orig_blob)
                            b.make_public()
                            st.markdown(f"🔗 [Original image]({b.public_url})")
                        if bucket.blob(wm_blob).exists():
                            b = bucket.blob(wm_blob)
                            b.make_public()
                            st.markdown(f"🔗 [Watermarked image]({b.public_url})")
                    except Exception:
                        pass
                    st.json(entry)
        else:
            st.info("No assets registered yet.")


# ════════════════════════════════════════════════════════════════════════════
# MODE 4: SYSTEM STATUS
# ════════════════════════════════════════════════════════════════════════════

elif "Status" in mode:
    st.markdown("## 📊 System Status")

    if st.button("🔄 Refresh Status"):
        st.rerun()

    # ── Module Health ──────────────────────────────────────────────────────
    st.markdown("### 🔌 Module Health")
    try:
        from backend_cloud.integration import system_status
        status = system_status()
        for module, state in status.items():
            ok = "ok" in str(state).lower() or "configured" in str(state).lower()
            icon = "✅" if ok else "❌"
            st.markdown(f"{icon} **{module.upper()}**: `{state}`")
    except Exception as e:
        st.error(f"Could not load system status: {e}")

    # ── Environment Variables ──────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔑 Environment Variables")
    for var in ["GEMINI_API_KEY", "FIREBASE_PROJECT_ID", "GOOGLE_APPLICATION_CREDENTIALS", "FIREBASE_STORAGE_BUCKET"]:
        val = os.environ.get(var, "")
        if val:
            st.markdown(f"✅ `{var}` — set ({len(val)} chars)")
        else:
            st.markdown(f"❌ `{var}` — **not set**")

    # ── Firebase Storage Files ────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔥 Firebase Storage — Recent Uploads")
    try:
        blobs = firebase_list_blobs("uploads/")
        if blobs:
            # Sort newest first by name (UUIDs are time-based)
            for b in sorted(blobs, key=lambda x: x["updated"], reverse=True)[:10]:
                kb = b["size"] // 1024
                name = b["name"].replace("uploads/", "")
                st.markdown(f"- `{name}` ({kb} KB) — [view]({b['public_url']})")
        else:
            st.info("No files in Firebase Storage yet.")
    except Exception as e:
        st.warning(f"⚠️ Cannot list Firebase Storage files: {e}")

    # ── Blockchain Chain Status ────────────────────────────────────────────
    st.divider()
    st.markdown("### ⛓️ Blockchain Chain (Firebase Storage)")
    try:
        chain_data = firebase_download_json(CHAIN_BLOB_PATH)
        if chain_data:
            st.success(f"✅ Chain loaded — **{len(chain_data)} block(s)** found in `{CHAIN_BLOB_PATH}`")
            if len(chain_data) > 1:
                latest = chain_data[-1]
                st.markdown(f"Latest block: `{latest.get('block_hash','?')[:20]}…`  |  owner: **{latest.get('owner','?')}**")
        else:
            st.info("No chain stored in Firebase yet (will be created on first registration).")
    except Exception as e:
        st.warning(f"⚠️ Could not fetch chain from Firebase: {e}")

    # ── Firestore Records ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### ☁️ Firestore Records")
    try:
        from backend_cloud.firestore import list_media_records
        records = list_media_records(limit=5)
        if records:
            st.json(records[:3])
        else:
            st.info("No Firestore records yet (using in-memory fallback).")
    except ImportError as e:
        st.warning(f"⚠️ Firestore module not found: {e}")
    except Exception:
        st.warning("⚠️ Firestore unavailable — running in offline mode.")

    # ── Storage Note ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📝 Persistence Note")
    st.info(
        "**Streamlit Cloud has no persistent filesystem.** "
        "This app stores all uploaded images and the ownership chain in "
        "**Firebase Storage** (`uploads/` and `blockchain/ownership_chain.json`). "
        "Data survives app restarts, redeployments, and sleep cycles."
    )