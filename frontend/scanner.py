"""
member3_auto_scanner.py — Automated Media Scanner
Team Member 3 | Part 4: Frontend + Automation

Simulates a real-world background scanning service by:
  - Accepting a dataset of candidate URLs or local file paths
  - Fetching / loading each asset
  - Running the full detection pipeline (pHash + ORB + watermark + quality)
  - Persisting every result to Firestore (or in-memory fallback)
  - Providing a rich scan-summary report

Can be run as a standalone CLI script or imported and called
programmatically from the Streamlit UI (ui.py).
"""

import sys
import os
import time
import uuid
import json
import hashlib
import threading
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
for _part in ["ai_engine", "backend_cloud", "ai_services", "frontend"]:
    sys.path.insert(0, str(BASE_DIR / _part))

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Lazy imports (graceful degradation) ──────────────────────────────────────
try:
    from ai_engine.phash import compute_phash, phash_similarity
    _PHASH_OK = True
except ImportError:
    _PHASH_OK = False

try:
    from ai_engine.orb import compute_orb_similarity
    _ORB_OK = True
except ImportError:
    _ORB_OK = False

try:
    from ai_engine.similarity import SimilarityResult, ComponentScores, compute_final_score
    _SIM_OK = True
except ImportError:
    _SIM_OK = False

try:
    from ai_services.quality import detect_quality_issues
    _QUALITY_OK = True
except ImportError:
    _QUALITY_OK = False

try:
    import backend_cloud.firestore as fs
    _FS_OK = True
except ImportError:
    _FS_OK = False

try:
    from upload import fetch_image_from_url, preprocess_image
    _UPLOAD_OK = True
except ImportError:
    _UPLOAD_OK = False

try:
    from frontend.blockchain import BlockchainRegistry
    _CHAIN_OK = True
except ImportError:
    _CHAIN_OK = False


# ── Configuration ─────────────────────────────────────────────────────────────
SCAN_TIMEOUT_SEC   = 12          # per-URL network timeout
MAX_WORKERS        = 4           # parallel scan threads
DEFAULT_BATCH_SIZE = 20          # URLs per auto-scan batch
SCAN_INTERVAL_SEC  = 5           # simulated polling interval (demo)

# Simulated sports media asset URLs for demo / hackathon mode
_DEMO_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/240px-PNG_transparency_demonstration_1.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikepegsgrind.jpg/320px-Bikepegsgrind.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Dog_Breeds.jpg/320px-Dog_Breeds.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/320px-Good_Food_Display_-_NCI_Visuals_Online.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/240px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/320px-Cat_November_2010-1a.jpg",
]


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class ScanTarget:
    """A single asset to be scanned."""
    source: str                          # URL or local file path
    source_type: str = "url"             # "url" | "local"
    label: str = ""                      # human-readable tag
    expected_owner: str = "Unknown"      # owner hint for lookup

    def __post_init__(self):
        if not self.label:
            self.label = Path(self.source).name[:40]


@dataclass
class ScanRecord:
    """Result of scanning one asset against the registered reference."""
    scan_id: str       = field(default_factory=lambda: str(uuid.uuid4())[:8])
    target: str        = ""
    label: str         = ""
    timestamp: str     = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str        = "PENDING"       # PENDING | RUNNING | DONE | FAILED
    final_score: float = 0.0
    verdict: str       = "UNKNOWN"
    is_unauthorized: bool = False
    phash_score: float = 0.0
    orb_score: float   = 0.0
    quality_score: float = 1.0
    watermark_score: float = 0.0
    flags: list        = field(default_factory=list)
    error: str         = ""
    processing_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScanBatchReport:
    """Aggregated report for an entire scanning batch."""
    batch_id: str      = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: str    = field(default_factory=lambda: datetime.utcnow().isoformat())
    finished_at: str   = ""
    total: int         = 0
    succeeded: int     = 0
    failed: int        = 0
    unauthorized: int  = 0
    likely_match: int  = 0
    no_match: int      = 0
    avg_score: float   = 0.0
    records: list      = field(default_factory=list)   # list[ScanRecord]

    def summary_dict(self) -> dict:
        d = asdict(self)
        d.pop("records", None)          # keep summary light
        return d


# ── Image Loading ─────────────────────────────────────────────────────────────

def _load_from_url(url: str, timeout: int = SCAN_TIMEOUT_SEC) -> Optional[np.ndarray]:
    """Download an image from a URL and decode it via OpenCV."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MediaProScanner/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as exc:
        return None


def _load_from_path(path: str) -> Optional[np.ndarray]:
    """Load an image from a local file path."""
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _load_target(target: ScanTarget) -> Optional[np.ndarray]:
    """Dispatch to URL or local loader."""
    if target.source_type == "url":
        return _load_from_url(target.source)
    return _load_from_path(target.source)


# ── Core Detection Pipeline ───────────────────────────────────────────────────

def _run_detection(
    candidate_img: np.ndarray,
    reference_img: np.ndarray,
    watermark_key: str = "",
    reference_phash: str = "",
) -> dict:
    """
    Run pHash + ORB + quality checks against a reference image.

    Returns a flat dict of component scores.
    """
    scores = {
        "phash_score": 0.0,
        "orb_score": 0.0,
        "quality_score": 1.0,
        "watermark_score": 0.0,
        "flags": [],
    }

    # --- pHash -----------------------------------------------------------------
    if _PHASH_OK:
        try:
            ref_h = reference_phash or compute_phash(reference_img)
            cand_h = compute_phash(candidate_img)
            scores["phash_score"] = phash_similarity(ref_h, cand_h)
        except Exception:
            scores["flags"].append("PHASH_ERROR")

    # --- ORB -------------------------------------------------------------------
    if _ORB_OK:
        try:
            scores["orb_score"] = compute_orb_similarity(reference_img, candidate_img)
        except Exception:
            scores["flags"].append("ORB_ERROR")

    # --- Quality ---------------------------------------------------------------
    if _QUALITY_OK:
        try:
            q = detect_quality_issues(candidate_img, reference_img)
            scores["quality_score"] = q.get("overall_quality", 1.0)
            if q.get("is_screen_recording"):
                scores["flags"].append("SCREEN_RECORDING_DETECTED")
            if q.get("is_blurry"):
                scores["flags"].append("BLUR_DETECTED")
            if q.get("compression_artifacts"):
                scores["flags"].append("COMPRESSION_ARTIFACTS")
        except Exception:
            scores["flags"].append("QUALITY_ERROR")

    # --- Watermark (key-based pixel-pattern simulation) ------------------------
    if watermark_key:
        try:
            scores["watermark_score"] = _check_watermark(
                candidate_img, watermark_key
            )
            if scores["watermark_score"] > 0.6:
                scores["flags"].append("WATERMARK_FOUND")
        except Exception:
            scores["flags"].append("WATERMARK_ERROR")

    return scores


def _check_watermark(img: np.ndarray, key: str, n_samples: int = 512) -> float:
    """
    Key-based invisible watermark detection (seeded randomness, no pixel storage).
    Mirror of the embed logic: check whether seeded positions carry expected LSB parity.
    """
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    rng  = np.random.default_rng(seed)

    h, w = img.shape[:2]
    if h < 8 or w < 8:
        return 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int32)
    positions = rng.integers(0, h * w, size=n_samples)
    expected  = rng.integers(0, 2, size=n_samples)

    ys, xs = np.divmod(positions, w)
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)

    actual = gray[ys, xs] & 1
    match_rate = float(np.mean(actual == expected))
    # Normalise: 0.5 = random noise → 0.0, 1.0 = perfect match → 1.0
    normalised = max(0.0, (match_rate - 0.5) * 2.0)
    return round(normalised, 4)


def _compute_final(scores: dict) -> tuple[float, str, bool]:
    """Weighted combination → (final_score, verdict, is_unauthorized)."""
    W_WM, W_PH, W_ORB = 0.40, 0.30, 0.30

    final = (
        W_WM  * scores["watermark_score"]
        + W_PH  * scores["phash_score"]
        + W_ORB * scores["orb_score"]
    )
    final = round(min(final, 1.0), 4)

    if final >= 0.85:
        verdict, unauth = "UNAUTHORIZED_USE_DETECTED", True
    elif final >= 0.65:
        verdict, unauth = "LIKELY_UNAUTHORIZED", True
    elif final >= 0.45:
        verdict, unauth = "POSSIBLE_MATCH", False
    else:
        verdict, unauth = "NO_MATCH", False

    return final, verdict, unauth


# ── Single-Asset Scanner ──────────────────────────────────────────────────────

def scan_single(
    target: ScanTarget,
    reference_img: np.ndarray,
    watermark_key: str = "",
    reference_phash: str = "",
    persist: bool = True,
) -> ScanRecord:
    """
    Scan one asset against the reference image.

    Args:
        target:          URL / path + metadata.
        reference_img:   The registered (original) asset as BGR numpy array.
        watermark_key:   Watermark seed key used during registration.
        reference_phash: Pre-computed pHash of reference (speed optimisation).
        persist:         If True, write result to Firestore / fallback store.

    Returns:
        Populated ScanRecord.
    """
    rec = ScanRecord(target=target.source, label=target.label)
    rec.status = "RUNNING"
    t0 = time.perf_counter()

    # 1. Load candidate
    candidate = _load_target(target)
    if candidate is None:
        rec.status = "FAILED"
        rec.error  = "Could not load asset (network error or unsupported format)"
        rec.processing_ms = int((time.perf_counter() - t0) * 1000)
        return rec

    # 2. Resize for consistency
    candidate = cv2.resize(candidate, (512, 512), interpolation=cv2.INTER_AREA)

    # 3. Run detection
    scores = _run_detection(candidate, reference_img, watermark_key, reference_phash)

    # 4. Final score
    final, verdict, unauth = _compute_final(scores)

    rec.final_score      = final
    rec.verdict          = verdict
    rec.is_unauthorized  = unauth
    rec.phash_score      = scores["phash_score"]
    rec.orb_score        = scores["orb_score"]
    rec.quality_score    = scores["quality_score"]
    rec.watermark_score  = scores["watermark_score"]
    rec.flags            = scores["flags"]
    rec.status           = "DONE"
    rec.processing_ms    = int((time.perf_counter() - t0) * 1000)

    # 5. Persist
    if persist and _FS_OK:
        try:
            fs.write_document("scan_results", rec.scan_id, rec.to_dict())
        except Exception:
            pass

    return rec


# ── Batch Scanner ─────────────────────────────────────────────────────────────

def scan_batch(
    targets: list,              # list[ScanTarget | str]
    reference_img: np.ndarray,
    watermark_key: str = "",
    reference_phash: str = "",
    progress_cb: Optional[Callable[[int, int, ScanRecord], None]] = None,
    persist: bool = True,
    parallel: bool = False,
) -> ScanBatchReport:
    """
    Scan a batch of assets.

    Args:
        targets:      List of ScanTarget objects (or plain URL/path strings).
        reference_img: Original registered image.
        watermark_key: Watermark seed used during registration.
        reference_phash: Pre-computed reference pHash.
        progress_cb:  Optional callback(current, total, record) for UI updates.
        persist:      Persist each record to Firestore.
        parallel:     Run scans in parallel threads (faster, but may hit rate limits).

    Returns:
        ScanBatchReport with all records.
    """
    # Normalise input
    norm: list[ScanTarget] = []
    for t in targets:
        if isinstance(t, str):
            st = "url" if t.startswith("http") else "local"
            norm.append(ScanTarget(source=t, source_type=st))
        else:
            norm.append(t)

    report = ScanBatchReport(total=len(norm))

    # Guard: validate reference image before doing anything
    if reference_img is None or not isinstance(reference_img, np.ndarray) or reference_img.size == 0:
        raise ValueError(
            "reference_img is None or invalid. "
            "Ensure the uploaded image was loaded correctly before calling scan_batch()."
        )

    if len(reference_img.shape) < 2:
        raise ValueError("reference_img has an invalid shape — must be at least 2D.")

    ref_resized = cv2.resize(reference_img, (512, 512), interpolation=cv2.INTER_AREA)

    def _process(idx: int, tgt: ScanTarget):
        rec = scan_single(tgt, ref_resized, watermark_key, reference_phash, persist)
        report.records.append(rec)

        if rec.status == "DONE":
            report.succeeded += 1
            if rec.is_unauthorized and rec.final_score >= 0.65:
                report.unauthorized += 1
            elif rec.final_score >= 0.45:
                report.likely_match += 1
            else:
                report.no_match += 1
        else:
            report.failed += 1

        if progress_cb:
            progress_cb(idx + 1, len(norm), rec)

    if parallel and len(norm) > 1:
        threads = []
        semaphore = threading.Semaphore(MAX_WORKERS)

        def _worker(i, t):
            with semaphore:
                _process(i, t)

        for i, tgt in enumerate(norm):
            th = threading.Thread(target=_worker, args=(i, tgt), daemon=True)
            threads.append(th)
            th.start()
        for th in threads:
            th.join()
    else:
        for i, tgt in enumerate(norm):
            _process(i, tgt)

    # Finalise report
    done = [r for r in report.records if r.status == "DONE"]
    report.avg_score   = round(sum(r.final_score for r in done) / max(len(done), 1), 4)
    report.finished_at = datetime.utcnow().isoformat()

    if persist and _FS_OK:
        try:
            fs.write_document("scan_batches", report.batch_id, report.summary_dict())
        except Exception:
            pass

    return report


# ── Demo / Simulation Helpers ─────────────────────────────────────────────────

def get_demo_targets(n: int = 8) -> list:
    """Return a list of demo ScanTarget objects using Wikipedia images."""
    targets = []
    for i, url in enumerate(_DEMO_URLS[:n]):
        targets.append(ScanTarget(
            source=url,
            source_type="url",
            label=f"Demo Asset #{i+1}",
            expected_owner="Demo Owner",
        ))
    return targets


def simulate_live_scan(
    reference_img: np.ndarray,
    watermark_key: str = "demo-key",
    n_assets: int = 6,
    progress_cb: Optional[Callable] = None,
) -> ScanBatchReport:
    """
    Simulate a live automatic scan using demo URLs.

    Each URL is scanned with a brief delay to mimic a background crawler.
    Ideal for hackathon demo mode.
    """
    targets = get_demo_targets(n_assets)
    ref_ph  = ""
    if _PHASH_OK:
        try:
            ref_ph = compute_phash(cv2.resize(reference_img, (512, 512)))
        except Exception:
            pass

    return scan_batch(
        targets,
        reference_img,
        watermark_key=watermark_key,
        reference_phash=ref_ph,
        progress_cb=progress_cb,
        persist=True,
        parallel=False,
    )


def scan_url_list(
    reference_img: np.ndarray,
    urls: list,
    watermark_key: str = "",
    progress_cb: Optional[Callable] = None,
) -> ScanBatchReport:
    """
    Convenience wrapper: scan a plain list of URL strings.

    Automatically converts strings → ScanTarget objects.
    """
    targets = [
        ScanTarget(source=u, source_type="url", label=f"URL #{i+1}")
        for i, u in enumerate(urls)
    ]
    return scan_batch(
        targets,
        reference_img,
        watermark_key=watermark_key,
        progress_cb=progress_cb,
        persist=True,
    )


def scan_directory(
    directory: str,
    reference_img: np.ndarray,
    watermark_key: str = "",
    progress_cb: Optional[Callable] = None,
) -> ScanBatchReport:
    """
    Scan all image files in a local directory.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [
        str(p) for p in Path(directory).iterdir()
        if p.suffix.lower() in exts
    ]
    targets = [
        ScanTarget(source=p, source_type="local", label=Path(p).name)
        for p in sorted(paths)
    ]
    return scan_batch(
        targets,
        reference_img,
        watermark_key=watermark_key,
        progress_cb=progress_cb,
        persist=True,
    )


# ── Report Formatting ─────────────────────────────────────────────────────────

def format_report_text(report: ScanBatchReport) -> str:
    """Generate a human-readable plain-text summary of a scan batch."""
    lines = [
        "=" * 60,
        f"  SCAN BATCH REPORT  |  ID: {report.batch_id}",
        "=" * 60,
        f"  Started  : {report.started_at}",
        f"  Finished : {report.finished_at}",
        f"  Total    : {report.total}",
        f"  ✅ Done  : {report.succeeded}   ❌ Failed: {report.failed}",
        f"  🚨 Unauthorized : {report.unauthorized}",
        f"  ⚠️  Possible Match: {report.likely_match}",
        f"  ✅ No Match      : {report.no_match}",
        f"  Avg Score: {report.avg_score:.3f}",
        "-" * 60,
    ]
    for rec in report.records:
        icon = "🚨" if rec.is_unauthorized else ("⚠️" if rec.final_score > 0.45 else "✅")
        lines.append(
            f"  {icon} [{rec.status:6s}] {rec.label[:35]:<35} "
            f"score={rec.final_score:.3f}  {rec.verdict}"
        )
        if rec.flags:
            lines.append(f"         flags: {', '.join(rec.flags)}")
        if rec.error:
            lines.append(f"         error: {rec.error}")
    lines.append("=" * 60)
    return "\n".join(lines)


def get_violation_records(report: ScanBatchReport) -> list:
    """Return only records flagged as unauthorized use."""
    return [r for r in report.records if r.is_unauthorized]


def export_report_json(report: ScanBatchReport, path: Optional[str] = None) -> str:
    """
    Serialise the full report to JSON.
    If path is given, write to file and return the path; else return the JSON string.
    """
    payload = {
        "batch_id":    report.batch_id,
        "started_at":  report.started_at,
        "finished_at": report.finished_at,
        "summary": {
            "total":        report.total,
            "succeeded":    report.succeeded,
            "failed":       report.failed,
            "unauthorized": report.unauthorized,
            "likely_match": report.likely_match,
            "no_match":     report.no_match,
            "avg_score":    report.avg_score,
        },
        "records": [r.to_dict() for r in report.records],
    }
    json_str = json.dumps(payload, indent=2)
    if path:
        Path(path).write_text(json_str)
        return path
    return json_str


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick CLI demo:
        python auto_scanner.py [reference_image_path] [--key <watermark_key>]
    """
    import argparse

    parser = argparse.ArgumentParser(description="MediaPro Auto Scanner")
    parser.add_argument("reference", nargs="?", default="",
                        help="Path to reference image (leave blank for built-in test image)")
    parser.add_argument("--key", default="sports-media-2024",
                        help="Watermark seed key")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of demo URLs to scan")
    args = parser.parse_args()

    # Load reference
    if args.reference and Path(args.reference).exists():
        ref_img = cv2.imread(args.reference)
    else:
        print("[AutoScanner] No reference image supplied — generating synthetic reference.")
        rng = np.random.default_rng(42)
        ref_img = (rng.random((480, 640, 3)) * 255).astype(np.uint8)

    print(f"\n[AutoScanner] Scanning {args.n} demo assets …\n")

    def _print_progress(cur, total, rec: ScanRecord):
        icon = "🚨" if rec.is_unauthorized else ("⚠️" if rec.final_score > 0.45 else "✅")
        print(f"  [{cur:02d}/{total}] {icon}  {rec.label[:40]:<40} "
              f"score={rec.final_score:.3f}  {rec.verdict}")

    report = simulate_live_scan(
        ref_img,
        watermark_key=args.key,
        n_assets=args.n,
        progress_cb=_print_progress,
    )

    print("\n" + format_report_text(report))

    out_path = str(UPLOAD_DIR / f"scan_report_{report.batch_id}.json")
    export_report_json(report, out_path)
    print(f"\n[AutoScanner] Report saved → {out_path}")