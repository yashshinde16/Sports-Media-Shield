"""
member3_video.py — Video Frame Extraction & Sampling Engine
Team Member 3 | Part 1: AI Engine
Handles video input: frame extraction, keyframe detection, thumbnail generation,
and frame-level comparison pipelines.
"""

import os
import math
import tempfile
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np


# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_SAMPLE_RATE = 1       # extract 1 frame per second
MAX_FRAMES = 60               # hard cap to avoid memory issues
THUMBNAIL_SIZE = (320, 180)   # 16:9 thumbnail
SCENE_DIFF_THRESHOLD = 0.35   # mean absolute diff to trigger new scene


# ── Frame Extraction ─────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_frames: int = MAX_FRAMES,
    resize: Optional[tuple] = None,
) -> list[np.ndarray]:
    """
    Extract frames from a video at a given sample rate (frames per second).

    Args:
        video_path: Path to video file (.mp4, .avi, .mov, etc.)
        sample_rate: How many frames to extract per second of video.
        max_frames: Maximum number of frames to extract.
        resize: Optional (width, height) to resize each frame.

    Returns:
        List of BGR numpy arrays.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Interval between captured frames
    frame_interval = max(1, int(fps / sample_rate))

    frames = []
    frame_idx = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            if resize:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            frames.append(frame)

        frame_idx += 1

    cap.release()
    return frames


def frame_generator(
    video_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """
    Memory-efficient generator that yields (frame_number, frame_array).
    Use for large videos where loading all frames at once would OOM.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(fps / sample_rate))
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            yield frame_idx, frame
        frame_idx += 1

    cap.release()


# ── Video Metadata ────────────────────────────────────────────────────────────

def get_video_metadata(video_path: str) -> dict:
    """
    Return metadata dictionary for a video file.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Cannot open: {video_path}"}

    metadata = {
        "path": str(video_path),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_seconds": 0.0,
        "codec": "",
    }

    fps = metadata["fps"] or 1
    metadata["duration_seconds"] = round(metadata["total_frames"] / fps, 2)

    # FourCC codec
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_bytes = bytes([
        fourcc & 0xFF,
        (fourcc >> 8) & 0xFF,
        (fourcc >> 16) & 0xFF,
        (fourcc >> 24) & 0xFF,
    ])
    try:
        metadata["codec"] = codec_bytes.decode("ascii").strip()
    except Exception:
        metadata["codec"] = "UNKNOWN"

    cap.release()
    return metadata


# ── Keyframe Detection ────────────────────────────────────────────────────────

def extract_keyframes(
    video_path: str,
    diff_threshold: float = SCENE_DIFF_THRESHOLD,
    max_keyframes: int = 20,
) -> list[dict]:
    """
    Detect scene-change keyframes using mean absolute difference.

    Args:
        video_path: Path to video.
        diff_threshold: Frame difference (0-1) to trigger a new keyframe.
        max_keyframes: Maximum keyframes to extract.

    Returns:
        List of dicts: {frame_number, timestamp, frame}.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    prev_gray = None
    keyframes = []
    frame_idx = 0

    while cap.isOpened() and len(keyframes) < max_keyframes:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32) / 255.0

        if prev_gray is None:
            # Always capture first frame
            keyframes.append({
                "frame_number": frame_idx,
                "timestamp": round(frame_idx / fps, 3),
                "frame": frame.copy(),
            })
        else:
            diff = np.mean(np.abs(gray_f - prev_gray))
            if diff > diff_threshold:
                keyframes.append({
                    "frame_number": frame_idx,
                    "timestamp": round(frame_idx / fps, 3),
                    "frame": frame.copy(),
                    "diff_score": round(float(diff), 4),
                })

        prev_gray = gray_f
        frame_idx += 1

    cap.release()
    return keyframes


# ── Thumbnail Generation ─────────────────────────────────────────────────────

def generate_thumbnail(video_path: str, timestamp_s: float = 1.0) -> Optional[np.ndarray]:
    """
    Extract a thumbnail frame at the given timestamp (seconds).

    Returns:
        BGR frame resized to THUMBNAIL_SIZE, or None on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    target_frame = int(timestamp_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    return cv2.resize(frame, THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA)


def save_frames(frames: list[np.ndarray], output_dir: str, prefix: str = "frame") -> list[str]:
    """
    Save extracted frames as JPEG files.

    Returns:
        List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        path = os.path.join(output_dir, f"{prefix}_{i:04d}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(path)
    return paths


# ── Video-to-Video Comparison Pipeline ──────────────────────────────────────

def compare_videos_phash(video1_path: str, video2_path: str) -> dict:
    """
    Compare two videos using frame-level pHash similarity.

    Strategy:
      1. Extract frames from both videos (sampled)
      2. For each frame in video1, find best pHash match in video2
      3. Return mean/max similarity + per-frame details

    Returns dict with overall_similarity, frame_matches, verdict.
    """
    # Import here to avoid circular dependency
    from ai_engine.phash import compute_phash, phash_similarity

    frames1 = extract_frames(video1_path, sample_rate=1, max_frames=30)
    frames2 = extract_frames(video2_path, sample_rate=1, max_frames=30)

    if not frames1 or not frames2:
        return {"error": "Could not extract frames", "overall_similarity": 0.0}

    # Pre-compute hashes for video2
    hashes2 = [compute_phash(f) for f in frames2]

    frame_sims = []
    for f1 in frames1:
        h1 = compute_phash(f1)
        best_sim = max(phash_similarity(h1, h2) for h2 in hashes2)
        frame_sims.append(best_sim)

    overall = float(np.mean(frame_sims))
    max_sim = float(np.max(frame_sims))

    return {
        "frames_compared": len(frame_sims),
        "overall_similarity": round(overall, 4),
        "max_frame_similarity": round(max_sim, 4),
        "verdict": _verdict(overall),
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _verdict(sim: float) -> str:
    if sim >= 0.90:
        return "HIGH_SIMILARITY"
    elif sim >= 0.65:
        return "MODERATE_SIMILARITY"
    else:
        return "LOW_SIMILARITY"


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Video Engine Self-Test ===")

    # Create a small synthetic video for testing
    tmp = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    tmp.close()

    out = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    for i in range(50):
        frame = np.full((240, 320, 3), (i * 5 % 255), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        out.write(frame)
    out.release()

    meta = get_video_metadata(tmp.name)
    print(f"Metadata: {meta}")

    frames = extract_frames(tmp.name, sample_rate=2, max_frames=10)
    print(f"Extracted {len(frames)} frames")

    kf = extract_keyframes(tmp.name, diff_threshold=0.05)
    print(f"Keyframes detected: {len(kf)}")

    thumb = generate_thumbnail(tmp.name, timestamp_s=1.0)
    print(f"Thumbnail shape: {thumb.shape if thumb is not None else None}")

    os.unlink(tmp.name)
    print("Self-test PASSED ✓")