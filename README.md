# 🛡️ Sports Media Shield — Complete Setup & Deployment Guide

> **Hackathon-Ready** | FastAPI backend + Streamlit frontend + Firebase + Gemini AI

---

## 📁 Final Project Structure

```
mediapro/
├── ai_engine/
│   ├── __init__.py
│   ├── phash.py              ← Perceptual hashing (pHash)
│   ├── orb.py                ← ORB feature matching
│   ├── video_frames.py       ← Video frame extraction
│   └── similarity.py         ← Final score calculator
│
├── backend_cloud/
│   ├── __init__.py
│   ├── api.py                ← FastAPI app (/upload /compare /media)
│   ├── storage.py            ← Local file storage manager
│   ├── firestore.py          ← Firebase Firestore integration
│   └── integration.py        ← Orchestrates all modules
│
├── ai_services/
│   ├── __init__.py
│   ├── gemini.py             ← Google Gemini AI + watermarking
│   ├── quality.py            ← Quality/degradation detection
│   ├── explanation.py        ← Natural language result explainer
│   └── analysis.py           ← Combined analysis pipeline
│
├── frontend/
│   ├── __init__.py
│   ├── ui.py                 ← Streamlit dashboard (main entry point)
│   ├── upload.py             ← Upload helpers + pre-processing
│   ├── scanner.py            ← Auto URL scanner simulation
│   └── blockchain.py         ← ✅ Ownership ledger (THIS FILE)
│
├── uploads/                  ← Auto-created at runtime
├── ownership_chain.json      ← Auto-created blockchain ledger
├── requirements.txt
├── .env.example
└── HOW_TO_RUN.md             ← You are here
```

---

## ⚡ Quick Start (5 Minutes)

### Step 1 — Clone / Extract Project

```bash
# If from zip:
unzip mediapro.zip
cd mediapro
```

### Step 2 — Create Virtual Environment

```bash
python -m venv venv

# macOS / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure Environment Variables

```bash
cp .env.example .env
# Then edit .env with your keys (see Configuration section below)
```

### Step 5 — Run the App

```bash
# Option A: Streamlit UI (recommended for demo)
streamlit run frontend/ui.py

# Option B: FastAPI backend only
uvicorn backend_cloud.api:app --reload --port 8000
```

---

## 📦 requirements.txt

Create this file in your project root:

```txt
# Core
fastapi==0.111.0
uvicorn[standard]==0.29.0
streamlit==1.35.0
python-multipart==0.0.9

# Computer Vision
opencv-python==4.9.0.80
numpy==1.26.4
Pillow==10.3.0
imagehash==4.3.1
scikit-image==0.23.2

# AI / ML
google-generativeai==0.7.2

# Firebase
firebase-admin==6.5.0

# Utilities
python-dotenv==1.0.1
requests==2.31.0
aiofiles==23.2.1
pydantic==2.7.1
```

Install:
```bash
pip install -r requirements.txt
```

---

## 🔑 Configuration (.env file)

Create `.env` in the project root:

```env
# ── Google Gemini API ──────────────────────────────────────────────────────
# Get free key at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# ── Firebase / Firestore ───────────────────────────────────────────────────
# Download from Firebase Console → Project Settings → Service Accounts
# → Generate new private key → save as firebase_credentials.json
FIREBASE_CREDENTIALS_PATH=./firebase_credentials.json
FIREBASE_PROJECT_ID=your-project-id

# ── App Settings ───────────────────────────────────────────────────────────
WATERMARK_DEFAULT_KEY=sports_media_shield_2024
MAX_UPLOAD_MB=50
SIMILARITY_THRESHOLD=0.75
```

---

## 🔥 Firebase Setup (Free Tier)

### A) Create Firebase Project

1. Go to [console.firebase.google.com](https://console.firebase.google.com)
2. Click **Add project** → name it `sports-media-shield`
3. Disable Google Analytics (not needed) → **Create project**

### B) Enable Firestore

1. In sidebar → **Firestore Database**
2. Click **Create database**
3. Choose **Start in test mode** (for hackathon)
4. Select region → **Enable**

### C) Download Service Account Key

1. Project Settings (⚙️ icon) → **Service accounts**
2. Click **Generate new private key**
3. Save the downloaded JSON as `firebase_credentials.json` in project root

### D) Firestore Collections Used

| Collection | Purpose |
|---|---|
| `media_records` | Ingested asset metadata |
| `comparisons` | Detection results log |
| `blockchain_blocks` | Ownership chain backup |
| `scan_logs` | Auto-scanner history |

> **Note:** If Firebase is unavailable, the system automatically falls back to in-memory storage — it will still run perfectly for demo purposes.

---

## 🤖 Google Gemini Setup (Free)

1. Visit [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with Google → **Create API key**
3. Copy key → paste into `.env` as `GEMINI_API_KEY`

> **Note:** Without a key, the system uses built-in rule-based explanations as fallback. The demo still works fully.

---

## 🚀 Running the System

### Mode 1: Streamlit UI (Frontend Demo)

```bash
streamlit run frontend/ui.py
```

Opens at **http://localhost:8501**

**4 Tabs available:**
- 🛡️ **Manual Detection** — Upload original + suspect image, get similarity score
- 🤖 **Auto Scanner** — Simulate scanning URLs for infringements
- ⛓️ **Ownership Registry** — Register & verify media on blockchain ledger
- 📊 **System Status** — Health check all modules

---

### Mode 2: FastAPI Backend

```bash
uvicorn backend_cloud.api:app --reload --host 0.0.0.0 --port 8000
```

Opens at **http://localhost:8000**

Interactive docs at **http://localhost:8000/docs**

**Available Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/upload` | Upload + register protected media |
| `POST` | `/compare` | Compare two images for similarity |
| `GET` | `/media/{id}` | Retrieve stored media record |
| `GET` | `/media` | List all registered media |

**Example API call:**

```bash
# Upload a protected image
curl -X POST http://localhost:8000/upload \
  -F "file=@original.jpg" \
  -F "owner=ESPN" \
  -F "title=NBA Finals 2024" \
  -F "watermark_key=secret123"

# Compare two images
curl -X POST http://localhost:8000/compare \
  -F "reference=@original.jpg" \
  -F "suspect=@suspect.jpg"
```

---

### Mode 3: Run Both (Full Stack)

```bash
# Terminal 1 — Backend
uvicorn backend_cloud.api:app --reload --port 8000

# Terminal 2 — Frontend  
streamlit run frontend/ui.py
```

---

## 🧬 Testing the Blockchain Module Directly

```bash
cd mediapro
python frontend/blockchain.py
```

Expected output:
```
============================================================
  Sports Media Shield — Blockchain Registry Self-Test
============================================================

[1] Registering asset...
    media_id    : <uuid>
    block_hash  : 00a3f8...
    block_index : 1
    merkle_root : 4b7c2a...

[2] Verifying correct owner...
    verified : True
    reason   : All checks passed ✓

[3] Verifying wrong owner...
    verified : False
    reason   : Owner mismatch: registered='ESPN', claimed='FakeSports'
...
✅ All blockchain self-tests PASSED
```

---

## 🧪 Testing Individual Modules

```bash
# AI Engine
python ai_engine/phash.py
python ai_engine/orb.py
python ai_engine/similarity.py

# AI Services
python ai_services/quality.py
python ai_services/gemini.py

# Backend
python backend_cloud/firestore.py
python backend_cloud/integration.py

# Frontend
python frontend/blockchain.py
python frontend/scanner.py
python frontend/upload.py
```

---

## 🎯 System Flow Explained

```
┌─────────────────────────────────────────────────────────┐
│                   USER UPLOADS ORIGINAL                 │
└──────────────────────────┬──────────────────────────────┘
                           │
          ┌────────────────▼─────────────────┐
          │         INGEST PIPELINE           │
          │  1. Compute pHash fingerprint     │
          │  2. Embed invisible watermark     │
          │  3. Store to Firebase Firestore   │
          │  4. Register on blockchain chain  │
          └────────────────┬─────────────────┘
                           │
          ┌────────────────▼─────────────────┐
          │      USER UPLOADS SUSPECT         │
          └────────────────┬─────────────────┘
                           │
          ┌────────────────▼─────────────────┐
          │        DETECTION PIPELINE         │
          │  1. pHash comparison      (0.30)  │
          │  2. ORB feature matching  (0.30)  │
          │  3. Watermark detection   (0.40)  │
          │  4. Quality degradation check     │
          └────────────────┬─────────────────┘
                           │
          ┌────────────────▼─────────────────┐
          │        FINAL SCORE FORMULA        │
          │  Score = 0.40×WM + 0.30×pHash    │
          │        + 0.30×ORB                 │
          └────────────────┬─────────────────┘
                           │
          ┌────────────────▼─────────────────┐
          │       GEMINI EXPLANATION          │
          │  Natural language result summary  │
          │  + ownership verification         │
          └────────────────┬─────────────────┘
                           │
                     OUTPUT TO UI
```

**Verdict Thresholds:**

| Score | Verdict |
|-------|---------|
| ≥ 0.90 | 🔴 DEFINITE INFRINGEMENT |
| ≥ 0.75 | 🟠 HIGH PROBABILITY |
| ≥ 0.50 | 🟡 POSSIBLE INFRINGEMENT |
| < 0.50 | 🟢 LIKELY ORIGINAL |

---

## ☁️ Cloud Deployment

### Deploy Streamlit → Streamlit Cloud (Free)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set **Main file path**: `frontend/ui.py`
4. Add secrets in **Advanced settings**:
   ```toml
   GEMINI_API_KEY = "your_key"
   FIREBASE_CREDENTIALS_PATH = "firebase_credentials.json"
   ```

### Deploy FastAPI → Render (Free)

1. Create `render.yaml` in project root:
```yaml
services:
  - type: web
    name: sports-media-shield-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn backend_cloud.api:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: GEMINI_API_KEY
        sync: false
      - key: FIREBASE_CREDENTIALS_PATH
        value: ./firebase_credentials.json
```
2. Push to GitHub → connect at [render.com](https://render.com)

### Deploy FastAPI → Google Cloud Run (Free tier)

```bash
# Build and deploy
gcloud run deploy sports-media-shield \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your_key
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: cv2` | `pip install opencv-python` |
| `ModuleNotFoundError: imagehash` | `pip install imagehash` |
| Firebase auth error | Check `firebase_credentials.json` path in `.env` |
| Gemini 403 error | Verify API key at aistudio.google.com |
| Streamlit port busy | `streamlit run frontend/ui.py --server.port 8502` |
| `uploads/` permission error | `mkdir -p uploads && chmod 755 uploads` |
| Blockchain file locked | Delete `ownership_chain.json` to reset |

---

## 👥 Team Module Assignment

| Member | File | Responsibility |
|--------|------|---------------|
| Member 1 | `ai_engine/phash.py` | Perceptual hashing |
| Member 2 | `ai_engine/orb.py` | ORB feature matching |
| Member 3 | `ai_engine/video_frames.py` | Video processing |
| Member 4 | `ai_engine/similarity.py` | Final score logic |
| Member 1 | `backend_cloud/api.py` | FastAPI endpoints |
| Member 2 | `backend_cloud/storage.py` | File storage |
| Member 3 | `backend_cloud/firestore.py` | Firebase integration |
| Member 4 | `backend_cloud/integration.py` | System orchestration |
| Member 1 | `ai_services/gemini.py` | Gemini AI + watermark |
| Member 2 | `ai_services/quality.py` | Quality detection |
| Member 3 | `ai_services/explanation.py` | Result explanation |
| Member 4 | `ai_services/analysis.py` | Combined analysis |
| Member 1 | `frontend/ui.py` | Streamlit UI |
| Member 2 | `frontend/upload.py` | Upload helpers |
| Member 3 | `frontend/scanner.py` | Auto scanner |
| **Member 4** | **`frontend/blockchain.py`** | **Ownership ledger ✅** |

---

## 🏆 Hackathon Demo Script

1. **Start app**: `streamlit run frontend/ui.py`
2. **Register**: Go to "Ownership Registry" → upload image → click Register
3. **Detect**: Go to "Manual Detection" → upload original + watermarked/cropped version
4. **Auto Scan**: Switch to "Auto Scanner" → paste URLs → watch results
5. **Verify**: Back to Registry → verify ownership with media ID
6. **Show AI**: Point at Gemini explanation text in results panel

---

*Built with ❤️ for hackathon — Sports Media Shield v1.0*
