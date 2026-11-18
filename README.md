# Edit Helper

Edit Helper makes still images for video edits. Add a video or audio file to a project, analyze its speech with Gemini, review the suggested scenes, and generate 16:9 images for selected moments. Images are saved in the project's library and can be downloaded for use in an editor.

Run `run_app.bat`, then open `http://localhost:8501`. A Gemini API key is needed for audio analysis and image generation. The older stock search and sourcing pipeline remain under separate tabs.

---

## Setup

Local machine (full pipeline - what the app has always used):
```bash
pip install -r requirements-full.txt
```

Hosted deployment (slim set, covers the Gemini Create-images flow):
```bash
pip install -r requirements.txt
```

The two files are separate on purpose: `requirements.txt` is the slim hosted
set, `requirements-full.txt` is the full local pipeline (stock sourcing,
Whisper transcription, CLIP aesthetic memory, Pinterest). On a local machine,
install the full set or the stock pages stay hidden.

### Hosted deployment (production branch)

`gui/app.py` supports a hosted mode for putting the app on a URL. Set these on
the host (environment variables or `.streamlit/secrets.toml`):

| Setting | Effect |
| --- | --- |
| `EDIT_HELPER_HOSTED=1` | Keys are session-only: the Remember checkboxes are hidden and keys are never read from or written to the local database. Projects, library files and other settings still write to the shared server-side database and folders |
| `EDIT_HELPER_ACCESS_CODE=...` | Shared code visitors must enter before the app loads. REQUIRED in hosted mode - the app refuses to load without it. A simple gate, not per-user accounts |
| `GEMINI_API_KEY` / `PEXELS_API_KEY` | Optional server-side fallback keys, so visitors do not need their own |

In hosted mode the Find stock and Stock pipeline pages are hidden (they need
the full local install), Library copy-to-clipboard is Windows-local only and is
hidden, and uploads are fixed at a 200 MB cap. Local runs keep the upload limit
from `.streamlit/config.toml`.

If no `EDIT_HELPER_ACCESS_CODE` is set on a hosted deployment, the app
refuses to load until the owner configures one (fail closed).

Notes:
- The app is single-user by design: `data/edit_helper.db`, the library and the
  uploads folder are shared by every visitor of the same deployment. There is
  no per-visitor isolation of projects, files or settings.
- The access code is a shared gate, not production authentication: no
  throttling or lockout against brute force, one shared code with no
  per-visitor identity, and it does not isolate visitor data.
- If a key was ever persisted with Remember enabled, remove it from
  `data/edit_helper.db` (or rotate it) before exposing any hosted link. In
  hosted mode the app never reads keys from the database, but the file itself
  is still on the server.
- Streamlit Community Cloud: add the same values under Settings -> Secrets.

---

## 🛠️ Architecture & Implemented Capabilities

### 1. Phoneme & Word-Level Narrative Beat Alignment
- **Module:** `core/audio_aligner.py`
- **Engine:** WhisperX / `faster-whisper`
- Ingests `.mp4` or `.wav` files, transcribes speech with word-level start and end timestamps, and intelligently groups words into narrative beats (2.5s – 6.0s) based on pauses, sentence punctuation, and cadence.

### 2. Multimodal Semantic Embeddings
- **Module:** `core/multimodal_embedder.py`
- **Engine:** OpenCLIP `ViT-B-32` (PyTorch)
- Normalizes text and visual media into unit L2 embedding vectors (512D) for cosine similarity calculation.
- Includes representative video frame sampling for video assets.

### 3. Aesthetic Memory (Local Vector DB)
- **Module:** `core/aesthetic_memory.py`
- **Engine:** `qdrant-client` (embedded local filesystem database in `qdrant_storage/`)
- Ingests mood board images and sample clips from `reference_assets/`.
- Calculates and caches the mathematical **Aesthetic Profile centroid** for style matching.

### 4. LangGraph Autonomous Sourcing & Director Workflow
- **State Definition:** `agent/state.py`
- **Sourcer Node:** `agent/stock_sourcer.py` (queries Pexels Video & Image API with curated royalty-free fallback catalog).
- **Director Node:** `agent/director_evaluator.py`
  - Evaluates candidate visual embeddings against transcript text ($S_{\text{semantic}}$) and Qdrant aesthetic memory ($S_{\text{aesthetic}}$).
  - Enforces the **0.82 threshold** ($S_{\text{composite}} \ge 0.82$).
  - Downloads approved assets directly into `b-roll_assets/`.
- **Workflow StateGraph:** `agent/workflow.py`

### 5. Interactive GUI & Windows NLE Clipboard Integration
- **Dashboard:** `gui/app.py`
- **Clipboard Utility:** `core/clipboard_helper.py`
  - Supports Windows `CF_HDROP` native file drop and `pyperclip` text path.
  - Video editors can click **"📋 Copy to Clipboard"** and press `Ctrl+V` directly in Adobe Premiere Pro, DaVinci Resolve, or Final Cut to paste the asset into their timeline/project bin.

---

## 📂 Project Directory Structure

```
c:\Rajendran\apps\edit_helper\
├── .venv\                           # Python 3.11 virtual environment
├── requirements.txt                 # Slim hosted dependencies (streamlit pinned)
├── run_app.bat                      # 1-click Windows launcher
├── config/
│   └── settings.py                  # Pydantic settings & paths
├── core/
│   ├── audio_aligner.py             # WhisperX phoneme/word alignment
│   ├── multimodal_embedder.py       # OpenCLIP ViT-B-32 encoder
│   ├── aesthetic_memory.py          # Local Qdrant aesthetic memory
│   └── clipboard_helper.py          # Windows CF_HDROP & pyperclip helper
├── agent/
│   ├── state.py                     # LangGraph AgentState & schemas
│   ├── stock_sourcer.py             # Pexels API + curated catalog
│   ├── director_evaluator.py        # 0.82 threshold & multimodal scoring
│   └── workflow.py                  # LangGraph StateGraph pipeline
├── gui/
│   └── app.py                       # Streamlit web dashboard
├── reference_assets/                # Client aesthetic reference folder
├── b-roll_assets/                   # Downloaded approved B-roll assets
└── sample_inputs/                   # Test speech/voiceover audio
```
