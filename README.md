# Edit Helper

Edit Helper makes still images for video edits. Add a video or audio file to a project, analyze its speech with Gemini, review the suggested scenes, and generate 16:9 images for selected moments. Images are saved in the project's library and can be downloaded for use in an editor.

Run `run_app.bat`, then open `http://localhost:8501`. A Gemini API key is needed for audio analysis and image generation. The older stock search and sourcing pipeline remain under separate tabs.

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
├── requirements.txt                 # Pinned dependencies
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
