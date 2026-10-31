import sys
from pathlib import Path

# Ensure project root is in sys.path so local packages (config, db, core, agent) resolve
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import streamlit.components.v1 as components
import time
import json
from datetime import datetime
from uuid import uuid4
from dataclasses import asdict
import numpy as np

from config.settings import settings
from db.database import get_db, Project, Message, Resource
from core.audio_aligner import AudioBeatAligner, NarrativeBeat
from core.multimodal_embedder import MultimodalEmbedder
from core.aesthetic_memory import AestheticMemory
from core.clipboard_helper import copy_file_to_windows_clipboard
from core.pinterest_searcher import PinterestSearcher
from core.query_reasoner import QueryReasoner
from core.llm_translator import (
    LLMTranslator, is_non_english, list_ollama_models,
    GEMINI_MODELS, GEMINI_TRANSCRIBE_MODELS,
    AudioTranscription, AudioSegment, TranslatedBeat,
)
from core.image_generator import generate_image
from core.character_sheet import draft_character_sheet, compose_scene_prompt, compose_compact_block
from agent.stock_sourcer import StockSourcer
from agent.director_evaluator import DirectorEvaluator
from agent.workflow import BRollAgentGraph

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Edit Helper",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="auto",
)

# ─── Design System ────────────────────────────────────────────

_THEMES = {
    "light": ":root {"
        "--mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;"
        "--bg: #FAFAF8; --surface: #FFFFFF; --surface-2: #F4F3EF; --surface-3: #E9E7E1;"
        "--border: #E4E1DA; --border-strong: #CFCBC0;"
        "--text: #1B1C1F; --muted: #5B5E65; --faint: #8B8E96;"
        "--accent: #C2530C; --accent-2: #9E4207;"
        "--accent-soft: rgba(194,83,12,0.08); --accent-border: rgba(194,83,12,0.34);"
        "--accent-ink: #FFFFFF; --green: #1C7C4B; --badge-red: #C41E3A; --badge-green: #0E7A55;"
        "--radius-lg: 14px; --radius-md: 10px; --radius-sm: 6px; }",
    "dark": ":root {"
        "--mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;"
        "--bg: #131417; --surface: #1A1C20; --surface-2: #21242A; --surface-3: #2A2E36;"
        "--border: rgba(255,255,255,0.09); --border-strong: rgba(255,255,255,0.17);"
        "--text: #ECEDEF; --muted: #A2A8B1; --faint: #7B818A;"
        "--accent: #FF8A3D; --accent-2: #FFB27A;"
        "--accent-soft: rgba(255,138,61,0.10); --accent-border: rgba(255,138,61,0.36);"
        "--accent-ink: #1F0F03; --green: #34D399; --badge-red: #FF8FA0; --badge-green: #45D99A;"
        "--radius-lg: 14px; --radius-md: 10px; --radius-sm: 6px; }",
}

if "theme" not in st.session_state:
    _qp = st.query_params.get("theme", "light")
    st.session_state.theme = _qp if _qp in ("light", "dark") else "light"

_CSS_FONTS = """@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
"""

_CSS_BASE = """@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');


/* ── Canvas ── */
.stApp, [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text); }
[data-testid="stAppViewContainer"] * { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }
[data-testid="stIconMaterial"], [class*="material-symbols"], [data-testid^="stExpanderIcon"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', sans-serif !important;
}
.block-container { max-width: 1080px; padding: 2.2rem 2.5rem 6rem; }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; display: none; }
[class*="stDeployButton"], [data-testid="stManageAppButton"] { display: none !important; }

h1, h2, h3 { letter-spacing: -0.03em; color: var(--text); }
p, li { color: var(--muted); }
hr { border-color: var(--border); margin: 1.6rem 0; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container { padding: 1.4rem 1.1rem; }

.brand { display: flex; align-items: center; gap: 12px; padding: 4px 2px 18px; }
.brand-mark {
    width: 38px; height: 38px; border-radius: 11px; flex: none;
    background: var(--accent);
    color: var(--accent-ink); font-weight: 800; font-size: 15px;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 16px rgba(255,122,26,0.35);
}
.brand-name { font-weight: 700; font-size: 15.5px; color: var(--text); letter-spacing: -0.02em; }
.brand-tag { font-size: 11.5px; color: var(--muted); margin-top: 1px; }

.side-label {
    font-size: 10.5px; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--muted); margin: 18px 2px 8px;
}

/* Sidebar project list buttons */
[data-testid="stSidebar"] .stButton > button {
    width: 100%; border-radius: 10px; border: 1px solid transparent;
    background: transparent; color: var(--muted);
    text-align: left; justify-content: flex-start;
    padding: 8px 12px; font-size: 13.5px; font-weight: 500;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--surface-2); color: var(--text); border-color: transparent;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: var(--accent-soft); color: var(--accent-2);
    border: 1px solid var(--accent-border);
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: rgba(255,122,26,0.18); color: var(--accent-2);
}

/* ── Buttons (main) ── */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    border-radius: 999px; border: 1px solid var(--border-strong);
    background: var(--surface-2); color: var(--text);
    font-weight: 600; font-size: 13.5px; padding: 9px 18px;
    transition: all 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
    border-color: var(--accent-border); color: var(--accent-2);
    background: var(--surface-3);
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
    background: var(--accent);
    border: none; color: var(--accent-ink);
    box-shadow: none;
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {
    background: var(--accent-2);
    color: var(--accent-ink); transform: none;
    box-shadow: none;
}
.stButton > button:disabled { opacity: 0.45; }

/* ── Navigation (desktop underline bar + mobile page picker) ── */
.st-key-nav_seg { margin-bottom: 26px; }
.st-key-nav_seg [role="radiogroup"] {
    background: transparent !important; border: none !important; border-radius: 0 !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0 !important; gap: 30px !important; width: 100% !important;
    box-shadow: none !important;
}
.st-key-nav_seg [role="radiogroup"] button {
    background: transparent !important; border: none !important; border-radius: 0 !important;
    box-shadow: none !important; margin: 0 !important; padding: 0 !important;
    min-height: 0 !important; height: auto !important;
}
.st-key-nav_seg [role="radiogroup"] button > div {
    background: transparent !important; border-radius: 0 !important; box-shadow: none !important;
    padding: 11px 2px 12px !important; margin: 0 !important;
    border-bottom: 3px solid transparent !important;
}
.st-key-nav_seg [role="radiogroup"] button p {
    color: var(--muted) !important; font-weight: 600 !important; font-size: 15px !important;
    letter-spacing: 0.01em !important;
}
.st-key-nav_seg [role="radiogroup"] button:hover { background: transparent !important; }
.st-key-nav_seg [role="radiogroup"] button:hover p { color: var(--text) !important; }
.st-key-nav_seg [role="radiogroup"] button[data-selected="true"] > div {
    border-bottom-color: var(--accent) !important; background: transparent !important;
}
.st-key-nav_seg [role="radiogroup"] button[data-selected="true"] p {
    color: var(--text) !important;
}
/* mobile: labelled page picker replaces the bar */
.st-key-nav_sel { display: none; }
@media (max-width: 700px) {
    .st-key-nav_seg { display: none; }
    .st-key-nav_sel { display: block; margin-bottom: 22px; }
    .st-key-nav_sel [data-testid="stWidgetLabel"] p {
        font-size: 14px !important; font-weight: 600 !important; color: var(--muted) !important;
    }
}

/* ── Inputs ── */
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div, textarea {
    background: var(--surface-2) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important; color: var(--text) !important;
}
[data-testid="stTextInput"] input:focus, textarea:focus,
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(255,122,26,0.18) !important;
}
[data-testid="stTextInput"] label, [data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label, [data-testid="stFileUploader"] label,
.stTextArea label, [data-testid="stRadio"] label, [data-testid="stCheckbox"] label,
[data-testid="stSlider"] label {
    color: var(--muted) !important; font-size: 12.5px !important; font-weight: 600 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface); border: 1.5px dashed var(--border-strong) !important;
    border-radius: var(--radius-lg); padding: 28px;
}
[data-testid="stFileUploaderDropzone"]:hover, [data-testid="stFileUploaderDropzone"]:focus-within { border-color: var(--accent-border) !important; }
[data-testid="stFileUploaderDropzone"] button {
    border-radius: 999px; border: 1px solid var(--border-strong);
    background: var(--surface-3); color: var(--text); font-weight: 600;
}

/* ── Expanders / cards ── */
[data-testid="stExpander"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius-lg); overflow: hidden;
}
[data-testid="stExpander"] summary { padding: 14px 18px; font-weight: 600; color: var(--text); }
[data-testid="stExpander"] summary:hover { color: var(--accent-2); }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] { padding: 4px 18px 16px; }

[data-testid="stStatusWidget"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg);
}

/* ── Progress ── */
[data-testid="stProgress"] > div > div { background: var(--surface-3); border-radius: 999px; }
[data-testid="stProgress"] > div > div > div {
    background: var(--accent); border-radius: 999px;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius-md); color: var(--muted);
}

/* ── Chat ── */
[data-testid="stChatMessage"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 12px 16px;
}

/* ── Hero & product components ── */
.hero { padding: 34px 0 30px; }
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 11.5px; font-weight: 700; letter-spacing: 0.13em; text-transform: uppercase;
    color: var(--accent-2); background: var(--accent-soft);
    border: 1px solid var(--accent-border); border-radius: 999px; padding: 6px 14px;
    margin-bottom: 20px;
}
.pulse-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 8px var(--accent); }
.hero-title {
    font-family: 'Inter', -apple-system, sans-serif; font-weight: 800;
    font-size: 3.1rem; line-height: 1.06; color: var(--text); margin: 0 0 14px;
    letter-spacing: -0.04em;
}
.hero-title em { font-style: italic; color: var(--accent-2); }
.hero-sub { font-size: 1.06rem; line-height: 1.65; color: var(--muted); max-width: 640px; margin: 0; }

.page-head { padding: 6px 0 22px; border-bottom: 1px solid var(--border); margin-bottom: 24px; }
.page-head h1 {
    font-family: 'Inter', -apple-system, sans-serif; font-weight: 800;
    font-size: 1.9rem; margin: 0; color: var(--text); letter-spacing: -0.035em;
}
.page-head p { color: var(--muted); font-size: 0.95rem; margin: 6px 0 0; }

.section-eyebrow {
    font-size: 11px; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--accent-2); margin: 30px 0 6px;
}
.section-title { font-size: 1.28rem; font-weight: 700; color: var(--text); margin: 0 0 4px; letter-spacing: -0.02em; }
.section-sub { color: var(--muted); font-size: 0.92rem; margin: 0 0 16px; }

.card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 20px 22px; margin-bottom: 14px;
}

/* Step strip */
.steps { display: flex; align-items: center; gap: 10px; margin: 4px 0 26px; flex-wrap: wrap; }
.step {
    display: inline-flex; align-items: center; gap: 9px;
    font-size: 12.5px; font-weight: 600; color: var(--muted);
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 999px; padding: 7px 14px 7px 8px;
}
.step-num {
    width: 22px; height: 22px; border-radius: 50%; flex: none;
    display: flex; align-items: center; justify-content: center;
    background: var(--surface-3); color: var(--muted); font-size: 11px; font-weight: 700;
}
.step.active { color: var(--text); border-color: var(--accent-border); background: var(--accent-soft); }
.step.active .step-num { background: linear-gradient(135deg, #FF9440, #FF5E0E); color: var(--accent-ink); }
.step.done { color: var(--muted); border-color: var(--border); }
.step.done .step-num { background: rgba(52,211,153,0.15); color: var(--green); }
.step-line { width: 22px; height: 1px; background: var(--border-strong); flex: none; }

/* Badges & pills */
.timestamp-badge {
    display: inline-block; background: var(--accent-soft); color: var(--accent-2);
    font-weight: 700; font-size: 12px; padding: 4px 11px;
    border-radius: 999px; border: 1px solid var(--accent-border); margin-right: 6px;
}
.duration-badge {
    display: inline-block; background: var(--surface-3); color: var(--muted);
    font-weight: 600; font-size: 11.5px; padding: 4px 10px;
    border-radius: 999px; border: 1px solid var(--border);
}
.transcript-box {
    background: var(--surface-2); border-left: 3px solid var(--accent);
    padding: 12px 16px; border-radius: 0 var(--radius-md) var(--radius-md) 0;
    margin: 12px 0 14px; font-size: 0.98rem; line-height: 1.55; color: var(--text);
}
.translation-box {
    background: rgba(52,211,153,0.07); border-left: 3px solid var(--green);
    padding: 10px 14px; border-radius: 0 var(--radius-md) var(--radius-md) 0;
    margin-bottom: 12px; font-size: 0.9rem; color: #A7D8C4;
}
.metric-pill {
    display: inline-block; font-size: 11.5px; font-weight: 600; padding: 3px 9px;
    border-radius: 999px; margin-right: 5px;
    background: var(--surface-3); border: 1px solid var(--border); color: var(--muted);
}
.source-badge {
    display: inline-block; font-size: 10.5px; padding: 3px 9px;
    border-radius: 999px; font-weight: 700; letter-spacing: 0.06em; margin-left: 4px;
}
.source-pinterest { background: rgba(230,0,35,0.10); color: var(--badge-red); border: 1px solid rgba(230,0,35,0.38); }
.source-pexels { background: rgba(5,185,125,0.10); color: var(--badge-green); border: 1px solid rgba(5,185,125,0.38); }
.source-fallback { background: var(--surface-3); color: var(--muted); border: 1px solid var(--border); }
.source-gemini { background: var(--accent-soft); color: var(--accent-2); border: 1px solid var(--accent-border); }

/* Feature cards (landing) */
.feat-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 20px; height: 100%;
    transition: border-color 0.15s ease;
}
.feat-card:hover { border-color: var(--accent-border); }
.feat-num {
    font-family: 'Instrument Serif', Georgia, serif !important; font-style: italic;
    font-size: 1.6rem; color: var(--accent-2); line-height: 1;
}
.feat-title { font-weight: 700; font-size: 0.98rem; color: var(--text); margin: 10px 0 6px; }
.feat-desc { font-size: 0.85rem; color: var(--muted); line-height: 1.55; margin: 0; }

/* Project cards (landing) */
.proj-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 18px 20px;
}
.proj-name { font-weight: 700; font-size: 1rem; color: var(--text); }
.proj-meta { font-size: 12px; color: var(--muted); margin-top: 5px; }

/* Gallery caption chip */
.img-chip {
    display: inline-block; background: var(--accent-soft); color: var(--accent-2);
    font-size: 11px; font-weight: 700; border: 1px solid var(--accent-border);
    padding: 3px 10px; border-radius: 999px; margin-bottom: 6px;
}

.stButton > button[kind="primary"] p, .stButton > button[kind="primary"] span,
.stButton > button[kind="primary"] div, .stFormSubmitButton > button[kind="primary"] p {
    color: var(--accent-ink) !important; opacity: 1 !important;
}

.note-box {
    background: var(--surface-2); border: 1px solid var(--border);
    border-left: 2px solid var(--accent); border-radius: 10px;
    padding: 13px 18px; color: var(--text); font-size: 14px; margin: 6px 0 10px;
}

/* Disabled actions */
.stButton > button:disabled, .stFormSubmitButton > button:disabled {
    background: var(--surface-3) !important; color: var(--faint) !important;
    border: 1px solid var(--border) !important; opacity: 1 !important;
}

/* Images */
[data-testid="stImage"] img { border-radius: var(--radius-md); border: 1px solid var(--border); }
/* ----------------------------------------------------------
   v2 - editorial post-production tool direction.
   De-slop pass: no glows, no gradients, no lift-on-hover,
   hierarchy from type + hairlines, mono for time/data.
   ---------------------------------------------------------- */

/* no decorative glow or lift anywhere */
.stButton > button, .stButton > button:hover, .stButton > button:active,
.stDownloadButton > button, .stDownloadButton > button:hover,
.stFormSubmitButton > button, .stFormSubmitButton > button:hover,
.brand-mark, .pulse-dot, .proj-card, .proj-card:hover, .feat-card,
.img-chip, .metric-pill, .step, .step-num, .hero-eyebrow {
    box-shadow: none !important;
}
.stButton > button:hover, .stDownloadButton > button:hover,
.stFormSubmitButton > button:hover, .proj-card:hover {
    transform: none !important;
}

/* step strip: flat editorial tracker, not pills */
.step {
    border: none !important; background: transparent !important;
    padding: 4px 2px !important; font-size: 12.5px !important; font-weight: 500 !important;
}
.step.active { background: transparent !important; color: var(--accent-2) !important; }
.step-num {
    background: transparent !important; border: 1px solid var(--border-strong) !important;
    color: var(--muted) !important; border-radius: 6px !important;
    font-family: var(--mono) !important; font-size: 11px !important; font-weight: 500 !important;
    width: 24px !important; height: 24px !important;
}
.step.active .step-num {
    background: var(--accent) !important; border-color: var(--accent) !important;
    color: var(--accent-ink) !important;
}
.step.done .step-num { border-color: var(--accent-border) !important; color: var(--accent-2) !important; }
.step-line { background: var(--border-strong) !important; height: 1px !important; }

/* hero: plain grotesque, no serif-italic accent */
.hero-title em { font-family: inherit !important; font-style: normal !important; color: var(--accent) !important; }
.hero-eyebrow {
    font-family: var(--mono) !important; font-size: 11px !important; font-weight: 500 !important;
    letter-spacing: 0.05em !important; color: var(--muted) !important;
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 6px !important; padding: 5px 10px !important;
}
.pulse-dot { width: 6px !important; height: 6px !important; border-radius: 2px !important; }

/* quiet section labels */
.section-eyebrow {
    color: var(--muted) !important; font-size: 11px !important;
    letter-spacing: 0.09em !important; font-weight: 600 !important;
}

/* feature list: hairline structure instead of cards */
.feat-card {
    background: transparent !important; border: none !important;
    border-top: 1px solid var(--border-strong) !important; border-radius: 0 !important;
    padding: 16px 16px 4px 0 !important;
}
.feat-num { font-family: var(--mono) !important; font-weight: 500 !important; font-size: 12px !important; color: var(--accent-2) !important; }
.feat-title { font-size: 15px !important; letter-spacing: -0.01em !important; }
.feat-desc { font-size: 13px !important; line-height: 1.6 !important; }

/* project rows: quieter meta */
.proj-card { border-radius: 10px !important; }
.proj-meta { font-family: var(--mono) !important; font-size: 11px !important; letter-spacing: 0 !important; }

/* page head as a workspace title bar */
.page-head { border-bottom: 1px solid var(--border) !important; padding-bottom: 16px !important; margin-bottom: 6px !important; }
.page-head p { font-family: var(--mono) !important; font-size: 12px !important; }

/* chips/badges: mono, squared, bordered (not tinted pills) */
.img-chip {
    font-family: var(--mono) !important; font-size: 11px !important; font-weight: 500 !important;
    letter-spacing: 0 !important; text-transform: none !important;
    border-radius: 6px !important; padding: 4px 9px !important;
    background: var(--surface-2) !important; color: var(--accent-2) !important;
    border: 1px solid var(--accent-border) !important;
}
.timestamp-badge, .duration-badge { font-family: var(--mono) !important; border-radius: 6px !important; letter-spacing: 0 !important; }
.metric-pill { font-family: var(--mono) !important; border-radius: 6px !important; letter-spacing: 0 !important; }
.source-badge { border-radius: 5px !important; font-family: var(--mono) !important; text-transform: none !important; letter-spacing: 0 !important; }
.translation-box { border-left-width: 2px !important; border-radius: 6px !important; }

/* tighter radii overall */
.feat-card, .proj-card, .transcript-box { border-radius: 10px !important; }


/* contrast + native-widget fixes (both themes) */
.translation-box { color: var(--text) !important; }
.translation-box b { color: var(--text) !important; }
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] summary:focus,
[data-testid="stExpander"] summary[aria-expanded="true"] {
    background: transparent !important; color: var(--text) !important;
}
[data-testid="stExpander"] summary:hover { background: var(--surface-2) !important; }
[data-testid="stAlert"] {
    background: var(--surface-2) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 10px !important;
}
[data-testid="stAlert"] * { color: var(--text) !important; }
[data-testid="stAlertContentInfo"], [data-testid="stAlertContentWarning"],
[data-testid="stAlertContentSuccess"], [data-testid="stAlertContentError"] {
    background: transparent !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] p,
[data-testid="stSidebar"] .stButton > button[kind="primary"] span,
[data-testid="stSidebar"] .stButton > button[kind="primary"] div {
    color: var(--accent-2) !important;
}

/* contrast repair: Streamlit's compiled light theme leaks into a few controls in dark mode */
/* 1. disabled primary buttons: descendants were keeping accent-ink (near-black on dark) */
.stButton > button[kind="primary"]:disabled, .stFormSubmitButton > button[kind="primary"]:disabled {
    background: var(--surface-3) !important; border: 1px solid var(--border) !important; opacity: 1 !important;
}
.stButton > button[kind="primary"]:disabled p, .stButton > button[kind="primary"]:disabled span,
.stButton > button[kind="primary"]:disabled div,
.stFormSubmitButton > button[kind="primary"]:disabled p, .stFormSubmitButton > button[kind="primary"]:disabled span,
.stFormSubmitButton > button[kind="primary"]:disabled div {
    color: var(--faint) !important;
}
/* 2. secondary buttons: background could fall back to white with light text on it.
   Buttons with help= are wrapped in stTooltipHoverTarget, so child selectors miss them. */
.stButton button[kind="secondary"], .stDownloadButton button,
.stFormSubmitButton button[kind="secondaryFormSubmit"] {
    background: var(--surface-2) !important; color: var(--text) !important;
    border: 1px solid var(--border-strong) !important; border-radius: 999px !important;
}
.stButton button[kind="secondary"]:hover, .stDownloadButton button:hover,
.stFormSubmitButton button[kind="secondaryFormSubmit"]:hover {
    background: var(--surface-3) !important; border-color: var(--accent-border) !important; color: var(--accent-2) !important;
}
/* 4. number-input steppers + chat submit: light-theme icon fallback */
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {
    background: var(--surface-2) !important; color: var(--muted) !important;
}
[data-testid="stChatInputSubmitButton"] {
    background: var(--surface-2) !important; color: var(--muted) !important;
}

/* audit repair: inputs, selectbox, slider, chips, links - pin Streamlit's compiled
   light-theme fallbacks to theme tokens. Focus indicators preserved. */
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input {
    background: var(--surface-2) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; caret-color: var(--text) !important;
}
[data-testid="stTextInput"] input:hover, [data-testid="stTextArea"] textarea:hover,
[data-testid="stNumberInput"] input:hover { border-color: var(--border-strong) !important; }
[data-testid="stTextInput"] input::placeholder, [data-testid="stTextArea"] textarea::placeholder { font-style: italic !important;
    color: var(--faint) !important; opacity: 1 !important;
}
[data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: var(--accent) !important; box-shadow: 0 0 0 1px var(--accent) !important;
}
[data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"],
[data-testid="stNumberInputContainer"] {
    background: var(--surface-2) !important; border-color: var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stTextInputRootElement"]:focus-within, [data-testid="stTextAreaRootElement"]:focus-within,
[data-testid="stNumberInputContainer"]:focus-within, .react-aria-TextField:focus-within {
    border-color: var(--accent) !important; box-shadow: 0 0 0 1px var(--accent) !important;
}
[data-testid="stSelectbox"] input[role="combobox"], [data-testid="stMultiSelect"] input[role="combobox"] {
    background: var(--surface-2) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 10px !important;
    caret-color: var(--text) !important;
}
[data-testid="stSelectbox"] input[role="combobox"]::placeholder,
[data-testid="stMultiSelect"] input[role="combobox"]::placeholder { color: var(--faint) !important; opacity: 1 !important; }
[data-testid="stSelectbox"] .react-aria-ComboBox button, [data-testid="stMultiSelect"] button {
    background: transparent !important; border: none !important; color: var(--muted) !important;
}
[data-testid="stSelectbox"] .react-aria-ComboBox svg, [data-testid="stMultiSelect"] svg { fill: var(--muted) !important; }
[data-baseweb="select"] > div {
    background: var(--surface-2) !important; border-color: var(--border-strong) !important;
}
[data-baseweb="select"] > div *, [data-baseweb="select"] input { color: var(--text) !important; }
[data-baseweb="select"] svg { fill: var(--muted) !important; }
[data-baseweb="popover"] [data-baseweb="menu"], [role="listbox"] {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
}
[data-baseweb="menu"] li, [role="listbox"] li, [role="option"] { color: var(--text) !important; background: transparent !important; }
[data-baseweb="menu"] li:hover, [role="option"]:hover { background: var(--surface-2) !important; }
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSlider"] [data-testid="stThumbValue"],
[data-testid="stSlider"] span { color: var(--text) !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"],
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] p {
    background: var(--surface-3) !important; color: var(--text) !important;
}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
code {
    background: var(--surface-3) !important; color: var(--text) !important;
    border-radius: 5px !important; padding: 2px 7px !important; font-size: 0.88em !important;
    font-weight: 500 !important;
}
.stMarkdown a, [data-testid="stCaptionContainer"] a { color: var(--accent-2) !important; }

/* audit repair round 2: chat input, avatars, collapse control, widget labels */
[data-testid="stBottom"] { background: var(--bg) !important; }
[data-testid="stChatInput"] {
    background: var(--surface-2) !important; border: 1px solid var(--border-strong) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] > div { background: transparent !important; }
[data-testid="stChatInputSubmitButton"] button, [data-testid="stChatInput"] button[kind="headerNoPadding"] {
    background: var(--accent) !important; color: var(--accent-ink) !important; border: none !important;
}
[data-testid="stChatInputSubmitButton"] svg, [data-testid="stChatInput"] button svg { fill: var(--accent-ink) !important; }
[data-testid="stChatInput"] textarea {
    background: transparent !important; color: var(--text) !important; caret-color: var(--text) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--faint) !important; opacity: 1 !important; }
[data-testid="stChatInput"]:focus-within { border-color: var(--accent) !important; }
[data-testid^="stChatMessageAvatar"] {
    background: var(--surface-3) !important; color: var(--muted) !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important;
}
[data-testid^="stChatMessageAvatar"] svg { fill: var(--muted) !important; color: var(--muted) !important; }
[data-testid="stSidebarCollapseButton"] button, [data-testid="collapsedControl"] button {
    color: var(--muted) !important; background: var(--surface) !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important;
}
[data-testid="stSidebarCollapseButton"] svg, [data-testid="collapsedControl"] svg { fill: var(--muted) !important; }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span { color: var(--muted) !important; }
[data-testid="stTooltipIcon"] { color: var(--muted) !important; }
/* help "?" icon buttons only: they have no kind attribute. Regular buttons with
   help= are also wrapped in stTooltipIcon/stTooltipHoverTarget - do not restyle those. */
[data-testid="stTooltipIcon"] [data-testid="stTooltipHoverTarget"] button:not([kind]),
[data-testid="stTooltipIcon"] button:not([kind]) {
    background: transparent !important; border: none !important; box-shadow: none !important;
    color: var(--muted) !important;
}
[data-testid="stTooltipIcon"] [data-testid="stTooltipHoverTarget"] button:not([kind]):hover,
[data-testid="stTooltipIcon"] button:not([kind]):hover { color: var(--text) !important; background: var(--surface-3) !important; }
[data-testid="stTooltipIcon"] button:not([kind]) svg, [data-testid="stTooltipIcon"] svg {
    stroke: var(--muted) !important; fill: none !important; color: var(--muted) !important;
}

/* audit repair round 3: uploader chip, password eye, steppers, avatar, focus rings */
[data-testid="stFileChip"] {
    background: var(--surface-2) !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important; color: var(--muted) !important;
}
[data-testid="stFileChip"] * { color: var(--muted) !important; }
[data-testid="stFileChipName"] { color: var(--text) !important; font-weight: 600 !important; }
[data-testid="stFileChip"] button, [data-testid="stFileChipDeleteBtn"] button {
    background: transparent !important; border: none !important; color: var(--muted) !important;
}
[data-testid="stFileUploaderDropzone"] button[kind="borderlessIcon"] {
    background: var(--surface-3) !important; border: 1px solid var(--border) !important;
    color: var(--muted) !important; border-radius: 8px !important;
}
[data-testid="stTextInput"] button {
    background: transparent !important; border: none !important; color: var(--muted) !important;
}
[data-testid="stTextInput"] button svg { stroke: var(--muted) !important; }
[data-testid="stNumberInput"] button {
    background: var(--surface-3) !important; color: var(--text) !important; border: none !important;
}
[data-testid="stNumberInput"] [data-testid="stNumberInputContainer"] button svg { fill: var(--text) !important; }
[data-testid="stNumberInput"] [data-testid="stTooltipIcon"] button svg {
    fill: none !important; stroke: var(--muted) !important; color: var(--muted) !important;
}
[data-testid="stChatMessage"] > div:first-child {
    background: var(--surface-3) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebarCollapsedControl"] button, [data-testid="stSidebarCollapsedControl"] svg {
    color: var(--muted) !important; fill: var(--muted) !important;
}
/* consistent theme-accent keyboard focus across all interactive elements */
*:focus-visible {
    outline: 2px solid var(--accent) !important; outline-offset: 1px !important;
    box-shadow: none !important;
}

/* theme radio in sidebar: square segmented look */
[data-testid="stSidebar"] [data-testid="stRadio"] label { font-family: var(--mono) !important; font-size: 12px !important; }

/* v9: toasts, tooltip/popover surfaces, dropzone hover/drag, code tokens */

/* toasts: themed surface + text (compiled light theme leaked white boxes) */
[data-testid="stToastContainer"] > div,
[data-baseweb="toast"],
[data-testid="stToast"] {
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border-strong) !important;
    box-shadow: none !important;
}
[data-testid="stToast"] *, [data-baseweb="toast"] * {
    background: transparent !important;
    color: var(--text) !important;
}

/* tooltip + popover wrapper surfaces */
[data-testid="stTooltipContent"],
[data-baseweb="tooltip"],
[data-baseweb="tooltip"] > div,
[data-baseweb="popover"] > div {
    background: var(--surface-3) !important;
    color: var(--text) !important;
    border: 1px solid var(--border-strong) !important;
    box-shadow: none !important;
}
[data-testid="stTooltipContent"] *, [data-baseweb="tooltip"] * {
    background: transparent !important;
    color: var(--text) !important;
}

/* file uploader: hover / drag / focus-within stay themed */
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stFileUploaderDropzone"]:focus-within {
    background: var(--surface-2) !important;
}
[data-testid="stFileUploaderDropzone"] * { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] button { color: var(--text) !important; }

/* code blocks: base surface + readable token colors in both themes
   (keys accent-2 5.9/9+, strings green 4.7/8.1, numbers accent-2,
    punctuation/base muted/text, all >=4.5 on surface-2) */
[data-testid="stCode"], [data-testid="stCode"] pre, [data-testid="stCode"] pre code {
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stCode"] pre span { color: var(--text) !important; }
[data-testid="stCode"] .hljs-attr, [data-testid="stCode"] .hljs-attribute,
[data-testid="stCode"] .hljs-name, [data-testid="stCode"] .token.attr-name,
[data-testid="stCode"] .token.property { color: var(--accent-2) !important; }
[data-testid="stCode"] .hljs-string, [data-testid="stCode"] .token.string { color: var(--green) !important; }
[data-testid="stCode"] .hljs-number, [data-testid="stCode"] .token.number { color: var(--accent-2) !important; }
[data-testid="stCode"] .hljs-literal, [data-testid="stCode"] .hljs-keyword,
[data-testid="stCode"] .token.boolean, [data-testid="stCode"] .token.keyword,
[data-testid="stCode"] .hljs-punctuation, [data-testid="stCode"] .token.punctuation,
[data-testid="stCode"] .token.operator { color: var(--muted) !important; }

/* v10 */
/* find-stock media type caption beside source badge (was inline #9AA3AF, 2.44:1 light) */
.media-type-tag { color: var(--muted) !important; font-size: 0.8rem; }

/* dropzone drag-active: Streamlit swaps a class that repaints the inner wrapper
   light; force all inner layers transparent so the themed section surface shows */
[data-testid="stFileUploaderDropzone"] > div,
[data-testid="stFileUploaderDropzone"] > div * { background-color: transparent !important; }
[data-testid="stFileUploaderDropzone"] > div button {
    background-color: var(--surface-3) !important;
    border: 1px solid var(--border-strong) !important;
    color: var(--text) !important;
}
/* drag-active inserts a "Drag and drop a file here" div before the limits line;
   Streamlit hides the limits line by color, which our muted-text guard defeats -
   hide it structurally and give the drag instruction full text color */
[data-testid="stFileUploaderDropzone"] > div:not(:only-of-type) ~ div { visibility: hidden !important; }
[data-testid="stFileUploaderDropzone"] > div:not(:only-of-type) span { color: var(--text) !important; font-weight: 600 !important; }
/* quiet back-to-landing text button in the sidebar */
.st-key-back_to_landing .stButton button {
    background: transparent !important; border: none !important; box-shadow: none !important;
    color: var(--muted) !important; font-size: 12.5px !important; font-weight: 500 !important;
    padding: 0 !important; min-height: 0 !important; height: auto !important;
}
.st-key-back_to_landing .stButton button:hover { color: var(--accent-2) !important; background: transparent !important; }
"""

st.markdown(
    "<style>" + _CSS_FONTS + _THEMES[st.session_state.theme] + _CSS_BASE + "</style>",
    unsafe_allow_html=True,
)


# ─── Database ─────────────────────────────────────────────────
db = get_db()

# ─── Session State Init ──────────────────────────────────────
defaults = {
    "active_project_id": None,
    "broll_results": [],
    "trace_logs": [],
    "is_processing": False,
    "pinterest_results": [],
    "show_new_project": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def get_active_project():
    pid = st.session_state.active_project_id
    if pid:
        return db.get_project(pid)
    return None


# ─── Landing (entry view) ─────────────────────────────────────
# Ported from the approved Toolcraft concept (v2): fixed dark palette, orange
# accent, waveform hero where pictures develop at key moments. "Try it" enters
# the real app; the sidebar gets a quiet way back.

_LND_ORANGE = "#FF8A3D"

_LANDING_CSS = """
<style>
/* landing-only chrome: full-bleed dark page, no sidebar */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stHeader"] { display: none !important; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #0c0b0d !important; }
[data-testid="stMainBlockContainer"] { max-width: 1180px !important; padding-top: 10px !important; }
.lnd { color: #f3efe9; font-family: 'Inter', sans-serif; }
.lnd-brand { font-weight: 700; font-size: 16px; padding: 12px 0 4px; }
.lnd-brand b { color: #FF8A3D; font-weight: 700; }
.lnd-eyebrow { color: #FF8A3D; font-size: 13px; letter-spacing: 0.14em; text-transform: uppercase; margin: 26px 0 14px; }
.lnd h1 { font-size: clamp(32px, 5.4vw, 66px); line-height: 1.02; letter-spacing: -0.04em; font-weight: 700; max-width: 860px; margin: 0; color: #f3efe9; }
.lnd-sub { max-width: 580px; margin: 20px 0 6px; font-size: 17px; line-height: 1.5; color: #bdb6ad; }
.lnd-cta-note { color: #9a9389; font-size: 14px; margin-top: -6px; }
.lnd-steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 24px; padding: 26px 0 20px; }
.lnd-step { border-top: 1px solid rgba(255,255,255,0.18); padding-top: 14px; }
.lnd-step-t { font-weight: 600; font-size: 17px; color: #f3efe9; }
.lnd-step-d { color: #aaa398; margin-top: 4px; line-height: 1.5; font-size: 15px; }
.st-key-landing_try .stButton button {
    background: #FF8A3D !important; color: #1a0e05 !important; border: none !important;
    border-radius: 999px !important; padding: 13px 30px !important; font-weight: 700 !important;
    font-size: 16px !important; min-height: 0 !important; height: auto !important;
}
.st-key-landing_try .stButton button:hover { background: #ff9a57 !important; }
.st-key-landing_try .stButton button p, .st-key-landing_try .stButton button span {
    color: #1a0e05 !important; font-weight: 700 !important; font-size: 16px !important;
}
.st-key-landing_try .stButton button:focus-visible { outline: 2px solid #ffd2b0 !important; outline-offset: 2px !important; }
</style>
"""

# Hero: vanilla-JS port of the concept's canvas (React wrapper removed, draw
# logic unchanged). Honors prefers-reduced-motion with the finished state.
_LANDING_HERO_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0;background:#0c0b0d;}
canvas{width:100%;height:440px;display:block;}
</style></head><body>
<canvas id="hero" role="img" aria-label="Animation: a talk's waveform where five pictures develop at its key moments"></canvas>
<script>
const ORANGE = "#FF8A3D";
const KEYS = [0.1, 0.3, 0.5, 0.7, 0.9];
const LABELS = ["the opening idea", "the turn", "the story", "the lesson", "the close"];
const TIMES = ["0:08", "0:31", "0:57", "1:24", "1:49"];
function rnd(n){ const x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); }
function scene(ctx, i, w, h){
  const pal = [["#2a1a3a","#ff8a3d","#ffd7a8"],["#0d1b26","#1f6f78","#8fd3c8"],["#2b1410","#e0522a","#ffb870"],["#101a12","#4c8a4a","#e6f0b0"],["#1c1a14","#ffb347","#fff1d6"]][i];
  const g = ctx.createLinearGradient(0,0,0,h); g.addColorStop(0,pal[0]); g.addColorStop(0.7,pal[1]); g.addColorStop(1,pal[2]);
  ctx.fillStyle = g; ctx.fillRect(0,0,w,h);
  ctx.fillStyle = pal[2]; ctx.beginPath(); ctx.arc(w*0.68, h*0.34, h*0.12, 0, 7); ctx.fill();
  const dark = a => `rgba(8,6,10,${a})`;
  if (i === 0){
    for (let k=0;k<3;k++){ ctx.fillStyle=dark(0.4+k*0.25); ctx.beginPath(); ctx.moveTo(0,h);
      for (let x=0;x<=w;x+=5) ctx.lineTo(x, h*(0.6+k*0.1) - Math.abs(Math.sin(x*0.03+k*2))*h*(0.28-k*0.07));
      ctx.lineTo(w,h); ctx.fill(); }
  } else if (i === 1){
    for (let r=0;r<14;r++){ const y=h*0.55+r*(h*0.035); ctx.strokeStyle=`rgba(255,255,255,${0.08+r*0.02})`; ctx.lineWidth=1.5;
      ctx.beginPath(); for (let x=0;x<=w;x+=4) ctx.lineTo(x, y+Math.sin(x*0.06+r)*(2+r*0.4)); ctx.stroke(); }
    ctx.fillStyle=dark(0.5); ctx.fillRect(0,h*0.54,w,2);
  } else if (i === 2){
    ctx.fillStyle=dark(0.85);
    for (let x=0;x<w;x+=14){ const bh=h*(0.18+rnd(x+5)*0.4); ctx.fillRect(x,h-bh,12,bh);
      ctx.fillStyle="rgba(255,200,120,0.8)";
      for (let y=h-bh+6;y<h-4;y+=10) if (rnd(x*y)>0.6) ctx.fillRect(x+3,y,3,4);
      ctx.fillStyle=dark(0.85); }
  } else if (i === 3){
    for (let x=-6;x<w+6;x+=11){ const th=h*(0.3+rnd(x+2)*0.35); ctx.fillStyle=dark(0.55+rnd(x)*0.35);
      ctx.beginPath(); ctx.moveTo(x,h); ctx.lineTo(x+5,h-th); ctx.lineTo(x+11,h); ctx.fill(); }
  } else {
    ctx.fillStyle=dark(0.9); ctx.fillRect(0,h*0.78,w,h*0.22);
    for (const [fx,fh] of [[0.42,0.34],[0.52,0.2]]){ ctx.beginPath();
      ctx.arc(w*fx, h*(0.78-fh)-6, 6, 0, 7); ctx.fill();
      ctx.fillRect(w*fx-5, h*(0.78-fh), 10, h*fh); }
  }
}
(function(){
  const cv = document.getElementById("hero");
  const ctx = cv.getContext("2d");
  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const fw = 220, fh = 140;
  const tiles = [0,1,2,3,4].map(i => { const c=document.createElement("canvas"); c.width=fw; c.height=fh;
    const x=c.getContext("2d"); if (x) scene(x,i,fw,fh); return c; });
  const t0 = performance.now(); const CYCLE = 13000, RUN = 6500;
  function draw(now){
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const W = cv.clientWidth, H = cv.clientHeight;
    if (cv.width !== Math.round(W*dpr)){ cv.width = Math.round(W*dpr); cv.height = Math.round(H*dpr); }
    ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,W,H);
    const t = (now - t0 + 0.4*RUN) % CYCLE;
    const prog = still ? 1.1 : Math.min(1.1, t/RUN);
    const fade = still ? 1 : (t > CYCLE-700 ? Math.max(0,(CYCLE-t)/700) : 1);
    const small = W < 640;
    const wy = H * (small ? 0.78 : 0.72);
    const pad = 24; const bars = Math.floor((W - pad*2) / 6); const cols = 2;
    for (let b=0;b<bars;b++){
      const u=b/bars; let a=0.2+0.8*Math.pow(rnd(b*1.7),1.5);
      for (const k of KEYS) a += Math.max(0, 0.5 - Math.abs(u-k)*6);
      const bh = Math.min(1.1,a) * H * 0.12;
      ctx.fillStyle = u < prog ? ORANGE : "rgba(255,255,255,0.24)";
      ctx.fillRect(pad + b*6, wy - bh, 3, bh*2);
    }
    if (prog < 1.02){ ctx.fillStyle = ORANGE;
      ctx.fillRect(pad + Math.min(prog,1)*(bars*6) - 1, wy - H*0.19, 2, H*0.38); }
    ctx.globalAlpha = fade;
    KEYS.forEach((k,i) => {
      if (small && i > 3) return;
      const dev = Math.max(0, Math.min(1, (prog-k)/0.12));
      const cx = small ? pad + ((i%cols)+0.5)*((W-pad*2)/cols) : pad + k*(bars*6);
      const colW = (W-pad*2)/cols;
      const sw = small ? colW-18 : 150, sh = small ? Math.round(sw*0.64) : 96;
      const sx = cx - sw/2, sy = small ? 30 + Math.floor(i/cols)*(sh+34) : wy - H*0.26 - sh - (i%2)*20;
      if (!small){ ctx.strokeStyle="rgba(255,255,255,0.18)"; ctx.beginPath();
        ctx.moveTo(cx, sy+sh); ctx.lineTo(cx, wy-H*0.12); ctx.stroke(); }
      ctx.fillStyle="rgba(255,255,255,0.06)"; ctx.fillRect(sx,sy,sw,sh);
      const cell=6;
      for (let yy=0;yy<sh;yy+=cell) for (let xx=0;xx<sw;xx+=cell)
        if (rnd(xx*3+yy*7+i*100) < dev)
          ctx.drawImage(tiles[i], (xx/sw)*fw, (yy/sh)*fh, (cell/sw)*fw, (cell/sh)*fh, sx+xx, sy+yy, cell, cell);
      ctx.strokeStyle = dev>0.99 ? ORANGE : "rgba(255,255,255,0.25)";
      ctx.strokeRect(sx+0.5, sy+0.5, sw-1, sh-1);
      ctx.fillStyle="rgba(255,255,255,0.75)"; ctx.font="11px ui-monospace, Menlo, monospace";
      ctx.fillText(small ? `${TIMES[i]}  idea ${i+1}` : `${TIMES[i]}  ${LABELS[i]}`, sx, sy-8);
    });
    ctx.globalAlpha = 1;
    ctx.fillStyle="rgba(255,255,255,0.45)"; ctx.font="11px ui-monospace, Menlo, monospace";
    ctx.fillText("Sample talk, sample pictures", pad, H-10);
    if (!still) requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
})();
</script></body></html>"""


def render_landing(db):
    """Marketing-style entry view. Fixed dark palette by design (the approved
    concept is dark-led); the editor keeps its own light/dark toggle."""
    st.markdown(_LANDING_CSS, unsafe_allow_html=True)
    st.markdown("""
    <div class="lnd">
        <div class="lnd-brand"><b>&#9679;</b> B-roll</div>
        <div class="lnd-eyebrow">Video in. Pictures out.</div>
        <h1>Every key moment of your video, as a picture.</h1>
        <p class="lnd-sub">Upload a video. B-roll picks out the key ideas in what is said and makes
        the images an editor can place over them. It handles audio in many languages; quality
        varies by language.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Try it", key="landing_try"):
        st.session_state["entered"] = True
        projs = db.list_projects()
        if projs:
            st.session_state.active_project_id = projs[0].id
            for k in ("nav_page", "nav_seg", "nav_sel"):
                st.session_state[k] = "Create images"
        else:
            st.session_state.show_new_project = True
        st.rerun()
    st.markdown('<div class="lnd-cta-note">Opens the generate page</div>', unsafe_allow_html=True)
    components.html(_LANDING_HERO_HTML, height=470)
    st.markdown("""
    <div class="lnd"><div class="lnd-steps">
        <div class="lnd-step"><div class="lnd-step-t">1&nbsp;&nbsp;Drop in a video</div>
            <div class="lnd-step-d">Or just its audio.</div></div>
        <div class="lnd-step"><div class="lnd-step-t">2&nbsp;&nbsp;Choose how many</div>
            <div class="lnd-step-d">You set the number of images.</div></div>
        <div class="lnd-step"><div class="lnd-step-t">3&nbsp;&nbsp;Place them</div>
            <div class="lnd-step-d">Images cover the key ideas across the whole video.</div></div>
    </div></div>
    """, unsafe_allow_html=True)



# Entry view: marketing landing until the user enters the app.
if not st.session_state.get("entered"):
    render_landing(db)
    st.stop()

# ─── Sidebar: Projects ───────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-mark">EH</div>
        <div>
            <div class="brand-name">Edit Helper</div>
            <div class="brand-tag">Narration to b-roll stills</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("&larr; Landing", key="back_to_landing"):
        st.session_state["entered"] = False
        st.rerun()

    # New project button
    if st.button("＋  New Project", use_container_width=True, type="primary"):
        st.session_state.show_new_project = True

    if st.session_state.show_new_project:
        with st.form("new_project_form"):
            new_name = st.text_input("Project Name", placeholder="e.g. Travel Vlog Episode 3")
            new_desc = st.text_area("Description (optional)", placeholder="B-roll for outdoor adventure segment", height=80)
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("Create", type="primary")
            with col2:
                cancelled = st.form_submit_button("Cancel")

            if submitted and new_name.strip():
                project = db.create_project(name=new_name.strip(), description=new_desc.strip())
                st.session_state.active_project_id = project.id
                st.session_state.show_new_project = False
                st.session_state.broll_results = []
                st.session_state.trace_logs = []
                db.add_message(project.id, "system", f"Project '{project.name}' created.")
                st.rerun()
            if cancelled:
                st.session_state.show_new_project = False
                st.rerun()

    st.markdown('<div class="side-label">Projects</div>', unsafe_allow_html=True)

    projects = db.list_projects()
    if not projects:
        st.caption("No projects yet. Create one above.")
    else:
        for proj in projects:
            is_active = st.session_state.active_project_id == proj.id
            label = f"{'●' if is_active else '○'}  {proj.name}"
            if st.button(label, key=f"proj_{proj.id}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.active_project_id = proj.id
                st.session_state.broll_results = []
                st.session_state.trace_logs = []
                st.session_state.pinterest_results = []
                st.rerun()

    # Global settings
    st.markdown("---")
    with st.expander("Global Settings"):
        # ── Search sources ──
        st.markdown("**Search Sources**")
        saved_pexels_key = db.get_setting("pexels_api_key", "")
        session_pexels_key = st.session_state.get("pexels_api_key_session", "")
        pexels_input = st.text_input("Pexels API Key", value=session_pexels_key or saved_pexels_key,
                                     type="password",
                                     help="Free from pexels.com/api. Also loadable from .env file.")
        remember_pexels = st.checkbox(
            "Remember Pexels key on this device", value=bool(saved_pexels_key), key="pexels_remember",
            help="Unchecked: kept for this session only. Checked: stored unencrypted in data/app.db.")
        if pexels_input != (session_pexels_key or saved_pexels_key):
            if remember_pexels:
                db.set_setting("pexels_api_key", pexels_input)
                st.session_state.pop("pexels_api_key_session", None)
            else:
                st.session_state["pexels_api_key_session"] = pexels_input
        if saved_pexels_key and not remember_pexels:
            if st.button("Remove stored Pexels key", key="forget_pexels"):
                db.set_setting("pexels_api_key", "")
                st.toast("Stored Pexels key removed.")
                st.rerun()

        enable_pinterest = st.checkbox("Enable Pinterest Search", value=db.get_setting("enable_pinterest", "1") == "1")
        db.set_setting("enable_pinterest", "1" if enable_pinterest else "0")

        enable_pexels = st.checkbox("Enable Pexels Search", value=db.get_setting("enable_pexels", "1") == "1")
        db.set_setting("enable_pexels", "1" if enable_pexels else "0")

        st.markdown("---")

        # ── Translation LLM ──
        st.markdown("**Translation LLM** (for Tamil/Hindi/non-English)")

        saved_backend = db.get_setting("llm_backend", "local")
        llm_backend = st.radio(
            "Backend",
            options=["local", "api"],
            index=0 if saved_backend == "local" else 1,
            format_func=lambda x: "Local (Ollama)" if x == "local" else "API (Google Gemini)",
            horizontal=True,
            key="llm_backend_radio",
        )
        if llm_backend != saved_backend:
            db.set_setting("llm_backend", llm_backend)

        if llm_backend == "local":
            # Ollama settings
            available_models = list_ollama_models()
            saved_ollama_model = db.get_setting("ollama_model", "")

            if available_models:
                default_idx = 0
                if saved_ollama_model in available_models:
                    default_idx = available_models.index(saved_ollama_model)
                ollama_model = st.selectbox(
                    "Ollama Model", options=available_models,
                    index=default_idx, key="ollama_model_select",
                )
                if ollama_model != saved_ollama_model:
                    db.set_setting("ollama_model", ollama_model)
                st.caption(f"Running locally on GPU. No API key needed.")
            else:
                st.warning("Ollama not running. Start it: `ollama serve`")
                ollama_model = st.text_input("Ollama Model (manual)", value=saved_ollama_model,
                                             placeholder="gemma4:e4b")
                if ollama_model != saved_ollama_model:
                    db.set_setting("ollama_model", ollama_model)
        else:
            # Gemini API settings
            saved_gemini_key = db.get_setting("gemini_api_key", "")
            session_gemini_key = st.session_state.get("gemini_api_key_session", "")
            gemini_input = st.text_input("Gemini API Key", value=session_gemini_key or saved_gemini_key,
                                         type="password",
                                         help="Free from aistudio.google.com/apikey. Also loadable from .env file.")
            remember_gemini = st.checkbox(
                "Remember Gemini key on this device", value=bool(saved_gemini_key), key="gemini_remember",
                help="Unchecked: kept for this session only. Checked: stored unencrypted in data/app.db.")
            if gemini_input != (session_gemini_key or saved_gemini_key):
                if remember_gemini:
                    db.set_setting("gemini_api_key", gemini_input)
                    st.session_state.pop("gemini_api_key_session", None)
                else:
                    st.session_state["gemini_api_key_session"] = gemini_input
            if saved_gemini_key and not remember_gemini:
                if st.button("Remove stored Gemini key", key="forget_gemini"):
                    db.set_setting("gemini_api_key", "")
                    st.toast("Stored Gemini key removed.")
                    st.rerun()
            st.caption("Session keys live in memory only. Remembered keys are stored unencrypted "
                       "in data/app.db - use a key you can rotate and keep this folder private. "
                       "Environment variables (.env) are never written anywhere by the app.")

            model_options = list(GEMINI_MODELS.keys())
            model_labels = list(GEMINI_MODELS.values())
            saved_gemini_model = db.get_setting("gemini_model", "gemini-3.8-flash")
            default_idx = model_options.index(saved_gemini_model) if saved_gemini_model in model_options else 0

            gemini_model = st.radio(
                "Gemini Model", options=model_options,
                index=default_idx,
                format_func=lambda x: GEMINI_MODELS.get(x, x),
                key="gemini_model_radio",
            )
            if gemini_model != saved_gemini_model:
                db.set_setting("gemini_model", gemini_model)

    st.markdown("---")
    st.markdown('<div class="side-label">Appearance</div>', unsafe_allow_html=True)
    def _persist_theme():
        st.query_params["theme"] = st.session_state.theme

    st.radio(
        "Theme", options=["light", "dark"],
        format_func=lambda x: "Light" if x == "light" else "Dark",
        horizontal=True, key="theme", label_visibility="collapsed",
        on_change=_persist_theme,
    )


def get_pexels_key() -> str:
    return (st.session_state.get("pexels_api_key_session", "")
            or settings.PEXELS_API_KEY
            or db.get_setting("pexels_api_key", ""))


def get_gemini_key() -> str:
    return (st.session_state.get("gemini_api_key_session", "")
            or settings.GEMINI_API_KEY
            or db.get_setting("gemini_api_key", ""))


def is_pinterest_enabled() -> bool:
    return db.get_setting("enable_pinterest", "1") == "1"


def is_pexels_enabled() -> bool:
    return db.get_setting("enable_pexels", "1") == "1"


def build_translator(custom_instructions: str = "") -> LLMTranslator:
    return LLMTranslator(
        backend=db.get_setting("llm_backend", "local"),
        api_key=get_gemini_key(),
        gemini_model=db.get_setting("gemini_model", "gemini-3.8-flash"),
        ollama_model=db.get_setting("ollama_model", ""),
        custom_instructions=custom_instructions,
    )


def render_character_sheet(project, scenes, gemini_key: str) -> str:
    """Character sheet + style anchor shared by every scene prompt.

    Drafted once from the selected scenes by the reasoning model, editable,
    and saved per project - so recurring people keep the same description in
    every image. Returns the current sheet text ("" when empty).
    """
    sheet_key = f"character_sheet_{project.id}"
    if sheet_key not in st.session_state:
        st.session_state[sheet_key] = db.get_memory(project.id, "character_sheet") or ""

    st.markdown(
        '<div class="section-sub"><b>Characters &amp; style</b> - one shared sheet keeps the '
        'same people looking the same in every image. Draft it from your scenes, then tweak '
        'anything - it goes into every prompt below.</div>',
        unsafe_allow_html=True,
    )

    def _save_sheet():
        db.set_memory(project.id, "character_sheet", st.session_state[sheet_key])

    if st.button("Draft character sheet from scenes", key=f"draft_sheet_{project.id}",
                 disabled=not gemini_key):
        translator = build_translator(project.custom_instructions)
        scene_lines = [f"{i + 1}. {s}" for i, s in enumerate(scenes)]
        with st.spinner("Drafting character sheet..."):
            drafted = draft_character_sheet(
                translator.generate_text, scene_lines, project.custom_instructions)
        if drafted:
            st.session_state[sheet_key] = drafted
            _save_sheet()
            st.rerun()
        else:
            st.error("The reasoning model returned nothing - check your Gemini key, "
                     "or write the sheet by hand below.")

    st.text_area(
        "Character sheet (applies to every image)",
        key=sheet_key, on_change=_save_sheet, height=120,
        placeholder=("MOTHER: woman in her 40s, blue cotton saree, shoulder-length black hair in a braid.\n"
                     "DAUGHTER: girl around 8, pink frock, two ponytails.\n"
                     "STYLE ANCHOR: warm natural window light, soft tones, cinematic 16:9 still, shallow depth of field."),
    )
    return st.session_state[sheet_key]


def step_strip(states, labels=None):
    """Render the 4-step create-flow tracker.
    states: list of 4 strings, each 'done', 'active', or 'todo'."""
    labels = labels or ["Upload media", "Understand speech", "Plan coverage", "Generate images"]
    parts = ['<div class="steps">']
    for i, (label, state) in enumerate(zip(labels, states)):
        parts.append(f'<div class="step {state}"><span class="step-num">{i + 1}</span>{label}</div>')
        if i < 3:
            parts.append('<div class="step-line"></div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def coverage_bar(segments, picked_indices):
    """Render a mini timeline showing where the picked moments sit across the video."""
    if not segments:
        return
    total = max(seg.end_time for seg in segments) or 1
    ticks = []
    for i, seg in enumerate(segments):
        left = (seg.start_time / total) * 100
        width = max(((seg.end_time - seg.start_time) / total) * 100, 0.8)
        picked = i in picked_indices
        color = "var(--accent)" if picked else "var(--surface-3)"
        opacity = "1" if picked else "0.7"
        ticks.append(
            f'<div style="position:absolute;left:{left:.2f}%;width:{width:.2f}%;top:0;bottom:0;'
            f'background:{color};opacity:{opacity};border-radius:3px;" '
            f'title="{seg.start_time:.0f}s"></div>'
        )
    st.markdown(
        f'<div style="margin:2px 0 6px;font-size:11px;font-family:var(--mono);'
        f'letter-spacing:0.03em;color:var(--muted);">COVERAGE ACROSS YOUR VIDEO / 0:00 - '
        f'{int(total // 60)}:{int(total % 60):02d}</div>'
        f'<div style="position:relative;height:16px;background:var(--surface-2);'
        f'border:1px solid var(--border);border-radius:8px;overflow:hidden;margin-bottom:18px;">'
        f'{"".join(ticks)}</div>',
        unsafe_allow_html=True,
    )


# ─── Main Content ─────────────────────────────────────────────
project = get_active_project()

if project is None:
    # ── Landing ──
    st.markdown("""
    <section class="hero">
        <div class="hero-eyebrow"><span class="pulse-dot"></span>Gemini-powered image studio</div>
        <h1 class="hero-title">Turn narration into a <em>visual story</em>.</h1>
        <p class="hero-sub">
            Upload a rough cut or voiceover. Gemini listens to the speech — including Tamil and
            other languages, model quality permitting — understands what it is about, and frames
            a set of stills you can drop anywhere into your edit.
        </p>
    </section>
    """, unsafe_allow_html=True)

    if not projects:
        st.markdown('<div class="note-box">Create a project in the sidebar to get started.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-eyebrow">How it works</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    features = [
        ("01", "Upload media", "Drop in a video or audio file (upload cap set in config, 10GB here). The audio track carries the story."),
        ("02", "Understand speech", "Gemini transcribes and reasons about the narration with timestamps. Multilingual, including Tamil — verify accuracy on your clips."),
        ("03", "Plan coverage", "Pick how many images you need. Moments spread across the whole video — you approve every prompt."),
        ("04", "Generate images", "Stills render inline on this page, matched to transcript passages, ready for your timeline."),
    ]
    for col, (num, title, desc) in zip((f1, f2, f3, f4), features):
        with col:
            st.markdown(f"""
            <div class="feat-card">
                <div class="feat-num">{num}</div>
                <div class="feat-title">{title}</div>
                <p class="feat-desc">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    # Show recent activity
    if projects:
        st.markdown('<div class="section-eyebrow">Recent projects</div>', unsafe_allow_html=True)
        for proj in projects[:5]:
            resource_count = len(db.get_resources(proj.id))
            st.markdown(f"""
            <div class="proj-card">
                <div class="proj-name">{proj.name}</div>
                <div class="proj-meta">{resource_count} resources · {proj.description[:80] if proj.description else 'No description'}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

else:
    # ─── Project Header ──────────────────────────────────────
    st.markdown(f"""
    <div class="page-head">
        <h1>{project.name}</h1>
        <p>{project.description or 'No description'}</p>
    </div>
    """, unsafe_allow_html=True)

    # ─── Navigation ───────────────────────────────────────────
    # Desktop: spacious underline bar (segmented control restyled).
    # Mobile: clearly labelled page picker (selectbox). Same state, CSS swaps them.
    _NAV_PAGES = ["Create images", "Library", "Find stock", "Stock pipeline", "Activity", "Settings"]
    if "nav_page" not in st.session_state or st.session_state["nav_page"] not in _NAV_PAGES:
        st.session_state["nav_page"] = "Create images"

    def _nav_from_seg():
        st.session_state["nav_page"] = st.session_state["nav_seg"]
        st.session_state["nav_sel"] = st.session_state["nav_seg"]

    def _nav_from_sel():
        st.session_state["nav_page"] = st.session_state["nav_sel"]
        st.session_state["nav_seg"] = st.session_state["nav_sel"]

    st.session_state.setdefault("nav_seg", st.session_state["nav_page"])
    st.session_state.setdefault("nav_sel", st.session_state["nav_page"])
    st.segmented_control(
        "Pages", _NAV_PAGES,
        key="nav_seg", on_change=_nav_from_seg, label_visibility="collapsed", width="content",
    )
    st.selectbox(
        "Go to page", _NAV_PAGES,
        key="nav_sel", on_change=_nav_from_sel,
    )
    _page = st.session_state["nav_page"]

    if _page == "Create images":
        st.markdown('<div class="section-title">Create images from a video</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Gemini listens to the audio, understands the story, and generates a set of stills that cover the whole video. Images appear right here on the page.</div>', unsafe_allow_html=True)

        gemini_key = get_gemini_key()
        if not gemini_key:
            key_input = st.text_input(
                "Gemini API key", type="password", key="create_gemini_key",
                help="Kept for this session only. To remember it on this device, use Global Settings.")
            if key_input:
                st.session_state["gemini_api_key_session"] = key_input.strip()
                gemini_key = key_input.strip()

        # ── Output mode: generate here, or copy prompts to your own generator ──
        mode_key = f"output_mode_{project.id}"
        if mode_key not in st.session_state:
            st.session_state[mode_key] = db.get_memory(project.id, "output_mode") or "Generate images"

        def _save_mode():
            db.set_memory(project.id, "output_mode", st.session_state[mode_key])

        mode = st.segmented_control(
            "Output mode", ["Generate images", "Copy prompts"],
            key=mode_key, on_change=_save_mode,
            help=("Generate images: Gemini creates the stills here. "
                  "Copy prompts: finished prompts to paste into your own image "
                  "generator - no image generation happens here."),
        ) or "Generate images"

        analysis_key = f"image_analysis_{project.id}"
        images_key = f"generated_images_{project.id}"
        analysis = st.session_state.get(analysis_key)
        generated = st.session_state.get(images_key, [])

        _done = [
            bool(st.session_state.get(f"image_source_{project.id}") or analysis or generated),
            bool(analysis),
            bool(analysis),
            bool(generated),
        ]
        _next = next((i for i, d in enumerate(_done) if not d), None)
        _step4 = "Generate images" if mode == "Generate images" else "Copy prompts"
        step_strip(["done" if d else ("active" if i == _next else "todo") for i, d in enumerate(_done)],
                   labels=["Upload media", "Understand speech", "Plan coverage", _step4])

        # ── Step 1: source media ──
        st.markdown('<div class="section-eyebrow">Step 1 · Source media</div>', unsafe_allow_html=True)
        source_video = st.file_uploader(
            "Video or audio file",
            type=["mp4", "mov", "mkv", "avi", "webm", "mp3", "wav", "flac", "m4a"],
            key="create_video_upload",
        )
        st.caption("The audio is analyzed by your chosen Gemini model. It handles multilingual speech including Tamil, but quality varies by model and clip — check the transcript before generating. Generated images are saved in this project's library.")

        if st.button("Analyze audio →", type="primary", disabled=not (gemini_key and source_video), key="analyze_video"):
            try:
                from core.audio_aligner import extract_audio

                suffix = Path(source_video.name).suffix.lower()
                media_path = settings.TEMP_DIR / f"{uuid4().hex}{suffix}"
                media_path.write_bytes(source_video.getbuffer())
                with st.spinner("Extracting and analyzing audio..."):
                    audio_path = extract_audio(media_path)
                    translator = build_translator(project.custom_instructions)
                    analysis_new = translator.transcribe_audio(
                        str(audio_path), transcribe_model=db.get_setting("gemini_model", "gemini-3.8-flash"))
                if not analysis_new or not analysis_new.segments:
                    st.error("Gemini could not find usable speech. Try a video with clear narration or dialogue.")
                else:
                    st.session_state[analysis_key] = analysis_new
                    st.session_state[f"image_source_{project.id}"] = source_video.name
                    st.session_state[images_key] = []
                    db.set_memory(project.id, "last_analysis", json.dumps({
                        "language": analysis_new.language, "backend": analysis_new.backend,
                        "source_file": source_video.name,
                        "segments": [asdict(seg) for seg in analysis_new.segments],
                    }))
                    db.add_message(project.id, "assistant", f"Analyzed {source_video.name}: {len(analysis_new.segments)} audio moments.")
                    st.rerun()
            except Exception as exc:
                st.error(f"Audio analysis failed: {exc}")

        analysis = st.session_state.get(analysis_key)
        if analysis is None:
            _saved_raw = db.get_memory(project.id, "last_analysis")
            if _saved_raw:
                try:
                    _saved = json.loads(_saved_raw)
                    if not (source_video and source_video.name != _saved.get("source_file")):
                        _restored = AudioTranscription(
                            language=_saved.get("language", ""),
                            backend=_saved.get("backend", ""),
                            segments=[AudioSegment(**seg) for seg in _saved.get("segments", [])],
                        )
                        if _restored.segments:
                            analysis = _restored
                            st.session_state[analysis_key] = analysis
                            st.session_state[f"image_source_{project.id}"] = _saved.get("source_file")
                except Exception:
                    analysis = None
        if source_video and st.session_state.get(f"image_source_{project.id}") != source_video.name:
            analysis = None

        # ── Step 2 + 3: moments and coverage plan ──
        if analysis and analysis.segments:
            st.markdown('<div class="section-eyebrow">Step 2 · What the speaker says</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">{len(analysis.segments)} moments found</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Review the transcript and visual ideas before generating. Edit any image prompt below.</div>', unsafe_allow_html=True)

            image_count = st.number_input(
                "Images to create — spread across the whole video", min_value=1, max_value=min(12, len(analysis.segments)),
                value=min(3, len(analysis.segments)), key=f"image_count_{project.id}",
            )
            count = int(image_count)
            indices = [round(i * (len(analysis.segments) - 1) / max(count - 1, 1)) for i in range(count)]
            if count == 1:
                indices = [len(analysis.segments) // 2]

            coverage_bar(analysis.segments, set(indices))

            st.markdown('<div class="section-eyebrow">Step 3 · Scene plan</div>', unsafe_allow_html=True)
            _scenes = [
                (analysis.segments[i].visual_description or analysis.segments[i].english_translation
                 or analysis.segments[i].original_text)
                for i in indices
            ]
            sheet = render_character_sheet(project, _scenes, gemini_key)
            _snap_key = f"sheet_snapshot_{project.id}"
            _ver_key = f"sheet_ver_{project.id}"
            st.session_state.setdefault(_ver_key, 0)
            if st.session_state.get(_snap_key) != sheet:
                st.session_state[_snap_key] = sheet
                st.session_state[_ver_key] += 1
            prompts = []
            for scene_no, index in enumerate(indices, 1):
                segment = analysis.segments[index]
                scene = segment.visual_description or segment.english_translation or segment.original_text
                with st.expander(f"Image {scene_no} · {segment.start_time:.0f}–{segment.end_time:.0f}s  ·  {segment.original_text[:80]}", expanded=True):
                    if segment.english_translation and segment.english_translation != segment.original_text:
                        st.markdown(f'<div class="translation-box"><b>English:</b> {segment.english_translation}</div>',
                                    unsafe_allow_html=True)
                    if mode == "Generate images":
                        prompt = st.text_area(
                            "Image prompt",
                            value=compose_scene_prompt(scene, sheet, project.custom_instructions),
                            key=f"scene_prompt_{project.id}_{index}_gen_v{st.session_state[_ver_key]}",
                            height=140,
                        )
                    else:
                        prompt = st.text_area(
                            "Scenario line",
                            value=scene,
                            key=f"scene_prompt_{project.id}_{index}_copy_v{st.session_state[_ver_key]}",
                            height=68,
                        )
                    prompts.append((segment, prompt))

            if mode == "Generate images":
                use_ref = st.checkbox(
                    "Use the previous image as a visual reference",
                    value=True, key=f"use_ref_{project.id}",
                    help=("Each image after the first is generated with the previous scene's "
                          "image attached - same people, same style, carried forward."))
                if st.button(f"Generate {count} image{'s' if count > 1 else ''} →", type="primary", key="generate_video_images"):
                    created = []
                    prev_image = None
                    with st.status("Generating images", expanded=True) as status:
                        for segment, prompt in prompts:
                            try:
                                output = settings.ASSETS_OUTPUT_DIR / project.id / f"{uuid4().hex}.png"
                                image_path = generate_image(gemini_key, prompt, output,
                                                            reference_image=prev_image if use_ref else None)
                                db.add_resource(
                                    project_id=project.id, resource_type="image",
                                    title=f"Generated image at {segment.start_time:.0f}s",
                                    file_path=str(image_path),
                                    metadata={"source": "Gemini", "start_time": segment.start_time,
                                              "end_time": segment.end_time, "prompt": prompt},
                                )
                                created.append((segment.start_time, str(image_path)))
                                prev_image = image_path
                                st.write(f"Created image for {segment.start_time:.0f}s")
                            except Exception as exc:
                                st.error(f"Image at {segment.start_time:.0f}s failed: {exc}")
                                prev_image = None  # a failed scene breaks the reference chain
                        failed = len(prompts) - len(created)
                        if failed:
                            status.update(
                                label=f"Created {len(created)} of {len(prompts)} images ({failed} failed)",
                                state="error", expanded=(len(created) == 0))
                        else:
                            status.update(label=f"Created {len(created)} of {len(prompts)} images",
                                          state="complete", expanded=False)
                    st.session_state[images_key] = created
                    if created:
                        db.add_message(project.id, "assistant", f"Generated {len(created)} images from audio moments.")
            else:
                st.markdown(
                    '<div class="section-sub">One block - paste it into your Gemini or ChatGPT chat. '
                    'The sheet on top carries the same characters through every image.</div>',
                    unsafe_allow_html=True,
                )
                compact_scenes = [
                    (f"IMAGE {scene_no} ({segment.start_time:.0f}s)", prompt_text)
                    for scene_no, (segment, prompt_text) in enumerate(prompts, 1)
                ]
                st.code(compose_compact_block(compact_scenes, sheet, project.custom_instructions),
                        language=None)

        # ── Step 4: results gallery (images visible on the page) ──
        generated = st.session_state.get(images_key, [])
        if generated:
            st.markdown('<div class="section-eyebrow">Step 4 · Ready for your edit</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Each image is matched to a transcript passage — drop it into your timeline at the marked time.</div>', unsafe_allow_html=True)
            gallery_cols = 3 if len(generated) >= 3 else len(generated)
            for row_start in range(0, len(generated), gallery_cols):
                cols = st.columns(gallery_cols)
                for col_idx, col in enumerate(cols):
                    idx = row_start + col_idx
                    if idx >= len(generated):
                        break
                    start_time, image_path = generated[idx]
                    with col:
                        if Path(image_path).exists():
                            st.markdown(f'<span class="img-chip">Place at ~{start_time:.0f}s</span>', unsafe_allow_html=True)
                            st.image(image_path, use_container_width=True)
                            with open(image_path, "rb") as image_file:
                                st.download_button("Download", image_file, file_name=Path(image_path).name,
                                                   mime="image/png", key=f"download_{Path(image_path).stem}",
                                                   use_container_width=True)

    # ═══════════════════════════════════════════════════════════
    # TAB: SEARCH (Pinterest + Pexels quick search)
    # ═══════════════════════════════════════════════════════════
    if _page == "Find stock":
        st.markdown('<div class="section-title">Search for B-Roll</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Search Pinterest and Pexels. The agent reasons about your query to find better visuals.</div>', unsafe_allow_html=True)

        search_col1, search_col2, search_col3 = st.columns([3, 1, 1])
        with search_col1:
            search_query = st.text_input("Search query", placeholder="Describe the visual you need",
                                         key="search_query_input")
            st.caption('Example: "cinematic mountain landscape drone aerial". Tamil or Hindi works too - it gets translated before searching.')
        with search_col2:
            fetch_count = st.number_input("Images to fetch", min_value=5, max_value=100, value=20, step=5,
                                          key="fetch_count_search")
        with search_col3:
            skip_duplicates = st.checkbox("Skip already saved", value=True, key="skip_dupes_search")

        if st.button("Search", type="primary", use_container_width=True, key="search_btn"):
            if search_query.strip():
                with st.spinner(f"Reasoning and searching for '{search_query}'..."):
                    db.add_message(project.id, "user", f"Search: {search_query}")

                    # Build dedup set
                    exclude_urls = set()
                    if skip_duplicates:
                        exclude_urls = db.get_saved_urls(project.id) | db.get_saved_download_urls(project.id)

                    # Translate non-English search query via LLM
                    effective_query = search_query
                    llm_search_queries = []
                    if is_non_english(search_query):
                        translator = build_translator(project.custom_instructions)
                        translated = translator.translate_and_reason(search_query)
                        if translated.english_translation != search_query:
                            effective_query = translated.english_translation
                            llm_search_queries = translated.search_queries
                            st.session_state["last_translation"] = translated

                    # Use reasoner to generate smart queries
                    reasoner = QueryReasoner(custom_instructions=project.custom_instructions)
                    reasoned = reasoner.reason(effective_query)
                    if llm_search_queries:
                        reasoned.queries = llm_search_queries + reasoned.queries
                    st.session_state["last_reasoning"] = reasoned

                    sourcer = StockSourcer(
                        api_key=get_pexels_key(),
                        enable_pinterest=is_pinterest_enabled(),
                        enable_pexels=is_pexels_enabled(),
                        custom_instructions=project.custom_instructions,
                    )

                    # Search using all reasoned queries
                    all_candidates = []
                    seen_ids = set()
                    for q in reasoned.queries[:4]:
                        found = sourcer.search_all(q, per_page=min(fetch_count // 2, 15),
                                                   exclude_urls=exclude_urls)
                        for c in found:
                            if c.id not in seen_ids:
                                seen_ids.add(c.id)
                                all_candidates.append(c)
                        if len(all_candidates) >= fetch_count:
                            break

                    st.session_state.pinterest_results = all_candidates[:fetch_count]

                    sources_summary = {}
                    for c in all_candidates[:fetch_count]:
                        sources_summary[c.source] = sources_summary.get(c.source, 0) + 1
                    summary_parts = [f"{v} from {k}" for k, v in sources_summary.items()]

                    dup_note = f" ({len(exclude_urls)} already-saved filtered)" if exclude_urls else ""
                    db.add_message(project.id, "assistant",
                                   f"Reasoned queries: {reasoned.queries} | "
                                   f"Found {len(all_candidates[:fetch_count])} results "
                                   f"({', '.join(summary_parts)}){dup_note}")

                    # Remember search
                    prev_searches = db.get_memory(project.id, "search_history") or "[]"
                    search_list = json.loads(prev_searches)
                    search_list.append({"query": search_query, "reasoned_queries": reasoned.queries,
                                        "mood": reasoned.mood, "results": len(all_candidates[:fetch_count]),
                                        "time": datetime.now().isoformat()})
                    db.set_memory(project.id, "search_history", json.dumps(search_list[-20:]))

                    st.rerun()

        # Show translation output
        if "last_translation" in st.session_state and st.session_state["last_translation"]:
            trans = st.session_state["last_translation"]
            with st.expander("Translation", expanded=True):
                st.markdown(f"**Language:** `{trans.language}`")
                st.markdown(f"**English:** {trans.english_translation}")
                if trans.visual_description:
                    st.markdown(f"**Visual reasoning:** {trans.visual_description}")
                if trans.search_queries:
                    st.markdown(f"**LLM search queries:** {', '.join(f'`{q}`' for q in trans.search_queries)}")

        # Show reasoning output
        if "last_reasoning" in st.session_state and st.session_state["last_reasoning"]:
            reasoned = st.session_state["last_reasoning"]
            with st.expander("Agent Reasoning", expanded=True):
                st.markdown(f"**Mood detected:** `{reasoned.mood}`")
                if reasoned.visual_concepts:
                    st.markdown(f"**Visual concepts:** {', '.join(f'`{c}`' for c in reasoned.visual_concepts)}")
                st.markdown(f"**Search queries generated:** {', '.join(f'`{q}`' for q in reasoned.queries)}")
                if reasoned.style_modifiers:
                    st.markdown(f"**Style:** {', '.join(reasoned.style_modifiers)}")
                st.caption(reasoned.analysis)

        # Display search results
        results = st.session_state.pinterest_results
        if results:
            st.markdown(f"**{len(results)} results found**")
            if any(getattr(c, "source", "") == "fallback" for c in results):
                st.info("No live stock source answered (offline or no Pexels key). Cards marked "
                        "SAMPLE are curated stand-ins from the app's built-in catalog - not live "
                        "search results. Add a Pexels key in Global Settings for real results.")

            # Grid display
            cols_per_row = 3
            for row_start in range(0, len(results), cols_per_row):
                cols = st.columns(cols_per_row)
                for col_idx, col in enumerate(cols):
                    idx = row_start + col_idx
                    if idx >= len(results):
                        break
                    cand = results[idx]
                    with col:
                        # Source badge
                        source_class = f"source-{cand.source}"
                        source_label = "SAMPLE" if cand.source == "fallback" else cand.source.upper()
                        st.markdown(
                            f'<span class="source-badge {source_class}">{source_label}</span> '
                            f'<span class="media-type-tag">{cand.media_type}</span>',
                            unsafe_allow_html=True,
                        )

                        # Preview image
                        try:
                            st.image(cand.preview_url, use_container_width=True)
                        except Exception:
                            st.info("Preview unavailable")

                        st.caption(cand.title[:80] if cand.title else "Untitled")

                        # Save button
                        if st.button(f"Save to Project", key=f"save_{idx}_{cand.id}"):
                            db.add_resource(
                                project_id=project.id,
                                resource_type=cand.media_type,
                                title=cand.title,
                                source_url=cand.source_url,
                                file_path="",
                                metadata=json.dumps({
                                    "download_url": cand.download_url,
                                    "preview_url": cand.preview_url,
                                    "source": cand.source,
                                    "width": cand.width,
                                    "height": cand.height,
                                }),
                            )
                            db.add_message(project.id, "assistant",
                                           f"Saved '{cand.title[:50]}' ({cand.source}) to resources.")
                            st.toast(f"Saved: {cand.title[:40]}")

    # ═══════════════════════════════════════════════════════════
    # TAB: PIPELINE (Audio → Beats → Auto Source)
    # ═══════════════════════════════════════════════════════════
    if _page == "Stock pipeline":
        st.markdown('<div class="section-title">Autonomous B-Roll Pipeline</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Upload audio/video, agent extracts narrative beats and auto-sources matching b-roll.</div>', unsafe_allow_html=True)

        pipe_col1, pipe_col2 = st.columns([1, 1])

        with pipe_col1:
            st.markdown("**Aesthetic Memory**")
            ref_folder = st.text_input(
                "Reference Assets Directory",
                value=str(settings.REFERENCE_ASSETS_DIR),
                key="ref_folder_pipeline",
            )
            if st.button("Sync Aesthetic Memory"):
                with st.spinner("Ingesting references into Qdrant..."):
                    embedder = MultimodalEmbedder()
                    if getattr(embedder, "is_mock", False):
                        st.warning("Embedding model not loaded - deterministic fallback mode. "
                                   "Similarity matching is approximate, not real visual judgment.")
                    memory = AestheticMemory(embedder=embedder)
                    res = memory.ingest_reference_folder(Path(ref_folder))
                    if res.get("status") == "success":
                        st.success(f"Ingested {res['ingested_count']} files.")
                    else:
                        st.warning(res.get("message", "Check folder path"))

        with pipe_col2:
            st.markdown("**Thresholds**")
            threshold_val = st.slider("Rejection Threshold", 0.50, 0.95, settings.SIMILARITY_THRESHOLD, 0.01,
                                      key="threshold_pipeline")
            weight_text = st.slider("Semantic Weight", 0.0, 1.0, settings.TEXT_WEIGHT, 0.05,
                                    key="weight_pipeline")
            st.caption(f"Aesthetic Weight: **{round(1.0 - weight_text, 2)}**")

        pipe_col3, pipe_col4, pipe_col5 = st.columns([1, 1, 1])
        with pipe_col3:
            pipeline_fetch_count = st.number_input("Candidates per beat", min_value=3, max_value=50,
                                                   value=10, step=2, key="pipeline_fetch_count",
                                                   help="How many images/videos to fetch and evaluate per narrative beat.")
        with pipe_col4:
            pipeline_skip_dupes = st.checkbox("Skip already-saved assets", value=True, key="pipeline_skip_dupes",
                                              help="Avoid fetching images already saved in this project.")
        with pipe_col5:
            transcribe_method = st.radio(
                "Transcription",
                options=["whisper", "gemini"],
                format_func=lambda x: "Whisper (local GPU)" if x == "whisper" else "Gemini (API)",
                index=0, key="transcribe_method",
                help="Gemini: API call = transcribe + translate + queries. Whisper: local GPU, separate translation.",
            )
            if transcribe_method == "gemini":
                transcribe_model_options = list(GEMINI_TRANSCRIBE_MODELS.keys())
                saved_transcribe_model = db.get_setting("gemini_transcribe_model", "gemini-3.8-flash")
                default_idx = transcribe_model_options.index(saved_transcribe_model) if saved_transcribe_model in transcribe_model_options else 0
                gemini_transcribe_model = st.selectbox(
                    "Transcribe model",
                    options=transcribe_model_options,
                    index=default_idx,
                    format_func=lambda x: GEMINI_TRANSCRIBE_MODELS.get(x, x),
                    key="gemini_transcribe_model_select",
                )
                if gemini_transcribe_model != saved_transcribe_model:
                    db.set_setting("gemini_transcribe_model", gemini_transcribe_model)

        st.markdown("---")

        st.markdown("**Input Media**")
        st.caption("Upload a video/audio file (up to 10GB, set in .streamlit config). Or paste a local file path for very large files.")

        input_method = st.radio("Input method", ["Upload file", "Local file path", "Demo sample"],
                                horizontal=True, key="input_method")

        uploaded_file = None
        local_file_path = ""
        if input_method == "Upload file":
            uploaded_file = st.file_uploader(
                "Upload Video or Audio (mp4, mkv, avi, mov, mp3, wav, flac — up to 10GB)",
                type=["mp4", "mkv", "avi", "mov", "webm", "mp3", "wav", "flac", "aac", "ogg", "m4a", "wma"],
                key="pipeline_upload",
            )
        elif input_method == "Local file path":
            local_file_path = st.text_input(
                "Full path to video/audio file on disk",
                placeholder=r"C:\Videos\my_project\episode_03.mp4",
                key="local_path_input",
            )

        if st.button("Run Pipeline", type="primary", use_container_width=True,
                     disabled=st.session_state.is_processing, key="run_pipeline_btn"):
            st.session_state.is_processing = True
            st.session_state.broll_results = []
            st.session_state.trace_logs = []

            db.add_message(project.id, "user", "Started autonomous b-roll pipeline.")

            progress = st.progress(0, text="Initializing...")

            with st.status("Agent Execution", expanded=True) as status_box:
                # Resolve media path
                media_path = None
                if input_method == "Upload file":
                    if uploaded_file is None:
                        progress.empty()
                        st.error("Upload a file first (or switch the input method).")
                        st.session_state.is_processing = False
                        st.stop()
                    temp_input = settings.TEMP_DIR / uploaded_file.name
                    with open(temp_input, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    media_path = temp_input
                elif input_method == "Local file path":
                    if not local_file_path.strip():
                        progress.empty()
                        st.error("Enter a local file path first (or switch the input method).")
                        st.session_state.is_processing = False
                        st.stop()
                    media_path = Path(local_file_path.strip())
                    if not media_path.exists():
                        progress.empty()
                        st.error(f"File not found: {media_path}")
                        st.session_state.is_processing = False
                        st.stop()
                else:
                    demo_path = settings.BASE_DIR / "sample_inputs" / "sample_voiceover.wav"
                    media_path = demo_path if demo_path.exists() else None
                    if media_path is None:
                        progress.empty()
                        st.error("Demo sample is missing from sample_inputs/.")
                        st.session_state.is_processing = False
                        st.stop()

                st.write(f"Ingesting: {media_path.name}")

                use_gemini_transcribe = (transcribe_method == "gemini" and get_gemini_key())
                gemini_segments = None

                if use_gemini_transcribe:
                    # ── Gemini path ──
                    selected_transcribe_model = db.get_setting("gemini_transcribe_model", "gemini-3.8-flash")
                    is_dedicated = "transcribe" in selected_transcribe_model
                    label = f"Gemini ({selected_transcribe_model}): transcribing"
                    if not is_dedicated:
                        label += " + translating"
                    progress.progress(15, text=f"{label}...")

                    from core.audio_aligner import extract_audio
                    audio_path = extract_audio(media_path)
                    st.write(f"Audio extracted: {audio_path.name}")

                    translator = build_translator(project.custom_instructions)
                    transcription = translator.transcribe_audio(str(audio_path), transcribe_model=selected_transcribe_model)

                    if transcription and transcription.segments:
                        gemini_segments = transcription
                        st.write(f"Gemini transcribed {len(transcription.segments)} segments ({transcription.language}).")
                        # Build NarrativeBeat objects from Gemini segments
                        beats = []
                        for idx, seg in enumerate(transcription.segments, 1):
                            beats.append(NarrativeBeat(
                                index=idx,
                                start_time=seg.start_time,
                                end_time=seg.end_time,
                                duration=round(seg.end_time - seg.start_time, 2),
                                text=seg.original_text,
                                search_keywords=seg.search_queries[:3],
                            ))
                    else:
                        st.warning("Gemini transcription failed. Falling back to Whisper.")
                        use_gemini_transcribe = False

                if not use_gemini_transcribe:
                    # ── Whisper path: local GPU transcription ──
                    progress.progress(15, text="Transcribing with Whisper (local GPU)...")
                    aligner = AudioBeatAligner(model_size=settings.WHISPER_MODEL_SIZE)
                    words = aligner.transcribe_and_align(media_path)
                    beats = aligner.segment_into_narrative_beats(words)

                if not beats:
                    st.error("Transcription produced no usable speech from this file. "
                             "Try a clip with clearer narration, or switch the transcription method.")
                    st.session_state.is_processing = False
                    st.stop()

                st.write(f"Found {len(beats)} narrative beats.")

                # Load models
                progress.progress(35, text="Loading CLIP & Aesthetic Memory...")
                embedder = MultimodalEmbedder()
                if getattr(embedder, "is_mock", False):
                    st.warning("Embedding model not loaded - scoring runs in deterministic fallback "
                               "mode, so candidate rankings are approximate, not real quality judgments.")
                aesthetic_mem = AestheticMemory(embedder=embedder)
                if aesthetic_mem.aesthetic_centroid is None:
                    aesthetic_mem.ingest_reference_folder(Path(ref_folder))

                # Custom instructions
                custom_instr = project.custom_instructions
                if custom_instr:
                    st.write(f"Custom instructions loaded ({len(custom_instr)} chars).")

                # Build dedup set
                exclude_urls = set()
                if pipeline_skip_dupes:
                    exclude_urls = db.get_saved_urls(project.id) | db.get_saved_download_urls(project.id)
                    if exclude_urls:
                        st.write(f"Filtering {len(exclude_urls)} already-saved asset URLs.")

                # Build translator (for Whisper path — Gemini path already has translations)
                translator = build_translator(project.custom_instructions)

                # Build agent
                progress.progress(50, text="Running reasoner → sourcer → director...")
                sourcer = StockSourcer(
                    api_key=get_pexels_key(),
                    custom_instructions=project.custom_instructions,
                )
                director = DirectorEvaluator(embedder=embedder, aesthetic_memory=aesthetic_mem)

                trace_logs = []

                def log_cb(msg):
                    trace_logs.append(msg)
                    st.write(msg)

                agent = BRollAgentGraph(
                    sourcer=sourcer,
                    director=director,
                    embedder=embedder,
                    aesthetic_memory=aesthetic_mem,
                    threshold=threshold_val,
                    log_callback=log_cb,
                    exclude_urls=exclude_urls,
                    translator=translator,
                )

                completed = []
                for i, beat in enumerate(beats):
                    pct = int(50 + (i / max(len(beats), 1)) * 45)
                    progress.progress(pct, text=f"Beat {beat.index}/{len(beats)}: '{beat.text[:30]}...'")

                    # If Gemini already translated, use English text for embedding
                    translation_info = None
                    if gemini_segments and i < len(gemini_segments.segments):
                        seg = gemini_segments.segments[i]
                        embed_text = seg.english_translation or beat.text
                        translation_info = TranslatedBeat(
                            original_text=seg.original_text,
                            language=gemini_segments.language,
                            english_translation=seg.english_translation,
                            visual_description=seg.visual_description,
                            search_queries=seg.search_queries,
                            mood=seg.mood,
                            backend=gemini_segments.backend,
                        )
                    else:
                        embed_text = beat.text

                    text_emb = embedder.embed_text(embed_text)
                    final_state = agent.process_beat(beat, text_emb)
                    # Merge translation from either Gemini or workflow
                    if not translation_info:
                        translation_info = getattr(final_state, '_translation', None)
                    completed.append({"beat": beat, "approved_asset": final_state.approved_asset,
                                      "evaluated_candidates": final_state.evaluated_assets,
                                      "translation": translation_info})

                    # Save approved asset as resource
                    if final_state.approved_asset and final_state.approved_asset.local_path:
                        db.add_resource(
                            project_id=project.id,
                            resource_type=final_state.approved_asset.candidate.media_type,
                            title=final_state.approved_asset.candidate.title,
                            file_path=final_state.approved_asset.local_path,
                            source_url=final_state.approved_asset.candidate.source_url,
                            metadata=json.dumps({
                                "beat_index": beat.index,
                                "composite_score": final_state.approved_asset.composite_score,
                                "source": final_state.approved_asset.candidate.source,
                            }),
                        )

                st.session_state.broll_results = completed
                st.session_state.trace_logs = trace_logs
                progress.progress(100, text="Done!")
                status_box.update(label="Pipeline Complete", state="complete", expanded=False)

            db.add_message(project.id, "assistant",
                           f"Pipeline finished. {len(completed)} beats processed, "
                           f"{sum(1 for c in completed if c['approved_asset'])} assets approved.")

            st.session_state.is_processing = False
            st.rerun()

        # Display results
        if st.session_state.broll_results:
            st.markdown('<div class="section-title">Sequenced B-Roll Feed</div>', unsafe_allow_html=True)
            for item in st.session_state.broll_results:
                beat = item["beat"]
                approved = item.get("approved_asset")

                with st.container():
                    col_meta, col_media = st.columns([1.3, 1.7])

                    with col_meta:
                        st.markdown(f"""
                        <span class="timestamp-badge">{beat.formatted_range}</span>
                        <span class="duration-badge">{beat.duration:.1f}s</span>
                        """, unsafe_allow_html=True)
                        st.markdown(f'<div class="transcript-box">"{beat.text}"</div>', unsafe_allow_html=True)

                        # Show translation if non-English
                        translation = item.get("translation")
                        if translation:
                            st.markdown(
                                f'<div class="translation-box">'
                                f'<b>English:</b> "{translation.english_translation}"</div>',
                                unsafe_allow_html=True,
                            )
                            if translation.visual_description:
                                st.caption(f"Visuals: {translation.visual_description}")
                            if translation.search_queries:
                                st.caption("LLM queries: " + ", ".join(f"`{q}`" for q in translation.search_queries))

                        if beat.search_keywords:
                            st.caption("Keywords: " + ", ".join(f"`{kw}`" for kw in beat.search_keywords))

                        if approved:
                            source_class = f"source-{approved.candidate.source}"
                            st.markdown(f"""
                            <span class="metric-pill">Text: {approved.semantic_score:.3f}</span>
                            <span class="metric-pill">Aesthetic: {approved.aesthetic_score:.3f}</span>
                            <span class="metric-pill">Composite: <b>{approved.composite_score:.3f}</b></span>
                            <span class="source-badge {source_class}">{"SAMPLE" if approved.candidate.source == "fallback" else approved.candidate.source.upper()}</span>
                            """, unsafe_allow_html=True)
                            st.caption(f"**{approved.candidate.title[:70]}** ({approved.candidate.media_type})")

                            if approved.local_path:
                                if st.button(f"Copy to Clipboard", key=f"copy_{beat.index}"):
                                    res = copy_file_to_windows_clipboard(approved.local_path)
                                    if res.get("success"):
                                        st.toast(f"Copied to clipboard!")
                                st.caption(f"`{approved.local_path}`")
                        else:
                            st.warning("No candidate passed threshold.")

                    with col_media:
                        if approved and approved.local_path:
                            p = Path(approved.local_path)
                            if p.exists() and p.suffix.lower() == ".mp4":
                                st.video(str(p))
                            elif p.exists():
                                st.image(str(p), use_container_width=True)
                            else:
                                st.image(approved.candidate.preview_url, use_container_width=True)
                        elif approved:
                            st.image(approved.candidate.preview_url, use_container_width=True)
                        else:
                            st.info("No media")
                    st.divider()

        if st.session_state.trace_logs:
            with st.expander("Execution Log"):
                st.code("\n".join(st.session_state.trace_logs), language="text")

    # ═══════════════════════════════════════════════════════════
    # TAB: LIBRARY (Project Resources)
    # ═══════════════════════════════════════════════════════════
    if _page == "Library":
        st.markdown('<div class="section-title">Library</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Every image and asset saved in this project, visible on the page.</div>', unsafe_allow_html=True)

        resources = db.get_resources(project.id)
        if not resources:
            st.info("No resources yet. Create images, search for b-roll or run the pipeline to add assets.")
        else:
            # Filter
            types = sorted(set(r.resource_type for r in resources))
            filter_type = st.selectbox("Filter by type", ["all"] + types, key="resource_filter")

            filtered = resources if filter_type == "all" else [r for r in resources if r.resource_type == filter_type]
            st.caption(f"Showing {len(filtered)} of {len(resources)} resources")

            def res_meta(res):
                if res.metadata:
                    try:
                        parsed = json.loads(res.metadata)
                        return parsed if isinstance(parsed, dict) else {}
                    except (json.JSONDecodeError, TypeError):
                        return {}
                return {}

            def res_preview(res, meta):
                preview_url = meta.get("preview_url", "")
                if preview_url:
                    return preview_url
                if res.file_path and Path(res.file_path).exists():
                    p = Path(res.file_path)
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                        return str(p)
                return None

            # Visual grid for assets that have a preview
            with_preview = [r for r in filtered if res_preview(r, res_meta(r))]
            without_preview = [r for r in filtered if not res_preview(r, res_meta(r))]

            cols_per_row = 3
            for row_start in range(0, len(with_preview), cols_per_row):
                cols = st.columns(cols_per_row)
                for col_idx, col in enumerate(cols):
                    idx = row_start + col_idx
                    if idx >= len(with_preview):
                        break
                    res = with_preview[idx]
                    meta = res_meta(res)
                    with col:
                        source = meta.get("source", "")
                        source_class = f"source-{source.lower()}" if source else "source-fallback"
                        if source:
                            st.markdown(f'<span class="source-badge {source_class}">{source.upper()}</span>',
                                        unsafe_allow_html=True)
                        try:
                            st.image(res_preview(res, meta), use_container_width=True)
                        except Exception:
                            st.info("Preview unavailable")
                        st.markdown(f"**{res.title[:60] or 'Untitled'}**")
                        st.caption(f"`{res.resource_type}`")
                        a1, a2 = st.columns(2)
                        with a1:
                            if res.file_path and Path(res.file_path).exists():
                                if st.button("📋 Copy", key=f"clip_res_{res.id}", help="Copy to clipboard",
                                             use_container_width=True):
                                    copy_file_to_windows_clipboard(res.file_path)
                                    st.toast("Copied!")
                        with a2:
                            if st.button("🗑 Delete", key=f"del_res_{res.id}", help="Delete resource",
                                         use_container_width=True):
                                db.delete_resource(res.id)
                                st.rerun()
                        if res.source_url:
                            st.caption(f"[View Original]({res.source_url})")

            # Compact rows for assets without a preview
            for res in without_preview:
                meta = res_meta(res)
                source = meta.get("source", "")
                st.markdown(f"**{res.title[:60] or 'Untitled'}** — `{res.resource_type}`")
                if source:
                    st.caption(f"Source: {source}")
                if res.file_path:
                    st.caption(f"File: `{res.file_path}`")
                if res.source_url:
                    st.caption(f"[View Original]({res.source_url})")
                if st.button("🗑", key=f"del_res_{res.id}", help="Delete resource"):
                    db.delete_resource(res.id)
                    st.rerun()
                st.divider()

    # ═══════════════════════════════════════════════════════════
    # TAB: ACTIVITY LOG (Chat History)
    # ═══════════════════════════════════════════════════════════
    if _page == "Activity":
        st.markdown('<div class="section-title">Activity Log</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">All actions and events in this project.</div>', unsafe_allow_html=True)

        messages = db.get_messages(project.id)
        if not messages:
            st.info("No activity yet.")
        else:
            for msg in messages:
                icon = {"user": "🧑", "assistant": "🤖", "system": "⚙️"}.get(msg.role, "💬")
                with st.chat_message(msg.role, avatar=icon):
                    st.markdown(msg.content)
                    ts = msg.created_at[:19].replace("T", " ") if msg.created_at else ""
                    st.caption(ts)

        # Manual note input
        note = st.chat_input("Add a note to this project...", key="chat_note_input")
        if note:
            db.add_message(project.id, "user", note)
            st.rerun()

        if messages and st.button("Clear Activity Log", key="clear_log"):
            db.clear_messages(project.id)
            st.rerun()

    # ═══════════════════════════════════════════════════════════
    # TAB: PROJECT SETTINGS (Custom Instructions)
    # ═══════════════════════════════════════════════════════════
    if _page == "Settings":
        st.markdown('<div class="section-title">Project Settings</div>', unsafe_allow_html=True)

        # Project info
        with st.form("project_info_form"):
            edit_name = st.text_input("Project Name", value=project.name)
            edit_desc = st.text_area("Description", value=project.description, height=80)
            if st.form_submit_button("Update Info"):
                db.update_project(project.id, name=edit_name, description=edit_desc)
                st.toast("Project info updated.")
                st.rerun()

        st.markdown("---")

        # Custom Instructions
        st.markdown('<div class="section-title">Custom Instructions</div>', unsafe_allow_html=True)
        st.caption(
            "Rules and preferences the agent follows for this project. "
            "Similar to custom instructions in ChatGPT or system prompts. "
            "Example: 'Prefer warm color tones', 'Only landscape orientation', "
            "'Avoid stock photos with visible watermarks'."
        )
        with st.form("custom_instructions_form"):
            custom_instr = st.text_area(
                "Custom Instructions",
                value=project.custom_instructions,
                height=200,
                placeholder=(
                    "Examples:\n"
                    "- Prefer cinematic, warm-toned visuals\n"
                    "- Only landscape orientation (16:9)\n"
                    "- No close-up faces\n"
                    "- Prioritize Pinterest results over Pexels\n"
                    "- Style: dark moody editorial"
                ),
            )
            if st.form_submit_button("Save Instructions"):
                db.update_project(project.id, custom_instructions=custom_instr)
                db.add_message(project.id, "system", "Custom instructions updated.")
                st.toast("Custom instructions saved.")
                st.rerun()

        st.markdown("---")

        # Agent Memory
        st.markdown('<div class="section-title">Agent Memory</div>', unsafe_allow_html=True)
        st.caption("What the agent remembers about this project.")
        memory = db.get_all_memory(project.id)
        if memory:
            for k, v in memory.items():
                with st.expander(f"`{k}`"):
                    st.code(v, language="json")
                    if st.button(f"Delete '{k}'", key=f"del_mem_{k}"):
                        db.delete_memory(project.id, k)
                        st.rerun()
        else:
            st.info("No agent memory entries yet. Memory builds as you use the agent.")

        st.markdown("---")

        # Danger zone
        st.markdown('<div class="section-title">Danger Zone</div>', unsafe_allow_html=True)
        if st.button("Delete This Project", type="secondary", key="delete_project_btn"):
            st.session_state["confirm_delete"] = True

        if st.session_state.get("confirm_delete"):
            st.warning(f"This will permanently delete '{project.name}' and all its data.")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes, Delete", type="primary", key="confirm_del_yes"):
                    db.delete_project(project.id)
                    st.session_state.active_project_id = None
                    st.session_state["confirm_delete"] = False
                    st.rerun()
            with col_no:
                if st.button("Cancel", key="confirm_del_no"):
                    st.session_state["confirm_delete"] = False
                    st.rerun()
