import sys
from pathlib import Path

# Ensure project root is in sys.path so local packages (config, db, core, agent) resolve
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import time
import json
from datetime import datetime
from uuid import uuid4
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
from agent.stock_sourcer import StockSourcer
from agent.director_evaluator import DirectorEvaluator
from agent.workflow import BRollAgentGraph

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Edit Helper",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #151a20; color: #f1f4f8; }
    .stApp { background-color: #151a20; }
    [data-testid="stSidebar"] { background-color: #1c232c; }
    .block-container { max-width: 1160px; padding-top: 2.5rem; }
    h1, h2, h3 { letter-spacing: -0.035em; }
    div.stButton > button[kind="primary"] { background: #6daee8; color: #101820; border: 0; }
    div.stButton > button[kind="primary"]:hover { background: #91c5f2; color: #101820; }

    .hero-container {
        padding: 0 0 22px;
        border-bottom: 1px solid #35404c;
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: 2rem; font-weight: 700; color: #f1f4f8;
        margin: 0;
    }
    .hero-sub { color: #94a3b8; font-size: 0.95rem; margin-top: 4px; }

    .project-card {
        background: rgba(30,41,59,0.6); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;
        cursor: pointer; transition: border-color 0.2s;
    }
    .project-card:hover { border-color: rgba(99,102,241,0.5); }

    .timestamp-badge {
        display: inline-block; background: rgba(56,189,248,0.15); color: #38bdf8;
        font-weight: 700; font-size: 0.85rem; padding: 3px 10px;
        border-radius: 16px; border: 1px solid rgba(56,189,248,0.3); margin-right: 6px;
    }
    .duration-badge {
        display: inline-block; background: rgba(168,85,247,0.15); color: #c084fc;
        font-weight: 600; font-size: 0.8rem; padding: 3px 8px;
        border-radius: 16px; border: 1px solid rgba(168,85,247,0.3);
    }
    .transcript-box {
        background: rgba(15,23,42,0.7); border-left: 4px solid #6366f1;
        padding: 10px 14px; border-radius: 0 8px 8px 0; margin: 10px 0 14px 0;
        font-size: 1rem; line-height: 1.4; color: #e2e8f0;
    }
    .metric-pill {
        display: inline-block; font-size: 0.78rem; padding: 2px 7px;
        border-radius: 5px; margin-right: 5px;
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    }
    .source-badge {
        display: inline-block; font-size: 0.72rem; padding: 2px 8px;
        border-radius: 10px; font-weight: 600; margin-left: 4px;
    }
    .source-pinterest { background: rgba(230,0,35,0.2); color: #e60023; border: 1px solid rgba(230,0,35,0.3); }
    .source-pexels { background: rgba(5,185,125,0.2); color: #05b97d; border: 1px solid rgba(5,185,125,0.3); }
    .source-fallback { background: rgba(255,255,255,0.1); color: #94a3b8; border: 1px solid rgba(255,255,255,0.15); }
</style>
""", unsafe_allow_html=True)

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


# ─── Sidebar: Projects ───────────────────────────────────────
with st.sidebar:
    st.markdown("### Edit Helper")
    st.caption("Images for your edit")
    st.markdown("---")

    # New project button
    if st.button("+ New Project", use_container_width=True, type="primary"):
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

    st.markdown("---")
    st.markdown("##### Your Projects")

    projects = db.list_projects()
    if not projects:
        st.caption("No projects yet. Create one above.")
    else:
        for proj in projects:
            is_active = st.session_state.active_project_id == proj.id
            label = f"{'📂' if is_active else '📁'} {proj.name}"
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
        pexels_input = st.text_input("Pexels API Key", value=saved_pexels_key, type="password",
                                     help="Free from pexels.com/api. Also loadable from .env file.")
        if pexels_input != saved_pexels_key:
            db.set_setting("pexels_api_key", pexels_input)

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
            gemini_input = st.text_input("Gemini API Key", value=saved_gemini_key, type="password",
                                         help="Free from aistudio.google.com/apikey. Also loadable from .env file.")
            if gemini_input != saved_gemini_key:
                db.set_setting("gemini_api_key", gemini_input)

            model_options = list(GEMINI_MODELS.keys())
            model_labels = list(GEMINI_MODELS.values())
            saved_gemini_model = db.get_setting("gemini_model", "gemini-2.5-flash")
            default_idx = model_options.index(saved_gemini_model) if saved_gemini_model in model_options else 0

            gemini_model = st.radio(
                "Gemini Model", options=model_options,
                index=default_idx,
                format_func=lambda x: GEMINI_MODELS.get(x, x),
                key="gemini_model_radio",
            )
            if gemini_model != saved_gemini_model:
                db.set_setting("gemini_model", gemini_model)


def get_pexels_key() -> str:
    return db.get_setting("pexels_api_key", "") or settings.PEXELS_API_KEY


def get_gemini_key() -> str:
    return db.get_setting("gemini_api_key", "") or settings.GEMINI_API_KEY


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


# ─── Main Content ─────────────────────────────────────────────
project = get_active_project()

if project is None:
    # Landing page
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">Make images for your video</h1>
        <p class="hero-sub">
            Upload a video. Review image ideas based on its audio. Save the images you want for your edit.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info("Create a project in the sidebar to get started.")

    # Show recent activity
    if projects:
        st.markdown("### Recent Projects")
        for proj in projects[:5]:
            resource_count = len(db.get_resources(proj.id))
            st.markdown(f"**{proj.name}** — {resource_count} resources — {proj.description[:80] if proj.description else 'No description'}")

else:
    # ─── Project Header ──────────────────────────────────────
    st.markdown(f"""
    <div class="hero-container">
        <h1 class="hero-title">{project.name}</h1>
        <p class="hero-sub">{project.description or 'No description'}</p>
    </div>
    """, unsafe_allow_html=True)

    # ─── Tabs ─────────────────────────────────────────────────
    tab_create, tab_resources, tab_search, tab_pipeline, tab_chat, tab_settings = st.tabs([
        "Create images", "Library", "Find stock", "Stock pipeline", "Activity", "Settings"
    ])

    with tab_create:
        st.subheader("Create images from a video")
        st.caption("Gemini listens to the video's audio, suggests visuals, and generates images for selected moments.")

        gemini_key = get_gemini_key()
        if not gemini_key:
            key_input = st.text_input("Gemini API key", type="password", key="create_gemini_key")
            if key_input:
                db.set_setting("gemini_api_key", key_input.strip())
                gemini_key = key_input.strip()

        source_video = st.file_uploader(
            "Video or audio file",
            type=["mp4", "mov", "mkv", "avi", "webm", "mp3", "wav", "flac", "m4a"],
            key="create_video_upload",
        )
        st.caption("The audio is used to understand the narration. Generated images are saved in this project's library.")

        analysis_key = f"image_analysis_{project.id}"
        images_key = f"generated_images_{project.id}"
        if st.button("Analyze audio", type="primary", disabled=not (gemini_key and source_video), key="analyze_video"):
            try:
                from core.audio_aligner import extract_audio

                suffix = Path(source_video.name).suffix.lower()
                media_path = settings.TEMP_DIR / f"{uuid4().hex}{suffix}"
                media_path.write_bytes(source_video.getbuffer())
                with st.spinner("Extracting and analyzing audio..."):
                    audio_path = extract_audio(media_path)
                    translator = build_translator(project.custom_instructions)
                    analysis = translator.transcribe_audio(str(audio_path), transcribe_model="gemini-3.8-flash")
                if not analysis or not analysis.segments:
                    st.error("Gemini could not find usable speech. Try a video with clear narration or dialogue.")
                else:
                    st.session_state[analysis_key] = analysis
                    st.session_state[f"image_source_{project.id}"] = source_video.name
                    st.session_state[images_key] = []
                    db.add_message(project.id, "assistant", f"Analyzed {source_video.name}: {len(analysis.segments)} audio moments.")
                    st.rerun()
            except Exception as exc:
                st.error(f"Audio analysis failed: {exc}")

        analysis = st.session_state.get(analysis_key)
        if source_video and st.session_state.get(f"image_source_{project.id}") != source_video.name:
            analysis = None
        if analysis and analysis.segments:
            st.markdown(f"#### {len(analysis.segments)} moments found")
            st.caption("Review the transcript and visual ideas before generating. Edit any image prompt below.")
            image_count = st.number_input(
                "Images to create", min_value=1, max_value=min(12, len(analysis.segments)),
                value=min(3, len(analysis.segments)), key=f"image_count_{project.id}",
            )
            count = int(image_count)
            indices = [round(i * (len(analysis.segments) - 1) / max(count - 1, 1)) for i in range(count)]
            if count == 1:
                indices = [len(analysis.segments) // 2]

            prompts = []
            for index in indices:
                segment = analysis.segments[index]
                scene = segment.visual_description or segment.english_translation or segment.original_text
                with st.expander(f"{segment.start_time:.0f}–{segment.end_time:.0f}s  {segment.original_text[:90]}", expanded=True):
                    if segment.english_translation and segment.english_translation != segment.original_text:
                        st.caption(f"English: {segment.english_translation}")
                    prompt = st.text_area(
                        "Image prompt",
                        value=(f"Create a cinematic 16:9 still image for a video edit. Show {scene}. "
                               f"Keep the scene visually clear, with no text, logos, or watermarks. "
                               f"{project.custom_instructions}"),
                        key=f"scene_prompt_{project.id}_{index}",
                        height=110,
                    )
                    prompts.append((segment, prompt))

            if st.button("Generate images", type="primary", key="generate_video_images"):
                created = []
                with st.status("Generating images", expanded=True) as status:
                    for segment, prompt in prompts:
                        try:
                            output = settings.ASSETS_OUTPUT_DIR / project.id / f"{uuid4().hex}.png"
                            image_path = generate_image(gemini_key, prompt, output)
                            db.add_resource(
                                project_id=project.id, resource_type="image",
                                title=f"Generated image at {segment.start_time:.0f}s",
                                file_path=str(image_path),
                                metadata={"source": "Gemini", "start_time": segment.start_time,
                                          "end_time": segment.end_time, "prompt": prompt},
                            )
                            created.append((segment.start_time, str(image_path)))
                            st.write(f"Created image for {segment.start_time:.0f}s")
                        except Exception as exc:
                            st.error(f"Image at {segment.start_time:.0f}s failed: {exc}")
                    status.update(label=f"Created {len(created)} of {len(prompts)} images", state="complete", expanded=False)
                st.session_state[images_key] = created
                if created:
                    db.add_message(project.id, "assistant", f"Generated {len(created)} images from audio moments.")

        generated = st.session_state.get(images_key, [])
        if generated:
            st.markdown("#### Ready for your edit")
            for start_time, image_path in generated:
                if Path(image_path).exists():
                    preview, action = st.columns([3, 1])
                    with preview:
                        st.image(image_path, caption=f"{start_time:.0f}s", use_container_width=True)
                    with action:
                        with open(image_path, "rb") as image_file:
                            st.download_button("Download image", image_file, file_name=Path(image_path).name,
                                               mime="image/png", key=f"download_{Path(image_path).stem}")

    # ═══════════════════════════════════════════════════════════
    # TAB 1: SEARCH (Pinterest + Pexels quick search)
    # ═══════════════════════════════════════════════════════════
    with tab_search:
        st.markdown("### Search for B-Roll")
        st.caption("Search Pinterest and Pexels. The agent reasons about your query to find better visuals.")

        search_col1, search_col2, search_col3 = st.columns([3, 1, 1])
        with search_col1:
            search_query = st.text_input("Search query", placeholder="cinematic mountain landscape drone aerial",
                                         key="search_query_input")
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
                        st.markdown(
                            f'<span class="source-badge {source_class}">{cand.source.upper()}</span> '
                            f'<span style="color:#94a3b8;font-size:0.8rem;">{cand.media_type}</span>',
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
    # TAB 2: PIPELINE (Audio → Beats → Auto Source)
    # ═══════════════════════════════════════════════════════════
    with tab_pipeline:
        st.markdown("### Autonomous B-Roll Pipeline")
        st.caption("Upload audio/video, agent extracts narrative beats and auto-sources matching b-roll.")

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
        st.caption("Upload any video/audio file (no size limit). Or paste a local file path for very large files.")

        input_method = st.radio("Input method", ["Upload file", "Local file path", "Demo sample"],
                                horizontal=True, key="input_method")

        uploaded_file = None
        local_file_path = ""
        if input_method == "Upload file":
            uploaded_file = st.file_uploader(
                "Upload Video or Audio (mp4, mkv, avi, mov, mp3, wav, flac — any size)",
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
                if input_method == "Upload file" and uploaded_file is not None:
                    temp_input = settings.TEMP_DIR / uploaded_file.name
                    with open(temp_input, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    media_path = temp_input
                elif input_method == "Local file path" and local_file_path.strip():
                    media_path = Path(local_file_path.strip())
                    if not media_path.exists():
                        st.error(f"File not found: {media_path}")
                        st.session_state.is_processing = False
                        st.stop()
                else:
                    demo_path = settings.BASE_DIR / "sample_inputs" / "sample_voiceover.wav"
                    media_path = demo_path if demo_path.exists() else None

                if media_path is None:
                    st.error("No input file provided.")
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

                st.write(f"Found {len(beats)} narrative beats.")

                # Load models
                progress.progress(35, text="Loading CLIP & Aesthetic Memory...")
                embedder = MultimodalEmbedder()
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
            st.markdown("### Sequenced B-Roll Feed")
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
                                f'<div style="background:rgba(56,189,248,0.08);border-left:4px solid #38bdf8;'
                                f'padding:8px 12px;border-radius:0 8px 8px 0;margin-bottom:10px;'
                                f'font-size:0.9rem;color:#93c5fd;">'
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
                            <span class="source-badge {source_class}">{approved.candidate.source.upper()}</span>
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
    # TAB 3: RESOURCES
    # ═══════════════════════════════════════════════════════════
    with tab_resources:
        st.markdown("### Project Resources")

        resources = db.get_resources(project.id)
        if not resources:
            st.info("No resources yet. Search for b-roll or run the pipeline to add assets.")
        else:
            # Filter
            types = sorted(set(r.resource_type for r in resources))
            filter_type = st.selectbox("Filter by type", ["all"] + types, key="resource_filter")

            filtered = resources if filter_type == "all" else [r for r in resources if r.resource_type == filter_type]
            st.caption(f"Showing {len(filtered)} of {len(resources)} resources")

            for res in filtered:
                with st.container():
                    rcol1, rcol2, rcol3 = st.columns([2, 1, 0.5])
                    with rcol1:
                        meta = {}
                        if res.metadata:
                            try:
                                parsed = json.loads(res.metadata)
                                meta = parsed if isinstance(parsed, dict) else {}
                            except (json.JSONDecodeError, TypeError):
                                meta = {}
                        source = meta.get("source", "")
                        st.markdown(f"**{res.title[:60] or 'Untitled'}** — `{res.resource_type}`")
                        if source:
                            st.caption(f"Source: {source}")
                        if res.file_path:
                            st.caption(f"File: `{res.file_path}`")
                        if res.source_url:
                            st.caption(f"[View Original]({res.source_url})")

                    with rcol2:
                        preview_url = meta.get("preview_url", "")
                        if preview_url:
                            try:
                                st.image(preview_url, width=150)
                            except Exception:
                                pass
                        elif res.file_path and Path(res.file_path).exists():
                            p = Path(res.file_path)
                            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                                st.image(str(p), width=150)

                    with rcol3:
                        if res.file_path and Path(res.file_path).exists():
                            if st.button("📋", key=f"clip_res_{res.id}", help="Copy to clipboard"):
                                copy_file_to_windows_clipboard(res.file_path)
                                st.toast("Copied!")
                        if st.button("🗑", key=f"del_res_{res.id}", help="Delete resource"):
                            db.delete_resource(res.id)
                            st.rerun()

                    st.divider()

    # ═══════════════════════════════════════════════════════════
    # TAB 4: ACTIVITY LOG (Chat History)
    # ═══════════════════════════════════════════════════════════
    with tab_chat:
        st.markdown("### Activity Log")
        st.caption("All actions and events in this project.")

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
    # TAB 5: PROJECT SETTINGS (Custom Instructions)
    # ═══════════════════════════════════════════════════════════
    with tab_settings:
        st.markdown("### Project Settings")

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
        st.markdown("### Custom Instructions")
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
        st.markdown("### Agent Memory")
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
        st.markdown("### Danger Zone")
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
