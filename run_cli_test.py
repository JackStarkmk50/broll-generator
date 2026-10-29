import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

# Windows UTF-8 console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from config.settings import settings
from core.audio_aligner import AudioBeatAligner
from core.multimodal_embedder import MultimodalEmbedder
from core.aesthetic_memory import AestheticMemory
from core.clipboard_helper import copy_file_to_windows_clipboard
from core.pinterest_searcher import PinterestSearcher
from agent.stock_sourcer import StockSourcer
from agent.director_evaluator import DirectorEvaluator
from agent.workflow import BRollAgentGraph
from db.database import get_db


def run_test():
    print("=" * 60)
    print("0. Database Init")
    print("=" * 60)
    db = get_db()
    project = db.create_project("CLI Test Project", "Automated test run")
    print(f"Created project: {project.name} ({project.id})")
    db.add_message(project.id, "system", "CLI test started.")

    print("\n" + "=" * 60)
    print("1. Phoneme & Narrative Beat Alignment")
    print("=" * 60)
    sample_audio = settings.BASE_DIR / "sample_inputs" / "sample_voiceover.wav"
    aligner = AudioBeatAligner(model_size="base")
    words = aligner.transcribe_and_align(sample_audio)
    beats = aligner.segment_into_narrative_beats(words)
    print(f"Extracted {len(words)} aligned words into {len(beats)} narrative beats:")
    for b in beats:
        print(f"  Beat #{b.index:02d} [{b.formatted_range}] ({b.duration:.1f}s): {b.text}")

    print("\n" + "=" * 60)
    print("2. Pinterest Search Test")
    print("=" * 60)
    pinterest = PinterestSearcher()
    pins = pinterest.search("cinematic mountain landscape", limit=5)
    print(f"Pinterest returned {len(pins)} results")
    for pin in pins[:3]:
        print(f"  - {pin.title[:60]} | {pin.image_url[:80]}")

    print("\n" + "=" * 60)
    print("3. Aesthetic Memory Ingestion (Qdrant)")
    print("=" * 60)
    embedder = MultimodalEmbedder()
    aesthetic_mem = AestheticMemory(embedder=embedder)
    res = aesthetic_mem.ingest_reference_folder(settings.REFERENCE_ASSETS_DIR)
    print(f"Qdrant result: {res}")

    print("\n" + "=" * 60)
    print("4. Agent Pipeline (Pinterest + Pexels)")
    print("=" * 60)
    sourcer = StockSourcer()
    director = DirectorEvaluator(embedder=embedder, aesthetic_memory=aesthetic_mem)

    agent = BRollAgentGraph(
        sourcer=sourcer,
        director=director,
        embedder=embedder,
        aesthetic_memory=aesthetic_mem,
        threshold=settings.SIMILARITY_THRESHOLD,
        log_callback=lambda msg: print(f"  [TRACE] {msg}"),
    )

    for beat in beats[:2]:
        print(f"\n--- Beat #{beat.index} ---")
        text_emb = embedder.embed_text(beat.text)
        final_state = agent.process_beat(beat, text_emb)
        if final_state.approved_asset:
            cand = final_state.approved_asset.candidate
            print(f"Approved: {cand.title} (source: {cand.source})")
            print(f"  File: {final_state.approved_asset.local_path}")
            print(f"  Score: {final_state.approved_asset.composite_score:.3f}")

            # Save to project resources
            if final_state.approved_asset.local_path:
                db.add_resource(project.id, cand.media_type, cand.title,
                                file_path=final_state.approved_asset.local_path,
                                source_url=cand.source_url)
        else:
            print("No asset approved.")

    # Verify DB
    resources = db.get_resources(project.id)
    messages = db.get_messages(project.id)
    print(f"\nProject '{project.name}': {len(resources)} resources, {len(messages)} messages")

    print("\n" + "=" * 60)
    print("SUCCESS: Full Pipeline Test Completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_test()
