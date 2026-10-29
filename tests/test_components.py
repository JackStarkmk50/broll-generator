import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
import json
import numpy as np

from config.settings import settings
from core.audio_aligner import AudioBeatAligner, NarrativeBeat, WordTimestamp
from core.multimodal_embedder import MultimodalEmbedder
from core.aesthetic_memory import AestheticMemory
from core.clipboard_helper import copy_file_to_windows_clipboard
from core.pinterest_searcher import PinterestSearcher
from agent.stock_sourcer import StockSourcer
from agent.director_evaluator import DirectorEvaluator
from agent.workflow import BRollAgentGraph
from db.database import Database


def test_database():
    """Test SQLite database CRUD operations."""
    import tempfile
    db = Database(db_path=Path(tempfile.mktemp(suffix=".db")))

    # Create project
    proj = db.create_project("Test Project", "desc", "Use warm tones")
    assert proj.id
    assert proj.name == "Test Project"
    assert proj.custom_instructions == "Use warm tones"

    # List
    projects = db.list_projects()
    assert len(projects) >= 1

    # Update
    db.update_project(proj.id, name="Updated Name")
    updated = db.get_project(proj.id)
    assert updated.name == "Updated Name"

    # Messages
    db.add_message(proj.id, "user", "Hello")
    db.add_message(proj.id, "assistant", "Found 5 results")
    msgs = db.get_messages(proj.id)
    assert len(msgs) == 2

    # Resources
    db.add_resource(proj.id, "image", "Test Image", source_url="https://example.com/img.jpg")
    resources = db.get_resources(proj.id)
    assert len(resources) == 1

    # Memory
    db.set_memory(proj.id, "last_search", "mountains")
    val = db.get_memory(proj.id, "last_search")
    assert val == "mountains"

    # Settings
    db.set_setting("api_key", "test123")
    assert db.get_setting("api_key") == "test123"

    # Delete project (cascades)
    db.delete_project(proj.id)
    assert db.get_project(proj.id) is None


def test_audio_alignment_and_beats():
    aligner = AudioBeatAligner()
    words = [
        WordTimestamp(word="Welcome", start=0.0, end=0.4),
        WordTimestamp(word="to", start=0.4, end=0.6),
        WordTimestamp(word="the", start=0.6, end=0.8),
        WordTimestamp(word="future", start=0.8, end=1.5),
        WordTimestamp(word="of", start=1.5, end=1.7),
        WordTimestamp(word="editing.", start=1.7, end=2.8),
        WordTimestamp(word="Autonomous", start=3.5, end=4.2),
        WordTimestamp(word="agents", start=4.2, end=4.8),
        WordTimestamp(word="curate", start=4.8, end=5.3),
        WordTimestamp(word="footage.", start=5.3, end=6.2),
    ]
    beats = aligner.segment_into_narrative_beats(words, min_beat_duration=2.0)
    assert len(beats) >= 2
    assert "editing." in beats[0].text
    assert beats[0].duration > 0
    assert len(beats[0].search_keywords) > 0


def test_multimodal_embedder():
    embedder = MultimodalEmbedder()
    text_vec = embedder.embed_text("cinematic drone shot over misty mountains")
    assert isinstance(text_vec, np.ndarray)
    assert text_vec.shape == (512,)
    norm = np.linalg.norm(text_vec)
    assert np.isclose(norm, 1.0, atol=1e-3)

    sim = MultimodalEmbedder.cosine_similarity(text_vec, text_vec)
    assert np.isclose(sim, 1.0, atol=1e-3)


def test_aesthetic_memory():
    embedder = MultimodalEmbedder()
    memory = AestheticMemory(embedder=embedder)
    res = memory.ingest_reference_folder(settings.REFERENCE_ASSETS_DIR)
    assert res["status"] == "success"
    assert res["ingested_count"] > 0
    assert memory.aesthetic_centroid is not None


def test_sourcer():
    sourcer = StockSourcer()
    queries = sourcer.generate_search_queries(
        "technology empowers creators to build stories",
        ["technology", "creators", "stories"],
    )
    assert len(queries) > 0

    candidates = sourcer.search_all(queries[0], per_page=2)
    assert len(candidates) > 0
    assert all(hasattr(c, "source") for c in candidates)


def test_sourcer_and_director():
    sourcer = StockSourcer()
    candidates = sourcer.search_all("technology creators", per_page=2)
    assert len(candidates) > 0

    embedder = MultimodalEmbedder()
    memory = AestheticMemory(embedder=embedder)
    memory.ingest_reference_folder(settings.REFERENCE_ASSETS_DIR)

    director = DirectorEvaluator(embedder=embedder, aesthetic_memory=memory)
    text_emb = embedder.embed_text("creators build stories")
    evaluated = director.evaluate(candidates[0], text_emb, threshold=0.60)
    assert evaluated.composite_score >= -1.0
    assert evaluated.status in ("approved", "rejected")


def test_pinterest_searcher_init():
    """Pinterest searcher instantiation (no live search — may be blocked)."""
    searcher = PinterestSearcher()
    assert searcher is not None


def test_clipboard_helper():
    test_file = settings.REFERENCE_ASSETS_DIR / "cinematic_nature.jpg"
    if test_file.exists():
        res = copy_file_to_windows_clipboard(test_file)
        assert res["success"] is True


if __name__ == "__main__":
    print("Running tests...")
    test_database()
    print("  database: OK")
    test_audio_alignment_and_beats()
    print("  audio alignment: OK")
    test_multimodal_embedder()
    print("  embedder: OK")
    test_aesthetic_memory()
    print("  aesthetic memory: OK")
    test_sourcer()
    print("  sourcer: OK")
    test_sourcer_and_director()
    print("  sourcer + director: OK")
    test_pinterest_searcher_init()
    print("  pinterest searcher: OK")
    test_clipboard_helper()
    print("  clipboard: OK")
    print("All tests passed!")
