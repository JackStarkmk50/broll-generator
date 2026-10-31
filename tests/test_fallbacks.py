"""Reproducible fallback-mode tests for the completion hardening (v13).

Runs without torch/open_clip, API keys, or network. These prove FALLBACK behavior
only - they say nothing about live-model quality.

Run from the repo root:  python -m tests.test_fallbacks
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.stock_sourcer import StockSourcer

# Dependency substitute: torch is absent here. multimodal_embedder imports it at
# module level, so install a minimal stand-in that supports only what the mock
# path touches; model loading itself is exercised through the module's own
# try/except chain (open_clip -> transformers -> mock).
if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401
    except ImportError:
        fake = types.ModuleType("torch")
        class _NoGrad:
            def __call__(self): return self
            def __enter__(self): return self
            def __exit__(self, *a): return False
        fake.no_grad = _NoGrad()
        fake.__version__ = "0.0-substitute"
        fake.cuda = types.SimpleNamespace(is_available=lambda: False)
        sys.modules["torch"] = fake

for _dep in ("cv2",):
    if _dep not in sys.modules:
        try:
            __import__(_dep)
        except ImportError:
            sys.modules[_dep] = types.ModuleType(_dep)

from core.multimodal_embedder import MultimodalEmbedder


def test_stock_sourcer_keyless_returns_only_fallback():
    """No Pexels key + Pinterest disabled: every candidate must be source='fallback'."""
    sourcer = StockSourcer(api_key="", enable_pinterest=False, enable_pexels=True)
    results = sourcer.search_all("cinematic mountain landscape", per_page=6)
    assert results, "fallback catalog should always return candidates"
    sources = {c.source for c in results}
    assert sources == {"fallback"}, f"expected only fallback sources, got {sources}"
    print(f"PASS: keyless search returned {len(results)} candidates, all source='fallback'")


def test_embedder_mock_mode_explicit():
    """Without torch/open_clip the embedder must be in mock mode and admit it."""
    emb = MultimodalEmbedder()
    assert emb.is_mock is True, "embedder should report mock mode when models unavailable"
    vec = emb.embed_text("a warm kitchen scene")
    assert vec.shape == (emb.embedding_dim,), f"unexpected embedding shape {vec.shape}"
    norm = float((vec ** 2).sum() ** 0.5)
    assert abs(norm - 1.0) < 1e-4, f"mock embedding should be normalized, norm={norm}"
    print(f"PASS: embedder is_mock=True, embed_text returns normalized {emb.embedding_dim}-d vector")


def test_embedder_mock_is_deterministic():
    emb = MultimodalEmbedder()
    a = emb.embed_text("same input")
    b = emb.embed_text("same input")
    assert (a == b).all(), "mock embeddings must be deterministic"
    print("PASS: mock embeddings deterministic")


if __name__ == "__main__":
    test_stock_sourcer_keyless_returns_only_fallback()
    test_embedder_mock_mode_explicit()
    test_embedder_mock_is_deterministic()
    print("ALL FALLBACK TESTS PASSED (fallback-mode evidence, not live-model proof)")
