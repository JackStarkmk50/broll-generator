"""Mocked audio-analysis test: the Create images transcription path with the
Gemini API boundary faked. Covers the transcribe call and response parsing
ONLY - not video/audio extraction (ffmpeg) and not image generation.
No real API key, network call, or user media.

Run:  python tests/test_mocked_audio_analysis.py
"""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm_translator import LLMTranslator

FAKE_RESPONSE = json.dumps({
    "language": "en",
    "segments": [
        {"start_time": 0.0, "end_time": 4.5,
         "original_text": "Welcome back to the vlog",
         "english_translation": "Welcome back to the vlog",
         "visual_description": "creator talking to camera at a desk",
         "mood": "upbeat", "search_queries": ["talking head desk vlog"]},
        {"start_time": 4.5, "end_time": 9.0,
         "original_text": "Today we review the new lens",
         "english_translation": "Today we review the new lens",
         "visual_description": "close-up of a camera lens",
         "mood": "curious", "search_queries": ["camera lens close up"]},
    ],
})


class _FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, model, contents, config=None):
        self.calls.append({"model": model, "config": config})
        return SimpleNamespace(text=FAKE_RESPONSE)


class _FakeClient:
    def __init__(self):
        self.models = _FakeModels()


def main():
    fake = _FakeClient()
    translator = LLMTranslator(backend="api", api_key="fake-key-for-test",
                               gemini_model="gemini-3.8-flash")
    translator._gemini._client = fake  # bypass genai.Client construction

    tmp = Path(tempfile.mkdtemp(prefix="edit-helper-flow-")) / "clip.wav"
    tmp.write_bytes(b"RIFF" + b"\0" * 128)  # dummy bytes; the fake API never reads them as audio

    result = translator.transcribe_audio(str(tmp), transcribe_model="gemini-3.8-flash")

    assert result is not None, "analysis should succeed with mocked Gemini"
    assert result.language == "en"
    assert len(result.segments) == 2
    seg = result.segments[0]
    assert seg.original_text == "Welcome back to the vlog"
    assert seg.search_queries == ["talking head desk vlog"]
    assert result.backend == "gemini/gemini-3.8-flash"
    assert fake.models.calls[0]["model"] == "gemini-3.8-flash", "chosen model must be used"
    print("PASS mocked audio analysis: audio -> Gemini(mock) -> parsed segments "
          f"({len(result.segments)} segments, backend={result.backend})")

    # no key -> analysis refuses instead of calling out
    translator2 = LLMTranslator(backend="api", api_key="", gemini_model="gemini-3.8-flash")
    assert translator2.transcribe_audio(str(tmp), transcribe_model="gemini-3.8-flash") is None
    print("PASS mocked audio analysis: missing key refuses cleanly, no API call")

    print("ALL MOCKED AUDIO-ANALYSIS CHECKS PASSED")


if __name__ == "__main__":
    main()
