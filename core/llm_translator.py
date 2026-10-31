"""
LLM-powered translator and visual query generator.

Backends:
1. Ollama local (gemma4:e4b, etc.) — no API key, runs on GPU
2. Google Gemini API (free tier) — needs GEMINI_API_KEY

User picks backend via UI toggle. Both produce the same output format.
"""
import json
import re
import requests
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class TranslatedBeat:
    original_text: str
    language: str
    english_translation: str
    visual_description: str
    search_queries: List[str]
    mood: str = ""
    keywords_en: List[str] = field(default_factory=list)
    backend: str = ""


# ── Script detection ──────────────────────────────────────────
_SCRIPT_RANGES = {
    "tamil": range(0x0B80, 0x0BFF + 1),
    "hindi": range(0x0900, 0x097F + 1),
    "telugu": range(0x0C00, 0x0C7F + 1),
    "malayalam": range(0x0D00, 0x0D7F + 1),
    "kannada": range(0x0C80, 0x0CFF + 1),
    "arabic": range(0x0600, 0x06FF + 1),
    "cjk": range(0x4E00, 0x9FFF + 1),
    "korean": range(0xAC00, 0xD7AF + 1),
}


def detect_script(text: str) -> str:
    counts = {"latin": 0}
    for name in _SCRIPT_RANGES:
        counts[name] = 0
    for ch in text:
        cp = ord(ch)
        matched = False
        for name, rng in _SCRIPT_RANGES.items():
            if cp in rng:
                counts[name] += 1
                matched = True
                break
        if not matched and ch.isalpha() and cp < 0x0250:
            counts["latin"] += 1
    if not any(counts.values()):
        return "unknown"
    best = max(counts, key=counts.get)
    return "english" if best == "latin" else best


def is_non_english(text: str) -> bool:
    return detect_script(text) not in ("english", "latin", "unknown")


# ── Prompt ────────────────────────────────────────────────────
VISUAL_PROMPT = """You are a visual content assistant for a b-roll sourcing tool.

Given a transcript segment (possibly in Tamil, Hindi, or another language):
1. Detect the language
2. Translate the full text to English accurately
3. Understand the MEANING and CONTEXT of what is being said
4. Suggest what visual b-roll footage would complement this narration
5. Generate 5 specific search queries for finding relevant stock images/videos

Respond in this exact JSON format (no markdown, no code fences):
{
  "language": "tamil",
  "english_translation": "...",
  "visual_description": "This segment talks about ... so visuals should show ...",
  "mood": "emotional|dramatic|calm|energetic|professional|informative",
  "search_queries": ["query1", "query2", "query3", "query4", "query5"],
  "keywords_en": ["keyword1", "keyword2", "keyword3"]
}

IMPORTANT:
- Search queries must be in ENGLISH regardless of input language
- Queries should describe VISUALS, not abstract concepts
- Think about what a video editor would search for on Pinterest or stock sites
- Be specific: "mother comforting crying child" not "parenting"
- Include variety: wide shots, close-ups, abstract, literal"""


AUDIO_TRANSCRIBE_PROMPT = """You are an audio transcription and visual content assistant.

Listen to this audio and:
1. Transcribe the speech accurately (detect language automatically — Tamil, Hindi, English, etc.)
2. Split the transcription into segments of 3-6 seconds each based on natural pauses
3. Translate each segment to English
4. For each segment, generate visual search queries for b-roll footage

Respond in this exact JSON format (no markdown, no code fences):
{
  "language": "tamil",
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 5.5,
      "original_text": "...",
      "english_translation": "...",
      "visual_description": "...",
      "mood": "emotional",
      "search_queries": ["query1", "query2", "query3"]
    }
  ]
}

IMPORTANT:
- Estimate start_time and end_time in seconds
- Search queries must be ENGLISH, describe VISUALS
- Be specific: "mother comforting crying child" not "parenting"
- Include variety per segment: wide shots, close-ups, abstract, literal"""


@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    original_text: str
    english_translation: str
    visual_description: str = ""
    mood: str = ""
    search_queries: List[str] = field(default_factory=list)


@dataclass
class AudioTranscription:
    language: str
    segments: List[AudioSegment]
    backend: str = ""


def _parse_audio_response(response_text: str) -> Optional[AudioTranscription]:
    """Parse Gemini audio transcription response."""
    text = response_text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "segments" in data:
            segments = []
            for seg in data["segments"]:
                segments.append(AudioSegment(
                    start_time=float(seg.get("start_time", 0)),
                    end_time=float(seg.get("end_time", 0)),
                    original_text=seg.get("original_text", ""),
                    english_translation=seg.get("english_translation", ""),
                    visual_description=seg.get("visual_description", ""),
                    mood=seg.get("mood", ""),
                    search_queries=seg.get("search_queries", []),
                ))
            return AudioTranscription(
                language=data.get("language", "unknown"),
                segments=segments,
            )
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"[AudioTranscription] Parse error: {e}")
    return None


def _parse_dedicated_transcribe(response_text: str) -> Optional[AudioTranscription]:
    """Parse output from dedicated transcribe model (raw text, no translation)."""
    text = response_text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "segments" in data:
            segments = []
            for seg in data["segments"]:
                segments.append(AudioSegment(
                    start_time=float(seg.get("start_time", 0)),
                    end_time=float(seg.get("end_time", 0)),
                    original_text=seg.get("text", seg.get("original_text", "")),
                    english_translation="",
                ))
            return AudioTranscription(
                language=data.get("language", "unknown"),
                segments=segments,
            )
    except (json.JSONDecodeError, TypeError) as e:
        print(f"[DedicatedTranscribe] Parse error: {e}")
    return None


def _parse_json_response(response_text: str, original: str) -> Optional[TranslatedBeat]:
    text = response_text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    # Find JSON object
    json_match = re.search(r'\{[^{}]*"english_translation"[^{}]*\}', text, re.DOTALL)
    if json_match:
        text = json_match.group(0)

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "english_translation" in data:
            return TranslatedBeat(
                original_text=original,
                language=data.get("language", detect_script(original)),
                english_translation=data["english_translation"],
                visual_description=data.get("visual_description", ""),
                search_queries=data.get("search_queries", []),
                mood=data.get("mood", ""),
                keywords_en=data.get("keywords_en", []),
            )
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback extraction
    trans_match = re.search(r'"english_translation"\s*:\s*"([^"]+)"', response_text)
    if trans_match:
        queries = re.findall(r'"([^"]{10,60})"', response_text)
        queries = [q for q in queries if "translation" not in q.lower() and "language" not in q.lower()]
        return TranslatedBeat(
            original_text=original,
            language=detect_script(original),
            english_translation=trans_match.group(1),
            visual_description="",
            search_queries=queries[:5],
            keywords_en=[],
        )
    return None


# ── Ollama backend ────────────────────────────────────────────
OLLAMA_MODEL_PREFERENCE = [
    "gemma4:e4b", "gemma4:12b", "gemma3:4b",
    "llama3:latest", "qwen2.5:7b",
]


def list_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    """Return list of locally available Ollama model names."""
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
        if resp.status_code == 200:
            return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pass
    return []


class OllamaBackend:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._available = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            models = list_ollama_models(self.base_url)
            if not self.model:
                for pref in OLLAMA_MODEL_PREFERENCE:
                    if pref in models:
                        self.model = pref
                        break
                if not self.model and models:
                    self.model = models[0]
            self._available = bool(self.model and self.model in models)
            if self._available:
                print(f"[Ollama] Using local model: {self.model}")
            return self._available
        except Exception:
            pass
        self._available = False
        return False

    def generate(self, prompt: str) -> Optional[str]:
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 1024},
                },
                timeout=300,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "")
        except requests.exceptions.Timeout:
            print(f"[Ollama] Timeout (300s) for model {self.model}")
        except Exception as e:
            print(f"[Ollama] Error: {e}")
        return None


# ── Gemini backend (new google-genai SDK) ─────────────────────
GEMINI_MODELS = {
    "gemini-3.8-flash": "Gemini 3.8 Flash (latest)",
    "gemini-3.6-flash": "Gemini 3.6 Flash",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite (fastest)",
}

# Deprecated model ids Google auto-routes; map them to the official replacement
# instead of silently falling back to the newest model.
DEPRECATED_MODEL_MAP = {
    "gemini-3.5-flash": "gemini-3.6-flash",
}

GEMINI_TRANSCRIBE_MODELS = {
    "gemini-3.5-transcribe": "Gemini 3.5 Transcribe (dedicated, fastest)",
    "gemini-3.8-flash": "Gemini 3.8 Flash (all-in-one: transcribe + translate + queries)",
    "gemini-3.6-flash": "Gemini 3.6 Flash (all-in-one)",
}


class GeminiBackend:
    def __init__(self, api_key: str, model: str = "gemini-3.8-flash"):
        self.api_key = api_key
        model = DEPRECATED_MODEL_MAP.get(model, model)
        self.model_name = model if model in GEMINI_MODELS else "gemini-3.8-flash"
        self._client = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _init(self):
        if self._client is not None:
            return
        from google import genai
        self._client = genai.Client(api_key=self.api_key)
        print(f"[Gemini] Initialized {self.model_name}")

    def generate(self, prompt: str) -> Optional[str]:
        try:
            self._init()
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            print(f"[Gemini] Error: {e}")
            return None

    def transcribe_audio(self, audio_path: str, custom_instructions: str = "",
                          transcribe_model: str = "", max_output_tokens: int = None) -> Optional[str]:
        """
        Send audio file to Gemini for transcription.
        If transcribe_model is a dedicated model (gemini-3.5-transcribe), returns raw transcript.
        Otherwise uses the all-in-one prompt for transcribe + translate + queries.
        """
        try:
            self._init()
            from google.genai import types
            import mimetypes
            from pathlib import Path as P

            path_obj = P(audio_path)
            mime = mimetypes.guess_type(str(path_obj))[0] or "audio/wav"
            size_mb = path_obj.stat().st_size / 1024 / 1024

            audio_bytes = path_obj.read_bytes()
            audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime)

            model_to_use = transcribe_model or self.model_name
            is_dedicated = "transcribe" in model_to_use

            if is_dedicated:
                prompt = (
                    "Transcribe this audio accurately. Detect the language automatically. "
                    "Include timestamps for each sentence/segment. "
                    "Output as JSON: {\"language\": \"...\", \"segments\": [{\"start_time\": 0.0, \"end_time\": 5.0, \"text\": \"...\"}]}"
                )
            else:
                prompt = AUDIO_TRANSCRIBE_PROMPT
                if custom_instructions:
                    prompt += f"\n\nUser style instructions:\n{custom_instructions}"

            print(f"[Gemini] Transcribing audio ({size_mb:.1f} MB) with {model_to_use}...")

            config = types.GenerateContentConfig(max_output_tokens=max_output_tokens) if max_output_tokens else None
            response = self._client.models.generate_content(
                model=model_to_use,
                contents=[audio_part, prompt],
                config=config,
            )
            return response.text
        except Exception as e:
            print(f"[Gemini] Audio transcription error: {e}")
            return None


# ── Main translator ───────────────────────────────────────────
class LLMTranslator:
    """
    Translates non-English transcript text and generates visual search queries.

    Args:
        backend: "local" (Ollama) or "api" (Gemini). Default "local".
        api_key: Gemini API key (only needed if backend="api").
        gemini_model: Which Gemini model to use.
        ollama_model: Which Ollama model to use (blank = auto-detect).
        ollama_url: Ollama server URL.
        custom_instructions: User style rules forwarded to LLM.
    """

    def __init__(
        self,
        backend: str = "local",
        api_key: str = "",
        gemini_model: str = "gemini-3.8-flash",
        ollama_model: str = "",
        ollama_url: str = "http://localhost:11434",
        custom_instructions: str = "",
    ):
        self.preferred_backend = backend
        self.custom_instructions = custom_instructions
        self._ollama = OllamaBackend(base_url=ollama_url, model=ollama_model)
        self._gemini = GeminiBackend(api_key=api_key, model=gemini_model) if api_key else None
        self._backend_name = ""

    def _get_backend(self):
        if self.preferred_backend == "api" and self._gemini and self._gemini.is_available():
            self._backend_name = f"gemini/{self._gemini.model_name}"
            return self._gemini
        if self.preferred_backend == "local" and self._ollama.is_available():
            self._backend_name = f"ollama/{self._ollama.model}"
            return self._ollama
        # Fallback: try the other one
        if self._ollama.is_available():
            self._backend_name = f"ollama/{self._ollama.model}"
            return self._ollama
        if self._gemini and self._gemini.is_available():
            self._backend_name = f"gemini/{self._gemini.model_name}"
            return self._gemini
        return None

    def translate_and_reason(self, text: str) -> TranslatedBeat:
        if not is_non_english(text):
            return TranslatedBeat(
                original_text=text, language="english",
                english_translation=text, visual_description="",
                search_queries=[], keywords_en=[], backend="skip",
            )

        backend = self._get_backend()
        if not backend:
            lang = detect_script(text)
            print(f"[LLMTranslator] No backend available. {lang} text goes to CLIP directly.")
            return TranslatedBeat(
                original_text=text, language=lang,
                english_translation=text, visual_description="",
                search_queries=[], keywords_en=[], backend="none",
            )

        prompt = VISUAL_PROMPT
        if self.custom_instructions:
            prompt += f"\n\nUser style instructions:\n{self.custom_instructions}"
        prompt += f'\n\nTranscript segment to process:\n"{text}"'

        print(f"[LLMTranslator] Translating via {self._backend_name}...")
        response = backend.generate(prompt)
        if response:
            result = _parse_json_response(response, text)
            if result:
                result.backend = self._backend_name
                return result
            print(f"[LLMTranslator] Parse failed, raw: {response[:200]}")

        lang = detect_script(text)
        return TranslatedBeat(
            original_text=text, language=lang,
            english_translation=text, visual_description="",
            search_queries=[], keywords_en=[], backend="failed",
        )

    def translate_beats(self, beat_texts: List[str]) -> List[TranslatedBeat]:
        return [self.translate_and_reason(text) for text in beat_texts]

    def transcribe_audio(self, audio_path: str, transcribe_model: str = "") -> Optional[AudioTranscription]:
        """
        Transcribe audio using Gemini.

        If transcribe_model is dedicated (gemini-3.5-transcribe):
          Step 1: transcribe with dedicated model → raw text segments
          Step 2: translate each segment with flash model (or Ollama) → English + queries

        If transcribe_model is a flash model (or empty):
          One call: audio → text + translation + queries all at once.
        """
        if not self._gemini or not self._gemini.is_available():
            print("[LLMTranslator] Gemini API required for audio transcription.")
            return None

        is_dedicated = "transcribe" in (transcribe_model or "")

        response = self._gemini.transcribe_audio(
            audio_path, self.custom_instructions, transcribe_model=transcribe_model
        )
        if not response:
            return None

        if is_dedicated:
            # Dedicated model returns raw transcript — need to translate segments
            raw = _parse_dedicated_transcribe(response)
            if not raw:
                print(f"[LLMTranslator] Dedicated transcribe parse failed: {response[:300]}")
                return None

            print(f"[LLMTranslator] Transcribed {len(raw.segments)} segments. Translating...")
            # Translate each segment using preferred backend
            for seg in raw.segments:
                if is_non_english(seg.original_text):
                    translated = self.translate_and_reason(seg.original_text)
                    seg.english_translation = translated.english_translation
                    seg.visual_description = translated.visual_description
                    seg.search_queries = translated.search_queries
                    seg.mood = translated.mood
                else:
                    seg.english_translation = seg.original_text
            raw.backend = f"gemini/{transcribe_model}+translate"
            return raw
        else:
            # All-in-one model
            result = _parse_audio_response(response)
            if result:
                result.backend = f"gemini/{transcribe_model or self._gemini.model_name}"
                return result
            print(f"[LLMTranslator] Audio parse failed: {response[:300]}")
        return None
