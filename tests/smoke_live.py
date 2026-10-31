"""
Bounded live smoke test: exactly 1 transcription + 2 image generations = 3 API
requests maximum. Aborts on the first failure. No retries exist anywhere in this
script or the called modules.

Requires GEMINI_API_KEY in the environment (collect via vault link, never chat/hardcode).
Run from the repo root:  python -m tests.smoke_live

Cost bound (official pricing, Oct 30 2026, promo rates through Dec 31 2026):
- Transcription, gemini-3.8-flash: ~570 input tokens + <=2048 output tokens
  -> worst case < $0.008 (free tier: $0).
- Images, gemini-3.1-flash-image at default 1K: ~70 input tokens + <=2560 output
  tokens each -> worst case $0.0384/image, realistic $0.0168/image.
- Worst case total < $0.09; approval cap $0.25 gives >2.5x margin.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MAX_REQUESTS = 3
TRANSCRIBE_MAX_OUT = 2048
IMAGE_MAX_OUT = 2560  # 1K image ~1120 tokens + text headroom
SAMPLE = Path("sample_inputs/sample_voiceover.wav")  # bundled 10.0 s clip, no user media

PROMPTS = [
    "Create a cinematic 16:9 still image for a video edit. Show a warm kitchen with two people "
    "cooking together, soft window light. Keep the scene visually clear, with no text, logos, or watermarks.",
    "Create a cinematic 16:9 still image for a video edit. Show a city apartment at dusk. "
    "Keep the scene visually clear, with no text, logos, or watermarks.",
]

_requests = 0


def counted(label: str) -> None:
    global _requests
    _requests += 1
    if _requests > MAX_REQUESTS:
        raise SystemExit(f"Refusing request #{_requests}: budget is {MAX_REQUESTS}.")
    print(f"[request {_requests}/{MAX_REQUESTS}] {label}", flush=True)


def main() -> None:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        sys.exit("GEMINI_API_KEY env var required. Collect via vault link; never paste in chat or files.")
    if not SAMPLE.exists():
        sys.exit(f"Bundled sample missing: {SAMPLE}")

    from core.llm_translator import GeminiBackend
    from core.image_generator import generate_image, IMAGE_MODEL

    counted(f"transcribe {SAMPLE.name} ({SAMPLE.stat().st_size} bytes) via gemini-3.8-flash, "
            f"max_output_tokens={TRANSCRIBE_MAX_OUT}")
    backend = GeminiBackend(api_key=key, model="gemini-3.8-flash")
    text = backend.transcribe_audio(str(SAMPLE), transcribe_model="gemini-3.8-flash",
                                    max_output_tokens=TRANSCRIBE_MAX_OUT)
    if not text:
        sys.exit("Transcription failed - aborting before any image call.")
    print(f"transcription ok: {len(text)} chars returned", flush=True)

    outdir = Path("temp_cache/smoke_live")
    outdir.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(PROMPTS, 1):
        counted(f"image {i}/{len(PROMPTS)} via {IMAGE_MODEL} (default 1K), max_output_tokens={IMAGE_MAX_OUT}")
        try:
            path = generate_image(key, prompt, outdir / f"smoke_{i}.png",
                                  max_output_tokens=IMAGE_MAX_OUT)
        except Exception as e:
            sys.exit(f"Image {i} failed - aborting, no further requests: {e}")
        print(f"image {i} saved: {path}", flush=True)

    print(f"DONE: {_requests} API requests used (budget {MAX_REQUESTS}).", flush=True)


if __name__ == "__main__":
    main()
