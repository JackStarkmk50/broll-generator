import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import wave
import math
from config.settings import settings

def create_sample_reference_images():
    """Generates sample aesthetic reference mood board images."""
    ref_dir = settings.REFERENCE_ASSETS_DIR
    ref_dir.mkdir(parents=True, exist_ok=True)

    presets = [
        ("cinematic_nature.jpg", (20, 45, 60), (45, 120, 110), "CINEMATIC MOUNTAIN MOOD"),
        ("cyberpunk_neon.jpg", (25, 10, 45), (140, 20, 120), "NEON URBAN CYBER AESTHETIC"),
        ("studio_workspace.jpg", (30, 35, 45), (90, 100, 120), "CREATIVE WORKSPACE EDITING"),
    ]

    for filename, col1, col2, text in presets:
        file_path = ref_dir / filename
        if not file_path.exists():
            img = Image.new("RGB", (800, 600), color=col1)
            draw = ImageDraw.Draw(img)
            # Create gradient effect
            for y in range(600):
                r = int(col1[0] + (col2[0] - col1[0]) * (y / 600))
                g = int(col1[1] + (col2[1] - col1[1]) * (y / 600))
                b = int(col1[2] + (col2[2] - col1[2]) * (y / 600))
                draw.line([(0, y), (800, y)], fill=(r, g, b))
            
            # Simple geometric shapes
            draw.rectangle([100, 150, 700, 450], outline=(255, 255, 255, 120), width=3)
            draw.text((160, 280), text, fill=(255, 255, 255))
            img.save(file_path, "JPEG", quality=90)
            print(f"Created reference asset: {file_path}")

def create_sample_audio_wav():
    """Generates a clean synthetic WAV file for testing speech ingestion."""
    inputs_dir = settings.BASE_DIR / "sample_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    wav_path = inputs_dir / "sample_voiceover.wav"
    
    if wav_path.exists():
        return

    # 16-bit mono 16000Hz PCM WAV (10 seconds)
    sample_rate = 16000
    duration_sec = 10.0
    num_samples = int(sample_rate * duration_sec)

    # Tone bursts representing speech cadences
    audio_data = bytearray()
    for i in range(num_samples):
        t = i / sample_rate
        # Intermittent tone bursts resembling spoken syllables
        envelope = (math.sin(2 * math.pi * 1.5 * t) > 0.1) * (0.5 + 0.5 * math.sin(2 * math.pi * 0.2 * t))
        val = int(envelope * 8000.0 * math.sin(2 * math.pi * 220.0 * t))
        val = max(-32767, min(32767, val))
        audio_data.extend(int(val).to_bytes(2, byteorder="little", signed=True))

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)

    print(f"Created sample audio voiceover: {wav_path}")

if __name__ == "__main__":
    create_sample_reference_images()
    create_sample_audio_wav()
