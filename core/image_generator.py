from pathlib import Path

from google import genai


IMAGE_MODEL = "gemini-3.1-flash-image"


def generate_image(api_key: str, prompt: str, output_path: Path, max_output_tokens: int = None) -> Path:
    from google.genai import types
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(max_output_tokens=max_output_tokens) if max_output_tokens else None
    response = client.models.generate_content(model=IMAGE_MODEL, contents=[prompt], config=config)
    for part in response.parts or []:
        if part.inline_data and part.inline_data.data:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(part.inline_data.data)
            return output_path
    raise RuntimeError("Gemini did not return an image. Try a different scene description.")
