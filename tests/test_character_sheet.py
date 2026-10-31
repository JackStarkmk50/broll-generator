"""Tests for character-sheet prompt chaining and reference-image generation.

Pure tests - no network, no API keys, no real google-genai package.

Run from the repo root:  python -m tests.test_character_sheet
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.character_sheet import (
    CHARACTER_SHEET_PROMPT, draft_character_sheet, compose_scene_prompt, compose_compact_block,
)

SHEET = ("CHARACTERS:\n"
         "MOTHER: woman in her 40s, blue cotton saree, shoulder-length black hair in a braid.\n"
         "DAUGHTER: girl around 8, pink frock, two ponytails.\n"
         "STYLE ANCHOR: warm natural window light, soft tones, cinematic 16:9 still.")


def test_compose_includes_sheet_verbatim_in_every_scene():
    """The identical sheet text must land in every scene prompt - that is the chaining."""
    scenes = ["mother and daughter cooking side by side",
              "daughter on a phone call in a city apartment"]
    prompts = [compose_scene_prompt(s, SHEET) for s in scenes]
    for prompt, scene in zip(prompts, scenes):
        assert scene in prompt
        assert SHEET in prompt
        assert "no text, logos, or watermarks" in prompt
    # consistency: the sheet substring is byte-identical across prompts
    assert prompts[0].index(SHEET) >= 0 and prompts[1].index(SHEET) >= 0
    print("PASS: sheet is chained identically into every scene prompt")


def test_compose_without_sheet_keeps_plain_prompt():
    prompt = compose_scene_prompt("a quiet beach at dawn")
    assert "a quiet beach at dawn" in prompt
    assert "CHARACTERS" not in prompt
    print("PASS: no sheet -> plain prompt, no sheet block")


def test_compose_appends_custom_instructions():
    prompt = compose_scene_prompt("a market scene", SHEET, "prefer muted colors")
    assert "prefer muted colors" in prompt
    print("PASS: custom instructions appended")


def test_draft_sends_scenes_and_returns_text():
    seen = {}

    def fake_generate(prompt):
        seen["prompt"] = prompt
        return "  CHARACTERS: none\nSTYLE ANCHOR: cool dusk tones  "

    scenes = ["1. scene one", "2. scene two"]
    out = draft_character_sheet(fake_generate, scenes, "be terse")
    assert out == "CHARACTERS: none\nSTYLE ANCHOR: cool dusk tones"
    assert "1. scene one" in seen["prompt"] and "2. scene two" in seen["prompt"]
    assert "be terse" in seen["prompt"]
    print("PASS: draft sends scenes + custom instructions, strips response")


def test_draft_empty_response_returns_none():
    assert draft_character_sheet(lambda p: None, ["1. x"]) is None
    assert draft_character_sheet(lambda p: "   ", ["1. x"]) is None
    print("PASS: empty model response -> None (UI can fall back to manual editing)")


def _fake_genai_module(record):
    """Minimal google.genai stand-in capturing generate_content calls."""
    fake = types.ModuleType("google.genai")

    class _Part:
        def __init__(self, data=None):
            self.inline_data = types.SimpleNamespace(data=data) if data else None

        @staticmethod
        def from_bytes(data, mime_type):
            return _Part(data=data)

    class _Models:
        def generate_content(self, model, contents, config=None):
            record["model"] = model
            record["contents"] = contents
            return types.SimpleNamespace(parts=[_Part(data=b"\x89PNG fake")])

    class Client:
        def __init__(self, api_key=None):
            record["api_key"] = api_key
            self.models = _Models()

    fake.Client = Client
    fake.types = types.SimpleNamespace(
        Part=_Part, GenerateContentConfig=lambda **kw: kw or None)
    return fake


def test_generate_image_reference_chaining(tmp_path=Path("/tmp/_cs_test")):
    record = {}
    fake = _fake_genai_module(record)
    google_mod = sys.modules.setdefault("google", types.ModuleType("google"))
    sys.modules["google.genai"] = fake
    google_mod.genai = fake
    for mod in ("core.image_generator",):
        sys.modules.pop(mod, None)
    from core.image_generator import generate_image

    tmp_path.mkdir(exist_ok=True)
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"\x89PNG ref")
    out = tmp_path / "out.png"

    generate_image("KEY", "scene two", out, reference_image=ref)
    contents = record["contents"]
    assert len(contents) == 2, f"expected prompt + reference part, got {len(contents)}"
    assert "keep the same people" in contents[0]
    assert out.read_bytes() == b"\x89PNG fake"

    generate_image("KEY", "scene one", tmp_path / "first.png")
    assert len(record["contents"]) == 1, "first scene must not attach a reference"
    print("PASS: reference image chained from the previous scene; first scene plain")




def test_compact_block_sheet_once_scenarios_below():
    scenes = [("IMAGE 1 (0s)", "mother and daughter cooking side by side"),
              ("IMAGE 2 (31s)", "daughter on a phone call in a city apartment")]
    block = compose_compact_block(scenes, SHEET)
    assert block.count("MOTHER: woman in her 40s") == 1, "sheet must appear exactly once"
    assert "SETUP" not in block and "Then generate" not in block, "no meta-instruction lines"
    assert "IMAGE 1 (0s): mother and daughter cooking side by side" in block
    assert "IMAGE 2 (31s): daughter on a phone call in a city apartment" in block
    assert "Every image: cinematic 16:9 still, no text, logos, or watermarks." in block
    assert block.index("CHARACTERS") < block.index("IMAGE 1") < block.index("IMAGE 2")
    print("PASS: compact block has sheet once on top, scenario-only lines below, no meta text")


def test_compact_block_without_sheet_still_has_rules():
    block = compose_compact_block([("IMAGE 1 (0s)", "a quiet beach")])
    assert "CHARACTERS" not in block
    assert "Every image: cinematic 16:9 still" in block
    print("PASS: compact block works without a sheet")

if __name__ == "__main__":
    test_compose_includes_sheet_verbatim_in_every_scene()
    test_compose_without_sheet_keeps_plain_prompt()
    test_compose_appends_custom_instructions()
    test_draft_sends_scenes_and_returns_text()
    test_draft_empty_response_returns_none()
    test_generate_image_reference_chaining()
    test_compact_block_sheet_once_scenarios_below()
    test_compact_block_without_sheet_still_has_rules()
    print("ALL CHARACTER SHEET TESTS PASSED")
