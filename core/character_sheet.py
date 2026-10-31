"""Character sheet + prompt chaining for consistent B-roll image sequences.

One shared sheet - canonical character descriptions plus a style anchor - is
injected into every scene prompt, so recurring people (e.g. the same mother
and daughter) look the same from image to image. This works in both output
modes: images generated in-app, or prompts pasted into the user's own image
generator.
"""
from typing import Callable, List, Optional

CHARACTER_SHEET_PROMPT = """You are planning a sequence of still images for one video.

Scenes:
{scenes}

Write a short character sheet that keeps people consistent across every image.

CHARACTERS: one line per recurring person, with a fixed description - age, build, hair, skin tone, clothing, distinctive items. If no specific people recur, write "CHARACTERS: none".
STYLE ANCHOR: one line fixing the shared look - lighting, color palette, camera style, mood.

Rules: under 120 words, plain text, exactly these two section labels, no other commentary. Keep every description acceptable to mainstream AI image generators (no sexually explicit or nude imagery, graphic gore, or hateful depictions); only recast a description symbolically if it would breach those rules."""


def draft_character_sheet(
    generate_fn: Callable[[str], Optional[str]],
    scene_lines: List[str],
    custom_instructions: str = "",
) -> Optional[str]:
    """Ask the reasoning model once for a shared character sheet.

    generate_fn: plain-text completion callable (LLMTranslator.generate_text).
    scene_lines: numbered scene descriptions, one per selected moment.
    Returns the sheet text, or None when the model returns nothing usable.
    """
    prompt = CHARACTER_SHEET_PROMPT.format(scenes="\n".join(scene_lines))
    if custom_instructions.strip():
        prompt += f"\n\nUser style instructions:\n{custom_instructions.strip()}"
    response = generate_fn(prompt)
    if not response:
        return None
    text = response.strip()
    return text or None


def compose_scene_prompt(scene: str, sheet: str = "", custom_instructions: str = "") -> str:
    """Build the final prompt for one scene, with the character sheet chained in.

    The sheet text is identical across scenes - that repetition is what keeps
    recurring characters consistent from image to image.
    """
    parts = [f"Create a cinematic 16:9 still image for a video edit. Show {scene}."]
    if sheet.strip():
        parts.append(
            "Keep these characters and this visual style identical across the whole image sequence:\n"
            + sheet.strip()
        )
    parts.append("Keep the scene visually clear, with no text, logos, or watermarks.")
    if custom_instructions.strip():
        parts.append(custom_instructions.strip())
    return "\n\n".join(parts)



def compose_compact_block(scenes, sheet: str = "", custom_instructions: str = "") -> str:
    """Compact combined block for stateful generators that remember context.

    The character sheet, style anchor and global rules appear ONCE at the top;
    each image is then just its scenario line - no repetition, fewer tokens.
    Paste the whole block into a chat-based generator (Gemini, ChatGPT) in one go.
    scenes: list of (label, scene_text), label e.g. "IMAGE 1 (0s)".
    """
    header = []
    if sheet.strip():
        header.append(sheet.strip())
    header.append("Every image: cinematic 16:9 still, no text, logos, or watermarks.")
    if custom_instructions.strip():
        header.append(custom_instructions.strip())
    blocks = ["\n".join(header)]
    blocks += [f"{label}: {scene}" for label, scene in scenes]
    return "\n\n".join(blocks)
