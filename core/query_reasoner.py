"""
Query Reasoner: analyzes transcript beat text and generates
intelligent visual search queries for b-roll sourcing.

Takes raw transcript text → reasons about visual concepts →
outputs diverse, targeted search queries with mood/style awareness.

Works without an LLM using visual concept mapping and text analysis.
"""
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ReasonedQuery:
    """Output of reasoning over a single narrative beat."""
    original_text: str
    analysis: str                          # Human-readable reasoning
    visual_concepts: List[str]             # Extracted visual ideas
    queries: List[str]                     # Final search queries
    mood: str = ""                         # Detected mood/tone
    style_modifiers: List[str] = field(default_factory=list)


# ── Visual Concept Mapping ────────────────────────────────────
# Maps abstract/narrative words to concrete visual search terms.
# Each key can appear in transcript; values are what to search for.

CONCEPT_TO_VISUAL = {
    # Abstract → concrete
    "technology": ["computer screen code", "digital interface", "circuit board macro", "person using laptop"],
    "innovation": ["futuristic lab", "3D printing", "robotics arm", "startup whiteboard"],
    "future": ["futuristic city skyline", "holographic display", "modern architecture glass"],
    "past": ["vintage film grain", "old photographs sepia", "antique clock"],
    "history": ["ancient ruins", "old manuscripts", "museum artifacts"],
    "growth": ["plant sprouting timelapse", "graph rising animation", "sunrise horizon"],
    "decline": ["wilting flower", "abandoned building", "sunset fading"],
    "speed": ["motion blur highway", "timelapse city traffic", "racing car closeup"],
    "slow": ["slow motion water droplets", "candle flame macro", "flowing silk fabric"],
    "power": ["lightning storm", "waterfall force", "engine turbine"],
    "peace": ["calm lake reflection", "meditation garden", "morning mist forest"],
    "danger": ["storm clouds gathering", "fire flames close", "cracked earth drought"],
    "success": ["mountain summit view", "trophy celebration", "finish line crossing"],
    "failure": ["broken glass", "rain on window", "empty chair spotlight"],
    "connection": ["handshake business", "network nodes", "bridge spanning river"],
    "isolation": ["lone tree field", "empty road desert", "single boat ocean"],
    "creativity": ["paint splatter canvas", "artist studio workspace", "colorful abstract"],
    "knowledge": ["library bookshelves", "open book pages", "classroom lecture"],
    "money": ["stock market screen", "coins stacking", "city financial district"],
    "health": ["running outdoor fitness", "fresh vegetables", "hospital corridor"],
    "love": ["couple silhouette sunset", "heart shape hands", "roses bouquet"],
    "war": ["chess pieces strategy", "storm approaching", "barbed wire fence"],
    "freedom": ["bird flying sky", "open road horizon", "balloon floating"],
    "change": ["butterfly metamorphosis", "seasons changing", "door opening light"],
    "time": ["clock gears mechanism", "hourglass sand", "calendar pages turning"],
    "music": ["piano keys close", "concert crowd lights", "vinyl record spinning"],
    "travel": ["airplane window clouds", "passport stamps", "world map pins"],
    "food": ["cooking flame kitchen", "fresh ingredients table", "restaurant plating"],
    "work": ["office desk organized", "factory production line", "team meeting room"],
    "dream": ["starry night sky", "clouds floating", "abstract light bokeh"],
    "struggle": ["climbing steep mountain", "storm waves ocean", "runner exhausted"],
}

# Mood detection keywords
MOOD_KEYWORDS = {
    "dramatic": ["dramatic", "intense", "powerful", "shocking", "devastating", "critical"],
    "uplifting": ["hope", "inspiring", "amazing", "wonderful", "beautiful", "incredible", "achieve"],
    "melancholic": ["sad", "loss", "missing", "gone", "empty", "alone", "dark"],
    "energetic": ["fast", "rapid", "exciting", "dynamic", "rush", "action", "speed"],
    "calm": ["peaceful", "quiet", "gentle", "soft", "serene", "tranquil", "still"],
    "mysterious": ["strange", "unknown", "hidden", "secret", "shadow", "enigma"],
    "professional": ["business", "corporate", "market", "industry", "company", "enterprise"],
    "cinematic": ["cinematic", "epic", "grand", "vast", "sweeping", "panoramic"],
}

# Style modifiers that improve search results
STYLE_MODIFIERS = {
    "dramatic": ["cinematic lighting", "dramatic angle", "high contrast"],
    "uplifting": ["golden hour", "bright", "warm tones"],
    "melancholic": ["moody", "dark tones", "rain"],
    "energetic": ["dynamic", "motion", "vibrant colors"],
    "calm": ["soft focus", "pastel", "minimal"],
    "mysterious": ["silhouette", "fog", "dark"],
    "professional": ["clean", "modern", "corporate"],
    "cinematic": ["wide angle", "4K cinematic", "film grain"],
}

# Stop words for visual query (more aggressive than general stop words)
VISUAL_STOP_WORDS = {
    "the", "a", "an", "in", "to", "and", "is", "of", "for", "with", "on", "at",
    "by", "from", "it", "this", "that", "are", "be", "was", "were", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "can", "shall", "its", "our", "their", "your", "my", "his",
    "her", "we", "they", "you", "i", "me", "us", "them", "he", "she", "but",
    "or", "nor", "not", "no", "so", "if", "then", "than", "when", "where",
    "how", "what", "which", "who", "whom", "why", "all", "each", "every",
    "both", "few", "more", "most", "some", "any", "such", "very", "just",
    "also", "about", "up", "out", "into", "over", "after", "before",
    "between", "through", "during", "while", "because", "since", "until",
    "really", "actually", "basically", "simply", "today", "now", "here",
    "there", "still", "already", "even", "much", "many", "well", "way",
    "thing", "things", "let", "get", "got", "going", "make", "made",
    "take", "come", "say", "said", "tell", "told", "know", "think",
    "see", "look", "want", "need", "use", "try", "keep", "give",
}


class QueryReasoner:
    """
    Analyzes narrative beat text and reasons about what visuals
    would best match, then generates targeted search queries.

    Pipeline:
    1. Clean and tokenize text
    2. Extract visual nouns/adjectives (content words)
    3. Map abstract concepts → concrete visual scenes
    4. Detect mood/tone
    5. Apply custom instruction filters
    6. Generate diverse search queries with style modifiers
    """

    def __init__(self, custom_instructions: str = ""):
        self.custom_instructions = custom_instructions
        self._instruction_keywords = self._parse_instructions(custom_instructions)

    def _parse_instructions(self, instructions: str) -> Dict[str, List[str]]:
        """Extract actionable preferences from custom instructions."""
        result = {"prefer": [], "avoid": [], "style": []}
        if not instructions:
            return result

        lines = instructions.lower().split("\n")
        for line in lines:
            line = line.strip().lstrip("-•*")
            if not line:
                continue
            if any(w in line for w in ["prefer", "prioritize", "focus on", "use more", "favor"]):
                # Extract the visual preference
                words = re.sub(r"(prefer|prioritize|focus on|use more|favor)\s*:?\s*", "", line)
                result["prefer"].append(words.strip())
            elif any(w in line for w in ["avoid", "no ", "don't", "exclude", "never"]):
                words = re.sub(r"(avoid|exclude|never|don't use|no)\s*:?\s*", "", line)
                result["avoid"].append(words.strip())
            elif any(w in line for w in ["style", "mood", "tone", "aesthetic", "look", "feel"]):
                result["style"].append(line.strip())
        return result

    def reason(self, beat_text: str, keywords: List[str] = None) -> ReasonedQuery:
        """
        Full reasoning pipeline over a narrative beat.
        Returns analysis + optimized search queries.
        """
        keywords = keywords or []
        clean_text = re.sub(r"[^\w\s-]", "", beat_text).strip().lower()
        words = clean_text.split()

        # Step 1: Extract content words (visual nouns/adjectives)
        content_words = [w for w in words if w not in VISUAL_STOP_WORDS and len(w) > 2]

        # Step 2: Find concept matches
        visual_concepts = []
        matched_concepts = {}
        for word in content_words:
            for concept, visuals in CONCEPT_TO_VISUAL.items():
                if word == concept or (len(word) > 4 and concept.startswith(word)):
                    matched_concepts[concept] = visuals
                    visual_concepts.extend(visuals[:2])  # Top 2 visuals per concept

        # Step 3: Detect mood
        mood = self._detect_mood(words)
        style_mods = STYLE_MODIFIERS.get(mood, ["cinematic"])

        # Step 4: Apply custom instruction preferences
        prefer_terms = " ".join(self._instruction_keywords.get("prefer", []))
        avoid_terms = self._instruction_keywords.get("avoid", [])
        style_terms = self._instruction_keywords.get("style", [])

        # Step 5: Generate diverse queries
        queries = []

        # Query 1: Direct content words (most specific)
        if content_words:
            q1 = " ".join(content_words[:4])
            if prefer_terms:
                q1 = f"{q1} {prefer_terms.split()[0]}" if prefer_terms.split() else q1
            queries.append(q1)

        # Query 2: Concept-mapped visual (more creative)
        if visual_concepts:
            queries.append(visual_concepts[0])
            if len(visual_concepts) > 2:
                queries.append(visual_concepts[2])

        # Query 3: Mood + subject combination
        if content_words and mood:
            mood_mod = style_mods[0] if style_mods else "cinematic"
            queries.append(f"{mood_mod} {content_words[0]} {content_words[-1] if len(content_words) > 1 else ''}")

        # Query 4: Original keywords with style
        if keywords:
            kw_query = " ".join(keywords[:3])
            if style_terms:
                kw_query = f"{kw_query} {style_terms[0].split()[-1]}"
            queries.append(kw_query)

        # Query 5: Broader scene description
        if content_words:
            queries.append(f"b-roll {' '.join(content_words[:2])}")

        # Deduplicate and clean
        seen = set()
        clean_queries = []
        for q in queries:
            q = q.strip()
            if q and q not in seen:
                seen.add(q)
                # Filter out avoid terms
                if not any(avoid in q for avoid in avoid_terms):
                    clean_queries.append(q)

        if not clean_queries:
            clean_queries = [" ".join(content_words[:3])] if content_words else ["cinematic b-roll"]

        # Build analysis string
        analysis_parts = [f"Text: \"{beat_text[:80]}\""]
        if matched_concepts:
            analysis_parts.append(f"Concepts: {', '.join(matched_concepts.keys())}")
        analysis_parts.append(f"Mood: {mood}")
        if visual_concepts:
            analysis_parts.append(f"Visual ideas: {', '.join(visual_concepts[:4])}")
        analysis_parts.append(f"Generated {len(clean_queries)} search queries")
        if avoid_terms:
            analysis_parts.append(f"Avoiding: {', '.join(avoid_terms)}")

        return ReasonedQuery(
            original_text=beat_text,
            analysis=" | ".join(analysis_parts),
            visual_concepts=visual_concepts[:6],
            queries=clean_queries[:5],
            mood=mood,
            style_modifiers=style_mods,
        )

    def _detect_mood(self, words: List[str]) -> str:
        """Detect dominant mood from word list."""
        scores = {}
        for mood, triggers in MOOD_KEYWORDS.items():
            score = sum(1 for w in words if w in triggers)
            if score > 0:
                scores[mood] = score

        if scores:
            return max(scores, key=scores.get)
        return "cinematic"  # default

    def reason_batch(self, beats: List[Dict], keywords_list: List[List[str]] = None) -> List[ReasonedQuery]:
        """Reason over multiple beats."""
        keywords_list = keywords_list or [[] for _ in beats]
        return [
            self.reason(beat["text"], kws)
            for beat, kws in zip(beats, keywords_list)
        ]
