from typing import Dict, Any, Callable, Optional, Set
import numpy as np
from pathlib import Path

from agent.state import AgentState, EvaluatedAsset
from agent.stock_sourcer import StockSourcer
from agent.director_evaluator import DirectorEvaluator
from core.audio_aligner import NarrativeBeat
from core.multimodal_embedder import MultimodalEmbedder
from core.aesthetic_memory import AestheticMemory
from core.llm_translator import LLMTranslator, is_non_english


class BRollAgentGraph:
    """
    LangGraph-style workflow orchestrating Sourcer, Director, and Downloader.
    Logs execution traces for terminal and UI streaming.

    Note: Runs as a simple state-machine loop instead of StateGraph to avoid
    Pydantic vs TypedDict compatibility issues across LangGraph versions.
    """

    def __init__(
        self,
        sourcer: Optional[StockSourcer] = None,
        director: Optional[DirectorEvaluator] = None,
        embedder: Optional[MultimodalEmbedder] = None,
        aesthetic_memory: Optional[AestheticMemory] = None,
        threshold: float = 0.82,
        log_callback: Optional[Callable[[str], None]] = None,
        exclude_urls: Optional[Set[str]] = None,
        translator: Optional[LLMTranslator] = None,
    ):
        self.embedder = embedder or MultimodalEmbedder()
        self.aesthetic_memory = aesthetic_memory or AestheticMemory(embedder=self.embedder)
        self.sourcer = sourcer or StockSourcer()
        self.director = director or DirectorEvaluator(
            embedder=self.embedder, aesthetic_memory=self.aesthetic_memory
        )
        self.threshold = threshold
        self.log_callback = log_callback or (lambda msg: print(f"[Agent] {msg}"))
        self.exclude_urls = exclude_urls or set()
        self.translator = translator

    def _log(self, state: AgentState, message: str):
        state.logs.append(message)
        self.log_callback(message)

    def _run_sourcer(self, state: AgentState):
        """Sourcer node: translate if needed, reason about text, generate queries, fetch candidates."""
        self._log(state, f"[Sourcer] Processing Beat {state.beat_index}: '{state.beat_text[:80]}'")

        search_text = state.beat_text
        search_queries = []

        # Translate non-English text via LLM
        if self.translator and is_non_english(state.beat_text):
            self._log(state, f"[Translator] Non-English detected. Translating...")
            translated = self.translator.translate_and_reason(state.beat_text)
            state._translation = translated
            if translated.english_translation != state.beat_text:
                search_text = translated.english_translation
                self._log(state, f"[Translator] ({translated.backend}) {translated.language}")
                self._log(state, f"[Translator] English: {translated.english_translation}")
                if translated.visual_description:
                    self._log(state, f"[Translator] Visuals: {translated.visual_description}")
                if translated.search_queries:
                    search_queries = translated.search_queries
                    self._log(state, f"[Translator] LLM queries: {search_queries}")
                # Re-embed with English text for CLIP matching
                state.text_embedding = self.embedder.embed_text(search_text).tolist()

        # Fall back to rule-based reasoner if LLM didn't provide queries
        if not search_queries:
            reasoned = self.sourcer.reason_and_explain(search_text, state.keywords)
            search_queries = reasoned.queries
            self._log(state, f"[Reasoner] {reasoned.analysis}")
            self._log(state, f"[Reasoner] Mood: {reasoned.mood} | Queries: {search_queries}")

        state.search_queries = search_queries

        candidates = []
        for q in search_queries[:4]:
            found = self.sourcer.search_all(q, per_page=3, exclude_urls=self.exclude_urls)
            candidates.extend(found)

        # Deduplicate by id
        seen = set()
        deduped = []
        for c in candidates:
            if c.id not in seen:
                seen.add(c.id)
                deduped.append(c)

        state.candidates = deduped
        state.current_idx = 0

        # Count sources
        pinterest_count = sum(1 for c in deduped if c.source == "pinterest")
        pexels_count = sum(1 for c in deduped if c.source == "pexels")
        fallback_count = sum(1 for c in deduped if c.source == "fallback")
        self._log(
            state,
            f"[Sourcer] {len(deduped)} candidates "
            f"(Pinterest: {pinterest_count}, Pexels: {pexels_count}, Fallback: {fallback_count})",
        )

    def _run_director(self, state: AgentState) -> bool:
        """
        Director node: evaluate candidates one by one.
        Returns True when done (approved or exhausted).
        """
        while state.current_idx < len(state.candidates):
            cand = state.candidates[state.current_idx]
            self._log(
                state,
                f"[Director] Evaluating #{state.current_idx + 1} ({cand.media_type.upper()}, {cand.source}): '{cand.title[:60]}'",
            )

            text_emb = np.array(state.text_embedding, dtype=np.float32)
            evaluated = self.director.evaluate(
                candidate=cand,
                text_embedding=text_emb,
                threshold=self.threshold,
            )
            state.evaluated_assets.append(evaluated)

            status_str = "APPROVED" if evaluated.status == "approved" else f"Below {self.threshold}"
            self._log(
                state,
                f"[Director] Semantic={evaluated.semantic_score:.3f} | "
                f"Aesthetic={evaluated.aesthetic_score:.3f} | "
                f"Composite={evaluated.composite_score:.3f} ({status_str})",
            )

            if evaluated.status == "approved":
                state.approved_asset = evaluated
                return True

            state.current_idx += 1

        self._log(state, f"[Director] Exhausted all candidates for Beat {state.beat_index}.")
        return True

    def _run_download(self, state: AgentState):
        """Download approved or best-effort asset."""
        target = state.approved_asset

        if target is None and state.evaluated_assets:
            best = max(state.evaluated_assets, key=lambda a: a.composite_score)
            target = best
            self._log(
                state,
                f"[Director] No candidate met {self.threshold:.2f}. "
                f"Using best: '{target.candidate.title[:50]}' (Score: {target.composite_score:.3f})",
            )

        if target:
            self._log(state, f"[Downloader] Downloading {target.candidate.media_type} from {target.candidate.source}...")
            saved_path = self.director.download_asset(target.candidate, state.beat_index)
            target.local_path = str(saved_path)
            state.approved_asset = target
            self._log(state, f"[Downloader] Saved: {saved_path.name}")

            # Track this URL so next beats won't reuse it
            self.exclude_urls.add(target.candidate.source_url)
            self.exclude_urls.add(target.candidate.download_url)

        state.finished = True

    def process_beat(self, beat: NarrativeBeat, text_emb: np.ndarray) -> AgentState:
        """Execute the full pipeline for a single narrative beat."""
        state = AgentState(
            beat_index=beat.index,
            beat_start=beat.start_time,
            beat_end=beat.end_time,
            beat_text=beat.text,
            keywords=beat.search_keywords,
            text_embedding=text_emb.tolist(),
        )

        # Step 1: Source candidates
        self._run_sourcer(state)

        # Step 2: Evaluate candidates
        self._run_director(state)

        # Step 3: Download best asset
        self._run_download(state)

        return state
