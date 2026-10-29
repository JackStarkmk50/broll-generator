from pathlib import Path
from typing import Optional
import requests
import re
import numpy as np
from PIL import Image
import io

from agent.state import StockCandidate, EvaluatedAsset
from core.multimodal_embedder import MultimodalEmbedder
from core.aesthetic_memory import AestheticMemory
from config.settings import settings

class DirectorEvaluator:
    """
    Director Agent node that performs multimodal semantic & aesthetic evaluation.
    Rejects media below the similarity threshold (default 0.82) and saves approved media.
    """
    def __init__(
        self,
        embedder: Optional[MultimodalEmbedder] = None,
        aesthetic_memory: Optional[AestheticMemory] = None,
        assets_dir: Optional[Path] = None,
        temp_dir: Optional[Path] = None
    ):
        self.embedder = embedder or MultimodalEmbedder()
        self.aesthetic_memory = aesthetic_memory or AestheticMemory(embedder=self.embedder)
        self.assets_dir = assets_dir or settings.ASSETS_OUTPUT_DIR
        self.temp_dir = temp_dir or settings.TEMP_DIR

        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        candidate: StockCandidate,
        text_embedding: np.ndarray,
        threshold: float = 0.82,
        text_weight: float = 0.60,
        aesthetic_weight: float = 0.40
    ) -> EvaluatedAsset:
        """
        Extracts visual embedding from candidate preview, computes cosine similarity against:
        1. Transcript semantic text embedding
        2. Client Aesthetic Profile vector (from Qdrant)
        """
        # Fetch candidate preview image/thumbnail
        preview_bytes = self._fetch_media_bytes(candidate.preview_url)
        if not preview_bytes:
            return EvaluatedAsset(
                candidate=candidate,
                semantic_score=0.0,
                aesthetic_score=0.0,
                composite_score=0.0,
                status="rejected",
                rejection_reason="Failed to download candidate preview media"
            )

        try:
            image = Image.open(io.BytesIO(preview_bytes)).convert("RGB")
            visual_emb = self.embedder.embed_image(image)
        except Exception as e:
            return EvaluatedAsset(
                candidate=candidate,
                semantic_score=0.0,
                aesthetic_score=0.0,
                composite_score=0.0,
                status="rejected",
                rejection_reason=f"Failed to extract visual embedding: {e}"
            )

        # 1. Semantic Match (Text <-> Visual)
        semantic_sim = MultimodalEmbedder.cosine_similarity(visual_emb, text_embedding)

        # 2. Aesthetic Profile Match (Visual <-> Qdrant Aesthetic Centroid)
        aesthetic_sim = self.aesthetic_memory.calculate_aesthetic_similarity(visual_emb)

        # 3. Composite Weighted Score
        composite_score = round((text_weight * semantic_sim) + (aesthetic_weight * aesthetic_sim), 4)

        is_approved = composite_score >= threshold
        status = "approved" if is_approved else "rejected"
        rejection_reason = None if is_approved else f"Score {composite_score:.3f} below threshold {threshold:.2f}"

        return EvaluatedAsset(
            candidate=candidate,
            semantic_score=round(semantic_sim, 4),
            aesthetic_score=round(aesthetic_sim, 4),
            composite_score=composite_score,
            status=status,
            rejection_reason=rejection_reason
        )

    def download_asset(self, candidate: StockCandidate, beat_index: int) -> Path:
        """
        Downloads the full approved asset to local /b-roll_assets folder with clean Windows filename.
        """
        slug = re.sub(r"[^\w\s-]", "", candidate.title).strip().replace(" ", "_")[:35]
        ext = "mp4" if candidate.media_type == "video" else "jpg"
        target_file = self.assets_dir / f"beat_{beat_index:02d}_{slug}.{ext}"

        # If already downloaded, return path
        if target_file.exists() and target_file.stat().st_size > 1000:
            return target_file

        url = candidate.download_url
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        try:
            res = requests.get(url, headers=headers, stream=True, timeout=25)
            if res.status_code == 200:
                with open(target_file, "wb") as f:
                    for chunk in res.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                return target_file
        except Exception as e:
            print(f"[DirectorEvaluator] Download failed for {candidate.id}: {e}")

        # Fallback: save preview if full download fails
        preview_bytes = self._fetch_media_bytes(candidate.preview_url)
        if preview_bytes:
            fallback_target = self.assets_dir / f"beat_{beat_index:02d}_{slug}_preview.jpg"
            with open(fallback_target, "wb") as f:
                f.write(preview_bytes)
            return fallback_target

        return target_file

    def _fetch_media_bytes(self, url: str) -> Optional[bytes]:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            r = requests.get(url, headers=headers, timeout=12)
            if r.status_code == 200:
                return r.content
        except Exception as e:
            print(f"[DirectorEvaluator] Fetch preview error for {url}: {e}")
        return None
