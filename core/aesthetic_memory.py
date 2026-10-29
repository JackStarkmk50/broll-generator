from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config.settings import settings
from core.multimodal_embedder import MultimodalEmbedder

COLLECTION_NAME = "client_aesthetic_profile"

class AestheticMemory:
    """
    Manages client aesthetic reference memory in a local Qdrant vector database.
    Establishes a mathematical 'Aesthetic Profile' centroid and computes visual similarity.
    """
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        embedder: Optional[MultimodalEmbedder] = None
    ):
        self.storage_path = storage_path or settings.QDRANT_STORAGE_DIR
        self.embedder = embedder or MultimodalEmbedder()
        self.vector_dim = self.embedder.embedding_dim
        
        # Initialize embedded local Qdrant client
        self.client = QdrantClient(path=str(self.storage_path))
        self._ensure_collection()
        self._aesthetic_centroid: Optional[np.ndarray] = None
        self._load_centroid()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )

    def ingest_reference_folder(self, folder_path: Path | str) -> Dict[str, Any]:
        """
        Scans a folder of client reference images and video clips,
        generates visual embeddings, stores them in Qdrant, and updates the Aesthetic Profile centroid.
        """
        ref_dir = Path(folder_path).resolve()
        if not ref_dir.exists():
            return {"status": "error", "message": f"Directory not found: {ref_dir}", "count": 0}

        supported_image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        supported_video_exts = {".mp4", ".mov", ".avi", ".mkv"}

        media_files = [
            f for f in ref_dir.iterdir()
            if f.is_file() and f.suffix.lower() in (supported_image_exts | supported_video_exts)
        ]

        if not media_files:
            return {"status": "warning", "message": f"No media files found in {ref_dir}", "count": 0}

        points = []
        all_embeddings: List[np.ndarray] = []

        for idx, media_file in enumerate(media_files):
            try:
                ext = media_file.suffix.lower()
                if ext in supported_video_exts:
                    emb = self.embedder.embed_video_preview(media_file)
                    media_type = "video"
                else:
                    emb = self.embedder.embed_image(media_file)
                    media_type = "image"

                all_embeddings.append(emb)
                points.append(
                    PointStruct(
                        id=idx + 1,
                        vector=emb.tolist(),
                        payload={
                            "filename": media_file.name,
                            "filepath": str(media_file),
                            "media_type": media_type
                        }
                    )
                )
            except Exception as e:
                print(f"[AestheticMemory] Error embedding {media_file.name}: {e}")

        if points:
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )

        # Compute centroid (Aesthetic Profile vector)
        if all_embeddings:
            mean_vec = np.mean(all_embeddings, axis=0)
            norm = np.linalg.norm(mean_vec)
            self._aesthetic_centroid = (mean_vec / norm).astype(np.float32) if norm > 1e-6 else mean_vec
            
            # Save centroid metadata file
            centroid_file = self.storage_path / "aesthetic_centroid.npy"
            np.save(centroid_file, self._aesthetic_centroid)

        return {
            "status": "success",
            "ingested_count": len(points),
            "aesthetic_profile_established": self._aesthetic_centroid is not None
        }

    def _load_centroid(self):
        centroid_file = self.storage_path / "aesthetic_centroid.npy"
        if centroid_file.exists():
            try:
                self._aesthetic_centroid = np.load(centroid_file)
            except Exception as e:
                print(f"[AestheticMemory] Failed loading cached centroid: {e}")

    @property
    def aesthetic_centroid(self) -> Optional[np.ndarray]:
        return self._aesthetic_centroid

    def calculate_aesthetic_similarity(self, visual_emb: np.ndarray) -> float:
        """
        Calculates cosine similarity of a candidate asset's visual embedding
        against the client's mathematical Aesthetic Profile.
        """
        if self._aesthetic_centroid is None:
            # If no reference assets have been ingested yet, default neutral baseline
            return 0.85

        return MultimodalEmbedder.cosine_similarity(visual_emb, self._aesthetic_centroid)

    def query_nearest_references(self, visual_emb: np.ndarray, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Returns nearest reference assets from Qdrant for diagnostic comparison.
        """
        try:
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=visual_emb.tolist(),
                limit=limit
            )
            return [
                {
                    "score": r.score,
                    "filename": r.payload.get("filename"),
                    "filepath": r.payload.get("filepath")
                }
                for r in results
            ]
        except Exception:
            return []
