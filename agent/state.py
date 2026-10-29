from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class StockCandidate(BaseModel):
    id: str
    media_type: Literal["video", "image"]
    title: str
    source_url: str
    download_url: str
    preview_url: str
    width: int = 1920
    height: int = 1080
    duration: Optional[float] = None
    source: str = "unknown"  # pinterest | pexels | fallback


class EvaluatedAsset(BaseModel):
    candidate: StockCandidate
    semantic_score: float
    aesthetic_score: float
    composite_score: float
    status: Literal["approved", "rejected"]
    rejection_reason: Optional[str] = None
    local_path: Optional[str] = None


class AgentState(BaseModel):
    beat_index: int
    beat_start: float
    beat_end: float
    beat_text: str
    keywords: List[str] = Field(default_factory=list)
    text_embedding: List[float] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)
    candidates: List[StockCandidate] = Field(default_factory=list)
    evaluated_assets: List[EvaluatedAsset] = Field(default_factory=list)
    current_idx: int = 0
    approved_asset: Optional[EvaluatedAsset] = None
    logs: List[str] = Field(default_factory=list)
    finished: bool = False
