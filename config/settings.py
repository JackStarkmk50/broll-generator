from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
import os


class Settings(BaseSettings):
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    ASSETS_OUTPUT_DIR: Path = BASE_DIR / "b-roll_assets"
    REFERENCE_ASSETS_DIR: Path = BASE_DIR / "reference_assets"
    QDRANT_STORAGE_DIR: Path = BASE_DIR / "qdrant_storage"
    TEMP_DIR: Path = BASE_DIR / "temp_cache"
    DATA_DIR: Path = BASE_DIR / "data"
    DB_PATH: Path = DATA_DIR / "edit_helper.db"

    # Model configs
    CLIP_MODEL_NAME: str = "ViT-B-32"
    CLIP_PRETRAINED: str = "openai"
    WHISPER_MODEL_SIZE: str = "large-v3"
    DEVICE: str = "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu"

    # Evaluation thresholds
    SIMILARITY_THRESHOLD: float = 0.82
    TEXT_WEIGHT: float = 0.60
    AESTHETIC_WEIGHT: float = 0.40

    # API keys (loaded from .env or UI)
    PEXELS_API_KEY: str = Field(default="", env="PEXELS_API_KEY")
    GEMINI_API_KEY: str = Field(default="", env="GEMINI_API_KEY")

    # Search sources
    ENABLE_PINTEREST: bool = True
    ENABLE_PEXELS: bool = True
    PINTEREST_RESULTS_PER_QUERY: int = 15
    PEXELS_RESULTS_PER_QUERY: int = 6

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Ensure required directories exist
for p in [
    settings.ASSETS_OUTPUT_DIR,
    settings.REFERENCE_ASSETS_DIR,
    settings.QDRANT_STORAGE_DIR,
    settings.TEMP_DIR,
    settings.DATA_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)
