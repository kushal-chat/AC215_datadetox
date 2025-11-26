"""Configuration settings for the model lineage pipeline."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings."""

    # HuggingFace
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")

    # Neo4j
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")

    # Data paths
    BASE_DATA_PATH: Path = Path("data/model-lineage")
    RAW_DATA_PATH: Path = BASE_DATA_PATH / "raw"
    PROCESSED_DATA_PATH: Path = BASE_DATA_PATH / "processed"

    # Scraping settings
    MAX_MODELS_PER_RUN: Optional[int] = None  # None = no limit
    SCRAPE_BATCH_SIZE: int = 100
    RATE_LIMIT_DELAY: float = 0.1  # seconds between requests

    # Graph settings
    NEO4J_DATABASE: str = "neo4j"  # Default database name

    @classmethod
    def validate(cls) -> bool:
        """Validate that required settings are present."""
        if not cls.HF_TOKEN:
            raise ValueError("HF_TOKEN environment variable is required")
        return True


settings = Settings()
