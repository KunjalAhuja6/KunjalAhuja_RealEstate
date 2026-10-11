import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Automatically load environment variables from .env file if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseModel):
    # Project paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    SAMPLE_DATA_DIR: Path = BASE_DIR / "data" / "sample_data"
    MODELS_DIR: Path = BASE_DIR / "data" / "models"
    RAW_DATA_PATH: Path = BASE_DIR / "data" / "sample_data" / "ames_housing.csv"
    CLEANED_DATA_PATH: Path = BASE_DIR / "data" / "sample_data" / "ames_housing_cleaned.csv"
    FEEDBACK_DATA_PATH: Path = BASE_DIR / "data" / "sample_data" / "feedback.json"
    ML_MODEL_PATH: Path = BASE_DIR / "data" / "models" / "property_price_model.joblib"
    DOCS_DIR: Path = BASE_DIR / "docs"
    ARCHITECTURE_DIR: Path = BASE_DIR / "architecture"
    DESIGN_DIR: Path = BASE_DIR / "design"
    PRESENTATION_DIR: Path = BASE_DIR / "presentation"

    # API configuration
    API_TITLE: str = "AI-Powered Real Estate Property Valuation Assistant"
    API_VERSION: str = "1.0.0"
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Retrieval configuration
    HYBRID_ALPHA: float = 0.55  # Weight on semantic vs keyword search (0.55 semantic, 0.45 keyword)
    DEFAULT_TOP_K: int = 5
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # LLM configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    USE_MOCK_FALLBACK: bool = True  # Always fallback smoothly to deterministic reasoning engine if no key

settings = Settings()
