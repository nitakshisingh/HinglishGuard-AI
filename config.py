"""
Configuration module for HinglishGuard AI.

Loads environment variables safely using python-dotenv, configures model
settings, paths, and policy thresholds without hardcoding secrets.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Load .env file from the project root if present
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# Google Gemini API Settings
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Flag indicating whether Gemini API is configured
IS_GEMINI_AVAILABLE: bool = bool(
    GEMINI_API_KEY and GEMINI_API_KEY.strip() and GEMINI_API_KEY != "your_gemini_api_key_here"
)

# Application & Environment Settings
APP_ENV: str = os.getenv("APP_ENV", "development")

# Database Configuration (Defaults to SQLite in data/ directory)
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DEFAULT_DB_PATH = DATA_DIR / "moderation.db"
DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH.as_posix()}")

# Model Artifact Paths
MODELS_DIR = BASE_DIR / "models"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline" / "logistic_regression_baseline.joblib"
BASELINE_VEC_PATH = MODELS_DIR / "baseline" / "tfidf_vectorizer.joblib"
TRANSFORMER_MODEL_DIR = MODELS_DIR / "transformers" / "xlm_roberta_hinglish" / "best_model"

# Configurable Moderation & Confidence Thresholds
# If classifier confidence is below UNCERTAINTY_THRESHOLD, route to human review
UNCERTAINTY_THRESHOLD: float = float(os.getenv("UNCERTAINTY_THRESHOLD", "0.70"))

# Threshold above which an item is considered high confidence
HIGH_CONFIDENCE_THRESHOLD: float = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.85"))

# Profanity intensity threshold (from HEOT lexicon scale 1-10) triggering escalation
PROFANITY_ESCALATION_THRESHOLD: int = int(os.getenv("PROFANITY_ESCALATION_THRESHOLD", "7"))

# Target Class Taxonomy
CLASS_NAMES = ["Neutral", "Offensive", "Hate"]
ID_TO_LABEL = {0: "Neutral", 1: "Offensive", 2: "Hate"}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}

# Human Review Status Constants
REVIEW_STATUS_PENDING = "PENDING"
REVIEW_STATUS_RESOLVED = "RESOLVED"
REVIEW_STATUS_SKIPPED = "SKIPPED"

# Policy Action Constants
POLICY_ACTION_APPROVE = "APPROVE"
POLICY_ACTION_FLAG = "FLAG"
POLICY_ACTION_ESCALATE = "ESCALATE_TO_HUMAN_REVIEW"
