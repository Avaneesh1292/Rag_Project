import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# API keys and runtime settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

# Keep backward compatibility for libraries that read env vars directly.
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

PDF_PATH = os.getenv("PDF_PATH", "").strip()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

SEMANTIC_K = int(os.getenv("SEMANTIC_K", "5"))
BM25_K = int(os.getenv("BM25_K", "5"))
RERANK_K = int(os.getenv("RERANK_K", "3"))

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
RETRY_BACKOFF_SECONDS = float(os.getenv("RETRY_BACKOFF_SECONDS", "1.0"))


def validate_config() -> None:
    missing = []
    if not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    if not PDF_PATH:
        missing.append("PDF_PATH")

    if missing:
        raise ValueError(
            "Missing required settings: "
            + ", ".join(missing)
            + ". Add them to your shell environment or .env file."
        )

    if not Path(PDF_PATH).exists():
        raise FileNotFoundError(f"PDF_PATH does not exist: {PDF_PATH}")
