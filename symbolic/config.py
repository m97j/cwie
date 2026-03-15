from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent  # ai_server/

# Hugging Face Token
HF_TOKEN = os.getenv("HF_TOKEN")

# Hugging Face Serve URL
HF_SERVE_URL = os.getenv("HF_SERVE_URL", "https://m97j-personachatengine-hf-serve.hf.space/api")

# Hugging Face Serve Timeout (초)
HF_TIMEOUT = float(os.getenv("HF_TIMEOUT", "25"))


# 모델 이름
FALLBACK_MODEL_NAME = os.getenv("FALLBACK_MODEL_NAME", "skt/ko-gpt-trinity-1.2B-v0.5")
EMBEDDER_MODEL_NAME = os.getenv("EMBEDDER_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 모델 디렉토리
FALLBACK_MODEL_DIR = Path(os.getenv("FALLBACK_MODEL_DIR", BASE_DIR / "models" / "fallback-npc-model"))
EMBEDDER_MODEL_DIR = Path(os.getenv("EMBEDDER_MODEL_DIR", BASE_DIR / "models" / "sentence-embedder"))

# ChromaDB 디렉토리
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "/app/rag/chroma_DB"))