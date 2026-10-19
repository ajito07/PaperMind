import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("RM_DATA_DIR", ROOT / "data"))
PAPERS_DIR = DATA_DIR / "papers"
DB_PATH = Path(os.environ.get("RM_DB_PATH", DATA_DIR / "researchmind.db"))

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.environ.get("RM_CHAT_MODEL", "qwen2.5:14b")   # synthesis, agent, judge
FAST_MODEL = os.environ.get("RM_FAST_MODEL", "llama3.1:8b")   # rewrite, routing, metadata
EMBED_MODEL = os.environ.get("RM_EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = 768
NUM_CTX = 8192
KEEP_ALIVE = "30m"

CHUNK_CHARS = 3500      # ~800 tokens
CHUNK_OVERLAP = 500
TOP_K_EACH = 20         # per retriever, before fusion
TOP_K_FUSED = 8
RRF_K = 60
# ponytail: calibration knob — max cosine below this means "not in library", tune per embedder.
# Calibrated for nomic-embed-text: off-corpus queries scored <=0.51, on-corpus >=0.63.
MIN_COSINE = float(os.environ.get("RM_MIN_COSINE", "0.55"))
MAX_AGENT_STEPS = 4
