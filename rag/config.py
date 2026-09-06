import os

from dotenv import load_dotenv

load_dotenv()

# Which chat-model provider the graph runs on: "groq" or "anthropic".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Sensible per-provider defaults, so switching LLM_PROVIDER doesn't require also
# rewriting the model IDs. Explicit env vars still win.
_DEFAULT_MODELS = {
    "anthropic": {"generation": "claude-opus-5", "critic": "claude-opus-5"},
    # Groq rotates its catalogue often; check `client.models.list()` if these 404.
    "groq": {
        "generation": "openai/gpt-oss-120b",
        "critic": "openai/gpt-oss-120b",
    },
}
_defaults = _DEFAULT_MODELS.get(LLM_PROVIDER, _DEFAULT_MODELS["groq"])

# Model used to generate answers from retrieved context.
GENERATION_MODEL = os.getenv("GENERATION_MODEL") or _defaults["generation"]

# Model used by the critic to judge groundedness. Must be at least as capable
# as GENERATION_MODEL, since it has to catch subtler mistakes than it makes.
CRITIC_MODEL = os.getenv("CRITIC_MODEL") or _defaults["critic"]

# Local, offline embedding model (BAAI/bge-m3) via sentence-transformers.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")

# Chunks retrieved per pass. Lower values make retrieval misses — and therefore the
# self-healing retry — more likely; raise it for large corpora where recall matters more.
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
