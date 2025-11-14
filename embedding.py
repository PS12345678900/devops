import os
from typing import List

import numpy as np
from langchain_openai import OpenAIEmbeddings  # type: ignore
import httpx  # type: ignore

# Support both OpenAI SDK styles:
# - v1.x: from openai import OpenAI
# - v0.x: import openai; openai.Embedding.create
_OPENAI_STYLE: str = "none"
try:
    from openai import OpenAI  # type: ignore

    _OPENAI_STYLE = "v1"
except Exception:
    try:
        import openai  # type: ignore

        _OPENAI_STYLE = "v0"
    except Exception:
        _OPENAI_STYLE = "none"


def _maybe_load_dotenv() -> None:
    """
    Lightweight .env loader without extra dependency.
    Looks for a .env file in CWD and in the project root (parent of this file's dir).
    Does not override variables already present in os.environ.
    """
    candidates = []
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(here)
        candidates.append(os.path.join(os.getcwd(), ".env"))
        candidates.append(os.path.join(project_root, ".env"))
    except Exception:
        candidates.append(os.path.join(os.getcwd(), ".env"))
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#") or "=" not in s:
                        continue
                    key, val = s.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("\"'")  # remove simple quotes
                    if key and key not in os.environ:
                        os.environ[key] = val
        except Exception:
            # Best-effort; ignore parse errors
            pass


class Embedder:
    def __init__(self, batch_size: int = 256) -> None:
        # Try to populate env from .env if present
        _maybe_load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Please set it in environment or .env")
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        env_model = os.getenv("EMBEDDING_MODEL") or os.getenv("OPENAI_EMBEDDING_MODEL")
        self.batch_size = batch_size
        self._openai_embed_model = env_model or "text-embedding-3-small"
        # LangChain embeddings with enterprise endpoint and TLS verify disabled
        http_client = httpx.Client(verify=False)
        self._lc_embeddings = OpenAIEmbeddings(
            api_key=api_key,
            base_url=base_url,
            model=self._openai_embed_model,
            http_client=http_client,
        )

    @property
    def name(self) -> str:
        return "openai"

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1536), dtype=np.float32)
        vectors = self._lc_embeddings.embed_documents(texts)
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vec = self._lc_embeddings.embed_query(text)
        return np.array(vec, dtype=np.float32)


