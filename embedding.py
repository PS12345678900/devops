import os
from typing import List

import numpy as np

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
        if _OPENAI_STYLE == "none":
            raise RuntimeError("OpenAI package not installed. Run: pip install --upgrade openai")
        # Try to populate env from .env if present
        _maybe_load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Please set it in environment or .env")
        base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        # Allow overriding embedding model via env
        env_model = os.getenv("EMBEDDING_MODEL") or os.getenv("OPENAI_EMBEDDING_MODEL")
        verify_env = os.getenv("OPENAI_VERIFY_SSL", "").strip().lower()
        # OPENAI_VERIFY_SSL=false disables TLS verification for enterprise proxies
        verify_ssl = not (verify_env in ("0", "false", "no"))

        self.batch_size = batch_size
        self._openai_embed_model = env_model or "text-embedding-3-small"
        if _OPENAI_STYLE == "v1":
            # v1 client supports base_url and http_client (for verify control)
            http_client = None
            try:
                import httpx  # type: ignore

                http_client = httpx.Client(verify=verify_ssl)
            except Exception:
                http_client = None
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            if http_client is not None:
                client_kwargs["http_client"] = http_client  # type: ignore[assignment]
            self._client = OpenAI(**client_kwargs)  # type: ignore[arg-type]
            self._legacy = False
        else:
            # legacy 0.x
            import openai  # type: ignore

            openai.api_key = api_key
            if base_url:
                openai.api_base = base_url
            # Legacy SDK doesn't accept http_client; consider proper CA config if verify fails
            self._legacy = True
            self._openai_mod = openai

    @property
    def name(self) -> str:
        return "openai"

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1536), dtype=np.float32)
        try:
            if self._legacy:
                res = self._openai_mod.Embedding.create(input=texts, model=self._openai_embed_model)
                vectors = [d["embedding"] for d in res["data"]]
            else:
                result = self._client.embeddings.create(input=texts, model=self._openai_embed_model)
                vectors = [d.embedding for d in result.data]
            return np.array(vectors, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding failed: {e}") from e

    def embed_query(self, text: str) -> np.ndarray:
        embs = self.embed_texts([text])
        return embs[0] if embs.shape[0] else np.zeros((1536,), dtype=np.float32)


