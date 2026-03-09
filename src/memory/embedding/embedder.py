from typing import Any
import os

import httpx


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


class EmbedderError(Exception):
    pass


async def embed_text(text: str) -> list[float]:
    if not text.strip():
        raise EmbedderError("empty text cannot be embedded")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Preferred endpoint in modern Ollama.
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
        )
        if resp.status_code == 200:
            data = resp.json()
            embeddings = data.get("embeddings")
            if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
                return [float(x) for x in embeddings[0]]

        # Fallback endpoint for compatibility.
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
        )
        if resp.status_code != 200:
            raise EmbedderError(f"ollama embedding failed: {resp.status_code} {resp.text}")

        data: dict[str, Any] = resp.json()
        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise EmbedderError("ollama returned invalid embedding payload")
        return [float(x) for x in embedding]
