from datetime import datetime, timezone
import os
from typing import Any

import httpx

from memory.embedding.embedder import embed_text
from memory.layers.qdrant_client import ensure_collection, upsert_point


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
GEN_MODEL = os.getenv("GEN_MODEL", "qwen3:8b")
COLLECTION = "reflective_memory"


class ReflectiveError(Exception):
    pass


async def generate_reflection_text(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": GEN_MODEL,
                "prompt": prompt,
                "stream": False,
            },
        )
    if resp.status_code != 200:
        raise ReflectiveError(f"ollama generate failed: {resp.status_code} {resp.text}")

    data = resp.json()
    text = (data.get("response") or "").strip()
    if not text:
        raise ReflectiveError("ollama returned empty reflection")
    return text


async def save_reflection(
    content: str,
    reflection_type: str,
    period_start: str,
    period_end: str,
    focus_areas: list[str] | None,
    source_layers: list[str] | None,
) -> str:
    vector = await embed_text(content)
    await ensure_collection(COLLECTION, len(vector))
    payload: dict[str, Any] = {
        "content": content,
        "layer": "L4",
        "reflection_type": reflection_type,
        "period_start": period_start,
        "period_end": period_end,
        "focus_areas": focus_areas or [],
        "source_layers": source_layers or [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return await upsert_point(COLLECTION, vector, payload)
