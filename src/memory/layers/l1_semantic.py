from datetime import datetime, timezone
from typing import Any

from memory.embedding.embedder import embed_text
from memory.layers.qdrant_client import ensure_collection, search_points, upsert_point


COLLECTION = "semantic_memory"


async def save_semantic_memory(content: str, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    vector = await embed_text(content)
    await ensure_collection(COLLECTION, len(vector))
    payload = {
        "content": content,
        "layer": "L1",
        "category": metadata.get("category", "skill"),
        "tags": metadata.get("tags", []),
        "source": metadata.get("source", "manual"),
        "importance": int(metadata.get("importance", 3)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return await upsert_point(COLLECTION, vector, payload)


async def recall_semantic_memory(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    vector = await embed_text(query)
    await ensure_collection(COLLECTION, len(vector))
    hits = await search_points(COLLECTION, vector, top_k)
    return [
        {
            "id": h.get("id"),
            "score": h.get("score", 0),
            "layer": "L1",
            "content": (h.get("payload") or {}).get("content", ""),
            "metadata": h.get("payload", {}),
        }
        for h in hits
    ]


async def recall_semantic_memory_with_filters(
    query: str,
    top_k: int = 5,
    categories: list[str] | None = None,
    min_importance: int | None = None,
    tags: list[str] | None = None,
    sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    vector = await embed_text(query)
    await ensure_collection(COLLECTION, len(vector))

    must: list[dict] = []
    if categories:
        must.append({"key": "category", "match": {"any": categories}})
    if min_importance is not None:
        must.append({"key": "importance", "range": {"gte": int(min_importance)}})
    if tags:
        must.append({"key": "tags", "match": {"any": tags}})
    if sources:
        must.append({"key": "source", "match": {"any": sources}})

    query_filter = {"must": must} if must else None
    hits = await search_points(COLLECTION, vector, top_k, query_filter=query_filter)
    return [
        {
            "id": h.get("id"),
            "score": h.get("score", 0),
            "layer": "L1",
            "content": (h.get("payload") or {}).get("content", ""),
            "metadata": h.get("payload", {}),
        }
        for h in hits
    ]
