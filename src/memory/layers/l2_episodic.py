from datetime import datetime, timezone
from typing import Any

from memory.embedding.embedder import embed_text
from memory.layers.qdrant_client import ensure_collection, search_points, upsert_point


COLLECTION = "episodic_memory"


async def save_episodic_memory(content: str, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    vector = await embed_text(content)
    await ensure_collection(COLLECTION, len(vector))
    payload = {
        "content": content,
        "layer": "L2",
        "event_type": metadata.get("event_type", "work"),
        "date": metadata.get("date", datetime.now(timezone.utc).date().isoformat()),
        "time_period": metadata.get("time_period", "full_day"),
        "emotion": metadata.get("emotion", "neutral"),
        "project": metadata.get("project", ""),
        "tags": metadata.get("tags", []),
        "source": metadata.get("source", "manual"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return await upsert_point(COLLECTION, vector, payload)


async def recall_episodic_memory(
    query: str,
    top_k: int = 5,
    date_from: str | None = None,
    date_to: str | None = None,
    event_types: list[str] | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
    sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    vector = await embed_text(query)
    await ensure_collection(COLLECTION, len(vector))

    query_filter: dict[str, Any] | None = None
    date_conditions: list[dict[str, Any]] = []
    if date_from:
        date_conditions.append({"key": "date", "range": {"gte": date_from}})
    if date_to:
        date_conditions.append({"key": "date", "range": {"lte": date_to}})
    if date_conditions:
        query_filter = {"must": date_conditions}

    must = [] if query_filter is None else query_filter["must"]
    if event_types:
        must.append({"key": "event_type", "match": {"any": event_types}})
    if project:
        must.append({"key": "project", "match": {"value": project}})
    if tags:
        must.append({"key": "tags", "match": {"any": tags}})
    if sources:
        must.append({"key": "source", "match": {"any": sources}})
    if must:
        query_filter = {"must": must}

    hits = await search_points(COLLECTION, vector, top_k, query_filter=query_filter)
    return [
        {
            "id": h.get("id"),
            "score": h.get("score", 0),
            "layer": "L2",
            "content": (h.get("payload") or {}).get("content", ""),
            "metadata": h.get("payload", {}),
        }
        for h in hits
    ]
