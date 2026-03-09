from typing import Any
from uuid import uuid4
import os

import httpx


QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")


class QdrantError(Exception):
    pass


async def ensure_collection(collection: str, vector_size: int) -> None:
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(f"{QDRANT_URL}/collections/{collection}")
        if r.status_code == 200:
            return

        if r.status_code != 404:
            raise QdrantError(f"qdrant check collection failed: {r.status_code} {r.text}")

        create_payload = {
            "vectors": {
                "size": vector_size,
                "distance": "Cosine",
            }
        }
        create_resp = await client.put(
            f"{QDRANT_URL}/collections/{collection}",
            json=create_payload,
        )
        if create_resp.status_code not in (200, 201):
            raise QdrantError(
                f"qdrant create collection failed: {create_resp.status_code} {create_resp.text}"
            )


async def upsert_point(
    collection: str,
    vector: list[float],
    payload: dict[str, Any],
    point_id: str | None = None,
) -> str:
    point_id = point_id or str(uuid4())
    body = {"points": [{"id": point_id, "vector": vector, "payload": payload}]}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.put(f"{QDRANT_URL}/collections/{collection}/points?wait=true", json=body)
    if r.status_code != 200:
        raise QdrantError(f"qdrant upsert failed: {r.status_code} {r.text}")
    return point_id


async def search_points(
    collection: str,
    query_vector: list[float],
    limit: int,
    query_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    body: dict[str, Any] = {
        "vector": query_vector,
        "limit": limit,
        "with_payload": True,
    }
    if query_filter:
        body["filter"] = query_filter

    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(f"{QDRANT_URL}/collections/{collection}/points/search", json=body)
    if r.status_code != 200:
        raise QdrantError(f"qdrant search failed: {r.status_code} {r.text}")
    data = r.json()
    return data.get("result", [])


async def get_collection_points_count(collection: str) -> int:
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(f"{QDRANT_URL}/collections/{collection}")
    if r.status_code == 404:
        return 0
    if r.status_code != 200:
        raise QdrantError(f"qdrant collection stats failed: {r.status_code} {r.text}")
    data = r.json()
    result = data.get("result", {})
    return int(result.get("points_count") or 0)
