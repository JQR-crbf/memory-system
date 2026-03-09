from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import yaml

from memory.embedding.embedder import embed_text
from memory.layers.qdrant_client import ensure_collection, search_points, upsert_point


GOALS_PATH = Path("/data/L5/goals.yaml")
COLLECTION = "aspiration_memory"


class AspirationError(Exception):
    pass


def load_goals() -> list[dict[str, Any]]:
    if not GOALS_PATH.exists():
        return []
    with GOALS_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    goals = data.get("goals")
    return goals if isinstance(goals, list) else []


def save_goals(goals: list[dict[str, Any]]) -> None:
    GOALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOALS_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"goals": goals}, f, allow_unicode=True, sort_keys=False)


def _goal_to_text(goal: dict[str, Any]) -> str:
    title = str(goal.get("title", "")).strip()
    notes = str(goal.get("notes", "")).strip()
    related = goal.get("related_skills") or []
    related_text = ", ".join(x for x in related if isinstance(x, str))
    return f"{title}。{notes}。相关技能：{related_text}"


async def upsert_goal_vector(goal: dict[str, Any]) -> str:
    goal_id = str(goal.get("id", "")).strip()
    if not goal_id:
        raise AspirationError("goal.id is required")
    vector = await embed_text(_goal_to_text(goal))
    await ensure_collection(COLLECTION, len(vector))

    payload = {
        "content": _goal_to_text(goal),
        "layer": "L5",
        "category": "goal",
        "goal_id": goal_id,
        "title": goal.get("title", ""),
        "priority": goal.get("priority", "medium"),
        "status": goal.get("status", "not_started"),
        "progress": int(goal.get("progress", 0)),
        "deadline": goal.get("deadline", ""),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    point_id = str(uuid5(NAMESPACE_URL, f"goal:{goal_id}"))
    return await upsert_point(COLLECTION, vector, payload, point_id=point_id)


async def recall_aspiration_memory(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    vector = await embed_text(query)
    await ensure_collection(COLLECTION, len(vector))
    hits = await search_points(COLLECTION, vector, top_k)
    return [
        {
            "id": h.get("id"),
            "score": h.get("score", 0),
            "layer": "L5",
            "content": (h.get("payload") or {}).get("content", ""),
            "metadata": h.get("payload", {}),
        }
        for h in hits
    ]
