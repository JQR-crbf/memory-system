from datetime import datetime, timezone
import os
from typing import Any
from uuid import uuid4

from neo4j import GraphDatabase


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "memory123456")
ME_NAME = os.getenv("ME_NAME", "xiaojin")


class RelationalError(Exception):
    pass


def _driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for x in value:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out


def save_relational_memory(content: str, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    people = _normalize_string_list(metadata.get("people"))
    skills = _normalize_string_list(metadata.get("skills"))
    source = str(metadata.get("source", "manual"))
    created_at = datetime.now(timezone.utc).isoformat()
    memory_id = str(uuid4())

    base_query = """
    MERGE (me:Person {name: $me_name})
    MERGE (m:Memory {id: $memory_id})
    SET m.content = $content, m.source = $source, m.created_at = $created_at
    MERGE (me)-[:RECORDED]->(m)
    RETURN m.id AS id
    """
    people_query = """
    MATCH (me:Person {name: $me_name}), (m:Memory {id: $memory_id})
    UNWIND $people AS person_name
      MERGE (p:Person {name: person_name})
      MERGE (me)-[:KNOWS]->(p)
      MERGE (m)-[:MENTIONS]->(p)
    """
    skills_query = """
    MATCH (me:Person {name: $me_name}), (m:Memory {id: $memory_id})
    UNWIND $skills AS skill_name
      MERGE (s:Skill {name: skill_name})
      MERGE (me)-[:HAS_SKILL]->(s)
      MERGE (m)-[:RELATES_TO_SKILL]->(s)
    """

    try:
        with _driver() as driver:
            records, _, _ = driver.execute_query(
                base_query,
                me_name=ME_NAME,
                memory_id=memory_id,
                content=content,
                source=source,
                created_at=created_at,
            )
            if not records:
                raise RelationalError("neo4j write returned no records")
            if people:
                driver.execute_query(
                    people_query,
                    me_name=ME_NAME,
                    memory_id=memory_id,
                    people=people,
                )
            if skills:
                driver.execute_query(
                    skills_query,
                    me_name=ME_NAME,
                    memory_id=memory_id,
                    skills=skills,
                )
            return str(records[0]["id"])
    except Exception as e:  # noqa: BLE001
        raise RelationalError(str(e)) from e


def recall_relational_memory(
    query_text: str,
    top_k: int = 5,
    sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    cypher = """
    MATCH (m:Memory)
    OPTIONAL MATCH (m)-[:MENTIONS]->(p:Person)
    OPTIONAL MATCH (m)-[:RELATES_TO_SKILL]->(s:Skill)
    WHERE toLower(m.content) CONTAINS toLower($q)
       OR toLower(coalesce(p.name, "")) CONTAINS toLower($q)
       OR toLower(coalesce(s.name, "")) CONTAINS toLower($q)
    WITH m, p, s
    WHERE $sources_count = 0 OR m.source IN $sources
    RETURN m.id AS id,
           m.content AS content,
           m.source AS source,
           m.created_at AS created_at,
           collect(DISTINCT p.name) AS people,
           collect(DISTINCT s.name) AS skills
    ORDER BY m.created_at DESC
    LIMIT $limit
    """

    try:
        with _driver() as driver:
            source_list = _normalize_string_list(sources)
            records, _, _ = driver.execute_query(
                cypher,
                q=query_text,
                limit=int(top_k),
                sources=source_list,
                sources_count=len(source_list),
            )
    except Exception as e:  # noqa: BLE001
        raise RelationalError(str(e)) from e

    results: list[dict[str, Any]] = []
    for r in records:
        results.append(
            {
                "id": r.get("id"),
                "score": 0.75,
                "layer": "L3",
                "content": r.get("content", ""),
                "metadata": {
                    "created_at": r.get("created_at"),
                    "source": r.get("source"),
                    "people": [p for p in (r.get("people") or []) if p],
                    "skills": [s for s in (r.get("skills") or []) if s],
                },
            }
        )
    return results


def get_relational_stats() -> dict[str, int]:
    cypher = """
    MATCH (n) WITH count(n) AS nodes
    MATCH ()-[r]->() RETURN nodes, count(r) AS relationships
    """
    try:
        with _driver() as driver:
            records, _, _ = driver.execute_query(cypher)
        if not records:
            return {"nodes": 0, "relationships": 0}
        return {
            "nodes": int(records[0].get("nodes", 0)),
            "relationships": int(records[0].get("relationships", 0)),
        }
    except Exception as e:  # noqa: BLE001
        raise RelationalError(str(e)) from e
