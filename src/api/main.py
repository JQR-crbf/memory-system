import asyncio
from contextlib import suppress
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import yaml
from memory.embedding.embedder import EmbedderError
from memory.ingest.worklog_parser import (
    extract_people,
    extract_project_tags,
    extract_skills,
    infer_event_type,
    infer_layer,
    split_worklog_points,
)
from memory.layers.l1_semantic import recall_semantic_memory_with_filters, save_semantic_memory
from memory.layers.l2_episodic import recall_episodic_memory, save_episodic_memory
from memory.layers.l3_relational import (
    RelationalError,
    get_relational_stats,
    recall_relational_memory,
    save_relational_memory,
)
from memory.layers.l4_reflective import ReflectiveError, generate_reflection_text, save_reflection
from memory.layers.l5_aspiration import (
    AspirationError,
    load_goals,
    recall_aspiration_memory,
    save_goals,
    upsert_goal_vector,
)
from memory.layers.qdrant_client import QdrantError, get_collection_points_count
from memory.router.classifier import (
    apply_manual_feedback_recall,
    apply_manual_feedback_remember,
    classify_for_recall_with_learning,
    classify_for_remember_with_learning,
    delete_rule,
    get_classifier_settings,
    get_rules,
    log_classification,
    normalize_layers,
    set_rule_enabled,
    update_classifier_settings,
)


app = FastAPI(title="Memory System API", version="0.1.0")
PROFILE_PATH = Path("/data/L0/profile.yaml")
CLASSIFICATION_LOG_PATH = Path("/data/system/classification_log.jsonl")
CLASSIFICATION_RULES_PATH = Path("/data/system/classification_rules.json")
REFLECTION_QUALITY_LOG_PATH = Path("/data/system/reflection_quality_log.jsonl")
PROFILE_SNAPSHOT_LOG_PATH = Path("/data/system/profile_snapshots.jsonl")
PROFILE_CONFLICT_RESOLUTION_LOG_PATH = Path("/data/system/profile_conflict_resolutions.jsonl")
MEMORY_API_KEY = os.getenv("MEMORY_API_KEY", "")
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*")
CHAT_AUTOSAVE_ENABLED = os.getenv("CHAT_AUTOSAVE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
CHAT_AUTOSAVE_MIN_SCORE = float(os.getenv("CHAT_AUTOSAVE_MIN_SCORE", "0.55"))
CHAT_AUTOSAVE_LLM_ENABLED = os.getenv("CHAT_AUTOSAVE_LLM_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
CHAT_AUTOSAVE_LLM_MODEL = os.getenv("CHAT_AUTOSAVE_LLM_MODEL", os.getenv("CLASSIFIER_MODEL", os.getenv("GEN_MODEL", "qwen3:8b")))
CHAT_AUTOSAVE_LLM_TIMEOUT = float(os.getenv("CHAT_AUTOSAVE_LLM_TIMEOUT", "20"))
CHAT_AUTOSAVE_REVIEW_BAND = float(os.getenv("CHAT_AUTOSAVE_REVIEW_BAND", "0.2"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
L4_AUTO_REFLECT_ENABLED = os.getenv("L4_AUTO_REFLECT_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
L4_AUTO_REFLECT_INTERVAL_MINUTES = int(os.getenv("L4_AUTO_REFLECT_INTERVAL_MINUTES", "360"))
L4_AUTO_REFLECT_TYPE = os.getenv("L4_AUTO_REFLECT_TYPE", "weekly")
L4_AUTO_REFLECT_FOCUS_AREAS = [
    x.strip()
    for x in os.getenv("L4_AUTO_REFLECT_FOCUS_AREAS", "核心成果,成长领域,盲区发现,下一步建议").split(",")
    if x.strip()
]

_auto_reflect_task: asyncio.Task | None = None

_origins = ["*"] if CORS_ALLOW_ORIGINS.strip() == "*" else [
    x.strip() for x in CORS_ALLOW_ORIGINS.split(",") if x.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProfileUpdateRequest(BaseModel):
    path: str
    value: Any


class RememberRequest(BaseModel):
    content: str
    layer_hint: str = "auto"
    metadata: dict[str, Any] | None = None


class RecallRequest(BaseModel):
    query: str
    layers: str | list[str] = "auto"
    top_k: int = 5
    filters: dict[str, Any] | None = None
    include_profile: bool = True


class ReflectRequest(BaseModel):
    type: str = "weekly"
    period_start: str | None = None
    period_end: str | None = None
    focus_areas: list[str] | None = None


class WorklogIngestRequest(BaseModel):
    report_text: str
    report_date: str = ""
    source: str = "import_manual_worklog"
    max_items: int = 30
    dry_run: bool = False


class SynthesizeResumeRequest(BaseModel):
    target_role: str = ""
    style: str = "professional"
    top_k: int = 8
    filters: dict[str, Any] | None = None
    include_profile: bool = True


class SynthesizeWorkStatusRequest(BaseModel):
    project: str = ""
    period_start: str | None = None
    period_end: str | None = None
    style: str = "concise"
    top_k: int = 8
    filters: dict[str, Any] | None = None
    include_profile: bool = False


class SynthesizeLearningPlanRequest(BaseModel):
    goal: str = ""
    horizon_days: int = 90
    style: str = "actionable"
    top_k: int = 8
    filters: dict[str, Any] | None = None
    include_profile: bool = True


class SynthesizeSelfProfileRequest(BaseModel):
    style: str = "insightful"
    top_k: int = 8
    filters: dict[str, Any] | None = None
    include_profile: bool = True


class SynthesizeProjectRadarRequest(BaseModel):
    project: str = ""
    period_start: str | None = None
    period_end: str | None = None
    style: str = "operational"
    top_k: int = 8
    filters: dict[str, Any] | None = None
    include_profile: bool = False


class SynthesizeDecisionCopilotRequest(BaseModel):
    decision_topic: str
    options: list[str] = []
    horizon_days: int = 30
    style: str = "pragmatic"
    top_k: int = 8
    filters: dict[str, Any] | None = None
    include_profile: bool = False


class SynthesizeCareerAssetsRequest(BaseModel):
    target_role: str = ""
    asset_types: list[str] = ["resume_bullets", "review_outline", "project_case_cards"]
    style: str = "professional"
    top_k: int = 8
    filters: dict[str, Any] | None = None
    include_profile: bool = True


class SynthesizeLearningEngineRequest(BaseModel):
    goal: str = ""
    period_days: int = 7
    style: str = "actionable"
    top_k: int = 8
    review_input: str = ""
    filters: dict[str, Any] | None = None
    include_profile: bool = True


class SynthesizeCopilotRequest(BaseModel):
    query: str
    task: str = "auto"  # auto/resume/work_status/learning_plan/learning_engine/self_profile/project_radar/decision_copilot/career_assets
    project: str = ""
    target_role: str = ""
    goal: str = ""
    options: list[str] = []
    top_k: int = 8
    style: str = ""
    period_start: str | None = None
    period_end: str | None = None
    period_days: int = 7
    horizon_days: int = 90
    filters: dict[str, Any] | None = None
    include_profile: bool = True


class ProfileConflictResolveRequest(BaseModel):
    snapshot_id: str
    action: str = "accepted_as_current"  # accepted_as_current / keep_previous / merged_after_review
    reviewer_note: str = ""


class GoalCreateRequest(BaseModel):
    id: str
    title: str
    category: str = "learning"
    priority: str = "medium"
    deadline: str = ""
    status: str = "not_started"
    progress: int = 0
    sub_goals: list[dict[str, Any]] = []
    related_skills: list[str] = []
    notes: str = ""


class GoalUpdateRequest(BaseModel):
    progress: int | None = None
    status: str | None = None
    sub_goals_update: list[dict[str, Any]] | None = None
    notes: str | None = None


class ClassifierFeedbackRequest(BaseModel):
    task: str = "recall"
    text: str
    layer: str | None = None
    layers: list[str] | None = None


class ClassifierRuleDeleteRequest(BaseModel):
    task: str
    pattern: str
    layer: str | None = None
    layers: list[str] | None = None


class ClassifierRuleToggleRequest(BaseModel):
    task: str
    pattern: str
    enabled: bool = False
    layer: str | None = None
    layers: list[str] | None = None


class ClassifierSettingsUpdateRequest(BaseModel):
    learning_enabled: bool | None = None
    vector_enabled: bool | None = None
    llm_fallback_enabled: bool | None = None
    vector_threshold: float | None = None


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_chat_source(source: Any) -> bool:
    """
    Treat both plain chat and session-scoped chat as chat source.

    Examples:
    - chat
    - chat::<scope_id>
    """
    raw = str(source or "").strip().lower()
    return raw == "chat" or raw.startswith("chat::")


def _load_profile() -> dict:
    if not PROFILE_PATH.exists():
        raise HTTPException(status_code=404, detail="profile.yaml not found")

    with PROFILE_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="profile.yaml format invalid")
    return data


def _save_profile(profile: dict) -> None:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROFILE_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(profile, f, allow_unicode=True, sort_keys=False)


def _load_goals_doc() -> dict:
    return {"goals": load_goals()}


def _set_by_dotted_path(data: dict, dotted_path: str, value: Any) -> None:
    keys = [k.strip() for k in dotted_path.split(".") if k.strip()]
    if not keys:
        raise HTTPException(status_code=400, detail="path is empty")

    cursor: Any = data
    for key in keys[:-1]:
        if not isinstance(cursor, dict):
            raise HTTPException(status_code=400, detail="path points to non-object")
        if key not in cursor:
            raise HTTPException(status_code=404, detail=f"path segment not found: {key}")
        cursor = cursor[key]

    if not isinstance(cursor, dict):
        raise HTTPException(status_code=400, detail="target parent is not an object")
    leaf = keys[-1]
    if leaf not in cursor:
        raise HTTPException(status_code=404, detail=f"path segment not found: {leaf}")
    cursor[leaf] = value


def _resolve_reflection_period(period_start: str | None, period_end: str | None) -> tuple[str, str]:
    today = datetime.now(timezone.utc).date()
    end = period_end or today.isoformat()
    start = period_start or (today - timedelta(days=6)).isoformat()
    return start, end


def _compact_results(items: list[dict[str, Any]], limit: int = 8) -> list[str]:
    lines: list[str] = []
    for item in items[:limit]:
        layer = item.get("layer", "Lx")
        content = str(item.get("content", "")).strip()
        if content:
            lines.append(f"- [{layer}] {content}")
    return lines


def _build_evidence(items: list[dict[str, Any]], max_items: int = 10) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items[:max_items]:
        meta = (item.get("metadata") or {}) if isinstance(item, dict) else {}
        out.append(
            {
                "id": item.get("id"),
                "layer": item.get("layer"),
                "score": round(float(item.get("score", 0.0)), 4),
                "source": meta.get("source", ""),
                "date": meta.get("date", ""),
                "project": meta.get("project", ""),
                "snippet": str(item.get("content", ""))[:180],
            }
        )
    return out


def _resume_confidence(source_counts: dict[str, int], evidence: list[dict[str, Any]]) -> float:
    total = sum(source_counts.values())
    if total <= 0:
        return 0.25
    avg_score = 0.0
    if evidence:
        avg_score = sum(float(x.get("score", 0.0)) for x in evidence) / len(evidence)
    coverage = 0
    for key in ["L1", "L2", "L3", "L5"]:
        if source_counts.get(key, 0) > 0:
            coverage += 1
    # weighted: retrieval quality + layer coverage
    conf = 0.4 + min(0.35, avg_score * 0.35) + min(0.25, coverage * 0.06)
    return round(max(0.0, min(1.0, conf)), 3)


def _learning_priority(evidence: list[dict[str, Any]], l5_count: int) -> str:
    if not evidence:
        return "medium"
    avg_score = sum(float(x.get("score", 0.0)) for x in evidence[:6]) / max(1, min(6, len(evidence)))
    if l5_count > 0 and avg_score >= 0.6:
        return "high"
    if avg_score >= 0.45:
        return "medium"
    return "medium"


def _extract_focus_topics(evidence: list[dict[str, Any]], max_items: int = 4) -> list[str]:
    topics: list[str] = []
    for item in evidence:
        project = str(item.get("project", "")).strip()
        if project and project not in topics:
            topics.append(project)
        if len(topics) >= max_items:
            break
    return topics


def _risk_signal_scan(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    risk_words = ["风险", "阻塞", "延期", "延迟", "问题", "bug", "失败", "依赖", "卡点", "异常"]
    positive_words = ["完成", "上线", "优化", "推进", "稳定", "达成", "已解决"]
    risk_hits = 0
    positive_hits = 0
    for item in evidence:
        text = str(item.get("snippet", "")).lower()
        if any(k in text for k in risk_words):
            risk_hits += 1
        if any(k in text for k in positive_words):
            positive_hits += 1
    score = max(0, min(100, 50 + risk_hits * 10 - positive_hits * 4))
    if score >= 70:
        level = "high"
    elif score >= 45:
        level = "medium"
    else:
        level = "low"
    return {
        "score": score,
        "level": level,
        "risk_hits": risk_hits,
        "positive_hits": positive_hits,
    }


def _detect_copilot_task(query: str) -> str:
    q = query.strip().lower()
    if re.search(r"(简历|cv|履历)", q):
        return "resume"
    if re.search(r"(述职提纲|项目案例|案例卡|简历要点|职业资产)", q):
        return "career_assets"
    if re.search(r"(我是谁|你怎么看我|自我画像|我的画像|了解我)", q):
        return "self_profile"
    if re.search(r"(风险|阻塞|卡点|里程碑|延期|项目状态|项目雷达)", q):
        return "project_radar"
    if re.search(r"(怎么决策|该选哪个|方案对比|取舍|决策建议)", q):
        return "decision_copilot"
    if re.search(r"(学什么|怎么学|学习任务|学习路径|复盘计划)", q):
        return "learning_engine"
    if re.search(r"(学习计划|90天|60天|30天)", q):
        return "learning_plan"
    if re.search(r"(最近进展|当前进展|工作情况|到哪个阶段|处理到哪|状态)", q):
        return "work_status"
    return "work_status"


def _build_weekly_tasks(evidence: list[dict[str, Any]], max_items: int = 4) -> list[str]:
    tasks: list[str] = []
    for item in evidence:
        snippet = str(item.get("snippet", "")).strip()
        if not snippet:
            continue
        task = f"围绕「{snippet[:36]}」完成一个可验证的小闭环（输出文档/结果记录）。"
        if task not in tasks:
            tasks.append(task)
        if len(tasks) >= max_items:
            break
    if not tasks:
        tasks = [
            "梳理本周学习目标，明确1个核心能力点。",
            "完成1次与当前项目强相关的实战练习并记录复盘。",
            "将学习产出沉淀为可复用模板或笔记。",
        ]
    return tasks


def _to_norm_set(values: list[str] | None) -> set[str]:
    out: set[str] = set()
    for v in values or []:
        text = str(v).strip().lower()
        if text:
            out.add(text)
    return out


def _detect_profile_conflicts(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    if not previous:
        return {"has_conflict": False, "items": []}

    items: list[dict[str, Any]] = []

    prev_name = str(previous.get("name", "")).strip()
    curr_name = str(current.get("name", "")).strip()
    if prev_name and curr_name and prev_name != curr_name:
        items.append(
            {
                "field": "name",
                "severity": "high",
                "previous": prev_name,
                "current": curr_name,
                "reason": "name_changed",
            }
        )

    prev_role = str(previous.get("current_role", "")).strip()
    curr_role = str(current.get("current_role", "")).strip()
    if prev_role and curr_role and prev_role != curr_role:
        items.append(
            {
                "field": "current_role",
                "severity": "medium",
                "previous": prev_role,
                "current": curr_role,
                "reason": "role_changed",
            }
        )

    prev_topics = _to_norm_set(previous.get("focus_topics", []))
    curr_topics = _to_norm_set(current.get("focus_topics", []))
    if prev_topics and curr_topics:
        overlap = len(prev_topics & curr_topics) / max(1, len(prev_topics | curr_topics))
        if overlap < 0.2:
            items.append(
                {
                    "field": "focus_topics",
                    "severity": "medium",
                    "previous": sorted(prev_topics),
                    "current": sorted(curr_topics),
                    "reason": "topic_drift",
                    "overlap": round(overlap, 3),
                }
            )

    prev_strength = _to_norm_set(previous.get("strength_signals", []))
    curr_strength = _to_norm_set(current.get("strength_signals", []))
    if prev_strength and curr_strength:
        overlap = len(prev_strength & curr_strength) / max(1, len(prev_strength | curr_strength))
        if overlap < 0.15:
            items.append(
                {
                    "field": "strength_signals",
                    "severity": "low",
                    "reason": "strength_shift",
                    "overlap": round(overlap, 3),
                }
            )

    return {"has_conflict": bool(items), "items": items}


def _merge_list_filters(base: list[str] | None, extra: list[str] | None) -> list[str] | None:
    merged: list[str] = []
    for item in (base or []) + (extra or []):
        if item and item not in merged:
            merged.append(item)
    return merged or None


def _has_explicit_sources(filters: dict[str, Any]) -> bool:
    """Check if caller already provided any explicit source filters."""
    return bool(
        filters.get("l1_sources")
        or filters.get("l2_sources")
        or filters.get("l3_sources")
        or filters.get("sources")
    )


def _business_filter_hint(query: str, filters: dict[str, Any] | None) -> dict[str, Any]:
    """
    Enrich caller-provided filters with business hints, but never override explicit intent.

    - If query中包含“51talk”，且调用方没有指定任何 sources，则优先使用 acceptance 知识库。
    - 如果 filters 里传入了 date，但没有 date_from/date_to，则自动映射为一个日范围。
    """
    base = filters or {}
    out: dict[str, Any] = dict(base)

    # Normalize single "date" into date_from/date_to for episodic queries.
    raw_date = out.get("date")
    if isinstance(raw_date, str) and raw_date and not out.get("date_from") and not out.get("date_to"):
        out["date_from"] = raw_date
        out["date_to"] = raw_date

    q = query.lower()
    # For user-facing business queries, prefer business-tagged memories,
    # but only when caller did NOT specify custom sources.
    if "51talk" in q and not _has_explicit_sources(base):
        out["l1_tags"] = _merge_list_filters(base.get("l1_tags"), ["51Talk"])
        out["l1_sources"] = _merge_list_filters(base.get("l1_sources"), ["acceptance"])
        out["l2_sources"] = _merge_list_filters(base.get("l2_sources"), ["acceptance"])
        if not out.get("project"):
            out["project"] = "51Talk AI平台"

    return out


def _score_chat_usefulness(text: str, metadata: dict[str, Any]) -> tuple[float, list[str]]:
    content = text.strip()
    lowered = content.lower()
    reasons: list[str] = []
    score = 0.2

    # Strong noise patterns for chit-chat / acknowledgements.
    noise_patterns = [
        r"^(好|好的|收到|明白了|嗯|嗯嗯|ok|okay|thanks|谢谢|谢了)[!！。,. ]*$",
        r"^(哈哈|呵呵|😂|👍|👌|🙏|[~\.\!\?！？]+)$",
        r"^(在吗|有人吗|测试|test)[!！。,. ]*$",
    ]
    if any(re.match(p, content, flags=re.IGNORECASE) for p in noise_patterns):
        return 0.05, ["noise_pattern"]

    # Useful intent clues.
    if any(k in content for k in ["记住", "请记住", "我学会", "我负责", "我的目标", "我计划", "我完成"]):
        score += 0.45
        reasons.append("explicit_memory_intent")
    if any(k in content for k in ["今天", "本周", "昨天", "刚刚", "进展", "上线", "复盘", "问题", "风险"]):
        score += 0.2
        reasons.append("event_signal")
    if any(k in content for k in ["同事", "协作", "合作", "关系", "团队", "客户"]):
        score += 0.2
        reasons.append("relation_signal")
    if any(k in content for k in ["目标", "计划", "里程碑", "deadline", "学习", "优先级"]):
        score += 0.2
        reasons.append("goal_signal")

    # Structural features.
    if len(content) >= 20:
        score += 0.08
        reasons.append("length_ok")
    if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", content):
        score += 0.08
        reasons.append("has_date")
    if re.search(r"[A-Za-z]{2,}", lowered):
        score += 0.04
        reasons.append("has_terms")

    # If metadata explicitly marks importance, trust it.
    if metadata.get("importance") is not None:
        score += 0.15
        reasons.append("metadata_importance")
    if metadata.get("source") == "acceptance":
        score += 0.2
        reasons.append("acceptance_source")

    final_score = max(0.0, min(1.0, score))
    return final_score, reasons


async def _llm_judge_chat_usefulness(text: str, metadata: dict[str, Any]) -> tuple[float, str]:
    prompt = f"""
你是记忆系统的“聊天价值评估器”。
请判断这条用户输入是否值得写入长期记忆。

高价值样例：
- 事实、经验、结论、偏好、目标、计划、关系、关键事件进展
低价值样例：
- 寒暄、确认语、口头语、纯情绪符号、无上下文短句

用户输入：
{text}

metadata:
{json.dumps(metadata, ensure_ascii=False)}

仅输出 JSON:
{{"score":0.0,"reason":"...","useful":false}}
score 范围 0~1。
"""
    try:
        async with httpx.AsyncClient(timeout=CHAT_AUTOSAVE_LLM_TIMEOUT) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": CHAT_AUTOSAVE_LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0, "top_p": 1},
                },
            )
        if resp.status_code != 200:
            return -1.0, "llm_unavailable"
        raw = (resp.json().get("response") or "").strip()
        data = json.loads(raw) if raw else {}
        score = float(data.get("score", -1))
        reason = str(data.get("reason", "llm_scored"))
        if score < 0 or score > 1:
            return -1.0, "llm_invalid_score"
        return score, reason
    except Exception:
        return -1.0, "llm_exception"


def _expanded_layers_for_empty_result(layers: list[str]) -> list[str]:
    if not layers:
        return layers
    if layers == ["L3"]:
        return ["L1", "L3"]
    if layers == ["L2"]:
        return ["L1", "L2"]
    if layers == ["L5"]:
        return ["L1", "L2", "L5"]
    if layers == ["L1"]:
        return ["L1", "L2"]

    expanded = list(layers)
    if "L1" not in expanded:
        expanded.append("L1")
    if "L2" not in expanded:
        expanded.append("L2")
    return normalize_layers(expanded)


async def _recall_by_layers(
    query: str,
    top_k: int,
    layers: list[str],
    filters: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    tasks: list[asyncio.Task] = []
    if "L1" in layers:
        tasks.append(
            asyncio.create_task(
                recall_semantic_memory_with_filters(
                    query,
                    top_k,
                    categories=filters.get("categories"),
                    min_importance=_to_int_or_none(filters.get("min_importance")),
                    tags=filters.get("l1_tags") or filters.get("tags"),
                    sources=filters.get("l1_sources") or filters.get("sources"),
                )
            )
        )
    if "L2" in layers:
        tasks.append(
            asyncio.create_task(
                recall_episodic_memory(
                    query,
                    top_k,
                    date_from=filters.get("date_from"),
                    date_to=filters.get("date_to"),
                    event_types=filters.get("event_types"),
                    project=filters.get("project"),
                    tags=filters.get("l2_tags") or filters.get("tags"),
                    sources=filters.get("l2_sources") or filters.get("sources"),
                )
            )
        )
    if "L3" in layers:
        tasks.append(
            asyncio.create_task(
                asyncio.to_thread(
                    recall_relational_memory,
                    query,
                    top_k,
                    filters.get("l3_sources") or filters.get("sources"),
                )
            )
        )
    if "L5" in layers:
        tasks.append(asyncio.create_task(recall_aspiration_memory(query, top_k)))

    if tasks:
        grouped = await asyncio.gather(*tasks)
        for one_layer_results in grouped:
            results.extend(one_layer_results)
    return results


def _load_classification_events(action: str | None = None) -> list[dict[str, Any]]:
    if not CLASSIFICATION_LOG_PATH.exists():
        return []
    items: list[dict[str, Any]] = []
    with CLASSIFICATION_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if action and str(event.get("action", "")).lower() != action.lower():
                continue
            items.append(event)
    return items


def _load_classification_rules() -> dict[str, Any]:
    if not CLASSIFICATION_RULES_PATH.exists():
        return {"remember": [], "recall": []}
    try:
        with CLASSIFICATION_RULES_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except (json.JSONDecodeError, OSError):
        return {"remember": [], "recall": []}
    remember = data.get("remember")
    recall = data.get("recall")
    return {
        "remember": remember if isinstance(remember, list) else [],
        "recall": recall if isinstance(recall, list) else [],
    }


def _append_reflection_quality_log(payload: dict[str, Any]) -> None:
    REFLECTION_QUALITY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REFLECTION_QUALITY_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _append_profile_snapshot(payload: dict[str, Any]) -> None:
    PROFILE_SNAPSHOT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROFILE_SNAPSHOT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_reflection_quality_log(recent: int = 20) -> list[dict[str, Any]]:
    if not REFLECTION_QUALITY_LOG_PATH.exists():
        return []
    records: list[dict[str, Any]] = []
    with REFLECTION_QUALITY_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    recent = max(1, min(200, int(recent)))
    return records[-recent:]


def _load_profile_snapshots(recent: int = 20) -> list[dict[str, Any]]:
    if not PROFILE_SNAPSHOT_LOG_PATH.exists():
        return []
    records: list[dict[str, Any]] = []
    with PROFILE_SNAPSHOT_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    recent = max(1, min(500, int(recent)))
    return records[-recent:]


def _load_profile_conflict_resolutions() -> list[dict[str, Any]]:
    if not PROFILE_CONFLICT_RESOLUTION_LOG_PATH.exists():
        return []
    records: list[dict[str, Any]] = []
    with PROFILE_CONFLICT_RESOLUTION_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return records


def _append_profile_conflict_resolution(payload: dict[str, Any]) -> None:
    PROFILE_CONFLICT_RESOLUTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROFILE_CONFLICT_RESOLUTION_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _profile_conflict_strategy(conflict: dict[str, Any]) -> dict[str, Any]:
    if not conflict.get("has_conflict"):
        return {"recommended_action": "accepted_as_current", "reason": "no_conflict"}
    items = conflict.get("items", [])
    has_high = any(str(x.get("severity", "")).lower() == "high" for x in items if isinstance(x, dict))
    has_medium = any(str(x.get("severity", "")).lower() == "medium" for x in items if isinstance(x, dict))
    if has_high:
        return {"recommended_action": "manual_review_required", "reason": "high_severity_conflict"}
    if has_medium:
        return {"recommended_action": "review_suggested", "reason": "medium_severity_conflict"}
    return {"recommended_action": "accepted_as_current", "reason": "low_severity_only"}


def _evaluate_reflection_quality(
    reflection_text: str,
    source_counts: dict[str, int],
    focus_areas: list[str],
) -> dict[str, Any]:
    text = reflection_text.strip()
    score = 0.35
    signals: list[str] = []

    # Length quality
    text_len = len(text)
    if 180 <= text_len <= 900:
        score += 0.2
        signals.append("length_good")
    elif text_len >= 120:
        score += 0.1
        signals.append("length_ok")
    else:
        signals.append("length_short")

    # Structure quality
    structure_keys = ["核心成果", "成长领域", "盲区发现", "下一步", "建议", "行动"]
    hits = sum(1 for k in structure_keys if k in text)
    if hits >= 4:
        score += 0.2
        signals.append("structure_complete")
    elif hits >= 2:
        score += 0.1
        signals.append("structure_partial")

    # Actionability quality
    action_patterns = [r"\d\)", r"第[一二三四五六七八九十]", r"建议", r"行动", r"下一步"]
    if any(re.search(p, text) for p in action_patterns):
        score += 0.15
        signals.append("actionable")

    # Source coverage quality
    non_empty_layers = sum(1 for _, c in source_counts.items() if int(c) > 0)
    if non_empty_layers >= 3:
        score += 0.15
        signals.append("source_coverage_good")
    elif non_empty_layers >= 2:
        score += 0.08
        signals.append("source_coverage_ok")

    # Focus coverage
    focus_hits = sum(1 for fa in focus_areas if fa and fa in text)
    if focus_areas and focus_hits >= max(1, len(focus_areas) // 2):
        score += 0.1
        signals.append("focus_covered")

    final_score = round(max(0.0, min(1.0, score)), 3)
    level = "high" if final_score >= 0.8 else ("medium" if final_score >= 0.6 else "low")
    return {
        "score": final_score,
        "level": level,
        "signals": signals,
        "text_length": text_len,
        "non_empty_layers": non_empty_layers,
        "focus_hits": focus_hits,
    }


async def _run_reflection_job(
    reflect_type: str,
    period_start: str | None,
    period_end: str | None,
    focus_areas: list[str] | None,
    trigger: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    p_start, p_end = _resolve_reflection_period(period_start, period_end)
    focus = focus_areas or ["核心成果", "成长领域", "盲区发现", "下一步建议"]
    is_self_profile = reflect_type in {"self_profile", "profile", "self_summary"}

    l1_task = asyncio.create_task(
        recall_semantic_memory_with_filters(
            "本周我掌握了哪些知识和方法",
            top_k=6,
        )
    )
    l2_task = asyncio.create_task(
        recall_episodic_memory(
            "本周我做了什么",
            top_k=8,
            date_from=p_start,
            date_to=p_end,
        )
    )
    l3_task = asyncio.create_task(asyncio.to_thread(recall_relational_memory, "合作 关系 技能", 6))
    l1_results, l2_results, l3_results = await asyncio.gather(l1_task, l2_task, l3_task)
    goals = _load_goals_doc()
    goals_preview = goals.get("goals", [])[:5]
    if is_self_profile:
        prompt = "\n".join(
            [
                "你是一个个人画像与成长档案整理器。",
                "请基于以下记忆，为“我”生成一份最新的自我画像总结，用中文，结构清晰，控制在500字以内。",
                "",
                f"时间范围：{p_start} 到 {p_end}",
                "你的目标是回答：我最近这段时间“在做什么、擅长什么、正在变成一个什么样的人”。",
                "",
                "## L2 事件记录（按时间线）",
                *(_compact_results(l2_results) or ["- 暂无"]),
                "",
                "## L1 知识与技能（抽象能力）",
                *(_compact_results(l1_results) or ["- 暂无"]),
                "",
                "## L3 关系与协作（和谁一起做事）",
                *(_compact_results(l3_results) or ["- 暂无"]),
                "",
                f"## L5 目标（前5条）\n{goals_preview}",
                "",
                "请严格按以下结构输出，不要加序言或总结性客套话：",
                "1) 角色与身份画像：我目前在做什么角色、主要负责哪些方向。",
                "2) 核心能力与代表性经验：用条目列出我的关键技能、代表项目或成果。",
                "3) 当前关注主题与长期项目：列出当前 2-4 个正在推进的主题（如学情数据、RAG、S9 平台等）。",
                "4) 工作与学习偏好：从记忆中总结出我的做事风格、沟通偏好、规划方式。",
                "5) 风险与盲区（可选）：如果从这些记忆中能看到明显短板或风险，请简要指出。",
            ]
        )
    else:
        prompt = "\n".join(
            [
                f"请基于以下信息生成一份{reflect_type}反思总结，语言用中文，结构清晰，控制在400字内。",
                "",
                f"时间范围：{p_start} 到 {p_end}",
                f"重点关注：{', '.join(focus)}",
                "",
                "## L2 事件记录",
                *(_compact_results(l2_results) or ["- 暂无"]),
                "",
                "## L1 知识与技能",
                *(_compact_results(l1_results) or ["- 暂无"]),
                "",
                "## L3 关系与协作",
                *(_compact_results(l3_results) or ["- 暂无"]),
                "",
                f"## L5 目标（前5条）\n{goals_preview}",
                "",
                "请按以下结构输出：",
                "1) 核心成果",
                "2) 成长领域",
                "3) 盲区发现",
                "4) 下一步行动建议（3条）",
            ]
        )

    reflection_text = await generate_reflection_text(prompt)
    reflection_id = await save_reflection(
        content=reflection_text,
        reflection_type=reflect_type,
        period_start=p_start,
        period_end=p_end,
        focus_areas=focus,
        source_layers=["L1", "L2", "L3", "L5"],
    )
    source_counts = {
        "L1": len(l1_results),
        "L2": len(l2_results),
        "L3": len(l3_results),
        "L5": len(goals.get("goals", [])),
    }
    quality = _evaluate_reflection_quality(reflection_text, source_counts, focus)
    quality_record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trigger": trigger,
        "reflection_id": reflection_id,
        "reflection_type": reflect_type,
        "period_start": p_start,
        "period_end": p_end,
        "source_counts": source_counts,
        "quality": quality,
    }
    _append_reflection_quality_log(quality_record)

    result: dict[str, Any] = {
        "id": reflection_id,
        "type": reflect_type,
        "period": f"{p_start} ~ {p_end}",
        "reflection": {"content": reflection_text},
        "source_counts": source_counts,
        "quality": quality,
        "status": "saved_to_L4",
        "trigger": trigger,
        "total_time_ms": round((time.perf_counter() - started) * 1000, 2),
    }

    # 对于 self_profile 类型反思，再额外写入一条 L1 语义记忆，作为“关于我的最新画像”。
    if is_self_profile:
        profile_meta: dict[str, Any] = {
            "source": "reflect_self_profile",
            "category": "self_profile",
            "importance": 5,
            "period_start": p_start,
            "period_end": p_end,
            "reflection_id": reflection_id,
        }
        profile_memory_id = await save_semantic_memory(reflection_text, profile_meta)
        log_classification(
            {
                "action": "remember",
                "classification_method": "reflect:self_profile",
                "layer": "L1",
                "content_preview": reflection_text[:80],
            }
        )
        result["profile_memory_id"] = profile_memory_id

    return result


async def _ingest_worklog_by_shared_rules(
    report_text: str,
    report_date: str = "",
    source: str = "import_manual_worklog",
    max_items: int = 30,
    dry_run: bool = False,
) -> dict[str, Any]:
    points = split_worklog_points(report_text, max_points=max(1, min(int(max_items), 200)))
    if not points:
        return {"status": "skipped", "reason": "no_valid_points", "input_points": 0}

    saved = 0
    errors = 0
    by_layer = {"L1": 0, "L2": 0, "L3": 0}
    failed: list[str] = []

    for title, point in points:
        people = extract_people(point)
        skills = extract_skills(point)
        tags = extract_project_tags(point)
        layer = infer_layer(title, point, people)
        event_type = infer_event_type(title, point)

        metadata: dict[str, Any] = {
            "source": source or "import_manual_worklog",
            "event_type": event_type,
            "tags": tags,
            "project": tags[0] if tags else "51Talk",
            "importance": 3,
        }
        if report_date:
            metadata["date"] = report_date
        if layer == "L3":
            metadata["people"] = people
            metadata["skills"] = skills
        elif layer == "L1":
            metadata["category"] = "knowledge"
            metadata["importance"] = 4

        if dry_run:
            saved += 1
            by_layer[layer] += 1
            continue

        try:
            if layer == "L1":
                await save_semantic_memory(point, metadata)
            elif layer == "L2":
                await save_episodic_memory(point, metadata)
            elif layer == "L3":
                await asyncio.to_thread(save_relational_memory, point, metadata)
            else:
                raise HTTPException(status_code=400, detail=f"unsupported layer: {layer}")
            saved += 1
            by_layer[layer] += 1
        except (EmbedderError, QdrantError, RelationalError, AspirationError, HTTPException) as e:
            errors += 1
            failed.append(str(e))

    return {
        "status": "done",
        "input_points": len(points),
        "saved": saved,
        "errors": errors,
        "by_layer": by_layer,
        "failed_samples": failed[:3],
        "dry_run": dry_run,
    }


async def _auto_reflect_loop() -> None:
    interval_s = max(60, L4_AUTO_REFLECT_INTERVAL_MINUTES * 60)
    # Delay first run a bit to avoid startup spikes.
    await asyncio.sleep(20)
    while True:
        try:
            result = await _run_reflection_job(
                reflect_type=L4_AUTO_REFLECT_TYPE,
                period_start=None,
                period_end=None,
                focus_areas=L4_AUTO_REFLECT_FOCUS_AREAS,
                trigger="scheduler",
            )
            log_classification(
                {
                    "action": "reflect_auto",
                    "reflection_id": result.get("id"),
                    "quality_score": result.get("quality", {}).get("score"),
                    "status": "saved_to_L4",
                }
            )
        except (EmbedderError, QdrantError, RelationalError, ReflectiveError) as e:
            log_classification(
                {
                    "action": "reflect_auto",
                    "status": "failed",
                    "error": str(e),
                }
            )
        await asyncio.sleep(interval_s)


@app.on_event("startup")
async def startup_auto_reflect() -> None:
    global _auto_reflect_task
    if not L4_AUTO_REFLECT_ENABLED:
        return
    if _auto_reflect_task is None or _auto_reflect_task.done():
        _auto_reflect_task = asyncio.create_task(_auto_reflect_loop())


@app.on_event("shutdown")
async def shutdown_auto_reflect() -> None:
    global _auto_reflect_task
    if _auto_reflect_task is None:
        return
    _auto_reflect_task.cancel()
    with suppress(asyncio.CancelledError):
        await _auto_reflect_task
    _auto_reflect_task = None


@app.middleware("http")
async def optional_api_key_guard(request: Request, call_next):
    if not MEMORY_API_KEY:
        return await call_next(request)

    public_paths = {"/health", "/docs", "/openapi.json"}
    if request.url.path in public_paths:
        return await call_next(request)

    api_key = request.headers.get("x-api-key", "")
    auth_header = request.headers.get("authorization", "")
    bearer_key = ""
    if auth_header.lower().startswith("bearer "):
        bearer_key = auth_header[7:].strip()

    if api_key != MEMORY_API_KEY and bearer_key != MEMORY_API_KEY:
        return JSONResponse(status_code=401, content={"detail": "invalid or missing API key"})

    return await call_next(request)


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {
            "fastapi": "running",
            "qdrant_url": os.getenv("QDRANT_URL", ""),
            "neo4j_uri": os.getenv("NEO4J_URI", ""),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", ""),
        },
    }


@app.get("/stats")
async def stats() -> dict:
    try:
        qdrant_counts = await asyncio.gather(
            get_collection_points_count("semantic_memory"),
            get_collection_points_count("episodic_memory"),
            get_collection_points_count("reflective_memory"),
            get_collection_points_count("aspiration_memory"),
        )
        l1_count, l2_count, l4_count, l5_vector_count = qdrant_counts
        l3_stats = await asyncio.to_thread(get_relational_stats)
    except (QdrantError, RelationalError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    goals = load_goals()
    total_memories = l1_count + l2_count + l4_count + l5_vector_count

    return {
        "total_memories": total_memories,
        "by_layer": {
            "L0": {"profile_exists": PROFILE_PATH.exists()},
            "L1": l1_count,
            "L2": l2_count,
            "L3": l3_stats,
            "L4": l4_count,
            "L5": {
                "goals": len(goals),
                "vectors": l5_vector_count,
            },
        },
    }


@app.get("/classifier/debug/last")
def classifier_debug_last(action: str | None = None, recent: int = 10, with_rules: bool = True) -> dict:
    valid_actions = {"remember", "recall", "reflect"}
    if action and action not in valid_actions:
        raise HTTPException(status_code=400, detail=f"unsupported action: {action}")

    recent = max(1, min(50, int(recent)))
    events = _load_classification_events(action=action)
    last_event = events[-1] if events else None
    recent_events = events[-recent:] if events else []

    response: dict[str, Any] = {
        "filter": {
            "action": action or "all",
            "recent": recent,
        },
        "total_matched_events": len(events),
        "last_event": last_event,
        "recent_events": recent_events,
    }

    if with_rules:
        rules = _load_classification_rules()
        response["rules_overview"] = {
            "remember_count": len(rules.get("remember", [])),
            "recall_count": len(rules.get("recall", [])),
            "remember_top3": sorted(
                rules.get("remember", []), key=lambda x: int(x.get("hits", 0)), reverse=True
            )[:3],
            "recall_top3": sorted(
                rules.get("recall", []), key=lambda x: int(x.get("hits", 0)), reverse=True
            )[:3],
        }

    return response


@app.get("/classifier/rules")
def classifier_rules(task: str | None = None, limit: int = 200, include_disabled: bool = True) -> dict:
    if task and task not in {"remember", "recall"}:
        raise HTTPException(status_code=400, detail=f"unsupported task: {task}")
    return {
        "task": task or "all",
        "rules": get_rules(task=task, limit=limit, include_disabled=include_disabled),
    }


@app.post("/classifier/feedback")
async def classifier_feedback(payload: ClassifierFeedbackRequest) -> dict:
    task = payload.task.strip().lower()
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text cannot be empty")

    if task == "remember":
        if not payload.layer:
            raise HTTPException(status_code=400, detail="remember feedback requires layer")
        result = await apply_manual_feedback_remember(text, payload.layer)
    elif task == "recall":
        if not payload.layers:
            raise HTTPException(status_code=400, detail="recall feedback requires layers")
        result = await apply_manual_feedback_recall(text, payload.layers)
    else:
        raise HTTPException(status_code=400, detail=f"unsupported task: {payload.task}")

    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=str(result.get("detail", "invalid feedback")))

    log_classification(
        {
            "action": "classifier_feedback",
            "task": task,
            "text_preview": text[:120],
            "layer": payload.layer,
            "layers": payload.layers,
            "status": "accepted",
        }
    )
    return {"status": "accepted", "result": result}


@app.post("/classifier/rules/delete")
def classifier_rules_delete(payload: ClassifierRuleDeleteRequest) -> dict:
    task = payload.task.strip().lower()
    if task not in {"remember", "recall"}:
        raise HTTPException(status_code=400, detail=f"unsupported task: {payload.task}")
    deleted = delete_rule(
        task=task,
        pattern=payload.pattern,
        layer=payload.layer,
        layers=payload.layers,
    )
    if deleted:
        log_classification(
            {
                "action": "classifier_rule_delete",
                "task": task,
                "pattern": payload.pattern,
                "layer": payload.layer,
                "layers": payload.layers,
                "status": "deleted",
            }
        )
    return {"status": "ok", "deleted": deleted}


@app.post("/classifier/rules/toggle")
def classifier_rules_toggle(payload: ClassifierRuleToggleRequest) -> dict:
    task = payload.task.strip().lower()
    if task not in {"remember", "recall"}:
        raise HTTPException(status_code=400, detail=f"unsupported task: {payload.task}")
    changed = set_rule_enabled(
        task=task,
        pattern=payload.pattern,
        enabled=payload.enabled,
        layer=payload.layer,
        layers=payload.layers,
    )
    if changed:
        log_classification(
            {
                "action": "classifier_rule_toggle",
                "task": task,
                "pattern": payload.pattern,
                "enabled": payload.enabled,
                "layer": payload.layer,
                "layers": payload.layers,
            }
        )
    return {"status": "ok", "changed": changed}


@app.get("/classifier/settings")
def classifier_settings() -> dict:
    return {"settings": get_classifier_settings()}


@app.post("/classifier/settings")
def classifier_settings_update(payload: ClassifierSettingsUpdateRequest) -> dict:
    updates = payload.model_dump(exclude_none=True)
    if "vector_threshold" in updates:
        v = float(updates["vector_threshold"])
        if v < 0 or v > 1:
            raise HTTPException(status_code=400, detail="vector_threshold must be between 0 and 1")
    settings = update_classifier_settings(updates)
    log_classification(
        {
            "action": "classifier_settings_update",
            "updates": updates,
        }
    )
    return {"status": "updated", "settings": settings}


@app.get("/profile")
def get_profile() -> dict:
    return {"profile": _load_profile()}


@app.put("/profile")
def update_profile(payload: ProfileUpdateRequest) -> dict:
    profile = _load_profile()
    _set_by_dotted_path(profile, payload.path, payload.value)
    _save_profile(profile)
    return {
        "status": "updated",
        "path": payload.path,
        "value": payload.value,
    }


@app.get("/profile/snapshots")
def get_profile_snapshots(recent: int = 20) -> dict:
    records = _load_profile_snapshots(recent=recent)
    enriched: list[dict[str, Any]] = []
    for idx, rec in enumerate(records):
        one = dict(rec)
        one.setdefault("snapshot_id", f"snapshot_{idx+1}")
        enriched.append(one)
    return {
        "recent": recent,
        "count": len(enriched),
        "records": enriched,
        "snapshot_log_exists": PROFILE_SNAPSHOT_LOG_PATH.exists(),
    }


@app.get("/profile/conflicts")
def get_profile_conflicts(recent: int = 50, only_unresolved: bool = True) -> dict:
    snapshots = _load_profile_snapshots(recent=recent)
    resolutions = _load_profile_conflict_resolutions()
    resolution_map = {
        str(x.get("snapshot_id", "")).strip(): x
        for x in resolutions
        if isinstance(x, dict) and str(x.get("snapshot_id", "")).strip()
    }

    items: list[dict[str, Any]] = []
    for idx, snap in enumerate(snapshots):
        snapshot_id = str(snap.get("snapshot_id") or f"snapshot_{idx+1}")
        conflict = snap.get("conflict") or {}
        if not bool(conflict.get("has_conflict")):
            continue
        resolution = resolution_map.get(snapshot_id)
        if only_unresolved and resolution:
            continue
        items.append(
            {
                "snapshot_id": snapshot_id,
                "created_at": snap.get("created_at"),
                "conflict": conflict,
                "strategy": _profile_conflict_strategy(conflict),
                "resolution": resolution,
                "profile": snap.get("profile", {}),
            }
        )

    return {
        "recent": recent,
        "count": len(items),
        "only_unresolved": only_unresolved,
        "items": items,
        "resolution_log_exists": PROFILE_CONFLICT_RESOLUTION_LOG_PATH.exists(),
    }


@app.post("/profile/conflicts/resolve")
def resolve_profile_conflict(payload: ProfileConflictResolveRequest) -> dict:
    snapshot_id = payload.snapshot_id.strip()
    if not snapshot_id:
        raise HTTPException(status_code=400, detail="snapshot_id cannot be empty")
    allowed_actions = {"accepted_as_current", "keep_previous", "merged_after_review"}
    action = payload.action.strip() or "accepted_as_current"
    if action not in allowed_actions:
        raise HTTPException(status_code=400, detail=f"unsupported action: {action}")

    resolution = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": snapshot_id,
        "action": action,
        "reviewer_note": payload.reviewer_note.strip(),
    }
    _append_profile_conflict_resolution(resolution)
    return {"status": "resolved", "resolution": resolution}


@app.get("/goals")
def get_goals() -> dict:
    goals = load_goals()
    return {"goals": goals}


@app.post("/goals")
async def create_goal(payload: GoalCreateRequest) -> dict:
    goals = load_goals()
    if any(str(g.get("id")) == payload.id for g in goals):
        raise HTTPException(status_code=409, detail=f"goal already exists: {payload.id}")

    goal = payload.model_dump()
    goals.append(goal)
    save_goals(goals)
    try:
        vector_id = await upsert_goal_vector(goal)
    except (EmbedderError, QdrantError, AspirationError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return {
        "status": "created",
        "goal_id": payload.id,
        "vector_id": vector_id,
    }


@app.put("/goals/{goal_id}")
async def update_goal(goal_id: str, payload: GoalUpdateRequest) -> dict:
    goals = load_goals()
    idx = next((i for i, g in enumerate(goals) if str(g.get("id")) == goal_id), -1)
    if idx < 0:
        raise HTTPException(status_code=404, detail=f"goal not found: {goal_id}")

    goal = goals[idx]
    if payload.progress is not None:
        goal["progress"] = max(0, min(100, int(payload.progress)))
    if payload.status is not None:
        goal["status"] = payload.status
    if payload.notes is not None:
        goal["notes"] = payload.notes
    if payload.sub_goals_update:
        existing = goal.get("sub_goals")
        if not isinstance(existing, list):
            existing = []
        status_map = {
            str(x.get("title")): str(x.get("status"))
            for x in payload.sub_goals_update
            if x.get("title") and x.get("status")
        }
        updated: list[dict[str, Any]] = []
        for item in existing:
            title = str(item.get("title", ""))
            if title in status_map:
                item["status"] = status_map[title]
            updated.append(item)
        goal["sub_goals"] = updated

    goals[idx] = goal
    save_goals(goals)
    try:
        vector_id = await upsert_goal_vector(goal)
    except (EmbedderError, QdrantError, AspirationError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return {"status": "updated", "goal_id": goal_id, "vector_id": vector_id, "goal": goal}


@app.post("/remember")
async def remember(payload: RememberRequest) -> dict:
    if not payload.content.strip():
        raise HTTPException(status_code=400, detail="content cannot be empty")

    metadata = payload.metadata or {}
    if CHAT_AUTOSAVE_ENABLED and _is_chat_source(metadata.get("source", "")):
        usefulness_score, reasons = _score_chat_usefulness(payload.content, metadata)
        llm_score = -1.0
        llm_reason = ""
        lower = max(0.0, CHAT_AUTOSAVE_MIN_SCORE - CHAT_AUTOSAVE_REVIEW_BAND)
        upper = min(1.0, CHAT_AUTOSAVE_MIN_SCORE + CHAT_AUTOSAVE_REVIEW_BAND)
        if CHAT_AUTOSAVE_LLM_ENABLED and lower <= usefulness_score <= upper:
            llm_score, llm_reason = await _llm_judge_chat_usefulness(payload.content, metadata)
            if llm_score >= 0:
                # Blend rule score with LLM score, prefer LLM slightly.
                usefulness_score = round(usefulness_score * 0.4 + llm_score * 0.6, 3)
                reasons.append(f"llm_review:{llm_reason}")
            else:
                reasons.append(f"llm_review_skipped:{llm_reason}")

        if usefulness_score < CHAT_AUTOSAVE_MIN_SCORE:
            log_classification(
                {
                    "action": "remember_skip",
                    "reason": "low_usefulness",
                    "usefulness_score": round(usefulness_score, 3),
                    "threshold": CHAT_AUTOSAVE_MIN_SCORE,
                    "llm_score": llm_score if llm_score >= 0 else None,
                    "llm_reason": llm_reason or None,
                    "content_preview": payload.content[:80],
                    "reasons": reasons,
                }
            )
            return {
                "status": "skipped",
                "reason": "low_usefulness",
                "usefulness_score": round(usefulness_score, 3),
                "threshold": CHAT_AUTOSAVE_MIN_SCORE,
                "llm_score": llm_score if llm_score >= 0 else None,
                "reasons": reasons,
            }
        metadata["usefulness_score"] = round(usefulness_score, 3)
        metadata["usefulness_reasons"] = reasons
        if llm_score >= 0:
            metadata["usefulness_llm_score"] = round(llm_score, 3)
            metadata["usefulness_llm_reason"] = llm_reason

    if payload.layer_hint == "auto":
        layer, classification_method = await classify_for_remember_with_learning(payload.content, metadata)
    else:
        layer = payload.layer_hint
        classification_method = "hint:manual"

    try:
        if layer == "L1":
            memory_id = await save_semantic_memory(payload.content, metadata)
        elif layer == "L2":
            memory_id = await save_episodic_memory(payload.content, metadata)
        elif layer == "L3":
            memory_id = await asyncio.to_thread(save_relational_memory, payload.content, metadata)
        elif layer == "L5":
            goal_id = str((metadata or {}).get("goal_id", "")).strip()
            if not goal_id:
                if _is_chat_source(metadata.get("source", "")):
                    # Chat autosave has no explicit goal_id most of the time.
                    # Fallback to semantic memory instead of failing the request.
                    layer = "L1"
                    classification_method = f"{classification_method}:l5_to_l1_no_goal"
                    memory_id = await save_semantic_memory(payload.content, metadata)
                else:
                    raise HTTPException(status_code=400, detail="L5 remember requires metadata.goal_id")
            else:
                goals = load_goals()
                goal = next((g for g in goals if str(g.get("id")) == goal_id), None)
                if goal is None:
                    raise HTTPException(status_code=404, detail=f"goal not found: {goal_id}")
                goal["notes"] = payload.content
                save_goals(goals)
                memory_id = await upsert_goal_vector(goal)
        else:
            raise HTTPException(status_code=400, detail=f"unsupported layer_hint: {layer}")
    except (EmbedderError, QdrantError, RelationalError, AspirationError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    log_classification(
        {
            "action": "remember",
            "classification_method": classification_method,
            "layer": layer,
            "content_preview": payload.content[:80],
        }
    )

    return {
        "id": memory_id,
        "layer": layer,
        "status": "saved",
        "classification_method": classification_method,
    }


@app.post("/recall")
async def recall(payload: RecallRequest) -> dict:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query cannot be empty")

    if payload.layers == "auto":
        routed_layers, classification_method = await classify_for_recall_with_learning(query)
    elif isinstance(payload.layers, list):
        routed_layers = normalize_layers(payload.layers)
        classification_method = "manual:list"
    else:
        routed_layers = normalize_layers(payload.layers)
        classification_method = "manual:single"

    if not routed_layers:
        raise HTTPException(status_code=400, detail="no supported layers in request")

    results: list[dict[str, Any]] = []
    filters = _business_filter_hint(query, payload.filters or {})
    started = time.perf_counter()
    expanded_layers: list[str] | None = None

    try:
        results = await _recall_by_layers(query, payload.top_k, routed_layers, filters)
        # P3: when auto-routing returns empty, broaden the scope once.
        if payload.layers == "auto" and not results:
            candidate = _expanded_layers_for_empty_result(routed_layers)
            if candidate != routed_layers:
                expanded_layers = candidate
                results = await _recall_by_layers(query, payload.top_k, expanded_layers, filters)
    except (EmbedderError, QdrantError, RelationalError, AspirationError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    # Merge and keep top_k globally by score.
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    results = results[: payload.top_k]

    response: dict[str, Any] = {
        "query": payload.query,
        "routed_layers": routed_layers,
        "classification_method": classification_method,
        "results": results,
        "total_search_time_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    if expanded_layers:
        response["auto_expanded"] = True
        response["expanded_from_layers"] = routed_layers
        response["expanded_to_layers"] = expanded_layers
    if payload.include_profile:
        response["profile"] = _load_profile()

    log_classification(
        {
            "action": "recall",
            "classification_method": classification_method,
            "routed_layers": routed_layers,
            "query": payload.query,
            "auto_expanded": bool(expanded_layers),
            "expanded_to_layers": expanded_layers,
        }
    )
    return response


@app.post("/ingest/worklog")
async def ingest_worklog(payload: WorklogIngestRequest) -> dict:
    text = payload.report_text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="report_text cannot be empty")
    return await _ingest_worklog_by_shared_rules(
        report_text=text,
        report_date=payload.report_date.strip(),
        source=payload.source.strip() or "import_manual_worklog",
        max_items=payload.max_items,
        dry_run=payload.dry_run,
    )


@app.post("/synthesize/resume")
async def synthesize_resume(payload: SynthesizeResumeRequest) -> dict:
    top_k = max(3, min(int(payload.top_k), 20))
    filters = _business_filter_hint("resume synthesis", payload.filters or {})
    target_role = payload.target_role.strip()
    style = payload.style.strip() or "professional"

    try:
        l1_task = asyncio.create_task(
            recall_semantic_memory_with_filters(
                "我的核心能力 代表成果 方法论 经验",
                top_k=top_k,
                tags=filters.get("l1_tags") or filters.get("tags"),
                sources=filters.get("l1_sources") or filters.get("sources"),
            )
        )
        l2_task = asyncio.create_task(
            recall_episodic_memory(
                "我的关键项目经历 工作进展 可量化成果",
                top_k=top_k,
                date_from=filters.get("date_from"),
                date_to=filters.get("date_to"),
                project=filters.get("project"),
                tags=filters.get("l2_tags") or filters.get("tags"),
                sources=filters.get("l2_sources") or filters.get("sources"),
            )
        )
        l3_task = asyncio.create_task(
            asyncio.to_thread(
                recall_relational_memory,
                "协作 跨团队 沟通 影响力",
                top_k,
                filters.get("l3_sources") or filters.get("sources"),
            )
        )
        l5_task = asyncio.create_task(recall_aspiration_memory("目标 成长 学习计划", top_k))
        l1_results, l2_results, l3_results, l5_results = await asyncio.gather(
            l1_task, l2_task, l3_task, l5_task
        )
    except (EmbedderError, QdrantError, RelationalError, AspirationError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    merged = list(l2_results) + list(l1_results) + list(l3_results) + list(l5_results)
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    evidence = _build_evidence(merged, max_items=12)
    source_counts = {
        "L1": len(l1_results),
        "L2": len(l2_results),
        "L3": len(l3_results),
        "L5": len(l5_results),
    }

    missing_info: list[str] = []
    if not l2_results:
        missing_info.append("缺少可量化项目经历（L2）")
    if not l1_results:
        missing_info.append("缺少能力与方法论沉淀（L1）")
    if not l3_results:
        missing_info.append("缺少协作关系与影响力证据（L3）")
    if not l5_results:
        missing_info.append("缺少目标与成长方向信息（L5）")

    profile = _load_profile() if payload.include_profile else {}
    profile_identity = profile.get("identity", {}) if isinstance(profile, dict) else {}
    profile_career = profile.get("career", {}) if isinstance(profile, dict) else {}

    prompt = "\n".join(
        [
            "你是职业简历写作助手。请基于给定证据生成一份中文简历草稿。",
            f"风格：{style}",
            f"目标岗位：{target_role or '未指定（按现有经历提炼）'}",
            "",
            "要求：",
            "- 使用简洁、专业、可投递语气；",
            "- 优先写有证据的成果，不要编造；",
            "- 尽量包含可量化表达（如数量、范围、效率变化）；",
            "- 输出结构：个人定位 / 核心能力 / 代表项目 / 协作与影响力 / 学习与成长。",
            "",
            f"身份信息：name={profile_identity.get('name', '')}, role={profile_career.get('current_role', '')}",
            "",
            "L2 事件证据：",
            *(_compact_results(l2_results, limit=8) or ["- 暂无"]),
            "",
            "L1 能力证据：",
            *(_compact_results(l1_results, limit=8) or ["- 暂无"]),
            "",
            "L3 协作证据：",
            *(_compact_results(l3_results, limit=6) or ["- 暂无"]),
            "",
            "L5 目标证据：",
            *(_compact_results(l5_results, limit=5) or ["- 暂无"]),
        ]
    )

    try:
        resume_text = await generate_reflection_text(prompt)
    except ReflectiveError:
        resume_text = "\n".join(
            [
                "个人定位：基于当前记忆，已形成 AI 产品与数据平台方向的复合能力。",
                "核心能力：需求抽象、数据结构化、跨团队协作推进、接口与平台化思维。",
                "代表项目：S9 相关平台与学情数据处理、AI API 平台化建设等。",
                "协作与影响力：持续参与跨团队沟通与评审，推动流程与规范沉淀。",
                "学习与成长：围绕 RAG、记忆系统、产品化落地持续迭代。",
            ]
        )

    highlights: list[str] = []
    for ev in evidence[:5]:
        snip = str(ev.get("snippet", "")).strip()
        if snip:
            highlights.append(snip[:80])

    confidence = _resume_confidence(source_counts, evidence)
    return {
        "type": "resume",
        "target_role": target_role,
        "result": resume_text,
        "highlights": highlights,
        "evidence": evidence,
        "confidence": confidence,
        "missing_info": missing_info,
        "source_counts": source_counts,
    }


@app.post("/synthesize/work_status")
async def synthesize_work_status(payload: SynthesizeWorkStatusRequest) -> dict:
    top_k = max(3, min(int(payload.top_k), 20))
    p_start, p_end = _resolve_reflection_period(payload.period_start, payload.period_end)
    filters = _business_filter_hint("work status synthesis", payload.filters or {})
    project = payload.project.strip()
    style = payload.style.strip() or "concise"
    if project and not filters.get("project"):
        filters["project"] = project

    try:
        l2_task = asyncio.create_task(
            recall_episodic_memory(
                "最近工作进展 阶段成果 阻塞 风险 下一步",
                top_k=top_k,
                date_from=p_start,
                date_to=p_end,
                project=filters.get("project"),
                event_types=filters.get("event_types"),
                tags=filters.get("l2_tags") or filters.get("tags"),
                sources=filters.get("l2_sources") or filters.get("sources"),
            )
        )
        l1_task = asyncio.create_task(
            recall_semantic_memory_with_filters(
                "方法论 经验 教训 复盘",
                top_k=max(4, top_k // 2),
                tags=filters.get("l1_tags") or filters.get("tags"),
                sources=filters.get("l1_sources") or filters.get("sources"),
            )
        )
        l3_task = asyncio.create_task(
            asyncio.to_thread(
                recall_relational_memory,
                "协作 沟通 依赖 风险",
                max(4, top_k // 2),
                filters.get("l3_sources") or filters.get("sources"),
            )
        )
        l5_task = asyncio.create_task(recall_aspiration_memory("目标 计划 里程碑", max(4, top_k // 2)))
        l2_results, l1_results, l3_results, l5_results = await asyncio.gather(
            l2_task, l1_task, l3_task, l5_task
        )
    except (EmbedderError, QdrantError, RelationalError, AspirationError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    merged = list(l2_results) + list(l1_results) + list(l3_results) + list(l5_results)
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    evidence = _build_evidence(merged, max_items=12)
    source_counts = {
        "L1": len(l1_results),
        "L2": len(l2_results),
        "L3": len(l3_results),
        "L5": len(l5_results),
    }

    missing_info: list[str] = []
    if not l2_results:
        missing_info.append("缺少该时间段内的工作事件证据（L2）")
    if project and not any(str(x.get("project", "")).strip() for x in evidence):
        missing_info.append("未检索到明确项目标识，建议补充 metadata.project")

    profile = _load_profile() if payload.include_profile else {}
    profile_career = profile.get("career", {}) if isinstance(profile, dict) else {}
    prompt = "\n".join(
        [
            "你是工作状态总结助手。请基于证据生成中文周报式状态摘要。",
            f"时间范围：{p_start} 到 {p_end}",
            f"项目：{project or str(filters.get('project', '') or '未指定')}",
            f"风格：{style}",
            "",
            "要求：",
            "- 不要编造；只使用证据中可支持的信息；",
            "- 输出结构固定为：1) 当前进展 2) 风险与阻塞 3) 下一步行动（3条）;",
            "- 如果信息不足，明确指出不确定项。",
            "",
            f"职业信息（可选）：{profile_career.get('current_role', '')}",
            "",
            "L2 事件证据：",
            *(_compact_results(l2_results, limit=10) or ["- 暂无"]),
            "",
            "L1 经验证据：",
            *(_compact_results(l1_results, limit=6) or ["- 暂无"]),
            "",
            "L3 协作证据：",
            *(_compact_results(l3_results, limit=6) or ["- 暂无"]),
            "",
            "L5 目标证据：",
            *(_compact_results(l5_results, limit=5) or ["- 暂无"]),
        ]
    )

    try:
        status_text = await generate_reflection_text(prompt)
    except ReflectiveError:
        status_text = "\n".join(
            [
                "1) 当前进展：已基于本地记忆汇总近期工作事件，但部分信息仍需补充更细粒度项目标识。",
                "2) 风险与阻塞：跨团队依赖与数据口径差异可能影响推进速度。",
                "3) 下一步行动：",
                "- 明确本周优先级最高的 1-2 个里程碑；",
                "- 补全关键事件的 project/date/source 元数据；",
                "- 对阻塞事项设定负责人与截止时间并跟踪。",
            ]
        )

    highlights: list[str] = []
    for ev in evidence[:5]:
        snip = str(ev.get("snippet", "")).strip()
        if snip:
            highlights.append(snip[:80])

    confidence = _resume_confidence(source_counts, evidence)
    return {
        "type": "work_status",
        "project": project,
        "period": {"start": p_start, "end": p_end},
        "result": status_text,
        "highlights": highlights,
        "evidence": evidence,
        "confidence": confidence,
        "missing_info": missing_info,
        "source_counts": source_counts,
    }


@app.post("/synthesize/learning_plan")
async def synthesize_learning_plan(payload: SynthesizeLearningPlanRequest) -> dict:
    top_k = max(3, min(int(payload.top_k), 20))
    horizon = max(30, min(int(payload.horizon_days), 180))
    style = payload.style.strip() or "actionable"
    user_goal = payload.goal.strip()
    filters = _business_filter_hint("learning plan synthesis", payload.filters or {})
    profile = _load_profile() if payload.include_profile else {}

    try:
        l1_task = asyncio.create_task(
            recall_semantic_memory_with_filters(
                "我当前技能 能力短板 方法论 学习经验",
                top_k=top_k,
                tags=filters.get("l1_tags") or filters.get("tags"),
                sources=filters.get("l1_sources") or filters.get("sources"),
            )
        )
        l2_task = asyncio.create_task(
            recall_episodic_memory(
                "最近学习与项目进展 问题卡点 复盘",
                top_k=top_k,
                date_from=filters.get("date_from"),
                date_to=filters.get("date_to"),
                project=filters.get("project"),
                tags=filters.get("l2_tags") or filters.get("tags"),
                sources=filters.get("l2_sources") or filters.get("sources"),
            )
        )
        l3_task = asyncio.create_task(
            asyncio.to_thread(
                recall_relational_memory,
                "协作 反馈 指导 评审",
                max(4, top_k // 2),
                filters.get("l3_sources") or filters.get("sources"),
            )
        )
        l5_task = asyncio.create_task(recall_aspiration_memory("目标 计划 里程碑 学习", top_k))
        l1_results, l2_results, l3_results, l5_results = await asyncio.gather(
            l1_task, l2_task, l3_task, l5_task
        )
    except (EmbedderError, QdrantError, RelationalError, AspirationError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    merged = list(l5_results) + list(l2_results) + list(l1_results) + list(l3_results)
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    evidence = _build_evidence(merged, max_items=12)
    source_counts = {
        "L1": len(l1_results),
        "L2": len(l2_results),
        "L3": len(l3_results),
        "L5": len(l5_results),
    }

    missing_info: list[str] = []
    if not l5_results:
        missing_info.append("缺少明确学习目标（L5）")
    if not l1_results:
        missing_info.append("缺少技能现状证据（L1）")
    if not l2_results:
        missing_info.append("缺少近期实践/卡点证据（L2）")

    identity = profile.get("identity", {}) if isinstance(profile, dict) else {}
    career = profile.get("career", {}) if isinstance(profile, dict) else {}
    prompt = "\n".join(
        [
            "你是学习路径规划助手。请基于证据生成中文学习计划。",
            f"目标周期：{horizon}天",
            f"用户目标：{user_goal or '未指定（根据现有目标推断）'}",
            f"风格：{style}",
            "",
            "要求：",
            "- 输出必须包含 30/60/90 天三个阶段；",
            "- 每阶段给出 2-3 个可执行行动；",
            "- 每个行动给出预期产出和验收方式；",
            "- 明确优先级（high/medium/low）；",
            "- 不要编造经验，优先引用现有证据。",
            "",
            f"身份信息：name={identity.get('name','')}, role={career.get('current_role','')}",
            "",
            "L5 目标证据：",
            *(_compact_results(l5_results, limit=8) or ["- 暂无"]),
            "",
            "L1 技能证据：",
            *(_compact_results(l1_results, limit=8) or ["- 暂无"]),
            "",
            "L2 实践证据：",
            *(_compact_results(l2_results, limit=8) or ["- 暂无"]),
            "",
            "L3 协作证据：",
            *(_compact_results(l3_results, limit=6) or ["- 暂无"]),
        ]
    )

    try:
        plan_text = await generate_reflection_text(prompt)
    except ReflectiveError:
        plan_text = "\n".join(
            [
                "30天：聚焦关键基础能力，完成1个可验证的项目闭环。",
                "60天：将能力用于真实业务场景，形成可复用方法和文档。",
                "90天：沉淀标准化方案并进行复盘，输出可展示成果。",
            ]
        )

    highlights: list[str] = []
    for ev in evidence[:5]:
        snip = str(ev.get("snippet", "")).strip()
        if snip:
            highlights.append(snip[:80])

    confidence = _resume_confidence(source_counts, evidence)
    priority = _learning_priority(evidence, len(l5_results))
    return {
        "type": "learning_plan",
        "goal": user_goal,
        "horizon_days": horizon,
        "result": plan_text,
        "priority": priority,
        "highlights": highlights,
        "evidence": evidence,
        "confidence": confidence,
        "missing_info": missing_info,
        "source_counts": source_counts,
    }


@app.post("/synthesize/learning_engine")
async def synthesize_learning_engine(payload: SynthesizeLearningEngineRequest) -> dict:
    top_k = max(3, min(int(payload.top_k), 20))
    period_days = max(3, min(int(payload.period_days), 30))
    style = payload.style.strip() or "actionable"
    goal = payload.goal.strip()
    review_input = payload.review_input.strip()
    filters = _business_filter_hint("learning engine synthesis", payload.filters or {})
    profile = _load_profile() if payload.include_profile else {}

    try:
        l1_task = asyncio.create_task(
            recall_semantic_memory_with_filters(
                "技能差距 能力短板 学习方法",
                top_k=top_k,
                tags=filters.get("l1_tags") or filters.get("tags"),
                sources=filters.get("l1_sources") or filters.get("sources"),
            )
        )
        l2_task = asyncio.create_task(
            recall_episodic_memory(
                "最近学习实践 项目卡点 复盘记录",
                top_k=top_k,
                date_from=filters.get("date_from"),
                date_to=filters.get("date_to"),
                project=filters.get("project"),
                tags=filters.get("l2_tags") or filters.get("tags"),
                sources=filters.get("l2_sources") or filters.get("sources"),
            )
        )
        l5_task = asyncio.create_task(recall_aspiration_memory("学习目标 优先级 里程碑", top_k))
        l1_results, l2_results, l5_results = await asyncio.gather(l1_task, l2_task, l5_task)
    except (EmbedderError, QdrantError, AspirationError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    merged = list(l2_results) + list(l1_results) + list(l5_results)
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    evidence = _build_evidence(merged, max_items=12)
    source_counts = {"L1": len(l1_results), "L2": len(l2_results), "L5": len(l5_results)}

    missing_info: list[str] = []
    if not goal and not l5_results:
        missing_info.append("缺少明确学习目标（goal 或 L5）")
    if not l2_results:
        missing_info.append("缺少近期实战证据（L2）")

    weekly_tasks = _build_weekly_tasks(evidence, max_items=4)
    checkpoints = [
        "每个任务都有可见产出（文档、代码、记录或演示）。",
        "每个任务在48小时内可完成最小验证版本。",
        "周末完成一次复盘并更新下一周计划。",
    ]
    expected_outcomes = [
        "形成至少1个可复用的方法模板。",
        "在当前项目中验证至少1项新能力。",
        "对学习重点和薄弱点有可追踪结论。",
    ]
    next_review_date = (datetime.now(timezone.utc).date() + timedelta(days=period_days)).isoformat()

    identity = profile.get("identity", {}) if isinstance(profile, dict) else {}
    career = profile.get("career", {}) if isinstance(profile, dict) else {}
    prompt = "\n".join(
        [
            "你是学习引擎助手。请基于证据给出短周期学习执行方案。",
            f"周期：{period_days}天",
            f"目标：{goal or '未指定（根据现有证据推断）'}",
            f"风格：{style}",
            "",
            "输出结构固定：",
            "1) 本周学习任务（3-5条）",
            "2) 每条任务验收标准",
            "3) 预期成果",
            "4) 下次复盘问题（3条）",
            "",
            f"身份信息：name={identity.get('name','')}, role={career.get('current_role','')}",
            f"用户复盘输入：{review_input or '无'}",
            "",
            "L2 证据：",
            *(_compact_results(l2_results, limit=8) or ["- 暂无"]),
            "",
            "L1 证据：",
            *(_compact_results(l1_results, limit=8) or ["- 暂无"]),
            "",
            "L5 证据：",
            *(_compact_results(l5_results, limit=6) or ["- 暂无"]),
        ]
    )
    try:
        result_text = await generate_reflection_text(prompt)
    except ReflectiveError:
        result_text = "\n".join(
            [
                "1) 本周学习任务：围绕当前项目补齐1个关键能力短板并产出结果。",
                "2) 验收标准：每个任务都有可追踪产出与复盘记录。",
                "3) 预期成果：形成可复用模板并提升项目推进效率。",
                "4) 下次复盘问题：本周最有效学习动作是什么？最大阻塞是什么？下周如何调整？",
            ]
        )

    confidence = _resume_confidence(source_counts, evidence)
    priority = _learning_priority(evidence, len(l5_results))
    return {
        "type": "learning_engine",
        "goal": goal,
        "period_days": period_days,
        "priority": priority,
        "result": result_text,
        "weekly_tasks": weekly_tasks,
        "checkpoints": checkpoints,
        "expected_outcomes": expected_outcomes,
        "next_review_date": next_review_date,
        "evidence": evidence,
        "confidence": confidence,
        "missing_info": missing_info,
        "source_counts": source_counts,
    }


@app.post("/synthesize/copilot")
async def synthesize_copilot(payload: SynthesizeCopilotRequest) -> dict:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query cannot be empty")

    task = payload.task.strip().lower()
    if task == "auto":
        task = _detect_copilot_task(query)

    if task == "resume":
        result = await synthesize_resume(
            SynthesizeResumeRequest(
                target_role=payload.target_role,
                style=payload.style or "professional",
                top_k=payload.top_k,
                filters=payload.filters,
                include_profile=payload.include_profile,
            )
        )
    elif task == "work_status":
        result = await synthesize_work_status(
            SynthesizeWorkStatusRequest(
                project=payload.project,
                period_start=payload.period_start,
                period_end=payload.period_end,
                style=payload.style or "concise",
                top_k=payload.top_k,
                filters=payload.filters,
                include_profile=False,
            )
        )
    elif task == "learning_plan":
        result = await synthesize_learning_plan(
            SynthesizeLearningPlanRequest(
                goal=payload.goal or query,
                horizon_days=payload.horizon_days,
                style=payload.style or "actionable",
                top_k=payload.top_k,
                filters=payload.filters,
                include_profile=payload.include_profile,
            )
        )
    elif task == "learning_engine":
        result = await synthesize_learning_engine(
            SynthesizeLearningEngineRequest(
                goal=payload.goal or query,
                period_days=payload.period_days,
                style=payload.style or "actionable",
                top_k=payload.top_k,
                review_input="",
                filters=payload.filters,
                include_profile=payload.include_profile,
            )
        )
    elif task == "self_profile":
        result = await synthesize_self_profile(
            SynthesizeSelfProfileRequest(
                style=payload.style or "insightful",
                top_k=payload.top_k,
                filters=payload.filters,
                include_profile=payload.include_profile,
            )
        )
    elif task == "project_radar":
        result = await synthesize_project_radar(
            SynthesizeProjectRadarRequest(
                project=payload.project,
                period_start=payload.period_start,
                period_end=payload.period_end,
                style=payload.style or "operational",
                top_k=payload.top_k,
                filters=payload.filters,
                include_profile=False,
            )
        )
    elif task == "decision_copilot":
        result = await synthesize_decision_copilot(
            SynthesizeDecisionCopilotRequest(
                decision_topic=query,
                options=payload.options or [],
                horizon_days=max(7, payload.horizon_days if payload.horizon_days else 30),
                style=payload.style or "pragmatic",
                top_k=payload.top_k,
                filters=payload.filters,
                include_profile=False,
            )
        )
    elif task == "career_assets":
        result = await synthesize_career_assets(
            SynthesizeCareerAssetsRequest(
                target_role=payload.target_role,
                asset_types=["resume_bullets", "review_outline", "project_case_cards"],
                style=payload.style or "professional",
                top_k=payload.top_k,
                filters=payload.filters,
                include_profile=payload.include_profile,
            )
        )
    else:
        raise HTTPException(status_code=400, detail=f"unsupported copilot task: {task}")

    return {
        "copilot_task": task,
        "query": query,
        "result": result,
    }


@app.post("/synthesize/self_profile")
async def synthesize_self_profile(payload: SynthesizeSelfProfileRequest) -> dict:
    top_k = max(3, min(int(payload.top_k), 20))
    style = payload.style.strip() or "insightful"
    filters = _business_filter_hint("self profile synthesis", payload.filters or {})
    profile = _load_profile() if payload.include_profile else {}

    try:
        l1_task = asyncio.create_task(
            recall_semantic_memory_with_filters(
                "我的角色 能力 优势 工作风格",
                top_k=top_k,
                categories=["self_profile", "knowledge", "skill"],
                tags=filters.get("l1_tags") or filters.get("tags"),
                sources=filters.get("l1_sources") or filters.get("sources"),
            )
        )
        l2_task = asyncio.create_task(
            recall_episodic_memory(
                "我最近在做什么 进展 复盘",
                top_k=top_k,
                date_from=filters.get("date_from"),
                date_to=filters.get("date_to"),
                project=filters.get("project"),
                tags=filters.get("l2_tags") or filters.get("tags"),
                sources=filters.get("l2_sources") or filters.get("sources"),
            )
        )
        l3_task = asyncio.create_task(
            asyncio.to_thread(
                recall_relational_memory,
                "协作 关系 伙伴 团队",
                max(4, top_k // 2),
                filters.get("l3_sources") or filters.get("sources"),
            )
        )
        l5_task = asyncio.create_task(recall_aspiration_memory("目标 计划 学习方向", max(4, top_k // 2)))
        l1_results, l2_results, l3_results, l5_results = await asyncio.gather(
            l1_task, l2_task, l3_task, l5_task
        )
    except (EmbedderError, QdrantError, RelationalError, AspirationError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    merged = list(l1_results) + list(l2_results) + list(l3_results) + list(l5_results)
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    evidence = _build_evidence(merged, max_items=12)
    source_counts = {
        "L1": len(l1_results),
        "L2": len(l2_results),
        "L3": len(l3_results),
        "L5": len(l5_results),
    }

    missing_info: list[str] = []
    if not l1_results:
        missing_info.append("缺少能力/画像证据（L1）")
    if not l2_results:
        missing_info.append("缺少近期行为证据（L2）")
    if not l3_results:
        missing_info.append("缺少协作关系证据（L3）")

    identity = profile.get("identity", {}) if isinstance(profile, dict) else {}
    career = profile.get("career", {}) if isinstance(profile, dict) else {}
    focus_topics = _extract_focus_topics(evidence, max_items=4)

    prompt = "\n".join(
        [
            "你是个人画像分析助手。请基于证据生成一份中文“自我画像”总结。",
            f"风格：{style}",
            "",
            "要求：",
            "- 输出结构：1) 我是谁 2) 我擅长什么 3) 我最近在推进什么 4) 我的工作风格 5) 盲区与下一步；",
            "- 不要编造事实；证据不足时说明不确定项；",
            "- 语气务实、可执行。",
            "",
            f"身份信息：name={identity.get('name','')}, role={career.get('current_role','')}",
            "",
            "L1 画像证据：",
            *(_compact_results(l1_results, limit=8) or ["- 暂无"]),
            "",
            "L2 行为证据：",
            *(_compact_results(l2_results, limit=8) or ["- 暂无"]),
            "",
            "L3 协作证据：",
            *(_compact_results(l3_results, limit=6) or ["- 暂无"]),
            "",
            "L5 目标证据：",
            *(_compact_results(l5_results, limit=5) or ["- 暂无"]),
        ]
    )

    try:
        profile_text = await generate_reflection_text(prompt)
    except ReflectiveError:
        profile_text = "\n".join(
            [
                "1) 我是谁：当前处于 AI 产品与数据平台融合实践的角色。",
                "2) 我擅长什么：需求抽象、跨团队协同、数据与接口体系化推进。",
                "3) 最近推进：围绕学情数据、记忆系统、平台化能力持续迭代。",
                "4) 工作风格：偏结构化与证据驱动，注重可落地与复盘。",
                "5) 下一步：补齐薄弱能力闭环并沉淀可复用方法。",
            ]
        )

    strengths: list[str] = []
    for item in (l1_results + l2_results)[:5]:
        txt = str(item.get("content", "")).strip()
        if txt:
            strengths.append(txt[:60])

    collaboration: list[str] = []
    for item in l3_results[:4]:
        txt = str(item.get("content", "")).strip()
        if txt:
            collaboration.append(txt[:60])

    growth_goals: list[str] = []
    for item in l5_results[:4]:
        txt = str(item.get("content", "")).strip()
        if txt:
            growth_goals.append(txt[:60])

    structured_profile = {
        "name": identity.get("name", ""),
        "current_role": career.get("current_role", ""),
        "focus_topics": focus_topics,
        "strength_signals": strengths,
        "collaboration_signals": collaboration,
        "growth_goals": growth_goals,
    }

    previous_snapshot = (_load_profile_snapshots(recent=1) or [None])[-1]
    previous_profile = (previous_snapshot or {}).get("profile", {}) if isinstance(previous_snapshot, dict) else {}
    conflict = _detect_profile_conflicts(previous_profile, structured_profile)
    conflict_strategy = _profile_conflict_strategy(conflict)

    confidence = _resume_confidence(source_counts, evidence)
    snapshot_id = f"ps_{uuid4().hex[:12]}"
    snapshot_payload = {
        "snapshot_id": snapshot_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "type": "self_profile_snapshot",
        "style": style,
        "confidence": confidence,
        "source_counts": source_counts,
        "profile": structured_profile,
        "missing_info": missing_info,
        "result_preview": profile_text[:200],
        "conflict": conflict,
        "conflict_strategy": conflict_strategy,
    }
    _append_profile_snapshot(snapshot_payload)

    return {
        "type": "self_profile",
        "result": profile_text,
        "profile": structured_profile,
        "evidence": evidence,
        "confidence": confidence,
        "missing_info": missing_info,
        "source_counts": source_counts,
        "snapshot_created": True,
        "snapshot_id": snapshot_id,
        "snapshot_log": str(PROFILE_SNAPSHOT_LOG_PATH),
        "conflict": conflict,
        "conflict_strategy": conflict_strategy,
    }


@app.post("/synthesize/project_radar")
async def synthesize_project_radar(payload: SynthesizeProjectRadarRequest) -> dict:
    top_k = max(3, min(int(payload.top_k), 20))
    p_start, p_end = _resolve_reflection_period(payload.period_start, payload.period_end)
    project = payload.project.strip()
    style = payload.style.strip() or "operational"
    filters = _business_filter_hint("project radar synthesis", payload.filters or {})
    if project and not filters.get("project"):
        filters["project"] = project

    try:
        l2_task = asyncio.create_task(
            recall_episodic_memory(
                "项目进展 风险 阻塞 依赖 延期",
                top_k=top_k,
                date_from=p_start,
                date_to=p_end,
                project=filters.get("project"),
                event_types=filters.get("event_types"),
                tags=filters.get("l2_tags") or filters.get("tags"),
                sources=filters.get("l2_sources") or filters.get("sources"),
            )
        )
        l3_task = asyncio.create_task(
            asyncio.to_thread(
                recall_relational_memory,
                "协作 依赖 风险 负责人",
                max(4, top_k // 2),
                filters.get("l3_sources") or filters.get("sources"),
            )
        )
        l5_task = asyncio.create_task(recall_aspiration_memory("里程碑 目标 进度", max(4, top_k // 2)))
        l1_task = asyncio.create_task(
            recall_semantic_memory_with_filters(
                "复盘 方法 经验 教训",
                top_k=max(4, top_k // 2),
                tags=filters.get("l1_tags") or filters.get("tags"),
                sources=filters.get("l1_sources") or filters.get("sources"),
            )
        )
        l2_results, l3_results, l5_results, l1_results = await asyncio.gather(
            l2_task, l3_task, l5_task, l1_task
        )
    except (EmbedderError, QdrantError, RelationalError, AspirationError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    merged = list(l2_results) + list(l3_results) + list(l5_results) + list(l1_results)
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    evidence = _build_evidence(merged, max_items=12)
    source_counts = {
        "L1": len(l1_results),
        "L2": len(l2_results),
        "L3": len(l3_results),
        "L5": len(l5_results),
    }
    risk = _risk_signal_scan(evidence)

    missing_info: list[str] = []
    if not l2_results:
        missing_info.append("缺少近期项目事件证据（L2）")
    if not l3_results:
        missing_info.append("缺少协作依赖证据（L3）")
    if not l5_results:
        missing_info.append("缺少目标与里程碑证据（L5）")

    prompt = "\n".join(
        [
            "你是项目雷达助手。请基于证据输出项目状态与风险预警。",
            f"项目：{project or str(filters.get('project', '') or '未指定')}",
            f"时间范围：{p_start} 到 {p_end}",
            f"风格：{style}",
            "",
            "输出结构固定：",
            "1) 当前状态（绿/黄/红 + 判断理由）",
            "2) 核心风险（最多3条）",
            "3) 关键阻塞与依赖（负责人/对象可为空）",
            "4) 接下来7天行动建议（3条）",
            "",
            "L2 项目事件：",
            *(_compact_results(l2_results, limit=10) or ["- 暂无"]),
            "",
            "L3 协作依赖：",
            *(_compact_results(l3_results, limit=6) or ["- 暂无"]),
            "",
            "L5 里程碑：",
            *(_compact_results(l5_results, limit=6) or ["- 暂无"]),
            "",
            "L1 经验复盘：",
            *(_compact_results(l1_results, limit=6) or ["- 暂无"]),
        ]
    )

    try:
        radar_text = await generate_reflection_text(prompt)
    except ReflectiveError:
        radar_text = "\n".join(
            [
                "1) 当前状态：黄灯（存在推进与风险并存）。",
                "2) 核心风险：跨团队依赖未完全对齐；关键数据口径可能不一致。",
                "3) 关键阻塞与依赖：部分事项缺少明确负责人和截止时间。",
                "4) 接下来7天行动建议：明确优先级、锁定负责人、建立每日风险同步机制。",
            ]
        )

    confidence = _resume_confidence(source_counts, evidence)
    return {
        "type": "project_radar",
        "project": project,
        "period": {"start": p_start, "end": p_end},
        "result": radar_text,
        "risk": risk,
        "evidence": evidence,
        "confidence": confidence,
        "missing_info": missing_info,
        "source_counts": source_counts,
    }


@app.post("/synthesize/decision_copilot")
async def synthesize_decision_copilot(payload: SynthesizeDecisionCopilotRequest) -> dict:
    topic = payload.decision_topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="decision_topic cannot be empty")
    top_k = max(3, min(int(payload.top_k), 20))
    horizon = max(7, min(int(payload.horizon_days), 180))
    style = payload.style.strip() or "pragmatic"
    filters = _business_filter_hint("decision copilot synthesis", payload.filters or {})
    options = [x.strip() for x in payload.options if str(x).strip()]

    try:
        l1_task = asyncio.create_task(
            recall_semantic_memory_with_filters(
                f"{topic} 方法 经验 方案 取舍",
                top_k=top_k,
                tags=filters.get("l1_tags") or filters.get("tags"),
                sources=filters.get("l1_sources") or filters.get("sources"),
            )
        )
        l2_task = asyncio.create_task(
            recall_episodic_memory(
                f"{topic} 实践 进展 问题 结果",
                top_k=top_k,
                date_from=filters.get("date_from"),
                date_to=filters.get("date_to"),
                project=filters.get("project"),
                tags=filters.get("l2_tags") or filters.get("tags"),
                sources=filters.get("l2_sources") or filters.get("sources"),
            )
        )
        l5_task = asyncio.create_task(recall_aspiration_memory(f"{topic} 目标 优先级", max(4, top_k // 2)))
        l1_results, l2_results, l5_results = await asyncio.gather(l1_task, l2_task, l5_task)
    except (EmbedderError, QdrantError, AspirationError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    merged = list(l2_results) + list(l1_results) + list(l5_results)
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    evidence = _build_evidence(merged, max_items=12)
    source_counts = {"L1": len(l1_results), "L2": len(l2_results), "L5": len(l5_results)}

    missing_info: list[str] = []
    if not options:
        missing_info.append("未提供明确候选方案（options），系统将基于证据给默认建议")
    if not l2_results:
        missing_info.append("缺少实践结果证据（L2）")

    option_text = "；".join(options) if options else "未提供候选方案，请基于证据给出2-3个可选方案。"
    prompt = "\n".join(
        [
            "你是决策副驾驶助手。请基于证据给出决策建议。",
            f"决策主题：{topic}",
            f"候选方案：{option_text}",
            f"决策周期：{horizon}天",
            f"风格：{style}",
            "",
            "输出结构固定：",
            "1) 方案对比（收益/风险/成本）",
            "2) 推荐方案（只选1个）",
            "3) 备选方案与触发条件",
            "4) 本周执行清单（3条）",
            "",
            "L2 证据：",
            *(_compact_results(l2_results, limit=8) or ["- 暂无"]),
            "",
            "L1 证据：",
            *(_compact_results(l1_results, limit=8) or ["- 暂无"]),
            "",
            "L5 证据：",
            *(_compact_results(l5_results, limit=6) or ["- 暂无"]),
        ]
    )

    try:
        decision_text = await generate_reflection_text(prompt)
    except ReflectiveError:
        decision_text = "\n".join(
            [
                "1) 方案对比：建议在低风险可回滚前提下优先试点推进。",
                "2) 推荐方案：先做可验证最小闭环，再逐步扩展。",
                "3) 备选方案：若资源不足，采用分阶段交付。",
                "4) 本周执行清单：明确目标、拆解任务、设定验收标准。",
            ]
        )

    confidence = _resume_confidence(source_counts, evidence)
    recommendation = "evidence_based_incremental"
    return {
        "type": "decision_copilot",
        "decision_topic": topic,
        "options": options,
        "horizon_days": horizon,
        "result": decision_text,
        "recommendation": recommendation,
        "evidence": evidence,
        "confidence": confidence,
        "missing_info": missing_info,
        "source_counts": source_counts,
    }


@app.post("/synthesize/career_assets")
async def synthesize_career_assets(payload: SynthesizeCareerAssetsRequest) -> dict:
    top_k = max(3, min(int(payload.top_k), 20))
    style = payload.style.strip() or "professional"
    target_role = payload.target_role.strip()
    filters = _business_filter_hint("career assets synthesis", payload.filters or {})
    profile = _load_profile() if payload.include_profile else {}
    asset_types = [str(x).strip() for x in payload.asset_types if str(x).strip()]
    if not asset_types:
        asset_types = ["resume_bullets", "review_outline", "project_case_cards"]

    try:
        l1_task = asyncio.create_task(
            recall_semantic_memory_with_filters(
                "能力 方法论 经验 成果",
                top_k=top_k,
                tags=filters.get("l1_tags") or filters.get("tags"),
                sources=filters.get("l1_sources") or filters.get("sources"),
            )
        )
        l2_task = asyncio.create_task(
            recall_episodic_memory(
                "关键项目经历 量化成果 工作进展",
                top_k=top_k,
                date_from=filters.get("date_from"),
                date_to=filters.get("date_to"),
                project=filters.get("project"),
                tags=filters.get("l2_tags") or filters.get("tags"),
                sources=filters.get("l2_sources") or filters.get("sources"),
            )
        )
        l3_task = asyncio.create_task(
            asyncio.to_thread(
                recall_relational_memory,
                "协作 影响力 跨团队",
                max(4, top_k // 2),
                filters.get("l3_sources") or filters.get("sources"),
            )
        )
        l5_task = asyncio.create_task(recall_aspiration_memory("目标 里程碑 成长", max(4, top_k // 2)))
        l1_results, l2_results, l3_results, l5_results = await asyncio.gather(
            l1_task, l2_task, l3_task, l5_task
        )
    except (EmbedderError, QdrantError, RelationalError, AspirationError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    merged = list(l2_results) + list(l1_results) + list(l3_results) + list(l5_results)
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    evidence = _build_evidence(merged, max_items=14)
    source_counts = {
        "L1": len(l1_results),
        "L2": len(l2_results),
        "L3": len(l3_results),
        "L5": len(l5_results),
    }
    confidence = _resume_confidence(source_counts, evidence)
    missing_info: list[str] = []
    if not l2_results:
        missing_info.append("缺少量化项目经历证据（L2）")
    if not l1_results:
        missing_info.append("缺少能力方法论证据（L1）")

    identity = profile.get("identity", {}) if isinstance(profile, dict) else {}
    career = profile.get("career", {}) if isinstance(profile, dict) else {}
    prompt = "\n".join(
        [
            "你是职业资产工厂助手，请基于证据产出三类内容。",
            f"目标岗位：{target_role or career.get('current_role', '') or '未指定'}",
            f"风格：{style}",
            f"输出资产类型：{', '.join(asset_types)}",
            "",
            "请使用以下结构输出：",
            "## resume_bullets",
            "- 6~10条可直接放简历的要点（尽量量化）",
            "## review_outline",
            "- 述职提纲：背景、目标、关键成果、挑战、反思、下一步",
            "## project_case_cards",
            "- 至少2个项目案例卡（项目名/问题/行动/结果/复用价值）",
            "",
            f"身份信息：name={identity.get('name','')}, role={career.get('current_role','')}",
            "",
            "L2 证据：",
            *(_compact_results(l2_results, limit=10) or ["- 暂无"]),
            "",
            "L1 证据：",
            *(_compact_results(l1_results, limit=8) or ["- 暂无"]),
            "",
            "L3 证据：",
            *(_compact_results(l3_results, limit=6) or ["- 暂无"]),
            "",
            "L5 证据：",
            *(_compact_results(l5_results, limit=6) or ["- 暂无"]),
        ]
    )
    try:
        assets_text = await generate_reflection_text(prompt)
    except ReflectiveError:
        assets_text = "\n".join(
            [
                "## resume_bullets",
                "- 基于本地记忆已形成 AI 产品与数据平台融合能力。",
                "## review_outline",
                "- 背景 -> 目标 -> 成果 -> 挑战 -> 反思 -> 下一步。",
                "## project_case_cards",
                "- 项目A：问题/行动/结果/复用价值（待补充量化细节）。",
            ]
        )

    highlights: list[str] = []
    for ev in evidence[:6]:
        snip = str(ev.get("snippet", "")).strip()
        if snip:
            highlights.append(snip[:80])

    return {
        "type": "career_assets",
        "target_role": target_role,
        "asset_types": asset_types,
        "result": assets_text,
        "highlights": highlights,
        "evidence": evidence,
        "confidence": confidence,
        "missing_info": missing_info,
        "source_counts": source_counts,
    }


@app.post("/reflect")
async def reflect(payload: ReflectRequest) -> dict:
    period_start, period_end = _resolve_reflection_period(payload.period_start, payload.period_end)
    focus_areas = payload.focus_areas or ["核心成果", "成长领域", "盲区发现", "下一步建议"]
    try:
        result = await _run_reflection_job(
            reflect_type=payload.type,
            period_start=period_start,
            period_end=period_end,
            focus_areas=focus_areas,
            trigger="manual",
        )
    except (EmbedderError, QdrantError, RelationalError, ReflectiveError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    log_classification(
        {
            "action": "reflect",
            "reflection_type": payload.type,
            "period_start": period_start,
            "period_end": period_end,
            "focus_areas": focus_areas,
            "status": "saved_to_L4",
            "reflection_id": result.get("id"),
            "quality_score": result.get("quality", {}).get("score"),
        }
    )
    return result


@app.get("/reflect/quality")
def reflect_quality(recent: int = 20) -> dict:
    records = _load_reflection_quality_log(recent=recent)
    if not records:
        return {"recent": recent, "count": 0, "avg_score": 0.0, "records": []}
    avg = round(sum(float((r.get("quality") or {}).get("score", 0)) for r in records) / len(records), 3)
    return {
        "recent": recent,
        "count": len(records),
        "avg_score": avg,
        "records": records,
    }


@app.get("/reflect/schedule")
def reflect_schedule_status() -> dict:
    return {
        "enabled": L4_AUTO_REFLECT_ENABLED,
        "interval_minutes": L4_AUTO_REFLECT_INTERVAL_MINUTES,
        "type": L4_AUTO_REFLECT_TYPE,
        "focus_areas": L4_AUTO_REFLECT_FOCUS_AREAS,
        "task_running": bool(_auto_reflect_task and not _auto_reflect_task.done()),
        "quality_log_exists": REFLECTION_QUALITY_LOG_PATH.exists(),
    }

