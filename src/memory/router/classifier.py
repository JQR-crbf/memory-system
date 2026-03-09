from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import httpx
from memory.embedding.embedder import EmbedderError, embed_text
from memory.layers.qdrant_client import QdrantError, ensure_collection, search_points, upsert_point


LOG_PATH = Path("/data/system/classification_log.jsonl")
RULES_PATH = Path("/data/system/classification_rules.json")
SETTINGS_PATH = Path("/data/system/classifier_settings.json")
SUPPORTED_LAYERS = {"L1", "L2", "L3", "L5"}
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", os.getenv("GEN_MODEL", "qwen3:8b"))
CLASSIFIER_LLM_TEMPERATURE = float(os.getenv("CLASSIFIER_LLM_TEMPERATURE", "0"))
CLASSIFIER_LLM_TOP_P = float(os.getenv("CLASSIFIER_LLM_TOP_P", "1"))
CLASSIFIER_VECTOR_COLLECTION = os.getenv("CLASSIFIER_VECTOR_COLLECTION", "classification_router")
CLASSIFIER_VECTOR_THRESHOLD = float(os.getenv("CLASSIFIER_VECTOR_THRESHOLD", "0.78"))
CLASSIFIER_VECTOR_TOP_K = int(os.getenv("CLASSIFIER_VECTOR_TOP_K", "3"))
CLASSIFIER_LEARNING_ENABLED = os.getenv("CLASSIFIER_LEARNING_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CLASSIFIER_VECTOR_ENABLED = os.getenv("CLASSIFIER_VECTOR_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CLASSIFIER_LLM_FALLBACK_ENABLED = os.getenv("CLASSIFIER_LLM_FALLBACK_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

L2_KEYWORDS = [
    "今天",
    "昨天",
    "昨日",
    "本周",
    "刚刚",
    "完成了",
    "遇到了",
    "进展",
    "日报",
    "周报",
]

L1_KEYWORDS = [
    "技能",
    "知识",
    "方法论",
    "经验",
    "我会什么",
    "怎么做",
    "最佳实践",
]

L3_KEYWORDS = [
    "关系",
    "人脉",
    "同事",
    "合作",
    "认识谁",
    "谁认识",
    "依赖",
    "图谱",
]

L5_KEYWORDS = [
    "目标",
    "计划",
    "进度",
    "deadline",
    "截止",
    "里程碑",
    "想学",
    "学习计划",
]

LOW_SIGNAL_PATTERNS = {
    "你好",
    "您好",
    "早上好",
    "晚上好",
    "谢谢",
    "好的",
    "收到",
    "在吗",
    "测试",
    "ok",
    "yes",
}

MIN_LEARNED_PATTERN_LEN = 4


def _to_text(value: str) -> str:
    return value.strip().lower()


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_rules() -> dict[str, list[dict[str, Any]]]:
    if not RULES_PATH.exists():
        return {"remember": [], "recall": []}
    try:
        with RULES_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"remember": [], "recall": []}
    remember = data.get("remember")
    recall = data.get("recall")
    return {
        "remember": remember if isinstance(remember, list) else [],
        "recall": recall if isinstance(recall, list) else [],
    }


def _default_settings() -> dict[str, Any]:
    return {
        "learning_enabled": CLASSIFIER_LEARNING_ENABLED,
        "vector_enabled": CLASSIFIER_VECTOR_ENABLED,
        "llm_fallback_enabled": CLASSIFIER_LLM_FALLBACK_ENABLED,
        "vector_threshold": CLASSIFIER_VECTOR_THRESHOLD,
    }


def get_classifier_settings() -> dict[str, Any]:
    settings = _default_settings()
    if not SETTINGS_PATH.exists():
        return settings
    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f) or {}
    except Exception:
        return settings
    if isinstance(raw.get("learning_enabled"), bool):
        settings["learning_enabled"] = raw["learning_enabled"]
    if isinstance(raw.get("vector_enabled"), bool):
        settings["vector_enabled"] = raw["vector_enabled"]
    if isinstance(raw.get("llm_fallback_enabled"), bool):
        settings["llm_fallback_enabled"] = raw["llm_fallback_enabled"]
    try:
        threshold = float(raw.get("vector_threshold", settings["vector_threshold"]))
        settings["vector_threshold"] = max(0.0, min(1.0, threshold))
    except (TypeError, ValueError):
        pass
    return settings


def update_classifier_settings(updates: dict[str, Any]) -> dict[str, Any]:
    settings = get_classifier_settings()
    if "learning_enabled" in updates and isinstance(updates["learning_enabled"], bool):
        settings["learning_enabled"] = updates["learning_enabled"]
    if "vector_enabled" in updates and isinstance(updates["vector_enabled"], bool):
        settings["vector_enabled"] = updates["vector_enabled"]
    if "llm_fallback_enabled" in updates and isinstance(updates["llm_fallback_enabled"], bool):
        settings["llm_fallback_enabled"] = updates["llm_fallback_enabled"]
    if "vector_threshold" in updates and updates["vector_threshold"] is not None:
        settings["vector_threshold"] = max(0.0, min(1.0, float(updates["vector_threshold"])))
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SETTINGS_PATH.open("w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
    return settings


def get_rules(task: str | None = None, limit: int = 200, include_disabled: bool = True) -> dict[str, list[dict[str, Any]]]:
    rules = _load_rules()
    limit = max(1, min(1000, int(limit)))
    if not include_disabled:
        rules = {
            "remember": [x for x in rules.get("remember", []) if not bool(x.get("disabled", False))],
            "recall": [x for x in rules.get("recall", []) if not bool(x.get("disabled", False))],
        }
    if task in {"remember", "recall"}:
        return {
            task: sorted(rules.get(task, []), key=lambda x: int(x.get("hits", 0)), reverse=True)[:limit]
        }
    return {
        "remember": sorted(rules.get("remember", []), key=lambda x: int(x.get("hits", 0)), reverse=True)[:limit],
        "recall": sorted(rules.get("recall", []), key=lambda x: int(x.get("hits", 0)), reverse=True)[:limit],
    }


def delete_rule(
    task: str,
    pattern: str,
    layer: str | None = None,
    layers: list[str] | None = None,
) -> bool:
    if task not in {"remember", "recall"}:
        return False
    rules = _load_rules()
    items = rules.get(task, [])
    original = len(items)
    if task == "remember":
        items = [
            x
            for x in items
            if not (
                str(x.get("pattern", "")).strip() == pattern.strip()
                and (layer is None or str(x.get("layer", "")) == layer)
            )
        ]
    else:
        normalized = normalize_layers(layers or [])
        items = [
            x
            for x in items
            if not (
                str(x.get("pattern", "")).strip() == pattern.strip()
                and (not normalized or normalize_layers(x.get("layers", [])) == normalized)
            )
        ]
    rules[task] = items
    _save_rules(rules)
    return len(items) != original


def set_rule_enabled(
    task: str,
    pattern: str,
    enabled: bool,
    layer: str | None = None,
    layers: list[str] | None = None,
) -> bool:
    if task not in {"remember", "recall"}:
        return False
    rules = _load_rules()
    items = rules.get(task, [])
    changed = False
    normalized_layers = normalize_layers(layers or [])
    for item in items:
        if str(item.get("pattern", "")).strip() != pattern.strip():
            continue
        if task == "remember":
            if layer and str(item.get("layer", "")) != layer:
                continue
        else:
            if normalized_layers and normalize_layers(item.get("layers", [])) != normalized_layers:
                continue
        item["disabled"] = not enabled
        item["updated_at"] = _now_iso()
        changed = True
    if changed:
        rules[task] = items
        _save_rules(rules)
    return changed


def _save_rules(rules: dict[str, list[dict[str, Any]]]) -> None:
    RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RULES_PATH.open("w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)


def _derive_patterns(text: str) -> list[str]:
    text = _to_text(text)
    raw_parts = [text]
    raw_parts.extend(re.split(r"[，。！？、；;,\.\s]+", text))

    out: list[str] = []
    for part in raw_parts:
        part = part.strip()
        if len(part) < MIN_LEARNED_PATTERN_LEN:
            continue
        if part in LOW_SIGNAL_PATTERNS:
            continue
        if len(part) > 40:
            part = part[:40]
        if part not in out:
            out.append(part)
        if len(out) >= 8:
            break

    # Add a few short windows so similar Chinese queries can hit learned rules.
    if len(text) >= 8:
        window = 10
        starts = [0, max(0, len(text) // 3 - 2), max(0, (len(text) * 2) // 3 - 2)]
        for s in starts:
            if s + 4 >= len(text):
                continue
            piece = text[s : s + window].strip()
            if (
                len(piece) >= MIN_LEARNED_PATTERN_LEN
                and piece not in LOW_SIGNAL_PATTERNS
                and piece not in out
            ):
                out.append(piece)
            if len(out) >= 12:
                break
    return out


def _char_bigrams(text: str) -> set[str]:
    text = _to_text(text)
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(0, len(text) - 1)}


def _similarity(a: str, b: str) -> float:
    a_set = _char_bigrams(a)
    b_set = _char_bigrams(b)
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    if union == 0:
        return 0.0
    return inter / union


def _is_low_signal_pattern(pattern: str) -> bool:
    return len(pattern) < MIN_LEARNED_PATTERN_LEN or pattern in LOW_SIGNAL_PATTERNS


def _dynamic_similarity_threshold(pattern: str, text: str) -> float:
    """
    Raise similarity threshold for short texts to avoid noisy over-matching.
    """
    threshold = 0.45
    if len(pattern) <= 6:
        threshold = max(threshold, 0.6)
    if len(text) <= 8:
        threshold = max(threshold, 0.62)
    return threshold


def _match_dynamic_rule(
    text: str,
    task: str,
) -> tuple[str | list[str] | None, str | None]:
    rules = _load_rules()
    task_rules = rules.get(task, [])
    for item in sorted(task_rules, key=lambda x: x.get("hits", 0), reverse=True):
        if bool(item.get("disabled", False)):
            continue
        pattern = str(item.get("pattern", "")).strip().lower()
        if not pattern:
            continue
        if _is_low_signal_pattern(pattern):
            continue
        score = _similarity(pattern, text)
        threshold = _dynamic_similarity_threshold(pattern, text)
        if pattern in text or score >= threshold:
            if task == "remember":
                return str(item.get("layer", "")), f"rule:learned:{pattern}:{round(score, 2)}"
            return item.get("layers", []), f"rule:learned:{pattern}:{round(score, 2)}"
    return None, None


def _extract_json_block(text: str) -> dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return {}


async def _learn_vector_rule_remember(text: str, layer: str, source: str) -> None:
    if layer not in SUPPORTED_LAYERS:
        return
    try:
        vector = await embed_text(text)
        await ensure_collection(CLASSIFIER_VECTOR_COLLECTION, len(vector))
        point_id = str(uuid5(NAMESPACE_URL, f"remember:{_to_text(text)}:{layer}"))
        await upsert_point(
            CLASSIFIER_VECTOR_COLLECTION,
            vector,
            {
                "task": "remember",
                "text": text[:200],
                "layer": layer,
                "source": source,
                "updated_at": _now_iso(),
            },
            point_id=point_id,
        )
    except (EmbedderError, QdrantError):
        return


async def _learn_vector_rule_recall(text: str, layers: list[str], source: str) -> None:
    layers = normalize_layers(layers)
    if not layers:
        return
    try:
        vector = await embed_text(text)
        await ensure_collection(CLASSIFIER_VECTOR_COLLECTION, len(vector))
        joined = ",".join(layers)
        point_id = str(uuid5(NAMESPACE_URL, f"recall:{_to_text(text)}:{joined}"))
        await upsert_point(
            CLASSIFIER_VECTOR_COLLECTION,
            vector,
            {
                "task": "recall",
                "text": text[:200],
                "layers": layers,
                "source": source,
                "updated_at": _now_iso(),
            },
            point_id=point_id,
        )
    except (EmbedderError, QdrantError):
        return


async def _match_vector_rule_remember(text: str) -> tuple[str | None, str | None]:
    settings = get_classifier_settings()
    if not settings.get("vector_enabled", True):
        return None, None
    try:
        vector = await embed_text(text)
        hits = await search_points(
            CLASSIFIER_VECTOR_COLLECTION,
            vector,
            limit=CLASSIFIER_VECTOR_TOP_K,
            query_filter={"must": [{"key": "task", "match": {"value": "remember"}}]},
        )
    except (EmbedderError, QdrantError):
        return None, None
    if not hits:
        return None, None
    best = hits[0]
    score = float(best.get("score") or 0)
    payload = best.get("payload") or {}
    layer = str(payload.get("layer", ""))
    if layer in SUPPORTED_LAYERS and score >= float(settings.get("vector_threshold", CLASSIFIER_VECTOR_THRESHOLD)):
        return layer, f"rule:learned:vector:{round(score, 3)}"
    return None, None


async def _match_vector_rule_recall(text: str) -> tuple[list[str] | None, str | None]:
    settings = get_classifier_settings()
    if not settings.get("vector_enabled", True):
        return None, None
    try:
        vector = await embed_text(text)
        hits = await search_points(
            CLASSIFIER_VECTOR_COLLECTION,
            vector,
            limit=CLASSIFIER_VECTOR_TOP_K,
            query_filter={"must": [{"key": "task", "match": {"value": "recall"}}]},
        )
    except (EmbedderError, QdrantError):
        return None, None
    if not hits:
        return None, None
    best = hits[0]
    score = float(best.get("score") or 0)
    payload = best.get("payload") or {}
    layers = normalize_layers(payload.get("layers", []))
    if layers and score >= float(settings.get("vector_threshold", CLASSIFIER_VECTOR_THRESHOLD)):
        return layers, f"rule:learned:vector:{round(score, 3)}"
    return None, None


def _learn_rule_remember(text: str, layer: str, source: str) -> None:
    if layer not in SUPPORTED_LAYERS:
        return
    rules = _load_rules()
    items = rules["remember"]
    patterns = _derive_patterns(text)
    for pattern in patterns:
        found = next((x for x in items if x.get("pattern") == pattern and x.get("layer") == layer), None)
        if found:
            found["hits"] = int(found.get("hits", 0)) + 1
            found["updated_at"] = _now_iso()
            found["source"] = source
        else:
            items.append(
                {
                    "pattern": pattern,
                    "layer": layer,
                    "hits": 1,
                    "source": source,
                    "created_at": _now_iso(),
                    "updated_at": _now_iso(),
                }
            )
    rules["remember"] = items[-400:]
    _save_rules(rules)


def _learn_rule_recall(text: str, layers: list[str], source: str) -> None:
    layers = [x for x in layers if x in SUPPORTED_LAYERS]
    if not layers:
        return
    rules = _load_rules()
    items = rules["recall"]
    patterns = _derive_patterns(text)
    for pattern in patterns:
        found = next((x for x in items if x.get("pattern") == pattern and x.get("layers") == layers), None)
        if found:
            found["hits"] = int(found.get("hits", 0)) + 1
            found["updated_at"] = _now_iso()
            found["source"] = source
        else:
            items.append(
                {
                    "pattern": pattern,
                    "layers": layers,
                    "hits": 1,
                    "source": source,
                    "created_at": _now_iso(),
                    "updated_at": _now_iso(),
                }
            )
    rules["recall"] = items[-400:]
    _save_rules(rules)


async def _llm_classify_remember(content: str, metadata: dict[str, Any] | None = None) -> tuple[str, float]:
    metadata = metadata or {}
    prompt = f"""
你是记忆路由分类器。请将输入文本分类到以下层之一：
- L1: 知识/技能/方法论
- L2: 事件/时间线/日报周报
- L3: 人际关系/协作关系
- L5: 目标/计划/进度

输入文本：
{content}

metadata:
{json.dumps(metadata, ensure_ascii=False)}

仅返回 JSON，格式：
{{"layer":"L1","confidence":0.82}}
"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": CLASSIFIER_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": CLASSIFIER_LLM_TEMPERATURE,
                    "top_p": CLASSIFIER_LLM_TOP_P,
                },
            },
        )
    if r.status_code != 200:
        return "L1", 0.0
    text = (r.json().get("response") or "").strip()
    data = _extract_json_block(text)
    layer = str(data.get("layer", "L1"))
    confidence = float(data.get("confidence", 0.6))
    if layer not in SUPPORTED_LAYERS:
        return "L1", 0.2
    return layer, confidence


async def _llm_classify_recall(query: str) -> tuple[list[str], float]:
    prompt = f"""
你是检索路由分类器。请根据用户问题判断应查询哪些层（可多选）：
- L1: 知识/技能
- L2: 事件/时间线
- L3: 人际关系/协作
- L5: 目标计划

用户问题：
{query}

仅返回 JSON，格式：
{{"layers":["L1"],"confidence":0.82}}
"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": CLASSIFIER_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": CLASSIFIER_LLM_TEMPERATURE,
                    "top_p": CLASSIFIER_LLM_TOP_P,
                },
            },
        )
    if r.status_code != 200:
        return ["L1"], 0.0
    text = (r.json().get("response") or "").strip()
    data = _extract_json_block(text)
    layers = data.get("layers", [])
    if isinstance(layers, str):
        layers = [layers]
    out = [x for x in layers if x in SUPPORTED_LAYERS]
    confidence = float(data.get("confidence", 0.6))
    return (out or ["L1"]), confidence


def classify_for_remember(content: str, metadata: dict[str, Any] | None = None) -> tuple[str, str]:
    metadata = metadata or {}
    if metadata.get("people"):
        return "L3", "rule:metadata.people"
    if metadata.get("date"):
        return "L2", "rule:metadata.date"

    text = _to_text(content)
    if _contains_any(text, L3_KEYWORDS):
        return "L3", "rule:l3_keywords"
    if _contains_any(text, L2_KEYWORDS):
        return "L2", "rule:l2_keywords"
    learned_layer, learned_method = _match_dynamic_rule(text, "remember")
    if isinstance(learned_layer, str) and learned_layer in SUPPORTED_LAYERS:
        return learned_layer, learned_method or "rule:learned"
    return "L1", "rule:default_l1"


def classify_for_recall(query: str) -> tuple[list[str], str]:
    text = _to_text(query)
    learned_layers, learned_method = _match_dynamic_rule(text, "recall")
    if isinstance(learned_layers, list) and learned_layers:
        return normalize_layers(learned_layers), learned_method or "rule:learned"

    has_l5 = _contains_any(text, L5_KEYWORDS)
    has_l3 = _contains_any(text, L3_KEYWORDS)
    has_l2 = _contains_any(text, L2_KEYWORDS)
    has_l1 = _contains_any(text, L1_KEYWORDS)

    if has_l5 and has_l1:
        return ["L1", "L5"], "rule:mixed_l1_l5"
    if has_l5 and has_l2:
        return ["L2", "L5"], "rule:mixed_l2_l5"
    if has_l5:
        return ["L5"], "rule:l5_keywords"
    if has_l3 and has_l1:
        return ["L1", "L3"], "rule:mixed_l1_l3"
    if has_l3 and has_l2:
        return ["L2", "L3"], "rule:mixed_l2_l3"
    if has_l3:
        return ["L3"], "rule:l3_keywords"
    if has_l1 and has_l2:
        return ["L1", "L2"], "rule:mixed_l1_l2"
    if has_l2:
        return ["L2"], "rule:l2_keywords"
    if has_l1:
        return ["L1"], "rule:l1_keywords"
    return ["L1"], "rule:default_l1"


def normalize_layers(raw_layers: str | list[str]) -> list[str]:
    if isinstance(raw_layers, str):
        layers = [raw_layers]
    else:
        layers = raw_layers
    deduped: list[str] = []
    for layer in layers:
        if layer in SUPPORTED_LAYERS and layer not in deduped:
            deduped.append(layer)
    return deduped


async def classify_for_remember_with_learning(
    content: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, str]:
    settings = get_classifier_settings()
    learning_enabled = bool(settings.get("learning_enabled", True))
    llm_fallback_enabled = bool(settings.get("llm_fallback_enabled", True))
    layer, method = classify_for_remember(content, metadata)
    if method.startswith("rule:default_l1"):
        learned_layer, learned_method = await _match_vector_rule_remember(content)
        if learned_layer:
            if learning_enabled:
                await _learn_vector_rule_remember(content, learned_layer, source="vector_hit")
                _learn_rule_remember(content, learned_layer, source="vector_hit")
            return learned_layer, learned_method or "rule:learned:vector"
        if not llm_fallback_enabled:
            return layer, "rule:default_l1:fallback_disabled"
        llm_layer, confidence = await _llm_classify_remember(content, metadata)
        if learning_enabled:
            _learn_rule_remember(content, llm_layer, source="llm_fallback")
            await _learn_vector_rule_remember(content, llm_layer, source="llm_fallback")
        return llm_layer, f"llm:learned:{round(confidence, 3)}"
    if learning_enabled and method.startswith("rule:learned"):
        _learn_rule_remember(content, layer, source="learned_hit")
        await _learn_vector_rule_remember(content, layer, source="learned_hit")
    return layer, method


async def classify_for_recall_with_learning(query: str) -> tuple[list[str], str]:
    settings = get_classifier_settings()
    learning_enabled = bool(settings.get("learning_enabled", True))
    llm_fallback_enabled = bool(settings.get("llm_fallback_enabled", True))
    layers, method = classify_for_recall(query)
    if method.startswith("rule:default_l1"):
        learned_layers, learned_method = await _match_vector_rule_recall(query)
        if learned_layers:
            if learning_enabled:
                await _learn_vector_rule_recall(query, learned_layers, source="vector_hit")
                _learn_rule_recall(query, learned_layers, source="vector_hit")
            return learned_layers, learned_method or "rule:learned:vector"
        if not llm_fallback_enabled:
            return layers, "rule:default_l1:fallback_disabled"
        llm_layers, confidence = await _llm_classify_recall(query)
        llm_layers = normalize_layers(llm_layers)
        if learning_enabled:
            _learn_rule_recall(query, llm_layers, source="llm_fallback")
            await _learn_vector_rule_recall(query, llm_layers, source="llm_fallback")
        return llm_layers, f"llm:learned:{round(confidence, 3)}"
    if learning_enabled and method.startswith("rule:learned"):
        _learn_rule_recall(query, layers, source="learned_hit")
        await _learn_vector_rule_recall(query, layers, source="learned_hit")
    return layers, method


async def apply_manual_feedback_remember(content: str, layer: str) -> dict[str, Any]:
    if layer not in SUPPORTED_LAYERS:
        return {"ok": False, "detail": f"unsupported layer: {layer}"}
    _learn_rule_remember(content, layer, source="manual_feedback")
    await _learn_vector_rule_remember(content, layer, source="manual_feedback")
    return {"ok": True, "task": "remember", "layer": layer, "content": content[:120]}


async def apply_manual_feedback_recall(query: str, layers: list[str]) -> dict[str, Any]:
    normalized = normalize_layers(layers)
    if not normalized:
        return {"ok": False, "detail": "no supported layers in feedback"}
    _learn_rule_recall(query, normalized, source="manual_feedback")
    await _learn_vector_rule_recall(query, normalized, source="manual_feedback")
    return {"ok": True, "task": "recall", "layers": normalized, "query": query[:120]}


def log_classification(event: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
