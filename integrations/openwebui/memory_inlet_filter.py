"""
Open WebUI Inlet Filter: auto recall memory and inject context.

Create a new Filter (Inlet enabled) in Open WebUI and paste this file content.
"""

import os
import re
import requests
from pydantic import BaseModel, Field


def _format_memory_context(recall_data: dict) -> str:
    profile = recall_data.get("profile", {})
    results = recall_data.get("results", [])

    profile_lines = []
    identity = profile.get("identity", {})
    career = profile.get("career", {})
    if identity:
        profile_lines.append(f"姓名: {identity.get('name', '')}")
    if career:
        profile_lines.append(f"职业: {career.get('current_role', '')}")

    memory_lines = []
    for item in results[:8]:
        layer = item.get("layer", "Lx")
        content = item.get("content", "")
        score = item.get("score", 0)
        memory_lines.append(f"- [{layer}] ({score:.2f}) {content}")

    return "\n".join(
        [
            "你正在使用六层记忆系统，以下是本轮可用记忆上下文：",
            "## L0 身份信息",
            *(profile_lines or ["- 暂无"]),
            "",
            "## 相关记忆",
            *(memory_lines or ["- 暂无"]),
            "",
            "回答时请优先利用这些记忆，并保持中文输出。",
        ]
    )


def _extract_scope_id(body: dict, user: dict | None) -> str:
    candidates = [
        body.get("chat_id"),
        body.get("conversation_id"),
        body.get("id"),
    ]
    if isinstance(user, dict):
        candidates.extend([user.get("chat_id"), user.get("id")])
    for c in candidates:
        if c is not None and str(c).strip():
            return str(c).strip()
    return "global"


class Filter:
    class Valves(BaseModel):
        memory_api_base: str = Field(
            default="http://host.docker.internal:8000",
            description="Memory API base URL",
        )
        memory_api_key: str = Field(
            default="",
            description="Memory API key",
        )
        memory_global_sources: str = Field(
            default="import_51talk_worklog,import_51talk_reference,acceptance,manual,chat",
            description="Comma-separated global memory sources",
        )
        timeout_seconds: int = Field(
            default=30,
            description="HTTP request timeout in seconds",
        )

    def __init__(self):
        self.valves = self.Valves()

    def _base_url(self) -> str:
        return (self.valves.memory_api_base or os.getenv("MEMORY_API_BASE", "")).strip().rstrip("/")

    def _timeout(self) -> int:
        return int(self.valves.timeout_seconds or 30)

    def _api_key(self) -> str:
        return (self.valves.memory_api_key or os.getenv("MEMORY_API_KEY", "")).strip()

    def _global_sources(self) -> list[str]:
        raw = (self.valves.memory_global_sources or os.getenv("MEMORY_GLOBAL_SOURCES", "")).strip()
        if not raw:
            raw = "import_51talk_worklog,import_51talk_reference,acceptance,manual,chat"
        return [x.strip() for x in raw.split(",") if x.strip()]

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        api_key = self._api_key()
        if api_key:
            headers["x-api-key"] = api_key
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _detect_intent(self, query: str) -> str:
        q = query.strip().lower()
        if re.search(r"(简历|cv|履历|述职)", q):
            return "resume"
        if re.search(r"(述职提纲|项目案例|案例卡|简历要点|职业资产)", q):
            return "career_assets"
        if re.search(r"(学什么|怎么学|学习计划|学习路径|复盘计划|学习任务)", q):
            return "learning_engine"
        if re.search(r"(我是谁|了解我|你怎么看我|我的画像|我是什么样的人|自我画像)", q):
            return "self_profile"
        if re.search(r"(风险|阻塞|卡点|里程碑|延期|延迟|项目状态|项目雷达)", q):
            return "project_radar"
        if re.search(r"(怎么决策|该选哪个|方案对比|决策建议|取舍)", q):
            return "decision_copilot"
        if re.search(r"(最近进展|当前进展|工作情况|到哪个阶段|处理到哪|状态|里程碑)", q):
            return "work_status"
        return "recall"

    def _is_relation_query(self, query: str) -> bool:
        q = query.strip().lower()
        return bool(
            re.search(
                r"(同事|关系|协作|合作|团队|领导|直属|汇报线|谁是|是谁|人际|伙伴|通讯录)",
                q,
            )
        )

    def _is_person_lookup_query(self, query: str) -> bool:
        q = query.strip()
        # Typical short follow-up questions like:
        # "谢雯萱", "那谢雯萱呢？", "刘京楠是谁"
        return bool(
            re.match(r"^(那|这|还有)?[\u4e00-\u9fff]{2,4}(是谁|是谁啊|是谁呀|呢)?[？?]?$", q)
        )

    async def inlet(self, body: dict, user=None) -> dict:
        messages = body.get("messages") or []
        if not messages:
            return body

        last = messages[-1]
        if last.get("role") != "user":
            return body

        query = (last.get("content") or "").strip()
        if not query:
            return body
        base_url = self._base_url()
        timeout = self._timeout()
        scope_id = _extract_scope_id(body, user if isinstance(user, dict) else None)
        scoped_source = f"chat::{scope_id}"
        allowed_sources = self._global_sources() + [scoped_source]
        intent = self._detect_intent(query)

        try:
            if intent != "recall":
                resp = requests.post(
                    f"{base_url}/synthesize/copilot",
                    json={
                        "query": query,
                        "task": intent,
                        "top_k": 8,
                        "style": "",
                        "include_profile": True,
                        "filters": {
                            "l1_sources": allowed_sources,
                            "l2_sources": allowed_sources,
                            "l3_sources": allowed_sources,
                        },
                    },
                    headers=self._headers(),
                    timeout=timeout,
                )
                resp.raise_for_status()
                wrapper = resp.json()
                if not isinstance(wrapper, dict):
                    wrapper = {}
                data = wrapper.get("result", {}) if isinstance(wrapper, dict) else {}
                highlights = data.get("highlights", []) or data.get("weekly_tasks", []) or []
                evidence = data.get("evidence", []) or []
                evidence_lines = [f"- [{x.get('layer','Lx')}] {x.get('snippet','')}" for x in evidence[:8]]
                memory_context = "\n".join(
                    [
                        "你正在使用六层记忆系统，以下是本轮 Copilot 合成上下文：",
                        f"task: {wrapper.get('copilot_task', intent)}",
                        f"confidence: {data.get('confidence', 0)}",
                        "## 合成结果",
                        str(data.get("result", "")).strip() or "- 暂无",
                        "",
                        "## 关键要点",
                        *(highlights or ["- 暂无"]),
                        "",
                        "## 证据摘要",
                        *(evidence_lines or ["- 暂无"]),
                        "",
                        "回答时优先基于上述结果，证据不足时请明确说明。",
                    ]
                )
            else:
                recall_filters = {
                    "l1_sources": allowed_sources,
                    "l2_sources": allowed_sources,
                    "l3_sources": allowed_sources,
                }
                # For people/relationship questions, do not restrict L3 to current scope
                # so historical chat-scoped relation memories can be recalled.
                relation_like = self._is_relation_query(query) or self._is_person_lookup_query(query)
                if relation_like:
                    recall_filters.pop("l3_sources", None)
                recall_layers = ["L3", "L1"] if relation_like else "auto"

                resp = requests.post(
                    f"{base_url}/recall",
                    json={
                        "query": query,
                        "layers": recall_layers,
                        "top_k": 6,
                        "include_profile": True,
                        "filters": recall_filters,
                    },
                    headers=self._headers(),
                    timeout=timeout,
                )
                resp.raise_for_status()
                recall_data = resp.json()
                memory_context = _format_memory_context(recall_data)
        except Exception as e:  # noqa: BLE001
            memory_context = (
                "六层记忆系统暂时不可用，请先按当前对话回答。\n"
                f"错误信息: {str(e)}"
            )

        messages.insert(
            0,
            {
                "role": "system",
                "content": memory_context,
            },
        )
        body["messages"] = messages
        return body
