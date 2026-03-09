"""
Open WebUI Outlet Filter: auto save memory from conversation.

Create a new Filter (Outlet enabled) in Open WebUI and paste this file content.
"""

import os
import requests
from pydantic import BaseModel, Field


def _extract_user_message(messages: list[dict]) -> str:
    for item in reversed(messages):
        if item.get("role") == "user":
            return (item.get("content") or "").strip()
    return ""


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
        timeout_seconds: int = Field(
            default=30,
            description="HTTP request timeout in seconds",
        )
        min_user_text_len: int = Field(
            default=6,
            description="Minimum user text length to autosave",
        )

    def __init__(self):
        self.valves = self.Valves()

    def _base_url(self) -> str:
        return (self.valves.memory_api_base or os.getenv("MEMORY_API_BASE", "")).strip().rstrip("/")

    def _timeout(self) -> int:
        return int(self.valves.timeout_seconds or 30)

    def _api_key(self) -> str:
        return (self.valves.memory_api_key or os.getenv("MEMORY_API_KEY", "")).strip()

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        api_key = self._api_key()
        if api_key:
            headers["x-api-key"] = api_key
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def outlet(self, body: dict, user=None) -> dict:
        messages = body.get("messages") or []
        if not messages:
            return body

        user_text = _extract_user_message(messages)
        if not user_text:
            return body

        # Basic noise filter for short chit-chat.
        if len(user_text) < int(self.valves.min_user_text_len or 6):
            return body
        scope_id = _extract_scope_id(body, user if isinstance(user, dict) else None)
        source = f"chat::{scope_id}"

        try:
            requests.post(
                f"{self._base_url()}/remember",
                json={
                    "content": user_text,
                    "layer_hint": "auto",
                    "metadata": {
                        "source": source,
                        "scope_id": scope_id,
                        "scope_type": "chat",
                    },
                },
                headers=self._headers(),
                timeout=self._timeout(),
            )
        except Exception:
            # Do not block chat if memory save fails.
            pass

        return body
