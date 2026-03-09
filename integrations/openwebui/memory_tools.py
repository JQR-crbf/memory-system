"""
Open WebUI Tool: Memory System integration

Create a new Tool in Open WebUI and paste this file content.
"""

import os
import json
import ast
import requests


class Tools:
    def __init__(self):
        self.base_url = os.getenv("MEMORY_API_BASE", "http://host.docker.internal:8000")
        self.timeout = 30
        self.api_key = os.getenv("MEMORY_API_KEY", "")

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def recall_memory(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant memories from the six-layer memory system."""
        payload = {
            "query": query,
            "layers": "auto",
            "top_k": top_k,
            "include_profile": True,
        }
        resp = requests.post(
            f"{self.base_url}/recall",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return str(data)

    async def save_memory(
        self,
        content: str,
        layer_hint: str = "auto",
        source: str = "manual",
    ) -> str:
        """Save one memory into the memory system."""
        payload = {
            "content": content,
            "layer_hint": layer_hint,
            "metadata": {"source": source},
        }
        resp = requests.post(
            f"{self.base_url}/remember",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return str(resp.json())

    async def ingest_worklog_text(
        self,
        report_text: str,
        report_date: str = "",
        source: str = "import_manual_worklog",
        max_items: int = 30,
    ) -> str:
        """
        Parse and ingest one report using backend shared rules.
        This keeps Tool-import and script-import consistent.
        """
        payload = {
            "report_text": report_text,
            "report_date": report_date,
            "source": source,
            "max_items": max_items,
            "dry_run": False,
        }
        resp = requests.post(
            f"{self.base_url}/ingest/worklog",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return str(resp.json())

    async def ingest_reports_json(
        self,
        reports_json: str,
        max_reports: int = 20,
        source: str = "import_manual_reports_json",
    ) -> str:
        """
        Ingest Chronicle-like reports JSON string.
        Expects payload with top-level key: reports[] containing date/type/content.
        """
        try:
            data = json.loads(reports_json)
        except Exception:
            return str({"status": "failed", "reason": "invalid_json"})

        reports = data.get("reports", [])
        if not isinstance(reports, list):
            return str({"status": "failed", "reason": "reports_not_list"})

        total_saved = 0
        total_errors = 0
        processed = 0
        for r in reports:
            if processed >= max_reports:
                break
            if not isinstance(r, dict):
                continue
            content = str(r.get("content", "")).strip()
            date = str(r.get("date", "")).strip()
            typ = str(r.get("type", "")).strip()
            if not content:
                continue
            if typ == "ai_insight":
                continue
            one = await self.ingest_worklog_text(
                report_text=content,
                report_date=date,
                source=source,
                max_items=20,
            )
            processed += 1
            try:
                parsed = ast.literal_eval(one) if one.startswith("{") else {}
            except Exception:
                parsed = {}
            total_saved += int(parsed.get("saved", 0))
            total_errors += int(parsed.get("errors", 0))

        return str(
            {
                "status": "done",
                "processed_reports": processed,
                "saved_points": total_saved,
                "errors": total_errors,
            }
        )

    async def first_event_date(
        self,
        keyword: str,
        source: str = "import_51talk_worklog",
        project: str | None = None,
        max_hits: int = 50,
    ) -> str:
        """
        Find the earliest date in episodic (L2) memories matching a keyword.

        Typical use:
        - When did I first start处理某个主题 (如学情数据)?
        - Scope can be limited by source and project.
        """
        filters: dict[str, object] = {"sources": [source]}
        if project:
            filters["project"] = project

        payload = {
            "query": keyword,
            "layers": ["L2"],
            "top_k": max_hits,
            "filters": filters,
            "include_profile": False,
        }
        resp = requests.post(
            f"{self.base_url}/recall",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []

        dates: list[str] = []
        for item in results:
            meta = (item or {}).get("metadata") or {}
            d = str(meta.get("date", "")).strip()
            if d:
                dates.append(d)

        if not dates:
            return str(
                {
                    "status": "not_found",
                    "reason": "no_dated_events",
                    "hits": len(results),
                }
            )

        first_date = min(dates)
        return str(
            {
                "status": "ok",
                "first_date": first_date,
                "total_hits": len(results),
            }
        )

    async def last_event_date(
        self,
        keyword: str,
        source: str = "import_51talk_worklog",
        project: str | None = None,
        max_hits: int = 50,
    ) -> str:
        """
        Find the latest date in episodic (L2) memories matching a keyword.
        Useful for checking "最近一次在做这个主题是什么时候".
        """
        filters: dict[str, object] = {"sources": [source]}
        if project:
            filters["project"] = project

        payload = {
            "query": keyword,
            "layers": ["L2"],
            "top_k": max_hits,
            "filters": filters,
            "include_profile": False,
        }
        resp = requests.post(
            f"{self.base_url}/recall",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []

        dates: list[str] = []
        for item in results:
            meta = (item or {}).get("metadata") or {}
            d = str(meta.get("date", "")).strip()
            if d:
                dates.append(d)

        if not dates:
            return str(
                {
                    "status": "not_found",
                    "reason": "no_dated_events",
                    "hits": len(results),
                }
            )

        latest_date = max(dates)
        return str(
            {
                "status": "ok",
                "last_date": latest_date,
                "total_hits": len(results),
            }
        )
