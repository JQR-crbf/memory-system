#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"

echo "[1/5] health"
curl -s "${BASE_URL}/health" | jq .

echo "[2/5] create goal"
curl -s -X POST "${BASE_URL}/goals" \
  -H "Content-Type: application/json" \
  -d '{
    "id":"goal-demo-001",
    "title":"完成六层记忆系统演示",
    "category":"project",
    "priority":"high",
    "deadline":"2026-12-31",
    "status":"in_progress",
    "progress":20,
    "related_skills":["FastAPI","Qdrant","Neo4j"],
    "notes":"用于端到端演示"
  }' | jq .

echo "[3/5] remember"
curl -s -X POST "${BASE_URL}/remember" \
  -H "Content-Type: application/json" \
  -d '{
    "content":"今天完成了记忆系统的端到端演示脚本",
    "layer_hint":"auto",
    "metadata":{"source":"manual","date":"2026-03-03"}
  }' | jq .

echo "[4/5] recall"
curl -s -X POST "${BASE_URL}/recall" \
  -H "Content-Type: application/json" \
  -d '{
    "query":"我的目标和今天进展",
    "layers":"auto",
    "top_k":5,
    "include_profile":false
  }' | jq .

echo "[5/5] stats"
curl -s "${BASE_URL}/stats" | jq .
