#!/usr/bin/env bash
set -euo pipefail

# Cleanup obvious test/demo noise while keeping 51Talk business memories.
# This script targets known QA/demo traces in Qdrant collections.

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"

echo "[1/3] cleanup semantic_memory: source=qa OR tags include FastAPI/Python"
curl -s -X POST "${QDRANT_URL}/collections/semantic_memory/points/delete?wait=true" \
  -H "Content-Type: application/json" \
  -d '{"filter":{"must":[{"key":"source","match":{"value":"qa"}}]}}' >/dev/null
curl -s -X POST "${QDRANT_URL}/collections/semantic_memory/points/delete?wait=true" \
  -H "Content-Type: application/json" \
  -d '{"filter":{"must":[{"key":"tags","match":{"any":["FastAPI","Python"]}}]}}' >/dev/null

echo "[2/3] cleanup episodic_memory: source=qa or project=memory-system demo traces"
curl -s -X POST "${QDRANT_URL}/collections/episodic_memory/points/delete?wait=true" \
  -H "Content-Type: application/json" \
  -d '{"filter":{"must":[{"key":"source","match":{"value":"qa"}}]}}' >/dev/null
curl -s -X POST "${QDRANT_URL}/collections/episodic_memory/points/delete?wait=true" \
  -H "Content-Type: application/json" \
  -d '{"filter":{"must":[{"key":"project","match":{"value":"memory-system"}}]}}' >/dev/null

echo "[3/3] done. current points:"
echo "semantic_memory=$(curl -s ${QDRANT_URL}/collections/semantic_memory | python3 -c 'import sys,json;print(json.load(sys.stdin).get(\"result\",{}).get(\"points_count\",0))')"
echo "episodic_memory=$(curl -s ${QDRANT_URL}/collections/episodic_memory | python3 -c 'import sys,json;print(json.load(sys.stdin).get(\"result\",{}).get(\"points_count\",0))')"
echo "reflective_memory=$(curl -s ${QDRANT_URL}/collections/reflective_memory | python3 -c 'import sys,json;print(json.load(sys.stdin).get(\"result\",{}).get(\"points_count\",0))')"
echo "aspiration_memory=$(curl -s ${QDRANT_URL}/collections/aspiration_memory | python3 -c 'import sys,json;print(json.load(sys.stdin).get(\"result\",{}).get(\"points_count\",0))')"
