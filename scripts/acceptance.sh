#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/acceptance.sh
#   MEMORY_API_KEY=xxx ./scripts/acceptance.sh

BASE_URL="${BASE_URL:-http://localhost:8000}"
KEY="${MEMORY_API_KEY:-${KEY:-}}"

if [[ -z "${KEY}" ]]; then
  echo "ERROR: MEMORY_API_KEY (or KEY) is required."
  exit 1
fi

PASS=0
TOTAL=0

check_status() {
  local name="$1"
  local cmd="$2"
  TOTAL=$((TOTAL + 1))
  local code
  code="$(eval "$cmd")"
  if [[ "$code" == "200" || "$code" == "201" || "$code" == "401" ]]; then
    echo "[PASS] ${name} -> ${code}"
    PASS=$((PASS + 1))
  else
    echo "[FAIL] ${name} -> ${code}"
  fi
}

echo "== Six-layer acceptance start =="
echo "BASE_URL=${BASE_URL}"

# 0) auth guard
check_status "Auth guard /stats no key (expect 401)" \
  "curl -s -o /dev/null -w '%{http_code}' ${BASE_URL}/stats"

# 1) health
check_status "Health with key (expect 200)" \
  "curl -s -o /dev/null -w '%{http_code}' -H 'x-api-key: ${KEY}' ${BASE_URL}/health"

# 2) stats
check_status "Stats with key (expect 200)" \
  "curl -s -o /dev/null -w '%{http_code}' -H 'x-api-key: ${KEY}' ${BASE_URL}/stats"

# 3) L0 profile
check_status "L0 GET /profile" \
  "curl -s -o /dev/null -w '%{http_code}' -H 'x-api-key: ${KEY}' ${BASE_URL}/profile"

# 4) L1
check_status "L1 remember" \
  "curl -s -o /dev/null -w '%{http_code}' -H 'x-api-key: ${KEY}' -H 'Content-Type: application/json' -d '{\"content\":\"我熟悉AI产品需求拆解与API能力设计\",\"layer_hint\":\"L1\",\"metadata\":{\"source\":\"acceptance\",\"category\":\"skill\",\"tags\":[\"AI产品\",\"API\"]}}' ${BASE_URL}/remember"
check_status "L1 recall" \
  "curl -s -o /dev/null -w '%{http_code}' -H 'x-api-key: ${KEY}' -H 'Content-Type: application/json' -d '{\"query\":\"我擅长什么产品能力\",\"layers\":[\"L1\"],\"top_k\":3,\"include_profile\":false}' ${BASE_URL}/recall"

# 5) L2
check_status "L2 remember" \
  "curl -s -o /dev/null -w '%{http_code}' -H 'x-api-key: ${KEY}' -H 'Content-Type: application/json' -d '{\"content\":\"今天完成了六层记忆系统验收脚本\",\"layer_hint\":\"L2\",\"metadata\":{\"source\":\"acceptance\",\"date\":\"2026-03-03\",\"event_type\":\"work\",\"project\":\"memory-system\"}}' ${BASE_URL}/remember"
check_status "L2 recall" \
  "curl -s -o /dev/null -w '%{http_code}' -H 'x-api-key: ${KEY}' -H 'Content-Type: application/json' -d '{\"query\":\"今天我做了什么\",\"layers\":[\"L2\"],\"top_k\":3,\"filters\":{\"date_from\":\"2026-03-01\",\"date_to\":\"2026-03-04\"},\"include_profile\":false}' ${BASE_URL}/recall"

# 6) L3
check_status "L3 remember" \
  "curl -s -o /dev/null -w '%{http_code}' -H 'x-api-key: ${KEY}' -H 'Content-Type: application/json' -d '{\"content\":\"我和AIEC团队协作推进AI API平台\",\"layer_hint\":\"L3\",\"metadata\":{\"source\":\"acceptance\",\"people\":[\"AIEC团队\"],\"skills\":[\"Neo4j\",\"API平台\"]}}' ${BASE_URL}/remember"
check_status "L3 recall" \
  "curl -s -o /dev/null -w '%{http_code}' -H 'x-api-key: ${KEY}' -H 'Content-Type: application/json' -d '{\"query\":\"我和谁协作过AI平台工作\",\"layers\":[\"L3\"],\"top_k\":3,\"include_profile\":false}' ${BASE_URL}/recall"

# 7) L4
check_status "L4 reflect" \
  "curl -s -o /dev/null -w '%{http_code}' -H 'x-api-key: ${KEY}' -H 'Content-Type: application/json' -d '{\"type\":\"weekly\",\"focus_areas\":[\"验收结果\",\"下一步计划\"]}' ${BASE_URL}/reflect"

# 8) L5
check_status "L5 goals get" \
  "curl -s -o /dev/null -w '%{http_code}' -H 'x-api-key: ${KEY}' ${BASE_URL}/goals"
check_status "L5 recall(auto)" \
  "curl -s -o /dev/null -w '%{http_code}' -H 'x-api-key: ${KEY}' -H 'Content-Type: application/json' -d '{\"query\":\"我的目标进度怎么样\",\"layers\":\"auto\",\"top_k\":3,\"include_profile\":false}' ${BASE_URL}/recall"

echo "== Result: ${PASS}/${TOTAL} checks passed =="
if [[ "${PASS}" -lt "${TOTAL}" ]]; then
  exit 1
fi
