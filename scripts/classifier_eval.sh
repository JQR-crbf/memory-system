#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   MEMORY_API_KEY=xxx ./scripts/classifier_eval.sh
#   BASE_URL=http://localhost:8000 MEMORY_API_KEY=xxx ./scripts/classifier_eval.sh

BASE_URL="${BASE_URL:-http://localhost:8000}"
KEY="${MEMORY_API_KEY:-${KEY:-}}"

if [[ -z "${KEY}" ]]; then
  echo "ERROR: MEMORY_API_KEY (or KEY) is required."
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required."
  exit 1
fi

post_recall() {
  local query="$1"
  curl -s -X POST "${BASE_URL}/recall" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${KEY}" \
    -d "{\"query\":\"${query}\",\"layers\":\"auto\",\"top_k\":5,\"include_profile\":false}"
}

echo "== Classifier Eval Start =="
echo "BASE_URL=${BASE_URL}"

# Warm-up sample to let learned routes kick in.
WARMUP_QUERY="结合我的长期轨迹，谁是对我产出影响最大的人物节点？"
echo "-- warmup query (x2): ${WARMUP_QUERY}"
post_recall "${WARMUP_QUERY}" >/dev/null
post_recall "${WARMUP_QUERY}" >/dev/null

declare -a QUERIES=(
  "我最好的同事是谁"
  "帮我回顾今天完成了哪些工作"
  "未来90天我最该优先推进什么"
  "从历史线索看谁最影响我的工作结果"
  "下一阶段学习计划我该怎么排"
  "我在人际协作网络里的关键伙伴是谁"
)

TOTAL=0
LLM_LEARNED=0
RULE_LEARNED=0
STATIC_RULE=0
CONSISTENT=0

for query in "${QUERIES[@]}"; do
  echo ""
  echo "-- query: ${query}"
  resp1="$(post_recall "${query}")"
  resp2="$(post_recall "${query}")"

  method1="$(echo "${resp1}" | jq -r '.classification_method // ""')"
  method2="$(echo "${resp2}" | jq -r '.classification_method // ""')"
  layers1="$(echo "${resp1}" | jq -c '.routed_layers // []')"
  layers2="$(echo "${resp2}" | jq -c '.routed_layers // []')"

  echo "   #1 method=${method1} layers=${layers1}"
  echo "   #2 method=${method2} layers=${layers2}"

  TOTAL=$((TOTAL + 2))
  if [[ "${method1}" == llm:learned* ]]; then LLM_LEARNED=$((LLM_LEARNED + 1)); fi
  if [[ "${method2}" == llm:learned* ]]; then LLM_LEARNED=$((LLM_LEARNED + 1)); fi
  if [[ "${method1}" == rule:learned* ]]; then RULE_LEARNED=$((RULE_LEARNED + 1)); fi
  if [[ "${method2}" == rule:learned* ]]; then RULE_LEARNED=$((RULE_LEARNED + 1)); fi
  if [[ "${method1}" == rule:* && "${method1}" != rule:learned* ]]; then STATIC_RULE=$((STATIC_RULE + 1)); fi
  if [[ "${method2}" == rule:* && "${method2}" != rule:learned* ]]; then STATIC_RULE=$((STATIC_RULE + 1)); fi

  if [[ "${layers1}" == "${layers2}" ]]; then
    CONSISTENT=$((CONSISTENT + 1))
  fi
done

consistency_rate="$(awk "BEGIN { printf \"%.2f\", (${CONSISTENT} / ${#QUERIES[@]}) * 100 }")"
llm_rate="$(awk "BEGIN { printf \"%.2f\", (${LLM_LEARNED} / ${TOTAL}) * 100 }")"
learned_rate="$(awk "BEGIN { printf \"%.2f\", (${RULE_LEARNED} / ${TOTAL}) * 100 }")"
static_rate="$(awk "BEGIN { printf \"%.2f\", (${STATIC_RULE} / ${TOTAL}) * 100 }")"

echo ""
echo "== Metrics =="
echo "total_calls=${TOTAL}"
echo "llm_fallback_count=${LLM_LEARNED} (${llm_rate}%)"
echo "learned_rule_count=${RULE_LEARNED} (${learned_rate}%)"
echo "static_rule_count=${STATIC_RULE} (${static_rate}%)"
echo "routing_consistency=${CONSISTENT}/${#QUERIES[@]} (${consistency_rate}%)"

echo ""
echo "== Debug Snapshot =="
curl -s "${BASE_URL}/classifier/debug/last?action=recall&recent=10" \
  -H "Authorization: Bearer ${KEY}" | jq '{last_event, rules_overview}'

echo ""
echo "== Classifier Eval Done =="
