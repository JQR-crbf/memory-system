# Memory System Copilot Reference

## Base URL

```text
http://localhost:8000
```

If the agent runs inside another container, replace with the reachable service URL.

## Headers

If API auth is enabled:

```text
x-api-key: <MEMORY_API_KEY>
Authorization: Bearer <MEMORY_API_KEY>
```

## General recall payload

```json
{
  "query": "帮我回顾一下最近在做什么",
  "layers": "auto",
  "top_k": 6,
  "include_profile": true
}
```

## Relationship recall payload

```json
{
  "query": "谁是我最关键的合作伙伴",
  "layers": ["L3", "L1"],
  "top_k": 6,
  "include_profile": true
}
```

## Copilot payload

```json
{
  "query": "帮我总结最近项目的风险和阻塞",
  "task": "project_radar",
  "top_k": 8,
  "include_profile": true,
  "filters": {}
}
```

## Supported copilot tasks

- `resume`
- `work_status`
- `learning_engine`
- `self_profile`
- `project_radar`
- `decision_copilot`
- `career_assets`

## Write memory payload

```json
{
  "content": "本周完成了 X 模块上线，主要解决了 Y 问题。",
  "layer_hint": "auto",
  "metadata": {
    "source": "chat",
    "importance": 4
  }
}
```

## Good usage patterns

- 先判断问题属于“回忆事实”还是“综合产出”
- 综合产出优先走 `/synthesize/copilot`
- 纯关系问题优先查 `L3`
- 简历、职业资产、述职提纲类问题，不要只查单层

## Bad usage patterns

- 所有问题都只调 `/recall`
- 所有问题都全层搜索
- 没有证据时假装很确定
- 把闲聊内容一股脑写入记忆
