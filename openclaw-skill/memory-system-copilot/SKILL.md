---
name: memory-system-copilot
description: Connects to the six-layer Memory System and routes user requests to the right recall or synthesis capability. Use when the user asks about long-term memory, recent progress, self profile, collaboration relationships, project risk, learning plans, decision support, resume generation, or career assets.
---

# Memory System Copilot

## Purpose

Use this skill to connect an agent to the local six-layer Memory System API.

This skill is designed for memory-aware agents that need more than plain vector recall. The backend separates memory into:
- L0 profile
- L1 semantic knowledge
- L2 episodic timeline
- L3 relational graph
- L4 reflection
- L5 goals and aspirations

The agent should prefer routed retrieval over generic prompting.

## Required environment

- `MEMORY_API_BASE`, default: `http://localhost:8000`
- `MEMORY_API_KEY`, optional unless API auth is enabled

## Routing rules

### Default memory recall

When the user is asking a general memory question, call:

```text
POST /recall
```

Use:
- `layers=auto`
- `top_k=6`
- `include_profile=true`

Examples:
- “我最近在做什么？”
- “帮我回顾一下我过去一周的重点”
- “你还记得我之前提过的项目吗？”

### Relationship or people queries

When the user asks about people, collaboration, reporting lines, influence, stakeholders, or key partners, prefer:

```text
POST /recall
```

Use:
- `layers=["L3","L1"]`
- `include_profile=true`

Examples:
- “谁是我最关键的合作伙伴？”
- “我和谁协作最紧密？”
- “某个人在我的项目里扮演什么角色？”

### Resume or career positioning

When the user asks for resume, CV, positioning, accomplishments, or professional summary, call:

```text
POST /synthesize/copilot
```

Use:
- `task="resume"`

### Work status or recent progress

When the user asks for current status, recent progress, milestones, or what has been done recently, call:

```text
POST /synthesize/copilot
```

Use:
- `task="work_status"`

### Self profile

When the user asks “我是谁”, “你怎么看我”, “我的画像”, or wants a structured self summary, call:

```text
POST /synthesize/copilot
```

Use:
- `task="self_profile"`

### Project risk or blockers

When the user asks for project radar, blockers, delays, milestones, or risk signals, call:

```text
POST /synthesize/copilot
```

Use:
- `task="project_radar"`

### Decision support

When the user asks for trade-offs, recommendations, choice comparison, or execution suggestions, call:

```text
POST /synthesize/copilot
```

Use:
- `task="decision_copilot"`

### Learning plan

When the user asks what to learn next, learning path, weekly learning tasks, or growth priorities, call:

```text
POST /synthesize/copilot
```

Use:
- `task="learning_engine"`

### Career assets

When the user asks for review outline, project case cards, resume bullets, or professional assets, call:

```text
POST /synthesize/copilot
```

Use:
- `task="career_assets"`

## Response rules

When answering:
- Prefer returned evidence over speculation
- If confidence is low or evidence is sparse, say so explicitly
- Do not invent facts missing from memory
- Distinguish facts, reflections, relationships, and goals when summarizing
- For project or career output, prioritize concrete evidence and quantifiable items

## Writeback rules

Only write memory when the new content is high-signal.

Suitable for writeback:
- stable profile facts
- important project milestones
- reusable methods or lessons
- collaboration relationships
- explicit future goals

Avoid writing:
- filler chat
- greetings
- repeated phrasing
- unsupported assumptions

When storing memory, call:

```text
POST /remember
```

Use:
- `layer_hint="auto"` unless the caller already knows the correct layer

## Recommended answer style

- Be concise but structured
- Mention which evidence is strong vs weak
- If multiple layers disagree, surface the conflict instead of hiding it
- For action-oriented tasks, end with concrete next steps

## Additional reference

Read `reference.md` in this directory for payload templates and endpoint examples.
