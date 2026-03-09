# OpenClaw Skill Package

本目录提供一个面向 `Memory System` 的可复用 skill 包，目标是让支持“目录式 skill / markdown skill / prompt skill”的 Agent 框架快速接入本项目。

## 目录说明

- `memory-system-copilot/SKILL.md`：主 skill 文件
- `memory-system-copilot/reference.md`：接口说明与调用建议

## 推荐使用方式

### 方式 1：目录式 skill

如果你的 OpenClaw 运行环境支持“技能目录 + 主 skill markdown”模式，可直接把：

```text
openclaw-skill/memory-system-copilot/
```

复制到 OpenClaw 的技能目录中使用。

### 方式 2：粘贴式 skill

如果你的 OpenClaw 只支持粘贴系统提示词或工具说明：

1. 打开 `memory-system-copilot/SKILL.md`
2. 复制正文
3. 粘贴到 OpenClaw 的 skill / system prompt / agent instruction 区域

## 依赖的环境变量

- `MEMORY_API_BASE`：默认 `http://localhost:8000`
- `MEMORY_API_KEY`：可选，启用鉴权时必填

## 适用场景

- 长期记忆召回
- 最近进展总结
- 自我画像生成
- 项目风险雷达
- 决策建议
- 学习路径与任务生成
- 简历和职业资产生成
